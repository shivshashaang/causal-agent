"""Structural equations of the synthetic data-generating process.

    T ~ Bernoulli(sigmoid(a * h(C)))
    Y = tau * T + b * g(C) + d * p + sigma * noise
    k = gamma * (T + (Y - mu_Y) / sd_Y) + noise     (trap scenario only)

Every normalizing constant (for h, g and Y) is a fixed reference statistic
computed from a fixed seed, so no row's value depends on any other row.

h and g are a linear index plus a weighted nonlinear part. A spec's
`nonlinear_family` picks the nonlinear parts:

    v1      h uses an interaction plus a threshold, g a square plus a
            saturating curve. The two parts are nearly unrelated.
    shared  h and g use the same function, so the confounders drive treatment
            and outcome through the same curve.

Everything here is private to the generator and its tests. The potential
outcomes `y0` and `y1` exist so tests can verify the true ATE; they must never
be written to a public dataset.
"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import cache

import numpy as np
import numpy.typing as npt

from causal_agent.bench.spec import DatasetSpec, NonlinearFamily

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

# Normalizing constants are computed once from a reference sample drawn with
# this fixed seed. That keeps the structural equations fixed functions: they
# do not depend on the rows of any particular dataset.
REFERENCE_SEED = 987_654_321
REFERENCE_ROWS = 1_000_000

# The knobs that determine the distribution of the observed outcome.
_OUTCOME_KNOBS = (
    "true_ate",
    "assignment_strength",
    "confounder_outcome_strength",
    "outcome_noise",
    "treatment_nonlinearity",
    "outcome_nonlinearity",
    "nonlinear_family",
    "outcome_predictor_strength",
)
_outcome_reference_cache: dict[tuple[object, ...], tuple[float, float]] = {}

# One independent random stream per variable, so changing a knob that affects
# one variable cannot shift the random numbers used by another.
_STREAMS = ("c1", "c2", "p", "n", "treatment", "outcome_noise", "trap_noise")


def linear_part(c1: FloatArray, c2: FloatArray) -> FloatArray:
    """Shared linear index of the two confounders; standard normal."""
    return (c1 + c2) / np.sqrt(2.0)


def nonlinear_treatment_part(c1: FloatArray, c2: FloatArray) -> FloatArray:
    """v1 family, treatment side: an interaction plus a threshold."""
    return c1 * c2 + (c1 > 0.5)


def nonlinear_outcome_part(c1: FloatArray, c2: FloatArray) -> FloatArray:
    """v1 family, outcome side: a square plus a saturating curve."""
    return c2**2 + np.tanh(c1 + 0.5)


def shared_nonlinear_part(c1: FloatArray, c2: FloatArray) -> FloatArray:
    """Shared family, both sides: a U-shape in c1 plus an interaction.

    It has mean zero and is uncorrelated with c1, with c2 and with the linear
    index, so no straight line through the confounders can stand in for it.
    """
    return (c1**2 - 1.0) / np.sqrt(2.0) + c1 * c2


NonlinearPart = Callable[[FloatArray, FloatArray], FloatArray]

# (treatment side, outcome side) for each family.
_NONLINEAR_PARTS: dict[NonlinearFamily, tuple[NonlinearPart, NonlinearPart]] = {
    NonlinearFamily.V1: (nonlinear_treatment_part, nonlinear_outcome_part),
    NonlinearFamily.SHARED: (shared_nonlinear_part, shared_nonlinear_part),
}


@cache
def _centre_and_scale(
    nonlinear_part: NonlinearPart,
    nonlinearity: float,
) -> tuple[float, float]:
    """Population mean and standard deviation of the raw index."""
    if nonlinearity == 0:
        return 0.0, 1.0
    rng = np.random.default_rng(REFERENCE_SEED)
    c1 = rng.standard_normal(REFERENCE_ROWS)
    c2 = rng.standard_normal(REFERENCE_ROWS)
    raw = linear_part(c1, c2) + nonlinearity * nonlinear_part(c1, c2)
    return float(raw.mean()), float(raw.std())


def _index(
    nonlinear_part: NonlinearPart,
    c1: FloatArray,
    c2: FloatArray,
    nonlinearity: float,
) -> FloatArray:
    raw = linear_part(c1, c2) + nonlinearity * nonlinear_part(c1, c2)
    centre, scale = _centre_and_scale(nonlinear_part, nonlinearity)
    return (raw - centre) / scale


def treatment_index(
    c1: FloatArray,
    c2: FloatArray,
    nonlinearity: float,
    family: NonlinearFamily = NonlinearFamily.V1,
) -> FloatArray:
    """h(C): mean zero, unit variance. Exactly linear when nonlinearity is 0."""
    return _index(_NONLINEAR_PARTS[family][0], c1, c2, nonlinearity)


def outcome_index(
    c1: FloatArray,
    c2: FloatArray,
    nonlinearity: float,
    family: NonlinearFamily = NonlinearFamily.V1,
) -> FloatArray:
    """g(C): mean zero, unit variance. Exactly linear when nonlinearity is 0."""
    return _index(_NONLINEAR_PARTS[family][1], c1, c2, nonlinearity)


def propensity(c1: FloatArray, c2: FloatArray, spec: DatasetSpec) -> FloatArray:
    """True probability of treatment for each row."""
    index = treatment_index(c1, c2, spec.treatment_nonlinearity, spec.nonlinear_family)
    return 1.0 / (1.0 + np.exp(-spec.assignment_strength * index))


def potential_outcome(
    treatment: int,
    c1: FloatArray,
    c2: FloatArray,
    p: FloatArray,
    noise: FloatArray,
    spec: DatasetSpec,
) -> FloatArray:
    """Outcome each row would have if its treatment were set to `treatment`."""
    return (
        spec.true_ate * treatment
        + spec.confounder_outcome_strength
        * outcome_index(c1, c2, spec.outcome_nonlinearity, spec.nonlinear_family)
        + spec.outcome_predictor_strength * p
        + spec.outcome_noise * noise
    )


@dataclass(frozen=True)
class StructuralSample:
    """One draw from the structural equations. Internal: never publish as-is."""

    c1: FloatArray
    c2: FloatArray
    p: FloatArray
    n: FloatArray
    propensity: FloatArray
    t: IntArray
    y0: FloatArray
    y1: FloatArray
    y: FloatArray
    k: FloatArray | None


def reference_sample(spec: DatasetSpec) -> StructuralSample:
    """A large sample drawn under a spec's equations with the fixed reference seed.

    It depends only on the knobs, never on the dataset's seed or row count,
    and it has no trap column. Used for population-level facts about a spec.
    """
    return _simulate(
        spec,
        seed=REFERENCE_SEED,
        n_rows=REFERENCE_ROWS,
        do_treatment=None,
        include_trap=False,
    )


def outcome_reference_stats(spec: DatasetSpec) -> tuple[float, float]:
    """Fixed mean and standard deviation of the observed outcome for a spec.

    Computed once from a reference sample drawn with `REFERENCE_SEED` under
    this spec's own treatment and outcome equations. The result depends only
    on the knobs, never on the dataset's seed, row count or trap strength.
    """
    key = tuple(getattr(spec, name) for name in _OUTCOME_KNOBS)
    if key not in _outcome_reference_cache:
        reference = reference_sample(spec)
        _outcome_reference_cache[key] = (
            float(reference.y.mean()),
            float(reference.y.std()),
        )
    return _outcome_reference_cache[key]


def simulate(spec: DatasetSpec, do_treatment: int | None = None) -> StructuralSample:
    """Draw one dataset from the structural equations.

    With `do_treatment` set to 0 or 1, every row receives that treatment
    regardless of its propensity. This is the intervention used to check the
    true ATE.
    """
    if do_treatment not in (None, 0, 1):
        raise ValueError("do_treatment must be None, 0 or 1")
    return _simulate(
        spec,
        seed=spec.seed,
        n_rows=spec.n_rows,
        do_treatment=do_treatment,
        include_trap=spec.trap_strength > 0,
    )


def _simulate(
    spec: DatasetSpec,
    *,
    seed: int,
    n_rows: int,
    do_treatment: int | None,
    include_trap: bool,
) -> StructuralSample:
    seeds = np.random.SeedSequence(seed).spawn(len(_STREAMS))
    rng = {
        name: np.random.default_rng(stream_seed)
        for name, stream_seed in zip(_STREAMS, seeds, strict=True)
    }
    size = n_rows

    c1 = rng["c1"].standard_normal(size)
    c2 = rng["c2"].standard_normal(size)
    p = rng["p"].standard_normal(size)
    n = rng["n"].standard_normal(size)

    prop = propensity(c1, c2, spec)
    if do_treatment is None:
        t = (rng["treatment"].random(size) < prop).astype(np.int64)
    else:
        t = np.full(size, do_treatment, dtype=np.int64)

    noise = rng["outcome_noise"].standard_normal(size)
    y0 = potential_outcome(0, c1, c2, p, noise, spec)
    y1 = potential_outcome(1, c1, c2, p, noise, spec)
    y = np.where(t == 1, y1, y0)

    k = None
    if include_trap:
        y_mean, y_sd = outcome_reference_stats(spec)
        y_standardized = (y - y_mean) / y_sd
        trap_noise = rng["trap_noise"].standard_normal(size)
        k = spec.trap_strength * (t + y_standardized) + trap_noise

    return StructuralSample(
        c1=c1, c2=c2, p=p, n=n, propensity=prop, t=t, y0=y0, y1=y1, y=y, k=k
    )
