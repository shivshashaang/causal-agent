"""Cross-fitted doubly robust (AIPW) estimator of the average treatment effect.

For each row the estimator combines an outcome model and a propensity model:

    psi = mu1(X) - mu0(X)
          + T * (Y - mu1(X)) / e(X)
          - (1 - T) * (Y - mu0(X)) / (1 - e(X))

    estimate = mean(psi),   standard error = sd(psi) / sqrt(n)

where mu1 and mu0 predict the outcome under treatment and control and e is the
probability of treatment. All three are gradient-boosted models fitted with
cross-fitting: the rows are split into folds, and each row's predictions come
from models trained on the other folds only (out-of-fold predictions).

Assumptions. The estimate is only as good as these:

1. The adjustment set is valid: it blocks every backdoor path and contains no
   variable affected by the treatment. This estimator cannot check that.
2. Overlap: every kind of unit has a real chance of both treatment arms.
3. The nuisance models approximate the truth well enough. "Doubly robust"
   means the estimate is consistent if either the outcome models or the
   propensity model is correct. It is not a guarantee of unbiasedness or
   accuracy in a finite sample, and it does not help if both are poor.

Numerical safeguards:

- Estimated propensities are clipped to [clip, 1 - clip] (default 0.01 and
  0.99) so that no row gets a weight above 1 / clip. Clipping trades variance
  for bias; the number of clipped rows is reported and a warning is raised
  when many rows are clipped.
- Folds are stratified by treatment so every training fold has both arms.
- Inputs must be finite, the treatment must be coded 0/1, and each arm needs
  enough rows to fit a model in every fold.
- All randomness (fold assignment, boosting) uses a fixed seed.

Limitations: the standard error treats the nuisance predictions as fixed and
assumes a valid adjustment set, so it understates uncertainty when either
fails. Boosted models regularize, which can leave some bias in finite samples.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.model_selection import StratifiedKFold

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

DEFAULT_FOLDS = 5
DEFAULT_SEED = 0
DEFAULT_CLIP = 0.01

# A warning is raised when more than this share of rows has a clipped propensity.
CLIPPED_SHARE_WARNING = 0.05

# Library defaults with shallow trees. Fixed in advance; not tuned on the benchmark.
BOOSTING_PARAMS = {"max_iter": 100, "max_depth": 3, "early_stopping": False}


@dataclass(frozen=True)
class EffectEstimate:
    """An ATE estimate with its standard error and diagnostics."""

    estimate: float
    standard_error: float
    n_rows: int
    n_treated: int
    diagnostics: dict[str, float] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    @property
    def confidence_interval(self) -> tuple[float, float]:
        """Normal-approximation 95% interval."""
        half_width = 1.96 * self.standard_error
        return self.estimate - half_width, self.estimate + half_width


@dataclass(frozen=True)
class Nuisances:
    """Out-of-fold nuisance predictions, one value per row."""

    mu0: FloatArray
    mu1: FloatArray
    propensity: FloatArray
    fold: IntArray


def prepare_arrays(
    frame: pd.DataFrame,
    treatment: str,
    outcome: str,
    adjustment_set: Sequence[str],
) -> tuple[IntArray, FloatArray, FloatArray]:
    """Validate the inputs and return (treatment, outcome, covariate matrix)."""
    adjustment = list(dict.fromkeys(adjustment_set))
    missing = [name for name in (treatment, outcome, *adjustment) if name not in frame]
    if missing:
        raise ValueError(f"columns not in the data: {missing}")
    if treatment in adjustment or outcome in adjustment:
        raise ValueError("the adjustment set cannot contain the treatment or outcome")

    t_raw = frame[treatment].to_numpy()
    if not np.isin(t_raw, (0, 1)).all():
        raise ValueError("the treatment must be coded 0/1")
    t = t_raw.astype(np.int64)
    y = frame[outcome].to_numpy(dtype=np.float64)
    x = frame[adjustment].to_numpy(dtype=np.float64).reshape(len(frame), -1)
    if not (np.isfinite(y).all() and np.isfinite(x).all()):
        raise ValueError("the outcome and adjustment columns must be finite")
    return t, y, x


def fold_assignments(t: IntArray, n_folds: int, seed: int) -> IntArray:
    """Assign each row to a fold, stratified by treatment."""
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold = np.empty(len(t), dtype=np.int64)
    for number, (_, test) in enumerate(splitter.split(np.zeros(len(t)), t)):
        fold[test] = number
    return fold


def cross_fit_nuisances(
    t: IntArray, y: FloatArray, x: FloatArray, n_folds: int, seed: int
) -> Nuisances:
    """Predict mu0, mu1 and the propensity for each row from the other folds."""
    fold = fold_assignments(t, n_folds, seed)
    mu0 = np.empty(len(t))
    mu1 = np.empty(len(t))
    propensity = np.empty(len(t))
    has_covariates = x.shape[1] > 0

    for number in range(n_folds):
        test = fold == number
        train = ~test
        treated = train & (t == 1)
        control = train & (t == 0)
        if not has_covariates:
            # Empty adjustment set: the models reduce to training-fold means.
            mu1[test] = y[treated].mean()
            mu0[test] = y[control].mean()
            propensity[test] = t[train].mean()
            continue
        model1 = HistGradientBoostingRegressor(random_state=seed, **BOOSTING_PARAMS)
        model0 = HistGradientBoostingRegressor(random_state=seed, **BOOSTING_PARAMS)
        classifier = HistGradientBoostingClassifier(
            random_state=seed, **BOOSTING_PARAMS
        )
        mu1[test] = model1.fit(x[treated], y[treated]).predict(x[test])
        mu0[test] = model0.fit(x[control], y[control]).predict(x[test])
        propensity[test] = classifier.fit(x[train], t[train]).predict_proba(x[test])[
            :, 1
        ]
    return Nuisances(mu0=mu0, mu1=mu1, propensity=propensity, fold=fold)


def aipw_estimate(
    frame: pd.DataFrame,
    treatment: str,
    outcome: str,
    adjustment_set: Sequence[str],
    *,
    n_folds: int = DEFAULT_FOLDS,
    seed: int = DEFAULT_SEED,
    clip: float = DEFAULT_CLIP,
) -> EffectEstimate:
    """Estimate the ATE with cross-fitted AIPW. See the module docstring."""
    if not 0 < clip < 0.5:
        raise ValueError("clip must be between 0 and 0.5")
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    t, y, x = prepare_arrays(frame, treatment, outcome, adjustment_set)

    n_rows = len(t)
    n_treated = int(t.sum())
    smaller_arm = min(n_treated, n_rows - n_treated)
    if smaller_arm < 2 * n_folds:
        raise ValueError(
            f"each treatment arm needs at least {2 * n_folds} rows; "
            f"the smaller arm has {smaller_arm}"
        )

    nuisances = cross_fit_nuisances(t, y, x, n_folds, seed)
    raw = nuisances.propensity
    e = np.clip(raw, clip, 1 - clip)
    n_clipped = int(np.sum((raw < clip) | (raw > 1 - clip)))

    psi = (
        nuisances.mu1
        - nuisances.mu0
        + t * (y - nuisances.mu1) / e
        - (1 - t) * (y - nuisances.mu0) / (1 - e)
    )
    estimate = float(psi.mean())
    standard_error = float(psi.std(ddof=1) / np.sqrt(n_rows))
    if not (np.isfinite(estimate) and np.isfinite(standard_error)):
        raise ValueError("the estimate is not finite")

    clipped_share = n_clipped / n_rows
    warnings: list[str] = []
    if clipped_share > CLIPPED_SHARE_WARNING:
        warnings.append(
            f"{clipped_share:.1%} of rows had an estimated propensity outside "
            f"[{clip}, {1 - clip}] and were clipped; overlap may be poor and "
            "the estimate may be biased"
        )

    return EffectEstimate(
        estimate=estimate,
        standard_error=standard_error,
        n_rows=n_rows,
        n_treated=n_treated,
        diagnostics={
            "propensity_min": float(raw.min()),
            "propensity_max": float(raw.max()),
            "clipped_rows": float(n_clipped),
            "clipped_share": clipped_share,
            "largest_weight": float(np.max(np.where(t == 1, 1 / e, 1 / (1 - e)))),
            "n_folds": float(n_folds),
        },
        warnings=tuple(warnings),
    )
