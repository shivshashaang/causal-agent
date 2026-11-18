"""Tests for the 20 Benchmark v2 specs (ADR 004).

Deterministic tests come first. Checks on generated data with bounds derived
from standard errors are marked `statistical` and sit at the bottom.

Nothing here asserts that any estimator fails. The tests check the properties
the benchmark must have for its true effects to be recoverable: a known
effect, overlap, a valid adjustment set made of observed columns, and the same
public interface as v1.
"""

from collections import Counter

import numpy as np
import pytest

from causal_agent.bench.equations import outcome_index, simulate
from causal_agent.bench.generate import PUBLIC_COVARIATES, generate
from causal_agent.bench.public import Timing
from causal_agent.bench.spec import (
    DEFAULT_N_ROWS,
    DEFAULT_OUTCOME_PREDICTOR_STRENGTH,
    DatasetSpec,
    NonlinearFamily,
    Scenario,
)
from causal_agent.bench.suite import (
    ASSIGNMENT_LEVELS,
    TRAP_LEVELS,
    V2_ASSIGNMENT_ENDS,
    V2_ATE_RANGE,
    V2_CONFOUNDER_OUTCOME_ENDS,
    V2_MASTER_SEED,
    V2_TRAP_NOISE,
    benchmark_specs,
    benchmark_v2_specs,
)

SPECS = benchmark_v2_specs()
SPEC_IDS = [spec.dataset_id for spec in SPECS]
V1_SPECS = benchmark_specs()


def _of(scenario: Scenario) -> list[DatasetSpec]:
    return [spec for spec in SPECS if spec.scenario is scenario]


def _sides(spec: DatasetSpec) -> tuple[float, float]:
    return spec.treatment_nonlinearity, spec.outcome_nonlinearity


def _nonlinear(sides: tuple[float, float]) -> list[DatasetSpec]:
    return [s for s in _of(Scenario.CONFOUNDED_NONLINEAR) if _sides(s) == sides]


# --- Shape of the benchmark ------------------------------------------------


def test_there_are_twenty_datasets_with_sequential_ids() -> None:
    assert SPEC_IDS == [f"ds_{number:03d}" for number in range(101, 121)]


def test_ids_and_seeds_do_not_overlap_with_v1() -> None:
    assert not {s.dataset_id for s in SPECS} & {s.dataset_id for s in V1_SPECS}
    assert not {s.seed for s in SPECS} & {s.seed for s in V1_SPECS}


def test_scenario_counts() -> None:
    assert Counter(spec.scenario for spec in SPECS) == {
        Scenario.RANDOMIZED: 2,
        Scenario.CONFOUNDED_LINEAR: 4,
        Scenario.CONFOUNDED_NONLINEAR: 10,
        Scenario.POST_OUTCOME_TRAP: 4,
    }


def test_nonlinear_subtype_counts() -> None:
    counts = Counter(_sides(s) for s in _of(Scenario.CONFOUNDED_NONLINEAR))
    assert counts == {(0.0, 1.0): 2, (1.0, 0.0): 2, (1.0, 1.0): 6}


def test_fixed_settings_are_the_same_as_v1() -> None:
    assert {spec.n_rows for spec in SPECS} == {DEFAULT_N_ROWS}
    assert {spec.outcome_predictor_strength for spec in SPECS} == {
        DEFAULT_OUTCOME_PREDICTOR_STRENGTH
    }


def test_every_nonlinear_dataset_uses_the_shared_family() -> None:
    for spec in SPECS:
        nonlinear = spec.treatment_nonlinearity > 0 or spec.outcome_nonlinearity > 0
        expected = NonlinearFamily.SHARED if nonlinear else NonlinearFamily.V1
        assert spec.nonlinear_family is expected, spec.dataset_id


def test_randomized_has_one_linear_and_one_curved_outcome() -> None:
    specs = _of(Scenario.RANDOMIZED)
    assert sorted(s.outcome_nonlinearity for s in specs) == [0.0, 1.0]
    assert {s.assignment_strength for s in specs} == {0.0}


def test_linear_grid_crosses_the_ends() -> None:
    grid = {
        (s.assignment_strength, s.confounder_outcome_strength)
        for s in _of(Scenario.CONFOUNDED_LINEAR)
    }
    assert grid == {
        (a, b) for a in V2_ASSIGNMENT_ENDS for b in V2_CONFOUNDER_OUTCOME_ENDS
    }


@pytest.mark.parametrize("sides", [(0.0, 1.0), (1.0, 0.0)])
def test_one_sided_grid_uses_both_assignment_ends(sides) -> None:
    specs = _nonlinear(sides)
    assert sorted(s.assignment_strength for s in specs) == list(V2_ASSIGNMENT_ENDS)
    assert {s.confounder_outcome_strength for s in specs} == {1.0}


def test_both_sides_grid_crosses_assignment_and_confounding() -> None:
    specs = _nonlinear((1.0, 1.0))
    grid = {(s.assignment_strength, s.confounder_outcome_strength) for s in specs}
    assert grid == {
        (a, b) for a in ASSIGNMENT_LEVELS for b in V2_CONFOUNDER_OUTCOME_ENDS
    }
    assert Counter(s.outcome_noise for s in specs) == {0.5: 2, 1.0: 2, 2.0: 2}


def test_trap_grid_uses_a_linear_base() -> None:
    specs = _of(Scenario.POST_OUTCOME_TRAP)
    grid = {(s.trap_strength, s.outcome_noise) for s in specs}
    assert grid == {(gamma, noise) for gamma in TRAP_LEVELS for noise in V2_TRAP_NOISE}
    assert {s.assignment_strength for s in specs} == {1.0}
    assert {s.confounder_outcome_strength for s in specs} == {1.0}


def test_every_noise_level_is_used() -> None:
    assert {spec.outcome_noise for spec in SPECS} == {0.5, 1.0, 2.0}


# --- True effects ----------------------------------------------------------


def test_exactly_two_datasets_have_a_zero_effect() -> None:
    assert sum(spec.true_ate == 0.0 for spec in SPECS) == 2


def test_nonzero_effects_are_varied_and_not_round() -> None:
    effects = [spec.true_ate for spec in SPECS if spec.true_ate != 0.0]
    assert len(set(effects)) == len(effects)
    assert all(V2_ATE_RANGE[0] <= effect <= V2_ATE_RANGE[1] for effect in effects)
    assert all(round(effect * 2) != effect * 2 for effect in effects)
    assert any(effect < 0 for effect in effects)
    assert any(effect > 0 for effect in effects)
    assert all(round(effect, 2) == effect for effect in effects)


# --- Seeds and IDs ---------------------------------------------------------


def test_every_dataset_has_its_own_seed() -> None:
    assert len({spec.seed for spec in SPECS}) == len(SPECS)


def test_specs_are_deterministic() -> None:
    assert benchmark_v2_specs() == SPECS
    assert benchmark_v2_specs(master_seed=V2_MASTER_SEED) == SPECS


def test_ids_do_not_group_datasets_by_scenario() -> None:
    scenarios = [spec.scenario for spec in SPECS]
    changes = sum(a is not b for a, b in zip(scenarios, scenarios[1:], strict=False))
    assert changes > 6


# --- Same public interface as v1 -------------------------------------------


@pytest.fixture(scope="module")
def datasets():
    return {spec.dataset_id: generate(spec) for spec in SPECS}


@pytest.fixture(scope="module")
def v1_example():
    return generate(V1_SPECS[0])


@pytest.mark.parametrize("dataset_id", SPEC_IDS)
def test_public_schema_matches_v1(datasets, v1_example, dataset_id) -> None:
    dataset = datasets[dataset_id]
    assert list(dataset.frame.columns) == list(v1_example.frame.columns)
    assert dataset.public.question == v1_example.public.question
    assert dataset.public.n_rows == v1_example.public.n_rows
    assert [(c.name, c.kind) for c in dataset.public.columns] == [
        (c.name, c.kind) for c in v1_example.public.columns
    ]


@pytest.mark.parametrize("spec", SPECS, ids=SPEC_IDS)
def test_only_the_trap_column_is_labelled_post_outcome(datasets, spec) -> None:
    covariates = datasets[spec.dataset_id].public.covariates
    post = [c.name for c in covariates if c.timing is Timing.POST_OUTCOME]
    expected = 1 if spec.scenario is Scenario.POST_OUTCOME_TRAP else 0
    assert len(post) == expected


# --- Identification --------------------------------------------------------


@pytest.mark.parametrize("spec", SPECS, ids=SPEC_IDS)
def test_true_effect_is_the_unit_level_effect(spec) -> None:
    sample = simulate(spec)
    np.testing.assert_allclose(sample.y1 - sample.y0, spec.true_ate, atol=1e-12)


@pytest.mark.parametrize("spec", SPECS, ids=SPEC_IDS)
def test_pre_treatment_columns_are_a_valid_adjustment_set(datasets, spec) -> None:
    """No hidden confounding: observed pre-treatment columns close every backdoor."""
    dataset = datasets[spec.dataset_id]
    pre = [
        c.name for c in dataset.public.covariates if c.timing is Timing.PRE_TREATMENT
    ]
    assert dataset.truth.is_valid_adjustment_set(pre)


@pytest.mark.parametrize("spec", SPECS, ids=SPEC_IDS)
def test_empty_set_is_valid_only_when_randomized(datasets, spec) -> None:
    truth = datasets[spec.dataset_id].truth
    assert truth.is_valid_adjustment_set([]) == (spec.scenario is Scenario.RANDOMIZED)


@pytest.mark.parametrize(
    "spec",
    _of(Scenario.POST_OUTCOME_TRAP),
    ids=[spec.dataset_id for spec in _of(Scenario.POST_OUTCOME_TRAP)],
)
def test_adjusting_for_the_trap_column_is_invalid(datasets, spec) -> None:
    dataset = datasets[spec.dataset_id]
    assert not dataset.truth.is_valid_adjustment_set(PUBLIC_COVARIATES)


@pytest.mark.parametrize("spec", SPECS, ids=SPEC_IDS)
def test_overlap_and_balance(spec) -> None:
    """The same thresholds as v1."""
    sample = simulate(spec)
    inside = np.mean((sample.propensity >= 0.05) & (sample.propensity <= 0.95))
    assert inside >= 0.90
    assert 0.2 <= sample.t.mean() <= 0.8


# --- Statistical checks ----------------------------------------------------


def _oracle_regression(spec: DatasetSpec) -> tuple[float, float]:
    """OLS of Y on T, the true g(C) and p: estimate of tau and its standard error."""
    sample = simulate(spec)
    design = np.column_stack(
        [
            np.ones(spec.n_rows),
            sample.t,
            outcome_index(
                sample.c1, sample.c2, spec.outcome_nonlinearity, spec.nonlinear_family
            ),
            sample.p,
        ]
    )
    coefficients, *_ = np.linalg.lstsq(design, sample.y, rcond=None)
    residuals = sample.y - design @ coefficients
    residual_variance = residuals @ residuals / (spec.n_rows - design.shape[1])
    covariance = residual_variance * np.linalg.inv(design.T @ design)
    return float(coefficients[1]), float(np.sqrt(covariance[1, 1]))


@pytest.mark.statistical
@pytest.mark.parametrize("spec", SPECS, ids=SPEC_IDS)
def test_oracle_regression_recovers_true_ate(spec) -> None:
    """Given the true function of the observed confounders, the data yields tau.

    This is the check that the effect is identified from observed columns: a
    model that knows the right curve needs nothing that is hidden.
    """
    estimate, standard_error = _oracle_regression(spec)
    assert abs(estimate - spec.true_ate) <= 4 * standard_error


@pytest.mark.statistical
@pytest.mark.parametrize(
    "spec",
    _of(Scenario.RANDOMIZED),
    ids=[spec.dataset_id for spec in _of(Scenario.RANDOMIZED)],
)
def test_naive_difference_recovers_true_ate_when_randomized(spec) -> None:
    sample = simulate(spec)
    treated, control = sample.y[sample.t == 1], sample.y[sample.t == 0]
    difference = treated.mean() - control.mean()
    standard_error = np.sqrt(
        treated.var(ddof=1) / len(treated) + control.var(ddof=1) / len(control)
    )
    assert abs(difference - spec.true_ate) <= 4 * standard_error
