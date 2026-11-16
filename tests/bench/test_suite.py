"""Tests for the 30 benchmark specs.

Deterministic tests come first. Checks on generated data with bounds derived
from standard errors are marked `statistical` and sit at the bottom.
"""

from collections import Counter

import numpy as np
import pytest

from causal_agent.bench.equations import outcome_index, simulate
from causal_agent.bench.spec import (
    DEFAULT_N_ROWS,
    DEFAULT_OUTCOME_PREDICTOR_STRENGTH,
    DatasetSpec,
    Scenario,
)
from causal_agent.bench.suite import (
    ASSIGNMENT_LEVELS,
    CONFOUNDER_OUTCOME_LEVELS,
    NOISE_LEVELS,
    NONLINEAR_SIDES,
    TRAP_LEVELS,
    benchmark_specs,
)

SPECS = benchmark_specs()
SPEC_IDS = [spec.dataset_id for spec in SPECS]


def _of(scenario: Scenario) -> list[DatasetSpec]:
    return [spec for spec in SPECS if spec.scenario is scenario]


def _knobs(spec: DatasetSpec) -> tuple:
    return tuple(
        value
        for name, value in spec.model_dump().items()
        if name not in {"dataset_id", "seed"}
    )


# --- Shape of the benchmark ------------------------------------------------


def test_there_are_thirty_datasets_with_sequential_ids() -> None:
    assert SPEC_IDS == [f"ds_{number:03d}" for number in range(1, 31)]


def test_scenario_counts() -> None:
    assert Counter(spec.scenario for spec in SPECS) == {
        Scenario.RANDOMIZED: 6,
        Scenario.CONFOUNDED_LINEAR: 9,
        Scenario.CONFOUNDED_NONLINEAR: 9,
        Scenario.POST_OUTCOME_TRAP: 6,
    }


def test_fixed_settings_are_shared_by_every_dataset() -> None:
    assert {spec.n_rows for spec in SPECS} == {DEFAULT_N_ROWS}
    assert {spec.outcome_predictor_strength for spec in SPECS} == {
        DEFAULT_OUTCOME_PREDICTOR_STRENGTH
    }


def test_randomized_grid() -> None:
    grid = {
        (spec.outcome_nonlinearity, spec.outcome_noise)
        for spec in _of(Scenario.RANDOMIZED)
    }
    assert grid == {(lam, noise) for lam in (0.0, 1.0) for noise in NOISE_LEVELS}


def test_linear_grid_crosses_assignment_and_confounding() -> None:
    specs = _of(Scenario.CONFOUNDED_LINEAR)
    grid = {(s.assignment_strength, s.confounder_outcome_strength) for s in specs}
    assert grid == {
        (a, b) for a in ASSIGNMENT_LEVELS for b in CONFOUNDER_OUTCOME_LEVELS
    }
    for level in ASSIGNMENT_LEVELS:
        row = [s.outcome_noise for s in specs if s.assignment_strength == level]
        assert sorted(row) == list(NOISE_LEVELS)
    for level in CONFOUNDER_OUTCOME_LEVELS:
        column = [
            s.outcome_noise for s in specs if s.confounder_outcome_strength == level
        ]
        assert sorted(column) == list(NOISE_LEVELS)


def test_nonlinear_grid_crosses_sides_and_assignment() -> None:
    specs = _of(Scenario.CONFOUNDED_NONLINEAR)
    grid = {
        ((s.treatment_nonlinearity, s.outcome_nonlinearity), s.assignment_strength)
        for s in specs
    }
    assert grid == {(side, a) for side in NONLINEAR_SIDES for a in ASSIGNMENT_LEVELS}
    for side in NONLINEAR_SIDES:
        row = [
            s.outcome_noise
            for s in specs
            if (s.treatment_nonlinearity, s.outcome_nonlinearity) == side
        ]
        assert sorted(row) == list(NOISE_LEVELS)


def test_trap_grid_uses_a_linear_base() -> None:
    specs = _of(Scenario.POST_OUTCOME_TRAP)
    grid = {(s.trap_strength, s.outcome_noise) for s in specs}
    assert grid == {(gamma, noise) for gamma in TRAP_LEVELS for noise in NOISE_LEVELS}
    assert {s.assignment_strength for s in specs} == {1.0}
    assert {s.confounder_outcome_strength for s in specs} == {1.0}


@pytest.mark.parametrize("scenario", list(Scenario))
def test_each_scenario_has_a_zero_and_a_negative_effect(scenario) -> None:
    effects = {spec.true_ate for spec in _of(scenario)}
    assert 0.0 in effects
    assert any(effect < 0 for effect in effects)


# --- Seeds and IDs ---------------------------------------------------------


def test_every_dataset_has_its_own_seed() -> None:
    assert len({spec.seed for spec in SPECS}) == len(SPECS)


def test_specs_are_deterministic() -> None:
    assert benchmark_specs() == SPECS
    assert benchmark_specs(master_seed=0) == SPECS


def test_master_seed_changes_ids_and_seeds_but_not_the_knob_grid() -> None:
    other = benchmark_specs(master_seed=1)
    assert [s.seed for s in other] != [s.seed for s in SPECS]
    assert [s.scenario for s in other] != [s.scenario for s in SPECS]
    assert sorted(map(_knobs, other), key=repr) == sorted(map(_knobs, SPECS), key=repr)


def test_ids_do_not_group_datasets_by_scenario() -> None:
    """In ID order, scenarios are interleaved rather than in four blocks."""
    scenarios = [spec.scenario for spec in SPECS]
    changes = sum(a is not b for a, b in zip(scenarios, scenarios[1:], strict=False))
    assert changes > 10


# --- Overlap and balance ---------------------------------------------------


@pytest.mark.parametrize("spec", SPECS, ids=SPEC_IDS)
def test_overlap_and_balance(spec) -> None:
    """Identification needs every kind of unit to have a chance of each arm."""
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
            outcome_index(sample.c1, sample.c2, spec.outcome_nonlinearity),
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
    """With the true outcome model, the observed data yields tau."""
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
