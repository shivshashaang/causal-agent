"""Tests for the structural equations.

Deterministic tests come first. Simulation-based checks with bounds derived
from standard errors are marked `statistical` and sit at the bottom.
"""

import dataclasses

import numpy as np
import pytest

from causal_agent.bench import equations
from causal_agent.bench.equations import (
    linear_part,
    outcome_index,
    outcome_reference_stats,
    reference_sample,
    shared_nonlinear_part,
    simulate,
    treatment_index,
)
from causal_agent.bench.spec import NonlinearFamily, Scenario

ALL_SCENARIOS = list(Scenario)
SHARED = {"nonlinear_family": NonlinearFamily.SHARED}


def _arrays(sample) -> dict[str, np.ndarray]:
    return {
        name: value
        for name, value in dataclasses.asdict(sample).items()
        if value is not None
    }


# --- Reproducibility -------------------------------------------------------


@pytest.mark.parametrize("scenario", ALL_SCENARIOS)
def test_same_spec_gives_identical_sample(make_spec, scenario) -> None:
    first = _arrays(simulate(make_spec(scenario)))
    second = _arrays(simulate(make_spec(scenario)))
    assert first.keys() == second.keys()
    for name in first:
        np.testing.assert_array_equal(first[name], second[name], err_msg=name)


def test_different_seed_gives_different_sample(make_spec) -> None:
    first = simulate(make_spec(seed=1))
    second = simulate(make_spec(seed=2))
    assert not np.array_equal(first.c1, second.c1)
    assert not np.array_equal(first.t, second.t)
    assert not np.array_equal(first.y, second.y)


def test_sample_has_requested_number_of_rows(make_spec) -> None:
    sample = simulate(make_spec(n_rows=123))
    assert all(len(array) == 123 for array in _arrays(sample).values())


# --- True ATE --------------------------------------------------------------


@pytest.mark.parametrize("scenario", ALL_SCENARIOS)
@pytest.mark.parametrize("true_ate", [0.0, -1.0, 2.0])
def test_unit_level_effect_equals_true_ate_for_every_row(
    make_spec, scenario, true_ate
) -> None:
    sample = simulate(make_spec(scenario, true_ate=true_ate))
    np.testing.assert_allclose(sample.y1 - sample.y0, true_ate, atol=1e-12)


@pytest.mark.parametrize("scenario", ALL_SCENARIOS)
def test_observed_outcome_is_the_potential_outcome_received(
    make_spec, scenario
) -> None:
    sample = simulate(make_spec(scenario))
    treated = sample.t == 1
    np.testing.assert_array_equal(sample.y[treated], sample.y1[treated])
    np.testing.assert_array_equal(sample.y[~treated], sample.y0[~treated])


def test_treatment_is_binary_with_both_arms_present(make_spec) -> None:
    sample = simulate(make_spec())
    assert set(np.unique(sample.t)) == {0, 1}


# --- Knob independence -----------------------------------------------------


@pytest.mark.parametrize(
    "outcome_side_change",
    [
        {"outcome_noise": 2.0},
        {"confounder_outcome_strength": 2.0},
        {"true_ate": -1.0},
        {"outcome_predictor_strength": 0.5},
    ],
)
def test_treatment_unchanged_when_only_outcome_knobs_change(
    make_spec, outcome_side_change
) -> None:
    base = simulate(make_spec())
    changed = simulate(make_spec(**outcome_side_change))
    np.testing.assert_array_equal(base.t, changed.t)
    np.testing.assert_array_equal(base.propensity, changed.propensity)
    assert not np.array_equal(base.y, changed.y)


def test_treatment_unchanged_when_outcome_nonlinearity_changes(make_spec) -> None:
    base = simulate(
        make_spec(
            Scenario.CONFOUNDED_NONLINEAR,
            treatment_nonlinearity=1.0,
            outcome_nonlinearity=0.0,
        )
    )
    changed = simulate(
        make_spec(
            Scenario.CONFOUNDED_NONLINEAR,
            treatment_nonlinearity=1.0,
            outcome_nonlinearity=1.0,
        )
    )
    np.testing.assert_array_equal(base.t, changed.t)
    assert not np.array_equal(base.y, changed.y)


def test_trap_strength_changes_neither_treatment_nor_outcome(make_spec) -> None:
    weak = simulate(make_spec(Scenario.POST_OUTCOME_TRAP, trap_strength=0.5))
    strong = simulate(make_spec(Scenario.POST_OUTCOME_TRAP, trap_strength=2.0))
    np.testing.assert_array_equal(weak.t, strong.t)
    np.testing.assert_array_equal(weak.y, strong.y)
    assert not np.array_equal(weak.k, strong.k)


def test_adding_the_trap_changes_neither_treatment_nor_outcome(make_spec) -> None:
    without_trap = simulate(make_spec(Scenario.CONFOUNDED_LINEAR))
    with_trap = simulate(make_spec(Scenario.POST_OUTCOME_TRAP))
    np.testing.assert_array_equal(without_trap.t, with_trap.t)
    np.testing.assert_array_equal(without_trap.y, with_trap.y)


# --- Structure of each equation --------------------------------------------


def test_randomized_propensity_is_one_half(make_spec) -> None:
    sample = simulate(make_spec(Scenario.RANDOMIZED))
    np.testing.assert_array_equal(sample.propensity, 0.5)


def test_propensity_is_logit_linear_without_treatment_nonlinearity(
    make_spec,
) -> None:
    spec = make_spec(Scenario.CONFOUNDED_LINEAR, assignment_strength=1.5)
    sample = simulate(spec)
    log_odds = np.log(sample.propensity / (1 - sample.propensity))
    np.testing.assert_allclose(
        log_odds, 1.5 * linear_part(sample.c1, sample.c2), atol=1e-9
    )


def test_indices_are_exactly_linear_at_zero_nonlinearity() -> None:
    rng = np.random.default_rng(3)
    c1, c2 = rng.standard_normal(500), rng.standard_normal(500)
    np.testing.assert_array_equal(treatment_index(c1, c2, 0.0), linear_part(c1, c2))
    np.testing.assert_array_equal(outcome_index(c1, c2, 0.0), linear_part(c1, c2))


def test_indices_are_not_linear_at_positive_nonlinearity() -> None:
    rng = np.random.default_rng(3)
    c1, c2 = rng.standard_normal(500), rng.standard_normal(500)
    assert not np.allclose(treatment_index(c1, c2, 1.0), linear_part(c1, c2))
    assert not np.allclose(outcome_index(c1, c2, 1.0), linear_part(c1, c2))


def test_family_defaults_to_v1() -> None:
    rng = np.random.default_rng(3)
    c1, c2 = rng.standard_normal(500), rng.standard_normal(500)
    for index in (treatment_index, outcome_index):
        np.testing.assert_array_equal(
            index(c1, c2, 1.0), index(c1, c2, 1.0, NonlinearFamily.V1)
        )


def test_shared_family_uses_one_function_for_both_indices() -> None:
    rng = np.random.default_rng(3)
    c1, c2 = rng.standard_normal(500), rng.standard_normal(500)
    shared_h = treatment_index(c1, c2, 1.0, NonlinearFamily.SHARED)
    shared_g = outcome_index(c1, c2, 1.0, NonlinearFamily.SHARED)
    np.testing.assert_array_equal(shared_h, shared_g)
    assert not np.allclose(shared_h, treatment_index(c1, c2, 1.0))
    assert not np.allclose(shared_g, outcome_index(c1, c2, 1.0))


def test_family_does_not_matter_at_zero_nonlinearity() -> None:
    rng = np.random.default_rng(3)
    c1, c2 = rng.standard_normal(500), rng.standard_normal(500)
    for index in (treatment_index, outcome_index):
        np.testing.assert_array_equal(
            index(c1, c2, 0.0, NonlinearFamily.SHARED), linear_part(c1, c2)
        )


def test_shared_family_reaches_both_treatment_and_outcome(make_spec) -> None:
    """Switching the family changes who is treated and what they score."""
    v1 = simulate(make_spec(Scenario.CONFOUNDED_NONLINEAR))
    shared = simulate(make_spec(Scenario.CONFOUNDED_NONLINEAR, **SHARED))
    np.testing.assert_array_equal(v1.c1, shared.c1)
    np.testing.assert_array_equal(v1.c2, shared.c2)
    assert not np.array_equal(v1.t, shared.t)
    assert not np.allclose(v1.y0, shared.y0)


def test_treatment_unchanged_when_only_the_outcome_side_is_nonlinear(
    make_spec,
) -> None:
    """With lambda_T at zero the family cannot reach the treatment."""
    knobs = {"treatment_nonlinearity": 0.0, "outcome_nonlinearity": 1.0}
    v1 = simulate(make_spec(Scenario.CONFOUNDED_NONLINEAR, **knobs))
    shared = simulate(make_spec(Scenario.CONFOUNDED_NONLINEAR, **knobs, **SHARED))
    np.testing.assert_array_equal(v1.t, shared.t)
    np.testing.assert_array_equal(v1.propensity, shared.propensity)


@pytest.mark.parametrize(
    "knobs",
    [
        {"treatment_nonlinearity": 0.0, "outcome_nonlinearity": 1.0},
        {"treatment_nonlinearity": 1.0, "outcome_nonlinearity": 0.0},
        {"treatment_nonlinearity": 1.0, "outcome_nonlinearity": 1.0},
    ],
)
def test_shared_family_keeps_a_constant_unit_level_effect(make_spec, knobs) -> None:
    spec = make_spec(Scenario.CONFOUNDED_NONLINEAR, true_ate=0.37, **knobs, **SHARED)
    sample = simulate(spec)
    np.testing.assert_allclose(sample.y1 - sample.y0, 0.37, atol=1e-12)
    np.testing.assert_array_equal(
        sample.y, np.where(sample.t == 1, sample.y1, sample.y0)
    )


def test_shared_family_has_its_own_outcome_reference_stats(make_spec) -> None:
    v1 = outcome_reference_stats(make_spec(Scenario.CONFOUNDED_NONLINEAR))
    shared = outcome_reference_stats(make_spec(Scenario.CONFOUNDED_NONLINEAR, **SHARED))
    assert v1 != shared


def test_reference_sample_ignores_seed_and_rows(make_spec) -> None:
    first = reference_sample(make_spec(seed=1, n_rows=50))
    second = reference_sample(make_spec(seed=2, n_rows=80))
    assert len(first.y) == equations.REFERENCE_ROWS
    assert first.k is None
    np.testing.assert_array_equal(first.y, second.y)


def test_trap_column_exists_only_in_the_trap_scenario(make_spec) -> None:
    assert simulate(make_spec(Scenario.CONFOUNDED_LINEAR)).k is None
    assert simulate(make_spec(Scenario.POST_OUTCOME_TRAP)).k is not None


def test_trap_column_follows_its_equation(make_spec) -> None:
    weak_spec = make_spec(Scenario.POST_OUTCOME_TRAP, trap_strength=0.5)
    weak = simulate(weak_spec)
    strong = simulate(make_spec(Scenario.POST_OUTCOME_TRAP, trap_strength=2.0))
    y_mean, y_sd = outcome_reference_stats(weak_spec)
    signal = weak.t + (weak.y - y_mean) / y_sd
    # Both specs share the trap noise, so removing the signal must leave it.
    np.testing.assert_allclose(
        weak.k - 0.5 * signal, strong.k - 2.0 * signal, atol=1e-9
    )


# --- Fixed reference statistics --------------------------------------------


@pytest.mark.parametrize("scenario", ALL_SCENARIOS)
def test_no_row_depends_on_any_other_row(make_spec, scenario) -> None:
    """A shorter dataset is exactly the first rows of a longer one."""
    short = _arrays(simulate(make_spec(scenario, n_rows=500)))
    long = _arrays(simulate(make_spec(scenario, n_rows=2000)))
    for name in short:
        np.testing.assert_array_equal(short[name], long[name][:500], err_msg=name)


def test_outcome_reference_stats_ignore_seed_rows_and_trap_strength(
    make_spec,
) -> None:
    base = outcome_reference_stats(make_spec(Scenario.POST_OUTCOME_TRAP))
    changed = outcome_reference_stats(
        make_spec(Scenario.POST_OUTCOME_TRAP, seed=99, n_rows=50, trap_strength=2.0)
    )
    assert base == changed


def test_outcome_reference_stats_follow_the_outcome_knobs(make_spec) -> None:
    base = outcome_reference_stats(make_spec(Scenario.POST_OUTCOME_TRAP))
    noisier = outcome_reference_stats(
        make_spec(Scenario.POST_OUTCOME_TRAP, outcome_noise=2.0)
    )
    assert noisier[1] > base[1]


def test_reference_constants_are_reproducible(make_spec) -> None:
    """Recomputing from scratch with the fixed seed gives identical constants."""
    spec = make_spec(Scenario.POST_OUTCOME_TRAP)
    rng = np.random.default_rng(3)
    c1, c2 = rng.standard_normal(500), rng.standard_normal(500)

    first_outcome_stats = outcome_reference_stats(spec)
    first_h = treatment_index(c1, c2, 1.0)
    first_g = outcome_index(c1, c2, 1.0)

    equations._outcome_reference_cache.clear()
    equations._centre_and_scale.cache_clear()

    assert outcome_reference_stats(spec) == first_outcome_stats
    np.testing.assert_array_equal(treatment_index(c1, c2, 1.0), first_h)
    np.testing.assert_array_equal(outcome_index(c1, c2, 1.0), first_g)


# --- Interventions ---------------------------------------------------------


@pytest.mark.parametrize("forced", [0, 1])
def test_do_treatment_sets_every_row(make_spec, forced) -> None:
    sample = simulate(make_spec(), do_treatment=forced)
    np.testing.assert_array_equal(sample.t, forced)
    expected = sample.y1 if forced == 1 else sample.y0
    np.testing.assert_array_equal(sample.y, expected)


def test_do_treatment_rejects_other_values(make_spec) -> None:
    with pytest.raises(ValueError, match="do_treatment"):
        simulate(make_spec(), do_treatment=2)


# --- Statistical checks ----------------------------------------------------


@pytest.mark.statistical
@pytest.mark.parametrize("index", [treatment_index, outcome_index])
def test_nonlinear_index_is_centred_with_unit_variance(index) -> None:
    rng = np.random.default_rng(11)
    c1, c2 = rng.standard_normal(200_000), rng.standard_normal(200_000)
    values = index(c1, c2, 1.0)
    assert abs(values.mean()) < 0.02
    assert abs(values.std() - 1.0) < 0.02


@pytest.mark.statistical
@pytest.mark.parametrize("index", [treatment_index, outcome_index])
def test_shared_index_is_centred_with_unit_variance(index) -> None:
    rng = np.random.default_rng(11)
    c1, c2 = rng.standard_normal(200_000), rng.standard_normal(200_000)
    values = index(c1, c2, 1.0, NonlinearFamily.SHARED)
    assert abs(values.mean()) < 0.02
    assert abs(values.std() - 1.0) < 0.02


@pytest.mark.statistical
def test_shared_part_is_invisible_to_a_straight_line() -> None:
    """Mean zero and uncorrelated with each confounder and the linear index.

    The covariances are exactly zero in the population. The part still
    depends on the confounders, so the sampling error of a covariance is not
    1 / sqrt(rows); it is measured from the products instead.
    """
    rows = 200_000
    rng = np.random.default_rng(12)
    c1, c2 = rng.standard_normal(rows), rng.standard_normal(rows)
    part = shared_nonlinear_part(c1, c2)
    assert abs(part.mean()) <= 4 * part.std() / np.sqrt(rows)
    for straight in (c1, c2, linear_part(c1, c2)):
        products = part * straight
        assert abs(products.mean()) <= 4 * products.std() / np.sqrt(rows)


@pytest.mark.statistical
@pytest.mark.parametrize(
    "knobs",
    [
        {"treatment_nonlinearity": 0.0, "outcome_nonlinearity": 1.0},
        {"treatment_nonlinearity": 1.0, "outcome_nonlinearity": 0.0},
        {"treatment_nonlinearity": 1.0, "outcome_nonlinearity": 1.0},
    ],
)
def test_forced_treatment_difference_recovers_true_ate_in_shared_family(
    make_spec, knobs
) -> None:
    true_ate, rows = -0.62, 200_000
    scenario = Scenario.CONFOUNDED_NONLINEAR
    treated = simulate(
        make_spec(
            scenario, true_ate=true_ate, n_rows=rows, seed=101, **knobs, **SHARED
        ),
        do_treatment=1,
    )
    control = simulate(
        make_spec(
            scenario, true_ate=true_ate, n_rows=rows, seed=202, **knobs, **SHARED
        ),
        do_treatment=0,
    )
    difference = treated.y.mean() - control.y.mean()
    standard_error = np.sqrt(treated.y.var() / rows + control.y.var() / rows)
    assert abs(difference - true_ate) <= 4 * standard_error


@pytest.mark.statistical
def test_outcome_reference_stats_match_a_fresh_large_sample(make_spec) -> None:
    spec = make_spec(Scenario.POST_OUTCOME_TRAP, n_rows=200_000, seed=303)
    sample = simulate(spec)
    y_mean, y_sd = outcome_reference_stats(spec)
    standard_error = sample.y.std() / np.sqrt(spec.n_rows)
    assert abs(sample.y.mean() - y_mean) <= 4 * standard_error
    assert abs(sample.y.std() / y_sd - 1.0) < 0.02


@pytest.mark.statistical
@pytest.mark.parametrize("scenario", ALL_SCENARIOS)
def test_forced_treatment_difference_recovers_true_ate(make_spec, scenario) -> None:
    """Independent samples under do(T=1) and do(T=0) differ by tau on average."""
    true_ate = 1.0
    rows = 200_000
    treated = simulate(
        make_spec(scenario, true_ate=true_ate, n_rows=rows, seed=101),
        do_treatment=1,
    )
    control = simulate(
        make_spec(scenario, true_ate=true_ate, n_rows=rows, seed=202),
        do_treatment=0,
    )
    difference = treated.y.mean() - control.y.mean()
    standard_error = np.sqrt(treated.y.var() / rows + control.y.var() / rows)
    assert abs(difference - true_ate) <= 4 * standard_error
