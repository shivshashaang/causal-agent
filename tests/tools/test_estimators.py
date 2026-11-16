"""Tests for the four estimators and the dispatcher."""

import numpy as np
import pandas as pd
import pytest

from causal_agent.agent.fixed_pipeline import ESTIMATOR as PIPELINE_ESTIMATOR
from causal_agent.bench.spec import Scenario
from causal_agent.tools.aipw import aipw_estimate
from causal_agent.tools.estimators import (
    ASSUMPTIONS,
    EstimatorName,
    difference_in_means,
    estimate_ate,
    linear_regression,
    propensity_weighting,
)

ADJUSTED = [
    EstimatorName.LINEAR_REGRESSION,
    EstimatorName.PROPENSITY_WEIGHTING,
    EstimatorName.DOUBLY_ROBUST,
]


def _frame(small_benchmark, scenario) -> pd.DataFrame:
    return small_benchmark.datasets[scenario].frame


def _valid_set(small_benchmark, scenario) -> list[str]:
    truth = small_benchmark.datasets[scenario].truth
    return list(min(truth.valid_adjustment_sets, key=len))


def _naive(frame: pd.DataFrame) -> float:
    return float(
        frame.loc[frame.t == 1, "y"].mean() - frame.loc[frame.t == 0, "y"].mean()
    )


# --- Names and documented assumptions --------------------------------------


def test_every_estimator_documents_its_assumptions() -> None:
    assert set(ASSUMPTIONS) == set(EstimatorName)
    for name in ADJUSTED:
        assert any("adjustment set must be valid" in text for text in ASSUMPTIONS[name])
    doubly_robust = " ".join(ASSUMPTIONS[EstimatorName.DOUBLY_ROBUST])
    assert "not a guarantee" in doubly_robust
    assert "randomized" in " ".join(ASSUMPTIONS[EstimatorName.DIFFERENCE_IN_MEANS])


def test_fixed_pipeline_uses_a_named_estimator() -> None:
    assert PIPELINE_ESTIMATOR == EstimatorName.DOUBLY_ROBUST.value


# --- Difference in means ---------------------------------------------------


def test_difference_in_means_is_the_gap_between_arm_means() -> None:
    frame = pd.DataFrame(
        {"t": [0, 0, 0, 1, 1, 1], "y": [1.0, 2.0, 3.0, 5.0, 6.0, 10.0]}
    )
    result = difference_in_means(frame, "t", "y")
    assert result.estimate == pytest.approx(7.0 - 2.0)
    assert result.standard_error == pytest.approx(np.sqrt(7.0 / 3 + 1.0 / 3))
    assert result.n_rows == 6
    assert result.n_treated == 3


def test_difference_in_means_always_uses_the_empty_adjustment_set(
    small_benchmark,
) -> None:
    frame = _frame(small_benchmark, Scenario.CONFOUNDED_LINEAR)
    with pytest.raises(ValueError, match="always uses the empty adjustment set"):
        estimate_ate(EstimatorName.DIFFERENCE_IN_MEANS, frame, "t", "y", ["x1"])
    result = estimate_ate(EstimatorName.DIFFERENCE_IN_MEANS, frame, "t", "y", [])
    assert result.estimate == pytest.approx(_naive(frame))


# --- Linear regression -----------------------------------------------------


def test_linear_regression_recovers_exact_coefficients_without_noise() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(200)
    t = (rng.random(200) < 0.5).astype(int)
    frame = pd.DataFrame({"x": x, "t": t, "y": 3.0 + 2.0 * t - 1.5 * x})
    result = linear_regression(frame, "t", "y", ["x"])
    assert result.estimate == pytest.approx(2.0)
    assert result.diagnostics["r_squared"] == pytest.approx(1.0)


def test_linear_regression_with_no_covariates_equals_difference_in_means(
    small_benchmark,
) -> None:
    frame = _frame(small_benchmark, Scenario.RANDOMIZED)
    assert linear_regression(frame, "t", "y", []).estimate == pytest.approx(
        _naive(frame)
    )


def test_linear_regression_rejects_dependent_columns(small_benchmark) -> None:
    frame = _frame(small_benchmark, Scenario.CONFOUNDED_LINEAR).assign(
        copy=lambda f: f.x1
    )
    with pytest.raises(ValueError, match="linearly dependent"):
        linear_regression(frame, "t", "y", ["x1", "copy"])


# --- Propensity weighting --------------------------------------------------


def test_weighting_with_no_covariates_equals_difference_in_means(
    small_benchmark,
) -> None:
    frame = _frame(small_benchmark, Scenario.CONFOUNDED_LINEAR)
    result = propensity_weighting(frame, "t", "y", [])
    assert result.estimate == pytest.approx(_naive(frame))
    assert result.diagnostics["clipped_rows"] == 0


def test_weighting_clips_extreme_propensities() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(1000)
    t = (x + 0.05 * rng.standard_normal(1000) > 0).astype(int)
    frame = pd.DataFrame({"x": x, "t": t, "y": x + t + rng.standard_normal(1000)})
    result = propensity_weighting(frame, "t", "y", ["x"])
    assert result.diagnostics["clipped_share"] > 0.05
    assert result.diagnostics["largest_weight"] <= 100 + 1e-9
    assert any("clipped" in warning for warning in result.warnings)


# --- Shared behaviour ------------------------------------------------------


@pytest.mark.parametrize("estimator", list(EstimatorName))
def test_estimators_are_deterministic(small_benchmark, estimator) -> None:
    frame = _frame(small_benchmark, Scenario.CONFOUNDED_LINEAR)
    adjustment = (
        []
        if estimator is EstimatorName.DIFFERENCE_IN_MEANS
        else _valid_set(small_benchmark, Scenario.CONFOUNDED_LINEAR)
    )
    first = estimate_ate(estimator, frame, "t", "y", adjustment)
    second = estimate_ate(estimator, frame, "t", "y", adjustment)
    assert first == second
    assert first.standard_error > 0


def test_dispatcher_runs_the_doubly_robust_estimator(small_benchmark) -> None:
    frame = _frame(small_benchmark, Scenario.CONFOUNDED_LINEAR)
    adjustment = _valid_set(small_benchmark, Scenario.CONFOUNDED_LINEAR)
    via_dispatch = estimate_ate(
        EstimatorName.DOUBLY_ROBUST, frame, "t", "y", adjustment
    )
    assert via_dispatch == aipw_estimate(frame, "t", "y", adjustment)


@pytest.mark.parametrize("estimator", ADJUSTED)
def test_invalid_columns_are_rejected(small_benchmark, estimator) -> None:
    frame = _frame(small_benchmark, Scenario.CONFOUNDED_LINEAR)
    with pytest.raises(ValueError, match="not in the data"):
        estimate_ate(estimator, frame, "t", "y", ["nope"])
    with pytest.raises(ValueError, match="cannot contain"):
        estimate_ate(estimator, frame, "t", "y", ["x1", "y"])


# --- Statistical checks ----------------------------------------------------


@pytest.mark.statistical
@pytest.mark.parametrize("estimator", ADJUSTED)
def test_adjusted_estimators_recover_the_effect_when_their_assumptions_hold(
    small_benchmark, estimator
) -> None:
    """Linear data, valid adjustment set, true ATE of 1."""
    frame = _frame(small_benchmark, Scenario.CONFOUNDED_LINEAR)
    adjustment = _valid_set(small_benchmark, Scenario.CONFOUNDED_LINEAR)
    result = estimate_ate(estimator, frame, "t", "y", adjustment)
    assert abs(result.estimate - 1.0) <= 4 * result.standard_error


@pytest.mark.statistical
def test_difference_in_means_recovers_the_effect_when_randomized(
    small_benchmark,
) -> None:
    frame = _frame(small_benchmark, Scenario.RANDOMIZED)
    result = difference_in_means(frame, "t", "y")
    assert abs(result.estimate - 1.0) <= 4 * result.standard_error
