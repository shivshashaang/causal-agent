"""Tests for the cross-fitted AIPW estimator."""

import numpy as np
import pandas as pd
import pytest

from causal_agent.bench.generate import generate
from causal_agent.bench.spec import DatasetSpec, Scenario
from causal_agent.tools.aipw import (
    DEFAULT_CLIP,
    aipw_estimate,
    cross_fit_nuisances,
    fold_assignments,
    prepare_arrays,
)


def _spec(scenario: Scenario, n_rows: int = 1000, **knobs) -> DatasetSpec:
    fields = {
        "dataset_id": "ds_test",
        "scenario": scenario,
        "seed": 5,
        "n_rows": n_rows,
        "true_ate": 1.0,
        "assignment_strength": 0.0 if scenario is Scenario.RANDOMIZED else 1.0,
        "confounder_outcome_strength": 1.0,
        "outcome_noise": 1.0,
        **knobs,
    }
    return DatasetSpec(**fields)


@pytest.fixture(scope="module")
def confounded():
    return generate(_spec(Scenario.CONFOUNDED_LINEAR))


def _valid_set(dataset) -> list[str]:
    return list(min(dataset.truth.valid_adjustment_sets, key=len))


def _poor_overlap_frame() -> pd.DataFrame:
    """Treatment is almost a deterministic function of x."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(1000)
    t = (x + 0.05 * rng.standard_normal(1000) > 0).astype(int)
    y = x + t + rng.standard_normal(1000)
    return pd.DataFrame({"x": x, "t": t, "y": y})


# --- Reproducibility -------------------------------------------------------


def test_same_inputs_give_the_same_estimate(confounded) -> None:
    first = aipw_estimate(confounded.frame, "t", "y", _valid_set(confounded))
    second = aipw_estimate(confounded.frame, "t", "y", _valid_set(confounded))
    assert first == second


def test_a_different_seed_changes_the_folds_and_the_estimate(confounded) -> None:
    first = aipw_estimate(confounded.frame, "t", "y", _valid_set(confounded), seed=0)
    second = aipw_estimate(confounded.frame, "t", "y", _valid_set(confounded), seed=1)
    assert first.estimate != second.estimate
    assert abs(first.estimate - second.estimate) < first.standard_error


def test_order_and_duplicates_in_the_adjustment_set_do_not_matter_for_validation(
    confounded,
) -> None:
    names = _valid_set(confounded)
    t, y, x = prepare_arrays(confounded.frame, "t", "y", [*names, names[0]])
    assert x.shape == (len(confounded.frame), len(names))


# --- Cross-fitting ---------------------------------------------------------


def test_folds_cover_every_row_and_are_stratified_by_treatment(confounded) -> None:
    t = confounded.frame["t"].to_numpy()
    fold = fold_assignments(t, n_folds=5, seed=0)
    assert set(fold) == {0, 1, 2, 3, 4}
    overall = t.mean()
    for number in range(5):
        in_fold = fold == number
        assert abs(in_fold.mean() - 0.2) < 0.01
        assert abs(t[in_fold].mean() - overall) < 0.01
    np.testing.assert_array_equal(fold, fold_assignments(t, n_folds=5, seed=0))


def test_nuisance_predictions_are_out_of_fold(confounded) -> None:
    """A row's own outcome cannot influence the predictions made for it."""
    t, y, x = prepare_arrays(confounded.frame, "t", "y", _valid_set(confounded))
    base = cross_fit_nuisances(t, y, x, n_folds=5, seed=0)

    in_fold = base.fold == 0
    changed_y = y.copy()
    changed_y[in_fold] += 100.0
    changed = cross_fit_nuisances(t, changed_y, x, n_folds=5, seed=0)

    # Predictions for fold 0 come from models that never saw fold 0.
    np.testing.assert_array_equal(changed.mu0[in_fold], base.mu0[in_fold])
    np.testing.assert_array_equal(changed.mu1[in_fold], base.mu1[in_fold])
    # Other folds trained on the changed rows, so their predictions move.
    assert not np.allclose(changed.mu1[~in_fold], base.mu1[~in_fold])
    # The propensity model never uses the outcome.
    np.testing.assert_array_equal(changed.propensity, base.propensity)


# --- Empty adjustment set --------------------------------------------------


def test_empty_adjustment_set_matches_the_difference_in_means() -> None:
    dataset = generate(_spec(Scenario.RANDOMIZED))
    frame = dataset.frame
    result = aipw_estimate(frame, "t", "y", [])
    naive = frame.loc[frame.t == 1, "y"].mean() - frame.loc[frame.t == 0, "y"].mean()
    assert abs(result.estimate - naive) < 0.1 * result.standard_error
    assert result.diagnostics["clipped_rows"] == 0
    assert result.diagnostics["propensity_min"] == pytest.approx(
        frame.t.mean(), abs=0.01
    )


def test_empty_adjustment_set_does_not_remove_confounding(confounded) -> None:
    """With no covariates the estimator has nothing to adjust with."""
    frame = confounded.frame
    result = aipw_estimate(frame, "t", "y", [])
    naive = frame.loc[frame.t == 1, "y"].mean() - frame.loc[frame.t == 0, "y"].mean()
    assert abs(result.estimate - naive) < 0.1 * result.standard_error


# --- Clipping and safeguards -----------------------------------------------


def test_extreme_propensities_are_clipped_and_reported() -> None:
    result = aipw_estimate(_poor_overlap_frame(), "t", "y", ["x"])
    assert result.diagnostics["clipped_rows"] > 0
    assert result.diagnostics["largest_weight"] <= 1 / DEFAULT_CLIP + 1e-9
    assert result.diagnostics["clipped_share"] > 0.05
    assert any("clipped" in warning for warning in result.warnings)
    assert np.isfinite(result.estimate)


def test_a_wider_clip_bounds_the_weights_more_tightly() -> None:
    result = aipw_estimate(_poor_overlap_frame(), "t", "y", ["x"], clip=0.1)
    assert result.diagnostics["largest_weight"] <= 10 + 1e-9


def test_good_overlap_raises_no_warning(confounded) -> None:
    result = aipw_estimate(confounded.frame, "t", "y", _valid_set(confounded))
    assert result.warnings == ()
    assert result.standard_error > 0
    low, high = result.confidence_interval
    assert low < result.estimate < high
    assert result.n_rows == 1000
    assert result.n_treated == int(confounded.frame["t"].sum())


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda f: f.assign(t=f.t * 2), "coded 0/1"),
        (lambda f: f.assign(y=f.y.where(f.index > 0, np.nan)), "finite"),
        (lambda f: f.assign(x1=f.x1.where(f.index > 0, np.inf)), "finite"),
        (lambda f: f.drop(columns=["x2"]), "not in the data"),
    ],
)
def test_invalid_data_is_rejected(confounded, change, message) -> None:
    with pytest.raises(ValueError, match=message):
        aipw_estimate(change(confounded.frame), "t", "y", ["x1", "x2"])


@pytest.mark.parametrize("bad_set", [["t"], ["x1", "y"]])
def test_treatment_and_outcome_cannot_be_adjusted_for(confounded, bad_set) -> None:
    with pytest.raises(ValueError, match="cannot contain"):
        aipw_estimate(confounded.frame, "t", "y", bad_set)


def test_a_tiny_treatment_arm_is_rejected(confounded) -> None:
    frame = confounded.frame.copy()
    frame["t"] = 0
    frame.loc[:4, "t"] = 1
    with pytest.raises(ValueError, match="at least 10 rows"):
        aipw_estimate(frame, "t", "y", ["x1"])


@pytest.mark.parametrize("options", [{"clip": 0.0}, {"clip": 0.5}, {"n_folds": 1}])
def test_invalid_options_are_rejected(confounded, options) -> None:
    with pytest.raises(ValueError):
        aipw_estimate(confounded.frame, "t", "y", ["x1"], **options)


# --- Statistical checks ----------------------------------------------------


@pytest.mark.statistical
@pytest.mark.parametrize(
    ("scenario", "knobs"),
    [
        (Scenario.RANDOMIZED, {}),
        (Scenario.CONFOUNDED_LINEAR, {}),
        (
            Scenario.CONFOUNDED_NONLINEAR,
            {"treatment_nonlinearity": 1.0, "outcome_nonlinearity": 1.0},
        ),
    ],
)
def test_estimate_is_within_four_standard_errors_of_the_true_ate(
    scenario, knobs
) -> None:
    """One fixed dataset per scenario, given a valid adjustment set.

    This checks the implementation on data where its assumptions hold. It is
    not a claim that the estimator is accurate in general.
    """
    dataset = generate(_spec(scenario, n_rows=5000, **knobs))
    result = aipw_estimate(dataset.frame, "t", "y", _valid_set(dataset))
    assert abs(result.estimate - 1.0) <= 4 * result.standard_error
