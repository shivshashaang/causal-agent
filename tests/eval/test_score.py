"""Tests for scoring answers against ground truth."""

import pytest

from causal_agent.bench.graph import Role
from causal_agent.bench.spec import Scenario
from causal_agent.eval.score import (
    Score,
    method_validity,
    score_answer,
    tolerance_key,
)
from causal_agent.tools.estimators import EstimatorName

TOLERANCES = (0.05, 0.10, 0.20)
DIM = EstimatorName.DIFFERENCE_IN_MEANS
OLS = EstimatorName.LINEAR_REGRESSION
IPW = EstimatorName.PROPENSITY_WEIGHTING
DR = EstimatorName.DOUBLY_ROBUST


def _truth(small_benchmark, scenario):
    return small_benchmark.datasets[scenario].truth


def _columns(truth, role) -> tuple[str, ...]:
    return tuple(sorted(n for n, r in truth.column_roles.items() if r is role))


def _confounders(truth) -> tuple[str, ...]:
    return _columns(truth, Role.CONFOUNDER)


# --- Accuracy --------------------------------------------------------------


def test_tolerance_is_a_multiple_of_the_outcome_sd(small_benchmark) -> None:
    truth = _truth(small_benchmark, Scenario.CONFOUNDED_LINEAR)
    sd = truth.diagnostics.outcome_sd
    just_inside = score_answer("a", truth, 1.0 + 0.099 * sd, OLS, (), TOLERANCES)
    just_outside = score_answer("a", truth, 1.0 + 0.101 * sd, OLS, (), TOLERANCES)
    assert just_inside.within_tolerance == {"0.05": False, "0.10": True, "0.20": True}
    assert just_outside.within_tolerance == {"0.05": False, "0.10": False, "0.20": True}
    assert just_inside.abs_error == pytest.approx(0.099 * sd)
    assert just_inside.abs_error_in_sd == pytest.approx(0.099)


def test_error_keeps_its_sign_and_exact_answers_pass(small_benchmark) -> None:
    truth = _truth(small_benchmark, Scenario.RANDOMIZED)
    low = score_answer("a", truth, 0.5, DIM, (), TOLERANCES)
    assert low.error == pytest.approx(-0.5)
    assert low.abs_error == pytest.approx(0.5)
    exact = score_answer("a", truth, truth.true_ate, DIM, (), TOLERANCES)
    assert all(exact.within_tolerance.values())


def test_no_answer_is_never_within_tolerance(small_benchmark) -> None:
    truth = _truth(small_benchmark, Scenario.RANDOMIZED)
    score = score_answer("a", truth, None, None, (), TOLERANCES)
    assert not score.answered
    assert score.estimate is None and score.error is None and score.abs_error is None
    assert score.within_tolerance == {"0.05": False, "0.10": False, "0.20": False}
    assert score.validity is None


def test_score_identifies_the_dataset_and_approach(small_benchmark) -> None:
    truth = _truth(small_benchmark, Scenario.POST_OUTCOME_TRAP)
    score = score_answer("agent", truth, 1.0, OLS, ("x1",), TOLERANCES)
    assert score.approach == "agent"
    assert score.dataset_id == "ds_904"
    assert score.scenario == "post_outcome_trap"
    assert score.true_ate == 1.0
    assert score.estimator == "linear_regression"
    assert Score.model_validate_json(score.model_dump_json()) == score
    assert tolerance_key(0.1) == "0.10"


# --- Adjustment-set validity -----------------------------------------------


def test_adjustment_validity_follows_the_backdoor_criterion(small_benchmark) -> None:
    truth = _truth(small_benchmark, Scenario.CONFOUNDED_LINEAR)
    confounders = _confounders(truth)
    assert method_validity(truth, OLS, confounders).adjustment_valid
    assert not method_validity(truth, OLS, confounders[:1]).adjustment_valid
    assert not method_validity(truth, OLS, ()).adjustment_valid


def test_adjusting_for_the_post_outcome_column_is_invalid(small_benchmark) -> None:
    truth = _truth(small_benchmark, Scenario.POST_OUTCOME_TRAP)
    confounders = _confounders(truth)
    collider = _columns(truth, Role.POST_OUTCOME_COLLIDER)
    assert method_validity(truth, DR, confounders).estimator_valid
    with_trap = method_validity(truth, DR, (*confounders, *collider))
    assert not with_trap.adjustment_valid
    assert not with_trap.estimator_valid


def test_difference_in_means_always_uses_the_empty_set(small_benchmark) -> None:
    confounded = _truth(small_benchmark, Scenario.CONFOUNDED_LINEAR)
    stated = _confounders(confounded)
    validity = method_validity(confounded, DIM, stated)
    assert not validity.adjustment_valid
    assert not validity.estimator_valid
    score = score_answer("a", confounded, 1.0, DIM, stated, TOLERANCES)
    assert score.adjustment_set == ()

    randomized = _truth(small_benchmark, Scenario.RANDOMIZED)
    assert method_validity(randomized, DIM, ()).estimator_valid


# --- Model assumptions, tracked separately ---------------------------------


@pytest.mark.parametrize(
    ("scenario", "outcome_correct", "propensity_correct"),
    [
        (Scenario.RANDOMIZED, True, True),
        (Scenario.CONFOUNDED_LINEAR, True, True),
        (Scenario.CONFOUNDED_NONLINEAR, False, False),
        (Scenario.POST_OUTCOME_TRAP, True, True),
    ],
)
def test_outcome_and_propensity_correctness_are_separate_facts(
    small_benchmark, scenario, outcome_correct, propensity_correct
) -> None:
    truth = _truth(small_benchmark, scenario)
    validity = method_validity(truth, OLS, _confounders(truth))
    assert validity.outcome_model_correct is outcome_correct
    assert validity.propensity_model_correct is propensity_correct


def test_each_estimator_is_judged_on_its_own_assumption(make_truth) -> None:
    """Nonlinear outcome only: regression's assumption fails, weighting's holds."""
    truth = make_truth(treatment_nonlinearity=0.0, outcome_nonlinearity=1.0)
    adjustment = _confounders(truth)
    assert method_validity(truth, OLS, adjustment).model_assumption_met is False
    assert method_validity(truth, OLS, adjustment).estimator_valid is False
    assert method_validity(truth, IPW, adjustment).model_assumption_met is True
    assert method_validity(truth, IPW, adjustment).estimator_valid is True


def test_nonlinear_assignment_does_not_invalidate_linear_regression(make_truth) -> None:
    """Nonlinear treatment only: regression's assumption holds, weighting's fails."""
    truth = make_truth(treatment_nonlinearity=1.0, outcome_nonlinearity=0.0)
    adjustment = _confounders(truth)
    assert method_validity(truth, OLS, adjustment).estimator_valid is True
    assert method_validity(truth, IPW, adjustment).model_assumption_met is False
    assert method_validity(truth, IPW, adjustment).estimator_valid is False


def test_randomization_rescues_a_misspecified_regression(make_truth) -> None:
    truth = make_truth(
        scenario=Scenario.RANDOMIZED,
        assignment_strength=0.0,
        treatment_nonlinearity=0.0,
        outcome_nonlinearity=1.0,
    )
    assert method_validity(truth, OLS, ()).model_assumption_met is True


def test_doubly_robust_model_fit_is_recorded_as_unknown(small_benchmark) -> None:
    """Flexible estimation is not guaranteed consistent; the truth cannot say."""
    truth = _truth(small_benchmark, Scenario.CONFOUNDED_NONLINEAR)
    validity = method_validity(truth, DR, _confounders(truth))
    assert validity.model_assumption_met is None
    assert validity.estimator_valid is True
    assert not method_validity(truth, DR, ()).estimator_valid


def test_unrecognized_estimator_has_no_estimator_verdict(small_benchmark) -> None:
    truth = _truth(small_benchmark, Scenario.CONFOUNDED_LINEAR)
    validity = method_validity(truth, None, _confounders(truth))
    assert validity.adjustment_valid
    assert validity.model_assumption_met is None
    assert validity.estimator_valid is None


def test_poor_overlap_invalidates_every_estimator(small_benchmark) -> None:
    truth = _truth(small_benchmark, Scenario.CONFOUNDED_LINEAR)
    poor = truth.model_copy(
        update={
            "diagnostics": truth.diagnostics.model_copy(update={"overlap_share": 0.5})
        }
    )
    validity = method_validity(poor, DR, _confounders(poor))
    assert validity.adjustment_valid
    assert not validity.overlap_adequate
    assert not validity.estimator_valid


def test_validity_and_accuracy_are_independent(small_benchmark) -> None:
    truth = _truth(small_benchmark, Scenario.CONFOUNDED_LINEAR)
    valid_but_wrong = score_answer(
        "a", truth, 99.0, OLS, _confounders(truth), TOLERANCES
    )
    assert valid_but_wrong.validity.estimator_valid
    assert not valid_but_wrong.within_tolerance["0.10"]
    invalid_but_right = score_answer("a", truth, 1.0, DIM, (), TOLERANCES)
    assert not invalid_but_right.validity.estimator_valid
    assert invalid_but_right.within_tolerance["0.10"]
