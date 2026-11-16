"""Score one answer against a dataset's ground truth.

Accuracy and validity are separate questions and are recorded separately:

- Accuracy: is the estimate within a tolerance of the true ATE? The tolerance
  is a multiple of the outcome's standard deviation.
- Validity: do the method's assumptions hold for this dataset? A valid method
  can still be inaccurate in a finite sample, and an invalid one can land
  close by luck.

Validity has three parts, each recorded on its own:

1. Adjustment set: passes the backdoor criterion on the true graph.
2. Model assumption of the chosen estimator:
   - difference_in_means: none (it always uses the empty adjustment set, so
     its validity is entirely the adjustment-set question).
   - linear_regression: the outcome model is linear, or treatment is
     randomized.
   - propensity_weighting: the propensity model is logit-linear.
   - doubly_robust: whether its boosted models fit well enough cannot be read
     from the ground truth, so this is recorded as unknown (None), never as
     guaranteed.
3. Overlap: the dataset's true propensities are mostly away from 0 and 1.

`estimator_valid` is True when the adjustment set is valid, overlap is
adequate and the model assumption is not known to fail. For the doubly robust
estimator that means "valid as far as the ground truth can show".
"""

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from causal_agent.bench.truth import GroundTruth
from causal_agent.tools.estimators import EstimatorName

MIN_OVERLAP_SHARE = 0.90


class MethodValidity(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    adjustment_valid: bool
    outcome_model_correct: bool
    """A linear outcome model would be correct for this dataset."""

    propensity_model_correct: bool
    """A logistic propensity model would be correct for this dataset."""

    model_assumption_met: bool | None
    """For the chosen estimator. None when it cannot be verified or no estimator."""

    overlap_adequate: bool
    estimator_valid: bool | None
    """None when the answer names no recognized estimator."""


class Score(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    approach: str
    dataset_id: str
    scenario: str
    true_ate: float
    outcome_sd: float
    answered: bool
    estimate: float | None
    error: float | None
    abs_error: float | None
    abs_error_in_sd: float | None
    within_tolerance: dict[str, bool]
    """Keyed by tolerance in SD units, such as "0.10". False when unanswered."""

    estimator: str | None
    adjustment_set: tuple[str, ...]
    validity: MethodValidity | None
    """None when there is no answer."""


def tolerance_key(tolerance_sd: float) -> str:
    return f"{tolerance_sd:.2f}"


def method_validity(
    truth: GroundTruth, estimator: EstimatorName | None, adjustment_set: Sequence[str]
) -> MethodValidity:
    """Check a method's assumptions against the ground truth."""
    assumptions = truth.assumptions
    outcome_model_correct = (
        assumptions.outcome_model_linear or assumptions.treatment_randomized
    )
    propensity_model_correct = assumptions.propensity_logit_linear
    overlap_adequate = truth.diagnostics.overlap_share >= MIN_OVERLAP_SHARE

    if estimator is EstimatorName.DIFFERENCE_IN_MEANS:
        # Difference in means always uses the empty adjustment set.
        adjustment_set = ()
    adjustment_valid = truth.is_valid_adjustment_set(adjustment_set)

    model_assumption_met: bool | None
    if estimator is EstimatorName.DIFFERENCE_IN_MEANS:
        model_assumption_met = True
    elif estimator is EstimatorName.LINEAR_REGRESSION:
        model_assumption_met = outcome_model_correct
    elif estimator is EstimatorName.PROPENSITY_WEIGHTING:
        model_assumption_met = propensity_model_correct
    else:
        model_assumption_met = None

    estimator_valid: bool | None
    if estimator is None:
        estimator_valid = None
    else:
        estimator_valid = (
            adjustment_valid and overlap_adequate and model_assumption_met is not False
        )

    return MethodValidity(
        adjustment_valid=adjustment_valid,
        outcome_model_correct=outcome_model_correct,
        propensity_model_correct=propensity_model_correct,
        model_assumption_met=model_assumption_met,
        overlap_adequate=overlap_adequate,
        estimator_valid=estimator_valid,
    )


def score_answer(
    approach: str,
    truth: GroundTruth,
    estimate: float | None,
    estimator: EstimatorName | None,
    adjustment_set: Sequence[str],
    tolerances_sd: Sequence[float],
) -> Score:
    """Score one approach's answer for one dataset. `estimate` None = no answer."""
    outcome_sd = truth.diagnostics.outcome_sd
    answered = estimate is not None
    error = estimate - truth.true_ate if answered else None
    abs_error = abs(error) if answered else None
    if estimator is EstimatorName.DIFFERENCE_IN_MEANS:
        adjustment_set = ()

    return Score(
        approach=approach,
        dataset_id=truth.dataset_id,
        scenario=truth.spec.scenario.value,
        true_ate=truth.true_ate,
        outcome_sd=outcome_sd,
        answered=answered,
        estimate=estimate,
        error=error,
        abs_error=abs_error,
        abs_error_in_sd=abs_error / outcome_sd if answered else None,
        within_tolerance={
            tolerance_key(tolerance): answered and abs_error <= tolerance * outcome_sd
            for tolerance in tolerances_sd
        },
        estimator=estimator.value if estimator else None,
        adjustment_set=tuple(adjustment_set),
        validity=(
            method_validity(truth, estimator, adjustment_set) if answered else None
        ),
    )
