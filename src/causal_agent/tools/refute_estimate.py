"""Tool: refute_estimate.

Re-runs an estimate under a change that should not matter if the estimate is
sound, and reports whether the estimate held up. Three checks:

- `placebo_treatment`: shuffle the treatment column, which destroys any real
  effect. The estimator should now find an effect near zero.
- `random_common_cause`: add a column of pure noise to the adjustment set. The
  estimate should barely move.
- `data_subset`: re-estimate on random 80% subsets of the rows. The estimate
  should barely move.

Each check is repeated with fixed seeds. The yardstick is the original
estimate's own standard error: a check fails when the placebo effect, or the
average shift of the estimate, is larger than one standard error.

What a pass does not mean. These checks detect an estimator that finds
effects in noise and estimates that are unstable. They cannot detect a wrong
adjustment set (a missed confounder, or a column affected by the treatment)
or a misspecified model: a biased estimate is usually just as stable as an
unbiased one. Passing is not evidence that the estimate is unbiased.

A fourth check, `estimator_disagreement`, targets a misspecified model. It
re-estimates the effect with an estimator from the other model family on the
same adjustment set (the cross-fitted doubly robust estimator, or linear
regression when the original is doubly robust) and asks whether the two
answers differ by more than a materiality bound.

    difference = original estimate - comparison estimate
    paired SE  = sd(row contributions of original - row contributions of
                 comparison) / sqrt(n)
    interval   = difference +/- 2 * paired SE
    bound      = 0.10 * standard deviation of the outcome

    disagreement   the whole interval lies outside [-bound, +bound]
    agreement      the whole interval lies inside  [-bound, +bound]
    inconclusive   the interval straddles the bound; or the adjustment set is
                   empty, so there is no model to compare; or more than 5% of
                   rows had a clipped propensity, so overlap is too poor for
                   the comparison to mean much

Both estimates come from the same rows, so they are correlated. The paired
standard error is computed from each row's contribution to each estimate (the
contributions of an estimator average to its estimate), which accounts for
that. It is approximate: it treats the fitted outcome and propensity models as
fixed, so it understates the uncertainty, most when overlap is weak.

What the verdicts do not mean. Disagreement shows that the two estimators
cannot both be right; it does not show which one is wrong and is not proof
that the original estimate is biased. Agreement is not evidence of an unbiased
estimate: both estimators use the same adjustment set, so a wrong adjustment
set biases both. The bound and the 2-standard-error interval are exploratory
choices.
"""

from enum import StrEnum
from pathlib import Path
from typing import Literal, Self

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from causal_agent.bench.public import load_public
from causal_agent.tools.aipw import CLIPPED_SHARE_WARNING, EffectEstimate
from causal_agent.tools.estimators import EstimatorName, estimate_ate

N_REPETITIONS = 10
SUBSET_FRACTION = 0.8
FAIL_THRESHOLD_IN_STANDARD_ERRORS = 1.0
REFUTER_SEED = 2026
NOISE_COLUMN = "_random_common_cause"

# Thresholds of the estimator-disagreement check. Fixed before Benchmark v2
# was run; exploratory, not calibrated.
MATERIALITY_IN_OUTCOME_SD = 0.10
"""Differences smaller than this many outcome standard deviations do not matter."""

INTERVAL_STANDARD_ERRORS = 2.0
"""Half-width of the interval around the difference, in paired standard errors."""

MAX_CLIPPED_SHARE = CLIPPED_SHARE_WARNING
"""Above this share of clipped propensities the check is inconclusive."""

CONTRIBUTION_TOLERANCE = 1e-6
"""How closely an estimator's row contributions must average to its estimate."""

LIMITATIONS = (
    "A pass is not evidence that the estimate is unbiased.",
    "These checks detect estimators that find effects in noise and estimates "
    "that are unstable.",
    "They cannot detect a wrong adjustment set (a missed confounder, or a "
    "column affected by the treatment) or a misspecified model.",
)


DISAGREEMENT_LIMITATIONS = (
    "Disagreement is not proof that the original estimate is biased. It shows "
    "that the two estimators cannot both be right, not which one is wrong.",
    "Agreement is not evidence that the estimate is unbiased. Both estimators "
    "use the same adjustment set, so a wrong adjustment set biases both.",
    "The paired standard error is approximate: it treats the fitted outcome "
    "and propensity models as fixed, so it understates the uncertainty, most "
    "when overlap is weak.",
    "The materiality bound of 0.10 outcome standard deviations and the "
    "2-standard-error interval are exploratory choices.",
)


class RefuterName(StrEnum):
    PLACEBO_TREATMENT = "placebo_treatment"
    RANDOM_COMMON_CAUSE = "random_common_cause"
    DATA_SUBSET = "data_subset"
    ESTIMATOR_DISAGREEMENT = "estimator_disagreement"


# The three original checks: each re-runs the same estimator and passes or fails.
STABILITY_REFUTERS = (
    RefuterName.PLACEBO_TREATMENT,
    RefuterName.RANDOM_COMMON_CAUSE,
    RefuterName.DATA_SUBSET,
)

Verdict = Literal["disagreement", "agreement", "inconclusive"]


class RefuteEstimateInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_id: str
    estimator: EstimatorName
    adjustment_set: tuple[str, ...] = Field(default=())
    refuter: RefuterName

    @model_validator(mode="after")
    def _check_combination(self) -> Self:
        if len(set(self.adjustment_set)) != len(self.adjustment_set):
            raise ValueError("adjustment_set contains a column more than once")
        if self.estimator is EstimatorName.DIFFERENCE_IN_MEANS:
            if self.adjustment_set:
                raise ValueError(
                    "difference_in_means always uses the empty adjustment set; "
                    "pass an empty adjustment_set or choose another estimator"
                )
            if self.refuter is RefuterName.RANDOM_COMMON_CAUSE:
                raise ValueError(
                    "random_common_cause does not apply to difference_in_means, "
                    "which cannot take an extra adjustment column"
                )
        return self


class RefuteEstimateOutput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_id: str
    estimator: EstimatorName
    adjustment_set: tuple[str, ...]
    refuter: RefuterName
    original_estimate: float
    original_standard_error: float
    n_repetitions: int
    refuted_mean: float
    """Mean of the re-estimates."""

    refuted_sd: float
    """Standard deviation of the re-estimates."""

    expected: float
    """Value the re-estimates should be near: 0 for the placebo, else the original."""

    shift_in_standard_errors: float
    """|refuted_mean - expected| divided by the original standard error."""

    threshold_in_standard_errors: float
    passed: bool
    interpretation: str
    limitations: tuple[str, ...]


class EstimatorDisagreementOutput(BaseModel):
    """Result of the `estimator_disagreement` check."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_id: str
    estimator: EstimatorName
    adjustment_set: tuple[str, ...]
    refuter: RefuterName
    original_estimate: float
    comparison_estimator: EstimatorName | None
    """The estimator the original was compared with. None when not compared."""

    comparison_estimate: float | None
    difference: float | None
    """Original estimate minus comparison estimate."""

    paired_standard_error: float | None
    """Approximate standard error of the difference, from paired row contributions."""

    interval_low: float | None
    interval_high: float | None
    outcome_sd: float
    materiality_bound: float
    """0.10 outcome standard deviations, in outcome units."""

    difference_in_outcome_sd: float | None
    verdict: Verdict
    reason: str
    """Which rule produced the verdict."""

    interpretation: str
    limitations: tuple[str, ...]


def comparison_estimator(estimator: EstimatorName) -> EstimatorName:
    """The estimator from the other model family."""
    if estimator is EstimatorName.DOUBLY_ROBUST:
        return EstimatorName.LINEAR_REGRESSION
    return EstimatorName.DOUBLY_ROBUST


def checked_contributions(result: EffectEstimate) -> np.ndarray:
    """An estimate's row contributions, verified to average to the estimate."""
    contributions = result.contributions
    if contributions is None or len(contributions) != result.n_rows:
        raise ValueError("the estimator did not return one contribution per row")
    gap = abs(float(contributions.mean()) - result.estimate)
    if gap > CONTRIBUTION_TOLERANCE * max(1.0, abs(result.estimate)):
        raise ValueError(
            "the row contributions do not average to the estimate "
            f"(off by {gap:.2e}); the paired standard error would be wrong"
        )
    return contributions


def paired_standard_error(first: EffectEstimate, second: EffectEstimate) -> float:
    """Approximate standard error of the difference of two estimates on one dataset.

    Uses the row-by-row difference of the two estimators' contributions, so the
    correlation between the two estimates is accounted for.
    """
    difference = checked_contributions(first) - checked_contributions(second)
    return float(difference.std(ddof=1) / np.sqrt(len(difference)))


def disagreement_verdict(
    difference: float, standard_error: float, bound: float
) -> tuple[Verdict, str]:
    """Compare the interval around a difference with the materiality bound."""
    half_width = INTERVAL_STANDARD_ERRORS * standard_error
    if abs(difference) - half_width > bound:
        return "disagreement", "the whole interval lies outside the materiality bound"
    if abs(difference) + half_width < bound:
        return "agreement", "the whole interval lies inside the materiality bound"
    return "inconclusive", "the interval straddles the materiality bound"


def _disagreement_interpretation(
    verdict: Verdict,
    reason: str,
    original: EstimatorName,
    comparison: EstimatorName | None,
    difference_in_sd: float | None,
) -> str:
    if comparison is None or difference_in_sd is None:
        return f"Inconclusive: {reason}. No comparison was made."
    pair = f"{original.value} and {comparison.value}"
    size = f"{abs(difference_in_sd):.3f} outcome standard deviations"
    if verdict == "disagreement":
        return (
            f"Disagreement: {pair} differ by {size}, and {reason}. At least one "
            "of the two does not fit this data. This does not show which one, "
            "and it is not proof that the original estimate is biased."
        )
    if verdict == "agreement":
        return (
            f"Agreement: {pair} differ by {size}, and {reason}. This does not "
            "show that the estimate is unbiased: both use the same adjustment "
            "set."
        )
    return (
        f"Inconclusive: {pair} differ by {size}, and {reason}. The check "
        "cannot say whether the two estimators materially agree."
    )


def _estimator_disagreement(
    request: RefuteEstimateInput,
    frame: pd.DataFrame,
    treatment: str,
    outcome: str,
) -> EstimatorDisagreementOutput:
    original = estimate_ate(
        request.estimator, frame, treatment, outcome, request.adjustment_set
    )
    outcome_sd = float(frame[outcome].std(ddof=1))
    bound = MATERIALITY_IN_OUTCOME_SD * outcome_sd

    def output(
        verdict: Verdict,
        reason: str,
        comparison_name: EstimatorName | None = None,
        comparison: EffectEstimate | None = None,
        standard_error: float | None = None,
    ) -> EstimatorDisagreementOutput:
        difference = (
            None if comparison is None else original.estimate - comparison.estimate
        )
        compared = difference is not None and standard_error is not None
        half_width = INTERVAL_STANDARD_ERRORS * standard_error if compared else None
        in_sd = difference / outcome_sd if difference is not None else None
        return EstimatorDisagreementOutput(
            dataset_id=request.dataset_id,
            estimator=request.estimator,
            adjustment_set=request.adjustment_set,
            refuter=request.refuter,
            original_estimate=original.estimate,
            comparison_estimator=comparison_name,
            comparison_estimate=None if comparison is None else comparison.estimate,
            difference=difference,
            paired_standard_error=standard_error,
            interval_low=difference - half_width if compared else None,
            interval_high=difference + half_width if compared else None,
            outcome_sd=outcome_sd,
            materiality_bound=bound,
            difference_in_outcome_sd=in_sd,
            verdict=verdict,
            reason=reason,
            interpretation=_disagreement_interpretation(
                verdict, reason, request.estimator, comparison_name, in_sd
            ),
            limitations=DISAGREEMENT_LIMITATIONS,
        )

    if not request.adjustment_set:
        return output(
            "inconclusive",
            "the adjustment set is empty, so there is no model to compare",
        )

    comparison_name = comparison_estimator(request.estimator)
    comparison = estimate_ate(
        comparison_name, frame, treatment, outcome, request.adjustment_set
    )
    standard_error = paired_standard_error(original, comparison)
    clipped_share = max(
        original.diagnostics.get("clipped_share", 0.0),
        comparison.diagnostics.get("clipped_share", 0.0),
    )
    if clipped_share > MAX_CLIPPED_SHARE:
        return output(
            "inconclusive",
            f"{clipped_share:.1%} of rows had a clipped propensity, so overlap "
            "is too poor for the comparison",
            comparison_name,
            comparison,
            standard_error,
        )
    label, reason = disagreement_verdict(
        original.estimate - comparison.estimate, standard_error, bound
    )
    return output(label, reason, comparison_name, comparison, standard_error)


def verdict(shift_in_standard_errors: float) -> bool:
    """A check passes when the shift is at most the threshold."""
    return shift_in_standard_errors <= FAIL_THRESHOLD_IN_STANDARD_ERRORS


def _re_estimate(
    request: RefuteEstimateInput,
    frame: pd.DataFrame,
    treatment: str,
    outcome: str,
    repetition: int,
) -> float:
    rng = np.random.default_rng([REFUTER_SEED, repetition])
    adjustment_set = request.adjustment_set
    if request.refuter is RefuterName.PLACEBO_TREATMENT:
        changed = frame.assign(**{treatment: rng.permutation(frame[treatment])})
    elif request.refuter is RefuterName.RANDOM_COMMON_CAUSE:
        changed = frame.assign(**{NOISE_COLUMN: rng.standard_normal(len(frame))})
        adjustment_set = (*adjustment_set, NOISE_COLUMN)
    else:
        size = int(round(SUBSET_FRACTION * len(frame)))
        rows = rng.choice(len(frame), size=size, replace=False)
        changed = frame.iloc[np.sort(rows)].reset_index(drop=True)
    return estimate_ate(
        request.estimator, changed, treatment, outcome, adjustment_set
    ).estimate


def _interpretation(request: RefuteEstimateInput, passed: bool, shift: float) -> str:
    if request.refuter is RefuterName.PLACEBO_TREATMENT:
        what = f"the placebo effect was {shift:.2f} standard errors from zero"
    else:
        what = f"the estimate moved by {shift:.2f} standard errors on average"
    if passed:
        return (
            f"Passed: {what}. The estimate is stable under this check, which "
            "does not show that it is unbiased."
        )
    return (
        f"Failed: {what}, more than the threshold of "
        f"{FAIL_THRESHOLD_IN_STANDARD_ERRORS:g}. Treat the estimate with caution."
    )


def refute_estimate(
    request: RefuteEstimateInput, root: Path
) -> RefuteEstimateOutput | EstimatorDisagreementOutput:
    """Run one refutation check on one estimate."""
    frame, metadata = load_public(request.dataset_id, root)
    covariates = {column.name for column in metadata.covariates}
    unknown = [name for name in request.adjustment_set if name not in covariates]
    if unknown:
        raise ValueError(
            f"adjustment_set may only contain covariate columns "
            f"{sorted(covariates)}; got {unknown}"
        )
    treatment, outcome = metadata.treatment, metadata.outcome
    if request.refuter is RefuterName.ESTIMATOR_DISAGREEMENT:
        return _estimator_disagreement(request, frame, treatment, outcome)

    original = estimate_ate(
        request.estimator, frame, treatment, outcome, request.adjustment_set
    )
    re_estimates = np.array(
        [
            _re_estimate(request, frame, treatment, outcome, repetition)
            for repetition in range(N_REPETITIONS)
        ]
    )
    expected = (
        0.0 if request.refuter is RefuterName.PLACEBO_TREATMENT else original.estimate
    )
    shift = abs(float(re_estimates.mean()) - expected) / original.standard_error
    passed = verdict(shift)
    return RefuteEstimateOutput(
        dataset_id=request.dataset_id,
        estimator=request.estimator,
        adjustment_set=request.adjustment_set,
        refuter=request.refuter,
        original_estimate=original.estimate,
        original_standard_error=original.standard_error,
        n_repetitions=N_REPETITIONS,
        refuted_mean=float(re_estimates.mean()),
        refuted_sd=float(re_estimates.std(ddof=1)),
        expected=expected,
        shift_in_standard_errors=shift,
        threshold_in_standard_errors=FAIL_THRESHOLD_IN_STANDARD_ERRORS,
        passed=passed,
        interpretation=_interpretation(request, passed, shift),
        limitations=LIMITATIONS,
    )
