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
"""

from enum import StrEnum
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from causal_agent.bench.public import load_public
from causal_agent.tools.estimators import EstimatorName, estimate_ate

N_REPETITIONS = 10
SUBSET_FRACTION = 0.8
FAIL_THRESHOLD_IN_STANDARD_ERRORS = 1.0
REFUTER_SEED = 2026
NOISE_COLUMN = "_random_common_cause"

LIMITATIONS = (
    "A pass is not evidence that the estimate is unbiased.",
    "These checks detect estimators that find effects in noise and estimates "
    "that are unstable.",
    "They cannot detect a wrong adjustment set (a missed confounder, or a "
    "column affected by the treatment) or a misspecified model.",
)


class RefuterName(StrEnum):
    PLACEBO_TREATMENT = "placebo_treatment"
    RANDOM_COMMON_CAUSE = "random_common_cause"
    DATA_SUBSET = "data_subset"


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


def refute_estimate(request: RefuteEstimateInput, root: Path) -> RefuteEstimateOutput:
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
