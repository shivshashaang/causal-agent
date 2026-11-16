"""Tool: estimate_effect.

Estimates the average treatment effect on one dataset with a chosen estimator
and adjustment set. The caller chooses both; the tool does not judge whether
the adjustment set is valid, and it does not hide any column. Its output
states the assumptions the chosen estimator relies on.

The adjustment set and estimator are echoed in the output so that every call
can be logged and scored later.
"""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from causal_agent.bench.public import load_public
from causal_agent.tools.estimators import ASSUMPTIONS, EstimatorName, estimate_ate


class EstimateEffectInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_id: str
    estimator: EstimatorName
    adjustment_set: tuple[str, ...] = Field(default=())
    """Covariate columns to adjust for. Must be empty for difference_in_means."""

    @model_validator(mode="after")
    def _check_adjustment_set(self) -> "EstimateEffectInput":
        if len(set(self.adjustment_set)) != len(self.adjustment_set):
            raise ValueError("adjustment_set contains a column more than once")
        if self.estimator is EstimatorName.DIFFERENCE_IN_MEANS and self.adjustment_set:
            raise ValueError(
                "difference_in_means always uses the empty adjustment set; "
                "pass an empty adjustment_set or choose another estimator"
            )
        return self


class EstimateEffectOutput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_id: str
    estimator: EstimatorName
    adjustment_set: tuple[str, ...]
    estimate: float
    standard_error: float
    ci_low: float
    ci_high: float
    n_rows: int
    n_treated: int
    diagnostics: dict[str, float]
    warnings: tuple[str, ...]
    assumptions: tuple[str, ...]


def estimate_effect(request: EstimateEffectInput, root: Path) -> EstimateEffectOutput:
    """Run one estimator on one dataset."""
    frame, metadata = load_public(request.dataset_id, root)
    covariates = {column.name for column in metadata.covariates}
    unknown = [name for name in request.adjustment_set if name not in covariates]
    if unknown:
        raise ValueError(
            f"adjustment_set may only contain covariate columns "
            f"{sorted(covariates)}; got {unknown}"
        )

    result = estimate_ate(
        request.estimator,
        frame,
        metadata.treatment,
        metadata.outcome,
        request.adjustment_set,
    )
    ci_low, ci_high = result.confidence_interval
    return EstimateEffectOutput(
        dataset_id=request.dataset_id,
        estimator=request.estimator,
        adjustment_set=request.adjustment_set,
        estimate=result.estimate,
        standard_error=result.standard_error,
        ci_low=ci_low,
        ci_high=ci_high,
        n_rows=result.n_rows,
        n_treated=result.n_treated,
        diagnostics=result.diagnostics,
        warnings=result.warnings,
        assumptions=ASSUMPTIONS[request.estimator],
    )
