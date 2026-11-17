"""Fixed-pipeline baseline: no LLM, the same rule for every dataset.

Rule: adjust for every covariate the public metadata labels `pre_treatment`,
and estimate the ATE with the cross-fitted doubly robust (AIPW) estimator.

The pipeline sees exactly what the agent sees: the public CSV and the public
metadata, including timing labels. It never reads ground truth.

The rule is valid only when every pre-treatment covariate is safe to adjust
for and together they block all backdoor paths. That holds in this benchmark
because its graphs have no hidden variables and no pre-treatment colliders;
it is not true of observational data in general.
"""

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from causal_agent.bench.public import Timing, load_public
from causal_agent.tools.aipw import aipw_estimate

ESTIMATOR = "doubly_robust"


class PipelineResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_id: str
    estimate: float
    standard_error: float
    ci_low: float
    ci_high: float
    estimator: str
    adjustment_set: tuple[str, ...]
    diagnostics: dict[str, float]
    warnings: tuple[str, ...]


def run_fixed_pipeline(dataset_id: str, root: Path) -> PipelineResult:
    """Estimate the ATE for one dataset with the fixed rule."""
    frame, metadata = load_public(dataset_id, root)
    adjustment_set = tuple(
        column.name
        for column in metadata.covariates
        if column.timing is Timing.PRE_TREATMENT
    )
    result = aipw_estimate(frame, metadata.treatment, metadata.outcome, adjustment_set)
    ci_low, ci_high = result.confidence_interval
    return PipelineResult(
        dataset_id=dataset_id,
        estimate=result.estimate,
        standard_error=result.standard_error,
        ci_low=ci_low,
        ci_high=ci_high,
        estimator=ESTIMATOR,
        adjustment_set=adjustment_set,
        diagnostics=result.diagnostics,
        warnings=result.warnings,
    )
