"""Population-level facts about a spec, recorded but never used to pick datasets.

`linear_regression_bias` answers one question about a spec: with unlimited
data, how far from the true effect would a linear regression land if it
adjusted for every pre-treatment covariate? A linear regression is biased when
the part of the outcome that a straight line cannot explain is correlated with
the part of the treatment that a straight line cannot explain.

The number is an approximation from the fixed 1,000,000-row reference sample,
so it comes with a standard error. It describes the data-generating process;
it is not a result of any approach, and no dataset is included or excluded
because of it (ADR 004).

This module is private: it reads specs and potential-outcome parameters.
"""

from dataclasses import dataclass

import numpy as np

from causal_agent.bench.equations import outcome_reference_stats, reference_sample
from causal_agent.bench.graph import TRAP, covariate_names
from causal_agent.bench.spec import DatasetSpec

# Knobs that do not change the joint distribution of pre-treatment covariates,
# treatment and outcome.
_IRRELEVANT_KNOBS = frozenset({"dataset_id", "seed", "n_rows", "trap_strength"})
_cache: dict[tuple[object, ...], "LinearRegressionBias"] = {}


@dataclass(frozen=True)
class LinearRegressionBias:
    """Large-sample error of linear regression on all pre-treatment covariates."""

    bias: float
    """Estimate minus the true effect, in outcome units."""

    standard_error: float
    """Sampling error of `bias` from the finite reference sample."""

    outcome_sd: float
    """Standard deviation of the outcome under the spec."""

    @property
    def bias_in_sd(self) -> float:
        """The bias in the unit the tolerance uses: outcome standard deviations."""
        return self.bias / self.outcome_sd


def linear_regression_bias(spec: DatasetSpec) -> LinearRegressionBias:
    """Approximate population bias of linear regression for a spec."""
    key = tuple(
        value
        for name, value in sorted(spec.model_dump().items())
        if name not in _IRRELEVANT_KNOBS
    )
    if key not in _cache:
        sample = reference_sample(spec)
        pre_treatment = [name for name in covariate_names(spec) if name != TRAP]
        design = np.column_stack(
            [
                np.ones(len(sample.y)),
                sample.t,
                *(getattr(sample, name) for name in pre_treatment),
            ]
        )
        gram = design.T @ design
        coefficients = np.linalg.solve(gram, design.T @ sample.y)
        residuals = sample.y - design @ coefficients
        # Robust (sandwich) variance: valid when the linear model is wrong,
        # which is the case this number exists to describe.
        scores = design * residuals[:, None]
        bread = np.linalg.inv(gram)
        covariance = bread @ (scores.T @ scores) @ bread
        _cache[key] = LinearRegressionBias(
            bias=float(coefficients[1] - spec.true_ate),
            standard_error=float(np.sqrt(covariance[1, 1])),
            outcome_sd=outcome_reference_stats(spec)[1],
        )
    return _cache[key]
