"""Typed specification of one synthetic dataset.

A spec holds every knob of the data-generating process, including the true
ATE. It is private to the generator and the evaluation code and must never
reach the agent or a baseline.
"""

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_N_ROWS = 5000

# Strength `d` of the outcome-only predictor `p`. It adds variance d**2 to the
# outcome and never causes bias: adjusting for `p` only improves precision.
DEFAULT_OUTCOME_PREDICTOR_STRENGTH = 1.0


class Scenario(StrEnum):
    """The four causal structures in the benchmark."""

    RANDOMIZED = "randomized"
    CONFOUNDED_LINEAR = "confounded_linear"
    CONFOUNDED_NONLINEAR = "confounded_nonlinear"
    POST_OUTCOME_TRAP = "post_outcome_trap"


class NonlinearFamily(StrEnum):
    """Which nonlinear functions the treatment and outcome indices use.

    The family only matters when `treatment_nonlinearity` or
    `outcome_nonlinearity` is positive.
    """

    V1 = "v1"
    """Benchmark v1: the treatment and the outcome each have their own function."""

    SHARED = "shared"
    """Benchmark v2: the treatment and the outcome use the same function."""


class DatasetSpec(BaseModel):
    """Every parameter needed to generate one dataset reproducibly."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    dataset_id: str = Field(min_length=1)
    scenario: Scenario
    seed: int = Field(ge=0)
    n_rows: int = Field(default=DEFAULT_N_ROWS, ge=10)

    true_ate: float
    """tau: the constant treatment effect, which is also the true ATE."""

    assignment_strength: float = Field(ge=0)
    """a: how strongly the confounders drive treatment. Zero means randomized."""

    confounder_outcome_strength: float = Field(ge=0)
    """b: how strongly the confounders drive the outcome."""

    outcome_noise: float = Field(gt=0)
    """sigma: standard deviation of the outcome noise."""

    treatment_nonlinearity: float = Field(default=0.0, ge=0)
    """lambda_T: weight of the nonlinear part of the treatment index."""

    outcome_nonlinearity: float = Field(default=0.0, ge=0)
    """lambda_Y: weight of the nonlinear part of the outcome index."""

    nonlinear_family: NonlinearFamily = NonlinearFamily.V1
    """Which nonlinear functions lambda_T and lambda_Y switch on."""

    outcome_predictor_strength: float = Field(
        default=DEFAULT_OUTCOME_PREDICTOR_STRENGTH, ge=0
    )
    """d: strength of the outcome-only predictor."""

    trap_strength: float = Field(default=0.0, ge=0)
    """gamma: how closely the post-outcome collider tracks T and Y."""

    @model_validator(mode="after")
    def _scenario_matches_knobs(self) -> Self:
        confounded = (
            self.assignment_strength > 0 and self.confounder_outcome_strength > 0
        )
        nonlinear = self.treatment_nonlinearity > 0 or self.outcome_nonlinearity > 0
        has_trap = self.trap_strength > 0

        match self.scenario:
            case Scenario.RANDOMIZED:
                consistent = (
                    self.assignment_strength == 0
                    and self.treatment_nonlinearity == 0
                    and not has_trap
                )
                rule = "a == 0, lambda_T == 0, gamma == 0"
            case Scenario.CONFOUNDED_LINEAR:
                consistent = confounded and not nonlinear and not has_trap
                rule = "a > 0, b > 0, lambda_T == lambda_Y == 0, gamma == 0"
            case Scenario.CONFOUNDED_NONLINEAR:
                consistent = confounded and nonlinear and not has_trap
                rule = "a > 0, b > 0, lambda_T > 0 or lambda_Y > 0, gamma == 0"
            case Scenario.POST_OUTCOME_TRAP:
                consistent = confounded and not nonlinear and has_trap
                rule = "a > 0, b > 0, lambda_T == lambda_Y == 0, gamma > 0"

        if not consistent:
            raise ValueError(
                f"scenario '{self.scenario.value}' requires {rule}; "
                "the knobs given do not match"
            )
        if self.nonlinear_family is not NonlinearFamily.V1 and not nonlinear:
            raise ValueError(
                f"nonlinear_family '{self.nonlinear_family.value}' requires "
                "lambda_T > 0 or lambda_Y > 0; with both at zero it has no effect"
            )
        return self
