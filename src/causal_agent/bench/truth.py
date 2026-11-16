"""Private ground truth of a benchmark dataset.

Only the generator and the evaluation code may use this module. Nothing here
may be shown to the agent or to a baseline: it holds the true ATE, the
generating parameters and the true causal graph.

Column names in a `GroundTruth` are the public names (`x1..x4`, `t`, `y`), so
evaluation code can compare them directly with what a method used.
"""

from collections.abc import Iterable, Mapping
from pathlib import Path

import networkx as nx
import numpy as np
from pydantic import BaseModel, ConfigDict

from causal_agent.bench.equations import StructuralSample
from causal_agent.bench.graph import (
    OUTCOME,
    TREATMENT,
    Role,
    column_roles,
    covariate_names,
    true_graph,
    valid_adjustment_sets,
)
from causal_agent.bench.public import validate_dataset_id
from causal_agent.bench.spec import DatasetSpec

TRUTH_DIRECTORY = "truth"

# Propensities inside these bounds count as having adequate overlap.
OVERLAP_LOWER = 0.05
OVERLAP_UPPER = 0.95


class Assumptions(BaseModel):
    """Facts about the data-generating process that estimators rely on.

    Derived from the knobs, not from the scenario label. They say whether an
    assumption holds; they do not say that any estimator will be accurate.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    treatment_randomized: bool
    """Treatment is independent of every covariate."""

    outcome_model_linear: bool
    """E[Y | T, covariates] is linear in the covariates with an additive T."""

    propensity_logit_linear: bool
    """The log-odds of treatment are linear in the covariates."""


class Diagnostics(BaseModel):
    """Descriptive facts about one generated dataset. Reported, never asserted."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    naive_difference: float
    naive_bias: float
    outcome_sd: float
    treated_share: float
    propensity_min: float
    propensity_max: float
    overlap_share: float


class GroundTruth(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_id: str
    spec: DatasetSpec
    true_ate: float
    treatment: str
    outcome: str
    internal_names: dict[str, str]
    """Public column name -> the generator's internal variable name."""

    column_roles: dict[str, Role]
    edges: tuple[tuple[str, str], ...]
    valid_adjustment_sets: tuple[tuple[str, ...], ...]
    assumptions: Assumptions
    diagnostics: Diagnostics

    def is_valid_adjustment_set(self, adjustment: Iterable[str]) -> bool:
        """Whether a set of public column names passes the backdoor criterion."""
        return tuple(sorted(set(adjustment))) in self.valid_adjustment_sets

    def graph(self) -> nx.DiGraph:
        """The true causal graph, in public column names."""
        graph = nx.DiGraph()
        graph.add_nodes_from([self.treatment, self.outcome, *self.column_roles])
        graph.add_edges_from(self.edges)
        return graph


def derive_assumptions(spec: DatasetSpec) -> Assumptions:
    randomized = spec.assignment_strength == 0
    return Assumptions(
        treatment_randomized=randomized,
        outcome_model_linear=spec.outcome_nonlinearity == 0,
        propensity_logit_linear=randomized or spec.treatment_nonlinearity == 0,
    )


def _diagnostics(spec: DatasetSpec, sample: StructuralSample) -> Diagnostics:
    treated = sample.t == 1
    naive_difference = float(sample.y[treated].mean() - sample.y[~treated].mean())
    inside = (sample.propensity >= OVERLAP_LOWER) & (sample.propensity <= OVERLAP_UPPER)
    return Diagnostics(
        naive_difference=naive_difference,
        naive_bias=naive_difference - spec.true_ate,
        outcome_sd=float(sample.y.std(ddof=1)),
        treated_share=float(treated.mean()),
        propensity_min=float(sample.propensity.min()),
        propensity_max=float(sample.propensity.max()),
        overlap_share=float(np.mean(inside)),
    )


def build_truth(
    spec: DatasetSpec, sample: StructuralSample, public_names: Mapping[str, str]
) -> GroundTruth:
    """Assemble the ground truth for one generated dataset.

    `public_names` maps every internal variable name (covariates, treatment
    and outcome) to its public column name.
    """
    internal = (*covariate_names(spec), TREATMENT, OUTCOME)
    if set(public_names) != set(internal):
        raise ValueError(
            f"public_names must cover exactly {sorted(internal)}, "
            f"got {sorted(public_names)}"
        )
    if len(set(public_names.values())) != len(internal):
        raise ValueError("public column names must be unique")

    graph = nx.relabel_nodes(true_graph(spec), dict(public_names))
    treatment, outcome = public_names[TREATMENT], public_names[OUTCOME]
    covariates = [public_names[name] for name in covariate_names(spec)]

    return GroundTruth(
        dataset_id=spec.dataset_id,
        spec=spec,
        true_ate=spec.true_ate,
        treatment=treatment,
        outcome=outcome,
        internal_names={public: name for name, public in public_names.items()},
        column_roles=column_roles(graph, treatment, outcome),
        edges=tuple(sorted(graph.edges)),
        valid_adjustment_sets=tuple(
            valid_adjustment_sets(graph, covariates, treatment, outcome)
        ),
        assumptions=derive_assumptions(spec),
        diagnostics=_diagnostics(spec, sample),
    )


def truth_path(dataset_id: str, root: Path) -> Path:
    return Path(root) / TRUTH_DIRECTORY / f"{validate_dataset_id(dataset_id)}.json"


def load_truth(dataset_id: str, root: Path) -> GroundTruth:
    """Load the ground truth. For evaluation code only."""
    text = truth_path(dataset_id, root).read_text(encoding="utf-8")
    return GroundTruth.model_validate_json(text)
