"""Tool: discover_graph.

Runs the PC algorithm (causal-learn, Fisher-z partial-correlation test) on the
observed columns and reports which columns stay statistically associated
after conditioning on the others.

Timing tiers. The public metadata orders the columns in time: pre-treatment
covariates, then the treatment, then the outcome, then post-outcome columns.
Whether two columns are linked is tested by conditioning only on columns
measured no later than the later of the two. A later column is a consequence,
not a cause, of earlier ones, and conditioning on a consequence can hide or
create associations between its causes. In practice PC is run once per tier
on the columns up to that tier, and each run contributes the edges that
involve its newest columns.

What this tool can and cannot contribute:

- It gives statistical evidence about association. It does not prove that a
  relationship is causal, and it does not prove a causal direction.
- Where two columns were measured at different times, the edge is directed
  from the earlier to the later column and marked `timing`: the direction
  comes from the dataset's timing labels, not from the data.
- Where PC itself orients an edge between columns measured at the same time,
  the edge is marked `data`. Such orientations rest on independence patterns
  and can be wrong.
- Everything else is reported as undirected. An edge is never silently
  oriented.

Its practical uses are screening for covariates associated with both
treatment and outcome (candidate confounders) and checking whether any
covariate is associated with treatment at all.
"""

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from causallearn.search.ConstraintBased.PC import pc
from pydantic import BaseModel, ConfigDict, Field

from causal_agent.bench.public import PublicDataset, Timing, load_public

METHOD = (
    "PC algorithm with Fisher-z partial-correlation tests, conditioning only "
    "on columns measured no later than the pair being tested"
)
DEFAULT_ALPHA = 0.05

CAVEATS = (
    "Edges are statistical evidence of association that conditioning on the "
    "other columns did not remove. They are not proof of a causal relationship.",
    "Directions marked 'timing' come from the dataset's timing labels, not "
    "from the data. Directions marked 'data' come from independence patterns "
    "and can be wrong.",
    "The independence test assumes linear relationships. Nonlinear dependence "
    "can be missed, and binary columns are handled only approximately.",
    "A missing edge is not proof of no effect, and unmeasured variables are "
    "not considered.",
)

_TIMING_ORDER = {
    Timing.PRE_TREATMENT: 0,
    Timing.TREATMENT: 1,
    Timing.OUTCOME: 2,
    Timing.POST_OUTCOME: 3,
}


class DiscoverGraphInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_id: str
    alpha: float = Field(default=DEFAULT_ALPHA, ge=0.001, le=0.2)
    """Significance level of the independence tests."""


class DirectedEdge(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str
    target: str
    basis: Literal["timing", "data"]


class UndirectedEdge(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    first: str
    second: str


class DiscoverGraphOutput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_id: str
    method: str
    alpha: float
    directed_edges: tuple[DirectedEdge, ...]
    undirected_edges: tuple[UndirectedEdge, ...]
    associated_with_treatment: tuple[str, ...]
    associated_with_outcome: tuple[str, ...]
    treatment_outcome_associated: bool
    unconnected_columns: tuple[str, ...]
    caveats: tuple[str, ...]


def _edges_for_tier(
    matrix: np.ndarray, names: list[str], order: dict[str, int], tier: int
) -> tuple[list[DirectedEdge], list[UndirectedEdge]]:
    """Typed edges from one PC run, keeping only pairs whose later column is in `tier`.

    causal-learn encodes i -> j as matrix[j, i] == 1 and matrix[i, j] == -1,
    and an undirected edge as -1 in both cells.
    """
    directed: list[DirectedEdge] = []
    undirected: list[UndirectedEdge] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            first, second = names[i], names[j]
            if max(order[first], order[second]) != tier:
                continue
            if matrix[i, j] == 0 and matrix[j, i] == 0:
                continue
            if order[first] != order[second]:
                source, target = sorted((first, second), key=order.__getitem__)
                directed.append(
                    DirectedEdge(source=source, target=target, basis="timing")
                )
            elif matrix[j, i] == 1 and matrix[i, j] == -1:
                directed.append(DirectedEdge(source=first, target=second, basis="data"))
            elif matrix[i, j] == 1 and matrix[j, i] == -1:
                directed.append(DirectedEdge(source=second, target=first, basis="data"))
            else:
                undirected.append(UndirectedEdge(first=first, second=second))
    return directed, undirected


def _tiered_pc(
    frame: pd.DataFrame, metadata: PublicDataset, alpha: float
) -> tuple[list[DirectedEdge], list[UndirectedEdge]]:
    """Run PC once per timing tier on the columns up to that tier."""
    order = {column.name: _TIMING_ORDER[column.timing] for column in metadata.columns}
    directed: list[DirectedEdge] = []
    undirected: list[UndirectedEdge] = []
    for tier in sorted(set(order.values())):
        names = [name for name in frame.columns if order[name] <= tier]
        if len(names) < 2:
            continue
        result = pc(
            frame[names].to_numpy(dtype=np.float64),
            alpha=alpha,
            indep_test="fisherz",
            show_progress=False,
            node_names=names,
        )
        tier_directed, tier_undirected = _edges_for_tier(
            result.G.graph, names, order, tier
        )
        directed.extend(tier_directed)
        undirected.extend(tier_undirected)
    return directed, undirected


def discover_graph(request: DiscoverGraphInput, root: Path) -> DiscoverGraphOutput:
    """Run tier-aware PC on one dataset's observed columns."""
    frame, metadata = load_public(request.dataset_id, root)
    names = list(frame.columns)
    directed, undirected = _tiered_pc(frame, metadata, request.alpha)

    neighbours: dict[str, set[str]] = {name: set() for name in names}
    for edge in directed:
        neighbours[edge.source].add(edge.target)
        neighbours[edge.target].add(edge.source)
    for edge in undirected:
        neighbours[edge.first].add(edge.second)
        neighbours[edge.second].add(edge.first)

    covariates = [column.name for column in metadata.covariates]
    treatment, outcome = metadata.treatment, metadata.outcome
    return DiscoverGraphOutput(
        dataset_id=request.dataset_id,
        method=METHOD,
        alpha=request.alpha,
        directed_edges=tuple(directed),
        undirected_edges=tuple(undirected),
        associated_with_treatment=tuple(
            name for name in covariates if treatment in neighbours[name]
        ),
        associated_with_outcome=tuple(
            name for name in covariates if outcome in neighbours[name]
        ),
        treatment_outcome_associated=outcome in neighbours[treatment],
        unconnected_columns=tuple(name for name in names if not neighbours[name]),
        caveats=CAVEATS,
    )
