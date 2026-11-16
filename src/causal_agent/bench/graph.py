"""The true causal graph of a dataset, column roles, and the backdoor check.

This module is private: the true graph must never reach the agent or a
baseline. Variable names here are the generator's internal names; the public
dataset uses neutral names.
"""

from collections.abc import Iterable
from enum import StrEnum
from itertools import combinations

import networkx as nx

from causal_agent.bench.spec import DatasetSpec

TREATMENT = "t"
OUTCOME = "y"
CONFOUNDERS = ("c1", "c2")
OUTCOME_PREDICTOR = "p"
IRRELEVANT = "n"
TRAP = "k"


class Role(StrEnum):
    """Causal role of a covariate, derived from the true graph."""

    CONFOUNDER = "confounder"
    OUTCOME_PREDICTOR = "outcome_predictor"
    IRRELEVANT = "irrelevant"
    POST_OUTCOME_COLLIDER = "post_outcome_collider"


def covariate_names(spec: DatasetSpec) -> tuple[str, ...]:
    """The four covariates of a dataset. The trap column replaces `n`."""
    last = TRAP if spec.trap_strength > 0 else IRRELEVANT
    return (*CONFOUNDERS, OUTCOME_PREDICTOR, last)


def true_graph(spec: DatasetSpec) -> nx.DiGraph:
    """Build the causal graph implied by the structural equations.

    An edge exists exactly when its coefficient is non-zero. The edge from
    treatment to outcome is always present, even when the true ATE is zero:
    it is the effect under study and does not affect which adjustment sets
    are valid.
    """
    graph = nx.DiGraph()
    graph.add_nodes_from([TREATMENT, OUTCOME, *covariate_names(spec)])
    graph.add_edge(TREATMENT, OUTCOME)
    for confounder in CONFOUNDERS:
        if spec.assignment_strength > 0:
            graph.add_edge(confounder, TREATMENT)
        if spec.confounder_outcome_strength > 0:
            graph.add_edge(confounder, OUTCOME)
    if spec.outcome_predictor_strength > 0:
        graph.add_edge(OUTCOME_PREDICTOR, OUTCOME)
    if spec.trap_strength > 0:
        graph.add_edge(TREATMENT, TRAP)
        graph.add_edge(OUTCOME, TRAP)
    return graph


def is_valid_adjustment_set(
    graph: nx.DiGraph,
    adjustment: Iterable[str],
    treatment: str = TREATMENT,
    outcome: str = OUTCOME,
) -> bool:
    """Backdoor criterion for the effect of `treatment` on `outcome`.

    A set is valid when (1) it contains no descendant of the treatment and
    (2) it blocks every path between treatment and outcome that starts with
    an arrow into the treatment. Condition (2) is checked as d-separation in
    the graph with the treatment's outgoing edges removed.
    """
    adjustment = set(adjustment)
    unknown = adjustment - set(graph.nodes)
    if unknown:
        raise ValueError(f"not in the graph: {sorted(unknown)}")
    if treatment in adjustment or outcome in adjustment:
        return False
    if adjustment & nx.descendants(graph, treatment):
        return False

    backdoor_graph = graph.copy()
    backdoor_graph.remove_edges_from(list(graph.out_edges(treatment)))
    return nx.is_d_separator(backdoor_graph, {treatment}, {outcome}, adjustment)


def valid_adjustment_sets(
    graph: nx.DiGraph,
    candidates: Iterable[str],
    treatment: str = TREATMENT,
    outcome: str = OUTCOME,
) -> list[tuple[str, ...]]:
    """Every subset of `candidates` that passes the backdoor criterion."""
    candidates = sorted(candidates)
    return [
        subset
        for size in range(len(candidates) + 1)
        for subset in combinations(candidates, size)
        if is_valid_adjustment_set(graph, subset, treatment, outcome)
    ]


def column_roles(
    graph: nx.DiGraph, treatment: str = TREATMENT, outcome: str = OUTCOME
) -> dict[str, Role]:
    """Derive each covariate's causal role from the graph."""
    after_treatment = nx.descendants(graph, treatment)
    after_outcome = nx.descendants(graph, outcome)
    without_treatment = graph.copy()
    without_treatment.remove_node(treatment)

    roles: dict[str, Role] = {}
    for node in graph.nodes:
        if node in (treatment, outcome):
            continue
        causes_treatment = nx.has_path(graph, node, treatment)
        causes_outcome = nx.has_path(without_treatment, node, outcome)
        if node in after_treatment and node in after_outcome:
            roles[node] = Role.POST_OUTCOME_COLLIDER
        elif node in after_treatment or (causes_treatment and not causes_outcome):
            raise ValueError(
                f"'{node}' has a role outside the approved benchmark design"
            )
        elif causes_treatment and causes_outcome:
            roles[node] = Role.CONFOUNDER
        elif causes_outcome:
            roles[node] = Role.OUTCOME_PREDICTOR
        else:
            roles[node] = Role.IRRELEVANT
    return roles
