"""Tests for the true graph, column roles and the backdoor check."""

from itertools import combinations

import networkx as nx
import pytest

from causal_agent.bench.graph import (
    Role,
    column_roles,
    covariate_names,
    is_valid_adjustment_set,
    true_graph,
    valid_adjustment_sets,
)
from causal_agent.bench.spec import Scenario
from causal_agent.bench.suite import benchmark_specs

SPECS = benchmark_specs()
SPEC_IDS = [spec.dataset_id for spec in SPECS]


def _all_subsets(names) -> set[tuple[str, ...]]:
    names = sorted(names)
    return {
        subset for size in range(len(names) + 1) for subset in combinations(names, size)
    }


# Hand-derived from the backdoor criterion (ADR 001).
EXPECTED_VALID_SETS = {
    Scenario.RANDOMIZED: _all_subsets(["c1", "c2", "p", "n"]),
    Scenario.CONFOUNDED_LINEAR: {
        ("c1", "c2"),
        ("c1", "c2", "p"),
        ("c1", "c2", "n"),
        ("c1", "c2", "n", "p"),
    },
    Scenario.CONFOUNDED_NONLINEAR: {
        ("c1", "c2"),
        ("c1", "c2", "p"),
        ("c1", "c2", "n"),
        ("c1", "c2", "n", "p"),
    },
    Scenario.POST_OUTCOME_TRAP: {("c1", "c2"), ("c1", "c2", "p")},
}

EXPECTED_ROLES = {
    Scenario.RANDOMIZED: {
        "c1": Role.OUTCOME_PREDICTOR,
        "c2": Role.OUTCOME_PREDICTOR,
        "p": Role.OUTCOME_PREDICTOR,
        "n": Role.IRRELEVANT,
    },
    Scenario.CONFOUNDED_LINEAR: {
        "c1": Role.CONFOUNDER,
        "c2": Role.CONFOUNDER,
        "p": Role.OUTCOME_PREDICTOR,
        "n": Role.IRRELEVANT,
    },
    Scenario.CONFOUNDED_NONLINEAR: {
        "c1": Role.CONFOUNDER,
        "c2": Role.CONFOUNDER,
        "p": Role.OUTCOME_PREDICTOR,
        "n": Role.IRRELEVANT,
    },
    Scenario.POST_OUTCOME_TRAP: {
        "c1": Role.CONFOUNDER,
        "c2": Role.CONFOUNDER,
        "p": Role.OUTCOME_PREDICTOR,
        "k": Role.POST_OUTCOME_COLLIDER,
    },
}


# --- Graph construction ----------------------------------------------------


def test_confounded_graph_edges(make_spec) -> None:
    graph = true_graph(make_spec(Scenario.CONFOUNDED_LINEAR))
    assert set(graph.edges) == {
        ("c1", "t"),
        ("c2", "t"),
        ("c1", "y"),
        ("c2", "y"),
        ("p", "y"),
        ("t", "y"),
    }
    assert set(graph.nodes) == {"t", "y", "c1", "c2", "p", "n"}


def test_randomized_graph_has_no_edges_into_treatment(make_spec) -> None:
    graph = true_graph(make_spec(Scenario.RANDOMIZED))
    assert list(graph.predecessors("t")) == []
    assert ("c1", "y") in graph.edges


def test_trap_graph_replaces_the_irrelevant_column_with_a_collider(
    make_spec,
) -> None:
    spec = make_spec(Scenario.POST_OUTCOME_TRAP)
    graph = true_graph(spec)
    assert covariate_names(spec) == ("c1", "c2", "p", "k")
    assert "n" not in graph.nodes
    assert set(graph.predecessors("k")) == {"t", "y"}


def test_treatment_edge_is_present_even_when_the_effect_is_zero(make_spec) -> None:
    graph = true_graph(make_spec(true_ate=0.0))
    assert ("t", "y") in graph.edges


@pytest.mark.parametrize("spec", SPECS, ids=SPEC_IDS)
def test_true_graph_is_acyclic(spec) -> None:
    assert nx.is_directed_acyclic_graph(true_graph(spec))


# --- Valid adjustment sets and roles for the benchmark ---------------------


@pytest.mark.parametrize("spec", SPECS, ids=SPEC_IDS)
def test_valid_adjustment_sets_match_the_hand_derived_table(spec) -> None:
    found = valid_adjustment_sets(true_graph(spec), covariate_names(spec))
    assert set(found) == EXPECTED_VALID_SETS[spec.scenario]
    assert len(found) == len(set(found))


@pytest.mark.parametrize("spec", SPECS, ids=SPEC_IDS)
def test_column_roles_match_the_hand_derived_table(spec) -> None:
    assert column_roles(true_graph(spec)) == EXPECTED_ROLES[spec.scenario]


def test_empty_set_is_valid_only_when_randomized(make_spec) -> None:
    assert is_valid_adjustment_set(true_graph(make_spec(Scenario.RANDOMIZED)), [])
    assert not is_valid_adjustment_set(
        true_graph(make_spec(Scenario.CONFOUNDED_LINEAR)), []
    )


def test_one_confounder_alone_is_not_enough(make_spec) -> None:
    graph = true_graph(make_spec(Scenario.CONFOUNDED_LINEAR))
    assert not is_valid_adjustment_set(graph, ["c1"])
    assert not is_valid_adjustment_set(graph, ["c2", "p", "n"])


def test_any_set_with_the_trap_column_is_invalid(make_spec) -> None:
    graph = true_graph(make_spec(Scenario.POST_OUTCOME_TRAP))
    assert is_valid_adjustment_set(graph, ["c1", "c2"])
    assert not is_valid_adjustment_set(graph, ["c1", "c2", "k"])
    assert not is_valid_adjustment_set(graph, ["k"])


def test_treatment_and_outcome_are_never_valid_adjustments(make_spec) -> None:
    graph = true_graph(make_spec(Scenario.CONFOUNDED_LINEAR))
    assert not is_valid_adjustment_set(graph, ["c1", "c2", "y"])
    assert not is_valid_adjustment_set(graph, ["c1", "c2", "t"])


def test_unknown_column_raises(make_spec) -> None:
    graph = true_graph(make_spec())
    with pytest.raises(ValueError, match="not in the graph"):
        is_valid_adjustment_set(graph, ["c1", "c2", "nope"])


# --- The checker is a real backdoor check, not a lookup --------------------


def test_m_bias_graph() -> None:
    """Adjusting for a pre-treatment collider opens a path."""
    graph = nx.DiGraph([("u1", "t"), ("u1", "m"), ("u2", "m"), ("u2", "y"), ("t", "y")])
    assert is_valid_adjustment_set(graph, [])
    assert not is_valid_adjustment_set(graph, ["m"])
    assert is_valid_adjustment_set(graph, ["m", "u1"])
    assert is_valid_adjustment_set(graph, ["m", "u2"])


def test_mediator_is_not_a_valid_adjustment() -> None:
    graph = nx.DiGraph([("t", "m"), ("m", "y"), ("t", "y")])
    assert is_valid_adjustment_set(graph, [])
    assert not is_valid_adjustment_set(graph, ["m"])


def test_confounder_reached_through_a_chain() -> None:
    """Blocking any node on the backdoor path is enough."""
    graph = nx.DiGraph([("u", "w"), ("w", "t"), ("u", "y"), ("t", "y")])
    assert not is_valid_adjustment_set(graph, [])
    assert is_valid_adjustment_set(graph, ["w"])
    assert is_valid_adjustment_set(graph, ["u"])


def test_role_outside_the_design_is_reported() -> None:
    instrument_like = nx.DiGraph([("z", "t"), ("t", "y")])
    with pytest.raises(ValueError, match="outside the approved"):
        column_roles(instrument_like)
