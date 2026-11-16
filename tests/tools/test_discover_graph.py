"""Tests for the discover_graph tool."""

import numpy as np
import pytest
from pydantic import ValidationError

from causal_agent.bench.graph import Role
from causal_agent.bench.spec import Scenario
from causal_agent.tools.discover_graph import (
    CAVEATS,
    DirectedEdge,
    DiscoverGraphInput,
    DiscoverGraphOutput,
    UndirectedEdge,
    _edges_for_tier,
    discover_graph,
)

ALL_SCENARIOS = list(Scenario)
CONFOUNDED = [
    Scenario.CONFOUNDED_LINEAR,
    Scenario.CONFOUNDED_NONLINEAR,
    Scenario.POST_OUTCOME_TRAP,
]
TIMING_ORDER = {"pre_treatment": 0, "treatment": 1, "outcome": 2, "post_outcome": 3}


@pytest.fixture(scope="module")
def outputs(small_benchmark) -> dict[Scenario, DiscoverGraphOutput]:
    """Run against a root that has no truth directory."""
    return {
        scenario: discover_graph(
            DiscoverGraphInput(dataset_id=small_benchmark.dataset_id(scenario)),
            small_benchmark.public_only_root,
        )
        for scenario in ALL_SCENARIOS
    }


def _names_with_role(small_benchmark, scenario, role) -> set[str]:
    roles = small_benchmark.datasets[scenario].truth.column_roles
    return {name for name, found in roles.items() if found is role}


# --- Limitations are part of the output ------------------------------------


@pytest.mark.parametrize("scenario", ALL_SCENARIOS)
def test_output_carries_the_caveats(outputs, scenario) -> None:
    output = outputs[scenario]
    assert output.caveats == CAVEATS
    text = " ".join(output.caveats)
    assert "not proof" in text
    assert "timing labels, not from the data" in text
    assert "linear" in text
    assert "unmeasured" in text
    assert "Fisher-z" in output.method


@pytest.mark.parametrize("scenario", ALL_SCENARIOS)
def test_no_edge_points_backwards_in_time(small_benchmark, outputs, scenario) -> None:
    metadata = small_benchmark.datasets[scenario].public
    order = {c.name: TIMING_ORDER[c.timing.value] for c in metadata.columns}
    for edge in outputs[scenario].directed_edges:
        assert order[edge.source] <= order[edge.target]
        if order[edge.source] < order[edge.target]:
            assert edge.basis == "timing"
        else:
            assert edge.basis == "data"


# --- What the tool finds on data with strong, known structure --------------


@pytest.mark.parametrize("scenario", ALL_SCENARIOS)
def test_treatment_to_outcome_edge_is_found_and_directed_by_timing(
    outputs, scenario
) -> None:
    """These datasets have a true effect of 1."""
    output = outputs[scenario]
    assert output.treatment_outcome_associated
    assert DirectedEdge(source="t", target="y", basis="timing") in output.directed_edges


@pytest.mark.parametrize("scenario", CONFOUNDED)
def test_true_confounders_are_associated_with_treatment_and_outcome(
    small_benchmark, outputs, scenario
) -> None:
    confounders = _names_with_role(small_benchmark, scenario, Role.CONFOUNDER)
    assert len(confounders) == 2
    assert confounders <= set(outputs[scenario].associated_with_treatment)
    assert confounders <= set(outputs[scenario].associated_with_outcome)


def test_post_outcome_column_does_not_hide_the_confounders(
    small_benchmark, outputs
) -> None:
    """Tests between earlier columns never condition on a later column."""
    scenario = Scenario.POST_OUTCOME_TRAP
    output = outputs[scenario]
    (collider,) = _names_with_role(
        small_benchmark, scenario, Role.POST_OUTCOME_COLLIDER
    )
    assert DirectedEdge(source="t", target=collider, basis="timing") in (
        output.directed_edges
    )
    assert DirectedEdge(source="y", target=collider, basis="timing") in (
        output.directed_edges
    )
    assert all(edge.source != collider for edge in output.directed_edges)


def test_outcome_only_predictor_is_linked_to_the_outcome_only(
    small_benchmark, outputs
) -> None:
    scenario = Scenario.CONFOUNDED_LINEAR
    (predictor,) = _names_with_role(small_benchmark, scenario, Role.OUTCOME_PREDICTOR)
    assert predictor in outputs[scenario].associated_with_outcome
    assert predictor not in outputs[scenario].associated_with_treatment


def test_irrelevant_column_is_unconnected(small_benchmark, outputs) -> None:
    scenario = Scenario.CONFOUNDED_LINEAR
    (irrelevant,) = _names_with_role(small_benchmark, scenario, Role.IRRELEVANT)
    assert irrelevant in outputs[scenario].unconnected_columns


# --- Edge encoding ---------------------------------------------------------


def _matrix(size: int, *cells: tuple[int, int, int]) -> np.ndarray:
    matrix = np.zeros((size, size), dtype=int)
    for row, column, value in cells:
        matrix[row, column] = value
    return matrix


def test_same_tier_edges_keep_what_pc_reported() -> None:
    names = ["a", "b", "c"]
    order = {"a": 0, "b": 0, "c": 0}
    # a -> b is matrix[b, a] == 1 and matrix[a, b] == -1; b - c is -1 both ways.
    matrix = _matrix(3, (1, 0, 1), (0, 1, -1), (1, 2, -1), (2, 1, -1))
    directed, undirected = _edges_for_tier(matrix, names, order, tier=0)
    assert directed == [DirectedEdge(source="a", target="b", basis="data")]
    assert undirected == [UndirectedEdge(first="b", second="c")]


def test_cross_tier_edges_are_directed_by_timing_whatever_pc_reported() -> None:
    names = ["late", "early"]
    order = {"late": 1, "early": 0}
    # PC claims late -> early; timing says the opposite.
    matrix = _matrix(2, (1, 0, 1), (0, 1, -1))
    directed, undirected = _edges_for_tier(matrix, names, order, tier=1)
    assert directed == [DirectedEdge(source="early", target="late", basis="timing")]
    assert undirected == []


def test_a_run_contributes_only_edges_whose_later_column_is_in_its_tier() -> None:
    names = ["a", "b", "c"]
    order = {"a": 0, "b": 0, "c": 1}
    matrix = _matrix(3, (0, 1, -1), (1, 0, -1), (0, 2, -1), (2, 0, -1))
    directed, undirected = _edges_for_tier(matrix, names, order, tier=1)
    assert directed == [DirectedEdge(source="a", target="c", basis="timing")]
    assert undirected == []


def test_bidirected_edges_are_reported_as_undirected() -> None:
    matrix = _matrix(2, (0, 1, 1), (1, 0, 1))
    directed, undirected = _edges_for_tier(matrix, ["a", "b"], {"a": 0, "b": 0}, 0)
    assert directed == []
    assert undirected == [UndirectedEdge(first="a", second="b")]


# --- Schema and behaviour --------------------------------------------------


def test_discovery_is_deterministic(small_benchmark, outputs) -> None:
    scenario = Scenario.CONFOUNDED_NONLINEAR
    again = discover_graph(
        DiscoverGraphInput(dataset_id=small_benchmark.dataset_id(scenario)),
        small_benchmark.root,
    )
    assert again == outputs[scenario]


def test_output_round_trips_through_json(outputs) -> None:
    output = outputs[Scenario.POST_OUTCOME_TRAP]
    assert DiscoverGraphOutput.model_validate_json(output.model_dump_json()) == output


@pytest.mark.parametrize(
    "fields",
    [
        {"dataset_id": "ds_902", "alpha": 0.0},
        {"dataset_id": "ds_902", "alpha": 0.5},
        {"dataset_id": "ds_902", "unknown": 1},
        {"alpha": 0.05},
    ],
)
def test_invalid_inputs_are_rejected(fields) -> None:
    with pytest.raises(ValidationError):
        DiscoverGraphInput(**fields)


def test_path_like_dataset_id_is_rejected(small_benchmark) -> None:
    with pytest.raises(ValueError, match="invalid dataset id"):
        discover_graph(
            DiscoverGraphInput(dataset_id="../truth/ds_902"), small_benchmark.root
        )
