"""Tests for the private ground truth."""

import pytest
from pydantic import ValidationError

from causal_agent.bench.equations import simulate
from causal_agent.bench.graph import Role
from causal_agent.bench.spec import Scenario
from causal_agent.bench.truth import (
    GroundTruth,
    build_truth,
    derive_assumptions,
    load_truth,
    truth_path,
)

IDENTITY = {name: name for name in ("c1", "c2", "p", "n", "t", "y")}
SHUFFLED = {"c1": "x3", "c2": "x1", "p": "x4", "n": "x2", "t": "t", "y": "y"}
SHUFFLED_TRAP = {"c1": "x3", "c2": "x1", "p": "x4", "k": "x2", "t": "t", "y": "y"}


def _truth(make_spec, scenario=Scenario.CONFOUNDED_LINEAR, names=None, **overrides):
    spec = make_spec(scenario, **overrides)
    return build_truth(spec, simulate(spec), names or SHUFFLED)


# --- Assumption flags ------------------------------------------------------


@pytest.mark.parametrize(
    ("scenario", "overrides", "randomized", "outcome_linear", "propensity_linear"),
    [
        (Scenario.RANDOMIZED, {}, True, True, True),
        (Scenario.RANDOMIZED, {"outcome_nonlinearity": 1.0}, True, False, True),
        (Scenario.CONFOUNDED_LINEAR, {}, False, True, True),
        (
            Scenario.CONFOUNDED_NONLINEAR,
            {"treatment_nonlinearity": 0.0, "outcome_nonlinearity": 1.0},
            False,
            False,
            True,
        ),
        (
            Scenario.CONFOUNDED_NONLINEAR,
            {"treatment_nonlinearity": 1.0, "outcome_nonlinearity": 0.0},
            False,
            True,
            False,
        ),
        (Scenario.CONFOUNDED_NONLINEAR, {}, False, False, False),
        (Scenario.POST_OUTCOME_TRAP, {}, False, True, True),
    ],
)
def test_assumptions_follow_from_the_knobs(
    make_spec, scenario, overrides, randomized, outcome_linear, propensity_linear
) -> None:
    assumptions = derive_assumptions(make_spec(scenario, **overrides))
    assert assumptions.treatment_randomized is randomized
    assert assumptions.outcome_model_linear is outcome_linear
    assert assumptions.propensity_logit_linear is propensity_linear


# --- Building the truth ----------------------------------------------------


def test_truth_carries_the_true_ate_and_the_spec(make_spec) -> None:
    truth = _truth(make_spec, true_ate=-1.0)
    assert truth.true_ate == -1.0
    assert truth.spec.true_ate == -1.0
    assert truth.dataset_id == truth.spec.dataset_id


def test_truth_uses_public_column_names(make_spec) -> None:
    truth = _truth(make_spec)
    assert truth.internal_names == {
        "x3": "c1",
        "x1": "c2",
        "x4": "p",
        "x2": "n",
        "t": "t",
        "y": "y",
    }
    assert truth.column_roles == {
        "x3": Role.CONFOUNDER,
        "x1": Role.CONFOUNDER,
        "x4": Role.OUTCOME_PREDICTOR,
        "x2": Role.IRRELEVANT,
    }
    assert set(truth.edges) == {
        ("x3", "t"),
        ("x1", "t"),
        ("x3", "y"),
        ("x1", "y"),
        ("x4", "y"),
        ("t", "y"),
    }
    assert set(truth.valid_adjustment_sets) == {
        ("x1", "x3"),
        ("x1", "x3", "x4"),
        ("x1", "x2", "x3"),
        ("x1", "x2", "x3", "x4"),
    }


def test_trap_truth_forbids_the_collider(make_spec) -> None:
    truth = _truth(make_spec, Scenario.POST_OUTCOME_TRAP, SHUFFLED_TRAP)
    assert truth.column_roles["x2"] is Role.POST_OUTCOME_COLLIDER
    assert truth.is_valid_adjustment_set(["x1", "x3"])
    assert not truth.is_valid_adjustment_set(["x1", "x2", "x3"])


def test_adjustment_check_ignores_order_and_duplicates(make_spec) -> None:
    truth = _truth(make_spec)
    assert truth.is_valid_adjustment_set(["x3", "x1"])
    assert truth.is_valid_adjustment_set(["x3", "x1", "x3"])
    assert truth.is_valid_adjustment_set({"x4", "x1", "x3"})


def test_adjustment_check_rejects_incomplete_or_unknown_sets(make_spec) -> None:
    truth = _truth(make_spec)
    assert not truth.is_valid_adjustment_set([])
    assert not truth.is_valid_adjustment_set(["x3"])
    assert not truth.is_valid_adjustment_set(["x1", "x3", "y"])
    assert not truth.is_valid_adjustment_set(["x1", "x3", "not_a_column"])


def test_empty_set_is_valid_when_randomized(make_spec) -> None:
    truth = _truth(make_spec, Scenario.RANDOMIZED)
    assert truth.is_valid_adjustment_set([])
    assert len(truth.valid_adjustment_sets) == 16


def test_graph_round_trips_through_the_edge_list(make_spec) -> None:
    truth = _truth(make_spec)
    graph = truth.graph()
    assert set(graph.edges) == set(truth.edges)
    assert set(graph.nodes) == {"x1", "x2", "x3", "x4", "t", "y"}


def test_build_truth_requires_a_complete_unique_naming(make_spec) -> None:
    spec = make_spec()
    sample = simulate(spec)
    with pytest.raises(ValueError, match="must cover exactly"):
        build_truth(spec, sample, {"c1": "x1"})
    with pytest.raises(ValueError, match="unique"):
        build_truth(spec, sample, {**IDENTITY, "c2": "c1"})


# --- Diagnostics -----------------------------------------------------------


def test_diagnostics_describe_the_sample(make_spec) -> None:
    spec = make_spec(true_ate=2.0)
    sample = simulate(spec)
    diagnostics = build_truth(spec, sample, IDENTITY).diagnostics
    naive = sample.y[sample.t == 1].mean() - sample.y[sample.t == 0].mean()
    assert diagnostics.naive_difference == pytest.approx(naive)
    assert diagnostics.naive_bias == pytest.approx(naive - 2.0)
    assert diagnostics.treated_share == pytest.approx(sample.t.mean())
    assert diagnostics.outcome_sd == pytest.approx(sample.y.std(ddof=1))
    assert 0.0 <= diagnostics.overlap_share <= 1.0


def test_randomized_diagnostics_show_a_constant_propensity(make_spec) -> None:
    diagnostics = _truth(make_spec, Scenario.RANDOMIZED).diagnostics
    assert diagnostics.propensity_min == diagnostics.propensity_max == 0.5
    assert diagnostics.overlap_share == 1.0


# --- Serialization ---------------------------------------------------------


@pytest.mark.parametrize(
    ("scenario", "names"),
    [
        (Scenario.RANDOMIZED, SHUFFLED),
        (Scenario.CONFOUNDED_NONLINEAR, SHUFFLED),
        (Scenario.POST_OUTCOME_TRAP, SHUFFLED_TRAP),
    ],
)
def test_truth_round_trips_through_json(make_spec, scenario, names) -> None:
    truth = _truth(make_spec, scenario, names)
    assert GroundTruth.model_validate_json(truth.model_dump_json()) == truth


def test_load_truth_reads_from_the_truth_directory(make_spec, tmp_path) -> None:
    truth = _truth(make_spec, dataset_id="ds_042")
    path = truth_path("ds_042", tmp_path)
    assert path == tmp_path / "truth" / "ds_042.json"
    path.parent.mkdir(parents=True)
    path.write_text(truth.model_dump_json(), encoding="utf-8")
    assert load_truth("ds_042", tmp_path) == truth


def test_load_truth_rejects_path_like_ids(tmp_path) -> None:
    with pytest.raises(ValueError, match="invalid dataset id"):
        load_truth("../public/ds_001", tmp_path)


def test_truth_is_frozen(make_spec) -> None:
    with pytest.raises(ValidationError):
        _truth(make_spec).true_ate = 0.0
