"""Tests for DatasetSpec: defaults, immutability and scenario consistency."""

import pytest
from pydantic import ValidationError

from causal_agent.bench.spec import (
    DEFAULT_N_ROWS,
    DEFAULT_OUTCOME_PREDICTOR_STRENGTH,
    DatasetSpec,
    Scenario,
)


@pytest.mark.parametrize("scenario", list(Scenario))
def test_each_scenario_accepts_its_matching_knobs(make_spec, scenario) -> None:
    assert make_spec(scenario).scenario is scenario


def test_defaults() -> None:
    spec = DatasetSpec(
        dataset_id="ds_test",
        scenario=Scenario.RANDOMIZED,
        seed=0,
        true_ate=1.0,
        assignment_strength=0.0,
        confounder_outcome_strength=1.0,
        outcome_noise=1.0,
    )
    assert spec.n_rows == DEFAULT_N_ROWS == 5000
    assert spec.outcome_predictor_strength == DEFAULT_OUTCOME_PREDICTOR_STRENGTH
    assert spec.treatment_nonlinearity == 0.0
    assert spec.outcome_nonlinearity == 0.0
    assert spec.trap_strength == 0.0


def test_spec_is_frozen(make_spec) -> None:
    spec = make_spec()
    with pytest.raises(ValidationError):
        spec.true_ate = 99.0


def test_unknown_fields_are_rejected(make_spec) -> None:
    with pytest.raises(ValidationError):
        make_spec(not_a_knob=1.0)


def test_randomized_may_have_a_nonlinear_outcome(make_spec) -> None:
    spec = make_spec(Scenario.RANDOMIZED, outcome_nonlinearity=1.0)
    assert spec.outcome_nonlinearity == 1.0


@pytest.mark.parametrize(
    ("nonlinear_knobs"),
    [
        {"treatment_nonlinearity": 0.0, "outcome_nonlinearity": 1.0},
        {"treatment_nonlinearity": 1.0, "outcome_nonlinearity": 0.0},
        {"treatment_nonlinearity": 1.0, "outcome_nonlinearity": 1.0},
    ],
)
def test_nonlinear_scenario_accepts_either_or_both_sides(
    make_spec, nonlinear_knobs
) -> None:
    make_spec(Scenario.CONFOUNDED_NONLINEAR, **nonlinear_knobs)


@pytest.mark.parametrize(
    ("scenario", "overrides"),
    [
        (Scenario.RANDOMIZED, {"assignment_strength": 1.0}),
        (Scenario.RANDOMIZED, {"treatment_nonlinearity": 1.0}),
        (Scenario.RANDOMIZED, {"trap_strength": 0.5}),
        (Scenario.CONFOUNDED_LINEAR, {"assignment_strength": 0.0}),
        (Scenario.CONFOUNDED_LINEAR, {"confounder_outcome_strength": 0.0}),
        (Scenario.CONFOUNDED_LINEAR, {"outcome_nonlinearity": 1.0}),
        (Scenario.CONFOUNDED_LINEAR, {"treatment_nonlinearity": 1.0}),
        (Scenario.CONFOUNDED_LINEAR, {"trap_strength": 0.5}),
        (
            Scenario.CONFOUNDED_NONLINEAR,
            {"treatment_nonlinearity": 0.0, "outcome_nonlinearity": 0.0},
        ),
        (Scenario.CONFOUNDED_NONLINEAR, {"assignment_strength": 0.0}),
        (Scenario.CONFOUNDED_NONLINEAR, {"trap_strength": 0.5}),
        (Scenario.POST_OUTCOME_TRAP, {"trap_strength": 0.0}),
        (Scenario.POST_OUTCOME_TRAP, {"assignment_strength": 0.0}),
        (Scenario.POST_OUTCOME_TRAP, {"outcome_nonlinearity": 1.0}),
    ],
)
def test_scenario_label_must_match_knobs(make_spec, scenario, overrides) -> None:
    with pytest.raises(ValidationError, match="requires"):
        make_spec(scenario, **overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"outcome_noise": 0.0},
        {"outcome_noise": -1.0},
        {"assignment_strength": -0.5},
        {"confounder_outcome_strength": -0.5},
        {"outcome_predictor_strength": -1.0},
        {"n_rows": 5},
        {"seed": -1},
        {"true_ate": float("nan")},
        {"true_ate": float("inf")},
        {"dataset_id": ""},
    ],
)
def test_out_of_range_values_are_rejected(make_spec, overrides) -> None:
    with pytest.raises(ValidationError):
        make_spec(**overrides)
