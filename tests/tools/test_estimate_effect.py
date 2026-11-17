"""Tests for the estimate_effect tool."""

import pytest
from pydantic import ValidationError

from causal_agent.agent.fixed_pipeline import run_fixed_pipeline
from causal_agent.bench.graph import Role
from causal_agent.bench.spec import Scenario
from causal_agent.tools.estimate_effect import (
    EstimateEffectInput,
    EstimateEffectOutput,
    estimate_effect,
)
from causal_agent.tools.estimators import ASSUMPTIONS, EstimatorName, estimate_ate


def _run(small_benchmark, scenario, estimator, adjustment_set=()):
    request = EstimateEffectInput(
        dataset_id=small_benchmark.dataset_id(scenario),
        estimator=estimator,
        adjustment_set=adjustment_set,
    )
    return estimate_effect(request, small_benchmark.public_only_root)


def _confounders(small_benchmark, scenario) -> tuple[str, ...]:
    roles = small_benchmark.datasets[scenario].truth.column_roles
    return tuple(sorted(n for n, role in roles.items() if role is Role.CONFOUNDER))


# --- Input schema ----------------------------------------------------------


def test_input_accepts_estimator_names_as_strings() -> None:
    request = EstimateEffectInput.model_validate(
        {
            "dataset_id": "ds_902",
            "estimator": "linear_regression",
            "adjustment_set": ["x1"],
        }
    )
    assert request.estimator is EstimatorName.LINEAR_REGRESSION
    assert request.adjustment_set == ("x1",)


def test_adjustment_set_defaults_to_empty() -> None:
    request = EstimateEffectInput(dataset_id="ds_902", estimator="difference_in_means")
    assert request.adjustment_set == ()


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        (
            {"estimator": "difference_in_means", "adjustment_set": ["x1"]},
            "always uses the empty adjustment set",
        ),
        (
            {"estimator": "linear_regression", "adjustment_set": ["x1", "x1"]},
            "more than once",
        ),
        ({"estimator": "matching"}, "estimator"),
        ({"estimator": "linear_regression", "code": "print(1)"}, "code"),
        ({"adjustment_set": ["x1"]}, "estimator"),
    ],
)
def test_invalid_inputs_are_rejected(fields, message) -> None:
    with pytest.raises(ValidationError, match=message):
        EstimateEffectInput(dataset_id="ds_902", **fields)


# --- Behaviour -------------------------------------------------------------


@pytest.mark.parametrize("estimator", list(EstimatorName))
def test_output_echoes_the_method_and_states_its_assumptions(
    small_benchmark, estimator
) -> None:
    scenario = Scenario.CONFOUNDED_LINEAR
    adjustment = (
        ()
        if estimator is EstimatorName.DIFFERENCE_IN_MEANS
        else _confounders(small_benchmark, scenario)
    )
    output = _run(small_benchmark, scenario, estimator, adjustment)
    assert output.estimator is estimator
    assert output.adjustment_set == adjustment
    assert output.assumptions == ASSUMPTIONS[estimator]
    assert output.ci_low < output.estimate < output.ci_high
    assert output.n_rows == 2000


@pytest.mark.parametrize("estimator", list(EstimatorName))
def test_tool_matches_a_direct_estimator_call(small_benchmark, estimator) -> None:
    scenario = Scenario.CONFOUNDED_NONLINEAR
    adjustment = (
        ()
        if estimator is EstimatorName.DIFFERENCE_IN_MEANS
        else _confounders(small_benchmark, scenario)
    )
    output = _run(small_benchmark, scenario, estimator, adjustment)
    frame = small_benchmark.datasets[scenario].frame
    direct = estimate_ate(estimator, frame, "t", "y", adjustment)
    assert output.estimate == direct.estimate
    assert output.standard_error == direct.standard_error


def test_tool_reproduces_the_fixed_pipeline(small_benchmark) -> None:
    """Same estimator and adjustment set give the same number by either route."""
    scenario = Scenario.POST_OUTCOME_TRAP
    pipeline = run_fixed_pipeline(
        small_benchmark.dataset_id(scenario), small_benchmark.public_only_root
    )
    output = _run(
        small_benchmark, scenario, EstimatorName.DOUBLY_ROBUST, pipeline.adjustment_set
    )
    assert output.estimate == pipeline.estimate


def test_tool_does_not_block_the_post_outcome_column(small_benchmark) -> None:
    """Choosing the adjustment set is the caller's job, mistakes included."""
    scenario = Scenario.POST_OUTCOME_TRAP
    roles = small_benchmark.datasets[scenario].truth.column_roles
    collider = next(n for n, r in roles.items() if r is Role.POST_OUTCOME_COLLIDER)
    confounders = _confounders(small_benchmark, scenario)
    good = _run(small_benchmark, scenario, EstimatorName.LINEAR_REGRESSION, confounders)
    bad = _run(
        small_benchmark,
        scenario,
        EstimatorName.LINEAR_REGRESSION,
        (*confounders, collider),
    )
    assert bad.adjustment_set == (*confounders, collider)
    assert bad.estimate != good.estimate


@pytest.mark.parametrize("bad_set", [("t",), ("y",), ("x1", "not_a_column")])
def test_adjustment_set_must_be_covariate_columns(small_benchmark, bad_set) -> None:
    with pytest.raises(ValueError, match="may only contain covariate columns"):
        _run(
            small_benchmark,
            Scenario.CONFOUNDED_LINEAR,
            EstimatorName.LINEAR_REGRESSION,
            bad_set,
        )


def test_path_like_dataset_id_is_rejected(small_benchmark) -> None:
    request = EstimateEffectInput(
        dataset_id="../truth/ds_902", estimator="difference_in_means"
    )
    with pytest.raises(ValueError, match="invalid dataset id"):
        estimate_effect(request, small_benchmark.root)


def test_output_is_a_closed_schema_without_ground_truth(small_benchmark) -> None:
    output = _run(
        small_benchmark, Scenario.RANDOMIZED, EstimatorName.DIFFERENCE_IN_MEANS
    )
    assert EstimateEffectOutput.model_validate_json(output.model_dump_json()) == output
    assert set(EstimateEffectOutput.model_fields) == {
        "dataset_id",
        "estimator",
        "adjustment_set",
        "estimate",
        "standard_error",
        "ci_low",
        "ci_high",
        "n_rows",
        "n_treated",
        "diagnostics",
        "warnings",
        "assumptions",
    }
