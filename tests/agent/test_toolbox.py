"""Tests for the model-facing tool definitions and tool execution."""

import json

import pytest

from causal_agent.agent.toolbox import (
    TOOLS,
    execute_tool,
    input_schema,
    tool_definitions,
)
from causal_agent.bench.spec import Scenario
from causal_agent.tools.estimate_effect import EstimateEffectInput
from causal_agent.tools.estimators import EstimatorName
from causal_agent.tools.refute_estimate import RefuterName

PRIVATE_WORDS = ("true_ate", "ground truth", "seed", "scenario", "truth/")


def _definition(name: str) -> dict:
    return next(tool for tool in tool_definitions() if tool["name"] == name)


def _run(small_benchmark, name, tool_input, scenario=Scenario.CONFOUNDED_LINEAR):
    return execute_tool(
        name,
        tool_input,
        small_benchmark.dataset_id(scenario),
        small_benchmark.public_only_root,
    )


# --- Definitions -----------------------------------------------------------


def test_exactly_three_tools_are_exposed() -> None:
    assert [tool["name"] for tool in tool_definitions()] == [
        "discover_graph",
        "estimate_effect",
        "refute_estimate",
    ]
    assert len(TOOLS) == 3


def test_disagreement_check_is_part_of_refute_estimate_not_a_new_tool(
    small_benchmark,
) -> None:
    definition = _definition("refute_estimate")
    assert "estimator_disagreement" in json.dumps(definition["input_schema"])
    assert "not proof of bias" in definition["description"]

    covariates = [
        column.name
        for column in small_benchmark.datasets[
            Scenario.CONFOUNDED_LINEAR
        ].public.covariates
    ]
    outcome = _run(
        small_benchmark,
        "refute_estimate",
        {
            "estimator": "linear_regression",
            "adjustment_set": covariates,
            "refuter": "estimator_disagreement",
        },
    )
    assert not outcome.is_error
    assert outcome.output["verdict"] in ("disagreement", "agreement", "inconclusive")
    assert outcome.output["comparison_estimator"] == "doubly_robust"
    assert "passed" not in outcome.output


def test_no_tool_accepts_code_or_a_dataset_id() -> None:
    for tool in tool_definitions():
        properties = tool["input_schema"]["properties"]
        assert "dataset_id" not in properties
        assert "dataset_id" not in tool["input_schema"]["required"]
        assert not {"code", "script", "command", "query", "path"} & set(properties)
        assert tool["input_schema"]["additionalProperties"] is False


def test_schemas_are_self_contained_json() -> None:
    text = json.dumps(tool_definitions())
    assert "$ref" not in text
    assert "$defs" not in text


def test_estimate_effect_schema_lists_the_estimators() -> None:
    schema = _definition("estimate_effect")["input_schema"]
    assert schema["properties"]["estimator"]["enum"] == [e.value for e in EstimatorName]
    assert schema["required"] == ["estimator"]
    assert schema["properties"]["adjustment_set"]["type"] == "array"


def test_refute_estimate_schema_lists_the_refuters() -> None:
    schema = _definition("refute_estimate")["input_schema"]
    assert schema["properties"]["refuter"]["enum"] == [r.value for r in RefuterName]
    assert set(schema["required"]) == {"estimator", "refuter"}


def test_schema_fields_match_the_pydantic_model() -> None:
    schema = input_schema(EstimateEffectInput)
    expected = set(EstimateEffectInput.model_fields) - {"dataset_id"}
    assert set(schema["properties"]) == expected


def test_descriptions_state_limits_and_reveal_nothing_private() -> None:
    descriptions = {tool["name"]: tool["description"] for tool in tool_definitions()}
    assert "not proof" in descriptions["discover_graph"]
    assert (
        "does not check whether the adjustment set" in descriptions["estimate_effect"]
    )
    assert (
        "not evidence that the estimate is unbiased" in descriptions["refute_estimate"]
    )
    text = json.dumps(tool_definitions()).lower()
    assert [word for word in PRIVATE_WORDS if word in text] == []


# --- Execution -------------------------------------------------------------


def test_each_tool_runs_and_returns_json(small_benchmark) -> None:
    calls = {
        "discover_graph": {},
        "estimate_effect": {"estimator": "linear_regression", "adjustment_set": ["x1"]},
        "refute_estimate": {
            "estimator": "linear_regression",
            "adjustment_set": ["x1"],
            "refuter": "data_subset",
        },
    }
    for name, tool_input in calls.items():
        outcome = _run(small_benchmark, name, tool_input)
        assert not outcome.is_error, outcome.content
        assert json.loads(outcome.content) == outcome.output
        assert outcome.output["dataset_id"] == "ds_902"


def test_dataset_is_fixed_by_the_loop_not_the_model(small_benchmark) -> None:
    outcome = _run(
        small_benchmark,
        "estimate_effect",
        {"dataset_id": "ds_901", "estimator": "difference_in_means"},
    )
    assert outcome.is_error
    assert "dataset_id is fixed" in outcome.content


@pytest.mark.parametrize(
    ("name", "tool_input", "message"),
    [
        ("run_python", {"code": "print(1)"}, "Unknown tool"),
        ("estimate_effect", {"estimator": "magic"}, "Invalid input"),
        ("estimate_effect", {}, "Invalid input"),
        ("estimate_effect", {"estimator": "linear_regression", "code": "x"}, "code"),
        (
            "estimate_effect",
            {"estimator": "difference_in_means", "adjustment_set": ["x1"]},
            "always uses the empty adjustment set",
        ),
        (
            "estimate_effect",
            {"estimator": "linear_regression", "adjustment_set": ["nope"]},
            "may only contain covariate columns",
        ),
        ("estimate_effect", "not an object", "Expected a JSON object"),
        ("discover_graph", {"alpha": 5}, "Invalid input"),
        ("refute_estimate", {"estimator": "linear_regression"}, "refuter"),
    ],
)
def test_bad_calls_return_readable_errors_instead_of_raising(
    small_benchmark, name, tool_input, message
) -> None:
    outcome = _run(small_benchmark, name, tool_input)
    assert outcome.is_error
    assert outcome.output is None
    assert message in outcome.content


def test_error_messages_do_not_expose_file_paths(small_benchmark) -> None:
    outcome = _run(small_benchmark, "estimate_effect", {"estimator": "magic"})
    assert str(small_benchmark.public_only_root) not in outcome.content
    assert "Traceback" not in outcome.content


def test_tool_output_contains_no_ground_truth(small_benchmark) -> None:
    for name, tool_input in [
        ("discover_graph", {}),
        ("estimate_effect", {"estimator": "difference_in_means"}),
    ]:
        outcome = _run(small_benchmark, name, tool_input)
        assert not {"true_ate", "spec", "column_roles", "edges"} & set(outcome.output)
        assert "confounder" not in outcome.content
        assert "true_ate" not in outcome.content
