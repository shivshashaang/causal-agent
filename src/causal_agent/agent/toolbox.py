"""The three tools as the model sees them, and the code that runs a tool call.

The model never supplies a dataset ID: the loop fixes it for the question
being answered, so a tool call cannot reach any other dataset. Tool inputs are
validated by the same pydantic models the tools use. There is no tool that
runs arbitrary code.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from causal_agent.tools.discover_graph import DiscoverGraphInput, discover_graph
from causal_agent.tools.estimate_effect import EstimateEffectInput, estimate_effect
from causal_agent.tools.refute_estimate import RefuteEstimateInput, refute_estimate


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_model: type[BaseModel]
    run: Callable[[Any, Path], BaseModel]


TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="discover_graph",
        description=(
            "Run a causal discovery algorithm (PC with partial-correlation "
            "tests) on the dataset's columns. Returns which columns remain "
            "statistically associated after conditioning on others, including "
            "which covariates are associated with the treatment and with the "
            "outcome. The result is statistical evidence, not proof of "
            "causation or of causal direction; directions marked 'timing' "
            "come from the timing labels, not from the data."
        ),
        input_model=DiscoverGraphInput,
        run=discover_graph,
    ),
    ToolSpec(
        name="estimate_effect",
        description=(
            "Estimate the average treatment effect with a chosen estimator and "
            "adjustment set. Estimators: difference_in_means (no adjustment; "
            "adjustment_set must be empty), linear_regression, "
            "propensity_weighting (logistic propensity model), doubly_robust "
            "(cross-fitted, gradient-boosted outcome and propensity models). "
            "Returns the estimate, its standard error, a 95% interval, "
            "diagnostics and the assumptions the estimator relies on. The tool "
            "does not check whether the adjustment set is appropriate."
        ),
        input_model=EstimateEffectInput,
        run=estimate_effect,
    ),
    ToolSpec(
        name="refute_estimate",
        description=(
            "Check an estimate's robustness by re-running it under a change "
            "that should not matter: placebo_treatment (shuffled treatment; "
            "the effect should vanish), random_common_cause (an added noise "
            "covariate), or data_subset (random 80% subsets). Reports whether "
            "the estimate held up. A pass is not evidence that the estimate is "
            "unbiased."
        ),
        input_model=RefuteEstimateInput,
        run=refute_estimate,
    ),
)

_TOOLS_BY_NAME = {tool.name: tool for tool in TOOLS}


@dataclass(frozen=True)
class ToolOutcome:
    """The result of one tool call, ready to send back and to log."""

    content: str
    is_error: bool
    output: dict[str, Any] | None


def _inline_references(node: Any, definitions: dict[str, Any]) -> Any:
    """Replace `$ref` pointers with the definitions they name."""
    if isinstance(node, dict):
        if "$ref" in node:
            name = node["$ref"].rsplit("/", 1)[-1]
            return _inline_references(definitions[name], definitions)
        return {
            key: _inline_references(value, definitions)
            for key, value in node.items()
            if key not in ("title", "$defs")
        }
    if isinstance(node, list):
        return [_inline_references(item, definitions) for item in node]
    return node


def input_schema(input_model: type[BaseModel]) -> dict[str, Any]:
    """JSON schema of a tool's input as shown to the model, without dataset_id."""
    schema = input_model.model_json_schema()
    schema = _inline_references(schema, schema.get("$defs", {}))
    schema["properties"].pop("dataset_id")
    schema["required"] = [name for name in schema["required"] if name != "dataset_id"]
    return schema


def tool_definitions() -> list[dict[str, Any]]:
    """Tool definitions in the Messages API format."""
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": input_schema(tool.input_model),
        }
        for tool in TOOLS
    ]


def _validation_message(error: ValidationError) -> str:
    parts = []
    for issue in error.errors():
        location = ".".join(str(item) for item in issue["loc"])
        parts.append(f"{location}: {issue['msg']}" if location else issue["msg"])
    return "Invalid input. " + "; ".join(parts)


def execute_tool(
    name: str, tool_input: Any, dataset_id: str, root: Path
) -> ToolOutcome:
    """Validate and run one tool call for the given dataset.

    Failures come back as an error result the model can read and correct;
    they never raise.
    """
    tool = _TOOLS_BY_NAME.get(name)
    if tool is None:
        known = ", ".join(_TOOLS_BY_NAME)
        return ToolOutcome(f"Unknown tool '{name}'. Available: {known}.", True, None)
    if not isinstance(tool_input, dict):
        return ToolOutcome("Invalid input. Expected a JSON object.", True, None)
    if "dataset_id" in tool_input:
        return ToolOutcome(
            "Invalid input. dataset_id is fixed for this question; do not pass it.",
            True,
            None,
        )
    try:
        request = tool.input_model.model_validate(
            {**tool_input, "dataset_id": dataset_id}
        )
        output = tool.run(request, root)
    except ValidationError as error:
        return ToolOutcome(_validation_message(error), True, None)
    except ValueError as error:
        return ToolOutcome(f"Tool error. {error}", True, None)
    return ToolOutcome(output.model_dump_json(), False, output.model_dump(mode="json"))
