"""The agent: one tool-using loop on the raw Anthropic SDK.

For one dataset the model gets the public summary and the three tools. Each
model call is one step; the number of steps is capped. On the last step the
model can no longer call tools and must answer. Every tool call is recorded
with its input and output for failure analysis.
"""

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from causal_agent.agent.answer import ANSWER_FORMAT, FinalAnswer, parse_final_answer
from causal_agent.agent.llm import assistant_content, response_text, tool_uses
from causal_agent.agent.summary import describe_dataset
from causal_agent.agent.toolbox import execute_tool, tool_definitions
from causal_agent.bench.public import load_public
from causal_agent.config import Settings

SYSTEM_PROMPT = """\
You are a careful causal inference analyst. You are given one tabular dataset \
and one question: what is the average treatment effect of the treatment column \
on the outcome column?

You cannot see the rows. You have a statistical summary of the data and three \
tools: discover_graph, estimate_effect and refute_estimate. Decide which \
columns to adjust for and which estimator to use, and say briefly why. Use \
causal reasoning and the timing labels as well as the tool outputs: the tools \
report statistics, and they do not decide for you what is appropriate to \
adjust for. Check your estimate before you commit to it.

You have at most {max_steps} turns. A turn may contain several tool calls. \
Your last turn must be your final answer.

{answer_format}"""

LAST_STEP_NOTICE = (
    "Your next turn is your last. You cannot call tools in it. Give your final "
    "answer in the required JSON form."
)

REPAIR_PROMPT = (
    "Your message did not contain the required JSON object. Reply with only "
    "that JSON object."
)

Outcome = Literal["answered", "no_answer", "refusal", "max_tokens"]

# How close the final estimate must be to a logged tool estimate to count as
# coming from that call (the model may round what it reports).
SUPPORT_TOLERANCE = 0.01


class ToolCallRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    step: int
    tool: str
    input: Any
    output: dict[str, Any] | None
    error: str | None


class AgentRun(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_id: str
    model: str
    steps: int
    outcome: Outcome
    final_answer: FinalAnswer | None
    final_text: str
    tool_calls: tuple[ToolCallRecord, ...]
    input_tokens: int
    output_tokens: int

    def _calls_for_final_method(self, tool: str) -> list[ToolCallRecord]:
        answer = self.final_answer
        if answer is None or answer.estimator is None:
            return []
        return [
            call
            for call in self.tool_calls
            if call.tool == tool
            and call.output is not None
            and call.output["estimator"] == answer.estimator.value
            and set(call.output["adjustment_set"]) == set(answer.adjustment_set)
        ]

    def answer_is_supported(self) -> bool:
        """Whether an estimate_effect call produced the reported estimate."""
        if self.final_answer is None:
            return False
        return any(
            abs(call.output["estimate"] - self.final_answer.estimate)
            <= SUPPORT_TOLERANCE
            for call in self._calls_for_final_method("estimate_effect")
        )

    def refutations_of_final_method(self) -> dict[str, bool]:
        """Pass-or-fail refuters the agent ran on its final method.

        The estimator-disagreement check has a three-way verdict, not a pass
        or fail, and is reported by `disagreement_checks`.
        """
        return {
            call.output["refuter"]: call.output["passed"]
            for call in self._calls_for_final_method("refute_estimate")
            if "passed" in call.output
        }

    def disagreement_checks(self) -> tuple[tuple[str, str], ...]:
        """Every estimator-disagreement check the agent ran: (estimator, verdict)."""
        return tuple(
            (call.output["estimator"], call.output["verdict"])
            for call in self.tool_calls
            if call.tool == "refute_estimate"
            and call.output is not None
            and "verdict" in call.output
        )


def system_prompt(settings: Settings) -> str:
    return SYSTEM_PROMPT.format(
        max_steps=settings.max_agent_steps, answer_format=ANSWER_FORMAT
    )


def run_agent(dataset_id: str, root: Path, llm: Any, settings: Settings) -> AgentRun:
    """Answer one dataset's question with the tool-using agent.

    `llm` is anything with a `create(request) -> response` method, normally a
    `CachedLLM`.
    """
    frame, metadata = load_public(dataset_id, root)
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": describe_dataset(frame, metadata)}
    ]
    tools = tool_definitions()
    records: list[ToolCallRecord] = []
    input_tokens = output_tokens = 0
    final_answer: FinalAnswer | None = None
    final_text = ""
    outcome: Outcome = "no_answer"
    repair_requested = False
    steps = 0

    for step in range(1, settings.max_agent_steps + 1):
        steps = step
        is_last_step = step == settings.max_agent_steps
        request: dict[str, Any] = {
            "model": settings.agent_model,
            "max_tokens": settings.max_output_tokens,
            "system": system_prompt(settings),
            "messages": list(messages),
            "tools": tools,
            "output_config": {"effort": settings.effort},
        }
        if is_last_step:
            request["tool_choice"] = {"type": "none"}

        response = llm.create(request)
        usage = response.get("usage") or {}
        input_tokens += usage.get("input_tokens", 0)
        output_tokens += usage.get("output_tokens", 0)
        messages.append({"role": "assistant", "content": assistant_content(response)})
        stop_reason = response.get("stop_reason")

        if stop_reason == "refusal":
            outcome = "refusal"
            break

        calls = tool_uses(response)
        if stop_reason == "tool_use" and calls and not is_last_step:
            results: list[dict[str, Any]] = []
            for call in calls:
                result = execute_tool(call["name"], call["input"], dataset_id, root)
                records.append(
                    ToolCallRecord(
                        step=step,
                        tool=call["name"],
                        input=call["input"],
                        output=result.output,
                        error=result.content if result.is_error else None,
                    )
                )
                block: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": call["id"],
                    "content": result.content,
                }
                if result.is_error:
                    block["is_error"] = True
                results.append(block)
            if step == settings.max_agent_steps - 1:
                results.append({"type": "text", "text": LAST_STEP_NOTICE})
            messages.append({"role": "user", "content": results})
            continue

        final_text = response_text(response)
        final_answer = parse_final_answer(final_text)
        if final_answer is not None:
            outcome = "answered"
            break
        if stop_reason == "max_tokens":
            outcome = "max_tokens"
            break
        if is_last_step or repair_requested:
            break
        repair_requested = True
        messages.append({"role": "user", "content": REPAIR_PROMPT})

    return AgentRun(
        dataset_id=dataset_id,
        model=settings.agent_model,
        steps=steps,
        outcome=outcome,
        final_answer=final_answer,
        final_text=final_text,
        tool_calls=tuple(records),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def write_run_log(run: BaseModel, directory: Path, dataset_id: str) -> Path:
    """Write one question's run, including every tool call, as JSON."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{dataset_id}.json"
    path.write_text(run.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path
