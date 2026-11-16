"""No-tools baseline: the same model and the same summary, without any tool.

The model sees exactly the dataset summary the agent sees and must answer
from it alone. It cannot run an estimator, so the estimator and adjustment
set in its answer are what it says its reasoning corresponds to.
"""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from causal_agent.agent.answer import ANSWER_FORMAT, FinalAnswer, parse_final_answer
from causal_agent.agent.llm import response_text
from causal_agent.agent.loop import REPAIR_PROMPT, Outcome
from causal_agent.agent.summary import describe_dataset
from causal_agent.bench.public import load_public
from causal_agent.config import Settings

SYSTEM_PROMPT = """\
You are a careful causal inference analyst. You are given one tabular dataset \
and one question: what is the average treatment effect of the treatment column \
on the outcome column?

You cannot see the rows and you have no tools. Work from the statistical \
summary alone. Use causal reasoning and the timing labels to decide what the \
summary implies, say briefly how you reached your number, and give your best \
estimate.

{answer_format}
For estimator, name the method your reasoning is closest to."""

MAX_CALLS = 2


class NoToolsRun(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_id: str
    model: str
    steps: int
    outcome: Outcome
    final_answer: FinalAnswer | None
    final_text: str
    input_tokens: int
    output_tokens: int


def system_prompt() -> str:
    return SYSTEM_PROMPT.format(answer_format=ANSWER_FORMAT)


def run_no_tools(
    dataset_id: str, root: Path, llm: Any, settings: Settings
) -> NoToolsRun:
    """Answer one dataset's question from the summary alone."""
    frame, metadata = load_public(dataset_id, root)
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": describe_dataset(frame, metadata)}
    ]
    input_tokens = output_tokens = 0
    final_answer: FinalAnswer | None = None
    final_text = ""
    outcome: Outcome = "no_answer"
    steps = 0

    for step in range(1, MAX_CALLS + 1):
        steps = step
        response = llm.create(
            {
                "model": settings.agent_model,
                "max_tokens": settings.max_output_tokens,
                "system": system_prompt(),
                "messages": list(messages),
                "output_config": {"effort": settings.effort},
            }
        )
        usage = response.get("usage") or {}
        input_tokens += usage.get("input_tokens", 0)
        output_tokens += usage.get("output_tokens", 0)
        stop_reason = response.get("stop_reason")
        if stop_reason == "refusal":
            outcome = "refusal"
            break

        final_text = response_text(response)
        final_answer = parse_final_answer(final_text)
        if final_answer is not None:
            outcome = "answered"
            break
        if stop_reason == "max_tokens":
            outcome = "max_tokens"
            break
        messages.append({"role": "assistant", "content": final_text or "(no text)"})
        messages.append({"role": "user", "content": REPAIR_PROMPT})

    return NoToolsRun(
        dataset_id=dataset_id,
        model=settings.agent_model,
        steps=steps,
        outcome=outcome,
        final_answer=final_answer,
        final_text=final_text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
