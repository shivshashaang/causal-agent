"""Tests for the no-tools baseline, driven by a scripted model."""

import json

from causal_agent.agent.answer import ANSWER_FORMAT
from causal_agent.agent.loop import REPAIR_PROMPT, run_agent
from causal_agent.agent.no_tools import run_no_tools, system_prompt
from causal_agent.bench.spec import Scenario
from causal_agent.config import Settings

SETTINGS = Settings()


class ScriptedLLM:
    def __init__(self, responses) -> None:
        self._responses = list(responses)
        self.requests: list[dict] = []

    def create(self, request: dict) -> dict:
        self.requests.append(json.loads(json.dumps(request)))
        return self._responses.pop(0)


def text_turn(text: str, stop_reason: str = "end_turn") -> dict:
    return {
        "content": [
            {"type": "thinking", "thinking": "", "signature": "sig"},
            {"type": "text", "text": text},
        ],
        "stop_reason": stop_reason,
        "usage": {"input_tokens": 40, "output_tokens": 15},
    }


ANSWER = text_turn(
    'The raw gap is confounded. {"estimate": 0.9, "estimator": '
    '"linear_regression", "adjustment_set": ["x1", "x3"]}'
)


def _run(small_benchmark, llm, scenario=Scenario.CONFOUNDED_LINEAR):
    return run_no_tools(
        small_benchmark.dataset_id(scenario),
        small_benchmark.public_only_root,
        llm,
        SETTINGS,
    )


def test_baseline_answers_in_one_call_without_tools(small_benchmark) -> None:
    llm = ScriptedLLM([ANSWER])
    run = _run(small_benchmark, llm)
    assert run.outcome == "answered"
    assert run.steps == 1
    assert run.final_answer.estimate == 0.9
    assert run.final_answer.adjustment_set == ("x1", "x3")
    assert (run.input_tokens, run.output_tokens) == (40, 15)

    request = llm.requests[0]
    assert "tools" not in request
    assert "tool_choice" not in request
    assert not {"temperature", "top_p", "top_k", "thinking"} & set(request)
    assert request["model"] == "claude-haiku-5-5"
    assert request["output_config"] == {"effort": "medium"}
    assert request["system"] == system_prompt()
    assert ANSWER_FORMAT in request["system"]
    assert "no tools" in request["system"]


def test_baseline_sees_exactly_the_summary_the_agent_sees(small_benchmark) -> None:
    baseline_llm = ScriptedLLM([ANSWER])
    agent_llm = ScriptedLLM([ANSWER])
    _run(small_benchmark, baseline_llm)
    run_agent(
        small_benchmark.dataset_id(Scenario.CONFOUNDED_LINEAR),
        small_benchmark.public_only_root,
        agent_llm,
        SETTINGS,
    )
    assert baseline_llm.requests[0]["messages"] == agent_llm.requests[0]["messages"]


def test_baseline_request_reveals_nothing_private(small_benchmark) -> None:
    llm = ScriptedLLM([ANSWER])
    _run(small_benchmark, llm, Scenario.POST_OUTCOME_TRAP)
    sent = json.dumps(llm.requests[0]).lower()
    for word in ("true_ate", "confound", "collider", "seed", "scenario", "randomized"):
        assert word not in sent


def test_missing_json_gets_one_repair_request(small_benchmark) -> None:
    llm = ScriptedLLM([text_turn("Roughly one."), ANSWER])
    run = _run(small_benchmark, llm)
    assert run.outcome == "answered"
    assert run.steps == 2
    assert llm.requests[1]["messages"][-2:] == [
        {"role": "assistant", "content": "Roughly one."},
        {"role": "user", "content": REPAIR_PROMPT},
    ]


def test_two_unusable_answers_end_the_run(small_benchmark) -> None:
    llm = ScriptedLLM([text_turn("Roughly one."), text_turn("One, I think.")])
    run = _run(small_benchmark, llm)
    assert run.outcome == "no_answer"
    assert run.final_answer is None
    assert len(llm.requests) == 2


def test_refusal_and_truncation_are_reported(small_benchmark) -> None:
    refused = _run(
        small_benchmark, ScriptedLLM([{"content": [], "stop_reason": "refusal"}])
    )
    assert refused.outcome == "refusal"
    truncated = _run(
        small_benchmark, ScriptedLLM([text_turn("Let me", stop_reason="max_tokens")])
    )
    assert truncated.outcome == "max_tokens"
