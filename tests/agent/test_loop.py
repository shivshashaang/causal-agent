"""Tests for the agent loop, driven by a scripted model. No real API calls."""

import json

import pytest

from causal_agent.agent.answer import ANSWER_FORMAT
from causal_agent.agent.loop import (
    LAST_STEP_NOTICE,
    REPAIR_PROMPT,
    AgentRun,
    run_agent,
    system_prompt,
    write_run_log,
)
from causal_agent.bench.spec import Scenario
from causal_agent.config import Settings
from causal_agent.tools.estimators import EstimatorName

SETTINGS = Settings()


class ScriptedLLM:
    """Returns prepared responses in order and records every request."""

    def __init__(self, responses) -> None:
        self._responses = list(responses)
        self.requests: list[dict] = []

    def create(self, request: dict) -> dict:
        self.requests.append(json.loads(json.dumps(request)))
        response = self._responses.pop(0)
        return response(request) if callable(response) else response


def tool_call(name: str, tool_input: dict, call_id: str = "toolu_1") -> dict:
    return {"type": "tool_use", "id": call_id, "name": name, "input": tool_input}


def tool_turn(*calls: dict, thinking: bool = False) -> dict:
    content = list(calls)
    if thinking:
        content.insert(0, {"type": "thinking", "thinking": "", "signature": "sig"})
    return {
        "content": content,
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 100, "output_tokens": 20},
    }


def text_turn(text: str, stop_reason: str = "end_turn") -> dict:
    return {
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop_reason,
        "usage": {"input_tokens": 50, "output_tokens": 10},
    }


def answer_turn(estimate, estimator="linear_regression", adjustment_set=("x1", "x3")):
    payload = {
        "estimate": estimate,
        "estimator": estimator,
        "adjustment_set": list(adjustment_set),
    }
    return text_turn("I adjusted for the confounders.\n" + json.dumps(payload))


def _run(small_benchmark, llm, settings=SETTINGS, scenario=Scenario.CONFOUNDED_LINEAR):
    return run_agent(
        small_benchmark.dataset_id(scenario),
        small_benchmark.public_only_root,
        llm,
        settings,
    )


def estimate_then_answer(request: dict) -> dict:
    """Answer with whatever the last estimate_effect call returned."""
    result = json.loads(request["messages"][-1]["content"][0]["content"])
    return answer_turn(round(result["estimate"], 3))


ESTIMATE_CALL = tool_call(
    "estimate_effect",
    {"estimator": "linear_regression", "adjustment_set": ["x1", "x3"]},
)


# --- A normal run ----------------------------------------------------------


def test_agent_discovers_estimates_refutes_and_answers(small_benchmark) -> None:
    refute = tool_call(
        "refute_estimate",
        {
            "estimator": "linear_regression",
            "adjustment_set": ["x1", "x3"],
            "refuter": "data_subset",
        },
        "toolu_3",
    )

    def final(request: dict) -> dict:
        estimate = json.loads(request["messages"][4]["content"][0]["content"])
        return answer_turn(round(estimate["estimate"], 3))

    llm = ScriptedLLM(
        [
            tool_turn(tool_call("discover_graph", {})),
            tool_turn(ESTIMATE_CALL),
            tool_turn(refute),
            final,
        ]
    )
    run = _run(small_benchmark, llm)

    assert run.outcome == "answered"
    assert run.steps == 4
    assert [call.tool for call in run.tool_calls] == [
        "discover_graph",
        "estimate_effect",
        "refute_estimate",
    ]
    assert [call.step for call in run.tool_calls] == [1, 2, 3]
    assert all(call.error is None and call.output for call in run.tool_calls)
    assert run.final_answer.estimator is EstimatorName.LINEAR_REGRESSION
    assert run.final_answer.adjustment_set == ("x1", "x3")
    assert run.answer_is_supported()
    assert run.refutations_of_final_method() == {"data_subset": True}
    assert run.input_tokens == 350
    assert run.output_tokens == 70


def test_first_request_follows_the_project_rules(small_benchmark) -> None:
    llm = ScriptedLLM([answer_turn(1.0)])
    _run(small_benchmark, llm)
    request = llm.requests[0]
    assert request["model"] == SETTINGS.agent_model == "claude-haiku-5-5"
    assert request["max_tokens"] == SETTINGS.max_output_tokens
    assert request["output_config"] == {"effort": "medium"}
    assert [tool["name"] for tool in request["tools"]] == [
        "discover_graph",
        "estimate_effect",
        "refute_estimate",
    ]
    assert "tool_choice" not in request
    assert not {"temperature", "top_p", "top_k", "thinking"} & set(request)
    assert request["system"] == system_prompt(SETTINGS)
    assert ANSWER_FORMAT in request["system"]
    assert "at most 8 turns" in request["system"]
    assert len(request["messages"]) == 1
    assert request["messages"][0]["content"].startswith("Dataset: ds_902")


def test_model_name_comes_from_settings(small_benchmark) -> None:
    llm = ScriptedLLM([answer_turn(1.0)])
    run = _run(small_benchmark, llm, Settings(agent_model="another-model"))
    assert llm.requests[0]["model"] == "another-model"
    assert run.model == "another-model"


def test_prompt_and_summary_reveal_nothing_private(small_benchmark) -> None:
    llm = ScriptedLLM([answer_turn(1.0)])
    _run(small_benchmark, llm, scenario=Scenario.POST_OUTCOME_TRAP)
    sent = json.dumps(llm.requests[0]).lower()
    for word in ("true_ate", "confound", "collider", "seed", "scenario", "randomized"):
        assert word not in sent


# --- Tool results ----------------------------------------------------------


def test_parallel_tool_calls_are_answered_in_one_message(small_benchmark) -> None:
    second = tool_call(
        "estimate_effect", {"estimator": "difference_in_means"}, "toolu_2"
    )
    llm = ScriptedLLM([tool_turn(ESTIMATE_CALL, second), answer_turn(1.0)])
    run = _run(small_benchmark, llm)
    reply = llm.requests[1]["messages"][-1]
    assert reply["role"] == "user"
    assert [block["type"] for block in reply["content"]] == ["tool_result"] * 2
    assert [block["tool_use_id"] for block in reply["content"]] == [
        "toolu_1",
        "toolu_2",
    ]
    assert len(run.tool_calls) == 2


def test_tool_errors_are_returned_to_the_model_and_logged(small_benchmark) -> None:
    bad = tool_call("estimate_effect", {"estimator": "magic"})
    llm = ScriptedLLM([tool_turn(bad), tool_turn(ESTIMATE_CALL), estimate_then_answer])
    run = _run(small_benchmark, llm)
    error_block = llm.requests[1]["messages"][-1]["content"][0]
    assert error_block["is_error"] is True
    assert "Invalid input" in error_block["content"]
    assert run.tool_calls[0].output is None
    assert "Invalid input" in run.tool_calls[0].error
    assert run.tool_calls[1].error is None
    assert run.outcome == "answered"


def test_model_cannot_point_a_tool_at_another_dataset(small_benchmark) -> None:
    other = tool_call(
        "estimate_effect", {"dataset_id": "ds_901", "estimator": "difference_in_means"}
    )
    llm = ScriptedLLM([tool_turn(other), answer_turn(1.0)])
    run = _run(small_benchmark, llm)
    assert "dataset_id is fixed" in run.tool_calls[0].error


def test_thinking_blocks_are_sent_back_unchanged(small_benchmark) -> None:
    llm = ScriptedLLM([tool_turn(ESTIMATE_CALL, thinking=True), answer_turn(1.0)])
    _run(small_benchmark, llm)
    assistant = llm.requests[1]["messages"][1]
    assert assistant["role"] == "assistant"
    assert assistant["content"][0] == {
        "type": "thinking",
        "thinking": "",
        "signature": "sig",
    }


def test_history_only_grows(small_benchmark) -> None:
    llm = ScriptedLLM(
        [tool_turn(ESTIMATE_CALL), tool_turn(ESTIMATE_CALL), answer_turn(1.0)]
    )
    _run(small_benchmark, llm)
    lengths = [len(request["messages"]) for request in llm.requests]
    assert lengths == [1, 3, 5]
    for earlier, later in zip(llm.requests, llm.requests[1:], strict=False):
        assert later["messages"][: len(earlier["messages"])] == earlier["messages"]


# --- Step cap --------------------------------------------------------------


def test_agent_stops_at_the_step_cap(small_benchmark) -> None:
    settings = Settings(max_agent_steps=4)
    llm = ScriptedLLM(
        [tool_turn(ESTIMATE_CALL)] * 3 + [text_turn("I ran out of time.")]
    )
    run = _run(small_benchmark, llm, settings)
    assert len(llm.requests) == 4
    assert run.steps == 4
    assert run.outcome == "no_answer"
    assert run.final_answer is None
    assert len(run.tool_calls) == 3


def test_last_step_cannot_call_tools_and_is_announced(small_benchmark) -> None:
    settings = Settings(max_agent_steps=3)
    llm = ScriptedLLM([tool_turn(ESTIMATE_CALL)] * 2 + [estimate_then_answer])
    run = _run(small_benchmark, llm, settings)
    assert "tool_choice" not in llm.requests[0]
    assert "tool_choice" not in llm.requests[1]
    assert llm.requests[2]["tool_choice"] == {"type": "none"}
    notice = llm.requests[2]["messages"][-1]["content"][-1]
    assert notice == {"type": "text", "text": LAST_STEP_NOTICE}
    assert llm.requests[1]["messages"][-1]["content"][-1]["type"] == "tool_result"
    assert run.outcome == "answered"
    assert run.steps == 3


def test_default_cap_is_eight_model_calls(small_benchmark) -> None:
    llm = ScriptedLLM([tool_turn(ESTIMATE_CALL)] * 7 + [estimate_then_answer])
    run = _run(small_benchmark, llm)
    assert len(llm.requests) == 8
    assert run.steps == 8
    assert run.outcome == "answered"


# --- Answers that go wrong -------------------------------------------------


def test_missing_json_gets_one_repair_request(small_benchmark) -> None:
    llm = ScriptedLLM([text_turn("The effect is about one."), answer_turn(1.0)])
    run = _run(small_benchmark, llm)
    assert llm.requests[1]["messages"][-1] == {"role": "user", "content": REPAIR_PROMPT}
    assert run.outcome == "answered"
    assert run.steps == 2


def test_two_unusable_answers_end_the_run(small_benchmark) -> None:
    llm = ScriptedLLM([text_turn("About one."), text_turn("Still about one.")])
    run = _run(small_benchmark, llm)
    assert run.outcome == "no_answer"
    assert run.final_answer is None
    assert run.final_text == "Still about one."
    assert run.steps == 2


def test_refusal_ends_the_run(small_benchmark) -> None:
    llm = ScriptedLLM([{"content": [], "stop_reason": "refusal"}])
    run = _run(small_benchmark, llm)
    assert run.outcome == "refusal"
    assert run.final_answer is None


def test_truncated_response_is_reported(small_benchmark) -> None:
    llm = ScriptedLLM([text_turn("Let me think", stop_reason="max_tokens")])
    run = _run(small_benchmark, llm)
    assert run.outcome == "max_tokens"


def test_answer_not_backed_by_a_tool_call_is_flagged(small_benchmark) -> None:
    llm = ScriptedLLM([tool_turn(ESTIMATE_CALL), answer_turn(123.0)])
    run = _run(small_benchmark, llm)
    assert run.outcome == "answered"
    assert not run.answer_is_supported()

    different_method = ScriptedLLM(
        [tool_turn(ESTIMATE_CALL), answer_turn(1.0, "doubly_robust")]
    )
    assert not _run(small_benchmark, different_method).answer_is_supported()


# --- Logging ---------------------------------------------------------------


def test_run_log_records_every_tool_call(small_benchmark, tmp_path) -> None:
    llm = ScriptedLLM([tool_turn(ESTIMATE_CALL), estimate_then_answer])
    run = _run(small_benchmark, llm)
    path = write_run_log(run, tmp_path / "runs" / "agent", run.dataset_id)
    assert path.name == "ds_902.json"
    logged = json.loads(path.read_text(encoding="utf-8"))
    assert logged["tool_calls"][0]["tool"] == "estimate_effect"
    assert logged["tool_calls"][0]["input"]["estimator"] == "linear_regression"
    assert logged["tool_calls"][0]["output"]["estimate"] == pytest.approx(
        run.tool_calls[0].output["estimate"]
    )
    assert AgentRun.model_validate(logged) == run
    assert "true_ate" not in path.read_text(encoding="utf-8")
