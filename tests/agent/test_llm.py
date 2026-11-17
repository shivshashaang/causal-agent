"""Tests for the cached model client. No test calls the real API."""

import json

import pytest

from causal_agent.agent.llm import (
    CachedLLM,
    assistant_content,
    request_key,
    response_text,
    tool_uses,
)


class CountingClient:
    def __init__(self) -> None:
        self.requests: list[dict] = []

    def create(self, request: dict) -> dict:
        self.requests.append(request)
        return {
            "content": [{"type": "text", "text": f"reply {len(self.requests)}"}],
            "stop_reason": "end_turn",
        }


def _request(**overrides) -> dict:
    return {
        "model": "model-a",
        "max_tokens": 100,
        "system": "be brief",
        "messages": [{"role": "user", "content": "hello"}],
        **overrides,
    }


def _cached(tmp_path, client=None):
    client = client or CountingClient()
    return CachedLLM(lambda: client, tmp_path / "cache"), client


def test_identical_request_is_served_from_disk(tmp_path) -> None:
    llm, client = _cached(tmp_path)
    first = llm.create(_request())
    second = llm.create(_request())
    assert first == second
    assert len(client.requests) == 1
    assert (llm.hits, llm.misses) == (1, 1)


def test_cache_survives_a_new_instance(tmp_path) -> None:
    first_llm, first_client = _cached(tmp_path)
    reply = first_llm.create(_request())
    second_llm, second_client = _cached(tmp_path)
    assert second_llm.create(_request()) == reply
    assert second_client.requests == []


@pytest.mark.parametrize(
    "change",
    [
        {"model": "model-b"},
        {"system": "be verbose"},
        {"messages": [{"role": "user", "content": "goodbye"}]},
        {"max_tokens": 200},
        {"tools": [{"name": "t"}]},
    ],
)
def test_any_change_to_prompt_or_model_is_a_new_request(tmp_path, change) -> None:
    llm, client = _cached(tmp_path)
    llm.create(_request())
    llm.create(_request(**change))
    assert len(client.requests) == 2


def test_key_does_not_depend_on_dictionary_order() -> None:
    forward = {"model": "m", "messages": [{"role": "user", "content": "x"}]}
    backward = {"messages": [{"content": "x", "role": "user"}], "model": "m"}
    assert request_key(forward) == request_key(backward)
    assert len(request_key(forward)) == 64


def test_client_is_not_created_when_everything_is_cached(tmp_path) -> None:
    llm, _ = _cached(tmp_path)
    llm.create(_request())

    def fail():
        raise AssertionError("the client must not be created on a cache hit")

    replay = CachedLLM(fail, tmp_path / "cache")
    assert replay.create(_request())["stop_reason"] == "end_turn"


def test_cache_file_holds_request_and_response_only(tmp_path) -> None:
    llm, _ = _cached(tmp_path)
    llm.create(_request())
    (path,) = (tmp_path / "cache").iterdir()
    entry = json.loads(path.read_text(encoding="utf-8"))
    assert set(entry) == {"request", "response"}
    assert entry["request"] == _request()
    assert path.stem == request_key(_request())
    assert "key" not in json.dumps(entry).lower()


def test_assistant_content_keeps_thinking_blocks_unchanged() -> None:
    response = {
        "content": [
            {"type": "thinking", "thinking": "", "signature": "sig-1"},
            {"type": "text", "text": "I will call a tool.", "citations": None},
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "estimate_effect",
                "input": {"estimator": "linear_regression"},
                "caller": {"type": "direct"},
            },
            {"type": "something_new", "payload": 1},
        ]
    }
    assert assistant_content(response) == [
        {"type": "thinking", "thinking": "", "signature": "sig-1"},
        {"type": "text", "text": "I will call a tool."},
        {
            "type": "tool_use",
            "id": "toolu_1",
            "name": "estimate_effect",
            "input": {"estimator": "linear_regression"},
        },
        {"type": "something_new", "payload": 1},
    ]


def test_text_and_tool_use_helpers_read_blocks_by_type() -> None:
    response = {
        "content": [
            {"type": "thinking", "thinking": "", "signature": "s"},
            {"type": "text", "text": "first"},
            {"type": "tool_use", "id": "a", "name": "n", "input": {}},
            {"type": "text", "text": "second"},
        ]
    }
    assert response_text(response) == "first\nsecond"
    assert [block["id"] for block in tool_uses(response)] == ["a"]
    assert response_text({"content": []}) == ""
