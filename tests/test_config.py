"""Tests for the project configuration."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from causal_agent import config
from causal_agent.config import API_KEY_VARIABLE, Settings, load_api_key


def test_defaults_match_the_project_rules() -> None:
    settings = Settings()
    assert settings.agent_model == "claude-haiku-5-5"
    assert settings.max_agent_steps == 8
    assert settings.ate_tolerance_sd == 0.10
    assert settings.tolerance_sensitivity == (0.05, 0.10, 0.20)
    assert settings.cache_dir == Path("results") / "cache"
    assert settings.runs_dir == Path("results") / "runs"


def test_settings_can_be_overridden_but_not_mutated() -> None:
    settings = Settings(agent_model="some-other-model", max_agent_steps=4)
    assert settings.agent_model == "some-other-model"
    with pytest.raises(ValidationError):
        settings.max_agent_steps = 99
    with pytest.raises(ValidationError):
        Settings(unknown_option=1)


def test_api_key_variable_is_the_project_specific_one() -> None:
    assert API_KEY_VARIABLE == "AGENT_LLM_KEY"


def test_key_is_read_from_the_project_variable(monkeypatch) -> None:
    monkeypatch.setattr(config, "load_dotenv", lambda: None)
    monkeypatch.setenv("AGENT_LLM_KEY", "  test-key-value  ")
    assert load_api_key() == "test-key-value"


def test_anthropic_api_key_is_never_used(monkeypatch) -> None:
    monkeypatch.setattr(config, "load_dotenv", lambda: None)
    monkeypatch.delenv("AGENT_LLM_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "should-be-ignored")
    with pytest.raises(RuntimeError, match="AGENT_LLM_KEY is not set") as raised:
        load_api_key()
    assert "should-be-ignored" not in str(raised.value)
