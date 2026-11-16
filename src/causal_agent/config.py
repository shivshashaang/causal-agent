"""Project configuration: every tunable value lives here.

The API key is read from the `AGENT_LLM_KEY` environment variable (loaded from
`.env` by python-dotenv). It is never printed, logged, cached or committed, and
`ANTHROPIC_API_KEY` is never used.
"""

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

API_KEY_VARIABLE = "AGENT_LLM_KEY"


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    agent_model: str = "claude-haiku-5-5"
    """Model used by the agent and by the no-tools baseline."""

    effort: Literal["low", "medium", "high"] = "medium"
    """How much the model thinks. Sent explicitly so runs are comparable."""

    max_output_tokens: int = Field(default=8000, ge=1000)
    """Per-response cap. Thinking counts toward it, so it is generous."""

    max_agent_steps: int = Field(default=8, ge=2)
    """Maximum number of model calls per question."""

    ate_tolerance_sd: float = Field(default=0.10, gt=0)
    """An estimate is correct within this many standard deviations of Y."""

    tolerance_sensitivity: tuple[float, ...] = (0.05, 0.10, 0.20)
    """Tolerances reported side by side in the results."""

    master_seed: int = 0
    benchmark_root: Path = Path("data/benchmark")
    results_dir: Path = Path("results")
    report_dir: Path = Path("docs/results")

    @property
    def cache_dir(self) -> Path:
        """Disk cache of model responses. Gitignored."""
        return self.results_dir / "cache"

    @property
    def runs_dir(self) -> Path:
        """Per-question logs of every tool call. Gitignored."""
        return self.results_dir / "runs"


def load_api_key() -> str:
    """Return the API key from `AGENT_LLM_KEY`, loading `.env` if present."""
    load_dotenv()
    key = os.environ.get(API_KEY_VARIABLE, "").strip()
    if not key:
        raise RuntimeError(
            f"{API_KEY_VARIABLE} is not set. Add it to .env or the environment."
        )
    return key
