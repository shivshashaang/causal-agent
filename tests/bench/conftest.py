"""Shared fixtures for the benchmark tests."""

from collections.abc import Callable
from typing import Any

import pytest

from causal_agent.bench.spec import DatasetSpec, Scenario

# Knobs that make each scenario label valid. Tests override single values.
SCENARIO_KNOBS: dict[Scenario, dict[str, float]] = {
    Scenario.RANDOMIZED: {"assignment_strength": 0.0},
    Scenario.CONFOUNDED_LINEAR: {"assignment_strength": 1.0},
    Scenario.CONFOUNDED_NONLINEAR: {
        "assignment_strength": 1.0,
        "treatment_nonlinearity": 1.0,
        "outcome_nonlinearity": 1.0,
    },
    Scenario.POST_OUTCOME_TRAP: {"assignment_strength": 1.0, "trap_strength": 0.5},
}

SpecFactory = Callable[..., DatasetSpec]


@pytest.fixture
def make_spec() -> SpecFactory:
    def _make(
        scenario: Scenario = Scenario.CONFOUNDED_LINEAR, **overrides: Any
    ) -> DatasetSpec:
        fields: dict[str, Any] = {
            "dataset_id": "ds_test",
            "scenario": scenario,
            "seed": 7,
            "n_rows": 2000,
            "true_ate": 1.0,
            "confounder_outcome_strength": 1.0,
            "outcome_noise": 1.0,
            **SCENARIO_KNOBS[scenario],
            **overrides,
        }
        return DatasetSpec(**fields)

    return _make
