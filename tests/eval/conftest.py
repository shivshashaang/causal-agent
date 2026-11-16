"""Fixtures for the evaluation tests."""

import pytest

from causal_agent.bench.generate import generate
from causal_agent.bench.spec import DatasetSpec, Scenario


@pytest.fixture
def make_truth():
    """Build the ground truth of a small dataset with the given knobs."""

    def _make(scenario: Scenario = Scenario.CONFOUNDED_NONLINEAR, **knobs):
        fields = {
            "dataset_id": "ds_950",
            "scenario": scenario,
            "seed": 950,
            "n_rows": 500,
            "true_ate": 1.0,
            "assignment_strength": 1.0,
            "confounder_outcome_strength": 1.0,
            "outcome_noise": 1.0,
            **knobs,
        }
        return generate(DatasetSpec(**fields)).truth

    return _make
