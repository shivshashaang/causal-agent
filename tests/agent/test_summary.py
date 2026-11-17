"""Tests for the public dataset summary."""

import pytest

from causal_agent.agent.summary import TIMING_LEGEND, describe_dataset
from causal_agent.bench.spec import Scenario

PRIVATE_WORDS = (
    "true_ate",
    "confound",
    "collider",
    "irrelevant",
    "propensity",
    "randomized",
    "scenario",
    "seed",
    "nonlinear",
)


def _summary(small_benchmark, scenario) -> str:
    dataset = small_benchmark.datasets[scenario]
    return describe_dataset(dataset.frame, dataset.public)


@pytest.mark.parametrize("scenario", list(Scenario))
def test_summary_contains_no_private_words(small_benchmark, scenario) -> None:
    text = _summary(small_benchmark, scenario).lower()
    assert [word for word in PRIVATE_WORDS if word in text] == []


@pytest.mark.parametrize("scenario", list(Scenario))
def test_summary_has_the_same_structure_for_every_scenario(
    small_benchmark, scenario
) -> None:
    text = _summary(small_benchmark, scenario)
    lines = text.splitlines()
    assert lines[0] == f"Dataset: {small_benchmark.dataset_id(scenario)}"
    assert lines[1].startswith("Question: What is the average treatment effect")
    assert "Rows: 2000" in lines
    for heading in (
        "Columns",
        "Column means by treatment group",
        "Pearson correlations",
    ):
        assert heading in lines
    assert TIMING_LEGEND in lines
    assert len(lines) == len(
        _summary(small_benchmark, Scenario.RANDOMIZED).splitlines()
    )


def test_summary_reports_the_naive_difference(small_benchmark) -> None:
    dataset = small_benchmark.datasets[Scenario.CONFOUNDED_LINEAR]
    frame = dataset.frame
    naive = frame.loc[frame.t == 1, "y"].mean() - frame.loc[frame.t == 0, "y"].mean()
    text = describe_dataset(frame, dataset.public)
    outcome_row = next(
        line
        for line in text.split("Column means by treatment group")[1].splitlines()
        if line.startswith("y ")
    )
    assert outcome_row.split()[-1] == f"{naive:.3f}"


def test_summary_shows_timing_labels_including_post_outcome(small_benchmark) -> None:
    text = _summary(small_benchmark, Scenario.POST_OUTCOME_TRAP)
    assert text.count("post_outcome") == 2  # one column row plus the legend
    assert (
        _summary(small_benchmark, Scenario.CONFOUNDED_LINEAR).count("post_outcome") == 1
    )


def test_summary_is_deterministic(small_benchmark) -> None:
    scenario = Scenario.CONFOUNDED_NONLINEAR
    assert _summary(small_benchmark, scenario) == _summary(small_benchmark, scenario)


def test_summary_counts_the_treatment_groups(small_benchmark) -> None:
    dataset = small_benchmark.datasets[Scenario.RANDOMIZED]
    treated = int(dataset.frame["t"].sum())
    text = describe_dataset(dataset.frame, dataset.public)
    assert f"{treated} treated rows, {2000 - treated} control rows" in text
