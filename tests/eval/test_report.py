"""Tests for the results table, CSV and plot."""

import csv

import pytest

from causal_agent.bench.spec import Scenario
from causal_agent.eval.report import (
    Result,
    accuracy_summary,
    plot_errors,
    results_markdown,
    write_scores_csv,
)
from causal_agent.eval.score import score_answer
from causal_agent.tools.estimators import EstimatorName

TOLERANCES = (0.05, 0.10, 0.20)
OLS = EstimatorName.LINEAR_REGRESSION
DIM = EstimatorName.DIFFERENCE_IN_MEANS


@pytest.fixture
def results(small_benchmark) -> list[Result]:
    """Hand-made answers with known errors, so every count can be checked."""
    found = []
    for scenario in Scenario:
        truth = small_benchmark.datasets[scenario].truth
        sd = truth.diagnostics.outcome_sd
        valid_set = min(truth.valid_adjustment_sets, key=len)

        accurate = score_answer(
            "agent", truth, 1.0 + 0.01 * sd, OLS, valid_set, TOLERANCES
        )
        found.append(
            Result(
                score=accurate,
                outcome="answered",
                steps=3,
                n_tool_calls=4,
                answer_supported=True,
                agent_refutations={"data_subset": True},
                eval_refutations={"placebo_treatment": True, "data_subset": True},
            )
        )
        pipeline = score_answer(
            "fixed_pipeline",
            truth,
            1.0 + 0.15 * sd,
            EstimatorName.DOUBLY_ROBUST,
            valid_set,
            TOLERANCES,
        )
        found.append(
            Result(
                score=pipeline,
                outcome="answered",
                eval_refutations={"placebo_treatment": True, "data_subset": False},
            )
        )
        if scenario is Scenario.RANDOMIZED:
            baseline = score_answer("no_tools", truth, None, None, (), TOLERANCES)
            found.append(Result(score=baseline, outcome="no_answer", steps=2))
        else:
            baseline = score_answer(
                "no_tools", truth, 1.0 + 0.5 * sd, DIM, (), TOLERANCES
            )
            found.append(Result(score=baseline, outcome="answered", steps=1))
    return found


def _approach(results, name):
    return [r for r in results if r.score.approach == name]


def test_accuracy_summary_counts_and_errors(results) -> None:
    agent = accuracy_summary(_approach(results, "agent"), 0.10)
    assert (agent["n"], agent["answered"], agent["within"]) == (4, 4, 4)
    assert agent["mean_abs_error_sd"] == pytest.approx(0.01)
    assert agent["median_abs_error_sd"] == pytest.approx(0.01)

    pipeline = _approach(results, "fixed_pipeline")
    assert accuracy_summary(pipeline, 0.10)["within"] == 0
    assert accuracy_summary(pipeline, 0.20)["within"] == 4

    baseline = accuracy_summary(_approach(results, "no_tools"), 0.10)
    assert (baseline["n"], baseline["answered"], baseline["within"]) == (4, 3, 0)
    assert baseline["mean_abs_error_sd"] == pytest.approx(0.5)


def test_summary_of_unanswered_group_has_no_error_statistics(results) -> None:
    unanswered = [r for r in results if not r.score.answered]
    summary = accuracy_summary(unanswered, 0.10)
    assert summary["answered"] == 0
    assert summary["mean_abs_error"] is None


def _row(markdown: str, section: str, label: str) -> list[str]:
    body = markdown.split(section)[1]
    line = next(row for row in body.splitlines() if row.startswith(f"| {label} |"))
    return [cell.strip() for cell in line.strip("|").split("|")]


def test_markdown_accuracy_table_matches_the_data(results) -> None:
    text = results_markdown(results, 0.10, TOLERANCES)
    agent = _row(text, "## Accuracy by approach", "Agent (tools)")
    assert agent[1:5] == ["4/4", "4/4", "4/4", "4/4"]
    pipeline = _row(text, "## Accuracy by approach", "Fixed pipeline")
    assert pipeline[1:5] == ["4/4", "0/4", "0/4", "4/4"]
    baseline = _row(text, "## Accuracy by approach", "LLM, no tools")
    assert baseline[1:5] == ["3/4", "0/4", "0/4", "0/4"]


def test_markdown_scenario_table_has_a_row_per_scenario(results) -> None:
    text = results_markdown(results, 0.10, TOLERANCES)
    for label in (
        "Randomized",
        "Confounded, linear",
        "Confounded, nonlinear",
        "Post-outcome trap",
    ):
        row = _row(text, "## Correct estimates by scenario", label)
        assert row[1:] == ["1/1", "0/1", "0/1"]


def test_markdown_validity_table_separates_validity_from_accuracy(results) -> None:
    text = results_markdown(results, 0.10, TOLERANCES)
    section = "## Validity, kept separate from accuracy"
    # Columns: adjustment valid, estimator valid, VA, VI, IA, II, none.
    # The nonlinear dataset breaks the linear regression's assumption.
    assert _row(text, section, "Agent (tools)")[1:] == [
        "4/4",
        "3/4",
        "3",
        "0",
        "1",
        "0",
        "0",
    ]
    assert _row(text, section, "Fixed pipeline")[1:] == [
        "4/4",
        "4/4",
        "0",
        "4",
        "0",
        "0",
        "0",
    ]
    # The unadjusted estimate is invalid on all three confounded datasets.
    assert _row(text, section, "LLM, no tools")[1:] == [
        "0/3",
        "0/3",
        "0",
        "0",
        "0",
        "3",
        "0",
    ]


def test_markdown_refutation_table_counts_flags(results) -> None:
    text = results_markdown(results, 0.10, TOLERANCES)
    lines = [
        line
        # Only the refutation section: stop at the next heading.
        for line in text.split("## Refutation outcomes")[1]
        .split("\n## ")[0]
        .splitlines()
        if line.startswith("| ")
    ]
    cells = [[c.strip() for c in line.strip("|").split("|")] for line in lines[1:]]
    assert [
        "Agent (tools)",
        "checks run by the agent itself",
        "4",
        "0/0",
        "0/4",
    ] in cells
    assert [
        "Fixed pipeline",
        "all checks, run by the evaluation",
        "4",
        "4/4",
        "0/0",
    ] in cells
    assert not any(row[0] == "LLM, no tools" for row in cells)


def test_markdown_reports_agent_behaviour_and_every_dataset(results) -> None:
    text = results_markdown(results, 0.10, TOLERANCES)
    assert "| Mean model calls per question | 3.0 |" in text
    assert "| Answers backed by a logged estimate_effect call | 4/4 |" in text
    assert "| Questions where it refuted its final method | 4/4 |" in text
    for dataset_id in ("ds_901", "ds_902", "ds_903", "ds_904"):
        assert f"| {dataset_id} |" in text
    assert "no answer" in text
    assert "true_ate" not in text.lower() or "True ATE" in text


def test_csv_has_one_row_per_result_with_separate_validity_columns(
    results, tmp_path
) -> None:
    path = tmp_path / "out" / "scores.csv"
    write_scores_csv(results, path)
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(results) == 12
    for column in (
        "abs_error_in_sd",
        "within_0.05_sd",
        "within_0.10_sd",
        "within_0.20_sd",
        "adjustment_valid",
        "outcome_model_correct",
        "propensity_model_correct",
        "model_assumption_met",
        "estimator_valid",
        "agent_refutations",
        "eval_refutations",
    ):
        assert column in rows[0]
    unanswered = next(row for row in rows if row["answered"] == "False")
    assert unanswered["estimate"] == "" and unanswered["estimator_valid"] == ""
    pipeline = next(row for row in rows if row["approach"] == "fixed_pipeline")
    assert pipeline["eval_refutations"] == "data_subset=fail placebo_treatment=pass"
    assert pipeline["model_assumption_met"] == ""


def test_plot_is_written(results, tmp_path) -> None:
    path = tmp_path / "plots" / "errors.png"
    plot_errors(results, 0.10, path)
    assert path.exists()
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert path.stat().st_size > 10_000
