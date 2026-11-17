"""Tests for the evaluation runner, with a scripted model. No real API calls."""

import json

import pytest

from causal_agent.bench.spec import DatasetSpec, Scenario
from causal_agent.config import Settings
from causal_agent.eval.run import (
    APPROACHES,
    evaluate,
    refute_final_method,
    write_report,
)
from causal_agent.tools.estimators import EstimatorName

SPECS = [
    DatasetSpec(
        dataset_id="ds_801",
        scenario=Scenario.RANDOMIZED,
        seed=801,
        n_rows=1000,
        true_ate=1.0,
        assignment_strength=0.0,
        confounder_outcome_strength=1.0,
        outcome_noise=1.0,
    ),
    DatasetSpec(
        dataset_id="ds_802",
        scenario=Scenario.POST_OUTCOME_TRAP,
        seed=802,
        n_rows=1000,
        true_ate=1.0,
        assignment_strength=1.0,
        confounder_outcome_strength=1.0,
        outcome_noise=1.0,
        trap_strength=2.0,
    ),
]


class AnsweringLLM:
    """Always answers at once: difference in means, estimate 1.0."""

    def __init__(self) -> None:
        self.requests: list[dict] = []

    def create(self, request: dict) -> dict:
        self.requests.append(request)
        answer = {
            "estimate": 1.0,
            "estimator": "difference_in_means",
            "adjustment_set": [],
        }
        return {
            "content": [{"type": "text", "text": json.dumps(answer)}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        benchmark_root=tmp_path / "data",
        results_dir=tmp_path / "results",
        report_dir=tmp_path / "report",
    )


@pytest.fixture
def results(settings):
    return evaluate(settings, AnsweringLLM(), SPECS, log=lambda _: None)


def test_every_approach_answers_every_dataset(results) -> None:
    assert len(results) == len(APPROACHES) * len(SPECS)
    assert {(r.score.approach, r.score.dataset_id) for r in results} == {
        (approach, spec.dataset_id) for approach in APPROACHES for spec in SPECS
    }
    assert all(r.score.answered for r in results)


def test_fixed_pipeline_needs_no_model(settings) -> None:
    def fail(request):
        raise AssertionError("the fixed pipeline must not call the model")

    class NoLLM:
        create = staticmethod(fail)

    results = evaluate(
        settings, NoLLM(), SPECS, approaches=["fixed_pipeline"], log=lambda _: None
    )
    assert [r.score.estimator for r in results] == ["doubly_robust"] * 2
    assert all(r.steps is None for r in results)
    assert all(r.score.validity.adjustment_valid for r in results)


def test_scores_use_the_ground_truth_and_the_settings_tolerances(results) -> None:
    for result in results:
        assert result.score.true_ate == 1.0
        assert set(result.score.within_tolerance) == {"0.05", "0.10", "0.20"}
    llm_trap = [
        r
        for r in results
        if r.score.approach != "fixed_pipeline"
        and r.score.scenario == "post_outcome_trap"
    ]
    # The scripted model reported an unadjusted estimate on confounded data.
    assert all(r.score.validity.estimator_valid is False for r in llm_trap)
    assert all(r.score.within_tolerance["0.10"] for r in llm_trap)


def test_agent_fields_are_recorded(results) -> None:
    agent = [r for r in results if r.score.approach == "agent"]
    assert all(r.steps == 1 and r.n_tool_calls == 0 for r in agent)
    assert all(r.answer_supported is False for r in agent)
    no_tools = [r for r in results if r.score.approach == "no_tools"]
    assert all(r.n_tool_calls is None and r.answer_supported is None for r in no_tools)


def test_a_log_is_written_per_question_without_ground_truth(settings, results) -> None:
    for approach in APPROACHES:
        files = sorted(p.name for p in (settings.runs_dir / approach).iterdir())
        assert files == ["ds_801.json", "ds_802.json"]
        for path in (settings.runs_dir / approach).iterdir():
            text = path.read_text(encoding="utf-8")
            assert "true_ate" not in text
            assert "confounder" not in text


def test_models_never_receive_ground_truth(settings) -> None:
    llm = AnsweringLLM()
    evaluate(settings, llm, SPECS, approaches=["agent", "no_tools"], log=lambda _: None)
    sent = json.dumps(llm.requests).lower()
    for word in ("true_ate", "confound", "collider", "scenario", "seed"):
        assert word not in sent


def test_unknown_approach_is_rejected(settings) -> None:
    with pytest.raises(ValueError, match="unknown approach"):
        evaluate(settings, AnsweringLLM(), SPECS, approaches=["oracle"])


def test_refutation_is_off_by_default_and_runs_on_request(settings, results) -> None:
    assert all(r.eval_refutations is None for r in results)
    refuted = evaluate(
        settings,
        AnsweringLLM(),
        SPECS,
        approaches=["no_tools"],
        refute=True,
        log=lambda _: None,
    )
    # Difference in means cannot take the random-common-cause check.
    assert all(
        set(r.eval_refutations) == {"placebo_treatment", "data_subset"} for r in refuted
    )


def test_refutation_results_are_cached(settings, results) -> None:
    arguments = (
        "ds_801",
        EstimatorName.LINEAR_REGRESSION,
        ("x1", "x2"),
        settings.benchmark_root,
        settings.cache_dir,
    )
    first = refute_final_method(*arguments)
    cache_files = list((settings.cache_dir / "refutations").iterdir())
    assert len(cache_files) == 3
    stamps = {path: path.stat().st_mtime_ns for path in cache_files}
    assert refute_final_method(*arguments) == first
    assert {p: p.stat().st_mtime_ns for p in cache_files} == stamps
    assert set(first) == {"placebo_treatment", "random_common_cause", "data_subset"}


def test_unrunnable_methods_are_not_refuted(settings, results) -> None:
    root, cache = settings.benchmark_root, settings.cache_dir
    assert refute_final_method("ds_801", None, (), root, cache) is None
    assert (
        refute_final_method(
            "ds_801", EstimatorName.LINEAR_REGRESSION, ("nope",), root, cache
        )
        is None
    )


def test_report_files_are_written(settings, results) -> None:
    write_report(results, settings)
    names = sorted(path.name for path in settings.report_dir.iterdir())
    assert names == ["error_by_scenario.png", "results.md", "scores.csv"]
    markdown = (settings.report_dir / "results.md").read_text(encoding="utf-8")
    assert "## Accuracy by approach" in markdown
    assert "| ds_801 |" in markdown
