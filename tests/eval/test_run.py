"""Tests for the evaluation runner, with a scripted model. No real API calls."""

import json
from pathlib import Path

import pytest

from causal_agent.bench.fingerprint import spec_digest
from causal_agent.bench.spec import DatasetSpec, Scenario
from causal_agent.config import Settings
from causal_agent.eval import run as run_module
from causal_agent.eval.report import ABLATION_NOTE, DISAGREEMENT_VERDICTS
from causal_agent.eval.run import (
    ABLATIONS,
    ALL_APPROACHES,
    APPROACHES,
    design_label,
    evaluate,
    refute_final_method,
    run_record,
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


def test_ablation_is_not_one_of_the_default_approaches(results) -> None:
    assert ABLATIONS == ("no_tools_no_correlations",)
    assert not set(ABLATIONS) & set(APPROACHES)
    assert ALL_APPROACHES == (*APPROACHES, *ABLATIONS)
    assert {r.score.approach for r in results} == set(APPROACHES)


def test_ablation_runs_beside_the_baseline_without_the_correlation_table(
    settings,
) -> None:
    llm = AnsweringLLM()
    results = evaluate(
        settings,
        llm,
        SPECS,
        approaches=["no_tools", "no_tools_no_correlations"],
        log=lambda _: None,
    )
    assert [r.score.approach for r in results] == (
        ["no_tools"] * 2 + ["no_tools_no_correlations"] * 2
    )
    summaries = [request["messages"][0]["content"] for request in llm.requests]
    assert all("Pearson correlations" in text for text in summaries[:2])
    assert all("Pearson correlations" not in text for text in summaries[2:])
    for approach in ("no_tools", "no_tools_no_correlations"):
        files = sorted(p.name for p in (settings.runs_dir / approach).iterdir())
        assert files == ["ds_801.json", "ds_802.json"]

    write_report(results, settings, title="Ablation")
    markdown = (settings.report_dir / "results.md").read_text(encoding="utf-8")
    assert markdown.startswith("# Ablation\n")
    assert ABLATION_NOTE in markdown
    assert "| LLM, no tools |" in markdown
    assert "| LLM, no tools, no correlations (ablation) |" in markdown


def test_report_without_the_ablation_has_no_ablation_note(settings, results) -> None:
    write_report(results, settings)
    markdown = (settings.report_dir / "results.md").read_text(encoding="utf-8")
    assert markdown.startswith("# Results\n")
    assert ABLATION_NOTE not in markdown
    assert "ablation" not in markdown.lower()


def test_an_existing_report_is_never_replaced_by_accident(settings, results) -> None:
    write_report(results, settings)
    before = {path.name: path.read_bytes() for path in settings.report_dir.iterdir()}
    with pytest.raises(FileExistsError, match="already holds a results report"):
        write_report(results[:1], settings)
    after = {path.name: path.read_bytes() for path in settings.report_dir.iterdir()}
    assert after == before

    write_report(results[:1], settings, overwrite=True)
    assert (settings.report_dir / "scores.csv").read_bytes() != before["scores.csv"]


def test_command_line_stops_before_running_when_a_report_exists(
    tmp_path, monkeypatch, capsys
) -> None:
    report_dir = tmp_path / "published"
    report_dir.mkdir()
    (report_dir / "results.md").write_text("published results", encoding="utf-8")

    def fail(*args, **kwargs):
        raise AssertionError("nothing may run when the report would be replaced")

    monkeypatch.setattr(run_module, "evaluate", fail)
    with pytest.raises(SystemExit):
        run_module.main(["--report-dir", str(report_dir)])
    assert "already holds a results report" in capsys.readouterr().err
    assert (report_dir / "results.md").read_text(encoding="utf-8") == (
        "published results"
    )


def test_refute_also_runs_the_disagreement_check_on_each_final_method(
    settings,
) -> None:
    results = evaluate(
        settings,
        AnsweringLLM(),
        SPECS,
        approaches=["fixed_pipeline", "no_tools"],
        refute=True,
        log=lambda _: None,
    )
    stability = {"placebo_treatment", "random_common_cause", "data_subset"}
    for result in results:
        assert result.eval_disagreement in DISAGREEMENT_VERDICTS
        assert result.eval_disagreement_reason
        assert set(result.eval_refutations) <= stability
    # The scripted model names difference in means, which has no adjustment
    # set, so there is nothing to compare.
    no_tools = [r for r in results if r.score.approach == "no_tools"]
    assert {r.eval_disagreement for r in no_tools} == {"inconclusive"}
    # The check's result is cached beside the other refutations.
    assert len(list((settings.cache_dir / "refutations").iterdir())) > 0

    write_report(results, settings)
    markdown = (settings.report_dir / "results.md").read_text(encoding="utf-8")
    assert "## Estimator-disagreement check" in markdown
    assert "not proof that the reported estimate is biased" in markdown
    assert "## Model calls, tool calls and errors" in markdown


def test_design_and_recorded_bias_are_attached_only_on_request(
    settings, results
) -> None:
    assert all(r.subtype is None and r.linear_bias_sd is None for r in results)

    class NoLLM:
        def create(self, request):
            raise AssertionError("no model call expected")

    recorded = evaluate(
        settings,
        NoLLM(),
        SPECS,
        approaches=["fixed_pipeline"],
        record_design=True,
        log=lambda _: None,
    )
    assert [r.subtype for r in recorded] == ["Randomized", "Post-outcome trap"]
    # Both designs have a linear outcome, so the recorded bias is about zero.
    assert all(abs(r.linear_bias_sd) < 0.02 for r in recorded)


def test_design_labels_split_the_nonlinear_scenario() -> None:
    def nonlinear(treatment: float, outcome: float) -> DatasetSpec:
        return DatasetSpec(
            dataset_id="ds_803",
            scenario=Scenario.CONFOUNDED_NONLINEAR,
            seed=803,
            true_ate=1.0,
            assignment_strength=1.0,
            confounder_outcome_strength=1.0,
            outcome_noise=1.0,
            treatment_nonlinearity=treatment,
            outcome_nonlinearity=outcome,
        )

    assert design_label(nonlinear(0.0, 1.0)) == "Confounded, nonlinear outcome only"
    assert design_label(nonlinear(1.0, 0.0)) == "Confounded, nonlinear treatment only"
    assert design_label(nonlinear(1.0, 1.0)) == "Confounded, nonlinear on both sides"
    assert design_label(SPECS[0]) == "Randomized"
    curved = SPECS[0].model_copy(update={"outcome_nonlinearity": 1.0})
    assert design_label(curved) == "Randomized, nonlinear outcome"


def test_run_record_names_the_model_settings_datasets_and_commit(
    settings, results
) -> None:
    llm = AnsweringLLM()
    record = run_record("v2", settings, SPECS, ["agent", "no_tools"], True, llm)
    assert record["Benchmark"] == "v2"
    assert record["Datasets"] == 2
    assert record["Dataset spec digest (SHA-256)"] == spec_digest(SPECS)
    assert record["Model requested"] == "claude-haiku-5-5"
    assert record["Effort"] == "medium"
    assert record["Max model calls per agent question"] == 8
    assert record["Tolerance (outcome SDs)"] == 0.10
    assert record["Approaches"] == "agent, no_tools"
    assert {"Code commit", "Working tree clean", "Model reported by the API"} <= set(
        record
    )

    write_report(results, settings, record=record)
    names = sorted(path.name for path in settings.report_dir.iterdir())
    assert names == [
        "error_by_scenario.png",
        "results.md",
        "run_record.json",
        "scores.csv",
    ]
    markdown = (settings.report_dir / "results.md").read_text(encoding="utf-8")
    assert "## Run record" in markdown
    assert spec_digest(SPECS) in markdown
    saved = json.loads((settings.report_dir / "run_record.json").read_text("utf-8"))
    assert saved == json.loads(json.dumps(record))


@pytest.mark.parametrize(
    ("arguments", "data", "report", "title", "first", "count"),
    [
        ([], "benchmark", "docs/results", "Results", "ds_001", 30),
        (
            ["--benchmark", "v2"],
            "benchmark_v2",
            "docs/results/v2",
            "Benchmark v2 results",
            "ds_101",
            20,
        ),
    ],
)
def test_command_line_keeps_the_two_benchmarks_apart(
    tmp_path, monkeypatch, arguments, data, report, title, first, count
) -> None:
    captured = {}

    def fake_evaluate(settings, llm, specs, approaches, refute, **options):
        captured.update(settings=settings, specs=specs, approaches=approaches)
        return []

    def fake_write_report(results, settings, title, overwrite, record):
        captured.update(title=title, record=record)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_module, "evaluate", fake_evaluate)
    monkeypatch.setattr(run_module, "write_report", fake_write_report)
    run_module.main(arguments)

    assert captured["settings"].benchmark_root == Path("data") / data
    assert captured["settings"].report_dir == Path(report)
    assert captured["title"] == title
    assert captured["specs"][0].dataset_id == first
    assert len(captured["specs"]) == count
    assert list(captured["approaches"]) == list(APPROACHES)
    assert captured["record"]["Benchmark"] == (arguments[-1] if arguments else "v1")
    assert captured["record"]["Dataset spec digest (SHA-256)"] == spec_digest(
        captured["specs"]
    )


def test_report_files_are_written(settings, results) -> None:
    write_report(results, settings)
    names = sorted(path.name for path in settings.report_dir.iterdir())
    assert names == ["error_by_scenario.png", "results.md", "scores.csv"]
    markdown = (settings.report_dir / "results.md").read_text(encoding="utf-8")
    assert "## Accuracy by approach" in markdown
    assert "| ds_801 |" in markdown
