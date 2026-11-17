"""Run the evaluation: three approaches on the benchmark, scored and reported.

    uv run python -m causal_agent.eval.run                  # Benchmark v1
    uv run python -m causal_agent.eval.run --refute         # also check each method
    uv run python -m causal_agent.eval.run --benchmark v2   # Benchmark v2

Each benchmark has its own data directory (`data/benchmark`,
`data/benchmark_v2`) and its own default report directory (`docs/results`,
`docs/results/v2`). Every report carries a run record: the model, the
settings, the digest of the dataset specs and the commit of the code.

A run never replaces an existing report: it stops unless `--report-dir` names
a directory without one or `--overwrite` is passed. Ablations run only when
named with `--approaches`.

The three approaches receive only public data. Ground truth is read here,
after an approach has answered, to score it.

Outputs:
- `results/runs/<approach>/<dataset>.json`: one log per question, with every
  tool call (gitignored).
- `results/cache/`: cached model responses and refutation results
  (gitignored).
- `docs/results/`: `results.md`, `scores.csv` and `error_by_scenario.png`.
"""

import argparse
import hashlib
import json
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import anthropic

from causal_agent.agent.fixed_pipeline import run_fixed_pipeline
from causal_agent.agent.llm import AnthropicClient, CachedLLM
from causal_agent.agent.loop import run_agent, write_run_log
from causal_agent.agent.no_tools import run_no_tools
from causal_agent.bench.design import linear_regression_bias
from causal_agent.bench.fingerprint import spec_digest
from causal_agent.bench.generate import write_benchmark
from causal_agent.bench.public import load_public
from causal_agent.bench.spec import DatasetSpec, Scenario
from causal_agent.bench.suite import benchmark_specs, benchmark_v2_specs
from causal_agent.bench.truth import GroundTruth, load_truth
from causal_agent.config import Settings, load_api_key
from causal_agent.eval.report import (
    SCENARIO_LABELS,
    Result,
    plot_errors,
    results_markdown,
    write_scores_csv,
)
from causal_agent.eval.score import score_answer
from causal_agent.tools.estimators import EstimatorName
from causal_agent.tools.refute_estimate import (
    STABILITY_REFUTERS,
    RefuteEstimateInput,
    RefuterName,
    refute_estimate,
)

APPROACHES = ("fixed_pipeline", "agent", "no_tools")

# Extra approaches that are reported separately and never run by default.
# `no_tools_no_correlations` is the no-tools baseline with the correlation
# table removed from its summary.
ABLATIONS = ("no_tools_no_correlations",)
ALL_APPROACHES = (*APPROACHES, *ABLATIONS)


@dataclass(frozen=True)
class Benchmark:
    """Where a benchmark's datasets come from and where its files go."""

    title: str
    data_name: str
    """Directory name of the generated datasets, beside the default one."""

    report_subdirectory: str
    """Default report directory, relative to the settings' report directory."""

    specs: Callable[[Settings], list[DatasetSpec]]


BENCHMARKS = {
    "v1": Benchmark(
        title="Results",
        data_name="benchmark",
        report_subdirectory="",
        specs=lambda settings: benchmark_specs(settings.master_seed),
    ),
    "v2": Benchmark(
        title="Benchmark v2 results",
        data_name="benchmark_v2",
        report_subdirectory="v2",
        specs=lambda settings: benchmark_v2_specs(),
    ),
}


def refute_final_method(
    dataset_id: str,
    estimator: EstimatorName | None,
    adjustment_set: Sequence[str],
    root: Path,
    cache_dir: Path,
) -> dict[str, bool] | None:
    """Run the three pass-or-fail refuters on a method; results are cached on disk.

    The estimator-disagreement check is separate (`disagreement_of_final_method`)
    so these results keep the meaning they had in the Benchmark v1 evaluation.


    Returns None when the method cannot be run: no recognized estimator, or an
    adjustment set that names something other than a covariate column.
    """
    if estimator is None:
        return None
    _, metadata = load_public(dataset_id, root)
    covariates = {column.name for column in metadata.covariates}
    if not set(adjustment_set) <= covariates:
        return None

    outcomes: dict[str, bool] = {}
    for refuter in STABILITY_REFUTERS:
        if (
            estimator is EstimatorName.DIFFERENCE_IN_MEANS
            and refuter is RefuterName.RANDOM_COMMON_CAUSE
        ):
            continue
        request = RefuteEstimateInput(
            dataset_id=dataset_id,
            estimator=estimator,
            adjustment_set=tuple(sorted(adjustment_set)),
            refuter=refuter,
        )
        key = hashlib.sha256(request.model_dump_json().encode("utf-8")).hexdigest()
        path = Path(cache_dir) / "refutations" / f"{key}.json"
        if path.exists():
            passed = json.loads(path.read_text(encoding="utf-8"))["passed"]
        else:
            output = refute_estimate(request, root)
            passed = output.passed
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(output.model_dump_json(indent=1), encoding="utf-8")
        outcomes[refuter.value] = passed
    return outcomes


def disagreement_of_final_method(
    dataset_id: str,
    estimator: EstimatorName | None,
    adjustment_set: Sequence[str],
    root: Path,
    cache_dir: Path,
) -> dict[str, Any] | None:
    """Run the estimator-disagreement check on a method; cached on disk.

    Returns the check's output as a dict, or None when the method cannot be
    run (same rule as `refute_final_method`).
    """
    if estimator is None:
        return None
    _, metadata = load_public(dataset_id, root)
    covariates = {column.name for column in metadata.covariates}
    if not set(adjustment_set) <= covariates:
        return None
    request = RefuteEstimateInput(
        dataset_id=dataset_id,
        estimator=estimator,
        adjustment_set=tuple(sorted(adjustment_set)),
        refuter=RefuterName.ESTIMATOR_DISAGREEMENT,
    )
    key = hashlib.sha256(request.model_dump_json().encode("utf-8")).hexdigest()
    path = Path(cache_dir) / "refutations" / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    output = refute_estimate(request, root).model_dump(mode="json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=1), encoding="utf-8")
    return output


def _run_approach(
    approach: str, dataset_id: str, root: Path, llm: Any, settings: Settings
) -> tuple[Any, dict[str, Any]]:
    """Run one approach on one dataset. Returns its run record and answer fields."""
    if approach == "fixed_pipeline":
        run = run_fixed_pipeline(dataset_id, root)
        return run, {
            "estimate": run.estimate,
            "estimator": EstimatorName(run.estimator),
            "adjustment_set": run.adjustment_set,
            "outcome": "answered",
        }
    if approach == "agent":
        run = run_agent(dataset_id, root, llm, settings)
    else:
        run = run_no_tools(
            dataset_id,
            root,
            llm,
            settings,
            include_correlations=approach != "no_tools_no_correlations",
        )
    answer = run.final_answer
    fields: dict[str, Any] = {
        "estimate": answer.estimate if answer else None,
        "estimator": answer.estimator if answer else None,
        "adjustment_set": answer.adjustment_set if answer else (),
        "outcome": run.outcome,
        "steps": run.steps,
        "input_tokens": run.input_tokens,
        "output_tokens": run.output_tokens,
    }
    if approach == "agent":
        fields["n_tool_calls"] = len(run.tool_calls)
        fields["n_tool_errors"] = sum(call.error is not None for call in run.tool_calls)
        fields["answer_supported"] = run.answer_is_supported()
        fields["agent_refutations"] = run.refutations_of_final_method()
        fields["agent_disagreement"] = run.disagreement_checks()
    return run, fields


def design_label(spec: DatasetSpec) -> str:
    """A finer description of a dataset's design than its scenario."""
    if spec.scenario is Scenario.CONFOUNDED_NONLINEAR:
        sides = (spec.treatment_nonlinearity > 0, spec.outcome_nonlinearity > 0)
        return {
            (False, True): "Confounded, nonlinear outcome only",
            (True, False): "Confounded, nonlinear treatment only",
            (True, True): "Confounded, nonlinear on both sides",
        }[sides]
    if spec.scenario is Scenario.RANDOMIZED and spec.outcome_nonlinearity > 0:
        return "Randomized, nonlinear outcome"
    return SCENARIO_LABELS[spec.scenario.value]


def evaluate(
    settings: Settings,
    llm: Any,
    specs: Sequence[DatasetSpec],
    approaches: Sequence[str] = APPROACHES,
    refute: bool = False,
    log: Callable[[str], None] = print,
    record_design: bool = False,
) -> list[Result]:
    """Generate the datasets, run each approach on each one, and score.

    With `refute`, the evaluation also runs the three pass-or-fail refuters
    and the estimator-disagreement check on each final method. With
    `record_design`, each result carries a finer design label and the
    recorded population bias of linear regression for its dataset; both come
    from the private spec and are attached only after the approach answered.
    """
    root = settings.benchmark_root
    write_benchmark(specs, root)
    results: list[Result] = []
    for approach in approaches:
        if approach not in ALL_APPROACHES:
            raise ValueError(f"unknown approach: {approach}")
        for spec in specs:
            dataset_id = spec.dataset_id
            try:
                run, answer = _run_approach(approach, dataset_id, root, llm, settings)
            except anthropic.APIError as error:
                log(f"{approach} {dataset_id}: API error ({type(error).__name__})")
                run, answer = (
                    None,
                    {
                        "estimate": None,
                        "estimator": None,
                        "adjustment_set": (),
                        "outcome": "api_error",
                    },
                )
            if run is not None:
                write_run_log(run, settings.runs_dir / approach, dataset_id)

            # Ground truth is read only now, after the approach has answered.
            truth: GroundTruth = load_truth(dataset_id, root)
            score = score_answer(
                approach,
                truth,
                answer["estimate"],
                answer["estimator"],
                answer["adjustment_set"],
                settings.tolerance_sensitivity,
            )
            eval_refutations = None
            disagreement = None
            if refute and score.answered:
                method = (
                    dataset_id,
                    answer["estimator"],
                    score.adjustment_set,
                    root,
                    settings.cache_dir,
                )
                eval_refutations = refute_final_method(*method)
                disagreement = disagreement_of_final_method(*method)
            result = Result(
                score=score,
                outcome=answer["outcome"],
                steps=answer.get("steps"),
                n_tool_calls=answer.get("n_tool_calls"),
                answer_supported=answer.get("answer_supported"),
                agent_refutations=answer.get("agent_refutations", {}),
                eval_refutations=eval_refutations,
                n_tool_errors=answer.get("n_tool_errors"),
                input_tokens=answer.get("input_tokens"),
                output_tokens=answer.get("output_tokens"),
                agent_disagreement=answer.get("agent_disagreement", ()),
                eval_disagreement=disagreement["verdict"] if disagreement else None,
                eval_disagreement_reason=(
                    disagreement["reason"] if disagreement else None
                ),
                subtype=design_label(spec) if record_design else None,
                linear_bias_sd=(
                    linear_regression_bias(spec).bias_in_sd if record_design else None
                ),
            )
            results.append(result)
            shown = "no answer" if score.estimate is None else f"{score.estimate:+.3f}"
            log(
                f"{approach:24s} {dataset_id} {score.scenario:22s} "
                f"estimate {shown} (true {score.true_ate:+.2f}) {answer['outcome']}"
            )
    return results


def _git(*arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def run_record(
    benchmark: str,
    settings: Settings,
    specs: Sequence[DatasetSpec],
    approaches: Sequence[str],
    refute: bool,
    llm: Any,
) -> dict[str, Any]:
    """What was run: model, settings, datasets and the commit of the code."""
    status = _git("status", "--porcelain")
    return {
        "Benchmark": benchmark,
        "Run date (UTC)": datetime.now(UTC).strftime("%Y-%m-%d"),
        "Datasets": len(specs),
        "Dataset spec digest (SHA-256)": spec_digest(specs),
        "Approaches": ", ".join(approaches),
        "Model requested": settings.agent_model,
        "Model reported by the API": ", ".join(
            sorted(getattr(llm, "response_models", ())) or ["none"]
        ),
        "Effort": settings.effort,
        "Max output tokens per response": settings.max_output_tokens,
        "Max model calls per agent question": settings.max_agent_steps,
        "Tolerance (outcome SDs)": settings.ate_tolerance_sd,
        "Tolerances reported": ", ".join(
            f"{value:g}" for value in settings.tolerance_sensitivity
        ),
        "Refutation and disagreement checks run by the evaluation": refute,
        "Model calls made in this run": getattr(llm, "misses", "n/a"),
        "Model calls replayed from cache": getattr(llm, "hits", "n/a"),
        "Code commit": _git("rev-parse", "HEAD") or "unknown",
        "Working tree clean": "unknown" if status is None else status == "",
    }


def report_exists(report_dir: Path) -> bool:
    """Whether a results report has already been written to this directory."""
    return (Path(report_dir) / "results.md").exists()


def write_report(
    results: Sequence[Result],
    settings: Settings,
    title: str = "Results",
    overwrite: bool = False,
    record: Mapping[str, Any] | None = None,
) -> None:
    """Write the Markdown report, the CSV and the plot.

    Refuses to replace an existing report unless `overwrite` is set, so a
    published set of results cannot be replaced by accident. With `record`,
    the run record is shown in the report and saved as `run_record.json`.
    """
    report_dir = settings.report_dir
    if report_exists(report_dir) and not overwrite:
        raise FileExistsError(
            f"{report_dir} already holds a results report; choose another "
            "report directory or pass overwrite"
        )
    report_dir.mkdir(parents=True, exist_ok=True)
    markdown = results_markdown(
        results,
        settings.ate_tolerance_sd,
        settings.tolerance_sensitivity,
        title,
        record,
    )
    if record:
        (report_dir / "run_record.json").write_text(
            json.dumps(dict(record), indent=1) + "\n", encoding="utf-8", newline="\n"
        )
    (report_dir / "results.md").write_text(markdown, encoding="utf-8", newline="\n")
    write_scores_csv(results, report_dir / "scores.csv")
    plot_errors(
        results, settings.ate_tolerance_sd, report_dir / "error_by_scenario.png"
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--approaches",
        nargs="+",
        default=list(APPROACHES),
        choices=ALL_APPROACHES,
        help="default: the three main approaches; ablations must be named",
    )
    parser.add_argument(
        "--benchmark",
        choices=sorted(BENCHMARKS),
        default="v1",
        help="v1: the original 30 datasets; v2: the 20 frozen datasets of ADR 004",
    )
    parser.add_argument("--datasets", nargs="+", help="dataset IDs (default: all)")
    parser.add_argument("--refute", action="store_true")
    parser.add_argument("--model", help="override the model in the settings")
    parser.add_argument("--report-dir", type=Path, help="where to write the report")
    parser.add_argument("--title", help="title of the report")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace a report that already exists in the report directory",
    )
    arguments = parser.parse_args(argv)

    benchmark = BENCHMARKS[arguments.benchmark]
    settings = Settings()
    if arguments.model:
        settings = settings.model_copy(update={"agent_model": arguments.model})
    # Each benchmark has its own data directory and its own default report
    # directory, so running one can never touch the other's files.
    settings = settings.model_copy(
        update={
            "benchmark_root": settings.benchmark_root.with_name(benchmark.data_name),
            "report_dir": arguments.report_dir
            or settings.report_dir / benchmark.report_subdirectory,
        }
    )
    if report_exists(settings.report_dir) and not arguments.overwrite:
        parser.error(
            f"{settings.report_dir} already holds a results report. Pass "
            "--report-dir to write somewhere else, or --overwrite to replace it."
        )
    specs = benchmark.specs(settings)
    if arguments.datasets:
        specs = [spec for spec in specs if spec.dataset_id in arguments.datasets]

    llm = CachedLLM(lambda: AnthropicClient(load_api_key()), settings.cache_dir)
    results = evaluate(
        settings,
        llm,
        specs,
        arguments.approaches,
        arguments.refute,
        record_design=True,
    )
    record = run_record(
        arguments.benchmark,
        settings,
        specs,
        arguments.approaches,
        arguments.refute,
        llm,
    )
    write_report(
        results,
        settings,
        arguments.title or benchmark.title,
        arguments.overwrite,
        record,
    )
    print(f"model calls: {llm.misses} new, {llm.hits} from cache")
    print(f"report written to {settings.report_dir}")


if __name__ == "__main__":
    main()
