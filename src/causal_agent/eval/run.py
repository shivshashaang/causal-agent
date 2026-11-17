"""Run the evaluation: three approaches on the benchmark, scored and reported.

    uv run python -m causal_agent.eval.run            # everything
    uv run python -m causal_agent.eval.run --refute   # also refute each final method

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
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import anthropic

from causal_agent.agent.fixed_pipeline import run_fixed_pipeline
from causal_agent.agent.llm import AnthropicClient, CachedLLM
from causal_agent.agent.loop import run_agent, write_run_log
from causal_agent.agent.no_tools import run_no_tools
from causal_agent.bench.generate import write_benchmark
from causal_agent.bench.public import load_public
from causal_agent.bench.spec import DatasetSpec
from causal_agent.bench.suite import benchmark_specs
from causal_agent.bench.truth import GroundTruth, load_truth
from causal_agent.config import Settings, load_api_key
from causal_agent.eval.report import (
    Result,
    plot_errors,
    results_markdown,
    write_scores_csv,
)
from causal_agent.eval.score import score_answer
from causal_agent.tools.estimators import EstimatorName
from causal_agent.tools.refute_estimate import (
    RefuteEstimateInput,
    RefuterName,
    refute_estimate,
)

APPROACHES = ("fixed_pipeline", "agent", "no_tools")


def refute_final_method(
    dataset_id: str,
    estimator: EstimatorName | None,
    adjustment_set: Sequence[str],
    root: Path,
    cache_dir: Path,
) -> dict[str, bool] | None:
    """Run every applicable refuter on a method; results are cached on disk.

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
    for refuter in RefuterName:
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
    runner = run_agent if approach == "agent" else run_no_tools
    run = runner(dataset_id, root, llm, settings)
    answer = run.final_answer
    fields: dict[str, Any] = {
        "estimate": answer.estimate if answer else None,
        "estimator": answer.estimator if answer else None,
        "adjustment_set": answer.adjustment_set if answer else (),
        "outcome": run.outcome,
        "steps": run.steps,
    }
    if approach == "agent":
        fields["n_tool_calls"] = len(run.tool_calls)
        fields["answer_supported"] = run.answer_is_supported()
        fields["agent_refutations"] = run.refutations_of_final_method()
    return run, fields


def evaluate(
    settings: Settings,
    llm: Any,
    specs: Sequence[DatasetSpec],
    approaches: Sequence[str] = APPROACHES,
    refute: bool = False,
    log: Callable[[str], None] = print,
) -> list[Result]:
    """Generate the datasets, run each approach on each one, and score."""
    root = settings.benchmark_root
    write_benchmark(specs, root)
    results: list[Result] = []
    for approach in approaches:
        if approach not in APPROACHES:
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
            if refute and score.answered:
                eval_refutations = refute_final_method(
                    dataset_id,
                    answer["estimator"],
                    score.adjustment_set,
                    root,
                    settings.cache_dir,
                )
            result = Result(
                score=score,
                outcome=answer["outcome"],
                steps=answer.get("steps"),
                n_tool_calls=answer.get("n_tool_calls"),
                answer_supported=answer.get("answer_supported"),
                agent_refutations=answer.get("agent_refutations", {}),
                eval_refutations=eval_refutations,
            )
            results.append(result)
            shown = "no answer" if score.estimate is None else f"{score.estimate:+.3f}"
            log(
                f"{approach:14s} {dataset_id} {score.scenario:22s} "
                f"estimate {shown} (true {score.true_ate:+.1f}) {answer['outcome']}"
            )
    return results


def write_report(results: Sequence[Result], settings: Settings) -> None:
    """Write the Markdown report, the CSV and the plot."""
    report_dir = settings.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    markdown = results_markdown(
        results, settings.ate_tolerance_sd, settings.tolerance_sensitivity
    )
    (report_dir / "results.md").write_text(markdown, encoding="utf-8", newline="\n")
    write_scores_csv(results, report_dir / "scores.csv")
    plot_errors(
        results, settings.ate_tolerance_sd, report_dir / "error_by_scenario.png"
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--approaches", nargs="+", default=list(APPROACHES))
    parser.add_argument("--datasets", nargs="+", help="dataset IDs (default: all)")
    parser.add_argument("--refute", action="store_true")
    parser.add_argument("--model", help="override the model in the settings")
    arguments = parser.parse_args(argv)

    settings = Settings()
    if arguments.model:
        settings = settings.model_copy(update={"agent_model": arguments.model})
    specs = benchmark_specs(settings.master_seed)
    if arguments.datasets:
        specs = [spec for spec in specs if spec.dataset_id in arguments.datasets]

    llm = CachedLLM(lambda: AnthropicClient(load_api_key()), settings.cache_dir)
    results = evaluate(settings, llm, specs, arguments.approaches, arguments.refute)
    write_report(results, settings)
    print(f"model calls: {llm.misses} new, {llm.hits} from cache")
    print(f"report written to {settings.report_dir}")


if __name__ == "__main__":
    main()
