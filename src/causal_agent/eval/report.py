"""Turn scored results into a results table, a CSV and one plot.

Every number here is computed from the results it is given. Nothing is
filled in by hand.
"""

import csv
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean, median

from pydantic import BaseModel, ConfigDict

from causal_agent.eval.score import Score, tolerance_key

APPROACH_LABELS = {
    "agent": "Agent (tools)",
    "fixed_pipeline": "Fixed pipeline",
    "no_tools": "LLM, no tools",
    "no_tools_no_correlations": "LLM, no tools, no correlations (ablation)",
}
# The ablation is an extra approach. It never replaces the no-tools baseline.
ABLATION_NOTE = (
    "`LLM, no tools, no correlations (ablation)` is the no-tools baseline with "
    "the correlation table removed from its summary and nothing else changed. "
    "It is reported in addition to `LLM, no tools`, not in place of it."
)
# Shorter names for the plot's axis, where the full label does not fit.
PLOT_LABELS = {
    "agent": "Agent\n(tools)",
    "fixed_pipeline": "Fixed\npipeline",
    "no_tools": "LLM,\nno tools",
    "no_tools_no_correlations": "No tools,\nno corr.\n(ablation)",
}
SCENARIO_LABELS = {
    "randomized": "Randomized",
    "confounded_linear": "Confounded, linear",
    "confounded_nonlinear": "Confounded, nonlinear",
    "post_outcome_trap": "Post-outcome trap",
}

# Categorical slots 1 to 4 of the validated default palette, in fixed order,
# with a distinct marker per approach so identity never rests on color alone.
SERIES_STYLE = {
    "agent": ("#2a78d6", "o"),
    "fixed_pipeline": ("#eb6834", "s"),
    "no_tools": ("#1baf7a", "^"),
    "no_tools_no_correlations": ("#eda100", "D"),
}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"

# Errors smaller than this are drawn at this height on the log axis.
PLOT_FLOOR_SD = 0.001


class Result(BaseModel):
    """One approach's scored answer to one question, with run details."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    score: Score
    outcome: str
    steps: int | None = None
    n_tool_calls: int | None = None
    answer_supported: bool | None = None
    """Agent only: an estimate_effect call produced the reported estimate."""

    agent_refutations: dict[str, bool] = {}
    """Agent only: refuters it ran itself on its final method, pass or fail."""

    eval_refutations: dict[str, bool] | None = None
    """Refuters the evaluation ran on the final method. None if not run."""

    n_tool_errors: int | None = None
    """Agent only: tool calls that came back as an error."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    agent_disagreement: tuple[tuple[str, str], ...] = ()
    """Agent only: (estimator, verdict) of each estimator-disagreement check it ran."""

    eval_disagreement: str | None = None
    """Verdict of the estimator-disagreement check the evaluation ran on the
    final method: disagreement, agreement or inconclusive. None if not run."""

    eval_disagreement_reason: str | None = None
    subtype: str | None = None
    """A finer description of the dataset's design than the scenario."""

    linear_bias_sd: float | None = None
    """Recorded population bias of linear regression for the dataset's spec, in
    outcome SDs. A property of the data-generating process, not a result."""


def _fraction(count: int, total: int) -> str:
    return f"{count}/{total}"


def _markdown_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _approaches(results: Sequence[Result]) -> list[str]:
    present = {result.score.approach for result in results}
    return [name for name in APPROACH_LABELS if name in present]


def _scenarios(results: Sequence[Result]) -> list[str]:
    present = {result.score.scenario for result in results}
    return [name for name in SCENARIO_LABELS if name in present]


def _of(results: Sequence[Result], approach: str, scenario: str | None = None):
    return [
        result
        for result in results
        if result.score.approach == approach
        and (scenario is None or result.score.scenario == scenario)
    ]


def accuracy_summary(results: Sequence[Result], tolerance_sd: float) -> dict:
    """Counts and error statistics for a group of results."""
    key = tolerance_key(tolerance_sd)
    answered = [r for r in results if r.score.answered]
    errors_sd = [r.score.abs_error_in_sd for r in answered]
    return {
        "n": len(results),
        "answered": len(answered),
        "within": sum(r.score.within_tolerance[key] for r in results),
        "mean_abs_error": mean(r.score.abs_error for r in answered)
        if answered
        else None,
        "mean_abs_error_sd": mean(errors_sd) if answered else None,
        "median_abs_error_sd": median(errors_sd) if answered else None,
    }


def _format(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _accuracy_table(results, tolerances_sd) -> str:
    header = [
        "Approach",
        "Answered",
        *(f"Within {tolerance_key(t)} SD" for t in tolerances_sd),
        "Mean abs. error",
        "Mean abs. error (SD)",
        "Median abs. error (SD)",
    ]
    rows = []
    for approach in _approaches(results):
        group = _of(results, approach)
        summary = accuracy_summary(group, tolerances_sd[0])
        rows.append(
            [
                APPROACH_LABELS[approach],
                _fraction(summary["answered"], summary["n"]),
                *(
                    _fraction(accuracy_summary(group, t)["within"], summary["n"])
                    for t in tolerances_sd
                ),
                _format(summary["mean_abs_error"]),
                _format(summary["mean_abs_error_sd"]),
                _format(summary["median_abs_error_sd"]),
            ]
        )
    return _markdown_table(header, rows)


def _scenario_table(results, tolerance_sd) -> str:
    approaches = _approaches(results)
    header = ["Scenario", *(APPROACH_LABELS[a] for a in approaches)]
    rows = []
    for scenario in _scenarios(results):
        row = [SCENARIO_LABELS[scenario]]
        for approach in approaches:
            summary = accuracy_summary(_of(results, approach, scenario), tolerance_sd)
            row.append(_fraction(summary["within"], summary["n"]))
        rows.append(row)
    return _markdown_table(header, rows)


def _validity_table(results, tolerance_sd) -> str:
    key = tolerance_key(tolerance_sd)
    header = [
        "Approach",
        "Adjustment set valid",
        "Estimator valid",
        "Valid and accurate",
        "Valid, inaccurate",
        "Invalid, accurate",
        "Invalid and inaccurate",
        "No recognized estimator",
    ]
    rows = []
    for approach in _approaches(results):
        answered = [r for r in _of(results, approach) if r.score.answered]
        judged = [r for r in answered if r.score.validity.estimator_valid is not None]
        cells = Counter(
            (r.score.validity.estimator_valid, r.score.within_tolerance[key])
            for r in judged
        )
        rows.append(
            [
                APPROACH_LABELS[approach],
                _fraction(
                    sum(r.score.validity.adjustment_valid for r in answered),
                    len(answered),
                ),
                _fraction(
                    sum(r.score.validity.estimator_valid for r in judged), len(judged)
                ),
                str(cells[(True, True)]),
                str(cells[(True, False)]),
                str(cells[(False, True)]),
                str(cells[(False, False)]),
                str(len(answered) - len(judged)),
            ]
        )
    return _markdown_table(header, rows)


def _estimator_table(results) -> str:
    estimators = sorted(
        {r.score.estimator or "none stated" for r in results if r.score.answered}
    )
    header = ["Approach", *estimators]
    rows = []
    for approach in _approaches(results):
        counts = Counter(
            r.score.estimator or "none stated"
            for r in _of(results, approach)
            if r.score.answered
        )
        rows.append([APPROACH_LABELS[approach], *(str(counts[e]) for e in estimators)])
    return _markdown_table(header, rows)


def _flag_counts(group, refutations, key) -> tuple[int, int, int, int]:
    """(inaccurate flagged, inaccurate, accurate flagged, accurate)."""
    checked = [(r, refutations(r)) for r in group if r.score.answered]
    checked = [(r, outcome) for r, outcome in checked if outcome]
    inaccurate = [o for r, o in checked if not r.score.within_tolerance[key]]
    accurate = [o for r, o in checked if r.score.within_tolerance[key]]
    flagged = lambda outcomes: sum(not all(o.values()) for o in outcomes)  # noqa: E731
    return flagged(inaccurate), len(inaccurate), flagged(accurate), len(accurate)


def _refutation_table(results, tolerance_sd) -> str | None:
    key = tolerance_key(tolerance_sd)
    sources = [
        ("checks run by the agent itself", lambda r: r.agent_refutations),
        ("all checks, run by the evaluation", lambda r: r.eval_refutations),
    ]
    rows = []
    for approach in _approaches(results):
        for label, refutations in sources:
            counts = _flag_counts(_of(results, approach), refutations, key)
            if counts[1] + counts[3] == 0:
                continue
            rows.append(
                [
                    APPROACH_LABELS[approach],
                    label,
                    str(counts[1] + counts[3]),
                    _fraction(counts[0], counts[1]),
                    _fraction(counts[2], counts[3]),
                ]
            )
    if not rows:
        return None
    header = [
        "Approach",
        "Refutation source",
        "Estimates checked",
        "Inaccurate estimates flagged",
        "Accurate estimates flagged",
    ]
    return _markdown_table(header, rows)


DISAGREEMENT_VERDICTS = ("disagreement", "agreement", "inconclusive")


def _disagreement_table(results, tolerance_sd) -> str | None:
    """The evaluation's estimator-disagreement check against accuracy.

    Disagreement on an inaccurate estimate is a detected problem; on an
    accurate one, a false alarm. Inconclusive results are counted on their
    own and never as either.
    """
    key = tolerance_key(tolerance_sd)
    rows = []
    for approach in _approaches(results):
        checked = [
            r
            for r in _of(results, approach)
            if r.score.answered and r.eval_disagreement is not None
        ]
        if not checked:
            continue
        counts = Counter(
            (r.eval_disagreement, r.score.within_tolerance[key]) for r in checked
        )
        inaccurate = sum(not r.score.within_tolerance[key] for r in checked)
        accurate = len(checked) - inaccurate
        rows.append(
            [
                APPROACH_LABELS[approach],
                str(len(checked)),
                _fraction(counts[("disagreement", False)], inaccurate),
                _fraction(counts[("disagreement", True)], accurate),
                _fraction(counts[("agreement", False)], inaccurate),
                _fraction(counts[("agreement", True)], accurate),
                _fraction(counts[("inconclusive", False)], inaccurate),
                _fraction(counts[("inconclusive", True)], accurate),
            ]
        )
    if not rows:
        return None
    header = [
        "Approach",
        "Estimates checked",
        "Detected: disagreement on an inaccurate estimate",
        "False alarm: disagreement on an accurate estimate",
        "Missed: agreement on an inaccurate estimate",
        "Agreement on an accurate estimate",
        "Inconclusive, inaccurate estimate",
        "Inconclusive, accurate estimate",
    ]
    return _markdown_table(header, rows)


def _subtype_table(results, tolerance_sd) -> str | None:
    """Correct estimates by the finer design description, when one is recorded."""
    subtypes = list(dict.fromkeys(r.subtype for r in results if r.subtype))
    if not subtypes:
        return None
    key = tolerance_key(tolerance_sd)
    approaches = _approaches(results)
    header = ["Design", "Datasets", *(APPROACH_LABELS[a] for a in approaches)]
    rows = []
    for subtype in sorted(subtypes):
        group = [r for r in results if r.subtype == subtype]
        row = [subtype, str(len({r.score.dataset_id for r in group}))]
        for approach in approaches:
            own = [r for r in group if r.score.approach == approach]
            within = sum(r.score.within_tolerance[key] for r in own)
            row.append(_fraction(within, len(own)))
        rows.append(row)
    return _markdown_table(header, rows)


def _totals_table(results) -> str:
    """Model calls, tool calls, errors and unanswered questions per approach."""
    header = [
        "Approach",
        "Questions",
        "Answered",
        "Not answered",
        "Model calls",
        "Tool calls",
        "Tool errors",
        "Input tokens",
        "Output tokens",
    ]

    def total(values) -> str:
        values = [value for value in values if value is not None]
        return str(sum(values)) if values else "n/a"

    rows = []
    for approach in _approaches(results):
        group = _of(results, approach)
        answered = sum(r.score.answered for r in group)
        failures = Counter(r.outcome for r in group if not r.score.answered)
        not_answered = (
            ", ".join(f"{count} {name}" for name, count in sorted(failures.items()))
            or "0"
        )
        model_calls = (
            "0" if approach == "fixed_pipeline" else total(r.steps for r in group)
        )
        rows.append(
            [
                APPROACH_LABELS[approach],
                str(len(group)),
                str(answered),
                not_answered,
                model_calls,
                total(r.n_tool_calls for r in group),
                total(r.n_tool_errors for r in group),
                total(r.input_tokens for r in group),
                total(r.output_tokens for r in group),
            ]
        )
    return _markdown_table(header, rows)


def _agent_table(results) -> str | None:
    runs = [r for r in _of(results, "agent") if r.steps is not None]
    if not runs:
        return None
    answered = [r for r in runs if r.score.answered]
    checks = Counter(verdict for r in runs for _, verdict in r.agent_disagreement)
    extra = []
    if any(r.agent_disagreement for r in runs):
        extra = [
            [
                "Questions where it ran the estimator-disagreement check",
                _fraction(sum(bool(r.agent_disagreement) for r in runs), len(runs)),
            ],
            [
                "Verdicts of the checks it ran "
                "(disagreement / agreement / inconclusive)",
                " / ".join(str(checks[v]) for v in DISAGREEMENT_VERDICTS),
            ],
        ]
    rows = [
        ["Questions", str(len(runs))],
        ["Mean model calls per question", f"{mean(r.steps for r in runs):.1f}"],
        ["Most model calls on one question", str(max(r.steps for r in runs))],
        ["Mean tool calls per question", f"{mean(r.n_tool_calls for r in runs):.1f}"],
        [
            "Answers backed by a logged estimate_effect call",
            _fraction(sum(bool(r.answer_supported) for r in answered), len(answered)),
        ],
        [
            "Questions where it refuted its final method",
            _fraction(sum(bool(r.agent_refutations) for r in runs), len(runs)),
        ],
        *extra,
    ]
    return _markdown_table(["Agent behaviour", "Value"], rows)


def _dataset_table(results, tolerance_sd) -> str:
    key = tolerance_key(tolerance_sd)
    approaches = _approaches(results)
    by_dataset: dict[str, dict[str, Result]] = {}
    for result in results:
        by_dataset.setdefault(result.score.dataset_id, {})[result.score.approach] = (
            result
        )
    with_bias = any(r.linear_bias_sd is not None for r in results)
    header = ["Dataset", "Scenario", "True ATE"]
    if with_bias:
        header.append("Recorded linear regression bias (SD)")
    for approach in approaches:
        header += [APPROACH_LABELS[approach], "Error (SD)", "Within"]
    rows = []
    for dataset_id in sorted(by_dataset):
        group = by_dataset[dataset_id]
        any_result = next(iter(group.values()))
        first = any_result.score
        design = any_result.subtype or SCENARIO_LABELS[first.scenario]
        row = [dataset_id, design, f"{first.true_ate:g}"]
        if with_bias:
            bias = any_result.linear_bias_sd
            row.append("n/a" if bias is None else f"{bias:+.3f}")
        for approach in approaches:
            result = group.get(approach)
            if result is None or not result.score.answered:
                row += ["no answer", "n/a", "no"]
                continue
            score = result.score
            row += [
                f"{score.estimate:.3f}",
                f"{score.abs_error_in_sd:.3f}",
                "yes" if score.within_tolerance[key] else "no",
            ]
        rows.append(row)
    return _markdown_table(header, rows)


def results_markdown(
    results: Sequence[Result],
    tolerance_sd: float,
    tolerances_sd: Sequence[float],
    title: str = "Results",
    run_record: Mapping[str, object] | None = None,
) -> str:
    """The full results report as Markdown.

    `run_record` is written out as a table: what was run, with which model
    and settings, on which datasets and from which commit.
    """
    n_datasets = len({r.score.dataset_id for r in results})
    sections = [
        f"# {title}",
        "",
        f"Generated by `python -m causal_agent.eval.run` from {n_datasets} datasets. "
        "Every number below comes from that run.",
        "",
    ]
    if any(r.score.approach == "no_tools_no_correlations" for r in results):
        sections += [ABLATION_NOTE, ""]
    if run_record:
        sections += [
            "## Run record",
            "",
            _markdown_table(
                ["Item", "Value"],
                [[str(name), f"`{value}`"] for name, value in run_record.items()],
            ),
            "",
        ]
    sections += [
        "An estimate is counted as correct when it is within "
        f"{tolerance_key(tolerance_sd)} standard deviations of the outcome of the "
        "true ATE. Errors marked (SD) are in those units.",
        "",
        "## Accuracy by approach",
        "",
        _accuracy_table(results, tolerances_sd),
        "",
        f"## Correct estimates by scenario (within {tolerance_key(tolerance_sd)} SD)",
        "",
        _scenario_table(results, tolerance_sd),
        "",
    ]
    subtypes = _subtype_table(results, tolerance_sd)
    if subtypes:
        sections += [
            f"## Correct estimates by design (within {tolerance_key(tolerance_sd)} SD)",
            "",
            "The design is a finer description of each dataset than its scenario.",
            "",
            subtypes,
            "",
        ]
    sections += [
        "## Validity, kept separate from accuracy",
        "",
        "A method is valid when its adjustment set passes the backdoor criterion, "
        "overlap is adequate, and its model assumption is not known to fail. For "
        "the doubly robust estimator the model-fit assumption cannot be read from "
        "the ground truth, so valid means valid as far as the truth can show. "
        "The no-tools LLM's method is self-reported.",
        "",
        _validity_table(results, tolerance_sd),
        "",
        "## Estimator behind each answer",
        "",
        _estimator_table(results),
    ]
    refutation = _refutation_table(results, tolerance_sd)
    if refutation:
        sections += [
            "",
            "## Refutation outcomes",
            "",
            "An estimate is flagged when at least one refutation check failed.",
            "",
            refutation,
        ]
    disagreement = _disagreement_table(results, tolerance_sd)
    if disagreement:
        sections += [
            "",
            "## Estimator-disagreement check",
            "",
            "Run by the evaluation on each final method. It compares the method's "
            "estimate with one from the other model family on the same adjustment "
            "set. Disagreement means the two estimators cannot both be right; it "
            "is not proof that the reported estimate is biased. Fractions are out "
            "of the inaccurate or the accurate estimates checked. Inconclusive "
            "results are counted on their own. For a no-tools approach the check "
            "runs the method the model named, not the number it reported.",
            "",
            disagreement,
        ]
    sections += [
        "",
        "## Model calls, tool calls and errors",
        "",
        _totals_table(results),
    ]
    agent = _agent_table(results)
    if agent:
        sections += ["", "## Agent behaviour", "", agent]
    sections += ["", "## Every dataset", "", _dataset_table(results, tolerance_sd), ""]
    return "\n".join(sections)


def write_scores_csv(results: Sequence[Result], path: Path) -> None:
    """One row per approach and dataset, with every recorded field."""
    tolerance_keys = sorted({k for r in results for k in r.score.within_tolerance})
    header = [
        "approach",
        "dataset_id",
        "scenario",
        "true_ate",
        "outcome_sd",
        "outcome",
        "answered",
        "estimate",
        "error",
        "abs_error",
        "abs_error_in_sd",
        *(f"within_{key}_sd" for key in tolerance_keys),
        "estimator",
        "adjustment_set",
        "adjustment_valid",
        "outcome_model_correct",
        "propensity_model_correct",
        "model_assumption_met",
        "overlap_adequate",
        "estimator_valid",
        "steps",
        "n_tool_calls",
        "answer_supported",
        "agent_refutations",
        "eval_refutations",
        "n_tool_errors",
        "input_tokens",
        "output_tokens",
        "agent_disagreement",
        "eval_disagreement",
        "eval_disagreement_reason",
        "design",
        "linear_regression_bias_sd",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        for result in results:
            score, validity = result.score, result.score.validity
            writer.writerow(
                [
                    score.approach,
                    score.dataset_id,
                    score.scenario,
                    score.true_ate,
                    score.outcome_sd,
                    result.outcome,
                    score.answered,
                    score.estimate,
                    score.error,
                    score.abs_error,
                    score.abs_error_in_sd,
                    *(score.within_tolerance[key] for key in tolerance_keys),
                    score.estimator,
                    " ".join(score.adjustment_set),
                    *(
                        getattr(validity, name) if validity else None
                        for name in (
                            "adjustment_valid",
                            "outcome_model_correct",
                            "propensity_model_correct",
                            "model_assumption_met",
                            "overlap_adequate",
                            "estimator_valid",
                        )
                    ),
                    result.steps,
                    result.n_tool_calls,
                    result.answer_supported,
                    _refutation_text(result.agent_refutations),
                    _refutation_text(result.eval_refutations),
                    result.n_tool_errors,
                    result.input_tokens,
                    result.output_tokens,
                    " ".join(
                        f"{estimator}={verdict}"
                        for estimator, verdict in result.agent_disagreement
                    ),
                    result.eval_disagreement,
                    result.eval_disagreement_reason,
                    result.subtype,
                    result.linear_bias_sd,
                ]
            )


def _refutation_text(refutations: dict[str, bool] | None) -> str:
    if not refutations:
        return ""
    return " ".join(
        f"{name}={'pass' if passed else 'fail'}"
        for name, passed in sorted(refutations.items())
    )


def plot_errors(results: Sequence[Result], tolerance_sd: float, path: Path) -> None:
    """Dot plot: each dataset's absolute error, by approach, one panel per scenario."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    approaches = _approaches(results)
    scenarios = _scenarios(results)
    figure, axes = plt.subplots(
        1,
        len(scenarios),
        # 2.9 inches per panel with three approaches; wider with more.
        figsize=((0.8 * len(approaches) + 0.5) * len(scenarios) + 0.8, 4.4),
        sharey=True,
        facecolor=SURFACE,
    )
    axes = [axes] if len(scenarios) == 1 else list(axes)

    for axis, scenario in zip(axes, scenarios, strict=True):
        axis.set_facecolor(SURFACE)
        unanswered = []
        for position, approach in enumerate(approaches):
            group = _of(results, approach, scenario)
            errors = sorted(
                max(r.score.abs_error_in_sd, PLOT_FLOOR_SD)
                for r in group
                if r.score.answered
            )
            missing = sum(not r.score.answered for r in group)
            if missing:
                unanswered.append(f"{APPROACH_LABELS[approach]}: {missing} unanswered")
            # Deterministic sideways spread so equal errors do not overlap.
            offsets = [
                (index - (len(errors) - 1) / 2) * 0.07 for index in range(len(errors))
            ]
            color, marker = SERIES_STYLE[approach]
            axis.scatter(
                [position + offset for offset in offsets],
                errors,
                s=46,
                marker=marker,
                color=color,
                edgecolors=SURFACE,
                linewidths=0.9,
                zorder=3,
                label=APPROACH_LABELS[approach],
            )
        axis.axhline(
            tolerance_sd, color=INK_SECONDARY, linewidth=1, linestyle=(0, (4, 3))
        )
        axis.set_yscale("log")
        axis.set_xlim(-0.6, len(approaches) - 0.4)
        axis.set_xticks(range(len(approaches)))
        axis.set_xticklabels(
            [PLOT_LABELS[a] for a in approaches],
            fontsize=8.5,
            color=INK_SECONDARY,
        )
        axis.set_title(SCENARIO_LABELS[scenario], fontsize=10.5, color=INK, pad=8)
        axis.grid(axis="y", which="major", color=GRIDLINE, linewidth=0.8)
        axis.grid(axis="x", visible=False)
        axis.tick_params(axis="both", length=0, labelsize=8.5, labelcolor=INK_MUTED)
        axis.tick_params(axis="x", labelcolor=INK_SECONDARY)
        axis.minorticks_off()
        for side in ("top", "right", "left"):
            axis.spines[side].set_visible(False)
        axis.spines["bottom"].set_color(AXIS)
        if unanswered:
            axis.text(
                0.5,
                0.98,
                "\n".join(unanswered),
                transform=axis.transAxes,
                ha="center",
                va="top",
                fontsize=7.5,
                color=INK_SECONDARY,
            )

    axes[0].set_ylabel(
        "Absolute error, in SDs of the outcome (log scale)",
        fontsize=9,
        color=INK_SECONDARY,
    )
    axes[-1].text(
        1.0,
        tolerance_sd,
        f" tolerance\n {tolerance_key(tolerance_sd)} SD",
        transform=axes[-1].get_yaxis_transform(),
        ha="left",
        va="center",
        fontsize=8,
        color=INK_SECONDARY,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(approaches),
        frameon=False,
        fontsize=9,
        labelcolor=INK_SECONDARY,
        bbox_to_anchor=(0.5, 0.0),
    )
    figure.suptitle(
        "Estimation error by approach and scenario: one mark per dataset",
        fontsize=12,
        color=INK,
        x=0.02,
        ha="left",
    )
    figure.text(
        0.02,
        0.905,
        f"Marks below the dashed line are within tolerance. "
        f"Errors under {PLOT_FLOOR_SD} SD are drawn at {PLOT_FLOOR_SD}.",
        fontsize=8.5,
        color=INK_SECONDARY,
    )
    figure.subplots_adjust(left=0.08, right=0.93, top=0.82, bottom=0.2, wspace=0.08)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(figure)
