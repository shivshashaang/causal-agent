"""A deterministic text summary of one dataset's public data.

The agent and the no-tools baseline receive exactly this text, so the only
difference between them is access to tools. It is built from the public CSV
and public metadata alone.
"""

import pandas as pd

from causal_agent.bench.public import PublicDataset

TIMING_LEGEND = (
    "Timing labels: pre_treatment = measured before the treatment was assigned; "
    "treatment = the treatment itself; outcome = the outcome itself; "
    "post_outcome = measured after the outcome was observed."
)


def _table(rows: list[list[str]]) -> str:
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    return "\n".join(
        "  ".join(
            cell.ljust(width) for cell, width in zip(row, widths, strict=True)
        ).rstrip()
        for row in rows
    )


def _number(value: float) -> str:
    return f"{value:.3f}"


def describe_dataset(frame: pd.DataFrame, metadata: PublicDataset) -> str:
    """Summarize the columns, the two treatment groups and the correlations."""
    treatment, outcome = metadata.treatment, metadata.outcome
    names = [column.name for column in metadata.columns]

    column_rows = [["column", "kind", "timing", "mean", "sd", "min", "max"]]
    for column in metadata.columns:
        values = frame[column.name]
        column_rows.append(
            [
                column.name,
                column.kind,
                column.timing.value,
                _number(values.mean()),
                _number(values.std()),
                _number(values.min()),
                _number(values.max()),
            ]
        )

    treated = frame[frame[treatment] == 1]
    control = frame[frame[treatment] == 0]
    group_rows = [["column", "treated_mean", "control_mean", "difference"]]
    for name in names:
        if name == treatment:
            continue
        treated_mean, control_mean = treated[name].mean(), control[name].mean()
        group_rows.append(
            [
                name,
                _number(treated_mean),
                _number(control_mean),
                _number(treated_mean - control_mean),
            ]
        )

    correlations = frame[names].corr()
    correlation_rows = [["", *names]]
    for name in names:
        correlation_rows.append(
            [name, *(_number(correlations.loc[name, other]) for other in names)]
        )

    return "\n".join(
        [
            f"Dataset: {metadata.dataset_id}",
            f"Question: {metadata.question}",
            f"Rows: {metadata.n_rows}",
            f"Treatment column: {treatment} (binary, 1 = treated).",
            f"Outcome column: {outcome}.",
            "",
            "Columns",
            _table(column_rows),
            TIMING_LEGEND,
            "",
            f"Treatment groups: {len(treated)} treated rows, "
            f"{len(control)} control rows.",
            "Column means by treatment group",
            _table(group_rows),
            "",
            "Pearson correlations",
            _table(correlation_rows),
        ]
    )
