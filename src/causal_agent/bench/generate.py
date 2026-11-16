"""Turn a spec into a dataset: observed data, public metadata, private truth.

`generate` is pure. `write_benchmark` puts public and private files in
separate directories:

    <root>/public/ds_001.csv    observed columns only
    <root>/public/ds_001.json   PublicDataset
    <root>/truth/ds_001.json    GroundTruth

Potential outcomes and true propensities never leave this module.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from causal_agent.bench.equations import simulate
from causal_agent.bench.graph import OUTCOME, TRAP, TREATMENT, covariate_names
from causal_agent.bench.public import ColumnInfo, PublicDataset, Timing, public_paths
from causal_agent.bench.spec import DatasetSpec
from causal_agent.bench.truth import GroundTruth, build_truth, truth_path

PUBLIC_TREATMENT = "t"
PUBLIC_OUTCOME = "y"
PUBLIC_COVARIATES = ("x1", "x2", "x3", "x4")

# Separates the column-naming stream from the data streams of the same seed.
_NAMING_STREAM = 7_001


@dataclass(frozen=True)
class GeneratedDataset:
    frame: pd.DataFrame
    public: PublicDataset
    truth: GroundTruth


def _public_names(spec: DatasetSpec) -> dict[str, str]:
    """Assign neutral names x1..x4 to the covariates in a seeded random order."""
    internal = covariate_names(spec)
    rng = np.random.default_rng([spec.seed, _NAMING_STREAM])
    order = rng.permutation(len(internal))
    names = {
        internal[position]: public
        for position, public in zip(order, PUBLIC_COVARIATES, strict=True)
    }
    return {**names, TREATMENT: PUBLIC_TREATMENT, OUTCOME: PUBLIC_OUTCOME}


def generate(spec: DatasetSpec) -> GeneratedDataset:
    """Generate one dataset. Same spec, same result."""
    sample = simulate(spec)
    names = _public_names(spec)
    internal_of = {public: internal for internal, public in names.items()}

    values = {name: getattr(sample, name) for name in covariate_names(spec)}
    values[TREATMENT] = sample.t
    values[OUTCOME] = sample.y

    column_order = (*PUBLIC_COVARIATES, PUBLIC_TREATMENT, PUBLIC_OUTCOME)
    frame = pd.DataFrame(
        {public: values[internal_of[public]] for public in column_order}
    )

    columns = [
        ColumnInfo(
            name=public,
            kind="continuous",
            timing=(
                Timing.POST_OUTCOME
                if internal_of[public] == TRAP
                else Timing.PRE_TREATMENT
            ),
        )
        for public in PUBLIC_COVARIATES
    ]
    columns.append(
        ColumnInfo(name=PUBLIC_TREATMENT, kind="binary", timing=Timing.TREATMENT)
    )
    columns.append(
        ColumnInfo(name=PUBLIC_OUTCOME, kind="continuous", timing=Timing.OUTCOME)
    )

    public = PublicDataset(
        dataset_id=spec.dataset_id,
        question=(
            f"What is the average treatment effect of '{PUBLIC_TREATMENT}' "
            f"on '{PUBLIC_OUTCOME}'?"
        ),
        treatment=PUBLIC_TREATMENT,
        outcome=PUBLIC_OUTCOME,
        columns=tuple(columns),
        n_rows=spec.n_rows,
    )
    truth = build_truth(spec, sample, names)
    return GeneratedDataset(frame=frame, public=public, truth=truth)


def write_dataset(dataset: GeneratedDataset, root: Path) -> None:
    """Write one dataset's public and private files under `root`."""
    dataset_id = dataset.public.dataset_id
    csv_path, metadata_path = public_paths(dataset_id, root)
    private_path = truth_path(dataset_id, root)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    private_path.parent.mkdir(parents=True, exist_ok=True)

    dataset.frame.to_csv(csv_path, index=False, lineterminator="\n")
    metadata_path.write_text(
        dataset.public.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    private_path.write_text(
        dataset.truth.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_benchmark(specs: Iterable[DatasetSpec], root: Path) -> None:
    """Generate and write every dataset in `specs` under `root`."""
    for spec in specs:
        write_dataset(generate(spec), Path(root))
