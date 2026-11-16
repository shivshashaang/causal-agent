"""Public view of a benchmark dataset: everything the agent and baselines see.

This is the only `causal_agent.bench` module that agent and tool code may
import. It must not import any private module (spec, equations, graph, truth,
suite, generate), and it reads files only from the `public/` directory.
"""

import re
from enum import StrEnum
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict

PUBLIC_DIRECTORY = "public"

_DATASET_ID_PATTERN = re.compile(r"ds_\d{3}")


class Timing(StrEnum):
    """When a column was measured, as a data dictionary would state it."""

    PRE_TREATMENT = "pre_treatment"
    TREATMENT = "treatment"
    OUTCOME = "outcome"
    POST_OUTCOME = "post_outcome"


class ColumnInfo(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    kind: Literal["binary", "continuous"]
    timing: Timing


class PublicDataset(BaseModel):
    """Metadata published next to the CSV. Contains no ground truth."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_id: str
    question: str
    treatment: str
    outcome: str
    columns: tuple[ColumnInfo, ...]
    n_rows: int

    @property
    def covariates(self) -> tuple[ColumnInfo, ...]:
        """Every column except the treatment and the outcome."""
        return tuple(
            column
            for column in self.columns
            if column.name not in (self.treatment, self.outcome)
        )


def validate_dataset_id(dataset_id: str) -> str:
    """Reject anything that is not a plain dataset ID, such as a path."""
    if not _DATASET_ID_PATTERN.fullmatch(dataset_id):
        raise ValueError(f"invalid dataset id: {dataset_id!r}")
    return dataset_id


def public_paths(dataset_id: str, root: Path) -> tuple[Path, Path]:
    """Paths of the public CSV and the public metadata file."""
    directory = Path(root) / PUBLIC_DIRECTORY
    stem = validate_dataset_id(dataset_id)
    return directory / f"{stem}.csv", directory / f"{stem}.json"


def load_public(dataset_id: str, root: Path) -> tuple[pd.DataFrame, PublicDataset]:
    """Load the observed data and its public metadata."""
    csv_path, metadata_path = public_paths(dataset_id, root)
    metadata = PublicDataset.model_validate_json(
        metadata_path.read_text(encoding="utf-8")
    )
    frame = pd.read_csv(csv_path)
    expected = [column.name for column in metadata.columns]
    if list(frame.columns) != expected:
        raise ValueError(
            f"{dataset_id}: CSV columns {list(frame.columns)} do not match "
            f"the metadata columns {expected}"
        )
    return frame, metadata
