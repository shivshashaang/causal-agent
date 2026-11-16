"""Tests for the public dataset view and its loader."""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from causal_agent.bench.public import (
    ColumnInfo,
    PublicDataset,
    Timing,
    load_public,
    public_paths,
    validate_dataset_id,
)

PRIVATE_MODULES = ("spec", "equations", "graph", "truth", "suite", "generate")


def _metadata(**overrides) -> PublicDataset:
    fields = {
        "dataset_id": "ds_001",
        "question": "What is the average treatment effect of t on y?",
        "treatment": "t",
        "outcome": "y",
        "columns": (
            ColumnInfo(name="x1", kind="continuous", timing=Timing.PRE_TREATMENT),
            ColumnInfo(name="x2", kind="continuous", timing=Timing.POST_OUTCOME),
            ColumnInfo(name="t", kind="binary", timing=Timing.TREATMENT),
            ColumnInfo(name="y", kind="continuous", timing=Timing.OUTCOME),
        ),
        "n_rows": 3,
        **overrides,
    }
    return PublicDataset(**fields)


def _write(root: Path, metadata: PublicDataset, frame: pd.DataFrame) -> None:
    csv_path, metadata_path = public_paths(metadata.dataset_id, root)
    csv_path.parent.mkdir(parents=True)
    frame.to_csv(csv_path, index=False, lineterminator="\n")
    metadata_path.write_text(metadata.model_dump_json(), encoding="utf-8")


FRAME = pd.DataFrame(
    {
        "x1": [0.1, -1.25, 0.3333333333333333],
        "x2": [1.0, 2.0, 3.0],
        "t": [0, 1, 1],
        "y": [0.5, 1.5, -2.123456789012345],
    }
)


def test_public_metadata_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        _metadata(true_ate=1.0)
    with pytest.raises(ValidationError):
        ColumnInfo(
            name="x1", kind="continuous", timing=Timing.PRE_TREATMENT, role="confounder"
        )


def test_public_metadata_is_frozen() -> None:
    with pytest.raises(ValidationError):
        _metadata().question = "changed"


def test_covariates_exclude_treatment_and_outcome() -> None:
    assert [column.name for column in _metadata().covariates] == ["x1", "x2"]


def test_load_public_round_trips_data_and_metadata(tmp_path) -> None:
    metadata = _metadata()
    _write(tmp_path, metadata, FRAME)
    frame, loaded = load_public("ds_001", tmp_path)
    assert loaded == metadata
    pd.testing.assert_frame_equal(frame, FRAME, check_exact=True)


def test_load_public_rejects_mismatched_columns(tmp_path) -> None:
    _write(tmp_path, _metadata(), FRAME.rename(columns={"x2": "other"}))
    with pytest.raises(ValueError, match="do not match"):
        load_public("ds_001", tmp_path)


@pytest.mark.parametrize(
    "bad_id",
    ["../truth/ds_001", "..\\truth\\ds_001", "ds_001.json", "ds_1", "", "truth/ds_001"],
)
def test_dataset_id_cannot_be_a_path(tmp_path, bad_id) -> None:
    with pytest.raises(ValueError, match="invalid dataset id"):
        validate_dataset_id(bad_id)
    with pytest.raises(ValueError, match="invalid dataset id"):
        load_public(bad_id, tmp_path)


def test_public_paths_stay_inside_the_public_directory(tmp_path) -> None:
    csv_path, metadata_path = public_paths("ds_007", tmp_path)
    assert csv_path == tmp_path / "public" / "ds_007.csv"
    assert metadata_path == tmp_path / "public" / "ds_007.json"


def test_importing_the_public_module_loads_no_private_module() -> None:
    """Run in a fresh interpreter so other tests' imports cannot interfere."""
    code = (
        "import sys, causal_agent.bench.public\n"
        f"private = {PRIVATE_MODULES!r}\n"
        "loaded = [m for m in private if f'causal_agent.bench.{m}' in sys.modules]\n"
        "print(','.join(loaded))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == ""
