"""Tests for dataset generation, writing, and public/private isolation."""

import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from causal_agent.bench.equations import simulate
from causal_agent.bench.generate import (
    PUBLIC_COVARIATES,
    GeneratedDataset,
    generate,
    write_benchmark,
)
from causal_agent.bench.graph import Role
from causal_agent.bench.public import PublicDataset, Timing, load_public
from causal_agent.bench.spec import DatasetSpec, Scenario
from causal_agent.bench.suite import benchmark_specs
from causal_agent.bench.truth import GroundTruth, load_truth

SPECS = benchmark_specs()
SPEC_IDS = [spec.dataset_id for spec in SPECS]
PUBLIC_COLUMNS = [*PUBLIC_COVARIATES, "t", "y"]

# Words that would reveal private information if they appeared in public
# metadata: private field names, role names, scenario names.
PRIVATE_FIELDS = (set(DatasetSpec.model_fields) | set(GroundTruth.model_fields)) - set(
    PublicDataset.model_fields
)
PRIVATE_WORDS = (
    PRIVATE_FIELDS
    | {role.value for role in Role}
    | {scenario.value for scenario in Scenario}
    | {"confound", "collider", "propensity", "randomized", "y0", "y1"}
)


@pytest.fixture(scope="module")
def datasets() -> dict[str, GeneratedDataset]:
    return {spec.dataset_id: generate(spec) for spec in SPECS}


@pytest.fixture(scope="module")
def benchmark_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("benchmark")
    write_benchmark(SPECS, root)
    return root


def _file_hashes(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


# --- Generation ------------------------------------------------------------


@pytest.mark.parametrize("scenario", list(Scenario))
def test_generate_is_deterministic(make_spec, scenario) -> None:
    first, second = generate(make_spec(scenario)), generate(make_spec(scenario))
    pd.testing.assert_frame_equal(first.frame, second.frame, check_exact=True)
    assert first.public == second.public
    assert first.truth == second.truth


def test_frame_holds_the_observed_variables_under_public_names(make_spec) -> None:
    spec = make_spec(Scenario.POST_OUTCOME_TRAP)
    dataset, sample = generate(spec), simulate(spec)
    np.testing.assert_array_equal(dataset.frame["t"], sample.t)
    np.testing.assert_array_equal(dataset.frame["y"], sample.y)
    for public, internal in dataset.truth.internal_names.items():
        if public in PUBLIC_COVARIATES:
            np.testing.assert_array_equal(
                dataset.frame[public], getattr(sample, internal), err_msg=public
            )


def test_trap_dataset_publishes_the_collider_instead_of_the_irrelevant_column(
    make_spec,
) -> None:
    truth = generate(make_spec(Scenario.POST_OUTCOME_TRAP)).truth
    assert sorted(truth.internal_names.values()) == ["c1", "c2", "k", "p", "t", "y"]


# --- Public schema is identical across scenarios ---------------------------


@pytest.mark.parametrize("dataset_id", SPEC_IDS)
def test_every_dataset_has_the_same_public_schema(datasets, dataset_id) -> None:
    dataset = datasets[dataset_id]
    assert list(dataset.frame.columns) == PUBLIC_COLUMNS
    assert [column.name for column in dataset.public.columns] == PUBLIC_COLUMNS
    assert dataset.public.treatment == "t"
    assert dataset.public.outcome == "y"
    assert dataset.public.n_rows == len(dataset.frame) == 5000
    assert set(dataset.frame["t"].unique()) == {0, 1}
    assert not dataset.frame.isna().any().any()


def test_question_text_is_the_same_for_every_dataset(datasets) -> None:
    assert len({dataset.public.question for dataset in datasets.values()}) == 1


@pytest.mark.parametrize("dataset_id", SPEC_IDS)
def test_timing_labels_mark_only_the_trap_column_as_post_outcome(
    datasets, dataset_id
) -> None:
    dataset = datasets[dataset_id]
    post_outcome = {
        column.name
        for column in dataset.public.covariates
        if column.timing is Timing.POST_OUTCOME
    }
    colliders = {
        name
        for name, role in dataset.truth.column_roles.items()
        if role is Role.POST_OUTCOME_COLLIDER
    }
    assert post_outcome == colliders
    is_trap = dataset.truth.spec.scenario is Scenario.POST_OUTCOME_TRAP
    assert len(post_outcome) == (1 if is_trap else 0)


def test_column_names_do_not_encode_roles(datasets) -> None:
    """Across the benchmark, each role appears under several public names."""
    names_by_role: dict[Role, Counter] = {role: Counter() for role in Role}
    for dataset in datasets.values():
        for name, role in dataset.truth.column_roles.items():
            names_by_role[role][name] += 1
    for role, names in names_by_role.items():
        assert len(names) >= 3, f"{role} always appears under {sorted(names)}"


# --- Public metadata contains no ground truth ------------------------------


@pytest.mark.parametrize("dataset_id", SPEC_IDS)
def test_public_metadata_contains_no_private_words(datasets, dataset_id) -> None:
    text = datasets[dataset_id].public.model_dump_json().lower()
    leaked = sorted(word for word in PRIVATE_WORDS if word.lower() in text)
    assert leaked == []


def test_private_word_list_covers_the_sensitive_fields() -> None:
    """Guard the guard: the list must include the fields that matter."""
    assert {"true_ate", "seed", "scenario", "edges", "column_roles"} <= PRIVATE_WORDS
    assert "dataset_id" not in PRIVATE_WORDS


def test_public_metadata_fields_are_an_explicit_whitelist() -> None:
    assert set(PublicDataset.model_fields) == {
        "dataset_id",
        "question",
        "treatment",
        "outcome",
        "columns",
        "n_rows",
    }


# --- Files on disk ---------------------------------------------------------


def test_public_and_truth_files_live_in_separate_directories(benchmark_root) -> None:
    public_files = sorted(p.name for p in (benchmark_root / "public").iterdir())
    truth_files = sorted(p.name for p in (benchmark_root / "truth").iterdir())
    assert public_files == sorted(
        f"{dataset_id}{suffix}"
        for dataset_id in SPEC_IDS
        for suffix in (".csv", ".json")
    )
    assert truth_files == [f"{dataset_id}.json" for dataset_id in SPEC_IDS]
    assert sorted(p.name for p in benchmark_root.iterdir()) == ["public", "truth"]


@pytest.mark.parametrize("dataset_id", SPEC_IDS)
def test_public_files_on_disk_contain_no_private_words(
    benchmark_root, dataset_id
) -> None:
    public_dir = benchmark_root / "public"
    metadata_text = (public_dir / f"{dataset_id}.json").read_text(encoding="utf-8")
    header = (
        (public_dir / f"{dataset_id}.csv").read_text(encoding="utf-8").split("\n")[0]
    )
    assert header == ",".join(PUBLIC_COLUMNS)
    leaked = sorted(w for w in PRIVATE_WORDS if w.lower() in metadata_text.lower())
    assert leaked == []


@pytest.mark.parametrize("dataset_id", SPEC_IDS)
def test_files_round_trip_exactly(benchmark_root, datasets, dataset_id) -> None:
    frame, public = load_public(dataset_id, benchmark_root)
    pd.testing.assert_frame_equal(frame, datasets[dataset_id].frame, check_exact=True)
    assert public == datasets[dataset_id].public
    assert load_truth(dataset_id, benchmark_root) == datasets[dataset_id].truth


def test_writing_twice_gives_identical_files(benchmark_root, tmp_path) -> None:
    write_benchmark(SPECS, tmp_path)
    assert _file_hashes(tmp_path) == _file_hashes(benchmark_root)


def test_public_loader_cannot_be_pointed_at_the_truth_directory(
    benchmark_root,
) -> None:
    with pytest.raises(ValueError, match="invalid dataset id"):
        load_public("../truth/ds_001", benchmark_root)


# --- Truth agrees with the spec --------------------------------------------


@pytest.mark.parametrize("spec", SPECS, ids=SPEC_IDS)
def test_truth_matches_its_spec(datasets, spec) -> None:
    truth = datasets[spec.dataset_id].truth
    assert truth.spec == spec
    assert truth.true_ate == spec.true_ate
    expected_sets = {
        Scenario.RANDOMIZED: 16,
        Scenario.CONFOUNDED_LINEAR: 4,
        Scenario.CONFOUNDED_NONLINEAR: 4,
        Scenario.POST_OUTCOME_TRAP: 2,
    }
    assert len(truth.valid_adjustment_sets) == expected_sets[spec.scenario]
