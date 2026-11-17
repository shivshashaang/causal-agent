"""Fixtures shared by the tool, baseline and agent tests."""

import shutil
from dataclasses import dataclass
from pathlib import Path

import pytest

from causal_agent.bench.generate import GeneratedDataset, generate, write_dataset
from causal_agent.bench.spec import DatasetSpec, Scenario

SMALL_ROWS = 2000


@dataclass(frozen=True)
class SmallBenchmark:
    """One small dataset per scenario, written to disk."""

    root: Path
    public_only_root: Path
    datasets: dict[Scenario, GeneratedDataset]

    def dataset_id(self, scenario: Scenario) -> str:
        return self.datasets[scenario].public.dataset_id


def _small_specs() -> list[DatasetSpec]:
    common = {
        "n_rows": SMALL_ROWS,
        "true_ate": 1.0,
        "confounder_outcome_strength": 1.0,
        "outcome_noise": 1.0,
    }
    return [
        DatasetSpec(
            dataset_id="ds_901",
            scenario=Scenario.RANDOMIZED,
            seed=901,
            assignment_strength=0.0,
            **common,
        ),
        DatasetSpec(
            dataset_id="ds_902",
            scenario=Scenario.CONFOUNDED_LINEAR,
            seed=902,
            assignment_strength=1.0,
            **common,
        ),
        DatasetSpec(
            dataset_id="ds_903",
            scenario=Scenario.CONFOUNDED_NONLINEAR,
            seed=903,
            assignment_strength=1.0,
            treatment_nonlinearity=1.0,
            outcome_nonlinearity=1.0,
            **common,
        ),
        DatasetSpec(
            dataset_id="ds_904",
            scenario=Scenario.POST_OUTCOME_TRAP,
            seed=904,
            assignment_strength=1.0,
            trap_strength=2.0,
            **common,
        ),
    ]


@pytest.fixture(scope="session")
def small_benchmark(tmp_path_factory) -> SmallBenchmark:
    """Four small datasets on disk, plus a copy with the truth directory removed.

    Tool and baseline code is run against `public_only_root` to prove it works
    without any ground truth on disk.
    """
    root = tmp_path_factory.mktemp("small_benchmark")
    datasets = {}
    for spec in _small_specs():
        dataset = generate(spec)
        write_dataset(dataset, root)
        datasets[spec.scenario] = dataset

    public_only_root = tmp_path_factory.mktemp("public_only")
    shutil.copytree(root / "public", public_only_root / "public")
    return SmallBenchmark(
        root=root, public_only_root=public_only_root, datasets=datasets
    )
