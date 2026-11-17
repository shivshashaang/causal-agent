"""The benchmarks are frozen: these tests fail if a change alters a dataset.

The records in `tests/bench/frozen/` were written once from the generator and
are never regenerated to make a test pass. The v1 record was written before
any v2 work touched the generator, from the code that produced the published
v1 results.
"""

import json
from pathlib import Path

import pytest

from causal_agent.bench.fingerprint import dataset_fingerprint, spec_digest
from causal_agent.bench.generate import generate
from causal_agent.bench.suite import benchmark_specs

FROZEN = Path(__file__).parent / "frozen"

# Sums of 5,000 doubles. Last-bit differences between machines move them by a
# relative amount many orders of magnitude below this.
RELATIVE_TOLERANCE = 1e-9
ABSOLUTE_TOLERANCE = 1e-6


def _record(name: str) -> dict:
    return json.loads((FROZEN / name).read_text(encoding="utf-8"))


def _assert_matches(fingerprint: dict, frozen: dict) -> None:
    for key in ("columns", "n_rows", "treatment_sha256", "metadata_sha256"):
        assert fingerprint[key] == frozen[key], key
    assert set(fingerprint["continuous"]) == set(frozen["continuous"])
    for column, sums in frozen["continuous"].items():
        assert fingerprint["continuous"][column] == pytest.approx(
            sums, rel=RELATIVE_TOLERANCE, abs=ABSOLUTE_TOLERANCE
        ), column


def test_fingerprint_and_digest_detect_a_change(make_spec) -> None:
    """Guard the guard: a different dataset must not match."""
    spec = make_spec()
    other_seed = spec.model_copy(update={"seed": spec.seed + 1})
    other_effect = spec.model_copy(update={"true_ate": spec.true_ate + 0.01})
    base = dataset_fingerprint(generate(spec))

    _assert_matches(dataset_fingerprint(generate(spec)), base)
    with pytest.raises(AssertionError):
        _assert_matches(dataset_fingerprint(generate(other_seed)), base)
    with pytest.raises(AssertionError):
        _assert_matches(dataset_fingerprint(generate(other_effect)), base)
    assert spec_digest([spec]) != spec_digest([other_seed])
    assert spec_digest([spec]) != spec_digest([other_effect])


V1_SPECS = benchmark_specs()
V1_RECORD = _record("v1.json")


def test_v1_spec_digest_is_unchanged() -> None:
    assert spec_digest(V1_SPECS) == V1_RECORD["spec_digest"]


def test_v1_record_covers_exactly_the_thirty_datasets() -> None:
    assert sorted(V1_RECORD["datasets"]) == [spec.dataset_id for spec in V1_SPECS]


@pytest.mark.parametrize("spec", V1_SPECS, ids=[spec.dataset_id for spec in V1_SPECS])
def test_v1_dataset_is_unchanged(spec) -> None:
    _assert_matches(
        dataset_fingerprint(generate(spec)), V1_RECORD["datasets"][spec.dataset_id]
    )
