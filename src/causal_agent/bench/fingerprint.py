"""Fingerprints that pin a benchmark so a later change cannot alter it unnoticed.

Two kinds:

- `spec_digest`: a SHA-256 over every knob of every spec. Specs hold only
  exact values (seeds, strengths, the true ATE), so the digest is identical on
  every machine.
- `dataset_fingerprint`: a summary of one generated dataset. The treatment
  column and the public metadata are hashed exactly. Continuous columns are
  summarized by three weighted sums, because floating-point functions such as
  `exp` and `tanh` may differ in the last bit between machines, which would
  change a hash of the raw bytes without changing the data in any way that
  matters. Compare the sums with a tight relative tolerance.

This module is private: it reads specs and generated data.
"""

import hashlib
import json
from collections.abc import Iterable
from typing import Any

import numpy as np

from causal_agent.bench.generate import GeneratedDataset
from causal_agent.bench.spec import DatasetSpec


def _spec_record(spec: DatasetSpec) -> dict[str, Any]:
    return spec.model_dump(mode="json")


def spec_digest(specs: Iterable[DatasetSpec]) -> str:
    """SHA-256 over the specs, in the order given."""
    records = [_spec_record(spec) for spec in specs]
    text = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def dataset_fingerprint(dataset: GeneratedDataset) -> dict[str, Any]:
    """Summary of one dataset's public data and metadata."""
    frame, public = dataset.frame, dataset.public
    positions = np.arange(1, len(frame) + 1, dtype=np.float64)
    treatment = frame[public.treatment].to_numpy(dtype="<i8")

    continuous = {}
    for name in frame.columns:
        if name == public.treatment:
            continue
        values = frame[name].to_numpy(dtype=np.float64)
        continuous[name] = [
            float(values.sum()),
            float((values**2).sum()),
            float((values * positions).sum()),
        ]

    return {
        "columns": list(frame.columns),
        "n_rows": int(len(frame)),
        "treatment_sha256": hashlib.sha256(treatment.tobytes()).hexdigest(),
        "metadata_sha256": hashlib.sha256(
            public.model_dump_json().encode("utf-8")
        ).hexdigest(),
        "continuous": continuous,
    }
