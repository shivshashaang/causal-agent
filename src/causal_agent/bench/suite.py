"""The 30 benchmark dataset specifications (see ADR 001).

| Scenario             | Count | What varies                                   |
|----------------------|-------|-----------------------------------------------|
| randomized           | 6     | lambda_Y in {0, 1} x sigma                    |
| confounded_linear    | 9     | a x b, sigma rotated                          |
| confounded_nonlinear | 9     | (lambda_T, lambda_Y) x a, sigma rotated       |
| post_outcome_trap    | 6     | gamma x sigma, linear base                    |

This module is private: specs contain the true ATE.
"""

from typing import Any

import numpy as np

from causal_agent.bench.spec import DatasetSpec, Scenario

ASSIGNMENT_LEVELS = (0.5, 1.0, 1.5)
CONFOUNDER_OUTCOME_LEVELS = (0.5, 1.0, 2.0)
NOISE_LEVELS = (0.5, 1.0, 2.0)
TRAP_LEVELS = (0.5, 2.0)
NONLINEAR_SIDES = ((0.0, 1.0), (1.0, 0.0), (1.0, 1.0))  # (lambda_T, lambda_Y)
ATE_CYCLE = (0.0, 0.5, 1.0, 2.0, -1.0)

# Strength used for a knob that a scenario does not vary.
BASE_STRENGTH = 1.0

DEFAULT_MASTER_SEED = 0


def _rotated_noise(row: int, column: int) -> float:
    """Latin-square rotation: each noise level once per row and per column."""
    return NOISE_LEVELS[(row + column) % len(NOISE_LEVELS)]


def _scenario_knobs() -> dict[Scenario, list[dict[str, Any]]]:
    randomized = [
        {
            "assignment_strength": 0.0,
            "confounder_outcome_strength": BASE_STRENGTH,
            "outcome_noise": noise,
            "outcome_nonlinearity": outcome_nonlinearity,
        }
        for outcome_nonlinearity in (0.0, 1.0)
        for noise in NOISE_LEVELS
    ]
    linear = [
        {
            "assignment_strength": assignment,
            "confounder_outcome_strength": confounding,
            "outcome_noise": _rotated_noise(row, column),
        }
        for row, assignment in enumerate(ASSIGNMENT_LEVELS)
        for column, confounding in enumerate(CONFOUNDER_OUTCOME_LEVELS)
    ]
    nonlinear = [
        {
            "assignment_strength": assignment,
            "confounder_outcome_strength": BASE_STRENGTH,
            "outcome_noise": _rotated_noise(row, column),
            "treatment_nonlinearity": treatment_nonlinearity,
            "outcome_nonlinearity": outcome_nonlinearity,
        }
        for row, (treatment_nonlinearity, outcome_nonlinearity) in enumerate(
            NONLINEAR_SIDES
        )
        for column, assignment in enumerate(ASSIGNMENT_LEVELS)
    ]
    trap = [
        {
            "assignment_strength": BASE_STRENGTH,
            "confounder_outcome_strength": BASE_STRENGTH,
            "outcome_noise": noise,
            "trap_strength": trap_strength,
        }
        for trap_strength in TRAP_LEVELS
        for noise in NOISE_LEVELS
    ]
    return {
        Scenario.RANDOMIZED: randomized,
        Scenario.CONFOUNDED_LINEAR: linear,
        Scenario.CONFOUNDED_NONLINEAR: nonlinear,
        Scenario.POST_OUTCOME_TRAP: trap,
    }


def benchmark_specs(master_seed: int = DEFAULT_MASTER_SEED) -> list[DatasetSpec]:
    """Return the 30 specs, sorted by dataset ID.

    IDs are assigned by a seeded shuffle so that an ID says nothing about the
    scenario. Each dataset gets its own seed derived from `master_seed`.
    """
    rows: list[dict[str, Any]] = []
    for scenario, knob_sets in _scenario_knobs().items():
        for position, knobs in enumerate(knob_sets):
            true_ate = ATE_CYCLE[position % len(ATE_CYCLE)]
            rows.append({"scenario": scenario, "true_ate": true_ate, **knobs})

    id_stream, seed_stream = np.random.SeedSequence(master_seed).spawn(2)
    id_numbers = np.random.default_rng(id_stream).permutation(len(rows)) + 1
    seeds = seed_stream.generate_state(len(rows))

    specs = [
        DatasetSpec(dataset_id=f"ds_{id_number:03d}", seed=int(seed), **row)
        for row, id_number, seed in zip(rows, id_numbers, seeds, strict=True)
    ]
    return sorted(specs, key=lambda spec: spec.dataset_id)
