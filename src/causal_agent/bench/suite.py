"""The benchmark dataset specifications.

`benchmark_specs` returns the 30 datasets of Benchmark v1 (ADR 001), described
in the table below. `benchmark_v2_specs` returns the 20 datasets of Benchmark
v2 (ADR 004). The two share nothing but the knob levels: different IDs,
different seeds.

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

from causal_agent.bench.spec import DatasetSpec, NonlinearFamily, Scenario

ASSIGNMENT_LEVELS = (0.5, 1.0, 1.5)
CONFOUNDER_OUTCOME_LEVELS = (0.5, 1.0, 2.0)
NOISE_LEVELS = (0.5, 1.0, 2.0)
TRAP_LEVELS = (0.5, 2.0)
NONLINEAR_SIDES = ((0.0, 1.0), (1.0, 0.0), (1.0, 1.0))  # (lambda_T, lambda_Y)
ATE_CYCLE = (0.0, 0.5, 1.0, 2.0, -1.0)

# Strength used for a knob that a scenario does not vary.
BASE_STRENGTH = 1.0

DEFAULT_MASTER_SEED = 0

# Benchmark v2 (ADR 004). It reuses the v1 levels above and adds these.
V2_MASTER_SEED = 2
V2_FIRST_ID = 101
V2_ASSIGNMENT_ENDS = (0.5, 1.5)
V2_CONFOUNDER_OUTCOME_ENDS = (0.5, 2.0)
V2_ONE_SIDED = ((0.0, 1.0), (1.0, 0.0))  # (lambda_T, lambda_Y)
V2_TRAP_NOISE = (0.5, 2.0)
V2_ATE_RANGE = (-2.0, 2.0)
V2_ZERO_EFFECT_DATASETS = 2


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


def _v2_scenario_knobs() -> dict[Scenario, list[dict[str, Any]]]:
    shared = {"nonlinear_family": NonlinearFamily.SHARED}
    randomized = [
        {
            "assignment_strength": 0.0,
            "confounder_outcome_strength": BASE_STRENGTH,
            "outcome_noise": _rotated_noise(0, column),
            **extra,
        }
        for column, extra in enumerate(({}, {"outcome_nonlinearity": 1.0, **shared}))
    ]
    linear = [
        {
            "assignment_strength": assignment,
            "confounder_outcome_strength": confounding,
            "outcome_noise": _rotated_noise(row, column),
        }
        for row, assignment in enumerate(V2_ASSIGNMENT_ENDS)
        for column, confounding in enumerate(V2_CONFOUNDER_OUTCOME_ENDS)
    ]
    one_sided = [
        {
            "assignment_strength": assignment,
            "confounder_outcome_strength": BASE_STRENGTH,
            "outcome_noise": _rotated_noise(row, column),
            "treatment_nonlinearity": treatment_nonlinearity,
            "outcome_nonlinearity": outcome_nonlinearity,
            **shared,
        }
        for row, (treatment_nonlinearity, outcome_nonlinearity) in enumerate(
            V2_ONE_SIDED
        )
        for column, assignment in enumerate(V2_ASSIGNMENT_ENDS)
    ]
    both_sides = [
        {
            "assignment_strength": assignment,
            "confounder_outcome_strength": confounding,
            "outcome_noise": _rotated_noise(row, column),
            "treatment_nonlinearity": 1.0,
            "outcome_nonlinearity": 1.0,
            **shared,
        }
        for row, assignment in enumerate(ASSIGNMENT_LEVELS)
        for column, confounding in enumerate(V2_CONFOUNDER_OUTCOME_ENDS)
    ]
    trap = [
        {
            "assignment_strength": BASE_STRENGTH,
            "confounder_outcome_strength": BASE_STRENGTH,
            "outcome_noise": noise,
            "trap_strength": trap_strength,
        }
        for trap_strength in TRAP_LEVELS
        for noise in V2_TRAP_NOISE
    ]
    return {
        Scenario.RANDOMIZED: randomized,
        Scenario.CONFOUNDED_LINEAR: linear,
        Scenario.CONFOUNDED_NONLINEAR: one_sided + both_sides,
        Scenario.POST_OUTCOME_TRAP: trap,
    }


def benchmark_v2_specs(master_seed: int = V2_MASTER_SEED) -> list[DatasetSpec]:
    """Return the 20 Benchmark v2 specs, sorted by dataset ID (see ADR 004).

    | Scenario                         | Count | What varies                  |
    |----------------------------------|-------|------------------------------|
    | randomized                       | 2     | linear or curved outcome     |
    | confounded_linear                | 4     | a x b at the ends            |
    | confounded_nonlinear, one side   | 4     | which side x a at the ends   |
    | confounded_nonlinear, both sides | 6     | a x b at the ends            |
    | post_outcome_trap                | 4     | gamma x sigma, linear base   |

    Every nonlinear dataset uses the shared family. Outcome noise is rotated
    by position, as in v1.

    The true effects are drawn once, by a seeded generator, from a uniform
    distribution and rounded to two decimals. Two datasets, also picked by a
    seeded generator, have their effect set to exactly zero. Nothing about
    which dataset gets which effect is chosen by hand.
    """
    rows = [
        {"scenario": scenario, **knobs}
        for scenario, knob_sets in _v2_scenario_knobs().items()
        for knobs in knob_sets
    ]

    effect_stream, zero_stream, id_stream, seed_stream = np.random.SeedSequence(
        master_seed
    ).spawn(4)
    effects = np.round(
        np.random.default_rng(effect_stream).uniform(*V2_ATE_RANGE, size=len(rows)), 2
    )
    zero_positions = np.random.default_rng(zero_stream).choice(
        len(rows), size=V2_ZERO_EFFECT_DATASETS, replace=False
    )
    effects[zero_positions] = 0.0
    id_numbers = np.random.default_rng(id_stream).permutation(len(rows)) + V2_FIRST_ID
    seeds = seed_stream.generate_state(len(rows))

    specs = [
        DatasetSpec(
            dataset_id=f"ds_{id_number:03d}",
            seed=int(seed),
            true_ate=float(effect) + 0.0,  # turns -0.0 into 0.0
            **row,
        )
        for row, effect, id_number, seed in zip(
            rows, effects, id_numbers, seeds, strict=True
        )
    ]
    return sorted(specs, key=lambda spec: spec.dataset_id)


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
