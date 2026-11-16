"""Tests for the fixed-pipeline baseline."""

import pytest

from causal_agent.agent.fixed_pipeline import (
    ESTIMATOR,
    PipelineResult,
    run_fixed_pipeline,
)
from causal_agent.bench.graph import Role
from causal_agent.bench.spec import Scenario
from causal_agent.tools.aipw import aipw_estimate

ALL_SCENARIOS = list(Scenario)


@pytest.fixture(scope="module")
def results(small_benchmark) -> dict[Scenario, PipelineResult]:
    """Run against a root that has no truth directory at all."""
    return {
        scenario: run_fixed_pipeline(
            small_benchmark.dataset_id(scenario), small_benchmark.public_only_root
        )
        for scenario in ALL_SCENARIOS
    }


def test_pipeline_runs_without_any_ground_truth_on_disk(
    small_benchmark, results
) -> None:
    assert not (small_benchmark.public_only_root / "truth").exists()
    assert set(results) == set(ALL_SCENARIOS)


@pytest.mark.parametrize("scenario", ALL_SCENARIOS)
def test_pipeline_adjusts_for_every_pre_treatment_covariate(
    small_benchmark, results, scenario
) -> None:
    metadata = small_benchmark.datasets[scenario].public
    expected = tuple(
        column.name
        for column in metadata.covariates
        if column.timing.value == "pre_treatment"
    )
    assert results[scenario].adjustment_set == expected
    assert results[scenario].estimator == ESTIMATOR == "doubly_robust"


def test_pipeline_leaves_out_the_post_outcome_column(small_benchmark, results) -> None:
    truth = small_benchmark.datasets[Scenario.POST_OUTCOME_TRAP].truth
    collider = next(
        name
        for name, role in truth.column_roles.items()
        if role is Role.POST_OUTCOME_COLLIDER
    )
    adjustment_set = results[Scenario.POST_OUTCOME_TRAP].adjustment_set
    assert collider not in adjustment_set
    assert len(adjustment_set) == 3


@pytest.mark.parametrize(
    "scenario", [s for s in ALL_SCENARIOS if s.name != "POST_OUTCOME_TRAP"]
)
def test_pipeline_uses_all_four_covariates_when_none_is_post_outcome(
    results, scenario
) -> None:
    assert results[scenario].adjustment_set == ("x1", "x2", "x3", "x4")


@pytest.mark.parametrize("scenario", ALL_SCENARIOS)
def test_pipeline_adjustment_set_is_valid_in_this_benchmark(
    small_benchmark, results, scenario
) -> None:
    """The test reads the truth; the pipeline did not."""
    truth = small_benchmark.datasets[scenario].truth
    assert truth.is_valid_adjustment_set(results[scenario].adjustment_set)


def test_pipeline_matches_a_direct_estimator_call(small_benchmark, results) -> None:
    dataset = small_benchmark.datasets[Scenario.CONFOUNDED_LINEAR]
    result = results[Scenario.CONFOUNDED_LINEAR]
    direct = aipw_estimate(dataset.frame, "t", "y", result.adjustment_set)
    assert result.estimate == direct.estimate
    assert result.standard_error == direct.standard_error
    assert (result.ci_low, result.ci_high) == direct.confidence_interval
    assert result.diagnostics == direct.diagnostics


def test_pipeline_is_deterministic(small_benchmark, results) -> None:
    again = run_fixed_pipeline(
        small_benchmark.dataset_id(Scenario.RANDOMIZED), small_benchmark.root
    )
    assert again == results[Scenario.RANDOMIZED]


def test_result_is_a_closed_schema(results) -> None:
    assert set(PipelineResult.model_fields) == {
        "dataset_id",
        "estimate",
        "standard_error",
        "ci_low",
        "ci_high",
        "estimator",
        "adjustment_set",
        "diagnostics",
        "warnings",
    }


def test_unknown_or_path_like_dataset_ids_are_rejected(small_benchmark) -> None:
    with pytest.raises(ValueError, match="invalid dataset id"):
        run_fixed_pipeline("../truth/ds_902", small_benchmark.root)
    with pytest.raises(FileNotFoundError):
        run_fixed_pipeline("ds_999", small_benchmark.root)


@pytest.mark.statistical
@pytest.mark.parametrize("scenario", ALL_SCENARIOS)
def test_estimate_is_within_four_standard_errors_on_the_small_datasets(
    results, scenario
) -> None:
    """Fixed datasets where the pipeline's assumptions hold; true ATE is 1."""
    result = results[scenario]
    assert abs(result.estimate - 1.0) <= 4 * result.standard_error
