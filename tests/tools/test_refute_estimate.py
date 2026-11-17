"""Tests for the refute_estimate tool."""

import pytest
from pydantic import ValidationError

from causal_agent.bench.graph import Role
from causal_agent.bench.spec import Scenario
from causal_agent.tools.estimate_effect import EstimateEffectInput, estimate_effect
from causal_agent.tools.estimators import EstimatorName
from causal_agent.tools.refute_estimate import (
    FAIL_THRESHOLD_IN_STANDARD_ERRORS,
    LIMITATIONS,
    N_REPETITIONS,
    RefuteEstimateInput,
    RefuteEstimateOutput,
    RefuterName,
    refute_estimate,
    verdict,
)

ALL_REFUTERS = list(RefuterName)


def _confounders(small_benchmark, scenario) -> tuple[str, ...]:
    roles = small_benchmark.datasets[scenario].truth.column_roles
    return tuple(sorted(n for n, role in roles.items() if role is Role.CONFOUNDER))


def _run(small_benchmark, scenario, estimator, adjustment_set, refuter):
    request = RefuteEstimateInput(
        dataset_id=small_benchmark.dataset_id(scenario),
        estimator=estimator,
        adjustment_set=adjustment_set,
        refuter=refuter,
    )
    return refute_estimate(request, small_benchmark.public_only_root)


@pytest.fixture(scope="module")
def regression_outputs(small_benchmark) -> dict[RefuterName, RefuteEstimateOutput]:
    """Linear regression with the true confounders on linear data: a sound estimate."""
    scenario = Scenario.CONFOUNDED_LINEAR
    return {
        refuter: _run(
            small_benchmark,
            scenario,
            EstimatorName.LINEAR_REGRESSION,
            _confounders(small_benchmark, scenario),
            refuter,
        )
        for refuter in ALL_REFUTERS
    }


# --- Input schema ----------------------------------------------------------


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        (
            {"estimator": "difference_in_means", "refuter": "random_common_cause"},
            "does not apply to difference_in_means",
        ),
        (
            {
                "estimator": "difference_in_means",
                "refuter": "data_subset",
                "adjustment_set": ["x1"],
            },
            "always uses the empty adjustment set",
        ),
        (
            {
                "estimator": "linear_regression",
                "refuter": "data_subset",
                "adjustment_set": ["x1", "x1"],
            },
            "more than once",
        ),
        ({"estimator": "linear_regression", "refuter": "bootstrap"}, "refuter"),
        ({"estimator": "linear_regression"}, "refuter"),
    ],
)
def test_invalid_inputs_are_rejected(fields, message) -> None:
    with pytest.raises(ValidationError, match=message):
        RefuteEstimateInput(dataset_id="ds_902", **fields)


# --- The decision rule -----------------------------------------------------


def test_verdict_fails_only_above_the_threshold() -> None:
    assert verdict(0.0)
    assert verdict(FAIL_THRESHOLD_IN_STANDARD_ERRORS)
    assert not verdict(FAIL_THRESHOLD_IN_STANDARD_ERRORS + 0.01)


@pytest.mark.parametrize("refuter", ALL_REFUTERS)
def test_shift_and_verdict_are_consistent_with_the_reported_numbers(
    regression_outputs, refuter
) -> None:
    output = regression_outputs[refuter]
    shift = abs(output.refuted_mean - output.expected) / output.original_standard_error
    assert output.shift_in_standard_errors == pytest.approx(shift)
    assert output.passed is verdict(shift)
    assert output.threshold_in_standard_errors == FAIL_THRESHOLD_IN_STANDARD_ERRORS
    assert output.n_repetitions == N_REPETITIONS
    assert output.refuted_sd > 0


# --- Behaviour on a sound estimate -----------------------------------------


def test_placebo_compares_against_zero(regression_outputs) -> None:
    output = regression_outputs[RefuterName.PLACEBO_TREATMENT]
    assert output.expected == 0.0
    assert abs(output.refuted_mean) < abs(output.original_estimate) / 5


@pytest.mark.parametrize(
    "refuter", [RefuterName.RANDOM_COMMON_CAUSE, RefuterName.DATA_SUBSET]
)
def test_other_checks_compare_against_the_original_estimate(
    regression_outputs, refuter
) -> None:
    output = regression_outputs[refuter]
    assert output.expected == output.original_estimate


@pytest.mark.parametrize("refuter", ALL_REFUTERS)
def test_a_sound_estimate_passes_every_check(regression_outputs, refuter) -> None:
    output = regression_outputs[refuter]
    assert output.passed
    assert output.interpretation.startswith("Passed")
    assert "does not show that it is unbiased" in output.interpretation


@pytest.mark.parametrize("refuter", ALL_REFUTERS)
def test_output_states_what_a_pass_does_not_mean(regression_outputs, refuter) -> None:
    output = regression_outputs[refuter]
    assert output.limitations == LIMITATIONS
    text = " ".join(output.limitations)
    assert "not evidence that the estimate is unbiased" in text
    assert "cannot detect a wrong adjustment set" in text


def test_original_estimate_matches_estimate_effect(
    small_benchmark, regression_outputs
) -> None:
    scenario = Scenario.CONFOUNDED_LINEAR
    estimate = estimate_effect(
        EstimateEffectInput(
            dataset_id=small_benchmark.dataset_id(scenario),
            estimator=EstimatorName.LINEAR_REGRESSION,
            adjustment_set=_confounders(small_benchmark, scenario),
        ),
        small_benchmark.public_only_root,
    )
    for output in regression_outputs.values():
        assert output.original_estimate == estimate.estimate
        assert output.original_standard_error == estimate.standard_error


def test_noise_column_is_not_reported_as_part_of_the_adjustment_set(
    small_benchmark, regression_outputs
) -> None:
    output = regression_outputs[RefuterName.RANDOM_COMMON_CAUSE]
    assert output.adjustment_set == _confounders(
        small_benchmark, Scenario.CONFOUNDED_LINEAR
    )


def test_refutation_is_deterministic(small_benchmark, regression_outputs) -> None:
    scenario = Scenario.CONFOUNDED_LINEAR
    again = _run(
        small_benchmark,
        scenario,
        EstimatorName.LINEAR_REGRESSION,
        _confounders(small_benchmark, scenario),
        RefuterName.DATA_SUBSET,
    )
    assert again == regression_outputs[RefuterName.DATA_SUBSET]


# --- Other estimators ------------------------------------------------------


@pytest.mark.parametrize(
    "refuter", [RefuterName.PLACEBO_TREATMENT, RefuterName.DATA_SUBSET]
)
def test_difference_in_means_can_be_refuted(small_benchmark, refuter) -> None:
    output = _run(
        small_benchmark,
        Scenario.RANDOMIZED,
        EstimatorName.DIFFERENCE_IN_MEANS,
        (),
        refuter,
    )
    assert output.adjustment_set == ()
    assert output.passed


def test_doubly_robust_estimate_can_be_refuted(small_benchmark) -> None:
    scenario = Scenario.CONFOUNDED_LINEAR
    output = _run(
        small_benchmark,
        scenario,
        EstimatorName.DOUBLY_ROBUST,
        _confounders(small_benchmark, scenario),
        RefuterName.PLACEBO_TREATMENT,
    )
    assert output.expected == 0.0
    assert abs(output.refuted_mean) < abs(output.original_estimate) / 5


# --- A known blind spot, stated as a test ----------------------------------


@pytest.mark.parametrize(
    "refuter", [RefuterName.PLACEBO_TREATMENT, RefuterName.DATA_SUBSET]
)
def test_checks_report_numbers_for_a_confounded_unadjusted_estimate(
    small_benchmark, refuter
) -> None:
    """The unadjusted estimate on confounded data is biased by construction.

    This test does not assert a pass or a fail. It only confirms the tool
    runs and reports consistent numbers; how often the checks flag biased
    estimates is measured on the benchmark, not assumed here.
    """
    output = _run(
        small_benchmark,
        Scenario.CONFOUNDED_LINEAR,
        EstimatorName.DIFFERENCE_IN_MEANS,
        (),
        refuter,
    )
    assert output.passed is verdict(output.shift_in_standard_errors)


# --- Guards ----------------------------------------------------------------


def test_adjustment_set_must_be_covariate_columns(small_benchmark) -> None:
    with pytest.raises(ValueError, match="may only contain covariate columns"):
        _run(
            small_benchmark,
            Scenario.CONFOUNDED_LINEAR,
            EstimatorName.LINEAR_REGRESSION,
            ("y",),
            RefuterName.DATA_SUBSET,
        )


def test_path_like_dataset_id_is_rejected(small_benchmark) -> None:
    request = RefuteEstimateInput(
        dataset_id="../truth/ds_902",
        estimator="linear_regression",
        refuter="data_subset",
    )
    with pytest.raises(ValueError, match="invalid dataset id"):
        refute_estimate(request, small_benchmark.root)


def test_output_round_trips_through_json(regression_outputs) -> None:
    output = regression_outputs[RefuterName.PLACEBO_TREATMENT]
    assert RefuteEstimateOutput.model_validate_json(output.model_dump_json()) == output
