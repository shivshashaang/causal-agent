"""Tests for the estimator-disagreement check of the refute_estimate tool.

The examples are datasets made for these tests. None of them is a Benchmark v1
or v2 dataset.
"""

import numpy as np
import pytest

from causal_agent.bench.generate import generate, write_dataset
from causal_agent.bench.spec import DatasetSpec, NonlinearFamily, Scenario
from causal_agent.tools.aipw import EffectEstimate
from causal_agent.tools.estimators import EstimatorName, estimate_ate
from causal_agent.tools.refute_estimate import (
    DISAGREEMENT_LIMITATIONS,
    INTERVAL_STANDARD_ERRORS,
    MATERIALITY_IN_OUTCOME_SD,
    STABILITY_REFUTERS,
    EstimatorDisagreementOutput,
    RefuteEstimateInput,
    RefuteEstimateOutput,
    RefuterName,
    checked_contributions,
    comparison_estimator,
    disagreement_verdict,
    paired_standard_error,
    refute_estimate,
)

CHECK = RefuterName.ESTIMATOR_DISAGREEMENT

ACCURATE = "ds_951"  # linear confounding: linear regression is right
BIASED = "ds_952"  # one curve drives treatment and outcome: it is not
SMALL = "ds_953"  # too few rows to tell
NO_OVERLAP = "ds_954"  # treatment almost determined by the confounders

SPECS = {
    ACCURATE: dict(scenario=Scenario.CONFOUNDED_LINEAR, assignment_strength=1.0),
    BIASED: dict(
        scenario=Scenario.CONFOUNDED_NONLINEAR,
        assignment_strength=1.5,
        confounder_outcome_strength=2.0,
        outcome_noise=0.5,
        treatment_nonlinearity=1.0,
        outcome_nonlinearity=1.0,
        nonlinear_family=NonlinearFamily.SHARED,
    ),
    SMALL: dict(
        scenario=Scenario.CONFOUNDED_LINEAR, assignment_strength=1.0, n_rows=300
    ),
    NO_OVERLAP: dict(scenario=Scenario.CONFOUNDED_LINEAR, assignment_strength=6.0),
}


@pytest.fixture(scope="module")
def examples(tmp_path_factory):
    """The example datasets on disk, with their ground truth kept in memory."""
    root = tmp_path_factory.mktemp("disagreement_examples")
    datasets = {}
    for number, (dataset_id, knobs) in enumerate(SPECS.items()):
        fields = {
            "dataset_id": dataset_id,
            "seed": 9500 + number,
            "n_rows": 5000,
            "true_ate": 0.8,
            "confounder_outcome_strength": 1.0,
            "outcome_noise": 1.0,
            **knobs,
        }
        dataset = generate(DatasetSpec(**fields))
        write_dataset(dataset, root)
        datasets[dataset_id] = dataset
    return root, datasets


def _pre_treatment(dataset) -> tuple[str, ...]:
    return tuple(column.name for column in dataset.public.covariates)


def _check(examples, dataset_id, estimator, adjustment_set=None):
    root, datasets = examples
    if adjustment_set is None:
        adjustment_set = _pre_treatment(datasets[dataset_id])
    request = RefuteEstimateInput(
        dataset_id=dataset_id,
        estimator=estimator,
        adjustment_set=adjustment_set,
        refuter=CHECK,
    )
    return refute_estimate(request, root)


def _abs_error_in_sd(examples, dataset_id, estimate) -> float:
    truth = examples[1][dataset_id].truth
    return abs(estimate - truth.true_ate) / truth.diagnostics.outcome_sd


# --- The verdict rule -------------------------------------------------------


@pytest.mark.parametrize(
    ("difference", "standard_error", "expected"),
    [
        (0.50, 0.05, "disagreement"),  # interval [0.40, 0.60], bound 0.10
        (-0.50, 0.05, "disagreement"),  # the sign does not matter
        (0.02, 0.01, "agreement"),  # interval [0.00, 0.04]
        (0.00, 0.049, "agreement"),  # interval just inside the bound
        (0.09, 0.02, "inconclusive"),  # small difference, interval crosses
        (0.15, 0.05, "inconclusive"),  # large difference, interval crosses
        (0.00, 0.30, "inconclusive"),  # nothing can be said: far too noisy
    ],
)
def test_verdict_rule(difference, standard_error, expected) -> None:
    label, reason = disagreement_verdict(difference, standard_error, bound=0.10)
    assert label == expected
    assert "materiality bound" in reason


def test_thresholds_are_the_documented_ones() -> None:
    assert MATERIALITY_IN_OUTCOME_SD == 0.10
    assert INTERVAL_STANDARD_ERRORS == 2.0


def test_comparison_is_always_the_other_model_family() -> None:
    assert comparison_estimator(EstimatorName.DOUBLY_ROBUST) is (
        EstimatorName.LINEAR_REGRESSION
    )
    for estimator in EstimatorName:
        if estimator is not EstimatorName.DOUBLY_ROBUST:
            assert comparison_estimator(estimator) is EstimatorName.DOUBLY_ROBUST


def test_the_original_three_checks_are_still_listed_separately() -> None:
    assert CHECK not in STABILITY_REFUTERS
    assert set(RefuterName) == {*STABILITY_REFUTERS, CHECK}


# --- Row contributions ------------------------------------------------------


@pytest.mark.parametrize("dataset_id", [ACCURATE, BIASED, SMALL, NO_OVERLAP])
@pytest.mark.parametrize("estimator", list(EstimatorName))
def test_contributions_average_to_the_estimate(examples, dataset_id, estimator) -> None:
    dataset = examples[1][dataset_id]
    adjustment = (
        ()
        if estimator is EstimatorName.DIFFERENCE_IN_MEANS
        else _pre_treatment(dataset)
    )
    result = estimate_ate(estimator, dataset.frame, "t", "y", adjustment)
    assert result.contributions.shape == (len(dataset.frame),)
    assert result.contributions.mean() == pytest.approx(result.estimate, abs=1e-9)
    assert checked_contributions(result) is result.contributions


def test_contributions_that_do_not_average_to_the_estimate_are_rejected() -> None:
    wrong = EffectEstimate(
        estimate=1.0,
        standard_error=0.1,
        n_rows=4,
        n_treated=2,
        contributions=np.array([1.0, 1.0, 1.0, 1.5]),
    )
    with pytest.raises(ValueError, match="do not average to the estimate"):
        checked_contributions(wrong)
    missing = EffectEstimate(estimate=1.0, standard_error=0.1, n_rows=4, n_treated=2)
    with pytest.raises(ValueError, match="one contribution per row"):
        checked_contributions(missing)


def test_paired_standard_error_accounts_for_the_correlation(examples) -> None:
    """On the same rows two sound estimators move together, so their
    difference is much less uncertain than if they were independent."""
    dataset = examples[1][ACCURATE]
    adjustment = _pre_treatment(dataset)
    linear = estimate_ate(
        EstimatorName.LINEAR_REGRESSION, dataset.frame, "t", "y", adjustment
    )
    robust = estimate_ate(
        EstimatorName.DOUBLY_ROBUST, dataset.frame, "t", "y", adjustment
    )
    paired = paired_standard_error(linear, robust)
    independent = float(np.hypot(linear.standard_error, robust.standard_error))
    by_hand = (linear.contributions - robust.contributions).std(ddof=1) / np.sqrt(5000)
    assert paired == pytest.approx(by_hand)
    assert paired == pytest.approx(paired_standard_error(robust, linear))
    assert 0 < paired < 0.75 * independent
    assert paired_standard_error(linear, linear) == 0.0


# --- Known examples ---------------------------------------------------------


def test_accurate_example_gives_agreement(examples) -> None:
    output = _check(examples, ACCURATE, EstimatorName.LINEAR_REGRESSION)
    assert isinstance(output, EstimatorDisagreementOutput)
    assert _abs_error_in_sd(examples, ACCURATE, output.original_estimate) < 0.10
    assert output.verdict == "agreement"
    assert output.comparison_estimator is EstimatorName.DOUBLY_ROBUST
    assert abs(output.difference_in_outcome_sd) < MATERIALITY_IN_OUTCOME_SD
    assert -output.materiality_bound < output.interval_low < output.interval_high
    assert output.interval_high < output.materiality_bound
    assert "does not show that the estimate is unbiased" in output.interpretation


def test_biased_example_gives_disagreement(examples) -> None:
    output = _check(examples, BIASED, EstimatorName.LINEAR_REGRESSION)
    # The example is biased by construction: the regression is far from the truth.
    assert _abs_error_in_sd(examples, BIASED, output.original_estimate) > 0.20
    assert output.verdict == "disagreement"
    assert abs(output.difference_in_outcome_sd) > MATERIALITY_IN_OUTCOME_SD
    assert min(abs(output.interval_low), abs(output.interval_high)) > (
        output.materiality_bound
    )
    assert "not proof that the original estimate is biased" in output.interpretation


def test_disagreement_does_not_say_which_estimator_is_wrong(examples) -> None:
    """Asked about the estimator that is close to the truth here, the check
    still reports disagreement: it compares, it does not judge."""
    output = _check(examples, BIASED, EstimatorName.DOUBLY_ROBUST)
    linear = _check(examples, BIASED, EstimatorName.LINEAR_REGRESSION)
    assert output.comparison_estimator is EstimatorName.LINEAR_REGRESSION
    assert output.verdict == "disagreement"
    assert output.difference == pytest.approx(-linear.difference)
    assert output.paired_standard_error == pytest.approx(linear.paired_standard_error)
    assert _abs_error_in_sd(examples, BIASED, output.original_estimate) < (
        _abs_error_in_sd(examples, BIASED, linear.original_estimate)
    )


def test_too_few_rows_is_inconclusive(examples) -> None:
    output = _check(examples, SMALL, EstimatorName.LINEAR_REGRESSION)
    assert output.verdict == "inconclusive"
    assert "straddles" in output.reason
    assert output.interval_low < output.materiality_bound < output.interval_high or (
        output.interval_low < -output.materiality_bound < output.interval_high
    )


def test_empty_adjustment_set_is_inconclusive_without_a_comparison(examples) -> None:
    for estimator in (
        EstimatorName.DIFFERENCE_IN_MEANS,
        EstimatorName.LINEAR_REGRESSION,
        EstimatorName.DOUBLY_ROBUST,
    ):
        output = _check(examples, ACCURATE, estimator, adjustment_set=())
        assert output.verdict == "inconclusive"
        assert "adjustment set is empty" in output.reason
        assert output.comparison_estimator is None
        assert output.difference is None
        assert output.paired_standard_error is None
        assert "No comparison was made" in output.interpretation


def test_poor_overlap_is_inconclusive(examples) -> None:
    output = _check(examples, NO_OVERLAP, EstimatorName.LINEAR_REGRESSION)
    assert output.verdict == "inconclusive"
    assert "overlap" in output.reason
    # The comparison is still reported, so the reader can see the numbers.
    assert output.difference is not None
    assert output.paired_standard_error is not None


# --- Output -----------------------------------------------------------------


def test_output_states_its_limits_and_has_no_pass_or_fail(examples) -> None:
    output = _check(examples, ACCURATE, EstimatorName.LINEAR_REGRESSION)
    assert output.limitations == DISAGREEMENT_LIMITATIONS
    text = " ".join(output.limitations)
    assert "not proof that the original estimate is biased" in text
    assert "not evidence that the estimate is unbiased" in text
    assert "approximate" in text
    assert "exploratory" in text
    assert "passed" not in output.model_dump()
    assert output.materiality_bound == pytest.approx(
        MATERIALITY_IN_OUTCOME_SD * output.outcome_sd
    )
    assert output.interval_high - output.interval_low == pytest.approx(
        2 * INTERVAL_STANDARD_ERRORS * output.paired_standard_error
    )


def test_check_is_deterministic(examples) -> None:
    first = _check(examples, BIASED, EstimatorName.PROPENSITY_WEIGHTING)
    second = _check(examples, BIASED, EstimatorName.PROPENSITY_WEIGHTING)
    assert first == second
    assert first.comparison_estimator is EstimatorName.DOUBLY_ROBUST


def test_original_checks_keep_their_pass_or_fail_output(examples) -> None:
    root, datasets = examples
    for refuter in STABILITY_REFUTERS:
        output = refute_estimate(
            RefuteEstimateInput(
                dataset_id=ACCURATE,
                estimator=EstimatorName.LINEAR_REGRESSION,
                adjustment_set=_pre_treatment(datasets[ACCURATE]),
                refuter=refuter,
            ),
            root,
        )
        assert isinstance(output, RefuteEstimateOutput)
        assert isinstance(output.passed, bool)


def test_check_needs_no_ground_truth_on_disk(examples, tmp_path) -> None:
    import shutil

    root, datasets = examples
    shutil.copytree(root / "public", tmp_path / "public")
    request = RefuteEstimateInput(
        dataset_id=ACCURATE,
        estimator=EstimatorName.LINEAR_REGRESSION,
        adjustment_set=_pre_treatment(datasets[ACCURATE]),
        refuter=CHECK,
    )
    assert refute_estimate(request, tmp_path) == refute_estimate(request, root)
