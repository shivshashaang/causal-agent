"""Tests for the recorded population bias of linear regression.

No test here requires linear regression to fail on any dataset. The bias is
asserted only where theory fixes it at zero: when the outcome really is linear
in the covariates, or when treatment is randomized.
"""

import pytest

from causal_agent.bench.design import linear_regression_bias
from causal_agent.bench.spec import NonlinearFamily, Scenario
from causal_agent.bench.suite import benchmark_specs, benchmark_v2_specs

ALL_SPECS = [*benchmark_specs(), *benchmark_v2_specs()]
UNBIASED_BY_THEORY = [
    spec
    for spec in ALL_SPECS
    if spec.outcome_nonlinearity == 0 or spec.scenario is Scenario.RANDOMIZED
]


def test_bias_ignores_seed_rows_and_trap_strength(make_spec) -> None:
    base = linear_regression_bias(make_spec(Scenario.POST_OUTCOME_TRAP))
    changed = linear_regression_bias(
        make_spec(Scenario.POST_OUTCOME_TRAP, seed=99, n_rows=50, trap_strength=2.0)
    )
    assert base == changed


def test_bias_is_reported_in_outcome_sd_units(make_spec) -> None:
    result = linear_regression_bias(make_spec(Scenario.CONFOUNDED_NONLINEAR))
    assert result.outcome_sd > 0
    assert result.standard_error > 0
    assert result.bias_in_sd == pytest.approx(result.bias / result.outcome_sd)


def test_bias_does_not_depend_on_the_true_effect(make_spec) -> None:
    """A constant effect shifts the estimate and the truth by the same amount."""
    knobs = {"nonlinear_family": NonlinearFamily.SHARED}
    small = linear_regression_bias(
        make_spec(Scenario.CONFOUNDED_NONLINEAR, true_ate=0.0, **knobs)
    )
    large = linear_regression_bias(
        make_spec(Scenario.CONFOUNDED_NONLINEAR, true_ate=1.7, **knobs)
    )
    assert small.bias == pytest.approx(large.bias, abs=1e-9)


@pytest.mark.statistical
@pytest.mark.parametrize(
    "spec", UNBIASED_BY_THEORY, ids=[spec.dataset_id for spec in UNBIASED_BY_THEORY]
)
def test_bias_is_zero_where_theory_says_so(spec) -> None:
    result = linear_regression_bias(spec)
    assert abs(result.bias) <= 4 * result.standard_error
