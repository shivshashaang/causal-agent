"""The four estimators behind the `estimate_effect` tool.

Each estimates the average treatment effect of a binary treatment. Each rests
on different assumptions, listed in `ASSUMPTIONS`. No estimator can check
whether its adjustment set is valid; that is the caller's responsibility.

- `difference_in_means`: no adjustment at all. Always uses the empty set.
- `linear_regression`: ordinary least squares of the outcome on the treatment
  and the adjustment columns; the treatment coefficient is the estimate.
- `propensity_weighting`: inverse probability weighting with a logistic
  propensity model, using normalized (Hajek) weights.
- `doubly_robust`: cross-fitted AIPW with gradient boosting (see `aipw.py`).
"""

from collections.abc import Sequence
from enum import StrEnum

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from causal_agent.tools.aipw import (
    CLIPPED_SHARE_WARNING,
    DEFAULT_CLIP,
    DEFAULT_SEED,
    EffectEstimate,
    aipw_estimate,
    prepare_arrays,
)


class EstimatorName(StrEnum):
    DIFFERENCE_IN_MEANS = "difference_in_means"
    LINEAR_REGRESSION = "linear_regression"
    PROPENSITY_WEIGHTING = "propensity_weighting"
    DOUBLY_ROBUST = "doubly_robust"


_SHARED = (
    "The adjustment set must be valid: it blocks every backdoor path and "
    "contains nothing affected by the treatment. The estimator cannot check this."
)

ASSUMPTIONS: dict[EstimatorName, tuple[str, ...]] = {
    EstimatorName.DIFFERENCE_IN_MEANS: (
        "Uses no adjustment, so it is valid only if treatment is unrelated to "
        "every cause of the outcome, as in a randomized experiment.",
    ),
    EstimatorName.LINEAR_REGRESSION: (
        _SHARED,
        "The outcome is linear in the adjustment columns and the treatment "
        "effect is the same for everyone. If the outcome depends on the "
        "columns nonlinearly, the estimate can be biased unless treatment is "
        "randomized.",
        "The standard error assumes constant error variance.",
    ),
    EstimatorName.PROPENSITY_WEIGHTING: (
        _SHARED,
        "The log-odds of treatment are linear in the adjustment columns. If "
        "treatment depends on the columns nonlinearly, the estimate can be "
        "biased.",
        "Overlap: every kind of unit has a real chance of both treatment arms. "
        "Estimated propensities are clipped to [0.01, 0.99], which bounds the "
        "weights but can add bias when overlap is poor.",
        "The standard error treats the estimated propensities as known.",
    ),
    EstimatorName.DOUBLY_ROBUST: (
        _SHARED,
        "Overlap: every kind of unit has a real chance of both treatment arms. "
        "Estimated propensities are clipped to [0.01, 0.99].",
        "The gradient-boosted outcome and propensity models must approximate "
        "the truth well enough. The estimate is consistent if either set of "
        "models is right; that is not a guarantee of unbiasedness or accuracy "
        "in a finite sample.",
        "The standard error treats the nuisance predictions as fixed.",
    ),
}


def difference_in_means(
    frame: pd.DataFrame, treatment: str, outcome: str
) -> EffectEstimate:
    """Mean outcome of the treated minus mean outcome of the controls."""
    t, y, _ = prepare_arrays(frame, treatment, outcome, [])
    treated, control = y[t == 1], y[t == 0]
    if len(treated) < 2 or len(control) < 2:
        raise ValueError("each treatment arm needs at least 2 rows")
    estimate = float(treated.mean() - control.mean())
    standard_error = float(
        np.sqrt(treated.var(ddof=1) / len(treated) + control.var(ddof=1) / len(control))
    )
    share = len(treated) / len(t)
    influence = t * (y - treated.mean()) / share - (1 - t) * (y - control.mean()) / (
        1 - share
    )
    return EffectEstimate(
        estimate=estimate,
        standard_error=standard_error,
        n_rows=len(t),
        n_treated=len(treated),
        diagnostics={
            "treated_mean": float(treated.mean()),
            "control_mean": float(control.mean()),
        },
        contributions=estimate + influence,
    )


def linear_regression(
    frame: pd.DataFrame, treatment: str, outcome: str, adjustment_set: Sequence[str]
) -> EffectEstimate:
    """OLS of the outcome on an intercept, the treatment and the adjustment set."""
    t, y, x = prepare_arrays(frame, treatment, outcome, adjustment_set)
    n_rows = len(t)
    design = np.column_stack([np.ones(n_rows), t, x])
    n_parameters = design.shape[1]
    if n_rows <= n_parameters:
        raise ValueError("not enough rows for the number of adjustment columns")
    if np.linalg.matrix_rank(design) < n_parameters:
        raise ValueError(
            "the treatment and adjustment columns are linearly dependent; "
            "remove constant or duplicated columns"
        )

    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    residuals = y - design @ coefficients
    residual_variance = float(residuals @ residuals) / (n_rows - n_parameters)
    covariance = residual_variance * np.linalg.inv(design.T @ design)
    total = float(((y - y.mean()) ** 2).sum())
    # Each row's influence on the treatment coefficient. The residuals are
    # orthogonal to every design column, so these sum to zero.
    treatment_row = np.linalg.inv(design.T @ design / n_rows)[1]
    influence = (design * residuals[:, None]) @ treatment_row
    estimate = float(coefficients[1])
    return EffectEstimate(
        estimate=estimate,
        standard_error=float(np.sqrt(covariance[1, 1])),
        n_rows=n_rows,
        n_treated=int(t.sum()),
        diagnostics={
            "r_squared": 1.0 - float(residuals @ residuals) / total,
            "n_parameters": float(n_parameters),
        },
        contributions=estimate + influence,
    )


def propensity_weighting(
    frame: pd.DataFrame,
    treatment: str,
    outcome: str,
    adjustment_set: Sequence[str],
    *,
    clip: float = DEFAULT_CLIP,
) -> EffectEstimate:
    """Normalized inverse probability weighting with a logistic propensity model."""
    if not 0 < clip < 0.5:
        raise ValueError("clip must be between 0 and 0.5")
    t, y, x = prepare_arrays(frame, treatment, outcome, adjustment_set)
    n_rows = len(t)
    n_treated = int(t.sum())
    if n_treated < 2 or n_rows - n_treated < 2:
        raise ValueError("each treatment arm needs at least 2 rows")

    if x.shape[1] == 0:
        raw = np.full(n_rows, n_treated / n_rows)
    else:
        # C=inf is scikit-learn's spelling of an unpenalized logistic regression.
        model = LogisticRegression(C=np.inf, max_iter=1000)
        raw = model.fit(x, t).predict_proba(x)[:, 1]
    e = np.clip(raw, clip, 1 - clip)
    n_clipped = int(np.sum((raw < clip) | (raw > 1 - clip)))

    weight_treated = t / e
    weight_control = (1 - t) / (1 - e)
    mean_treated = float(weight_treated @ y / weight_treated.sum())
    mean_control = float(weight_control @ y / weight_control.sum())
    estimate = mean_treated - mean_control

    influence = weight_treated * (y - mean_treated) / weight_treated.mean() - (
        weight_control * (y - mean_control) / weight_control.mean()
    )
    standard_error = float(influence.std(ddof=1) / np.sqrt(n_rows))

    # Row contributions for comparing this estimate with another one. Unlike
    # the standard error above, they allow for the propensities having been
    # estimated: the part of each row's influence that the logistic model's
    # score explains is removed. The result is centred so its mean is exactly
    # zero.
    score = (t - raw)[:, None] * np.column_stack([np.ones(n_rows), x])
    explained = score @ np.linalg.lstsq(score, influence, rcond=None)[0]
    adjusted = influence - explained
    adjusted = adjusted - adjusted.mean()

    clipped_share = n_clipped / n_rows
    warnings: list[str] = []
    if clipped_share > CLIPPED_SHARE_WARNING:
        warnings.append(
            f"{clipped_share:.1%} of rows had an estimated propensity outside "
            f"[{clip}, {1 - clip}] and were clipped; overlap may be poor and "
            "the estimate may be biased"
        )
    return EffectEstimate(
        estimate=estimate,
        standard_error=standard_error,
        n_rows=n_rows,
        n_treated=n_treated,
        diagnostics={
            "propensity_min": float(raw.min()),
            "propensity_max": float(raw.max()),
            "clipped_rows": float(n_clipped),
            "clipped_share": clipped_share,
            "largest_weight": float(np.max(np.where(t == 1, 1 / e, 1 / (1 - e)))),
        },
        warnings=tuple(warnings),
        contributions=estimate + adjusted,
    )


def estimate_ate(
    estimator: EstimatorName,
    frame: pd.DataFrame,
    treatment: str,
    outcome: str,
    adjustment_set: Sequence[str],
    *,
    seed: int = DEFAULT_SEED,
) -> EffectEstimate:
    """Run the named estimator."""
    if estimator is EstimatorName.DIFFERENCE_IN_MEANS:
        if len(adjustment_set) > 0:
            raise ValueError(
                "difference_in_means always uses the empty adjustment set; "
                "pass an empty adjustment_set or choose another estimator"
            )
        return difference_in_means(frame, treatment, outcome)
    if estimator is EstimatorName.LINEAR_REGRESSION:
        return linear_regression(frame, treatment, outcome, adjustment_set)
    if estimator is EstimatorName.PROPENSITY_WEIGHTING:
        return propensity_weighting(frame, treatment, outcome, adjustment_set)
    if estimator is EstimatorName.DOUBLY_ROBUST:
        return aipw_estimate(frame, treatment, outcome, adjustment_set, seed=seed)
    raise ValueError(f"unknown estimator: {estimator!r}")
