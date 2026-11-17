# 005 — Estimator-disagreement check in `refute_estimate`

- Status: Accepted. The owner approved the check, the paired standard error
  and the 0.10 SD materiality bound on 2026-11-16 and 2026-11-17. The items
  under "Choices made during implementation" were made by Claude and have not
  been reviewed by the owner.
- Date: 2026-11-17
- Scope: a fourth check inside the existing `refute_estimate` tool, how its
  uncertainty is computed, its three verdicts, and what they may be taken to
  mean.

## Context

The three original refutation checks (placebo treatment, random common cause,
data subset) all keep the adjustment set and the model fixed and re-run the
same estimator. A biased estimate is as stable under them as an unbiased one:

- A shuffled treatment is unrelated to everything, so any estimator returns
  about zero whether or not the original was confounded.
- An independent noise column cannot move an estimate beyond sampling noise.
- Every 80% subset carries the same bias.

Measured during development on Benchmark v1 data: they flagged 0 of 23
inaccurate unadjusted estimates and 0 of 4 inaccurate regressions that
adjusted for the trap column. In the v1 no-tools ablation they flagged 0 of
12 inaccurate answers.

The owner asked:

> Review why existing refuters missed deliberately biased estimates.
>
> Propose one targeted diagnostic for estimator misspecification, potentially
> comparing linear and flexible estimates.
>
> Separate detected problems, false alarms, and inconclusive results. Do not
> describe estimator disagreement as proof of bias.

## Options

**What to check**

- A. Compare the estimate with one from the other model family on the same
  adjustment set (linear regression against the cross-fitted doubly robust
  estimator). Targets the effect estimate itself. Noisier than B.
- B. A specification test on the regression: add squares and interactions of
  the covariates and test whether they matter. Cheaper and more sensitive to
  a curved outcome. It flags a curved outcome even when the effect estimate
  is unbiased, and it says nothing about a weighting estimator.

**Uncertainty of the difference between two estimates on the same rows**

- A. Treat the two estimates as independent: `sqrt(se_1^2 + se_2^2)`.
- B. A paired standard error from each row's contribution to each estimate.
- C. A bootstrap that refits both estimators on resampled rows.
- D. Report the difference descriptively, with no significance claim.

## Decision

**Check: option A, inside `refute_estimate`.** The owner: "I approve adding
an estimator-disagreement diagnostic within the existing refute_estimate
tool." The tool count stays at three.

**Uncertainty: option B.** The owner ruled out option A: "do not use an
invalid independence assumption when estimating uncertainty between
estimators fitted on the same data", and approved "the paired standard-error
approach for estimator disagreement".

How it works:

    comparison = doubly_robust, or linear_regression when the original
                 estimator is doubly_robust; same adjustment set
    difference = original estimate - comparison estimate
    paired SE  = sd(contributions of original - contributions of comparison)
                 / sqrt(n)
    interval   = difference +/- 2 * paired SE
    bound      = 0.10 * standard deviation of the outcome

    disagreement   the whole interval lies outside [-bound, +bound]
    agreement      the whole interval lies inside  [-bound, +bound]
    inconclusive   the interval straddles the bound; or the adjustment set is
                   empty; or more than 5% of rows had a clipped propensity

Each estimator returns one contribution per row: the estimate plus that row's
influence on it. The contributions of an estimator average to its estimate,
and the tool refuses to compute a paired standard error if they do not.

What the verdicts mean:

- **Disagreement** means the two estimators cannot both be right. It does not
  say which one is wrong and is not proof that the original estimate is
  biased.
- **Agreement** is not evidence of an unbiased estimate. Both estimators use
  the same adjustment set, so a wrong adjustment set biases both.
- **Inconclusive** is reported as its own category and is never counted as a
  detection or as a pass.

The agent can call the check. The owner: "Make the new diagnostic available
to the agent for v2 and document that its toolset differs from v1." The
Benchmark v1 results were produced without it (commit `67ee865`); the original
three checks and their published results are unchanged.

## Why

The owner's stated reasons and constraints, in the owner's words:

- "Do not describe estimator disagreement as proof of bias."
- "do not use an invalid independence assumption when estimating uncertainty
  between estimators fitted on the same data."
- "Treat the paired SE as approximate because fitted nuisance models
  introduce additional uncertainty."
- "Separate detected problems, false alarms, and inconclusive results."
- "Preserve the original refutation results."

Approved without a stated reason (Claude recommended each; the owner
approved):

- Comparing estimates (option A) over a specification test (option B).
- The materiality bound of 0.10 outcome standard deviations. The owner called
  it "the exploratory 0.10 SD materiality bound".

Evidence shown to the owner before approval, from a simulation on v1-family
data (150 datasets of 5,000 rows per setting):

| Setting | Actual spread of (linear − doubly robust) | Paired SE | Independence SE |
|---|---|---|---|
| Confounded, linear | 0.0203 | 0.0206 | 0.0497 |
| v1 nonlinear, both sides, a = 1.5 | 0.0391 | 0.0324 | 0.0560 |

The paired standard error matched reality in the first setting and was about
17% too small in the second. The independence formula was 2.4 times too large
in the first. In the second setting linear regression is biased by about 0.02
SD, well inside the tolerance, yet 37% of datasets differed by more than two
paired standard errors. That is why the verdict uses a materiality bound and
not significance alone.

## Choices made during implementation

Made by Claude. None was chosen by the owner.

- **A 2-standard-error interval.** A conventional width. Not calibrated.
- **The poor-overlap rule.** Inconclusive when more than 5% of rows had an
  estimated propensity clipped at 0.01 or 0.99. This reuses the threshold at
  which the estimators already warn.
- **Contributions for propensity weighting allow for the estimated
  propensities.** The part of each row's influence explained by the logistic
  model's score is removed. The estimator's own reported standard error is
  unchanged, so v1 numbers do not move.
- **Linear regression's contributions are the robust (sandwich) form.** Its
  reported standard error is still the classical one.
- **A separate output type.** The check returns a verdict and no `passed`
  field, so the three pass-or-fail checks and the code that reads them are
  untouched.
- **The evaluation runs the check on every final method**, cached on disk,
  and reports it in its own table.

## Tradeoffs

- **It cannot detect a wrong adjustment set.** Both estimators share the set.
  A missed confounder or an adjusted collider passes as agreement.
- **Two wrong models can agree.** If both miss the same structure, the check
  is silent.
- **The paired standard error understates uncertainty** when overlap is weak,
  because it treats the fitted models as fixed. Some verdicts that should be
  inconclusive will come out as disagreement or agreement.
- **The bound equals the benchmark's scoring tolerance.** A tool the agent
  can call uses the same 0.10 SD as the evaluation. That is convenient and it
  ties the tool to this benchmark.
- **The agent's toolset differs between v1 and v2.** v1 and v2 agent results
  are not a like-for-like comparison of the same agent.
- **It is exploratory.** The thresholds were fixed before Benchmark v2 was
  run and were not tuned afterwards, but they were not calibrated either.
