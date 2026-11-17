# 002 — Estimators, fixed-pipeline baseline, and the three tools

- Status: Accepted in part. The owner chose the flexible estimator and the
  fixed-pipeline rule. The implementation choices listed under "Choices made
  during implementation" were made by Claude and reported at checkpoint 2;
  the owner replied "continue" without commenting on them.
- Date: 2026-11-16
- Scope: how effects are estimated, what the no-LLM baseline does, and what
  the agent's three tools do.

## Context

The agent may use exactly three typed tools: `discover_graph`,
`estimate_effect` and `refute_estimate`. It is compared with a fixed pipeline
that uses no LLM. ADR 001 fixed four estimator families: difference in means,
linear outcome regression, logistic propensity weighting or matching, and
flexible adjustment. Two things were left open after the generator was built:
what "flexible adjustment" is concretely, and what rule the fixed pipeline
follows.

## Options

**Flexible adjustment**

- A. Cross-fitted doubly robust estimator (AIPW): an outcome model and a
  propensity model, both gradient boosted, combined so the estimate is
  consistent if either is right.
- B. A gradient-boosted outcome model alone. Simpler, but more exposed to
  regularization bias.

**Fixed-pipeline rule**

- A. Adjust for every pre-treatment column with the flexible estimator.
- B. Adjust for every pre-treatment column with linear regression. A more
  traditional default, but its assumption fails on the datasets with a
  nonlinear outcome.

## Decision

**Flexible adjustment: option A.** `tools/aipw.py` implements cross-fitted
AIPW with scikit-learn histogram gradient boosting.

- Five folds, stratified by treatment. Each row's nuisance predictions come
  from models trained on the other folds.
- Separate outcome models for the treated and control arms; one propensity
  classifier.
- Fixed seed for folds and boosting. Hyperparameters are library defaults with
  depth-3 trees, fixed in advance and not tuned on the benchmark.
- Estimated propensities are clipped to [0.01, 0.99]. The number of clipped
  rows is reported, and a warning is raised when more than 5% are clipped.
- With an empty adjustment set the models reduce to training-fold means.
- The standard error is the standard deviation of the per-row scores divided
  by the square root of the sample size.

**Fixed-pipeline rule: option A.** `agent/fixed_pipeline.py` adjusts for every
covariate the public metadata labels `pre_treatment` and estimates with AIPW.
It reads only public data.

## Why

The owner's stated requirements, quoted.

**Flexible adjustment:**

> Implement a cross-fitted doubly robust AIPW estimator using scikit-learn
> gradient boosting for outcome and propensity models.
>
> - Use out-of-fold nuisance predictions.
> - Use fixed random seeds.
> - Document propensity clipping and numerical safeguards.
> - Handle empty adjustment sets correctly.
> - Do not describe doubly robust estimation as guaranteed unbiased or
>   accurate.
> - Record estimator assumptions and limitations.

**Fixed-pipeline baseline:**

> Use all pre-treatment covariates with the flexible estimator.
>
> The pipeline must use the same public timing metadata available to the
> agent. It must not access private ground truth.
>
> Do not intentionally weaken the fixed baseline.

**Tool limit:**

> Keep the three-tool limit. No arbitrary code execution or agent frameworks.

**Constraints carried from ADR 001:**

> 1. Flexible estimation is not automatically guaranteed consistent. Its
>    validity depends on adjustment-set correctness, overlap, and estimator
>    assumptions.
> 2. Difference in means always uses the empty adjustment set.
> 3. Graph discovery provides statistical evidence, not proof of causal
>    direction.

**Approved without a stated reason** (to be filled in by the owner):

- Choosing AIPW over a boosted outcome model alone. Claude recommended it and
  the owner approved it.

## Choices made during implementation

Made by Claude while building the tools. None was chosen by the owner.

- **Histogram gradient boosting.** scikit-learn has two gradient boosting
  implementations. The histogram one timed about twice as fast on this data
  size, which matters because refutation re-runs the estimator ten times.
- **Timing-aware discovery.** `discover_graph` runs PC once per timing tier, so
  a pair of columns is never tested by conditioning on a column measured
  later. Plain PC conditioned on the outcome and on the post-outcome column,
  and on the benchmark it dropped real confounder-treatment edges on 3 of 24
  confounded datasets. Edges across tiers are directed by timing and marked
  `timing`; orientations PC finds within a tier are marked `data`.
- **No DoWhy.** The estimators and refuters are written directly with numpy
  and scikit-learn, so every step is short enough to read and all four
  estimators can be refuted the same way. `dowhy` remains a dependency and is
  unused.
- **Weighting, not matching.** `propensity_weighting` (normalized weights,
  logistic propensity model) is the only member of the logistic propensity
  family.
- **Three refuters.** Placebo treatment, random common cause and data subset,
  ten fixed-seed repetitions each. A check fails when the placebo effect or
  the average shift exceeds one standard error of the original estimate.
- **Tools do not block any column.** `estimate_effect` accepts any covariate,
  including the post-outcome column. Choosing the adjustment set is the
  caller's job.
- **Baseline location.** The fixed pipeline lives in `agent/` with the other
  two approaches, so the import scan that forbids reading ground truth covers
  it.

## Tradeoffs

- **The fixed pipeline's rule is valid only on this kind of benchmark.**
  Adjusting for every pre-treatment column is safe here because the graphs
  have no hidden variables and no pre-treatment colliders.
- **A strong baseline leaves the agent little room.** The rule satisfies the
  adjustment criterion on all 30 datasets, so the agent can match the baseline
  on validity but not beat it.
- **Doubly robust is not a guarantee.** Its consistency needs a valid
  adjustment set, overlap, and nuisance models that fit well enough. The
  standard error treats the nuisance predictions as fixed.
- **Clipping trades variance for bias** when overlap is poor.
- **The refuters detect instability, not bias.** They cannot detect a wrong
  adjustment set or a misspecified model.
- **Discovery uses a linear test.** Fisher-z can miss nonlinear dependence and
  handles the binary treatment only approximately.
