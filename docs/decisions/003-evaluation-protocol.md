# 003 — Evaluation protocol

- Status: Accepted in part. The owner set the tolerance, the separation of
  validity from accuracy, the fairness rules and the agent's limits. The
  items under "Choices made during implementation" were made by Claude and
  have not been reviewed by the owner.
- Date: 2026-11-16
- Scope: what counts as a correct answer, how method choice is judged, what
  each approach is given, and how the agent is run.

## Context

Three approaches answer the same question on each of the 30 datasets: the
tool-using agent, the same LLM without tools, and a fixed pipeline with no
LLM. The project's definitions require an objective notion of "correct": the
estimate is within a tolerance of the true ATE, and the tolerance is a config
value. The metrics are estimate accuracy, method choice, and whether
refutation caught bad estimates.

## Options

**Tolerance**

- A. A fixed absolute tolerance in outcome units.
- B. A tolerance relative to the true ATE.
- C. A tolerance in standard deviations of the outcome.

**Judging method choice**

- A. One "correct" method per dataset.
- B. Check each chosen method against the assumptions it actually relies on,
  and report that separately from accuracy.

## Decision

**Tolerance: option C.** An estimate is correct when it is within
`ate_tolerance_sd = 0.10` standard deviations of the outcome of the true ATE.
The results also report mean absolute error and the counts at 0.05, 0.10 and
0.20 SD. No answer counts as incorrect.

**Method choice: option B.** `eval/score.py` records, separately:

- **Adjustment-set validity:** the set passes the backdoor criterion on the
  true graph.
- **Outcome-model correctness:** a linear outcome model is correct for the
  dataset (or treatment is randomized).
- **Propensity-model correctness:** a logistic propensity model is correct for
  the dataset.
- **The chosen estimator's own assumption:**
  - difference in means has none beyond the empty set being valid;
  - linear regression needs outcome-model correctness;
  - propensity weighting needs propensity-model correctness;
  - for the doubly robust estimator, whether its boosted models fit well
    enough cannot be read from the ground truth, so it is recorded as unknown.
- **Overlap:** at least 90% of true propensities lie in [0.05, 0.95].
- **Estimator validity:** the adjustment set is valid, overlap is adequate and
  the estimator's assumption is not known to fail.

Validity and accuracy are reported as a two-by-two table: valid and accurate,
valid but inaccurate, invalid but accurate, invalid and inaccurate.

**Refutation outcomes** are recorded separately again: the checks the agent
ran itself on its final method, and, with `--refute`, every applicable check
run by the evaluation on each approach's final method.

**Fairness.** All three approaches read only the public CSV and public
metadata, including timing labels. Ground truth is loaded by the evaluation
after an approach has answered. An import scan forbids `agent/` and `tools/`
from importing ground-truth code.

**Agent limits.** One loop on the raw Anthropic SDK, at most 8 model calls per
question, three tools, no code execution. The model name is a setting
(`claude-haiku-5-5` by default). Responses are cached on disk keyed by the
whole request, and every tool call is logged.

## Why

The owner's stated reasoning and requirements, quoted.

**Tolerance:**

> Use 0.10 SD of Y as the initial ATE accuracy tolerance, but also report mean
> absolute error and sensitivity at 0.05, 0.10, and 0.20 SD.

**Method choice:**

> Multiple causal estimators may be valid for the same dataset. Do not assign
> one arbitrarily "correct" method.

> Outcome regression and propensity-score weighting have different
> assumptions. A nonlinear treatment-assignment mechanism does not
> automatically invalidate linear outcome regression, and a nonlinear outcome
> does not automatically invalidate correctly specified propensity weighting.
>
> Track outcome-model correctness and propensity-model correctness separately.
>
> A method should be considered valid based on its actual assumptions, not an
> arbitrary scenario label.

> Keep estimator validity separate from estimation accuracy.

**Fairness:**

> Ensure all three approaches receive comparable information. [...] Ground
> truth must remain inaccessible to all three.
>
> Do not artificially weaken the fixed-pipeline baseline to make the agent
> appear better.
>
> Record estimation accuracy, estimator validity, adjustment-set validity, and
> refutation outcomes separately.

**Reporting:**

> Never invent benchmark results.

**Approved without a stated reason** (to be filled in by the owner):

- The value 0.10 for the tolerance. Claude proposed it; the owner approved it
  as an initial value.

## Choices made during implementation

Made by Claude. The owner has not reviewed them.

- **What the LLM approaches see.** The agent and the no-tools LLM receive the
  same deterministic text summary of the public data: column statistics, means
  by treatment group, and the correlation matrix. The only difference between
  them is the tools. Claude proposed this during design; the owner did not
  confirm or reject it.
- **A neutral system prompt.** The agent's prompt explains the task, the tools
  and the answer format, and tells it to use causal reasoning and the timing
  labels. It gives no rule about which columns to adjust for.
- **Final-answer format.** Both LLM approaches end with a JSON object naming
  the estimate, the estimator and the adjustment set. One repair request is
  allowed if the object is missing.
- **The dataset is fixed by the loop.** The model cannot pass a dataset ID to
  a tool.
- **The last step cannot call tools.** On the eighth model call the request
  disables tool use, so the cap is exact.
- **Effort level `medium`** is sent explicitly on every request.
- **No sampling parameters.** `claude-haiku-5-5` rejects non-default
  `temperature`, so reproducibility comes from the response cache, not from
  deterministic sampling.

## Tradeoffs

- **The threshold is a judgement call.** Reporting three tolerances and the
  mean absolute error keeps the conclusion from resting on one number.
- **The no-tools LLM's method is self-reported.** It cannot run an estimator,
  so the estimator and adjustment set in its answer describe its reasoning,
  not a computation the evaluation can inspect.
- **The summary decides how strong the no-tools baseline is.** With the
  correlation matrix in hand, a model can work out a linear regression by
  hand.
- **"Valid" for the doubly robust estimator is weaker** than for the others:
  it means valid as far as the ground truth can show.
- **One response per request.** Each question is asked once and cached, so the
  results describe one draw of the model's behaviour.
- **30 datasets.** Per-scenario counts are small (6 or 9), so differences of
  one or two datasets are not meaningful.
