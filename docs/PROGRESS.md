# Progress

Read this at the start of every session. Updated after every commit.

Last updated: 2026-11-16

## Done

- **Step 0, infrastructure.** Merged into `main` through PR #1. Installable
  `src/causal_agent` package, ruff and pytest config, GitHub Actions
  workflow, ADR 001.
- **Step 1, generator and metadata.** Merged into `main` through PR #2
  (checkpoint 1 approved by the owner).
  - `bench/spec.py`: `DatasetSpec`, `Scenario`, scenario-versus-knob checks.
  - `bench/equations.py`: structural equations, fixed reference constants,
    one random stream per variable.
  - `bench/suite.py`: the 30 specs. All pass overlap and balance with no
    knob lowered.
  - `bench/graph.py`: true graph, column roles, backdoor check.
  - `bench/public.py`: public metadata and loader. The only bench module
    agent and tool code may import.
  - `bench/truth.py`: private ground truth and loader.
  - `bench/generate.py`: dataset assembly and benchmark writer.
  - `tests/test_isolation.py`: import scan for `agent/` and `tools/`.
- **Step 2 and 3, baseline and tools.** On branch `step-2-baseline-tools`
  (not pushed).
  - `tools/aipw.py`: cross-fitted doubly robust (AIPW) estimator with
    scikit-learn histogram gradient boosting, out-of-fold nuisance
    predictions, propensity clipping to [0.01, 0.99], fixed seeds. The module
    docstring records its assumptions and limitations.
  - `agent/fixed_pipeline.py`: the fixed-pipeline baseline. Adjusts for every
    covariate labelled `pre_treatment` in the public metadata and estimates
    with AIPW. Tested against a directory that has no truth files.
    `agent/` holds all three approaches (agent loop and both baselines) so
    the existing import scan covers them.
  - `tools/discover_graph.py`: PC with Fisher-z tests from causal-learn, run
    once per timing tier so that a pair of columns is never tested by
    conditioning on a later column. Edges across tiers are directed by timing
    and marked `timing`; same-tier orientations from PC are marked `data`;
    the rest stay undirected. A fixed list of caveats is part of the output.
    - Why tiers: plain PC conditioned on the outcome and the post-outcome
      column and dropped real confounder-treatment edges (3 of 24 confounded
      datasets, and the confounder-outcome edges on 2 trap datasets). With
      tiers, both confounders are found on all 24.
  - `tools/estimators.py`: `difference_in_means` (always the empty set),
    `linear_regression` (OLS), `propensity_weighting` (logistic propensity,
    normalized weights, clipping) and a dispatcher that also runs
    `doubly_robust`. `ASSUMPTIONS` states what each one relies on. Only
    weighting represents the "logistic propensity" family; matching is not
    implemented.
  - `tools/estimate_effect.py`: typed tool. Input: dataset ID, estimator,
    adjustment set. Output echoes both, adds the estimate, standard error,
    95% interval, diagnostics, warnings and the estimator's assumptions. It
    does not block any covariate, including the post-outcome column: choosing
    the adjustment set is the caller's job.
  - `tools/refute_estimate.py`: typed tool with three checks (placebo
    treatment, random common cause, data subset), 10 fixed-seed repetitions
    each. A check fails when the placebo effect or the average shift exceeds
    one standard error of the original estimate. The output states that a
    pass is not evidence of an unbiased estimate.
  - Discovery notes, continued:
    - Measured on the 30 datasets at alpha 0.05: 3 of 6 randomized datasets
      show one chance covariate-treatment edge (p = 0.047, 0.009, 0.039).
      The test is calibrated: 60 of 1,200 null tests rejected on fresh
      randomized data.

## Verification status

- CI passed on PR #1 and PR #2. PR #2 ran all 505 tests on Linux, so the
  generator's tests are not sensitive to the Windows/Linux difference.
- `step-2-baseline-tools`: local checks pass before every commit
  (`uv run ruff check .`, `uv run ruff format --check .`, `uv run pytest`).
  CI has not run on this branch.

## Next

1. Checkpoint 2 reached: the fixed baseline and the three tools are
   implemented. Waiting for the owner's review. Do not start the LLM agent
   before then.
2. After approval: agent loop on the raw Anthropic SDK (8-step cap, cached
   responses, logged tool calls), then the no-tools LLM baseline, then the
   evaluation script, results table, plot and README.

## Working rules set by the owner

- Never commit directly to `main`. Work on a branch per step.
- `CLAUDE.md` stays local and untracked (listed in `.git/info/exclude`).
  Never include it in a commit.
- No Claude or AI co-author attribution in commit messages.
- Do not push or merge without the owner's say-so.
- Stop for review at three checkpoints: generator and metadata (done), tools
  and baselines, full evaluation.
- Do not silently resolve major causal-inference decisions. Report design
  deviations at each checkpoint.

## Approved decisions not yet recorded as ADRs

Record each one when the part it belongs to is built.

- **ATE tolerance:** an estimate is correct if it is within 0.10 standard
  deviations of `Y` of the true ATE. Also report mean absolute error and
  sensitivity at 0.05, 0.10 and 0.20 SD. The owner has not yet stated a
  reason for 0.10; ask before writing the ADR.
- **Flexible adjustment:** cross-fitted doubly robust AIPW with scikit-learn
  gradient boosting for the outcome and propensity models. Out-of-fold
  nuisance predictions, fixed seeds, documented clipping, correct handling of
  an empty adjustment set. Never described as guaranteed unbiased or
  accurate.
- **Fixed-pipeline baseline:** all pre-treatment covariates with the flexible
  estimator. Uses the same public timing metadata as the agent and no private
  ground truth. Must not be weakened.
- **Method validity:** no single "correct" method. Record separately:
  estimation accuracy, adjustment-set validity, outcome-model correctness,
  propensity-model correctness, estimator validity, refutation outcome.
- **Fairness:** the agent, the no-tools LLM and the fixed pipeline get
  comparable public information. Ground truth is inaccessible to all three.
- **Constraints from the owner:**
  1. Flexible estimation is not automatically consistent. Its validity
     depends on adjustment-set correctness, overlap, and estimator
     assumptions.
  2. Difference in means always uses the empty adjustment set.
  3. Graph discovery gives statistical evidence, not proof of causal
     direction.

## Open questions

- ADR 001 keeps two items marked "approved without a stated reason"
  (5,000 rows; the nonlinear function forms). The owner chose to leave them
  marked that way. Do not fill them in.
- Where the generated benchmark lives on disk, and whether it is committed or
  regenerated from seeds. Decide when the evaluation script is built.
- What the no-tools LLM baseline sees (proposal: one deterministic summary of
  the public CSV, shared with the agent). Decide at step 4.
- `dowhy` is a dependency but the estimators are implemented directly with
  numpy and scikit-learn. Raise at checkpoint 2.
