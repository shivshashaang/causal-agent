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

- **Steps 4 to 6, agent, no-tools baseline and evaluation.** On branch
  `step-4-agent-eval`, stacked on `step-2-baseline-tools` (neither pushed).
  The owner replied "continue" to the checkpoint 2 report without answering
  its three questions, so the no-change options are in force until the owner
  says otherwise: no new refuter, nonlinear functions untouched, `dowhy`
  left in place.
  - `config.py`: `Settings` (model `claude-haiku-5-5`, 8-step cap, tolerance
    0.10 SD with sensitivity 0.05/0.10/0.20, paths) and `load_api_key`, which
    reads `AGENT_LLM_KEY` only.
  - `agent/llm.py`: `AnthropicClient` (raw SDK, key passed explicitly) and
    `CachedLLM`, a disk cache keyed by a hash of the whole request. A fully
    cached run needs no key. Thinking blocks are replayed unchanged.
  - `agent/summary.py`: deterministic text summary of the public data
    (column statistics, means by treatment group, correlations). The agent
    and the no-tools baseline both receive exactly this.
  - `agent/answer.py`: the final-answer JSON format (estimate, estimator,
    adjustment set) and a lenient parser shared by both LLM approaches.
  - `agent/toolbox.py`: the three tool definitions sent to the model (JSON
    schemas generated from the pydantic input models, minus `dataset_id`)
    and `execute_tool`. The loop fixes the dataset; the model cannot pass
    one. Bad calls come back as readable error results.
  - `agent/loop.py`: the agent loop on the raw Anthropic SDK. One model call
    is one step, capped at 8; the last step cannot call tools. Every tool
    call is recorded with input, output and error. The system prompt gives no
    rule about which columns to adjust for. Tested with a scripted model
    only; no real API call has been made yet.
  - `agent/no_tools.py`: the no-tools baseline. Same model, same summary, no
    tools, one call plus at most one repair request. Its estimator and
    adjustment set are self-reported.
  - `eval/score.py`: scores one answer against ground truth. Accuracy
    (within a multiple of the outcome SD) and validity are separate. Validity
    records adjustment-set validity, outcome-model correctness,
    propensity-model correctness, the chosen estimator's own assumption, and
    overlap. For the doubly robust estimator the model-fit assumption is
    recorded as unknown, never as guaranteed.
  - `eval/report.py`: results Markdown (accuracy with tolerance sensitivity,
    by scenario, validity 2x2, estimator choice, refutation outcomes, agent
    behaviour, every dataset), a CSV with one row per approach and dataset,
    and one dot plot of error by approach and scenario. Adds `matplotlib` as
    a dependency. Plot colours were checked with the dataviz validator.
  - `eval/run.py`: `python -m causal_agent.eval.run` generates the benchmark
    under `data/benchmark/` (gitignored), runs the three approaches, logs
    each question to `results/runs/`, reads ground truth only after an
    approach has answered, and writes `docs/results/`. `--refute` also runs
    every applicable refuter on each final method, cached on disk.
  - `docs/decisions/002-estimators-baseline-tools.md` and
    `003-evaluation-protocol.md`. Both quote the owner's stated requirements
    and list separately the choices Claude made during implementation that
    the owner has not reviewed.
  - First real API calls (one dataset, `ds_001`): the request shape works.
    The no-tools model solved a regression by hand from the correlation
    table and landed within tolerance, so the shared summary is informative
    enough for that baseline to be strong on linear data.

  - Full evaluation run (2026-11-16, `claude-haiku-5-5`, 113 model calls).
    Results are in `docs/results/` and summarized in `README.md`.
    - Accuracy: all three approaches were within 0.10 SD on all 30 datasets.
      The agent did not beat either baseline.
    - Validity: adjustment set valid 30/30 for all three. Estimator valid:
      fixed pipeline 30/30, agent 24/30, no-tools LLM 24/30 (self-reported).
      The agent reported linear regression on every dataset, which is
      misspecified on the 6 nonlinear-outcome datasets.
    - The no-tools LLM computed a regression by hand from the correlation
      table on every dataset; 8 of its answers equal the true effect exactly
      because it rounds and the true effects are round numbers.
    - No approach adjusted for the post-outcome column.
    - Refutation (`--refute`, every applicable check on each final method):
      no final estimate was inaccurate, so the catch rate is undefined. Two
      false alarms: the data-subset check failed on two accurate
      fixed-pipeline estimates (`ds_001`, `ds_027`). The agent refuted its
      own final method on 10 of 30 questions; all passed.

**Checkpoint 3 reviewed by the owner.** `step-2-baseline-tools` was merged
into `main` through PR #3 and `step-4-agent-eval` through PR #4, each after
CI passed. The owner approved Benchmark v2 (see Next).

- **Step 5, Benchmark v2.** On branch `step-5-benchmark-v2` (not pushed).
  - Removed the `dowhy` dependency. It was never imported in `src/` or
    `tests/`; every package the code imports is a direct dependency.

## Verification status

- CI passed on PR #1 and PR #2. PR #2 ran all 505 tests on Linux, so the
  generator's tests are not sensitive to the Windows/Linux difference.
- CI passed on PR #3 (`step-2-baseline-tools`) and on PR #4
  (`step-4-agent-eval`) while it was stacked on PR #3. Local checks pass
  before every commit (`uv run ruff check .`, `uv run ruff format --check .`,
  `uv run pytest`; 780 tests).
- The agent loop and no-tools baseline are unit-tested with a scripted model
  and were then run against the real API on all 30 datasets.

## Next

1. Done: PR #4 merged, `step-5-benchmark-v2` created from `main`.
2. Benchmark v2, approved by the owner: 20 datasets (`ds_101` to `ds_120`)
   with a nonlinear feature shared by the treatment and outcome equations,
   varied non-round true effects and two zero-effect datasets. v1 datasets,
   results and run settings stay as they are. Specs, validation tests and
   ADR 004 are frozen before any LLM evaluation.
3. Also approved: an estimator-disagreement check inside `refute_estimate`
   (uncertainty method to be proposed to the owner first), a no-tools
   ablation without the correlation matrix on v1 and v2, and removal of the
   unused `dowhy` dependency.
4. Stop for the owner's review before running the v2 LLM evaluation.

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
- The three questions from checkpoint 2 were answered at the checkpoint 3
  review: add an estimator-disagreement check to `refute_estimate`; leave the
  v1 nonlinear functions alone and add a second family for v2; remove
  `dowhy`.
- ADR 002 and ADR 003 each list choices Claude made during implementation
  that the owner has not reviewed, and each has one item marked "approved
  without a stated reason" (AIPW over an outcome model alone; the value 0.10
  for the tolerance).
- The shared summary includes the correlation matrix, which lets the no-tools
  LLM work out a regression by hand. The owner never confirmed this design.
- The benchmark does not separate the three approaches. Whether to make it
  harder is the owner's call.
- The generated benchmark lives in `data/benchmark/` and is gitignored; it is
  regenerated from seeds on every run.
