# Progress

Read this at the start of every session. Updated after every commit.

Last updated: 2026-11-18

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
  - `bench/fingerprint.py` and `tests/bench/test_frozen.py`: v1 is now
    frozen by a recorded spec digest and one fingerprint per dataset
    (`tests/bench/frozen/v1.json`). The record was written from the untouched
    generator, before any v2 change, and the regenerated public files were
    byte-identical to the ones the v1 evaluation used. Until this commit
    there was no stored hash: the older test only checked that writing twice
    gives the same files. Continuous columns are compared by weighted sums
    with a 1e-9 relative tolerance, not by a byte hash, because `exp` and
    `tanh` can differ in the last bit between machines.
  - `bench/spec.py` and `bench/equations.py`: a second nonlinear family.
    `nonlinear_family` is `v1` by default (the original functions) or
    `shared`, where the treatment index and the outcome index use the same
    function `(c1^2 - 1)/sqrt(2) + c1*c2`. `shared` is rejected unless
    `lambda_T` or `lambda_Y` is positive. `reference_sample(spec)` exposes
    the fixed-seed 1,000,000-row sample. v1 public files are byte-identical
    after this change; v1 truth files differ only by the new spec field
    `nonlinear_family: "v1"`.
  - `bench/suite.py`: `benchmark_v2_specs()` returns the 20 v2 specs,
    `ds_101` to `ds_120`, master seed 2. Randomized 2, confounded linear 4,
    confounded nonlinear 10 (outcome side only 2, treatment side only 2, both
    sides 6), post-outcome trap 4. True effects are a seeded draw from
    Uniform(-2, 2) rounded to two decimals; two datasets picked by a seeded
    draw are set to zero. That draw put both zeros on trap datasets
    (`ds_106`, `ds_113`); it was not redrawn.
    `tests/bench/test_suite_v2.py` checks shape, effects, the v1 public
    schema, valid adjustment sets, overlap (same thresholds as v1) and that a
    regression given the true curve recovers the effect.
  - `bench/design.py`: `linear_regression_bias(spec)` records how far a
    linear regression on all pre-treatment covariates would land from the
    true effect with unlimited data (1,000,000-row reference sample, robust
    standard error). Recorded only; the owner ruled that it must not be a
    dataset selection gate. Tests assert it only where theory fixes it at
    zero (linear outcome or randomized treatment).
  - **Benchmark v2 is frozen.** `tests/bench/frozen/v2.json` holds the spec
    digest (`e06a2ef6...6867`) and a fingerprint per dataset;
    `docs/decisions/004-benchmark-v2.md` holds the design, the full spec
    table, the validation list and the freeze rule. At the freeze no LLM
    approach and no doubly robust estimate had been run on any v2 dataset.
    The six "both sides" datasets have a recorded linear-regression bias of
    0.100 to 0.502 SD; every other v2 dataset is at or below 0.003 SD.

  - No-tools ablation. `describe_dataset` and `run_no_tools` take
    `include_correlations` (default true, so the original baseline's prompt
    is unchanged). The evaluation has a fourth approach,
    `no_tools_no_correlations`, listed under `ABLATIONS`: it never runs by
    default and is labelled "(ablation)" in every table and in the plot.
    `eval/run.py` now refuses to write into a directory that already holds a
    report unless `--overwrite` is passed, and takes `--report-dir` and
    `--title`, so the published v1 results cannot be replaced by accident.

  - Ablation run on Benchmark v1 (2026-11-16, `claude-haiku-5-5`, 30 new
    model calls; the original no-tools baseline replayed from cache with 0
    new calls, which confirms its prompt is unchanged). Results are in
    `docs/results/v1-ablation/`. The original files in `docs/results/` are
    byte-identical to before.
    - Without the correlation table the no-tools model was within 0.10 SD on
      18 of 30 datasets (13 at 0.05, 23 at 0.20; mean absolute error 0.124
      SD). By scenario: randomized 6/6, confounded linear 4/9, confounded
      nonlinear 6/9, trap 2/6.
    - It named a valid adjustment set on all 30. The three existing
      refutation checks flagged 0 of its 12 inaccurate answers.
  - `README.md`: the v1 results are labelled Benchmark v1 and kept as
    written; the ablation is a labelled addendum; a short Benchmark v2
    section says v2 is frozen and not yet run.

  - The owner approved the frozen v2 benchmark, the paired standard error
    and the 0.10 SD materiality bound, and authorized the final
    implementation and one v2 evaluation. `step-5-benchmark-v2` is pushed as
    PR #5.
  - Estimator-disagreement check (ADR 005). Every estimator now returns one
    contribution per row whose mean is its estimate. `refute_estimate` has a
    fourth check, `estimator_disagreement`: it compares the estimate with one
    from the other model family on the same adjustment set, computes a paired
    standard error from the row contributions, and returns `disagreement`,
    `agreement` or `inconclusive` against a bound of 0.10 outcome SDs. The
    three original checks, their outputs and their cached results are
    unchanged (`STABILITY_REFUTERS`). The agent can call the new check, so
    its toolset now differs from the one used for the v1 results (commit
    `67ee865`).
  - Evaluation runner for v2. `--benchmark v1|v2` picks the specs, the data
    directory (`data/benchmark`, `data/benchmark_v2`) and the default report
    directory (`docs/results`, `docs/results/v2`). With `--refute` the
    evaluation also runs the disagreement check on every final method and
    reports detected problems, false alarms, misses and inconclusive results
    in their own table. Every report now has a table of model calls, tool
    calls, tool errors and tokens, a breakdown by finer design, and a run
    record (`run_record.json`: model requested and model reported by the API,
    settings, spec digest, commit, whether the working tree was clean).

  - **Benchmark v2 run** (one run, 2026-11-18 UTC, `claude-haiku-5-5` as
    requested and as reported by the API, 98 new model calls, code commit
    `9a6d836`, clean working tree, spec digest `e06a2ef6...6867`). Results
    are in `docs/results/v2/`, including `run_record.json`. A first attempt
    was stopped by Claude Code for low system memory after the fixed
    pipeline had answered two datasets and before any model call; the owner
    approved the restart, which ran everything from the start.
    - Within 0.10 SD: fixed pipeline 20/20, agent 16/20, no-tools LLM 16/20,
      ablation 10/20. **The agent lost to the fixed pipeline** and tied the
      no-tools LLM. All four of the agent's misses are on the six
      "nonlinear on both sides" datasets, where it reported linear
      regression.
    - The agent ran the disagreement check on 12 of 20 questions. On each
      of its four misses the check said `disagreement`; the agent read it
      and still reported linear regression, calling the doubly robust
      estimate an outlier.
    - The disagreement check run by the evaluation: on agent answers it
      flagged 4 of 4 inaccurate and 0 of 16 accurate (1 inconclusive); on
      fixed-pipeline answers it flagged 4 of 20 accurate ones, on the four
      datasets where the comparison estimator (linear regression) is the
      biased one. The three original checks flagged no inaccurate estimate
      of any approach.
  - Final documentation: `README.md` has a summary across both benchmarks,
    the v1 results as first published, the v2 results, the comparison with
    both baselines, the ablation findings, a refutation failure analysis,
    the architecture, how to reproduce, and limitations. ADR 003 records the
    additions to the protocol, ADR 004 an "After the run" note, ADR 005 the
    measured behaviour of the check.

## Verification status

- CI passed on PR #1 and PR #2. PR #2 ran all 505 tests on Linux, so the
  generator's tests are not sensitive to the Windows/Linux difference.
- CI passed on PR #3 (`step-2-baseline-tools`) and on PR #4
  (`step-4-agent-eval`) while it was stacked on PR #3. Local checks pass
  before every commit (`uv run ruff check .`, `uv run ruff format --check .`,
  `uv run pytest`).
- `step-5-benchmark-v2` is PR #5. CI passed on it at `9a6d836`. Locally
  1,121 tests pass. The full suite took 24 minutes once while the machine was
  short of memory and about 5 minutes otherwise; on CI it takes about 2.
- The agent loop and no-tools baseline are unit-tested with a scripted model
  and were then run against the real API: 30 v1 datasets, 20 v2 datasets.

## Next

1. The owner authorized merging PR #5 once CI passes on its final commit.
2. Nothing else is planned. Possible follow-ups, only if the owner asks: a
   decision rule for the agent when the disagreement check fires, hidden
   confounding, effects that vary between units, more seeds per dataset,
   other models.

## Working rules set by the owner

- Never commit directly to `main`. Work on a branch per step.
- `CLAUDE.md` stays local and untracked (listed in `.git/info/exclude`).
  Never include it in a commit.
- No Claude or AI co-author attribution in commit messages.
- Do not push or merge without the owner's say-so.
- Stop for review at three checkpoints: generator and metadata (done), tools
  and baselines (done), full evaluation (done).
- Benchmark v2: do not tune datasets based on which approach performs better.
  The specs are frozen and must not change after any approach has run.
- Preserve all Benchmark v1 results, documentation and run settings. Never
  overwrite `docs/results/` from a v2 or ablation run.
- Do not silently resolve major causal-inference decisions. Report design
  deviations at each checkpoint.

## Approved decisions from the design stage

All of these are now recorded in ADR 002 and ADR 003.

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
- ADR 004 and ADR 005 each list items "approved without a stated reason"
  (option B, the size and composition of v2, the two zero-effect datasets;
  comparing estimates over a specification test, the 0.10 SD bound). Do not
  fill them in without the owner's words.
- The shared summary includes the correlation matrix, which lets the no-tools
  LLM work out a regression by hand. The owner never confirmed this design;
  the ablation measures its effect and the original baseline is unchanged.
- The generated benchmarks live in `data/benchmark/` and `data/benchmark_v2/`
  and are gitignored; they are regenerated from seeds on every run and
  checked against the frozen fingerprints by the tests.
