# causal-agent

A tool-using LLM agent that answers "what is the effect of X on Y?" on tabular
data. It chooses a causal method, runs it through three typed tools, and checks
its own answer. It is evaluated on 30 synthetic datasets where the true effect
is known, against two baselines: the same LLM with no tools, and a fixed
pipeline with no LLM.

There are two benchmarks. **Benchmark v1** (30 datasets) has been run; its
results are below and are kept exactly as first published. **Benchmark v2**
(20 harder datasets) is specified and frozen but has not been run yet; see
[Benchmark v2](#benchmark-v2-frozen-not-yet-run).

## Benchmark v1 results

**The agent did not beat either baseline.** All three approaches were within
tolerance on all 30 datasets, so the benchmark did not separate them on
accuracy. On method validity the agent was behind the fixed pipeline.

Every number here comes from one real run of
`python -m causal_agent.eval.run --refute` (model `claude-haiku-5-5`). The
full report is in [docs/results/results.md](docs/results/results.md) and the
per-dataset scores are in [docs/results/scores.csv](docs/results/scores.csv).

### Accuracy

An estimate is correct when it is within 0.10 standard deviations (SD) of the
outcome of the true effect.

| Approach | Within 0.05 SD | Within 0.10 SD | Within 0.20 SD | Mean abs. error (SD) | Median abs. error (SD) |
|---|---|---|---|---|---|
| Agent (tools) | 30/30 | 30/30 | 30/30 | 0.015 | 0.012 |
| Fixed pipeline | 28/30 | 30/30 | 30/30 | 0.020 | 0.016 |
| LLM, no tools | 30/30 | 30/30 | 30/30 | 0.014 | 0.014 |

Every approach was correct on every dataset in every scenario (6/6, 9/9, 9/9
and 6/6). The largest error by any approach was 0.06 SD.

![Estimation error by approach and scenario](docs/results/error_by_scenario.png)

### Method validity

Validity is whether a method's assumptions hold for the dataset, judged
against the true causal graph. It is scored separately from accuracy.

| Approach | Adjustment set valid | Estimator valid | Invalid but accurate |
|---|---|---|---|
| Agent (tools) | 30/30 | 24/30 | 6 |
| Fixed pipeline | 30/30 | 30/30 | 0 |
| LLM, no tools (self-reported) | 30/30 | 24/30 | 6 |

The agent reported a linear regression on all 30 datasets. On the 6 datasets
with a nonlinear outcome, a linear outcome model is misspecified, so those 6
answers count as invalid even though they were accurate. For the fixed
pipeline's doubly robust estimator, "valid" means a valid adjustment set and
adequate overlap; whether its boosted models fit well enough cannot be read
from the ground truth.

### Refutation

The evaluation ran every applicable refutation check on each approach's final
method. An estimate is flagged when at least one check fails.

| Approach | Estimates checked | Inaccurate estimates flagged | Accurate estimates flagged |
|---|---|---|---|
| Agent (tools) | 30 | 0 of 0 | 0 of 30 |
| Fixed pipeline | 30 | 0 of 0 | 2 of 30 |
| LLM, no tools | 30 | 0 of 0 | 0 of 30 |

Because no final estimate was inaccurate, this run cannot show whether
refutation catches bad estimates. It does show two false alarms: the
data-subset check failed on two accurate fixed-pipeline estimates (`ds_001`
and `ds_027`). The doubly robust estimator refits its models on each subset,
so it moves more between subsets than a regression does.

The agent ran a refutation check on its own final method in 10 of 30
questions; all 10 passed.

A separate check during development, not part of the committed evaluation,
ran the placebo and data-subset checks on estimates that are biased by
construction. They flagged 0 of 23 inaccurate unadjusted estimates and 0 of 4
inaccurate regressions that adjusted for the trap column. These checks detect
instability, not bias.

### Ablation: the no-tools baseline without the correlation table

This was run after the v1 results above, as a separate run. It adds a fourth
approach and changes nothing above. The ablation is the no-tools baseline with
the correlation table removed from its summary; the column statistics, the
means by treatment group and the timing labels stay. The full report is in
[docs/results/v1-ablation/results.md](docs/results/v1-ablation/results.md).

| Approach | Within 0.05 SD | Within 0.10 SD | Within 0.20 SD | Mean abs. error (SD) | Median abs. error (SD) |
|---|---|---|---|---|---|
| LLM, no tools (original, from above) | 30/30 | 30/30 | 30/30 | 0.014 | 0.014 |
| LLM, no tools, no correlations (ablation) | 13/30 | 18/30 | 23/30 | 0.124 | 0.079 |

| Scenario | LLM, no tools (original) | No correlations (ablation) |
|---|---|---|
| Randomized | 6/6 | 6/6 |
| Confounded, linear | 9/9 | 4/9 |
| Confounded, nonlinear | 9/9 | 6/9 |
| Post-outcome trap | 6/6 | 2/6 |

Without the correlation table the no-tools model was correct on 18 of 30
datasets. It stayed correct on all 6 randomized datasets, where the raw gap
between the groups is the answer, and lost accuracy wherever adjustment is
needed. So on v1 the no-tools baseline matched the agent because the summary
handed it the inputs of a regression, not because tools are unnecessary. With
that table removed, the agent (30/30) and the fixed pipeline (30/30) are both
ahead of the no-tools model on v1.

The ablation named a valid adjustment set on all 30 datasets and still missed
12, so its misses are not explained by which columns it said it adjusted for.
Its method is self-reported: it has no way to run one. The evaluation's
refutation checks flagged none of
those 12. For a no-tools answer the checks re-run the method the model named,
not the number it guessed, so this is weak evidence about the checks.

## Failure analysis (Benchmark v1)

There were no accuracy failures to analyse in the original v1 run, so this
section explains why the benchmark did not separate the approaches and where
the agent's method choice fell short. It was written before the ablation
above and is kept as written.

**The agent's tools added no measurable accuracy over the summary.** The
no-tools model received column statistics, group means and the correlation
matrix. From those it worked out a linear regression by hand on every dataset
and was as accurate as the agent. Its method is self-reported, but its written
working shows the calculation.

**Round true effects helped the no-tools model.** The true effects are 0, 0.5,
1, 2 and −1. The no-tools model rounds its hand calculation (for example 1.99
to 2.0), and 8 of its 30 answers equal the true effect exactly. The agent had
2 exact answers and the fixed pipeline none.

**The agent chose the more precise estimator, not the more robust one.** It
ran both linear regression and the doubly robust estimator on all 30 datasets
and reported the linear regression every time. Its stated reasons were that
the two estimates agreed and that the regression was more precise (27 of 30
answers cite precision or the standard error). That cost it validity on the 6
nonlinear-outcome datasets and cost it nothing in accuracy. None of the three
tools reports a diagnostic for a misspecified outcome model, so the agent had
no direct evidence to prefer the robust estimator; in at least one answer it
asserted that the outcome "looks roughly linear" without a tool result to
support it.

**The nonlinear scenario does not punish linear regression.** A separate check
during development ran linear regression with all pre-treatment covariates on
the 9 nonlinear datasets, and it was within tolerance on all 9. The scenario
breaks the estimator's assumption without producing enough bias to matter at
this tolerance.

**The trap was avoided by everyone.** No approach adjusted for the
post-outcome column on any of the 6 trap datasets. The timing label makes the
column easy to spot: the fixed pipeline excludes it by rule, and both LLM
approaches cited the label.

**Adjusting for everything measured before treatment is always right here.**
The agent's final adjustment set equalled all pre-treatment columns on 22 of
30 datasets and was a valid subset on the other 8. With no hidden confounders
and no pre-treatment colliders, there is no dataset where choosing the
adjustment set is hard.

**How the agent used its tools.** It needed 2 or 3 model calls per question
(23 questions took 3), well under the cap of 8, and made 125 tool calls with
no errors: `estimate_effect` on all 30 questions, `discover_graph` on 25 and
`refute_estimate` on 10. Every final answer matched a logged
`estimate_effect` call.

## Benchmark v2 (frozen, not yet run)

Benchmark v2 exists because v1 could not rank the approaches. A linear
regression is biased only when the same curve drives both who is treated and
the outcome, and v1's two curves are nearly unrelated. In v2 the treatment and
the outcome share one curved function of the confounders. Everything else is
as in v1: every confounder is observed, the treatment effect is constant, and
the public columns and timing labels are the same.

- 20 datasets, `ds_101` to `ds_120`: 2 randomized, 4 confounded linear, 10
  confounded nonlinear (2 with a curved outcome only, 2 with a curved
  treatment model only, 6 with both), 4 with the post-outcome trap.
- True effects are non-round values drawn by a seeded generator; two datasets
  have an effect of exactly zero.
- The specs, their validation tests and the decision record were committed
  before any approach was run on v2, and they do not change afterwards.
- For each dataset the record lists how far a linear regression would land
  from the truth with unlimited data. It is 0.100 to 0.502 SD on the six
  "both" datasets and at most 0.003 SD on the other fourteen. That number was
  recorded, not used to choose datasets.

The full specification is in
[docs/decisions/004-benchmark-v2.md](docs/decisions/004-benchmark-v2.md).
There are no v2 results yet.

## Limitations

These describe Benchmark v1 and its run.

- **The benchmark is too easy to rank the approaches.** All three scored
  30/30. The results show that the pipeline, tools and agent work end to end;
  they do not show that an agent is better than a fixed rule.
- **No hidden confounding.** Every confounder is observed, so the hardest
  problem in real causal inference is absent.
- **Constant treatment effect.** The average effect and the effect on the
  treated coincide, so the benchmark cannot detect confusion between them.
- **Synthetic data only.** Independent Gaussian covariates, a clean binary
  treatment and no missing values. Nothing here says how the approaches would
  do on real data.
- **Small sample.** 30 datasets, one seed each, and one model response per
  question. Per-scenario counts are 6 or 9.
- **The refuters detect instability, not bias.** They cannot detect a wrong
  adjustment set or a misspecified model.
- **Validity for the doubly robust estimator is weaker** than for the others,
  because the fit of its models cannot be checked against the ground truth.
- **The fixed pipeline's rule is specific to this benchmark.** Adjusting for
  every pre-treatment column is not safe in general.
- **One model.** Only `claude-haiku-5-5` was run.

## How it works

### The benchmark

Thirty synthetic datasets, each with a binary treatment `t`, a continuous
outcome `y` and four covariates `x1..x4`. The true average treatment effect
(ATE) is a number written into the outcome equation, so it is known exactly.

| Scenario | Datasets | What is hard about it |
|---|---|---|
| Randomized | 6 | Nothing: treatment is independent of the covariates |
| Confounded, linear | 9 | Two covariates drive both treatment and outcome |
| Confounded, nonlinear | 9 | The same, with a nonlinear treatment model, outcome model, or both |
| Post-outcome trap | 6 | One column is caused by both treatment and outcome and must not be adjusted for |

Confounding strength, outcome noise and nonlinearity are independent knobs.
Column names are neutral and shuffled, dataset IDs do not encode the scenario,
and every dataset has the same public schema. The only hint about the trap
column is a timing label (`post_outcome`), as a data dictionary would give.

A **confounder** is a variable that influences both who gets treated and the
outcome. It makes the raw gap between treated and untreated units differ from
the true effect. Adjusting for it removes that distortion; adjusting for a
variable that is itself caused by the treatment or outcome adds a new one.

### The three tools

| Tool | What it does | What it cannot do |
|---|---|---|
| `discover_graph` | Runs the PC algorithm and reports which columns stay associated after conditioning on others | Prove causation or direction; directions come from timing labels |
| `estimate_effect` | Estimates the ATE with a chosen estimator and adjustment set | Judge whether the adjustment set is appropriate |
| `refute_estimate` | Re-runs an estimate under a placebo treatment, an added noise covariate, or random subsets | Detect a wrong adjustment set or a misspecified model |

Estimators: difference in means, linear regression, propensity weighting, and
a cross-fitted doubly robust estimator with gradient-boosted models. There is
no tool that runs arbitrary code, and the model cannot choose which dataset a
tool reads.

### The three approaches

- **Agent.** One loop on the raw Anthropic SDK (`claude-haiku-5-5`), at most 8
  model calls per question. It sees a text summary of the data and can call
  the three tools.
- **LLM, no tools.** The same model and the same summary, and nothing else.
- **Fixed pipeline.** No LLM. It adjusts for every column labelled
  `pre_treatment` and uses the doubly robust estimator.

All three read only the public data. Ground truth is loaded by the evaluation
after an approach has answered.

### Scoring

An estimate is **correct** when it is within 0.10 standard deviations of the
outcome of the true ATE. Results are also shown at 0.05 and 0.20.

**Validity** is scored separately from accuracy. A method is valid when its
adjustment set passes the backdoor criterion on the true graph, overlap is
adequate, and the estimator's own assumption is not known to fail. A valid
method can be inaccurate, and an invalid one can land close by luck.

## Reproduce

Requires [uv](https://docs.astral.sh/uv/). Commands are for PowerShell.

```powershell
uv sync
uv run ruff check .
uv run pytest
```

To run the evaluation, put an Anthropic API key in `.env`:

```
AGENT_LLM_KEY=your-key
```

```powershell
# Benchmark v1, all three approaches, also refuting each final method
uv run python -m causal_agent.eval.run --refute --report-dir results\rerun-v1

# The no-tools ablation beside the original no-tools baseline
uv run python -m causal_agent.eval.run --refute --approaches no_tools no_tools_no_correlations --report-dir results\rerun-v1-ablation
```

A run stops before doing anything if its report directory already holds a
report, so the published results in `docs/results/` cannot be replaced by
accident. Pass `--report-dir` to write somewhere else, or `--overwrite` to
replace a report on purpose.

Model responses are cached under `results/cache/`, so a second run on the same
machine makes no API calls and gives the same answers. Each question's tool
calls are logged under `results/runs/`.

## Repository layout

```
src/causal_agent/
  bench/    synthetic datasets: specs, structural equations, true graphs,
            public metadata, private ground truth
  tools/    discover_graph, estimate_effect, refute_estimate, and the estimators
  agent/    the agent loop, the no-tools baseline, the fixed pipeline
  eval/     scoring, the report, the run script
  config.py settings: model, step cap, tolerance, paths
tests/      one test module per source module
docs/
  decisions/  decision records
  results/    Benchmark v1: results table, per-dataset scores, plot
    v1-ablation/  the no-tools ablation on Benchmark v1
  PROGRESS.md
```

## Decision records

- [001: Synthetic dataset generator](docs/decisions/001-synthetic-generator.md)
- [002: Estimators, fixed-pipeline baseline, and the three tools](docs/decisions/002-estimators-baseline-tools.md)
- [003: Evaluation protocol](docs/decisions/003-evaluation-protocol.md)
- [004: Benchmark v2](docs/decisions/004-benchmark-v2.md)
