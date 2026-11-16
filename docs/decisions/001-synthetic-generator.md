# 001 — Synthetic dataset generator

- Status: Accepted
- Date: 2026-11-16
- Scope: how benchmark datasets are generated and what metadata is public or
  private. Evaluation rules (tolerance, validity scoring, baselines) get their
  own records when those parts are built.

## Context

The agent is evaluated on about 30 synthetic datasets where the true effect is
known. Every question asks for the average treatment effect (ATE) of one
treatment on one outcome. The datasets must vary confounding, noise and
nonlinearity, and the ground truth must stay hidden from the agent and from
both baselines.

## Options

**A. Scenario templates with a constant treatment effect.** A few hand-written
causal structures crossed with difficulty knobs. The true ATE is the number
`τ` written into the outcome equation, so it is exact. Simplest; no effect
heterogeneity; only as much structural variety as is hand-written.

**B. Potential outcomes with a varying effect.** Generate both outcomes per
row and reveal one. The true ATE is the mean of `Y1 − Y0` over the rows. More
realistic; strong confounding breaks overlap; an instrument would no longer
identify the ATE.

**C. Random graphs with simulated interventions.** Sample a random graph and
mechanisms, then approximate the true ATE by forcing treatment to 1 and 0 on a
large sample. Best test of graph discovery; identifiability is not guaranteed
without a checker; most code; least control over difficulty.

## Decision

Design A, revised as follows.

### Variables and causal roles

Every dataset has a binary treatment `t`, a continuous outcome `y`, and four
covariates published as `x1..x4` in shuffled order.

| Internal variable | Edges in the true graph | Role | Timing label |
|---|---|---|---|
| `c1`, `c2` | → `T` (when `a > 0`), → `Y` | confounder; plain outcome predictor when `a = 0` | `pre_treatment` |
| `p` | → `Y` only | outcome-only predictor | `pre_treatment` |
| `n` | none | irrelevant | `pre_treatment` |
| `k` (trap datasets only, replaces `n`) | `T` → `k` ← `Y` | post-outcome collider | `post_outcome` |

### Structural equations

- `T ~ Bernoulli(sigmoid(a · h(C)))`
- `Y = τ·T + b · g(C) + d · p + σ · noise`
- `k = γ · (T + (Y − μ_Y)/sd_Y) + noise` (trap only; drawn last so it cannot
  change `T` or `Y`). `μ_Y` and `sd_Y` are fixed reference statistics, not
  the dataset's own mean and standard deviation.
- `h(C) = linear(C) + λ_T · nonlinear_T(C)`, centred and rescaled to unit
  variance
- `g(C) = linear(C) + λ_Y · nonlinear_Y(C)`, centred and rescaled to unit
  variance
- `linear(C) = (c1 + c2)/√2`
- `nonlinear_T(C)` = interaction `c1·c2` plus threshold `1[c1 > 0.5]`
- `nonlinear_Y(C)` = square `c2²` plus saturating curve `tanh(c1 + 0.5)`

The true ATE is `τ` exactly, because the effect is constant. The function
forms are fixed once and are not tuned after seeing estimator results.

Implementation details added with the first generator commit:

- **Normalizing constants** for `h`, `g` and the trap's `Y` come from a
  reference sample of 1,000,000 draws with a fixed seed, so every structural
  equation is a fixed function and no row depends on any other row. At
  `λ = 0` the index is exactly `linear(C)`. The owner required fixed
  reference statistics for `Y` in the trap equation, "so the structural
  equation does not depend on other generated rows."
- **Random streams:** each variable (`c1`, `c2`, `p`, `n`, treatment, outcome
  noise, trap noise) has its own stream derived from the dataset seed.
  Changing outcome noise or trap strength therefore leaves treatment
  assignments unchanged.
- **Potential outcomes** `Y(0)` and `Y(1)` are computed inside the generator
  for tests only and are never part of the public dataset.
- **Unvaried knobs:** randomized datasets use `b = 1`; trap datasets use
  `a = b = 1`; nonlinear datasets use `b = 1`.
- **Public metadata** also states each column's kind (`binary` or
  `continuous`).
- **Module layout:** `bench/public.py` is the only module agent and tool code
  may import. `load_truth` lives in `bench/truth.py`.

### Knobs (independent)

| Knob | Meaning | Benchmark levels |
|---|---|---|
| `a` | treatment-assignment strength | 0, 0.5, 1.0, 1.5 |
| `b` | confounder-outcome strength | 0.5, 1.0, 2.0 |
| `σ` | outcome noise | 0.5, 1.0, 2.0 |
| `λ_T` | treatment nonlinearity | 0, 1 |
| `λ_Y` | outcome nonlinearity | 0, 1 |
| `τ` | true ATE | 0, 0.5, 1, 2, −1 (cycled) |
| `γ` | trap strength | 0.5, 2.0 |
| `d` | outcome-only predictor strength | 1.0 (fixed) |

The values of `γ` and `d` were proposed by Claude with the first generator
commit and approved by the owner:

- **`d = 1.0`:** the outcome-only predictor adds variance `d² = 1` to the
  outcome, the same as a confounder index at `b = 1`. It never causes bias;
  adjusting for it only improves precision.
- **`γ = 0.5` (weak) and `2.0` (strong):** the trap column is
  `γ · (T + standardized Y)` plus unit-variance noise. At 0.5 the column is
  mostly noise; at 2.0 it tracks `T` and `Y` closely. A larger `γ` means a
  larger distortion if the column is adjusted for.

### The 30 datasets

| Scenario | Count | What varies |
|---|---|---|
| Randomized (`a = 0`) | 6 | `λ_Y` ∈ {0, 1} × `σ` (3) |
| Observed confounding, linear | 9 | `a` × `b` (3 × 3), `σ` rotated |
| Observed confounding, nonlinear | 9 | (`λ_T`, `λ_Y`) ∈ {(0,1), (1,0), (1,1)} × `a` (3), `σ` rotated |
| Post-outcome collider trap | 6 | `γ` (2) × `σ` (3), linear base |

5,000 rows per dataset, subject to practical runtime constraints. IDs
`ds_001..ds_030` are assigned by a seeded shuffle so the ID does not encode
the scenario.

### Public and private metadata

- **Public** (agent and both baselines): the CSV, the question, treatment and
  outcome names, column names with timing labels, row count.
- **Private** (evaluation code only): true ATE, generating parameters and
  seed, the true edge list, column roles derived from it, and three
  assumption flags derived from the knobs: `treatment_randomized`,
  `outcome_model_linear`, `propensity_logit_linear`.
- Public and private files live in separate directories. `agent/` and
  `tools/` may import only the public loader.

### Adjustment-set validity

A set is valid if it passes the backdoor criterion on the true graph: it
contains no descendant of `T`, and it d-separates `T` from `Y` once `T`'s
outgoing edges are removed. It is computed from the graph, not hard-coded.

| Scenario | Valid sets |
|---|---|
| Randomized | all 16 subsets, including the empty set |
| Observed confounding | the 4 sets containing `c1` and `c2`; `p` and `n` optional |
| Trap | `{c1, c2}` and `{c1, c2, p}`; any set with `k` is invalid |

### Validation

- **Deterministic unit tests:** reproducibility; `Y(1) − Y(0) = τ` for every
  row; knob independence; adjustment sets match the table above; assumption
  flags follow from the knobs; overlap (at least 90% of true propensities in
  [0.05, 0.95]) and balance (treated share in [0.2, 0.8]); isolation of
  private metadata.
- **Statistical checks** (separate pytest marker, fixed seeds, bounds from
  standard errors): forced-treatment simulation, oracle regression recovery,
  and the naive difference on randomized specs each land within 4 standard
  errors of `τ`.
- **Characterization only:** naive bias is stored as a diagnostic. No test
  requires any estimator to fail.
- If the overlap test fails for a spec, `a` is lowered for that spec, because
  overlap is an identification requirement.

## Why

The owner's stated reasoning, quoted.

**Choosing Design A:**

> It is simpler to implement and validate within one day, provides exact
> ground-truth ATEs, and lets me control confounding, noise, and nonlinearity
> independently. I prefer a smaller, reliable benchmark that I can fully
> explain in an interview over a more complex generator with uncertain
> correctness.

**Removing the instrumental-variable scenario from v1:**

> It introduces additional identification assumptions and implementation
> complexity.

**Including the post-outcome trap scenario:**

> I want to test whether the agent can recognize and avoid inappropriate
> adjustment variables rather than blindly adjusting for every column.

**Allowing timing labels as public metadata:**

> Timing information represents realistic background knowledge and should be
> available consistently to the agent and baselines.

**Tracking outcome-model and propensity-model correctness separately:**

> Outcome regression and propensity-score weighting have different
> assumptions. A nonlinear treatment-assignment mechanism does not
> automatically invalidate linear outcome regression, and a nonlinear outcome
> does not automatically invalidate correctly specified propensity weighting.

> A method should be considered valid based on its actual assumptions, not an
> arbitrary scenario label.

**Deriving adjustment sets from the graph:**

> Do not assume that every valid adjustment set must contain all three
> covariates. Explicitly define the causal role of each generated variable and
> identify valid adjustment sets using the underlying causal graph and
> backdoor criterion.

**Validation approach:**

> Avoid brittle pytest tests that require every confounded dataset to show
> naive estimation error above a fixed threshold.

> Do not modify the data-generating parameters merely to force a particular
> estimator to fail.

**Scope:**

> This remains a one-day portfolio project. Do not introduce additional
> frameworks, unnecessary abstractions, new estimator families, or a larger
> benchmark unless essential for causal correctness.

**Approved without a stated reason** (to be filled in by the owner):

- 5,000 rows per dataset.
- The specific function forms for `nonlinear_T` and `nonlinear_Y` (proposed by
  Claude, approved by the owner).

## Tradeoffs

Raised by Claude during design review and accepted when the owner approved
the design.

- **No effect heterogeneity.** With a constant effect, the ATE and the effect
  on the treated coincide, so the benchmark cannot detect confusion between
  them.
- **No hidden confounding.** With the instrument scenario removed, every
  pre-treatment column is safe to adjust for. That agrees with the backdoor
  criterion only because these graphs have no hidden variables and no
  pre-treatment colliders.
- **A strong fixed pipeline can be valid everywhere.** Adjusting for every
  pre-treatment column can satisfy the adjustment criterion on all 30
  datasets, so the agent may match that baseline but not beat it on validity.
- **The trap tests use of timing knowledge, not discovery.** The data alone
  cannot reveal the collider's role.
- **Small sample.** 30 datasets with one seed each give wide uncertainty on
  any per-scenario rate.
- **Hand-written structures only.** Variety is limited to what the four
  scenarios encode.

## Constraints carried into later steps

Stated by the owner when approving the design.

1. Flexible estimation is not automatically guaranteed consistent. Its
   validity depends on adjustment-set correctness, overlap, and estimator
   assumptions.
2. Difference in means always uses the empty adjustment set.
3. Graph discovery provides statistical evidence, not proof of causal
   direction.
