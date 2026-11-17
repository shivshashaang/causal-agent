# 004 — Benchmark v2: a shared nonlinear feature

- Status: Accepted. The owner approved the design on 2026-11-16. The items
  under "Choices made during implementation" were made by Claude and have not
  been reviewed by the owner.
- Date: 2026-11-16
- Scope: a second set of 20 synthetic datasets, how it differs from
  Benchmark v1 (ADR 001), how it is validated, and how it is frozen.
- Frozen: the 20 specs have the digest
  `e06a2ef6e3c56eca661f692b87960ee2c11768d3cfef4690d1c47f6eb5606867`
  (`tests/bench/frozen/v2.json`). No LLM approach, and no doubly robust
  estimate, had been run on any v2 dataset when this record was committed.

## Context

On Benchmark v1 all three approaches were within tolerance on all 30
datasets, so v1 could not rank them. The agent reported a linear regression
on every dataset, which is misspecified on the six v1 datasets with a
nonlinear outcome, and it was accurate anyway.

The reason was measured after the v1 evaluation. A linear regression that
adjusts for the right columns is biased only when the part of the outcome a
straight line cannot explain is correlated with the part of the treatment a
straight line cannot explain. In v1 the treatment's nonlinear part
(`c1*c2` plus a step in `c1`) and the outcome's nonlinear part (`c2^2` plus
`tanh(c1 + 0.5)`) are nearly unrelated. On 1,000,000 rows per spec, linear
regression's bias on the nine v1 nonlinear specs was at most 0.020 standard
deviations of the outcome, against a tolerance of 0.10.

A small example of the failure v1 lacks: suppose the very young and the very
old are both more likely to be treated, and both have worse outcomes. Age
confounds the effect, but a straight line through age is flat, so adjusting
for age linearly removes none of the confounding.

The owner's request:

> I want a more discriminative benchmark, but not one engineered to make the
> agent win.

with these requirements for v2:

> - Stronger nonlinear relationships that can meaningfully challenge
>   linear-model assumptions.
> - More varied, non-round true ATEs.
> - A range of confounding and noise difficulty.
> - The same fundamental identification assumptions as v1.
> - No hidden confounding for now.

## Options

For the nonlinear form:

- **A. Scale up v1's functions.** Raise `lambda_T` and `lambda_Y` on the
  existing functions. Smallest change. The two functions stay nearly
  unrelated, so linear regression's bias stays small however large they are.
- **B. A shared nonlinear feature.** The treatment and the outcome both depend
  on the same curved function of the confounders. This is the textbook case
  in which linear adjustment fails. Adjusting for the observed confounders
  still identifies the effect; only the functional form is hard.
- **C. Effects that vary between units.** Make the treatment effect depend on
  the covariates. A linear regression then estimates a weighted average that
  differs from the ATE. This changes how the true ATE is defined and computed,
  so it does not keep "the same fundamental identification assumptions as
  v1".

## Decision

**Option B.** The owner: "Approve Option B: the shared nonlinear feature."

### What changes in the generator

- A spec has a new field `nonlinear_family`: `v1` (the default, the original
  functions) or `shared`.
- In the shared family both indices use one function of the two confounders:

      s(C) = (c1^2 - 1) / sqrt(2) + c1 * c2

      h(C) = standardize(linear(C) + lambda_T * s(C))     treatment index
      g(C) = standardize(linear(C) + lambda_Y * s(C))     outcome index

  `s` has mean zero and is uncorrelated with `c1`, with `c2` and with the
  linear index, so no straight line through the confounders can stand in for
  it. The standardizing constants come from the same fixed reference sample
  as in v1.
- Everything else is the v1 generator: the same variables, the same structural
  equations for `T`, `Y` and the trap column, a constant treatment effect, the
  same public schema and timing labels, 5,000 rows per dataset.

### The 20 datasets

| Scenario | Datasets | What varies |
|---|---|---|
| Randomized | 2 | one linear outcome, one curved outcome |
| Confounded, linear | 4 | `a` in {0.5, 1.5} x `b` in {0.5, 2.0} |
| Confounded, nonlinear, outcome side only | 2 | `a` in {0.5, 1.5} |
| Confounded, nonlinear, treatment side only | 2 | `a` in {0.5, 1.5} |
| Confounded, nonlinear, both sides | 6 | `a` in {0.5, 1.0, 1.5} x `b` in {0.5, 2.0} |
| Post-outcome trap | 4 | `gamma` in {0.5, 2.0} x `sigma` in {0.5, 2.0}, linear base |

IDs are `ds_101` to `ds_120`, assigned by a seeded shuffle. Outcome noise is
rotated by position, as in v1.

True effects: a seeded draw from Uniform(-2, 2), rounded to two decimals. Two
datasets, picked by a seeded draw, have their effect set to exactly zero.

The frozen specs. "Overlap share" is the share of true propensities in
[0.05, 0.95] in the generated dataset. The last column is described under
"Recorded, not used".

| ID | Scenario | True effect | a | b | sigma | lambda_T | lambda_Y | gamma | Overlap share | Treated share | Outcome SD | Linear regression bias (SD) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ds_101` | nonlinear, treatment side | 1.48 | 1.5 | 1 | 2 | 1 | 0 | 0 | 0.947 | 0.472 | 2.61 | +0.002 |
| `ds_102` | post-outcome trap | 0.89 | 1 | 1 | 0.5 | 0 | 0 | 0.5 | 0.995 | 0.502 | 1.65 | +0.001 |
| `ds_103` | post-outcome trap | -1.64 | 1 | 1 | 2 | 0 | 0 | 2 | 0.996 | 0.505 | 2.46 | +0.002 |
| `ds_104` | nonlinear, both sides | 0.27 | 1.5 | 2 | 0.5 | 1 | 1 | 0 | 0.945 | 0.473 | 2.42 | +0.502 |
| `ds_105` | randomized | 1.74 | 0 | 1 | 0.5 | 0 | 0 | 0 | 1.000 | 0.497 | 1.75 | +0.001 |
| `ds_106` | post-outcome trap | 0 | 1 | 1 | 2 | 0 | 0 | 0.5 | 0.997 | 0.514 | 2.48 | +0.002 |
| `ds_107` | nonlinear, both sides | -1.97 | 0.5 | 2 | 1 | 1 | 1 | 0 | 0.997 | 0.503 | 2.55 | +0.232 |
| `ds_108` | confounded, linear | -0.26 | 0.5 | 0.5 | 0.5 | 0 | 0 | 0 | 1.000 | 0.505 | 1.20 | +0.001 |
| `ds_109` | nonlinear, both sides | -0.55 | 1 | 0.5 | 1 | 1 | 1 | 0 | 0.979 | 0.492 | 1.50 | +0.157 |
| `ds_110` | nonlinear, outcome side | 1.35 | 1.5 | 1 | 1 | 0 | 1 | 0 | 0.948 | 0.501 | 1.97 | +0.000 |
| `ds_111` | nonlinear, both sides | -1.03 | 1 | 2 | 2 | 1 | 1 | 0 | 0.979 | 0.479 | 2.91 | +0.318 |
| `ds_112` | nonlinear, outcome side | -1.83 | 0.5 | 1 | 0.5 | 0 | 1 | 0 | 1.000 | 0.512 | 1.68 | +0.002 |
| `ds_113` | post-outcome trap | 0 | 1 | 1 | 0.5 | 0 | 0 | 2 | 0.996 | 0.487 | 1.53 | +0.001 |
| `ds_114` | nonlinear, treatment side | 0.7 | 0.5 | 1 | 1 | 1 | 0 | 0 | 0.998 | 0.499 | 1.76 | +0.001 |
| `ds_115` | confounded, linear | 1.18 | 1.5 | 2 | 2 | 0 | 0 | 0 | 0.955 | 0.483 | 3.25 | +0.000 |
| `ds_116` | confounded, linear | 0.12 | 1.5 | 0.5 | 1 | 0 | 0 | 0 | 0.953 | 0.498 | 1.51 | +0.000 |
| `ds_117` | nonlinear, both sides | 1.34 | 0.5 | 0.5 | 0.5 | 1 | 1 | 0 | 0.999 | 0.484 | 1.43 | +0.100 |
| `ds_118` | confounded, linear | 0.41 | 0.5 | 2 | 1 | 0 | 0 | 0 | 1.000 | 0.496 | 2.46 | +0.001 |
| `ds_119` | nonlinear, both sides | -0.9 | 1.5 | 0.5 | 2 | 1 | 1 | 0 | 0.950 | 0.463 | 2.33 | +0.131 |
| `ds_120` | randomized, curved outcome | -1.41 | 0 | 1 | 1 | 0 | 1 | 0 | 1.000 | 0.500 | 1.89 | +0.003 |

### Validation

These are tests in `tests/bench/`, written before any approach ran on v2.
None of them requires an estimator to fail.

1. **v1 is unchanged.** `tests/bench/test_frozen.py` compares the v1 spec
   digest and a fingerprint of each v1 dataset with a record written from the
   generator before any v2 change.
2. **The true effect is exact.** For every row, `y1 - y0` equals the spec's
   true effect.
3. **Overlap, by the v1 rule.** At least 0.90 of true propensities lie in
   [0.05, 0.95], and the treated share is between 0.2 and 0.8.
4. **Observed columns identify the effect.** The set of all pre-treatment
   columns passes the backdoor check on the true graph, the empty set passes
   only when treatment is randomized, and any set with the trap column fails.
5. **A model with the right curve recovers the effect.** A regression given
   the true `g(C)` lands within four standard errors of the true effect on
   every dataset. Nothing hidden is needed.
6. **Same public interface as v1.** Same columns, question text and timing
   labels.
7. **Frozen.** The v2 spec digest and a fingerprint of each v2 dataset are
   recorded in `tests/bench/frozen/v2.json`.

### Recorded, not used

For each spec, `bench/design.py` records how far a linear regression on all
pre-treatment columns would land from the true effect with unlimited data
(approximated on the fixed 1,000,000-row reference sample). The owner:

> Record the population bias of linear regression but do not use it as a
> dataset selection gate.

It is a property of the data-generating process, not a result of any
approach. No dataset was added, removed or changed because of it. The tests
assert it only where theory fixes it at zero: a linear outcome or a randomized
treatment.

### Freeze

The owner:

> Freeze generator specifications, validation tests, and ADR 004 before
> running any LLM evaluation.

> Do not tune datasets based on which approach performs better.

The specs, the tests above and this record are committed before any LLM call
on v2. After that the specs do not change: the results are reported as they
come out, from one run.

### Benchmark v1 is preserved

The owner: "Keep the original 30 datasets, results, scores, plots, and
failure analysis." and "Do not overwrite or silently replace the original
results." The v1 specs, datasets and published results are untouched. The v1
public files regenerated after the generator change are byte-identical to the
files the v1 evaluation used.

## Why

The owner's stated reasons and constraints, in the owner's words:

- "I want a more discriminative benchmark, but not one engineered to make the
  agent win."
- "Do not tune scenarios based on whether the agent wins. Freeze the v2
  specifications before running the approaches."
- "The same fundamental identification assumptions as v1." and "No hidden
  confounding for now."
- "Preserve observed-confounder identifiability and overlap requirements."

Approved without a stated reason (Claude recommended each; the owner
approved):

- Choosing option B over options A and C.
- The size of 20 datasets and the scenario composition.
- Including two zero-effect datasets.

## Choices made during implementation

Made by Claude. None was chosen by the owner.

- **The function `s`.** A U-shape in one confounder plus an interaction. It
  was picked because it is uncorrelated with the linear index, so the
  nonlinear and linear parts of the confounding are separate.
- **One measurement was made before the design was approved.** To check that
  option B would do what the owner asked, linear regression and logistic
  weighting were run on 1,000,000-row samples of 27 knob combinations of this
  family. Linear regression's bias was at most 0.003 SD with a one-sided
  nonlinearity and 0.06 to 0.48 SD with both sides. The owner saw these
  numbers before approving. No LLM and no doubly robust estimate was run on
  the family. Only this one family was tried; nothing was searched over.
- **Master seed 2**, IDs from 101, and the same noise rotation rule as v1.
  The seed was fixed before the specs were printed and was not changed.
- **True effects by seeded draw**, range (-2, 2), two decimals. The two
  zero-effect datasets were also picked by a seeded draw. The draw put both
  on trap datasets (`ds_106`, `ds_113`). It was not redrawn.
- **Where the bias is recorded.** In `bench/design.py` and in the table above,
  computed on demand, not in each dataset's private truth file. Putting it in
  the truth file would have meant a 1,000,000-row simulation every time a
  dataset is generated.
- **Fingerprints, not byte hashes, in the tests.** `exp` and `tanh` may differ
  in the last bit between machines, which would change a hash of the CSV
  bytes without changing the data in a way that matters. The tests hash the
  treatment column and the public metadata exactly and compare three weighted
  sums of each continuous column to a relative tolerance of 1e-9.
- **The trap scenario keeps a linear base**, as in v1, so the existing spec
  rule for that scenario is unchanged.

## Tradeoffs

- **v2 targets one failure.** It is built around the case that breaks linear
  adjustment. It says nothing about hidden confounding, effects that vary
  between units, or measurement error.
- **The one-sided nonlinear datasets do not bias linear regression.** With a
  curved outcome and a logit-linear treatment, linear regression is
  misspecified and still unbiased here (recorded bias at most 0.002 SD). Those
  four datasets separate "invalid" from "inaccurate"; they do not punish a
  linear model.
- **Six datasets carry most of the difficulty.** Only the "both sides"
  datasets have a recorded linear-regression bias at or above the 0.10
  tolerance (0.100 to 0.502 SD). With six such datasets, a difference of one
  or two between approaches is not strong evidence.
- **Both zero-effect datasets are trap datasets.** A zero effect is not
  tested where the functional form is hard.
- **The trap is still easy.** The timing label marks the trap column, as in
  v1.
- **The recorded bias is an approximation.** It comes from a finite reference
  sample; its standard error is between 0.001 and 0.005 in outcome units.
- **Having measured linear-model bias before approval is a form of looking.**
  It was a check of the design goal the owner set, on estimators, not on
  approaches. It still means the family was chosen knowing it challenges a
  linear model.
