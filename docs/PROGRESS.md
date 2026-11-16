# Progress

Read this at the start of every session. Updated after every commit.

Last updated: 2026-11-16

## Done

- Design for the synthetic generator approved and recorded in
  `docs/decisions/001-synthetic-generator.md`.
- Step 0 infrastructure, merged into `main` through PR #1: installable
  `src/causal_agent` package, ruff and pytest config, GitHub Actions
  workflow, one smoke test, this file, ADR 001. The unused `main.py` stub was
  deleted after confirming nothing referenced it.
- Step 1, part 1, on branch `step-1-generator`: `bench/spec.py`
  (`DatasetSpec`, `Scenario`, scenario-versus-knob validation) and
  `bench/equations.py` (indices `h` and `g`, propensity, potential outcomes,
  trap column, `simulate`), with tests in `tests/bench/`.

- Step 1, part 2: `bench/suite.py` with `benchmark_specs()`, the 30 specs
  from ADR 001. Tests cover the grid, seeds, ID shuffling, overlap and
  balance, oracle-regression recovery of the true ATE on all 30 datasets, and
  the naive difference on the 6 randomized ones. All 30 specs met the overlap
  rule without lowering any knob (lowest share of propensities inside
  [0.05, 0.95] was 93.6%).

## Verification status

- GitHub Actions CI passed on PR #1 (lint, format check, 1 test) before the
  merge.
- Step 1: local checks pass before every commit (`uv run ruff check .`,
  `uv run ruff format --check .`, `uv run pytest`). CI has not run on
  `step-1-generator` yet because the branch has not been pushed.

## Next

1. Step 1, part 3: the true causal graph, column roles, and the backdoor
   adjustment-set check.
2. Step 1, part 4: public and private metadata separation and the loaders.
3. Checkpoint 1: stop for the owner's review once the generator and metadata
   are complete.

Do not start the agent, tools, or evaluation pipeline until step 1 is done.

## Working rules set by the owner

- Never commit directly to `main`. Work on a branch per step.
- `CLAUDE.md` stays local and untracked (listed in `.git/info/exclude`).
  Never include it in a commit.
- No Claude or AI co-author attribution in commit messages.
- Do not push or merge without the owner's say-so.

## Approved decisions not yet recorded as ADRs

Record each one when the part it belongs to is built.

- **ATE tolerance:** an estimate is correct if it is within 0.10 standard
  deviations of `Y` of the true ATE. Also report mean absolute error and
  sensitivity at 0.05, 0.10 and 0.20 SD. The owner has not yet stated a
  reason for 0.10; ask before writing the ADR.
- **Method validity:** no single "correct" method. Record separately:
  estimation accuracy, adjustment-set validity, outcome-model correctness,
  propensity-model correctness, estimator validity, refutation outcome.
- **Fairness:** the agent, the no-tools LLM and the fixed pipeline get
  comparable public information. Ground truth is inaccessible to all three.
  The fixed pipeline must not be artificially weakened.
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
- What the no-tools LLM baseline sees (proposal: one deterministic summary of
  the public CSV, shared with the agent). Decide at step 4.
- `estimate_effect` must log the adjustment set and estimator family so
  validity can be scored. Carry into step 3.
