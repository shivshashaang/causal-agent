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

- Step 1, part 3: `bench/graph.py` with the true causal graph built from a
  spec, column roles derived from the graph, and the backdoor adjustment-set
  check (networkx d-separation). Valid sets match the hand-derived table in
  ADR 001 on all 30 specs: 16 for randomized, 4 for confounded, 2 for the
  trap. Extra tests on M-bias, mediator and chain graphs show the checker is
  a real backdoor check, not a lookup.

- Step 1, part 4a: `bench/public.py` with `PublicDataset`, `ColumnInfo`,
  `Timing` and `load_public`. It is the only bench module agent and tool
  code may import. Dataset IDs are validated so an ID cannot be a path into
  the truth directory. A test in a fresh interpreter confirms importing it
  loads no private module.

- Step 1, part 4b: `bench/truth.py` with `GroundTruth`, `Assumptions`,
  `Diagnostics`, `build_truth` and `load_truth`. Truth uses public column
  names. Valid adjustment sets are computed from the graph and stored; the
  three assumption flags are derived from the knobs.

- Step 1, part 4c: `bench/generate.py` with `generate`, `write_dataset` and
  `write_benchmark`. Covariates get neutral names `x1..x4` in a seeded random
  order. Public files go to `<root>/public/`, truth to `<root>/truth/`.
  Isolation tests: identical public schema in all 30 datasets, no private
  words in public metadata, exact file round trip, byte-identical rewrite,
  and `tests/test_isolation.py`, an import scan that forbids `agent/` and
  `tools/` from importing private modules (vacuous until those packages
  exist).

**Checkpoint 1 reached: generator and metadata complete. Waiting for the
owner's review before starting the fixed-pipeline baseline.**

## Verification status

- GitHub Actions CI passed on PR #1 (lint, format check, 1 test) before the
  merge.
- Step 1: local checks pass before every commit (`uv run ruff check .`,
  `uv run ruff format --check .`, `uv run pytest`). CI has not run on
  `step-1-generator` yet because the branch has not been pushed.

## Next

1. Owner reviews checkpoint 1 and decides whether to push `step-1-generator`.
2. Step 2: fixed-pipeline baseline (options and a recommendation first; it
   must not be weakened to favour the agent).
3. Step 3: `discover_graph`, `estimate_effect`, `refute_estimate`, one tool
   per commit.
4. Checkpoint 2: stop once tools and baselines are complete.

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
- Where the generated benchmark lives on disk, and whether it is committed or
  regenerated from seeds. Nothing writes it outside tests yet. Decide when
  the evaluation script is built.
- `load_truth` lives in `bench/truth.py` next to the model, not in an `eval`
  package as ADR 001's interface sketch suggested. The import scan enforces
  the boundary either way.
- What the no-tools LLM baseline sees (proposal: one deterministic summary of
  the public CSV, shared with the agent). Decide at step 4.
- `estimate_effect` must log the adjustment set and estimator family so
  validity can be scored. Carry into step 3.
