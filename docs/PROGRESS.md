# Progress

Read this at the start of every session. Updated after every commit.

Last updated: 2026-11-16

## Done

- Design for the synthetic generator approved and recorded in
  `docs/decisions/001-synthetic-generator.md`.
- Step 0 infrastructure, committed on branch `step-0-infra` (not pushed, not
  merged): installable `src/causal_agent` package, ruff and pytest config,
  GitHub Actions workflow, one smoke test, this file, ADR 001. The unused
  `main.py` stub was deleted after confirming nothing referenced it.

## Verification status

- Local checks pass: `uv sync --locked`, `uv run ruff check .`,
  `uv run ruff format --check .`, `uv run pytest` (1 test).
- GitHub Actions CI has **not** been verified. The workflow has never run
  because the branch has not been pushed.

## Next

1. Owner reviews the step 0 commit, then decides when to push and merge.
   Confirm CI passes on GitHub after the push.
2. Step 1, first commit: `DatasetSpec` and the structural equations in
   `src/causal_agent/bench/`, with tests.
3. Step 1, later commits: scenario generation (the 30 specs), then public and
   private metadata separation, then the backdoor adjustment-set check.

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
- Numeric values for trap strength `γ` (two levels) and outcome-only
  predictor strength `d` are not fixed yet. Propose them with the generator
  code.
- What the no-tools LLM baseline sees (proposal: one deterministic summary of
  the public CSV, shared with the agent). Decide at step 4.
- `estimate_effect` must log the adjustment set and estimator family so
  validity can be scored. Carry into step 3.
