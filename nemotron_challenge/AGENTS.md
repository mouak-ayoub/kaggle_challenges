# AGENTS.md - Nemotron Challenge

> Scope: these instructions apply only to files inside `nemotron_challenge/`.

## Session Start Protocol

Always read these small router/state files first:

1. `doc/FOCUSED_CONTEXT.md`
2. `doc/AUTONOMY_POLICY.md`
3. `doc/LOCAL_PROJECT_MEMORY.md`
4. `doc/PROJECT_DECISION_LOG.md`

Then read task-specific files according to `doc/FOCUSED_CONTEXT.md`. Do not load every project note for every task.

Use them for different purposes:

- `doc/FOCUSED_CONTEXT.md`: decides which extra docs are relevant for the current task.
- `doc/AUTONOMY_POLICY.md`: bounded autonomy; strict for bookkeeping and flexible for strategy.
- `doc/PRECEDENT_FIRST_EDITING.md`: required before notebook/workflow edits; reuse known-good patterns before inventing new ones.
- `doc/AGENT_RUN_PROTOCOL.md`: post-run, score, bundle, and submission bookkeeping.
- `doc/STRATEGY_ESCALATION.md`: guidance for score stagnation and method changes.
- `doc/POST_TRAINING_METHOD_LADDER.md`: idea backlog and method proposal source; do not treat it as a fixed checklist.
- `doc/EXPERIMENT_CHECKLIST.md`: current experiment checklist and archive expectations.
- `doc/SUBMISSION_TRACKING.md`: Kaggle submission archive and scoring registry expectations.
- `doc/errors.md`: non-trivial errors and the fixes or lessons learned.
- `README.md`: broader architecture, roadmap, and challenge context.

## Working Rules

- Be critical of experiment proposals, even when the same idea is raised repeatedly. Separate what was exactly tested from what only sounds similar; say clearly when evidence supports an idea, when evidence contradicts it, and when the right answer is "not tested yet." Do not loosely accept a proposal just because it is plausible, and do not reject a proposal using evidence from a materially different experiment.
- Use Occam's razor: start with the simplest coherent change that can explain or fix the observed problem. Add complexity only when the simple path fails with evidence.
- Treat Occam's razor as the guiding principle after repeated notebook iterations: when a config switch or fallback is no longer needed, remove it rather than documenting around it.
- After each milestone or important workflow change, run a short Occam audit before moving on: check active notebooks, scripts, and docs for stale switches, duplicated outputs, unclear artifact paths, and complex branches that can be replaced by one explicit path.
- Keep the audit principle-based, not checklist-based. Specific fixes from the current run are examples; the durable rule is to prefer one coherent path, remove speculative branches, and keep boundaries clear.
- Occam's razor does not mean deleting useful material or minimizing file count at any cost. It means choosing the minimum non-trivial structure that preserves evidence, keeps intent clear, and avoids speculative complexity.
- Occam's razor does not mean staying conservative forever. If repeated runs are stagnant, use `doc/STRATEGY_ESCALATION.md` to propose method changes, not only hyperparameter changes.
- Use precedent-first editing for notebooks and workflow code: inspect the nearest working predecessor, reuse the same shape by default, and state any intentional deviation before or during the edit.
- Do not solve recurring workflow misses by adding dozens of tiny rules. Add or refine a precedent/invariant rule instead.
- Keep code simple by default. Do not add broad edge-case handling, fallback branches, or clever abstractions before the project has actually hit that case.
- Keep notebook configuration in one small top cell until there is a clear reason to move it elsewhere. Do not scatter model names, LoRA settings, path constants, and generation settings across many cells.
- Prefer one clear path that works over several partially supported paths. Add complexity only after a concrete failure or repeated need.
- For the current phase, keep the active workflow in notebooks. Do not refactor notebook logic into `src/` unless the user explicitly asks for that step.
- Keep scripts thin if scripts are needed later.
- Keep notebooks focused on the immediate experiment path.
- When shared helper code changes in the Colab notebook, mirror the same method in the local notebook when applicable. Keep only environment-specific install/load/path cells different.
- Treat problem type/category as a first-class dimension in reports.
- Save raw model completions before extracting answers.
- Track answer extraction failures separately from reasoning failures.
- Treat `data/outputs/` as local ignored result state. It may contain run bundles, dashboards, submission archives, and `data/outputs/submissions/submissions_registry.csv`. Do not commit it, and do not create tracked duplicate result registries under `doc/`. Summarize durable lessons in docs instead.

## Notebook Policy

- Use one notebook for deep inspection of a small sample.
- Use separate batch notebooks for category-level summaries and error maps.
- Temporary prompt experiments may stay notebook-local at first.
- Move stable prompt templates, parsers, validators, and scoring helpers into `src/` later only when the user asks to refactor.
- Before editing any notebook workflow cell, follow `doc/PRECEDENT_FIRST_EDITING.md`.
- Do not manually edit `.ipynb` JSON unless necessary. Prefer paired `.py` or notebook-aware tooling. If editing directly, validate the notebook JSON before stopping.
- After editing, report the precedent used, what stayed same-shape, what intentionally changed, and what validation ran.

## Memory Update Policy

Update `doc/LOCAL_PROJECT_MEMORY.md` when:

- official data is downloaded
- a baseline is reproduced
- a new active category is chosen
- a milestone is completed
- an experiment result changes the next recommended action

Update `doc/PROJECT_DECISION_LOG.md` when:

- a method decision becomes durable
- an approach is explicitly superseded
- there is evidence that a method is or is not sufficient
- a strategy escalation changes the project direction

Decision-log entries should contain:

- context
- decision
- evidence
- consequence
- status

## Cross-Project Lessons

Reusable lessons that should not live only in this project belong in:

- `../TRANSFERABLE_LESSONS.md`

## Current Pipeline Goal

`official data -> baseline traces -> LoRA training -> adapter validation -> submission.zip`
