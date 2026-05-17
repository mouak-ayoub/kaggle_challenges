# AGENTS.md - Nemotron Challenge

> Scope: these instructions apply only to files inside `nemotron_challenge/`.

## Session Start Protocol

Read these files at the start of each session:

1. `doc/LOCAL_PROJECT_MEMORY.md`
2. `doc/PROJECT_DECISION_LOG.md`
3. `doc/errors.md`
4. `README.md`

Use them for different purposes:

- `doc/LOCAL_PROJECT_MEMORY.md`: current project state, active work, recent milestones
- `doc/PROJECT_DECISION_LOG.md`: durable project-specific decisions
- `doc/errors.md`: non-trivial errors and the fixes or lessons learned
- `README.md`: broader architecture, roadmap, and challenge context

## Working Rules

- Use Occam's razor: start with the simplest coherent change that can explain or fix the observed problem. Add complexity only when the simple path fails with evidence.
- Treat Occam's razor as the guiding principle after repeated notebook iterations: when a config switch or fallback is no longer needed, remove it rather than documenting around it.
- After each milestone or important workflow change, run a short Occam audit before moving on: check active notebooks, scripts, and docs for stale switches, duplicated outputs, unclear artifact paths, and complex branches that can be replaced by one explicit path.
- Keep the audit principle-based, not checklist-based. Specific fixes from the current run are examples; the durable rule is to prefer one coherent path, remove speculative branches, and keep boundaries clear.
- Occam's razor does not mean deleting useful material or minimizing file count at any cost. It means choosing the minimum non-trivial structure that preserves evidence, keeps intent clear, and avoids speculative complexity.
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

## Notebook Policy

- Use one notebook for deep inspection of a small sample.
- Use separate batch notebooks for category-level summaries and error maps.
- Temporary prompt experiments may stay notebook-local at first.
- Move stable prompt templates, parsers, validators, and scoring helpers into `src/` later only when the user asks to refactor.

## Memory Update Policy

Update `doc/LOCAL_PROJECT_MEMORY.md` when:

- official data is downloaded
- a baseline is reproduced
- a new active category is chosen
- a milestone is completed

Update `doc/PROJECT_DECISION_LOG.md` when:

- a method decision becomes durable
- an approach is explicitly superseded
- there is evidence that a method is or is not sufficient

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
