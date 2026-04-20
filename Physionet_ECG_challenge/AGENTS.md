# AGENTS.md - Physionet ECG Challenge

> Scope: these instructions apply only to files inside `Physionet_ECG_challenge/`.

## Session Start Protocol

Read these files at the start of each session:
1. `LOCAL_PROJECT_MEMORY.md`
2. `PROJECT_DECISION_LOG.md`
3. `README.md`

Use them for different purposes:
- `LOCAL_PROJECT_MEMORY.md`: current project state, active work, recent milestones
- `PROJECT_DECISION_LOG.md`: durable project-specific technical decisions
- `README.md`: broader architecture and pipeline context

## Working Rules

- Keep reusable logic in `src/`.
- Keep notebook cells focused on visualization, step-by-step exploration, and temporary diagnostics.
- Do not duplicate stable method logic in notebooks once it has been extracted to `src/`.
- Prefer `@dataclass` configs for grouped parameters.
- Prefer small reusable functions over notebook-only helper sprawl.
- Prefer `scikit-image`, `opencv`, and `numpy` for classical CV work.

## Notebook Policy

- Keep notebooks step by step with short markdown sections.
- The unitary notebook is for one image and deeper inspection.
- Batch notebooks are for seeing method behavior across multiple samples.
- Temporary debug plots and hypothesis-testing cells may stay notebook-local.
- Stable selection or geometry logic should be moved to `src/` once validated.

## Memory Update Policy

Update `LOCAL_PROJECT_MEMORY.md` when:
- a milestone is completed
- a new insight is confirmed
- the active focus changes

Keep those updates concise and local to the existing sections.

Update `PROJECT_DECISION_LOG.md` when:
- a method decision becomes durable
- a previous approach is explicitly superseded
- there is enough evidence to state why a method is or is not sufficient

Decision-log entries should contain:
- context
- decision
- evidence
- consequence
- status

## Cross-Project Lessons

Reusable lessons that should not live only in this project belong in:
- `../TRANSFERABLE_LESSONS.md`

Examples:
- what made a CV method insufficient in practice
- debugging patterns that transfer across projects
- refactoring patterns for notebooks into reusable code

## Current Pipeline Goal

`hard ECG image -> canonical clean page -> lead signals -> CSV`
