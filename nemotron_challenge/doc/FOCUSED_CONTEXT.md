# Focused Context

Use task-specific context instead of reading every project note for every change.

## Normal code edits

Read:

- `AGENTS.md`
- the file being changed
- nearby helpers

## Notebook or workflow edits

Read:

- `AGENTS.md`
- `doc/PRECEDENT_FIRST_EDITING.md`
- the notebook/script being changed
- the closest working predecessor notebook/script
- `doc/errors.md` if the task is a bug fix

Do not invent a new workflow shape when a known-good local pattern exists. If the edit intentionally changes the pattern, state the reason and the invariant being preserved.

## Run result reviews

Read:

- `doc/AGENT_RUN_PROTOCOL.md`
- `doc/EXPERIMENT_CHECKLIST.md`
- `doc/SUBMISSION_TRACKING.md`
- `doc/LOCAL_PROJECT_MEMORY.md`
- `doc/PROJECT_DECISION_LOG.md`

## Strategy reviews

Read:

- `doc/AUTONOMY_POLICY.md`
- `doc/STRATEGY_ESCALATION.md`
- `doc/POST_TRAINING_METHOD_LADDER.md`
- `doc/LOCAL_PROJECT_MEMORY.md`
- `doc/PROJECT_DECISION_LOG.md`

## Known error reviews

Read:

- `doc/errors.md`
- the failing file or notebook section

The goal is to keep useful context while avoiding unrelated history in every task.
