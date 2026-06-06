# Codex Task Templates

Use these templates when starting a focused task. They are prompts, not permanent project decisions.

## Post-run update

Goal: update project records after a completed Nemotron run.

Context:

- `doc/AGENT_RUN_PROTOCOL.md`
- `doc/EXPERIMENT_CHECKLIST.md`
- `doc/SUBMISSION_TRACKING.md`
- `doc/LOCAL_PROJECT_MEMORY.md`
- `doc/PROJECT_DECISION_LOG.md`

Constraints:

- do not invent missing values;
- use `unknown` when a value is unavailable;
- do not commit files under `data/outputs/`;
- do not rewrite the method ladder as a checklist.

Done when:

- run status is classified;
- next action is explicit;
- required docs are updated or skipped with a reason;
- final response lists files changed and commands run.

## Stagnation escalation

Goal: choose the next method after score stagnation.

Context:

- `doc/AUTONOMY_POLICY.md`
- `doc/STRATEGY_ESCALATION.md`
- `doc/POST_TRAINING_METHOD_LADDER.md`
- `doc/LOCAL_PROJECT_MEMORY.md`
- `doc/PROJECT_DECISION_LOG.md`

Constraints:

- do not propose only rank, learning-rate, or epoch changes;
- do not treat the ladder as a fixed checklist;
- preserve deterministic templates for solved families unless evidence says otherwise;
- if proposing a new method outside the ladder, label it `new proposal`.

Done when:

- escalation level is identified;
- failed assumption is stated;
- conservative, medium-risk, and radical options are proposed;
- one recommended next run is selected;
- a small diagnostic is proposed when uncertainty is high;
- exact notebook or file changes are listed.

## Notebook or workflow edit

Goal: modify a notebook or workflow while preserving proven local patterns.

Context:

- `AGENTS.md`
- `doc/PRECEDENT_FIRST_EDITING.md`
- relevant notebook section or paired `.py` file
- the closest working predecessor notebook/script
- `doc/errors.md` if the task is a bug fix

Before editing, write:

- precedent used;
- intended delta;
- invariants that must be preserved;
- intentional deviations, if any.

Constraints:

- prefer paired `.py` or notebook-aware tooling;
- keep active configuration in one top cell;
- do not scatter constants across cells;
- do not invent broad fallback machinery when a previous simple pattern works;
- validate the notebook if edited directly.

Done when:

- notebook remains valid;
- changed cells are listed;
- same-shape behavior inherited from the precedent is listed;
- intentional deviations are listed with reasons;
- commands or validation steps are reported.
