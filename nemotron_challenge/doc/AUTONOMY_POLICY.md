# Autonomy Policy

This project wants bounded autonomy, not mechanical execution.

## Deterministic Areas

Be strict and repeatable for evidence preservation and bookkeeping:

- preserve raw model outputs before answer extraction;
- avoid inventing missing values; use `unknown` when needed;
- update run and submission tracking when a run or score is discussed;
- classify run status clearly;
- write one explicit next action.

## Creative Areas

Be proposal-driven for research strategy:

- do not blindly follow the method ladder in order;
- do not restrict proposals to the existing ladder;
- if evidence suggests a better method, propose it explicitly;
- if proposing a new method, explain why it may beat the current ladder option;
- mark it as `new proposal` until a diagnostic or run supports it;
- promote it to `PROJECT_DECISION_LOG.md` only after evidence exists.

## Precedent Boundary

Be creative about research methods, not careless about working infrastructure.

For notebooks, data builders, training flows, submission flows, and artifact handling:

- inspect the closest working predecessor before editing;
- reuse the same operational pattern by default;
- state any intentional deviation and why it is safer or better;
- do not introduce broad fallback machinery when a previous simple pattern works;
- do not silently change persistence, resume, archive, or output-location behavior.

The goal is not to freeze the code. The goal is to make innovation explicit and keep proven workflow invariants intact.

## When Strategy Is Uncertain

Report:

1. what the current protocol suggests;
2. what alternative is proposed, if any;
3. why the alternative may be better;
4. what small diagnostic would validate or reject it.

The goal is disciplined research assistance, not deterministic method choice.
