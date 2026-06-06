# Strategy Escalation Protocol

This file prevents the project from staying conservative when the score is stagnant.

## Goal

Reach at least `0.70` public score.

Current known pattern:

- raw-answer SFT on partial data reached about `0.62`
- full raw SFT dropped to about `0.54`
- boxed/private final-answer SFT dropped to about `0.53`
- deterministic/procedural templates helped some families
- cipher and bit manipulation remain hard families

## Ladder File Policy

Treat `doc/POST_TRAINING_METHOD_LADDER.md` as an idea backlog, not a mandatory checklist.

Do:

- read it when escalation triggers;
- extract candidate methods relevant to the current failure mode;
- explain which idea is selected, postponed, or rejected;
- promote an idea into `PROJECT_DECISION_LOG.md` only after evidence exists.

Do not:

- run ladder ideas in order without reasoning;
- hardcode ladder ideas into this file as permanent rules;
- keep retrying a ladder idea after evidence shows it is insufficient.

## Stagnation Rules

### Level 0: Normal iteration

Use when only one run failed or evidence is incomplete.

Allowed action:

- audit artifacts
- compare by family
- fix extraction or packaging
- choose best checkpoint
- make one small method change

### Level 1: Method adjustment

Trigger when two serious runs fail to beat the best baseline or show the same weak families.

Required action:

- stop scaling the same loss/format/hyperparameter path
- consult the method ladder
- propose one new supervision signal
- preserve the successful parts of the current pipeline

Examples are illustrative, not mandatory:

- keep deterministic templates for easy families
- add short traces only for weak families
- use generated-eval family accuracy to select checkpoints

### Level 2: Strategic pivot

Trigger when score stays below the best baseline for three serious runs, or when local loss improves but public score stays flat or drops.

Required action:

- do not propose another simple rank/LR/epoch change as the main idea
- consult the method ladder
- propose three alternatives:
  1. conservative
  2. medium-risk
  3. radical

At least one proposal must change the learning signal, not only the hyperparameters.

Examples are illustrative, not mandatory:

- STaR-like bootstrapped SFT
- vLLM offline sampling for cipher and bit only
- verifier-filtered traces
- category-specific mixture
- train only weak families plus replay of strong-family templates
- separate adapters by family if allowed and practical
- shorten targets to final-answer-only for sampling, then reconstruct clean traces

### Level 3: Radical pivot

Trigger when we are stuck below `0.60` for multiple runs or below `0.70` after several method families.

Required action:

- challenge the current assumption
- consult the method ladder
- propose a method that may abandon the current main path

Candidate pivots are illustrative, not mandatory:

- stop global SFT and train only weak-family targeted data
- use deterministic templates for solved families and vLLM/rejection sampling only for cipher + bit
- generate answer-only candidates, filter by exact answer, then reconstruct clean traces
- build a verifier/solver for synthetic data quality before training
- try a smaller/faster model for synthetic data exploration, then transfer data to Nemotron LoRA
- use family-balanced curriculum instead of raw train distribution
- compare adapter checkpoints by hidden-family proxy, not eval loss

## Family Policy

### Numeral, unit conversion, gravity, equation

Default:

- use deterministic or semi-deterministic templates if they already improve performance
- do not spend expensive offline sampling unless evidence changes

### Cipher

Default:

- use vLLM offline sampling only for cipher rows
- generate short final-answer candidates first
- validate against gold answer
- keep accepted rows
- reconstruct clean trace format
- deduplicate by question and answer

### Bit manipulation

Default:

- use vLLM offline sampling only for bit rows
- prefer lower temperature than cipher
- validate exact final answer
- reconstruct compact step-by-step trace
- balance with replay data so the model does not forget easy families

## Required Output When Escalating

When this protocol triggers, Codex must output:

1. Why escalation triggered.
2. What method is now considered insufficient.
3. What evidence supports that decision.
4. Whether the method ladder was consulted.
5. Conservative next option.
6. Medium-risk next option.
7. Radical next option.
8. Recommended next run name.
9. Exact files/notebook cells to change.
