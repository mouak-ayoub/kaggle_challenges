# Post-Training Method Ladder

Last updated: 2026-05-31

This note turns the SFT -> rejection sampling -> RL discussion into a project rule. It is not a literature review and it is not proof that any one method will work on this Kaggle challenge.

The purpose is practical: when we are stuck, stop repeating the same kind of run and move to the next method class with a clear diagnostic.

## Source Anchors

- User-provided lecture: <https://www.youtube.com/watch?v=ebnX5Ur1hBk>
- Nathan Lambert, RLHF Book, Lecture 1: <https://rlhfbook.com/teach/course/lec1-chap1-3/>
- Nathan Lambert, RLHF Book, Lecture 2 slides: <https://rlhfbook.com/teach/course/lec2-chap4-5-9/slides.pdf>
- DeepSeek-R1 staged pipeline summary: <https://www.martinfowler.com/articles/deepseek-papers.html>

The Lambert lecture frames rejection sampling as: generate many completions, score them, select the best, then fine-tune with ordinary SFT on the curated set. It also contrasts rejection sampling with online RL/PPO and DPO. The DeepSeek-R1 summary describes a staged recipe: cold-start SFT, reasoning-oriented RL, rejection sampling plus additional SFT, then broader RL.

## Ladder

### 1. SFT / IFT

What it does:

- trains the model to imitate given completions
- is stable and simple
- teaches format and local response style well

Failure pattern:

- low loss but weak generated reasoning
- template imitation without solving
- more rows or bigger LoRA rank do not move public score

Project evidence:

- raw-answer SFT and v2 trace SFT produced clean training signals but did not beat the older `0.62` baseline.
- long cipher traces showed strong teacher-forced learning but failed generation.

When stuck:

- do not keep scaling the same SFT format.
- change the data-generation mechanism or objective.

### 2. Rejection Sampling + SFT

What it does:

- samples several completions per prompt from the model or a model variant
- scores or verifies each completion
- keeps only successful completions
- trains later with standard SFT on the accepted completions

Our adaptation:

- use official train rows with known `gold_answer`
- sample model-native traces
- accept only traces whose extracted boxed answer verifies against gold and passes quality gates
- inspect acceptance by family before training

Why try this now:

- previous traces were often external: human-written templates, deterministic solver traces, or agent/code-generated traces.
- model-native successful traces may be more learnable because they match the model's own wording, decomposition style, and stopping behavior.

Failure pattern:

- accepted rows are mostly easy families
- hard families such as cipher, bit manipulation, and equation have near-zero acceptance
- accepted traces are correct but too short, lucky, or procedurally fake

When stuck:

- improve the proposal distribution before training: better prompt, better teacher adapter, more samples for the hard family, or verifier-guided repair.
- do not hide failure with boxed-only fallbacks unless intentionally making a mixed control dataset.

### 3. Preference Optimization / DPO

What it does:

- trains on preferred versus rejected responses for the same prompt
- does not require a separate reward-model-serving loop at training time
- can be simpler than online RL once preference pairs exist

Possible project use:

- create pairs from rejection sampling: accepted trace versus wrong trace for the same row
- use family verifiers to label preferred/rejected candidates

Do not jump here until:

- we have enough reliable accepted/rejected pairs by hard family
- the pair labels reflect reasoning quality, not only formatting

### 4. Online RL / RLVR

What it does:

- generates completions during training
- scores them with a reward/verifier
- updates the policy directly toward higher-reward completions

Possible project use:

- verifiable families are a natural fit: binary bit rules, ciphers with deterministic mapping checks, numeric gravity/unit tasks, maybe equation once a verifier exists.

Cost/risk:

- higher implementation complexity
- easier to destabilize
- harder to resume/debug in Colab
- needs reward design and KL/control decisions

Do not jump here until:

- offline rejection sampling shows a static teacher ceiling on hard families, or
- we have a strong verifier but SFT/rejection-sampling cannot convert it into useful generation behavior.

## Stop Rules

Stop or pause a method class when two of these are true:

- public score is flat or regresses across two clean submissions.
- generated eval/probe failures show the same family-level pattern after the intended fix.
- teacher-forced loss improves but raw generations do not solve more rows.
- the next proposal only changes scale, rank, epochs, or another knob without changing the failure mechanism.
- hard-family coverage remains near zero.

When a stop rule triggers, write the evidence in `doc/PROJECT_DECISION_LOG.md`, choose the next rung deliberately, and keep the next run small enough to diagnose.

## Current Recommendation

The current next rung is rejection sampling plus SFT, not more vanilla trace SFT and not immediate online RL.

First deliverable:

- `rejection_sampling_raw_candidates.csv`
- `rejection_sampling_accepted_candidates.csv`
- `rejection_sampling_acceptance_summary.csv`
- one trainable `id/question/trace/gold_answer` CSV only if hard-family accepted coverage is meaningful

Decision gate:

- if cipher/bit/equation accepted coverage is near zero, stop before training and improve candidate generation or add verifier-guided repair.
