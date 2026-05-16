# Project Decision Log

Last updated: 2026-05-16

This file records durable project decisions. It is not a scratchpad.

## Decision: Start With Instrumentation Before Training

### Context

The challenge is about improving reasoning performance, but the local repository has no official data yet.

### Decision

Do not begin with fine-tuning or heavy inference code. First build data inspection, answer normalization, baseline logging, and category-level reporting.

### Evidence

Public community material suggests multiple heterogeneous problem families. A single aggregate score can hide whether failures come from reasoning, formatting, extraction, or category-specific weakness.

### Consequence

The initial repository contains documentation, project memory, and lightweight reusable utilities rather than model training code.

### Status

Active.

## Decision: Keep Supporting Markdown Under `doc/`

### Context

The nearby PhysioNet project separates agent instructions, current memory, durable decisions, and reusable source code. The project root was becoming noisy with several Markdown support files.

### Decision

Keep root-level Markdown limited to entry points, and move supporting notes into `doc/`:

- `AGENTS.md`
- `doc/LOCAL_PROJECT_MEMORY.md`
- `doc/PROJECT_DECISION_LOG.md`
- `doc/errors.md`
- `README.md`
- `src/`

### Evidence

That split keeps current status, durable technical choices, and reusable lessons from collapsing into one unclear README while keeping the repository root easy to scan.

### Consequence

Future sessions should start by reading the memory, decision, and error files before editing code.

### Status

Active.

## Decision: Treat Problem Families As First-Class Units

### Context

Public artifacts list reasoning families including ciphers, numerals, unit conversion, bit manipulation, gravity, and equations.

### Decision

All analysis and validation should preserve problem type/category when available. Improvements should be evaluated by category before being trusted globally.

### Evidence

Different families likely need different failure analysis and may benefit from different prompting, validation, or synthetic data strategies.

### Consequence

The first utilities include schema detection and category-count reporting.

### Status

Active.

## Decision: Keep Shared Notebook Helpers In Sync

### Context

The project uses a local notebook for Windows/data dry runs and a Colab notebook for Nemotron GPU training. Some cells are environment-specific, but helper methods for formatting, inspection, answer generation, and diagnostics should behave the same in both notebooks.

### Decision

When shared helper code changes in one notebook, mirror the same method in the other notebook when applicable. Keep only install, runtime, path, and model-loading differences separate.

### Evidence

The module summary helper was improved in the Colab notebook to show dtypes after Nemotron BF16/FP32 issues. The same diagnostic is useful in the local notebook too.

### Consequence

Notebook edits should check whether the changed method is shared logic or environment logic before stopping.

### Status

Active.

## Decision: Submit A LoRA Adapter Zip, Not Predictions

### Context

The competition `test.csv` has example prompts, but Kaggle scoring does not use a normal prediction CSV for this challenge.

### Decision

The Colab training notebook should treat generated test answers as a sanity check only. The real submission artifact is `submission.zip` containing the required saved PEFT LoRA adapter files at the zip root.

### Evidence

The public Kaggle submission demo saves the PEFT adapter and zips the adapter files. The notebook output shows `adapter_config.json` and `adapter_model.safetensors` in `submission.zip`, with adapter rank capped at 32.

### Consequence

Do not spend work on a final `submission.csv` path for leaderboard submission. Validate the adapter directory, check `adapter_config.json` rank, zip `adapter_config.json` and `adapter_model.safetensors` at the zip root, and upload `submission.zip`.

### Status

Active.

## Decision: Train Toward The Official Boxed Answer Format

### Context

The official metric prompt asks the model to put the final answer inside `\boxed{}` and the extractor prioritizes boxed content.

### Decision

Notebook SFT examples should train assistant targets as `\boxed{answer}` rather than raw short answers. Local sanity extraction should mirror the official boxed extraction behavior, including answers that contain literal braces.

### Evidence

The official metric notebook and Kaggle discussion threads confirm boxed answers are the primary extraction path. The metric was updated after community reports that answers containing `}` were impossible or ambiguous under the old regex. Threads also report score drops when generated reasoning is truncated before the boxed answer.

### Consequence

Prompts should ask for a boxed answer with no trailing text. Generation checks should inspect raw outputs and boxed extraction separately from reasoning correctness.

### Status

Active.

## Decision: Treat Public LB Scores As Noisy Signals

### Context

Participants reported different scores for identical or near-identical submissions.

### Decision

Do not choose methods purely from single public leaderboard runs. Prefer local category-level validation, raw completion logs, and repeated checks when leaderboard differences are small.

### Evidence

Kaggle discussion confirms vLLM scoring is not deterministic even with `temperature=0.0`; the host declined deterministic vLLM settings because they would reduce throughput substantially. Submissions can also take roughly 70-90 minutes or longer to score.

### Consequence

Small leaderboard deltas should be treated cautiously. Track category-level local behavior and avoid overfitting to one public score.

### Status

Active.
