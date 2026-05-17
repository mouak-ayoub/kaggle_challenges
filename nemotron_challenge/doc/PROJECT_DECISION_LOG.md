# Project Decision Log

Last updated: 2026-05-17

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

## Decision: Keep Raw-Answer Targets As The Current Baseline

### Context

The official metric prompt asks the model to put the final answer inside `\boxed{}` and the extractor prioritizes boxed content. However, the accepted Colab Nemotron run that scored about 0.62 was trained with raw short-answer targets, before the notebook was changed to boxed SFT targets.

### Decision

Keep `TRAIN_TARGET_FORMAT = "raw"` as the active notebook default because it reproduces the best known submission. Keep `TRAIN_TARGET_FORMAT = "boxed"` available as an explicit experiment, and keep local sanity extraction boxed-aware so extraction failures can be analyzed separately.

### Evidence

The official metric notebook and Kaggle discussion threads confirm boxed answers are the primary extraction path. The metric was updated after community reports that answers containing `}` were impossible or ambiguous under the old regex. But the only observed higher score so far is the raw-answer Colab run at about 0.62; the boxed-target change has not yet been scored as an improvement.

### Consequence

Do not overwrite the known baseline with an unscored format change. When testing boxed targets, record it as a separate submission with its own metadata and compare against the raw-answer baseline.

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

## Decision: Scale The Known Raw Baseline Before CoT Variants

### Context

The next target is a public score near 0.7. Proposed changes include CoT-style prompting, training on more data, proportional logging/eval, and watching probe answers before and after training.

### Decision

Make the next default run a scaled version of the known 0.62 baseline: raw-answer targets, the same direct prompt, all available training rows except the eval split, and better instrumentation. Keep boxed and private-reasoning prompts available as explicit switches, not the default.

### Evidence

The only high-scoring observed submission so far is the raw-target Nemotron LoRA run at about 0.62. Kaggle discussions warn that longer reasoning traces can lower score if the model fails to finish with a clean final answer. More data and better logs are lower-risk changes than changing both reasoning style and target format at the same time.

### Consequence

Compare the full-row raw run against 0.62 first. If it does not improve enough, submit a separate prompt/target experiment and track it independently in `data/outputs/submissions/`.

### Status

Superseded by the `S1_raw_full_r4` result.

## Decision: Stop Scaling Raw-Only Final-Answer SFT

### Context

The known best Colab Nemotron submission scored about `0.62` with raw short-answer SFT on a 1,024-row subset. The next control run scaled the same direct/raw setup to the full train split minus 256 eval rows.

### Decision

Do not spend more submissions on raw-only final-answer SFT as the main path. Prioritize metric-aligned boxed outputs, expanded target modules, and short procedural trace supervision.

### Evidence

`S1_raw_full_r4` / `ACTIVE_02_raw_full_r4` trained cleanly and eval loss improved monotonically to about `0.7933`, but the fixed probe stayed `1/5` and the public leaderboard score was `0.54`, below the partial raw baseline at about `0.62`.

### Consequence

Eval loss on final-answer SFT is not a reliable proxy for hidden reasoning score. The next serious experiments should test whether the model learns procedures, not just answer priors and output brevity.

### Status

Active.

## Decision: Persist Colab Training Outputs To Google Drive

### Context

Long Nemotron LoRA runs in Colab can be interrupted or reclaimed automatically, especially when multiple heavy GPU sessions run in parallel. Runtime-local files under `/content` disappear when the VM is deleted.

### Decision

Write experiment outputs, checkpoints, run bundles, trainer logs, and probe evolution files to Google Drive by default. Keep raw competition data local for speed, but persist training outputs under `/content/drive/MyDrive/Colab_Notebooks/Kaggle challenges/nemotron_challenge/artefacts/outputs/{EXPERIMENT_NAME}`.

### Evidence

The `03` and `04` Colab sessions disconnected automatically and their runtime-local artifacts were lost. Resume is only possible if a full `checkpoint-*` directory, including trainer state, survives outside the VM.

### Consequence

The Colab notebook now mounts Drive and uses Drive-backed `OUTPUT_DIR`. Resume is opt-in: keep `RESUME_FROM_CHECKPOINT=False` for a fresh run, or set it true and choose `RESUME_CHECKPOINT_STEP` to continue from `OUTPUT_DIR/checkpoint-{step}`. If resume is false and old checkpoints exist, training stops instead of silently mixing runs. Starting a clean rerun with the same `EXPERIMENT_NAME` requires changing the experiment name or manually clearing the old Drive output directory.

Generated-answer checkpoint eval runs during training when `GENERATED_EVAL_ROWS_ON_SAVE` is above `0`. The current serious-run default is `EVAL_ROWS`, which scores the full fixed eval split at each saved checkpoint. Set it to `64` for cheaper tracking or `0` to disable. Checkpoint summaries are written to `checkpoint_eval/checkpoint_generated_eval_summary.csv` and included in the run bundle.

### Status

Active.

## Decision: Build Kaggle Submission Zips Locally From Colab Run Bundles

### Context

The Colab notebooks previously wrote both a standalone Kaggle `submission.zip` and a diagnostics zip that also embedded the submission zip. That created ambiguity about which zip should be uploaded and made Drive contain duplicate packaged artifacts for the same run.

### Decision

At the end of each Colab run, write one portable `{EXPERIMENT_NAME}_run_bundle.zip`. The bundle contains `adapter/adapter_config.json`, `adapter/adapter_model.safetensors`, run configuration, trainer logs, probe logs, generated-eval files, checkpoint-eval files when present, and TensorBoard events. Build the actual Kaggle `submission.zip` locally from the bundle adapter files.

### Evidence

The Kaggle artifact must contain only `adapter_config.json` and `adapter_model.safetensors` at the zip root. Diagnostics and checkpoint-eval files are needed for local reports but must not be mixed into the upload artifact.

### Consequence

Colab is responsible for training and producing one run bundle. The local repository is responsible for extracting/analyzing that bundle and creating the final Kaggle upload with `scripts/build_submission_from_run_bundle.py`.

### Status

Active.

## Decision: Do Not Continue Final-Answer Boxed SFT Without Procedural Supervision

### Context

The `S4_attention_expand_r8_private_boxed_max128_drive` run trained full-data boxed targets with a private-reasoning prompt, LoRA rank 8, and expanded attention target modules.

### Decision

Do not repeat this exact final-answer boxed SFT path as the main strategy. Future serious attempts should add procedural supervision, checkpoint selection from generated eval, or category-specific data rather than just stronger boxed-format LoRA.

### Evidence

The final/current S4 adapter had clean boxed outputs and local generated eval `95/256 = 0.371`, but the public score was only `0.53`. It was strong on numerals but weak on cipher, equation, gravity, and bit manipulation. The 5-row probe also showed loss improving while reasoning correctness oscillated and final step regressed to `1/5`.

### Consequence

Boxed formatting helps extraction discipline but is not enough to improve reasoning. Prioritize short procedural traces, STaR-like bootstrapping, and checkpoint selection by generated eval family accuracy.

### Status

Active.

## Decision: Use Brainstorm-Then-Notebook Experiment Loop

### Context

The project has already spent several notebook iterations on raw final-answer SFT and boxed/private-reasoning variants. The best observed public score is still about `0.62`, while full-data raw and final-answer boxed experiments regressed to about `0.54` and `0.53`.

### Decision

Use a deliberate loop for future work: brainstorm candidate ideas together, critique them as research hypotheses, choose one simple non-trivial idea, implement it notebook-first, run it in Colab, download the run bundle/diagnostics, build the Kaggle `submission.zip` locally, update the dashboard and experiment notes, then iterate toward at least `0.7`.

### Evidence

Recent runs show that changing several knobs without enough procedural signal can improve formatting or eval loss while hurting the public score. The current strongest directions are short procedural traces and STaR-like bootstrapped SFT, but each needs visible artifacts and category-level evidence before becoming the next submitted adapter.

### Consequence

Do not jump straight into broad refactors or factorial sweeps. For each iteration, keep the selected idea legible in the notebook top cell, preserve raw completions and generated-eval outputs, and archive the resulting bundle/submission before choosing the next idea.

### Status

Active.
