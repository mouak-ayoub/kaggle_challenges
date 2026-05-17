# Local Project Memory

Last updated: 2026-05-16

## Current State

This project has just been bootstrapped.

Known facts:

- The local repository started empty.
- The nearby `Physionet_ECG_challenge` project was used as inspiration for project organization.
- Official Kaggle competition data is present locally under `data/raw/`.
- The downloaded zip was extracted and deleted.
- Official `train.csv` has 9,500 rows with columns `id`, `prompt`, `answer`.
- Official `test.csv` has 3 rows with columns `id`, `prompt`.
- Kaggle requires a LoRA adapter packaged as `submission.zip`, not a prediction CSV.
- The Colab notebook now writes test predictions only as a sanity CSV and packages the saved adapter into `submission.zip`.
- Kaggle discussion access via API is working with local auth. Useful findings are now recorded in `doc/errors.md` and `doc/PROJECT_DECISION_LOG.md`.
- Submission artifacts are tracked under `outputs/submissions/`, with one folder per Kaggle upload plus `metadata.json` and a registry CSV.
- There is no official `problem_type` column in these CSV files, so categories need to be inferred from prompt text or external traces.
- Public challenge breadcrumbs indicate a Nemotron-3 Nano baseline and structured reasoning problem families.

## Active Focus

Prepare for a slow, reproducible start:

1. Inspect and classify prompt families from `train.csv`.
2. Build a tiny supervised fine-tuning JSONL from `prompt` -> `answer`.
3. Adapt the existing `lora_demo/Finetune_Gemma.ipynb` pattern into a competition-specific notebook.
4. Decide whether the first run targets a small local model, a Colab GPU model, or the official Nemotron baseline.

## Recent Milestones

- Created project documentation split:
  - `README.md`
  - `AGENTS.md`
  - `doc/PROJECT_DECISION_LOG.md`
  - `doc/LOCAL_PROJECT_MEMORY.md`
  - `doc/errors.md`
- Created initial reusable package shape under `src/nemotron_challenge/`.
- Added a dataset inspection script.
- Extracted official data into `data/raw/` and deleted the downloaded zip.
- Confirmed the official train/test schemas with `scripts/inspect_dataset.py`.
- Updated the Colab notebook submission path to validate and zip the required PEFT adapter config and weights.
- Read key Kaggle discussion threads on submission packaging, metric updates, vLLM nondeterminism, boxed-answer extraction, and Kaggle environment issues.
- Confirmed Kaggle accepted two adapter submissions:
  - local small-model test adapter scored about 0.50
  - Colab Nemotron adapter scored about 0.62
- Archived the two accepted submission zips under `outputs/submissions/`:
  - `2026-05-16_colab_nemotron_lora_score_0_62/`
  - `2026-05-16_local_smol_lora_score_0_50/`
- Restored the active notebook defaults to the known 0.62 raw-answer training target. Boxed-answer targets remain available through `TRAIN_TARGET_FORMAT = "boxed"` as an experiment, not the baseline.
- Added boxed-aware answer extraction with docstrings in both local and Colab notebooks for sanity checks and future metric-aligned experiments.
- Prepared the next 0.7-attempt Colab run:
  - `TRAIN_ROWS = None` removes the 1,024-row cap while keeping a 256-row eval split.
  - Probe inference now shows 5 before/after prompts with extracted answers and raw completions.
  - The exact probe set is saved to `outputs/colab_lora/probe_questions.csv`.
  - A trainer callback now runs those same 5 probes after every eval step and appends results to `probe_evolution.csv` and `probe_evolution.jsonl`.
  - Logging, eval, and save steps are proportional to estimated total training steps.
  - A post-training decision dashboard now displays trainer log history, train/eval loss curves, probe evolution summaries, final probe raw outputs, and a small submit-signal table.
  - A TensorBoard cell launches against `/content/nemotron_challenge/outputs/colab_lora` after training.
  - Post-training `test.csv` sanity inference displays raw completions as well as extracted answers.
  - The notebook writes `outputs/colab_lora/run_config.json` and `trainer_log_history.csv` for submission tracking.
  - Colab downloads both `submission.zip` and `{EXPERIMENT_NAME}_diagnostics.zip`, where the diagnostics zip includes probe evolution, run config, trainer logs, sanity predictions, TensorBoard event files, adapter config, and the submission zip.
- Added `doc/EXPERIMENT_CHECKLIST.md` as the run planning board with checkbox tables for full-data, boxed, private-reasoning, LoRA-rank, expanded-target-module, and later synthetic-CoT experiments. The checklist uses stable status marks instead of colors because Markdown color support is renderer-dependent.
- Configured the active Colab notebook for the next selected experiment `S2_private_reasoning_boxed_r8`: full data, boxed targets, private-reasoning prompt, LoRA rank 8/alpha 64, and experiment-specific output paths.
- Added planned expanded-attention experiment `S4_attention_expand_r8_private_boxed` to the local checklist. The active notebook still remains on `S2_private_reasoning_boxed_r8` unless intentionally changed before copying to a new Colab notebook.
- Added a display-only Colab runtime/GPU inventory cell at the top of `notebooks/01_colab_train_and_submit.ipynb` so Colab runs record Python, platform, CUDA, GPU, `nvidia-smi`, disk, and memory details before any config decisions. Outside Colab it only prints a short skip message.
- Began explicit tracking for the three active Colab experiments in `doc/EXPERIMENT_CHECKLIST.md`: `ACTIVE_02_raw_full_r4`, `ACTIVE_03_private_reasoning_boxed_r8`, and `ACTIVE_04_S4_attention_expand_r8_private_boxed_max128`.
- After automatic Colab disconnects lost the `03` and `04` runtime artifacts, updated the Colab notebook to write outputs/checkpoints/submission zips/diagnostics to Google Drive by default and to auto-resume from the latest `checkpoint-*` under the experiment output directory.
- The local Colab notebook `notebooks/01_colab_train_and_submit.ipynb` is currently configured for direct import as `S4_attention_expand_r8_private_boxed_max128_drive`: Drive-backed outputs, auto-resume enabled, boxed/private prompt, LoRA `r=8` alpha `64`, expanded attention targets `in_proj/out_proj/q_proj/k_proj/v_proj/o_proj`, `MAX_NEW_TOKENS=128`, and effective batch `16*3=48`.
- Created `notebooks/03_colab_short_trace_train_and_submit.ipynb` for the short procedural-trace experiment `S6_short_trace_boxed_r8_attention_drive`. It adds compact cipher traces before a final boxed answer for parseable cipher rows, falls back to boxed targets for other rows, and writes `trace_training_samples.csv` into diagnostics. Local audit found `1,576 / 9,500` train rows receive a cipher trace.

## Open Questions

- Does Kaggle provide a starter notebook or only data files?
- Is local validation possible from the official train file?
- Does the full-row raw-target run improve over the 0.62 partial baseline?
- Does a boxed or private-reasoning prompt improve the score, or does it mainly add format/extraction failures?
