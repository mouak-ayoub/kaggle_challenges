# Local Project Memory

Last updated: 2026-05-17

## Current State

This project is in the Colab LoRA experiment phase.

Known facts:

- The local repository started empty.
- The nearby `Physionet_ECG_challenge` project was used as inspiration for project organization.
- Official Kaggle competition data is present locally under `data/raw/`.
- The downloaded zip was extracted and deleted.
- Official `train.csv` has 9,500 rows with columns `id`, `prompt`, `answer`.
- Official `test.csv` has 3 rows with columns `id`, `prompt`.
- Kaggle requires a LoRA adapter packaged as `submission.zip`, not a prediction CSV.
- The active Colab notebooks write test predictions only as sanity CSVs and package one `{EXPERIMENT_NAME}_run_bundle.zip`; Kaggle `submission.zip` is built locally from the bundle adapter files.
- Kaggle discussion access via API is working with local auth. Useful findings are now recorded in `doc/errors.md` and `doc/PROJECT_DECISION_LOG.md`.
- Local Kaggle reference pulls live under `data/reference/kaggle/` for comparison with official notebooks; `data/` is ignored and should not be committed.
- Submission artifacts are tracked under `data/outputs/submissions/`, with one folder per Kaggle upload plus `metadata.json` and a registry CSV.
- There is no official `problem_type` column in these CSV files, so categories need to be inferred from prompt text or external traces.
- Public challenge breadcrumbs indicate a Nemotron-3 Nano baseline and structured reasoning problem families.

## Active Focus

Prepare the next serious method while keeping the workflow simple:

1. Keep the active Colab notebooks Drive-backed and directly runnable.
2. Use probe-on-log, generated-eval-on-save, and final generated eval to compare checkpoints and final adapters.
3. Build Kaggle submissions locally from Colab run bundles.
4. Start the next procedural-supervision direction, likely short traces or a STaR-like bootstrap, only after the notebook path stays simple.

Current collaboration loop:

1. Brainstorm candidate ideas together, with critique focused on simple non-trivial methods that can plausibly move the score.
2. Choose one idea before implementation; avoid mixing unrelated knobs unless there is a deliberate reason.
3. Implement the selected idea as a notebook-first experiment with Occam's razor.
4. Run it in Colab, then copy/download the run bundle and diagnostics locally.
5. Build the strict Kaggle `submission.zip` locally from the run bundle adapter files.
6. Update the experiment dashboard, checklist, memory, and decision log with the evidence.
7. Iterate until the public score reaches at least `0.7` or the evidence forces a better target strategy.

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
- Updated the submission path so Colab saves adapter files in a run bundle and the local helper builds the strict Kaggle upload zip.
- Read key Kaggle discussion threads on submission packaging, metric updates, vLLM nondeterminism, boxed-answer extraction, and Kaggle environment issues.
- Confirmed Kaggle accepted two adapter submissions:
  - local small-model test adapter scored about 0.50
  - Colab Nemotron adapter scored about 0.62
- The full-data raw Nemotron control `S1_raw_full_r4` / `ACTIVE_02_raw_full_r4` scored `0.54`, worse than the 0.62 partial baseline despite clean training and lower eval loss.
- The S4 final/current run `S4_attention_expand_r8_private_boxed_max128_drive` scored `0.53`; it learned clean boxed output formatting but did not improve reasoning enough to beat raw baselines.
- Partial S4 saved-checkpoint generated eval now shows checkpoint 96 at `79/256 = 0.308594` and checkpoint 144 at `90/256 = 0.351562`, both below final/current step 193 at `95/256 = 0.371094`; checkpoint 192 is still pending until its saved summary row appears.
- Archived the two accepted submission zips under `data/outputs/submissions/`:
  - `2026-05-16_colab_nemotron_lora_score_0_62/`
  - `2026-05-16_local_smol_lora_score_0_50/`
- Restored the active notebook defaults to the known 0.62 raw-answer training target. Boxed-answer targets remain available through `TRAIN_TARGET_FORMAT = "boxed"` as an experiment, not the baseline.
- Added boxed-aware answer extraction with docstrings in both local and Colab notebooks for sanity checks and future metric-aligned experiments.
- Prepared the next 0.7-attempt Colab run:
  - `TRAIN_ROWS = None` removes the 1,024-row cap while keeping a 256-row eval split.
  - Probe inference now shows 5 before/after prompts with extracted answers and raw completions.
  - The exact probe set is saved to `data/outputs/colab_lora/probe_questions.csv`.
  - A trainer callback now runs those same 5 probes after logging steps and appends results to `probe_evolution.csv`.
  - Logging, eval, and save steps are proportional to estimated total training steps.
  - A post-training decision dashboard now displays trainer log history, train/eval loss curves, probe evolution summaries, final probe raw outputs, and a small submit-signal table.
  - Generated-answer accuracy on the fixed 256-row eval split now runs inside the training flow for the final model when `RUN_GENERATED_EVAL = True`; it writes both aggregate and inferred-family accuracy, and should be set false only for quick smoke runs.
  - Post-training `test.csv` sanity inference displays raw completions as well as extracted answers.
  - The notebook writes `data/outputs/colab_lora/run_config.json` and `trainer_log_history.csv` for submission tracking.
  - Superseded packaging note: current Colab runs download one `{EXPERIMENT_NAME}_run_bundle.zip`; Kaggle `submission.zip` is built locally from that bundle.
- Added `doc/EXPERIMENT_CHECKLIST.md` as the run planning board with checkbox tables for full-data, boxed, private-reasoning, LoRA-rank, expanded-target-module, and later synthetic-CoT experiments. The checklist uses stable status marks instead of colors because Markdown color support is renderer-dependent.
- Configured the active Colab notebook for the next selected experiment `S2_private_reasoning_boxed_r8`: full data, boxed targets, private-reasoning prompt, LoRA rank 8/alpha 64, and experiment-specific output paths.
- Added planned expanded-attention experiment `S4_attention_expand_r8_private_boxed` to the local checklist. The active notebook still remains on `S2_private_reasoning_boxed_r8` unless intentionally changed before copying to a new Colab notebook.
- Added a display-only runtime/GPU inventory cell at the top of the active Colab notebooks so runs record Python, platform, GPU, `nvidia-smi`, disk, and memory details before any config decisions.
- Began explicit tracking for the three active Colab experiments in `doc/EXPERIMENT_CHECKLIST.md`: `ACTIVE_02_raw_full_r4`, `ACTIVE_03_private_reasoning_boxed_r8`, and `ACTIVE_04_S4_attention_expand_r8_private_boxed_max128`.
- After automatic Colab disconnects lost the `03` and `04` runtime artifacts, updated the Colab notebook to write outputs/checkpoints/run bundles/diagnostics to Google Drive by default and to support explicit resume from a chosen Drive checkpoint.
- The Drive project was reorganized under `/content/drive/MyDrive/Colab_Notebooks/Kaggle challenges/nemotron_challenge/`; notebooks live at that root and durable training artifacts live under `artefacts/`, with run outputs under `artefacts/outputs/{EXPERIMENT_NAME}`.
- The Colab notebooks now write one `{EXPERIMENT_NAME}_run_bundle.zip` per finished run instead of separate Colab submission and diagnostics zips. The run bundle includes adapter files plus diagnostics/eval files; local packaging builds the Kaggle `submission.zip` from the bundle adapter files with `scripts/build_submission_from_run_bundle.py`.
- Simplified the Colab notebooks around the current Drive-only workflow: removed the non-Colab GPU-inventory branch, removed the Google Drive backup switch, and replaced resume/fresh-run switches with `RESUME_FROM_CHECKPOINT=False` plus `RESUME_CHECKPOINT_STEP=None`. For a resume, set the boolean true and choose the checkpoint number; for a clean rerun, use a new `EXPERIMENT_NAME` or manually clear the old output directory.
- Added generated-answer checkpoint eval during training with one switch, `GENERATED_EVAL_ROWS_ON_SAVE`. Use `EVAL_ROWS` for full checkpoint summaries under `checkpoint_eval/checkpoint_generated_eval_summary.csv`, `64` for cheaper tracking, or `0` to disable.
- Added `scripts/build_experiment_dashboard.py`, a local HTML dashboard builder that scans `data/outputs/submissions/`, diagnostics zips, run bundles, and copied run folders. It shows public scores, train/eval loss by run/checkpoint, probe evolution, generated-eval summaries, and grouped responses to the three public sanity test prompts. S4 checkpoint generated-eval CSVs from Drive are now copied under the S4 archive's `checkpoint_eval/` folder and appear in the local dashboard. The older 0.62 run did not save a sanity CSV, so its final three sanity answers were recovered into a local ignored CSV from the displayed notebook output.
- The local dashboard remains a plain HTML report, but its graphs now use Chart.js canvases instead of hand-drawn SVG so axes, ticks, tooltips, legends, and small family charts are more readable without introducing a notebook report.
- Added `notebooks/tools/04_colab_backfill_generated_eval.ipynb` as the single Colab backfill artifact for old submissions, separated from normal training notebooks. It scores already submitted adapter zips on the same fixed 256-row generated-eval split and fixed five-row probe set. Its default run list backfills the 0.62 raw baseline (`00-raw-1024`) and the 0.54 full raw control (`02-raw-full`). The dashboard will pick up each resulting `generated_eval_summary.csv` and `probe_evolution.csv` after they are copied into the matching local submission archive folder.
- First Colab backfill result for `00-raw-1024` completed: generated eval `65/256 = 0.253906`. Family accuracy was bit manipulation `2/45`, cipher `0/43`, equation `2/39`, gravity `1/41`, numeral `52/52`, unit conversion `8/36`. This is lower than S4 local generated eval despite the much better public score, reinforcing that the 256-row local generated eval is diagnostic but not a public-score proxy.
- The downloaded `00-raw-1024` backfill CSVs were copied into `data/outputs/submissions/2026-05-16_colab_nemotron_lora_score_0_62/`. The downloaded `02-raw-full` backfill CSVs were copied into `data/outputs/submissions/2026-05-17_colab_raw_full_r4_score_0_54/`. The dashboard now compares `00-raw-1024`, `02-raw-full`, and S4 checkpoints on the same 256-row generated eval and five-row probe set.
- The local Colab notebook `notebooks/01_colab_train_and_submit.ipynb` is currently configured for direct import as `S4_attention_expand_r8_private_boxed_max128_drive`: Drive-backed outputs, explicit checkpoint resume disabled by default, boxed/private prompt, LoRA `r=8` alpha `64`, expanded attention targets `in_proj/out_proj/q_proj/k_proj/v_proj/o_proj`, `MAX_NEW_TOKENS=128`, and effective batch `16*3=48`.
- Created `notebooks/03_colab_short_trace_train_and_submit.ipynb` for the short procedural-trace experiment `S6_short_trace_boxed_r8_attention_drive`. It adds compact cipher traces before a final boxed answer for parseable cipher rows, falls back to boxed targets for other rows, and writes `trace_training_samples.csv` into diagnostics. Local audit found `1,576 / 9,500` train rows receive a cipher trace.
- Added the next planned method idea `S7_starish_short_trace_bootstrap`: a STaR-like training-time loop that generates candidate short traces, filters/checks them against known train answers, corrects failed cases using gold answers, then trains a Kaggle-compatible LoRA adapter with ordinary SFT. This is bootstrapped supervised fine-tuning, not inference-time agent execution.
- Naming convention decision: use the top-cell `EXPERIMENT_NAME` as the source of truth for all paths, diagnostics, zips, archive folders, and Kaggle descriptions. Future experiment names should use lowercase `expNN_short_method_key_knobs`; do not rename an already-running Drive-backed experiment because that changes the checkpoint directory.

## Open Questions

- Does Kaggle provide a starter notebook or only data files?
- Is local validation possible from the official train file?
- Superseded: the full-row raw-target run did not improve over the 0.62 partial baseline; it scored 0.54.
- Does a boxed or private-reasoning prompt improve the score, or does it mainly add format/extraction failures?
