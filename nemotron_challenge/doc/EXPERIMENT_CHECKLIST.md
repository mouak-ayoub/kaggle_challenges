# Experiment Checklist

Use this as the submission planning board. After each Kaggle score, copy the submitted zip and source run bundle into `data/outputs/submissions/`, then update the status and notes here.

## Status Legend

| Mark | Meaning |
| --- | --- |
| `[x]` | done / already scored |
| `[ ]` | not run yet |
| `NEXT` | recommended next |
| `RISK` | high-risk or needs preflight check |

## Resume Here

Last recorded project result was `2026-05-18`; today is `2026-05-24`, so this is a 6-day return. Start here before reading the lower historical notes.

| Current answer | Detail |
| --- | --- |
| Best public baseline | `00-raw-1024`, score about `0.62`. |
| Recent S4 result | S4 final scored `0.53`; S4 checkpoint-144 scored `0.55`. Checkpoint timing helped a little but did not rescue S4. |
| Current conclusion | Stop S4-style final-answer boxed SFT as the main path. |
| Next active task | Integrate `data/input/trace_training.csv` into the next trace-training notebook, then evaluate with `eval_v2_grouped_family_balanced` before trusting a submission. |
| After that | Use `eval_v2` to support solver-guided STaR seed data, starting with verifier-friendly families. |
| Do not resume from | The lower legacy S0-S7 checklist. It is historical context, not the active continuation path. |

## Current Rule

Do not run a full factorial grid. We cannot afford every combination of prompt, target format, rank, LR, sequence length, and epoch count. Use bundled runs where each bundle has a clear hypothesis and avoids changing unrelated risky knobs.

Originality rule: all outside research is allowed as literature, tools, and inspiration, including papers, open-source code, Reddit/Medium/Substack posts, YouTube talks, other-domain competition ideas, and reasoning-challenge methods that are not from this exact Nemotron Kaggle challenge. Do not inspect or copy another active Nemotron challenge competitor's high-scoring notebook, GitHub repo, trajectory dataset, or submission path as the main method unless the user explicitly approves that. Our trajectory should come from our own verifier-guided STaR experiments.

## Naming Rule

Use one top-cell variable as the source of truth:

```python
EXPERIMENT_NAME = "exp04_attention_boxed_r8_max128_drive"
```

All output paths, adapter folders, run bundles, locally built submission zips, and Kaggle descriptions should derive from that value. For future runs, prefer lowercase `expNN_short_method_key_knobs` names. Avoid `S2`/`S4` shorthand in new run names because it can be confused with notebook numbers or active run numbers.

For new Colab runs, the notebook should download one `{EXPERIMENT_NAME}_run_bundle.zip`. Build the Kaggle `submission.zip` locally from that bundle so diagnostics and upload artifacts stay separate.

Do not rename an experiment while it is already running or resumable from Drive; changing `EXPERIMENT_NAME` changes the checkpoint directory. The ongoing run keeps `S4_attention_expand_r8_private_boxed_max128_drive` until it finishes.

Before submitting a trained adapter, inspect the decision dashboard in the Colab notebook:

- train loss should decrease without obvious divergence
- eval loss should not be clearly worse than comparable runs
- probe evolution should not show formatting collapse or repeated junk
- generated eval accuracy on the fixed 256-row eval split should be checked overall and by inferred problem family when choosing a submission candidate
- final test sanity predictions should be non-empty and formatted as expected

These signals are filters, not a leaderboard proxy. A run can look good locally and still score poorly because the public evaluator is noisy and hidden prompts differ from the five probes.

`RUN_GENERATED_EVAL` is `True` by default for serious submission runs because it gives a better local signal than eval loss alone. At the end of training, the final model gets generated-answer accuracy on the fixed eval split and writes one `all` row plus per-family rows such as `cipher`, `bit_manipulation`, and `numeral`. Set it to `False` only for quick smoke tests.

`GENERATED_EVAL_ROWS_ON_SAVE = EVAL_ROWS` writes generated-answer accuracy at each saved checkpoint for the full fixed eval split. Set it to `64` for cheaper checkpoint tracking, or `0` to disable checkpoint generation. When enabled, it writes per-checkpoint predictions under `checkpoint_eval/` and appends aggregate/family rows to `checkpoint_eval/checkpoint_generated_eval_summary.csv`.

## Near-Term Research Plan

- [ ] `eval_v2_grouped_family_balanced` `NEXT`
  - Purpose: replace the current 256-row generated eval as the main local diagnostic.
  - Design: 1,000-2,000 rows if runtime allows, family-balanced, grouped by prompt/rule pattern where feasible, with overall accuracy, hard-family accuracy excluding numeral, per-family accuracy, extraction failures, max-token hits, and raw completions.
  - Baselines to score: base model, `00-raw-1024`, `02-raw-full`, `04-s4`, and future STaR adapters.
  - Risk: larger eval is slower; keep it diagnostic, not a perfect leaderboard proxy.

- [ ] `exp05_star_seed_512` `NEXT` `RISK`
  - Hypothesis: a small set of compact verified reasoning traces can teach procedure without damaging the base model behavior that gave the `0.62` public score.
  - Method: start from `data/input/trace_training.csv`; use original questions with target-side cipher character-position traces plus boxed answers, with non-cipher rows boxed-only unless a family-specific trace exists.
  - Key config candidate: LoRA `r=8`, targets `in_proj/out_proj/q_proj/k_proj/v_proj/o_proj`, LR `1e-4`, 1 epoch, max seq `2048` if memory allows.
  - Success signal: `eval_v2` hard-family accuracy improves and public score stays near or above `0.62`.

- [ ] `exp06_star_cipher_unit_bit_1500` `RISK`
  - Hypothesis: verifier-friendly families can improve first: cipher, unit conversion, and bit manipulation.
  - Method: about 500 verified traces per family, using family-specific solvers/verifiers before training.
  - Key config candidate: LoRA `r=8`, `in_proj/out_proj` first, LR `1e-4`, max seq `2048` or `4096` if memory allows, 1 epoch.
  - Risk: traces can be correct-final-answer but bad reasoning; rejected traces and verifier failure reasons must be saved.

## Submission Runs

| Done | Run | Public score / status | Hypothesis | Key config | Main evidence | Archive / next |
| --- | --- | --- | --- | --- | --- | --- |
| [x] | `ACTIVE_02_raw_full_r4` | `0.54` scored | Full official data improves the known raw-answer baseline without changing prompt/format behavior. | Raw target, direct prompt, full train minus 256 eval rows, LoRA `r=4`/alpha `32`, `in_proj/out_proj`, seq `512`, max new `64`, effective batch `48`, LR `3e-4`, 1 epoch. | Eval loss improved to about `0.7933`, but probe stayed about `1/5`; public score dropped below the `0.62` partial baseline. Full-data raw final-answer SFT is not sufficient. | Archive: `data/outputs/submissions/2026-05-17_colab_raw_full_r4_score_0_54/`. |
| [ ] | `ACTIVE_03_private_reasoning_boxed_r8` | Interrupted / artifacts likely lost | Private-reasoning prompt plus boxed targets improves final-answer discipline and gives rank 8 enough capacity to learn transformations. | Boxed target, private-reasoning boxed prompt, LoRA `r=8`/alpha `64`, `in_proj/out_proj`, seq `512`, full data, 256 eval rows, LR `3e-4`, 1 epoch. | Runtime-local artifacts appear lost after Colab disconnect unless a Drive/manual download copy exists. This run also combines rank and output-format changes, so it is hard to interpret. | Recover only if a Drive/manual copy exists; otherwise do not revive as a priority. |
| [x] | `ACTIVE_04_S4_attention_expand_r8_private_boxed_max128` | Final `0.53`; checkpoint-144 `0.55` | Expanded attention LoRA improves transformation learning while keeping private-reasoning boxed setup. | Boxed target, private-reasoning boxed prompt, LoRA `r=8`/alpha `64`, `in_proj/out_proj/q_proj/k_proj/v_proj/o_proj`, seq `512`, max new `128`, effective batch `48`, LR `3e-4`, 1 epoch. | Final step 193 had best local generated eval `95/256 = 0.371094` but scored `0.53`; checkpoint-144 had lower local generated eval `90/256 = 0.351562` but scored `0.55`. Checkpoint timing matters, but S4 still did not beat the `0.62` raw baseline. | Archives: `data/outputs/submissions/2026-05-17_colab_s4_attention_boxed_r8_final_score_0_53/`, `data/outputs/submissions/2026-05-17_colab_s4_checkpoint144_score_0_55/`. Shift to `eval_v2` and solver-guided STaR unless checkpoint 192 has a strong new signal. |

## Runtime Recovery Rule

Colab runtime-local files under `/content` are not durable. If Colab deletes the VM, those files are gone. A run can continue only if a full `checkpoint-*` directory was saved to Google Drive or downloaded before the disconnect.

The active notebook now uses Google Drive-backed output paths directly:

```python
DRIVE_PROJECT_ROOT = Path("/content/drive/MyDrive/Colab_Notebooks/Kaggle challenges/nemotron_challenge/artefacts")
OUTPUT_DIR = DRIVE_PROJECT_ROOT / "outputs" / EXPERIMENT_NAME
RUN_BUNDLE_ZIP_PATH = DRIVE_PROJECT_ROOT / f"{EXPERIMENT_NAME}_run_bundle.zip"
```

Checkpoint resume is explicit and disabled by default:

```python
RESUME_FROM_CHECKPOINT = False
RESUME_CHECKPOINT_STEP = None
```

To continue from a saved Drive checkpoint, set:

```python
RESUME_FROM_CHECKPOINT = True
RESUME_CHECKPOINT_STEP = 192
```

That resumes from `OUTPUT_DIR / "checkpoint-192"`. If resume is false and old checkpoints already exist in `OUTPUT_DIR`, the notebook stops instead of silently mixing old and new training. For a clean rerun, use a new `EXPERIMENT_NAME` or manually clear the old Drive output directory after confirming it is no longer needed.

## Legacy Experiment Notes

This table preserves older ideas for context only. It is not the active continuation plan. Use `Resume Here` and `Near-Term Research Plan` first.

| Status | Candidate | Outcome / role | Keep / superseded reason |
| --- | --- | --- | --- |
| [x] | `S0_colab_raw_1024_r4` | Scored about `0.62`; best known public baseline. | Keep as leaderboard baseline and do-no-harm reference. |
| [x] | `S1_raw_full_r4` | Scored `0.54`; full-data raw-answer control. | Superseded as a main path because more raw final-answer SFT hurt public score. |
| [ ] | `S2_raw_full_r8` | Capacity-only raw full-data idea. | Not active; raw full-data direction is weak without procedural supervision. |
| [ ] | `S2_private_reasoning_boxed_r8` | Rank + boxed/private prompt idea. | Not active; combines too many axes and the related S4 path underperformed. |
| [ ] | `S3_boxed_full_r4_or_r8` | Isolate boxed target/output format. | Not active; boxed final-answer SFT has not improved reasoning enough. |
| [ ] | `S4_attention_expand_r8_private_boxed` | Expanded attention modules with boxed/private setup. | Effectively tested by S4 final/checkpoint-144; still below `0.62`. |
| [ ] | `S5_private_reasoning_boxed` | Prompt encourages internal reasoning but trains only final boxed answers. | Not active; procedural supervision is the stronger next direction. |
| [ ] | `S6_short_cot_synthetic` | Generic short rationales for selected categories. | Fold into solver-guided STaR only after trace quality checks. |
| [ ] | `S6_short_trace_boxed_r8_attention_drive` | Notebook exists at `notebooks/03_colab_short_trace_train_and_submit.ipynb`; cipher short traces for about `1,576 / 9,500` rows. | Useful source notebook, but do not run directly before `eval_v2` and trace audit. |
| [ ] | `S7_solver_guided_star_bootstrap` | Generate, verify, filter, and train on compact traces. | Active research direction, but the immediate first task is `eval_v2_grouped_family_balanced`. |
| [ ] | `S7_best_config_eval0_final` | Use all rows after a winning config is known. | Later polish only; not exploration. |

## Knobs And Critique

## Nemotron Module Counts

Captured from the loaded `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` module summary in Colab.

| Module | Class | Count | Target-module note |
| --- | --- | ---: | --- |
| `down_proj` | `Linear4bit` | 2967 | Very large expansion. High memory/time risk. |
| `up_proj` | `Linear4bit` | 2967 | Very large expansion. High memory/time risk. |
| `conv1d` | `Conv1d` | 23 | Not a standard LoRA target for this run. |
| `in_proj` | `Linear4bit` | 23 | Known working target. |
| `out_proj` | `Linear4bit` | 23 | Known working target. |
| `k_proj` | `Linear4bit` | 6 | Candidate for attention-expansion experiment. |
| `q_proj` | `Linear4bit` | 6 | Candidate for attention-expansion experiment. |
| `o_proj` | `Linear4bit` | 6 | Candidate for attention-expansion experiment. |
| `v_proj` | `Linear4bit` | 6 | Candidate for attention-expansion experiment. |
| `lm_head` | `Linear` | 1 | Avoid for now. |

`gate_proj` was not present in the Nemotron module summary. It appeared in the local SmolLM run, so do not use `gate_proj` for Nemotron unless a future module audit shows it exists.

The preferred expanded-target experiment is:

```python
LORA_TARGET_MODULES = ["in_proj", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj"]
```

This adds 24 attention projection modules rather than thousands of `up_proj/down_proj` modules.

| Knob | Candidate Values | Critique | Current Recommendation |
| --- | --- | --- | --- |
| Training rows | `1024`, `None`, eval-free all rows | More data is the lowest-risk improvement. Eval-free all rows removes diagnostics. | Use `TRAIN_ROWS=None`, keep `EVAL_ROWS=256` for now. |
| Prompt style | `direct_raw_0_62`, `boxed_final`, `private_reasoning_boxed` | Prompt changes can affect extraction as much as reasoning. CoT-style prompts may waste tokens. | Start direct raw; test boxed/private reasoning separately. |
| Target format | `raw`, `boxed`, rationale + boxed | Raw partial scored 0.62, but raw full-data `S1` scored 0.54. Boxed matches metric but is unproven. Rationale targets need synthetic data quality. | Stop scaling raw-only SFT; prioritize boxed/trace experiments. |
| LoRA rank | `4`, `8`, `16`, `32` | Rank 4 worked. Larger rank gives capacity but costs memory/time and may overfit. Rank 32 is max allowed but too big a jump. | Try `r=8` next if capacity is the suspected limit. Do not jump to 32 yet. |
| LoRA alpha | `32`, `64`, `128` | Changing alpha changes update scale. If rank changes, alpha should usually change with it. | Keep alpha/r near baseline: r4/a32, r8/a64, r16/a128. |
| LR | `3e-4`, `2e-4`, `1e-4` | Lower LR may help larger ranks but adds another confound. | Keep `3e-4` until rank experiments show instability. |
| Epochs | `1`, `2` | Extra epochs can improve memorization but also overfit formatting quirks. | Stay at 1 epoch for full-data comparisons. |
| Sequence length | `512`, `768`, `1024` | Nemotron Mamba memory grows sharply with sequence length. Many answers are short. | Keep `512` until we prove truncation is common. |
| Target modules | `in_proj/out_proj`, attention expansion, MLP expansion | `in_proj/out_proj` is the known working target set. Attention expansion adds only `q/k/v/o` modules. `up_proj/down_proj` each appear 2967 times, so that is much more expensive. | First expanded target test should be `S4_attention_expand_r8_private_boxed`. |
| Dropout | `0.1`, `0.05`, `0.0` | Dropout may regularize small data; full data/rank changes might prefer lower dropout. | Keep `0.1` until rank/data effects are known. |

## Budget-Aware Path

If we only have two serious submissions:

| Done | Submission | Why |
| --- | --- | --- |
| [x] | `S1_raw_full_r4` | Scored `0.54`; more raw data alone did not help. |
| [ ] | `S4_attention_expand_r8_private_boxed_max128_drive` or `S6_short_trace_boxed_r8_attention_drive` | Because raw full-data degraded, move to metric-aligned boxed behavior and procedural traces rather than more raw-only SFT. |

If we have three serious submissions:

| Done | Submission | Why |
| --- | --- | --- |
| [x] | `S1_raw_full_r4` | Established full-data raw baseline; public score `0.54`. |
| [ ] | `S2_raw_full_r8` | Test capacity with minimal behavioral drift. |
| [ ] | `S3_boxed_full_r4_or_r8` | Test metric-aligned output as a separate axis. |

If we have four serious submissions, add:

| Done | Submission | Why |
| --- | --- | --- |
| [ ] | `S4_attention_expand_r8_private_boxed` | Tests whether capacity should be spread across Nemotron attention projection modules rather than only increasing rank. |

## Markdown Display Note

GitHub-style Markdown renders checkboxes reliably, but colors are not portable. Some Markdown viewers strip inline HTML styles, and GitHub does not allow arbitrary colored table cells. That is why this file uses stable marks like `[x]`, `[ ]`, `NEXT`, and `RISK` instead of color-coded rows.
