# Submission Tracking

Leaderboard uploads are archived under:

```text
data/outputs/submissions/
```

## Submission Intake Protocol

When the user says a checkpoint or run is ready for submission, do not stop at
the Kaggle upload zip. Ask for or search for the artifacts needed for both
leaderboard upload and local tracking.

First separate the two goals:

- Kaggle upload needs only `submission.zip`.
- Dashboard/history needs diagnostics and metadata too.

Ask these questions if the answer is not already visible from the thread or
local files:

1. Which run/checkpoint is this?
   - Example: `exp05_trace_occam_r4_inout`, `checkpoint-96`.
2. Where is the Kaggle `submission.zip`?
   - It must contain only `adapter_config.json` and `adapter_model.safetensors`
     at the zip root.
3. Which local diagnostics should be archived with it?
   - `run_config.json`
   - `trainer_log_history.csv`
   - `probe_evolution.csv`
   - `generated_eval_summary.csv`
   - `generated_eval_predictions.csv` when available
   - `checkpoint_generated_eval_summary.csv` or checkpoint-specific eval files
   - `sanity_test_predictions_raw.csv` when available
4. Is this a pending submission or does it already have a Kaggle public score?
5. Should the local dashboard be updated now, or only after the public score
   returns?

If only `submission.zip` is available, it is still enough to upload to Kaggle,
but archive the run locally as diagnostics-pending and state which files are
missing from the dashboard record.

Dashboard public-score charts should use Kaggle scores when present. Local
diagnostic charts should use every run with saved metrics, including pending
submissions. If a submitted archive contains diagnostics for many checkpoints,
comparison charts should use only the actual submitted step recorded in
`metadata.json`/`run_config.json` or implied by the submitted checkpoint name.

Each submission gets one folder named with the date, source, method, and score when known. Keep the submitted `submission.zip`, the source run bundle when available, and generated metadata together.

Current tracked submissions:

| Run folder | Public score | Source | Notes |
| --- | ---: | --- | --- |
| `2026-05-16_colab_nemotron_lora_score_0_62` | ~0.62 | Colab | Best known run so far. Nemotron LoRA, raw-answer SFT target, rank 4, `in_proj`/`out_proj`, `MAX_SEQ_LENGTH=512`, `MAX_NEW_TOKENS=64`, batch `2x16`. |
| `2026-05-16_local_smol_lora_score_0_50` | ~0.50 | Local PyCharm/Windows | Submission-mechanics control. Adapter config points to SmolLM, so do not treat it as a Nemotron modeling baseline. The notebook snapshot is closest available, not guaranteed exact pre-submit state. |
| `2026-05-17_colab_raw_full_r4_score_0_54` | 0.54 | Colab | Full-data raw-answer control. Training/eval looked clean, but score dropped below the 0.62 partial baseline; do not keep scaling raw-only final-answer SFT as the main path. |
| `2026-05-17_colab_s4_attention_boxed_r8_final_score_0_53` | 0.53 | Colab | S4 final adapter: boxed/private prompt, rank 8, expanded attention targets. It learned clean boxed format but scored below the 0.62 raw partial baseline and 0.54 raw full-data control; local current-193 generated eval was `95/256 = 0.371`, with strong numerals but weak cipher/equation/gravity. |
| `2026-05-17_colab_s4_checkpoint144_score_0_55` | 0.55 | Colab checkpoint | S4 checkpoint-144 adapter: boxed/private prompt, rank 8, expanded attention targets. Local generated eval was `90/256 = 0.351562`, lower than current-193 aggregate, but public score improved over S4 final/current 0.53. Still below the 0.62 raw-answer baseline. |
| `2026-05-25_exp05_trace_occam_checkpoint96_mamba_score_0_58` | 0.58 | Colab checkpoint + Mamba backfill | Exp05 trace-training checkpoint 96, LoRA rank 4 on `in_proj/out_proj`, LR `1e-4`, trace/boxed targets. Submitted to Kaggle on 2026-05-25 and returned public score `0.58`. Local generated eval is `134/256 = 0.5234375`; probe is `3/5`. Strong unit conversion, gravity, and numeral; cipher remains `0/40`, bit/equation weak. |
| `2026-05-25_exp05_trace_occam_checkpoint144_mamba_score_0_59` | 0.59 | Colab checkpoint + Mamba backfill | Exp05 trace-training checkpoint 144, same adapter setup as checkpoint 96. Public score returned `0.59`, slightly above checkpoint 96 at `0.58`. Local generated eval is also `134/256 = 0.5234375`; probe is `3/5`. Relative to checkpoint 96 it gained one numeral row and lost one bit-manipulation row; cipher remains `0/40`, bit/equation weak. |
| `2026-05-25_exp06_mamba_fused_bf16_attention_r4_trace_v2_submitted_pending` | pending | Colab run bundle | Fast BF16 fused-Mamba attention-LoRA information-gathering submission on `trace_training_v2_2500.csv`. Adapter is valid (`r=4`, `q_proj/k_proj/v_proj/o_proj`) and the strict `submission.zip` was submitted, but generated eval is only `36/256 = 0.140625`; cipher `0/65`, bit `2/91`, equation `1/30`. Do not treat as a serious candidate unless the public score unexpectedly says otherwise. |

Generated files:

- `metadata.json`: score, method, hyperparameters, adapter summary, SHA-256.
- `metadata.json` should also include `kaggle_description` when a description is used in the Kaggle upload form.
- `adapter_config.json`: extracted adapter config from the submitted zip.
- `zip_contents.txt`: zip root contents and uncompressed sizes.
- `submissions_registry.csv`: one-row summary per tracked upload.

## Kaggle Descriptions

Explicit Kaggle upload descriptions were not recorded for the older submissions.
For those, only the local `method` and `notes` fields are available, so the
description can be reconstructed but not proven exact.

Most recent checkpoint-144 description:

```text
exp05 checkpoint144 trace SFT: canonical trace_training.csv, completion-only boxed targets, LoRA r4 alpha32 dropout0.05 on in_proj/out_proj, lr1e-4, seq512, max_new384. Local gen eval 134/256=0.5234; probe 3/5; Mamba backfill diagnostics. Checkpoint96 public=0.58.
```

When a new Kaggle upload finishes scoring, add the submitted zip and source run bundle to a new folder, then update the metadata and registry before changing the active notebooks.

For current Colab training runs, keep the run bundle. It should contain:

- `run_config.json`
- `probe_questions.csv`
- `probe_evolution.csv`
- `trainer_log_history.csv`
- `sanity_test_predictions.csv`
- `sanity_test_predictions_raw.csv`
- TensorBoard event files
- `adapter/adapter_config.json`
- `adapter/adapter_model.safetensors`
- `checkpoint_eval/` when checkpoint generated eval was enabled
