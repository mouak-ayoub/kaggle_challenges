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

Remark: all tracked experiments have failed the three public `test.csv` sanity rows without exception. These rows remain useful for spotting obvious behavior, but passing local loss/probe/generated-eval checks has not made any submitted adapter solve all three public sanity prompts.

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
| `2026-05-25_exp06_mamba_fused_bf16_attention_r4_trace_v2_submitted_pending` | 0.58 | Colab run bundle | Fast BF16 fused-Mamba attention-LoRA information-gathering submission on `trace_training_v2_2500.csv`. Adapter is valid (`r=4`, `q_proj/k_proj/v_proj/o_proj`) and generated eval is only `36/256 = 0.140625`; cipher `0/65`, bit `2/91`, equation `1/30`. Public score returned `0.58`, close to exp05 checkpoint96 but still below the 0.62 baseline. |
| `2026-05-26_exp10_mamba_cipher_v4_natural2500_b12_seq1024_ep3_ready` | 0.49 | Colab run bundle | Cipher-only natural-alignment diagnostic. Local cipher generated eval was `26/64 = 0.40625`, but public score returned `0.49`, confirming single-family cipher fine-tuning is not leaderboard-safe. |
| `2026-05-26_exp11_mamba_trace_v2_aug25k_checkpoint195_pending` | 0.56 | Colab checkpoint | Broad 25k v2 trace Mamba checkpoint 195, rank 14 attention LoRA. Public score returned `0.56`. Backfill generated eval is `11/64 = 0.171875`, with bit `0/20`, cipher `3/26`, equation `1/8`, gravity `3/3`, numeral `1/3`, unit `3/4`, and one max-token hit. This scored better publicly than checkpoint 390 despite worse local generated eval. |
| `2026-05-26_exp11_mamba_trace_v2_aug25k_checkpoint390_ready` | 0.54 | Colab checkpoint | Broad 25k v2 trace Mamba checkpoint 390, rank 14 attention LoRA. Public score returned `0.54`; backfill generated eval is `17/64 = 0.265625`, with bit `4/20`, cipher `3/26`, equation `1/8`, gravity `3/3`, numeral `3/3`, unit `3/4`, and one max-token hit. Scaling v2 trace format did not beat exp05 checkpoint144 or the 0.62 raw partial baseline. |
| `2026-05-27_exp11_mamba_trace_v2_aug25k_checkpoint585_ready` | 0.56 | Colab checkpoint | Broad 25k v2 trace Mamba checkpoint 585, rank 14 attention LoRA. Public score returned `0.56`, matching checkpoint 195 and beating checkpoint 390. No checkpoint-585 generated-eval diagnostics were included in the download. |
| `2026-05-27_exp11_mamba_trace_v2_aug25k_checkpoint780_ready` | 0.54 | Colab checkpoint | Broad 25k v2 trace Mamba checkpoint 780, rank 14 attention LoRA. Public score returned `0.54`, matching checkpoint 390 and below checkpoints 195 and 585 at `0.56`. No checkpoint-780 generated-eval diagnostics were included in the download. |
| `2026-05-27_exp12_mamba_trace_v2_aug25k_inout_qkvo_r4_ep1_score_0_54` | 0.54 | Colab run bundle | Broad 25k v2 trace run with fused BF16 Mamba and LoRA `r=4` on `in_proj/out_proj/q_proj/k_proj/v_proj/o_proj`. Local final generated eval was strong at `36/64 = 0.5625`, with cipher `18/26`, bit `3/20`, equation `5/8`, and probe `5/5`, but public score returned only `0.54`. |
| `2026-05-27_exp12_mamba_trace_v2_aug25k_5k_inout_vo_checkpoint26_score_0_57` | 0.57 | Colab checkpoint | 5k-row v2 trace diagnostic, checkpoint 26, submitted from downloaded strict `submission.zip`. The zip is valid, but `adapter_config.json` reports LoRA `r=9` even though the intended thread config was `LORA_R=4`; targets are `in_proj/v_proj/o_proj/out_proj`. Public score returned `0.57`, better than exp12 full all-six at `0.54` but still below exp05 checkpoint144 at `0.59`. |
| `2026-05-27_exp12_mamba_trace_v2_aug25k_5k_inout_vo_checkpoint105_score_0_58` | 0.58 | Colab checkpoint | Final/checkpoint-105 entry for the same 5k-row v2 trace run, submitted from a newer downloaded strict `submission.zip`. The folder also contains the downloaded final run bundle as related diagnostics; for dashboard tracking this is treated as the same last-step submission. Public score returned `0.58`, beating checkpoint 26 at `0.57` and matching exp05 checkpoint96. |
| `2026-05-27_exp12_mamba_trace_v2_aug25k_5k_inout_vo_final105_local_rejected` | not submitted | Colab run bundle | Final step-105 diagnostics from the same 5k-row run. Final generated eval is only `8/64 = 0.125`, with cipher `0/26`, bit `0/20`, and `21` max-token hits. The final bundle is kept as related diagnostics and excluded from the dashboard because the submitted last-step archive above carries the Kaggle score. |
| `2026-05-28_exp13_trace_v2_aug25k_10k_inout_r25_checkpoint78_score_0_57_reported` | 0.57 | Downloaded strict submission zip | Exp13 checkpoint 78 from the 10k-row dense-BF16 Mamba `in_proj/out_proj` rank-25 run. The strict `submission.zip` was found in Downloads and archived locally; SHA-256 is `8be6648cd722f2c91ad3c7407bc1fe57812e261dcadad6c2fdabe511f4b6004c`. Trainer eval at step 78 was `eval_loss=0.068001` and token accuracy `0.976612`. |
| `2026-05-28_exp13_trace_v2_aug25k_10k_inout_r25_ep1_submitted_pending` | 0.57 | Colab run bundle | 10k-row v2 trace dense-BF16 Mamba capacity test with LoRA `r=25`, alpha `128`, dropout `0.05`, targets `in_proj/out_proj` only. Strict `submission.zip` was built from the run bundle; adapter audit found `46` LoRA modules and `11,371,200` trainable params. Local final generated eval is `35/64 = 0.546875`, with cipher `18/26`, bit `4/20`, equation `3/8`, easy families perfect on tiny samples, probe `4/5`, and `3` max-token hits. Public score returned `0.57`, matching checkpoint 78. |

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
