# Submission Tracking

Leaderboard uploads are archived under:

```text
data/outputs/submissions/
```

Each submission gets one folder named with the date, source, method, and score when known. Keep the submitted `submission.zip`, the source run bundle when available, and generated metadata together.

Current tracked submissions:

| Run folder | Public score | Source | Notes |
| --- | ---: | --- | --- |
| `2026-05-16_colab_nemotron_lora_score_0_62` | ~0.62 | Colab | Best known run so far. Nemotron LoRA, raw-answer SFT target, rank 4, `in_proj`/`out_proj`, `MAX_SEQ_LENGTH=512`, `MAX_NEW_TOKENS=64`, batch `2x16`. |
| `2026-05-16_local_smol_lora_score_0_50` | ~0.50 | Local PyCharm/Windows | Submission-mechanics control. Adapter config points to SmolLM, so do not treat it as a Nemotron modeling baseline. The notebook snapshot is closest available, not guaranteed exact pre-submit state. |
| `2026-05-17_colab_raw_full_r4_score_0_54` | 0.54 | Colab | Full-data raw-answer control. Training/eval looked clean, but score dropped below the 0.62 partial baseline; do not keep scaling raw-only final-answer SFT as the main path. |
| `2026-05-17_colab_s4_attention_boxed_r8_final_score_0_53` | 0.53 | Colab | S4 final adapter: boxed/private prompt, rank 8, expanded attention targets. It learned clean boxed format but scored below the 0.62 raw partial baseline and 0.54 raw full-data control; local current-193 generated eval was `95/256 = 0.371`, with strong numerals but weak cipher/equation/gravity. |

Generated files:

- `metadata.json`: score, method, hyperparameters, adapter summary, SHA-256.
- `adapter_config.json`: extracted adapter config from the submitted zip.
- `zip_contents.txt`: zip root contents and uncompressed sizes.
- `submissions_registry.csv`: one-row summary per tracked upload.

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
