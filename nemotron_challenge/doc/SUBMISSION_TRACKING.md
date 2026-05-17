# Submission Tracking

Leaderboard uploads are archived under:

```text
outputs/submissions/
```

Each submission gets one folder named with the date, source, method, and score when known. Keep the submitted `submission.zip`, the closest notebook snapshot, and generated metadata together.

Current tracked submissions:

| Run folder | Public score | Source | Notes |
| --- | ---: | --- | --- |
| `2026-05-16_colab_nemotron_lora_score_0_62` | ~0.62 | Colab | Best known run so far. Nemotron LoRA, raw-answer SFT target, rank 4, `in_proj`/`out_proj`, `MAX_SEQ_LENGTH=512`, `MAX_NEW_TOKENS=64`, batch `2x16`. |
| `2026-05-16_local_smol_lora_score_0_50` | ~0.50 | Local PyCharm/Windows | Submission-mechanics control. Adapter config points to SmolLM, so do not treat it as a Nemotron modeling baseline. The notebook snapshot is closest available, not guaranteed exact pre-submit state. |

Generated files:

- `metadata.json`: score, method, hyperparameters, adapter summary, SHA-256.
- `adapter_config.json`: extracted adapter config from the submitted zip.
- `zip_contents.txt`: zip root contents and uncompressed sizes.
- `submissions_registry.csv`: one-row summary per tracked upload.

When a new Kaggle upload finishes scoring, add the zip and notebook snapshot to a new folder, then update the metadata and registry before changing the active notebooks.

For Colab training runs, also keep the diagnostics zip when available. It should contain:

- `run_config.json`
- `probe_questions.csv`
- `probe_evolution.csv`
- `probe_evolution.jsonl`
- `trainer_log_history.csv`
- `sanity_test_predictions.csv`
- `sanity_test_predictions_raw.csv`
- TensorBoard event files
- `adapter_config.json`
- the submitted `submission.zip`
