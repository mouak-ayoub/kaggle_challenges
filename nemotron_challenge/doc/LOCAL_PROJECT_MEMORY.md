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
- Updated local and Colab notebooks to train on `\boxed{answer}` targets and use boxed answer extraction for sanity checks.
- Confirmed Kaggle accepted two adapter submissions:
  - local small-model test adapter scored about 0.50
  - Colab Nemotron adapter scored about 0.62

## Open Questions

- Does Kaggle provide a starter notebook or only data files?
- Is local validation possible from the official train file?
- Which problem family should be attacked first?
- Can a 30B Nemotron LoRA run fit in the intended Colab GPU environment, or should the first LoRA dry run use a smaller model?
