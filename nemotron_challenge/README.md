# NVIDIA Nemotron Model Reasoning Challenge

This repository is for a gradual Kaggle workflow around the NVIDIA Nemotron Model Reasoning Challenge.

The first goal is not to train immediately. The first goal is to make the project inspectable: understand the data, reproduce a simple baseline, record decisions, and only then move toward prompting, answer extraction, synthetic data, LoRA/SFT, or selection strategies.

## Current Milestone

Bootstrap the repository with:

- project memory and decision logs inspired by the PhysioNet project
- a small `src/` package for reusable logic
- dataset inspection utilities
- an initial reasoning-challenge roadmap

No competition data has been downloaded into this repository yet.

## 0) Context

Competition:

- Kaggle page: <https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge>
- Public description: improve reasoning techniques using NVIDIA Nemotron open models on a new benchmark.
- Public Kaggle/LinkedIn material says participants start from a Nemotron-3 Nano baseline and run on Google Cloud G4 VMs with NVIDIA RTX PRO 6000 Blackwell GPUs. Verify the exact runtime, rules, and allowed methods inside Kaggle after joining.

Public community artifacts suggest the benchmark contains structured reasoning families such as:

- `bit_manipulation`
- `gravity`
- `unit_conversion`
- `cipher`
- `numeral`
- `equation_symbolic`
- `equation_numeric`

Those categories are treated as provisional until confirmed from the official competition files.

## 1) High-Level Architecture

The project should grow in stages:

1. Data and schema inspection
   - load official Kaggle CSV files
   - detect columns and problem families
   - build small reports before modeling

2. Baseline reproduction
   - run the provided baseline notebook or model path
   - save prompts, raw completions, extracted answers, and scores
   - compare failures by problem type

3. Answer extraction and validation
   - make final-answer parsing deterministic and testable
   - separate model reasoning text from submission answer format
   - track extraction failures separately from reasoning failures

4. Category-specific reasoning
   - inspect each problem family independently
   - build lightweight solvers, validators, or prompt templates where useful
   - avoid one prompt/template pretending every family is the same

5. Training or adaptation
   - only after the baseline and error map are clear
   - likely candidates: LoRA/SFT, synthetic data, retry/selection, and verifier-guided generation

6. Submission assembly
   - produce a strict `submission.zip` containing a compatible Nemotron LoRA adapter
   - keep generation traces and scoring reports outside the submission artifact

## 2) Repo Layout

Planned layout:

- `config/`: small YAML configuration files
- `data/raw/`: official competition files, ignored by git
- `data/interim/`: parsed intermediate artifacts, ignored by git
- `data/processed/`: model-ready derived files, ignored by git
- `doc/`: project memory, decisions, and important error notes
- `notebooks/`: exploration and visual reports
- `scripts/`: small command-line helpers
- `src/nemotron_challenge/`: reusable project code
- `outputs/`: local runs, traces, and reports, ignored by git

## 3) Early Strategy

The safest first technical bets are:

- Treat problem type as a first-class feature.
- Preserve every raw model completion before extracting the final answer.
- Build a strict local answer extractor early.
- Measure errors by category, not only by overall score.
- Start with simple deterministic diagnostics before using GPU time.

For this competition, the likely failure split is:

- model reasons incorrectly
- model reasons correctly but outputs the wrong final format
- answer extractor fails
- retry/selection chooses the wrong candidate
- category-specific rule is misunderstood

Those need separate reports.

## 4) Validation Strategy

Initial validation should answer:

- How many rows are in train/test?
- What are the exact official column names?
- Which problem families exist?
- Is there a local train score we can reproduce?
- How often does answer extraction fail independently of model accuracy?
- Which categories are easiest to improve without fine-tuning?

Do not start by optimizing one global prompt without this breakdown.

## 5) Practical Roadmap

Milestone A: official data is downloaded and inspected.

Milestone B: baseline output is reproduced and logged with raw completions.

Milestone C: category-level error analysis identifies the first target family.

Milestone D: one small intervention improves local validation for one family.

Milestone E: submission generation becomes deterministic and repeatable.

## 6) Repo Notes

This repo intentionally mirrors useful habits from the PhysioNet project:

- `AGENTS.md` for working rules
- `doc/LOCAL_PROJECT_MEMORY.md` for current state
- `doc/PROJECT_DECISION_LOG.md` for durable decisions
- `doc/errors.md` for non-trivial failures and fixes
- `src/` for stable logic
- notebooks only for exploration and reporting

## TL;DR

Begin with data visibility, baseline reproduction, and answer extraction. Then improve one reasoning family at a time.
