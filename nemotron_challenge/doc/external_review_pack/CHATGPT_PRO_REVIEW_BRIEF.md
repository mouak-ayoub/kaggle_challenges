# Nemotron Kaggle Challenge - External Review Brief

Date prepared: 2026-05-18

## Request To Reviewer

Please review the experiment history below as an independent technical critic.

Your tasks:

1. Analyze the experimental design and results.
2. Identify weak assumptions, leakage risks, metric/proxy problems, and missing controls.
3. Explain what the results do and do not support.
4. Propose future runs, methods, or analysis that you believe are most likely to improve the Kaggle score.

Important: this brief intentionally does not include our own next-step conclusions. Please infer your own conclusions from the evidence.

## Challenge Context

Competition: NVIDIA Nemotron Model Reasoning Challenge on Kaggle.

Known submission format:

- Kaggle expects a LoRA adapter packaged as `submission.zip`.
- `submission.zip` must contain adapter files at the zip root, especially:
  - `adapter_config.json`
  - `adapter_model.safetensors`
- This is not a normal `submission.csv` prediction competition.
- Local test generation against Kaggle `test.csv` is only a sanity check; leaderboard scoring loads/evaluates the submitted adapter.

Official local data available:

- `train.csv`: 9,500 rows
- `test.csv`: 3 public sanity rows
- Main columns in both CSVs:
  - `id`
  - `prompt`
  - `answer` only in train

Problem families are not supplied as an official CSV column. We infer families from prompt patterns for diagnostics:

- `bit_manipulation`
- `cipher`
- `equation`
- `gravity`
- `numeral`
- `unit_conversion`

## Evidence Included In This Brief

This brief summarizes:

- Public score curve by submitted adapter
- Main training configuration for each completed run
- Local generated-eval accuracy by run/checkpoint
- Local generated-eval accuracy by inferred problem family
- Probe accuracy by run/checkpoint
- Training/eval loss by run/checkpoint where available
- Saved sanity responses on the 3 public `test.csv` rows

## Evaluation Artifacts Used Locally

### Kaggle Public Score

Public leaderboard score from Kaggle after uploading `submission.zip`.

Known caveat from competition discussions:

- The public scoring path uses vLLM and may not be perfectly deterministic even at `temperature=0.0`.
- Small public score differences may require caution.

### Public Test Sanity Rows

The official `test.csv` contains 3 public sanity examples. The expected answers recorded for these examples are:

| id       | family           | expected answer   |
|----------|------------------|-------------------|
| 00066667 | bit_manipulation | 10010111          |
| 000b53cf | bit_manipulation | 01000011          |
| 00189f6a | cipher           | cat imagines book |

These rows are not a meaningful validation set; they are only used to inspect obvious response behavior.

### Fixed Generated Eval Split

For local diagnostics, we use a fixed 256-row held-out eval split from training data.

For each adapter/checkpoint, we generate answers and score exact extracted-answer matches against known train answers.

The generated-eval summary records:

- `rows`
- `matches`
- `misses`
- `accuracy`
- `empty_answers`
- `hit_max_new_tokens`
- `avg_generated_tokens`
- `avg_seconds`
- same metrics by inferred family

Important limitation:

- This 256-row generated eval is local and diagnostic.
- It is not the hidden Kaggle test set.
- It may not match Kaggle public leaderboard ordering.

### Probe Set

A fixed 5-row probe set is used for some training runs and local diagnostics.

Probe output records:

- raw generation
- extracted answer
- gold answer
- match boolean
- step/checkpoint

The probe is too small for score estimation. It is used to track behavior changes during a run.

## Submitted And Candidate Runs

### Submission Registry

| id                                                        | public score | source              | method label                                               | model                                              | target format | max seq | max new | train rows | LoRA r | LoRA alpha | LoRA dropout | LoRA modules                                | effective batch | LR     | status/purpose |
|-----------------------------------------------------------|--------------|---------------------|------------------------------------------------------------|----------------------------------------------------|---------------|---------|---------|------------|--------|------------|--------------|---------------------------------------------|-----------------|--------|----------------|
| `2026-05-16_colab_nemotron_lora_score_0_62`               | 0.62         | Colab               | Nemotron LoRA SFT raw-answer baseline                      | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`       | raw           | 512     | 64      | 1024       | 4      | 32         | 0.1          | `in_proj`, `out_proj`                       | 32              | 3e-4   | scored |
| `2026-05-16_local_smol_lora_score_0_50`                   | 0.50         | local Windows       | SmolLM local adapter smoke-test baseline                   | `HuggingFaceTB/SmolLM2-135M-Instruct`             | raw           | 256     | 32      | 1000       | 4      | 32         | 0.1          | `q_proj`, `v_proj`                          | 32              | 3e-4   | scored / non-Nemotron control |
| `2026-05-17_colab_raw_full_r4_score_0_54`                 | 0.54         | Colab               | Nemotron LoRA SFT raw-answer full-data control             | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`       | raw           | 512     | 64      | 9244       | 4      | 32         | 0.1          | `in_proj`, `out_proj`                       | 48              | 3e-4   | scored |
| `2026-05-17_colab_s4_attention_boxed_r8_final_score_0_53` | 0.53         | Colab               | S4 expanded-attention private-reasoning boxed final adapter | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`       | boxed         | 512     | 128     | 9244       | 8      | 64         | 0.1          | `in_proj`, `out_proj`, `q_proj`, `k_proj`, `v_proj`, `o_proj` | 48 | 3e-4 | scored |
| `2026-05-17_colab_s4_checkpoint144_score_0_55`             | 0.55         | Colab checkpoint    | S4 checkpoint-144 candidate adapter                         | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`       | boxed         | 512     | 128     | 9244       | 8      | 64         | 0.1          | `o_proj`, `in_proj`, `out_proj`, `v_proj`, `q_proj`, `k_proj` | 48 | 3e-4 | scored |

## Main Training Config Details

### `00-raw-1024`: Colab Nemotron Raw Baseline, Public Score 0.62

Configuration:

```text
model_name: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
system_prompt: Solve the puzzle and output only the final answer value. Do not explain. Do not write prefixes like 'answer:' or 'the answer is'.
train_target_format: raw
max_seq_length: 512
max_new_tokens: 64
train_rows: 1024
eval_rows: 256
lora_r: 4
lora_alpha: 32
lora_dropout: 0.1
lora_target_modules: in_proj, out_proj
per_device_train_batch_size: 2
gradient_accumulation_steps: 16
effective_batch_size: 32
epochs: 1
learning_rate: 0.0003
precision_mode: bf16_fast
```

Diagnostics:

- Submitted adapter scored `0.62`.
- Generated eval and probe diagnostics were computed from the same submitted adapter.

### `02-raw-full`: Colab Nemotron Raw Full-Data Control, Public Score 0.54

Configuration:

```text
model_name: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
prompt_experiment: direct_raw_0_62
train_target_format: raw
max_seq_length: 512
max_new_tokens: 64
train_rows_actual: 9244
eval_rows: 256
lora_r: 4
lora_alpha: 32
lora_dropout: 0.1
lora_target_modules: in_proj, out_proj
per_device_train_batch_size: 12
gradient_accumulation_steps: 4
effective_batch_size: 48
epochs: 1
learning_rate: 0.0003
precision_mode: bf16_fast
```

Diagnostic availability:

- Submitted adapter scored `0.54`.
- Generated eval and probe diagnostics were computed later from the same submitted adapter.

### `04-s4`: S4 Expanded-Attention Private-Reasoning Boxed Run, Public Score 0.53

Configuration:

```text
experiment_name: S4_attention_expand_r8_private_boxed_max128_drive
model_name: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
prompt_experiment: private_reasoning_boxed
system_prompt: Solve the puzzle carefully. Reason internally if needed, but write only the final answer inside \boxed{} with no trailing text.
train_target_format: boxed
max_seq_length: 512
max_new_tokens: 128
train_rows_actual: 9244
eval_rows: 256
probe_rows: 5
lora_r: 8
lora_alpha: 64
lora_dropout: 0.1
lora_target_modules: in_proj, out_proj, q_proj, k_proj, v_proj, o_proj
per_device_train_batch_size: 16
gradient_accumulation_steps: 3
effective_batch_size: 48
epochs: 1
learning_rate: 0.0003
precision_mode: bf16_fast
```

Available checkpoints:

```text
checkpoint-48
checkpoint-96
checkpoint-144
checkpoint-192
checkpoint-193
adapter/final
```

Scored public submission:

- Final/current adapter scored `0.53`.

Local checkpoint generated-eval summaries are available for:

- `checkpoint-96`
- `checkpoint-144`
- `current-193`

### S4 Checkpoint-144 Candidate

Configuration:

```text
experiment_name: S4_attention_expand_r8_private_boxed_max128_drive
checkpoint: checkpoint-144
step: 144
model_name: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
prompt_experiment: private_reasoning_boxed
train_target_format: boxed
max_seq_length: 512
max_new_tokens: 128
train_rows_actual: 9244
eval_rows: 256
lora_r: 8
lora_alpha: 64
lora_dropout: 0.1
lora_target_modules: o_proj, in_proj, out_proj, v_proj, q_proj, k_proj
per_device_train_batch_size: 16
gradient_accumulation_steps: 3
effective_batch_size: 48
epochs: 1
learning_rate: 0.0003
```

Status:

- Adapter submitted and scored `0.55`.
- It scored above S4 final/current `0.53`, but still below the `00-raw-1024` public `0.62` baseline.

## Public Leaderboard Scores

| run label    | public score |
|--------------|--------------|
| `00-raw-1024` | 0.62 |
| `smol`        | 0.50 |
| `02-raw-full` | 0.54 |
| `04-s4`       | 0.53 |
| `04-s4-step-144` | 0.55 |

The S4 checkpoint-144 candidate scored `0.55`, so checkpoint timing improved S4 slightly but did not close the gap to the 0.62 raw baseline.

## Local Generated Eval Results

All rows below use the same fixed 256-row generated-eval split.

| run/checkpoint | matches | rows | accuracy |
|----------------|---------|------|----------|
| `00-raw-1024` | 65 | 256 | 0.25390625 |
| `02-raw-full` | 87 | 256 | 0.33984375 |
| `04-s4-step-96` | 79 | 256 | 0.30859375 |
| `04-s4-step-144` | 90 | 256 | 0.3515625 |
| `04-s4-step-193` | 95 | 256 | 0.37109375 |

Same values as percentages:

| run/checkpoint | generated eval accuracy |
|----------------|-------------------------|
| `00-raw-1024` | 25.390625% |
| `02-raw-full` | 33.984375% |
| `04-s4-step-96` | 30.859375% |
| `04-s4-step-144` | 35.15625% |
| `04-s4-step-193` | 37.109375% |

## Local Generated Eval By Family

Percent accuracy by inferred family:

| family | `00-raw-1024` | `02-raw-full` | `04-s4-step-96` | `04-s4-step-144` | `04-s4-step-193` |
|--------|---------------|---------------|-----------------|------------------|------------------|
| bit_manipulation | 4.4444% | 24.4444% | 13.3333% | 22.2222% | 31.1111% |
| cipher | 0.0000% | 9.3023% | 11.6279% | 16.2791% | 11.6279% |
| equation | 5.1282% | 12.8205% | 12.8205% | 15.3846% | 12.8205% |
| gravity | 2.4390% | 4.8780% | 2.4390% | 2.4390% | 4.8780% |
| numeral | 100.0000% | 100.0000% | 100.0000% | 100.0000% | 100.0000% |
| unit_conversion | 22.2222% | 36.1111% | 27.7778% | 38.8889% | 47.2222% |

Family row counts in the 256-row generated eval:

| family | rows |
|--------|------|
| bit_manipulation | 45 |
| cipher | 43 |
| equation | 39 |
| gravity | 41 |
| numeral | 52 |
| unit_conversion | 36 |

## Probe Results

Probe match rate on the fixed 5-row probe set:

| run/checkpoint | probe matches | probe rows | probe accuracy |
|----------------|---------------|------------|----------------|
| `00-raw-1024` | 1 | 5 | 0.20 |
| `02-raw-full` | 1 | 5 | 0.20 |
| `04-s4-step-96` | 2 | 5 | 0.40 |
| `04-s4-step-144` | 2 | 5 | 0.40 |
| `04-s4-step-193` | 1 | 5 | 0.20 |

## Training And Eval Loss Points

Available loss values from archived logs:

| run/checkpoint | train loss | eval loss |
|----------------|------------|-----------|
| `02-raw-full` | 3.4933451123805863 | 0.7932660579681396 |
| `04-s4-step-96` | 2.3756311734517417 | 0.7917672395706177 |
| `04-s4-step-144` | 2.2357492446899414 | 0.7761566638946533 |
| `04-s4-step-193` | 2.4857180946231505 | 0.7694065570831299 |

## Public Test Sanity Responses

Saved responses to the 3 official public sanity rows:

Prompt text:

| id | family | prompt |
|----|--------|--------|
| `00066667` | bit_manipulation | `In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers. The transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT, and possibly majority or choice functions. Examples: 01010001 -> 11011101; 00001001 -> 01101101; 00010101 -> 01010101; 11111111 -> 10000001; 10011101 -> 01000101; 00111011 -> 00001001; 10111101 -> 00000101; 00100110 -> 10110011. Determine the output for: 00110100` |
| `000b53cf` | bit_manipulation | `In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers. The transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT, and possibly majority or choice functions. Examples: 10001110 -> 00100110; 10011001 -> 01000100; 01100100 -> 00010001; 10000010 -> 00001010; 00011011 -> 01001100; 00111010 -> 10011100; 01101111 -> 00110111; 10010110 -> 01011010; 00001010 -> 00101100. Determine the output for: 11100000` |
| `00189f6a` | cipher | `In Alice's Wonderland, secret encryption rules are used on text. Examples: ucoov pwgtfyoqg vorq yrjjoe -> queen discovers near valley; pqrsfv pqorzg wvgwpo trgbjo -> dragon dreams inside castle; gbcpovb tqorbog bxo zrswtrj pffq -> student creates the magical door; bxo sfjpov pqrsfv dfjjfig -> the golden dragon follows; nqwvtogg qorpg bxo zegboqwfcg gotqob -> princess reads the mysterious secret. Decrypt: trb wzrswvog hffk` |

Saved model answers:

| id | family | expected | `00-raw-1024` answer | `02-raw-full` answer | `04-s4` answer |
|----|--------|----------|----------------------|----------------------|----------------|
| `00066667` | bit_manipulation | `10010111` | `01000101` | `01000011` | `00001101` |
| `000b53cf` | bit_manipulation | `01000011` | `10000100` | `00110000` | `00100000` |
| `00189f6a` | cipher | `cat imagines book` | `queen creates the secret door` | `cat explores book` | `cat explores book` |

Match counts:

| run | correct | total |
|-----|---------|-------|
| `00-raw-1024` | 0 | 3 |
| `02-raw-full` | 0 | 3 |
| `04-s4` | 0 | 3 |

## Known Missing Or Partial Items

- The 5-row probe set is not statistically meaningful; it is a behavior trace.
- The 3 public test sanity rows are not statistically meaningful; they are visible sanity examples.
- `smol` is included as a small non-Nemotron control, not as a direct Nemotron-method comparison.
- S4 checkpoint-144 scored `0.55`; this is better than S4 final/current `0.53` but still below the 0.62 raw-answer baseline.

## What To Analyze

Please analyze from the evidence above:

- Whether the experiment comparisons are fair.
- Whether the local generated-eval setup is useful, misleading, or needs redesign.
- Whether the differences between public score and local diagnostics suggest data-split, prompt, extraction, family-distribution, or scoring-proxy problems.
- Whether training more data, changing output format, increasing LoRA rank/modules, or using boxed targets is supported or contradicted by the evidence.
- Which experiment should be treated as baseline for future work.
- What concrete next methods or runs should be prioritized.
