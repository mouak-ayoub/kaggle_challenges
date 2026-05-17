# Important Errors

Last updated: 2026-05-16

This file records non-trivial errors we hit so the next session does not rediscover them.

## Windows UTF-8 / TRL Import

Symptom:

```text
utf8 mode: 0
preferred encoding: cp1252
UnicodeDecodeError / charmap decode error while importing TRL
```

Cause:

On this Windows Jupyter kernel, Python defaults to `cp1252`, not UTF-8. Setting `PYTHONUTF8=1` inside a notebook cell is too late because Python chooses UTF-8 mode when the kernel process starts.

Lesson:

Start Jupyter with `PYTHONUTF8=1`, or keep the small notebook-local `pathlib.Path.read_text` UTF-8 patch before importing TRL. Do not use the `os.execv` restart trick in Jupyter; it can hang or disconnect the kernel.

## Local Windows Mamba Install

Symptom:

```text
pip install causal-conv1d mamba-ssm --no-build-isolation
subprocess-exited-with-error
```

Cause:

The Mamba CUDA packages are not a good fit for the local Windows virtualenv. They are meant for the GPU Linux runtime.

Lesson:

Do Nemotron/Mamba training in Colab or another Linux GPU environment. Keep local Windows for data checks, notebook editing, and small dry runs.

## Full Nemotron Load OOM

Symptom:

Loading or moving the full Nemotron 30B BF16 model to GPU runs out of memory.

Cause:

The base model is too large for normal full-precision fine-tuning in the available environment.

Lesson:

Use 4-bit loading with `BitsAndBytesConfig`. Do not call `model.to("cuda")` after quantized loading; let `from_pretrained(..., quantization_config=..., device_map=...)` place the model.

## `device_map="auto"` And 4-Bit Training

Symptom:

4-bit loading can complain about CPU/disk offload or place layers in a way that is awkward for LoRA training.

Cause:

`device_map="auto"` is useful for very large inference, but for single-GPU LoRA training it may offload parts of the model.

Lesson:

If the model fits on one GPU, prefer:

```python
device_map = {"": 0}
```

Use `auto` only when deliberately doing multi-device/offload work.

## Fused Mamba Fast Path Shape Error

Symptom:

```text
RuntimeError: mat1 and mat2 shapes cannot be multiplied (4608x4096 and 1x5505024)
```

Cause:

With `causal-conv1d` installed, Nemotron uses a fused Mamba path. That path passed a packed bitsandbytes 4-bit projection weight to a kernel expecting a normal matrix.

Lesson:

For 4-bit Nemotron training, avoid the fused path by not installing `causal-conv1d` in the training notebook. Installing `mamba-ssm` alone is okay. The warning that the fast path is unavailable is acceptable here.

## Generation `cache_position` Error

Symptom:

```text
TypeError: 'NoneType' object is not subscriptable
cache_position[-1]
```

Cause:

The remote Nemotron model code expected generation cache metadata, but `cache_position` was `None` in the current Transformers path.

Lesson:

For notebook sanity inference, call generation with:

```python
use_cache=False
```

and also set:

```python
model.config.use_cache = False
model.generation_config.use_cache = False
```

## BF16 MoE `index_add_` Dtype Mismatch

Symptom:

```text
RuntimeError: index_add_(): self (BFloat16) and source (Float) must have the same scalar type
```

Cause:

In the Nemotron MoE path, the accumulator was BF16 while an expert-weighted temporary tensor became FP32, probably through router/top-k weights. `index_add_` requires matching dtypes.

Lesson:

HF precision flags do not force every temporary tensor in custom remote model code. BF16 is still the right memory target on A100/H100, but this specific MoE runtime tensor may need a small dtype cast if the error returns.

Current bypass:

Use BF16 settings, but patch only the Nemotron MoE `moe` method so `weighted_output` is cast to `final_hidden_states.dtype` immediately before `index_add_`:

```python
weighted_output = weighted_output.to(final_hidden_states.dtype)
final_hidden_states.index_add_(0, token_indices, weighted_output)
```

Do not cast the whole 4-bit model. The issue is a temporary tensor inside custom model code, not the packed `Linear4bit` weights.

## FP32 Avoids Dtype Mismatch But Costs Memory

Symptom:

Switching compute/training to FP32 avoids the BF16 `index_add_` error, then training OOMs later.

Cause:

FP32 doubles activation and temporary tensor memory compared with BF16/FP16. Nemotron's naive Mamba path creates large temporary tensors.

Lesson:

FP32 is a debugging workaround, not the desired training precision. Prefer BF16 on A100/H100 if the MoE dtype mismatch is fixed or avoided.

## PEFT `load_adapter` Fails On Nemotron Checkpoint Eval

Symptom:

```text
TypeError: WeightConverter.__init__() got an unexpected keyword argument 'distributed_operation'
```

Cause:

In the Colab PEFT/Transformers stack, calling `current_model.load_adapter(...)` on the already wrapped Nemotron PEFT model can hit a conversion-mapping incompatibility while loading saved LoRA checkpoints.

Lesson:

Do not rerun an expensive completed eval if the failure happens after predictions were appended in notebook memory. First save the partial `all_predictions` / `all_summaries` result. For checkpoint comparison, avoid `load_adapter` and directly load `adapter_model.safetensors` into the existing active LoRA adapter by mapping saved keys like `.lora_A.weight` to the in-memory keys like `.lora_A.default.weight`.

## Naive Mamba Path OOM At Long Sequence Length

Symptom:

```text
OutOfMemoryError: Tried to allocate 18.00 GiB
```

The failure happened in `modeling_nemotron_h.py` inside the torch Mamba forward path.

Cause:

Disabling the fused path makes Nemotron use the slower naive Mamba implementation. Sequence length drives the size of its temporary tensors. `MAX_SEQ_LENGTH=1024` was too expensive in this setup.

Lesson:

Use `MAX_SEQ_LENGTH=256` for the first debug step, then try `512` for real training. Row count affects total time, but sequence length affects first-step memory.

Also keep `per_device_train_batch_size=1`. A larger micro-batch such as 32 can make the naive Mamba temporary tensors explode even in BF16. Use gradient accumulation for effective batch size instead:

```python
per_device_train_batch_size = 1
gradient_accumulation_steps = 32
per_device_eval_batch_size = 1
```

Do not trust the GPU memory shown immediately after loading the model. Training forward/backward creates additional peak temporary tensors; an H100/A100 can look fine before `trainer.train()` and still OOM inside the first Mamba layer.

## Kaggle Submission Is Adapter Zip, Not CSV

Symptom:

The early notebook produced `submission.csv`, matching normal Kaggle prediction-file workflows.

Cause:

This competition is different. The Kaggle Evaluation page says submissions must be a LoRA adapter packaged as `submission.zip`, containing at least `adapter_config.json`, rank at most 32.

Lesson:

Use CSV generation only for local sanity checks. The real leaderboard artifact is:

```text
submission.zip
  adapter_config.json
  adapter_model.safetensors
```

## Output Format Drift During SFT

Symptom:

After local SFT, the model still generated text like explanations, prefixes, or unrelated prose instead of the short training answers.

Cause:

The training targets are short final answers, but the system prompt was too weak and generation allowed too many new tokens. A chat model may keep using its instruction/chat prior unless the prompt is strict.

Lesson:

For local sanity checks, use a strict final-answer-only prompt and keep `MAX_NEW_TOKENS` small. The metric prioritizes `\boxed{...}`, but the best observed submission so far was trained on raw short-answer targets. Treat boxed-target training as a separate experiment until it beats the raw-answer baseline.

## Kaggle vLLM Scoring Is Not Deterministic

Symptom:

Participants report different public LB scores for the same `submission.zip`, including differences around several hundredths in some cases.

Cause:

Kaggle discussion confirms the evaluation uses vLLM and is not fully deterministic even with `temperature=0.0`. The host said deterministic settings would reduce throughput too much and would not remove all order effects.

Lesson:

Do not over-interpret one public score. Use local category-level validation and raw completion logs. Treat small LB changes as noisy unless they are reproduced or backed by local evidence.

## Boxed Answer Brace Edge Cases

Symptom:

Answers containing literal `}` were not extracted correctly by the earlier metric. Outputs with extra closing braces can also score differently after the metric update.

Cause:

The original extractor stopped at the first `}` inside `\boxed{...}`. The updated official metric takes content up to the last closing brace before the next boxed answer or end of text.

Lesson:

If testing boxed-output training or prompting, train the model to output exactly one final `\boxed{...}` answer and no trailing text. Keep raw completions so extraction failures can be separated from reasoning failures.

## Kaggle External Training Submission

Symptom:

Participants asked whether training must happen inside a Kaggle notebook and what should be uploaded.

Cause:

This challenge evaluates the submitted adapter, not the training notebook. External training is allowed as long as the final adapter is compatible with the official Nemotron base model.

Lesson:

The final artifact should be a `submission.zip` with required adapter files at the zip root:

```text
adapter_config.json
adapter_model.safetensors
```

If the artifact is large or browser upload/download is unreliable, use Kaggle CLI submission from a persisted Kaggle session or local machine rather than rerunning training only to submit.

## Kaggle Utility Script Import Order

Symptom:

Kaggle notebooks can fail with errors such as:

```text
PermissionError: [Errno 13] Permission denied: .../ptxas-blackwell
ModuleNotFoundError: No module named 'cutlass'
ModuleNotFoundError: No module named 'mamba_ssm'
```

Cause:

The Blackwell utility script paths and Triton `ptxas` permissions must be configured before imports that touch CUDA, Triton, Torch, Transformers, or `mamba_ssm`. Setting environment variables after those imports may have no effect.

Lesson:

For Kaggle-hosted runs, start from the official submission demo or a known working notebook. If patching manually, put the `ptxas` copy/chmod and Cutlass path setup in the first cell, before heavy imports.

## Solver Correctness Does Not Guarantee Learnable SFT

Symptom:

Community reports show more accurate synthetic solvers or CoT traces can still lower leaderboard score after SFT.

Cause:

Long or mechanical traces may be correct but hard for the model to learn under LoRA capacity, context, and training-budget limits. Some reported success came from keeping traces shorter and aligned with model priors rather than only maximizing solver accuracy.

Lesson:

Evaluate synthetic data for learnability, not only correctness. Prefer short traces, strict final boxed answers, and category-level validation before trusting a new generated dataset.

## Colab Torch Import Corruption After Wrong Runtime Installs

Symptom:

```text
RuntimeError: Trying to override a python impl for DispatchKey.Autograd on operator aten::dropout
```

or the runtime reports CPU Torch, for example:

```text
torch 2.10.0+cpu
cuda available: False
```

Cause:

The Colab runtime is not in a clean GPU Torch state, usually after running install cells in a CPU runtime or after a failed/partial Torch-related package setup. Once `import torch` itself fails, later notebook cells cannot reliably repair the process.

Lesson:

Restart or factory-reset the Colab runtime, select a GPU runtime before running installs, and verify `import torch`, `torch.cuda.is_available()`, and BF16 support before any heavy package installs. Do not install or upgrade `torch` inside the Nemotron notebooks unless there is a concrete reason.
