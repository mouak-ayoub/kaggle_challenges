# Experiment Checklist

Use this as the submission planning board. After each Kaggle score, copy the submitted zip and source run bundle into `data/outputs/submissions/`, then update the status and notes here.

## Status Legend

| Mark | Meaning |
| --- | --- |
| `[x]` | done / already scored |
| `[ ]` | not run yet |
| `NEXT` | recommended next |
| `RISK` | high-risk or needs preflight check |

## Current Rule

Do not run a full factorial grid. We cannot afford every combination of prompt, target format, rank, LR, sequence length, and epoch count. Use bundled runs where each bundle has a clear hypothesis and avoids changing unrelated risky knobs.

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

## Submission Runs

- [x] `ACTIVE_02_raw_full_r4`
  - Status: done, scored `0.54` public LB.
  - Colab notebook label: user renamed this ongoing notebook with prefix `02`.
  - Hypothesis: full official data improves the known raw-answer baseline without changing prompt/format behavior.
  - Key config: `TRAIN_ROWS=None`, `EVAL_ROWS=256`, direct raw final-answer system prompt, `TRAIN_TARGET_FORMAT="raw"`, `LORA_R=4`, alpha `32`, `LORA_TARGET_MODULES=["in_proj", "out_proj"]`, seq `512`, `MAX_NEW_TOKENS=64`, effective batch `12*4=48`, LR `3e-4`, 1 epoch.
  - Latest pasted signal: train steps `193`; at step `169/193` elapsed about `2:12:30` with about `19:02` left.
  - Loss signal: eval loss improved monotonically from `0.921606` at step 24 to `0.852498`, `0.828395`, `0.815382`, `0.805131`, and `0.799496` at step 144.
  - Probe signal: still about `1/5` on the fixed probes through step 144. Outputs are short and formatted cleanly, but the two bit probes drifted to `00000000` at step 144 and the cipher/phrase probes remain near-pattern rather than correct.
  - Checkpoint-selection note: do not blindly submit final step if probe collapse persists. With `save_steps=48`, available checkpoint candidates are expected around steps `48`, `96`, `144`, and final; exact step `120` is not saved unless a manual save was made.
  - Result: despite monotonic eval-loss improvement to about `0.7933`, public score dropped below the `0.62` partial baseline. Do not treat full-data raw final-answer SFT as sufficient.
  - Archive: `data/outputs/submissions/2026-05-17_colab_raw_full_r4_score_0_54/`.

- [ ] `ACTIVE_03_private_reasoning_boxed_r8`
  - Status: interrupted by automatic Colab disconnect; runtime-local artifacts appear lost unless a Drive/manual download copy exists.
  - Hypothesis: private-reasoning prompt plus boxed targets improves final-answer discipline and gives rank 8 enough capacity to learn transformations.
  - Key config: private-reasoning boxed `SYSTEM_PROMPT`, `TRAIN_TARGET_FORMAT="boxed"`, `LORA_R=8`, alpha `64`, `LORA_TARGET_MODULES=["in_proj", "out_proj"]`, seq `512`, full data, 256 eval rows, LR `3e-4`, 1 epoch.
  - Risk: combines rank and output-format changes, so compare mainly against the raw full run and S4.
  - Next analysis need: first eval probe at step 24/36/48, final run bundle, Kaggle score.

- [x] `ACTIVE_04_S4_attention_expand_r8_private_boxed_max128`
  - Status: final/current adapter submitted and scored `0.53`; checkpoint selection still optional if checkpoint eval suggests a better package.
  - Hypothesis: expanding LoRA into Nemotron attention projections improves transformation learning while keeping the private-reasoning boxed setup.
  - Key config: private-reasoning boxed `SYSTEM_PROMPT`, `TRAIN_TARGET_FORMAT="boxed"`, `LORA_R=8`, alpha `64`, `LORA_TARGET_MODULES=["in_proj", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj"]`, seq `512`, `MAX_NEW_TOKENS=128`, effective batch `16*3=48`, LR `3e-4`, 1 epoch.
  - Runtime observed: single `NVIDIA RTX PRO 6000 Blackwell Server Edition`, about `95GB` VRAM, `176GiB` system RAM, PyTorch CUDA `12.8`, driver CUDA `13.0`.
  - Before-training probe signal: 4/5 rows hit max tokens at 64 before the max-token increase; base model was verbose and sometimes emitted reasoning/`</think>`.
  - Earlier lost-run signal: step 24 eval loss `0.870853`, step 48 eval loss `0.821435`; probe stayed `1/5`, but outputs became short boxed answers with `hit_max_new_tokens=False` for all five probes.
  - Current Drive rerun signal: step 24 eval loss `0.871280`, step 48 eval loss `0.823985`; probe stayed `1/5`, all outputs were short boxed answers with `hit_max_new_tokens=False`.
  - Current Drive rerun signal update: step 72 eval loss `0.806457`; probe still `1/5`, all outputs short boxed with no max-token hits.
  - Current Drive rerun signal update: step 96 eval loss `0.791767`; probe improved to `2/5` for the first time, with `XXXVIII` and `wizard creates secret` correct.
  - Current Drive rerun signal update: step 120 eval loss `0.781328`; probe regressed back to `1/5`, losing `wizard creates secret` and returning `wizard follows secret`.
  - Current Drive rerun signal update: step 144 eval loss `0.776157`; probe returned to `2/5`, now solving `cat imagines book` and `XXXVIII` but losing `wizard creates secret`.
  - Current Drive rerun signal update: step 168 eval loss `0.771446`; probe returned to `2/5`, solving `XXXVIII` and `wizard creates secret`, similar to step 96.
  - Final signal: step 193 eval loss `0.769407`; probe regressed to `1/5`, solving only `XXXVIII`. Final loss is best but final probe signal is worse than checkpoints 96/144/168.
  - Current probe content signal: all late checkpoints keep outputs short boxed with no max-token hits, but the correct cipher probe alternates across checkpoints while both bit probes remain wrong.
  - Checkpoint-selection note: steps `96`, `144`, and `192` are real saved checkpoints because `save_steps=48`. Step `168` is not normally saved unless manually saved. The final adapter at step 193 exists after `save_pretrained`. Choose using generated eval on all 256 eval rows by family, not eval loss alone.
  - Local generated eval for final/current step 193: `95/256 = 0.371094`; family accuracy was numeral `52/52`, unit conversion `17/36`, bit manipulation `14/45`, cipher `5/43`, equation `5/39`, gravity `2/41`.
  - Local generated eval for saved checkpoint 96: `79/256 = 0.308594`; this is clearly worse than final/current.
  - Local generated eval for saved checkpoint 144: `90/256 = 0.351562`; this improved over checkpoint 96 but still does not beat final/current.
  - Checkpoint 192 eval was launched after loading the S4 `checkpoint-192` directory before the Drive reorganization, but the pasted output does not yet show `saved progress after: checkpoint-192`; under the new layout the expected path is `/content/drive/MyDrive/Colab_Notebooks/Kaggle challenges/nemotron_challenge/artefacts/outputs/S4_attention_expand_r8_private_boxed_max128_drive/checkpoint-192`. Treat checkpoint 192 as pending until its row appears in the saved summary CSV.
  - Result: final/current adapter scored `0.53`, below both the 0.62 raw partial baseline and the 0.54 raw full-data control.
  - Archive: final/current submission zip copied to `data/outputs/submissions/2026-05-17_colab_s4_attention_boxed_r8_final_score_0_53/`.
  - Next analysis need: finish or recover generated eval for checkpoint 192 only; checkpoint 96 and 144 do not justify a superseding zip.
  - Local notebook state: `notebooks/01_colab_train_and_submit.ipynb` is currently set to `S4_attention_expand_r8_private_boxed_max128_drive` for direct Colab import with Drive output and explicit checkpoint resume disabled by default.

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

- [x] `S0_colab_raw_1024_r4`
  - Status: done, scored about 0.62.
  - Hypothesis: known working baseline.
  - Changes: 1,024 train rows, raw target, direct prompt, LoRA `r=4`, alpha `32`, `in_proj/out_proj`, seq `512`, 1 epoch, LR `3e-4`.
  - Risk: public LB noise.
  - Notes: archived.

- [x] `S1_raw_full_r4`
  - Hypothesis: more official data improves without changing behavior.
  - Changes: `TRAIN_ROWS=None`, keep 256 eval rows, same raw target, direct prompt, and LoRA rank.
  - Keep fixed: prompt, target format, LoRA rank, LR, seq length.
  - Risk: more data may mostly repeat easy patterns.
  - Result: scored `0.54`, worse than the 0.62 partial baseline.
  - Notes: full raw data alone appears to learn answer priors/format rather than better reasoning.

- [ ] `S2_raw_full_r8` `NEXT`
  - Hypothesis: extra LoRA capacity learns more patterns from full data.
  - Changes: `LORA_R=8`, `LORA_ALPHA=64`.
  - Keep fixed: raw target, direct prompt, full data, seq `512`, LR `3e-4`.
  - Risk: more capacity can overfit or destabilize; artifact is larger and training is slower.
  - Notes: run if `S1` is stable but not enough.

- [ ] `S2_private_reasoning_boxed_r8` `NEXT` `RISK`
  - Hypothesis: these tasks need internal reasoning, and rank 8 gives enough capacity to learn cleaner transformations.
  - Changes: `LORA_R=8`, `LORA_ALPHA=64`, `TRAIN_TARGET_FORMAT="boxed"`, private-reasoning boxed `SYSTEM_PROMPT`.
  - Keep fixed: full data, `in_proj/out_proj`, seq `512`, LR `3e-4`, 1 epoch.
  - Risk: this combines two axes at once: rank increase and output-format/prompt change. If it wins, we will know it works, but not which part helped.
  - Notes: this is private-reasoning training, not true visible CoT, because the official CSV has final answers but no rationale targets.

- [ ] `S3_boxed_full_r4_or_r8`
  - Hypothesis: metric-aligned output format improves extraction.
  - Changes: `TRAIN_TARGET_FORMAT="boxed"` and a boxed-final-only `SYSTEM_PROMPT`.
  - Keep fixed: full data, seq `512`, 1 epoch.
  - Risk: could lose the raw-answer behavior that scored 0.62; extra braces can hurt extraction.
  - Notes: use `r=4` if isolating format; use `r=8` only if budget forces bundling.

- [ ] `S4_attention_expand_r8_private_boxed` `RISK`
  - Hypothesis: adding Nemotron attention projections may improve transformations while keeping the selected private-reasoning boxed setup.
  - Changes: `LORA_TARGET_MODULES=["in_proj", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj"]`.
  - Keep fixed: private-reasoning prompt, boxed target, full data, seq `512`, LR `3e-4`, `r=8`, alpha `64`.
  - Risk: this changes target modules while also using the boxed/private setup, so compare it directly against `S2_private_reasoning_boxed_r8`.
  - Notes: preferred expanded-module experiment because it adds 24 attention modules, not thousands of `up_proj/down_proj` modules.

- [ ] `S5_private_reasoning_boxed`
  - Hypothesis: prompt encourages internal reasoning while still emitting only final boxed answer.
  - Changes: private-reasoning boxed `SYSTEM_PROMPT` with boxed target.
  - Keep fixed: full data, target modules, seq `512`.
  - Risk: this is not true CoT training; the model may ignore it or produce verbose text.
  - Notes: try after boxed-only is understood.

- [ ] `S6_short_cot_synthetic` `RISK`
  - Hypothesis: short rationales improve learnability for algorithmic families.
  - Changes: add generated short traces plus final boxed answer for selected categories.
  - Keep fixed: LoRA target modules and rank from best previous run.
  - Risk: wrong or too-long traces can lower score even if they look logical.
  - Notes: do not run until we inspect families and audit synthetic traces.

- [ ] `S6_short_trace_boxed_r8_attention_drive` `RISK`
  - Notebook: `notebooks/03_colab_short_trace_train_and_submit.ipynb`.
  - Hypothesis: compact supervised procedural traces teach algorithmic cipher solving better than final-answer-only SFT while preserving Kaggle boxed extraction.
  - Method: for parseable cipher rows, train targets look like `Trace: trb->cat; wzrswvog->imagines; hffk->book.\nFinal: \boxed{cat imagines book}`. Non-parseable rows fall back to `\boxed{answer}`.
  - Current local audit: trace generation works for `00189f6a`; `1,576 / 9,500` train rows receive a cipher trace.
  - Key config: Drive-backed outputs, explicit checkpoint resume disabled by default, `TRAIN_TARGET_FORMAT="short_trace_boxed"`, `TRACE_TARGET_MODE="cipher_short_trace_else_boxed"`, `LORA_R=8`, alpha `64`, attention-expanded target modules, seq `768`, max new tokens `256`, effective batch `8*6=48`.
  - Risk: visible traces can improve reasoning but may also produce extra text; metric should still extract the last boxed answer if the model finishes cleanly.
  - Paper anchor: STaR argues rationale fine-tuning can outperform direct final-answer fine-tuning on reasoning tasks.

- [ ] `S7_starish_short_trace_bootstrap` `RISK`
  - Hypothesis: a STaR-like loop can produce better rationale supervision than hand-built traces alone while still ending in a Kaggle-compatible LoRA adapter.
  - Training-time idea: generate candidate short traces/answers for train prompts, check against known `answer`, keep successful traces, and for failures provide the gold answer while asking the model or a helper to produce a compact rationale that reaches it. Then fine-tune LoRA with ordinary SFT on the selected/corrected traces.
  - Important distinction: this is not an inference-time agent and not classic PPO/GRPO. It is bootstrapped supervised fine-tuning with a correctness filter, repeated if useful.
  - Kaggle compatibility: final artifact remains `submission.zip` containing rank `<=32` LoRA adapter files for Nemotron-3-Nano-30B.
  - First implementation path: start only on cipher rows because we can verify exact answers and inspect short trace quality; expand to bit/equation families after trace quality is proven.
  - Risk: generated rationales can be plausible but wrong; every retained trace must be auditable or mechanically checked where possible.

- [ ] `S7_best_config_eval0_final`
  - Hypothesis: use all 9,500 rows once the best config is known.
  - Changes: set `EVAL_ROWS=0` only after choosing a winning config.
  - Keep fixed: winning prompt, rank, LR, seq length.
  - Risk: no eval callback during training unless we add a log-step probe callback.
  - Notes: final polish, not exploration.

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
