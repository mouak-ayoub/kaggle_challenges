# Project Decision Log

Last updated: 2026-05-27

This file records durable project decisions. It is not a scratchpad.

## Decision: Start With Instrumentation Before Training

### Context

The challenge is about improving reasoning performance, but the local repository has no official data yet.

### Decision

Do not begin with fine-tuning or heavy inference code. First build data inspection, answer normalization, baseline logging, and category-level reporting.

### Evidence

Public community material suggests multiple heterogeneous problem families. A single aggregate score can hide whether failures come from reasoning, formatting, extraction, or category-specific weakness.

### Consequence

The initial repository contains documentation, project memory, and lightweight reusable utilities rather than model training code.

### Status

Active.

## Decision: Do Not Keep Scaling V2 Trace Supervision As-Is

### Context

The project tested broad v2 trace supervision at two scales. `exp06_mamba_fused_bf16_attention_r4_trace_v2` used the 2,500-row v2 curriculum and scored `0.58`. `exp11_mamba_trace_v2_aug25k_b8_ep1` used the 25k augmented v2 curriculum; checkpoints 195 and 585 scored `0.56`, while checkpoints 390 and 780 scored `0.54`.

### Decision

Do not continue the current v2 trace format as the main scaling direction. Keep its useful parts, especially compact stopping behavior and easy-family rehearsal, but redesign the hard-family traces before another large run.

### Evidence

The exp11 step-205 generated eval had zero empty answers and zero max-token hits, showing the model learned the output format. But generated-answer accuracy was only `21/64 = 0.328125`, with bit manipulation `0/20` and cipher `8/26`. Checkpoint 195 had weaker backfill generated eval than checkpoint 390 (`11/64 = 0.171875` vs `17/64 = 0.265625`) but scored higher publicly (`0.56` vs `0.54`); checkpoint 585 also scored `0.56`, while checkpoint 780 returned to `0.54`. The small generated eval is not a reliable checkpoint selector. These public scores are below exp05 checkpoint144 at `0.59` and below the raw partial baseline at `0.62`. Error inspection showed bit traces taught execution of a stated rule, not search for the rule, and cipher traces taught citation style without reliable ordered character alignment. A later exp12 target-surface test with the same broad v2 data and all six projection targets improved local generated eval to `36/64 = 0.5625`, especially cipher `18/26`, but public score returned only `0.54`. This reinforces the decision: stronger small local generated eval is not enough evidence to keep scaling the current v2 trace format.

### Consequence

The next broad dataset should not simply add more v2 rows. It should use hard-family trace designs that teach the missing operation: bit rule-search before execution, cipher ordered word/position alignment, and equation candidate-transform selection. Natural cipher traces remain useful as an ingredient, but the exp10 cipher-only score `0.49` shows that single-family cipher fine-tuning is not a leaderboard-safe path.

### Status

Active.

## Decision: Keep Supporting Markdown Under `doc/`

### Context

The nearby PhysioNet project separates agent instructions, current memory, durable decisions, and reusable source code. The project root was becoming noisy with several Markdown support files.

### Decision

Keep root-level Markdown limited to entry points, and move supporting notes into `doc/`:

- `AGENTS.md`
- `doc/LOCAL_PROJECT_MEMORY.md`
- `doc/PROJECT_DECISION_LOG.md`
- `doc/errors.md`
- `README.md`
- `src/`

### Evidence

That split keeps current status, durable technical choices, and reusable lessons from collapsing into one unclear README while keeping the repository root easy to scan.

### Consequence

Future sessions should start by reading the memory, decision, and error files before editing code.

### Status

Active.

## Decision: Treat Problem Families As First-Class Units

### Context

Public artifacts list reasoning families including ciphers, numerals, unit conversion, bit manipulation, gravity, and equations.

### Decision

All analysis and validation should preserve problem type/category when available. Improvements should be evaluated by category before being trusted globally.

### Evidence

Different families likely need different failure analysis and may benefit from different prompting, validation, or synthetic data strategies.

### Consequence

The first utilities include schema detection and category-count reporting.

### Status

Active.

## Decision: Keep Shared Notebook Helpers In Sync

### Context

The project uses a local notebook for Windows/data dry runs and a Colab notebook for Nemotron GPU training. Some cells are environment-specific, but helper methods for formatting, inspection, answer generation, and diagnostics should behave the same in both notebooks.

### Decision

When shared helper code changes in one notebook, mirror the same method in the other notebook when applicable. Keep only install, runtime, path, and model-loading differences separate.

### Evidence

The module summary helper was improved in the Colab notebook to show dtypes after Nemotron BF16/FP32 issues. The same diagnostic is useful in the local notebook too.

### Consequence

Notebook edits should check whether the changed method is shared logic or environment logic before stopping.

### Status

Active.

## Decision: Submit A LoRA Adapter Zip, Not Predictions

### Context

The competition `test.csv` has example prompts, but Kaggle scoring does not use a normal prediction CSV for this challenge.

### Decision

The Colab training notebook should treat generated test answers as a sanity check only. The real submission artifact is `submission.zip` containing the required saved PEFT LoRA adapter files at the zip root.

### Evidence

The public Kaggle submission demo saves the PEFT adapter and zips the adapter files. The notebook output shows `adapter_config.json` and `adapter_model.safetensors` in `submission.zip`, with adapter rank capped at 32.

### Consequence

Do not spend work on a final `submission.csv` path for leaderboard submission. Validate the adapter directory, check `adapter_config.json` rank, zip `adapter_config.json` and `adapter_model.safetensors` at the zip root, and upload `submission.zip`.

### Status

Active.

## Decision: Keep Raw-Answer Targets As The Current Baseline

### Context

The official metric prompt asks the model to put the final answer inside `\boxed{}` and the extractor prioritizes boxed content. However, the accepted Colab Nemotron run that scored about 0.62 was trained with raw short-answer targets, before the notebook was changed to boxed SFT targets.

### Decision

Keep `TRAIN_TARGET_FORMAT = "raw"` as the active notebook default because it reproduces the best known submission. Keep `TRAIN_TARGET_FORMAT = "boxed"` available as an explicit experiment, and keep local sanity extraction boxed-aware so extraction failures can be analyzed separately.

### Evidence

The official metric notebook and Kaggle discussion threads confirm boxed answers are the primary extraction path. The metric was updated after community reports that answers containing `}` were impossible or ambiguous under the old regex. But the only observed higher score so far is the raw-answer Colab run at about 0.62; the boxed-target change has not yet been scored as an improvement.

### Consequence

Do not overwrite the known baseline with an unscored format change. When testing boxed targets, record it as a separate submission with its own metadata and compare against the raw-answer baseline.

### Status

Active.

## Decision: Treat Public LB Scores As Noisy Signals

### Context

Participants reported different scores for identical or near-identical submissions.

### Decision

Do not choose methods purely from single public leaderboard runs. Prefer local category-level validation, raw completion logs, and repeated checks when leaderboard differences are small.

### Evidence

Kaggle discussion confirms vLLM scoring is not deterministic even with `temperature=0.0`; the host declined deterministic vLLM settings because they would reduce throughput substantially. Submissions can also take roughly 70-90 minutes or longer to score.

### Consequence

Small leaderboard deltas should be treated cautiously. Track category-level local behavior and avoid overfitting to one public score.

### Status

Active.

## Decision: Scale The Known Raw Baseline Before CoT Variants

### Context

The next target is a public score near 0.7. Proposed changes include CoT-style prompting, training on more data, proportional logging/eval, and watching probe answers before and after training.

### Decision

Make the next default run a scaled version of the known 0.62 baseline: raw-answer targets, the same direct prompt, all available training rows except the eval split, and better instrumentation. Keep boxed and private-reasoning prompts available as explicit switches, not the default.

### Evidence

The only high-scoring observed submission so far is the raw-target Nemotron LoRA run at about 0.62. Kaggle discussions warn that longer reasoning traces can lower score if the model fails to finish with a clean final answer. More data and better logs are lower-risk changes than changing both reasoning style and target format at the same time.

### Consequence

Compare the full-row raw run against 0.62 first. If it does not improve enough, submit a separate prompt/target experiment and track it independently in `data/outputs/submissions/`.

### Status

Superseded by the `S1_raw_full_r4` result.

## Decision: Stop Scaling Raw-Only Final-Answer SFT

### Context

The known best Colab Nemotron submission scored about `0.62` with raw short-answer SFT on a 1,024-row subset. The next control run scaled the same direct/raw setup to the full train split minus 256 eval rows.

### Decision

Do not spend more submissions on raw-only final-answer SFT as the main path. Prioritize metric-aligned boxed outputs, expanded target modules, and short procedural trace supervision.

### Evidence

`S1_raw_full_r4` / `ACTIVE_02_raw_full_r4` trained cleanly and eval loss improved monotonically to about `0.7933`, but the fixed probe stayed `1/5` and the public leaderboard score was `0.54`, below the partial raw baseline at about `0.62`.

### Consequence

Eval loss on final-answer SFT is not a reliable proxy for hidden reasoning score. The next serious experiments should test whether the model learns procedures, not just answer priors and output brevity.

### Status

Active.

## Decision: Reject Long Per-Position Cipher Citation Traces As A Submission Path

### Context

The `exp09_mamba_cipher_v3_synth2500_b20_ep3` run trained on 2,500 augmented cipher rows with long position-level citations. Teacher-forced validation looked excellent, with final eval loss around `0.088` and mean token accuracy around `0.973`.

### Decision

Do not submit or continue scaling this long per-position cipher trace style as-is. Future cipher traces should be shorter, target-only, and closer to the base model's natural word-alignment reasoning, while still ending with one boxed answer.

### Evidence

Final generated eval was `0/64 = 0.0` on cipher rows, with `63/64` max-token hits at `384` tokens. The final probe failed before and after training; after training the model copied the citation template but produced invalid citations and never reached a correct boxed answer. Public sanity outputs also repeated the cipher template on non-cipher binary prompts.

### Consequence

Treat low teacher-forced loss on trace text as insufficient. The next cipher dataset should minimize trace length and directly test whether generated outputs reach `Final answer` without max-token hits.

### Status

Active.

## Decision: Use Compact Natural-Alignment Cipher Traces For The Next Cipher Probe

### Context

Before exp09 training, the base model's cipher probe naturally reasoned by aligning example words and deriving character mappings. It was too verbose and did not reach the target before max tokens, but the reasoning style itself was closer to the desired algorithm than the rigid position-citation template.

### Decision

Generate v4 cipher traces that imitate the useful natural alignment style: `We need to find mapping from cipher to plaintext`, explicitly align one example's cipher/plain words, map letters word by word with consistency notes, then apply target-needed mappings and produce one boxed final answer. Avoid `Thinking:/Category:` boilerplate and avoid per-position citations.

### Evidence

The exp09 v3 position trace achieved excellent teacher-forced loss but generated `0/64` correct cipher answers and hit max tokens on `63/64` generated-eval rows. The failure mode was template repetition and invalid citations, not lack of trace imitation.

### Consequence

The next cipher file is `data/input/traces/trace_cipher_v4_natural2500.csv`, and the matching notebook is `notebooks/08_colab_mamba_cipher_v4_natural_train_and_submit.ipynb`. Because v4 is longer than exp09, the notebook uses `MAX_SEQ_LENGTH=1024` with a smaller batch. The submit gate emphasizes generated outputs reaching a boxed answer without max-token hits.

### Status

Active.

## Decision: Add Ordered Alignment To The Next Cipher Trace Revision

### Context

The `exp10_mamba_cipher_v4_natural2500_b12_seq1024_ep3` run reinforced the base model's natural cipher alignment style. It was interrupted before normal final packaging, but the in-memory adapter was saved into a run bundle and evaluated locally.

### Decision

Keep natural alignment as the cipher trace direction, but do not scale v4 unchanged. The next cipher trace revision should preserve the v4 wording while adding compact ordered alignment lines such as `ynz -> the gives y->t, n->h, z->e` and target assembly lines such as `cipher: y z x w n z j` / `plain: t e a c h e r`.

### Evidence

Exp10 improved generated cipher behavior from exp09's `0/64` and `63/64` max-token hits to `26/64 = 0.40625` on a 64-row cipher generated eval, with `6/64` max-token hits. It also reached a boxed answer on the fixed probe instead of looping. However, that probe still failed: for `ynz -> the`, the model used `y->h, n->t` instead of the ordered mapping `y->t, n->h`, producing `hatter creates chase` instead of `teacher creates castle`. The run remains verbose, averaging `787` generated tokens.

### Consequence

Exp10 scored `0.49`, so cipher-only natural-alignment training is not leaderboard-safe. V5 should target shorter outputs and ordered character execution inside a broad all-family curriculum rather than more long natural explanation or another cipher-only run.

### Status

Active.

## Decision: Organize `data/input/` By Role

### Context

The input root accumulated official Kaggle files, trace datasets, verifier dependencies, and accidentally downloaded diagnostics. This made resume work slower and increased the risk of picking the wrong CSV.

### Decision

Use role-based subfolders:

- `data/input/official/` for `train.csv` and `test.csv`
- `data/input/traces/` for trainable trace CSVs
- `data/input/verifier/` for local verifier/builder dependencies

Downloaded diagnostics belong under `data/outputs/`, not `data/input/`.

### Evidence

The active workflow now has multiple trace generations (`trace_training.csv`, v2, all-hard, v3, v4), plus official data and bit-verifier audit data. Root-level input files were no longer easy to scan.

### Consequence

Script defaults and active notebook local fallbacks now point to subfolders. Colab `/content/*.csv` uploads remain unchanged.

### Status

Active.

## Decision: Persist Colab Training Outputs To Google Drive

### Context

Long Nemotron LoRA runs in Colab can be interrupted or reclaimed automatically, especially when multiple heavy GPU sessions run in parallel. Runtime-local files under `/content` disappear when the VM is deleted.

### Decision

Write experiment outputs, checkpoints, run bundles, trainer logs, and probe evolution files to Google Drive by default. Keep raw competition data local for speed, but persist training outputs under `/content/drive/MyDrive/Colab_Notebooks/Kaggle challenges/nemotron_challenge/artefacts/outputs/{EXPERIMENT_NAME}`.

### Evidence

The `03` and `04` Colab sessions disconnected automatically and their runtime-local artifacts were lost. Resume is only possible if a full `checkpoint-*` directory, including trainer state, survives outside the VM.

### Consequence

The Colab notebook now mounts Drive and uses Drive-backed `OUTPUT_DIR`. Resume is opt-in: keep `RESUME_FROM_CHECKPOINT=False` for a fresh run, or set it true and choose `RESUME_CHECKPOINT_STEP` to continue from `OUTPUT_DIR/checkpoint-{step}`. If resume is false and old checkpoints exist, training stops instead of silently mixing runs. Starting a clean rerun with the same `EXPERIMENT_NAME` requires changing the experiment name or manually clearing the old Drive output directory.

Generated-answer checkpoint eval runs during training when `GENERATED_EVAL_ROWS_ON_SAVE` is above `0`. The current serious-run default is `EVAL_ROWS`, which scores the full fixed eval split at each saved checkpoint. Set it to `64` for cheaper tracking or `0` to disable. Checkpoint summaries are written to `checkpoint_eval/checkpoint_generated_eval_summary.csv` and included in the run bundle.

### Status

Active.

## Decision: Build Kaggle Submission Zips Locally From Colab Run Bundles

### Context

The Colab notebooks previously wrote both a standalone Kaggle `submission.zip` and a diagnostics zip that also embedded the submission zip. That created ambiguity about which zip should be uploaded and made Drive contain duplicate packaged artifacts for the same run.

### Decision

At the end of each Colab run, write one portable `{EXPERIMENT_NAME}_run_bundle.zip`. The bundle contains `adapter/adapter_config.json`, `adapter/adapter_model.safetensors`, run configuration, trainer logs, probe logs, generated-eval files, checkpoint-eval files when present, and TensorBoard events. Build the actual Kaggle `submission.zip` locally from the bundle adapter files.

### Evidence

The Kaggle artifact must contain only `adapter_config.json` and `adapter_model.safetensors` at the zip root. Diagnostics and checkpoint-eval files are needed for local reports but must not be mixed into the upload artifact.

### Consequence

Colab is responsible for training and producing one run bundle. The local repository is responsible for extracting/analyzing that bundle and creating the final Kaggle upload with `scripts/build_submission_from_run_bundle.py`.

### Status

Active.

## Decision: Treat Submission As Upload Plus Archive Intake

### Context

A Kaggle upload only needs a strict `submission.zip`, but the local dashboard,
experiment history, and later decisions need diagnostics such as probe outputs,
generated-eval summaries, run config, trainer logs, and public score metadata.
During checkpoint submissions it is easy to upload the zip and forget the
supporting files.

### Decision

When the user says a run or checkpoint is ready for submission, treat that as a
small intake workflow, not only a zip-packaging action. First confirm or search
for the upload artifact, then ask for missing dashboard/archive files:
`run_config.json`, `trainer_log_history.csv`, `probe_evolution.csv`,
`generated_eval_summary.csv`, `generated_eval_predictions.csv` when available,
checkpoint eval summaries, and sanity raw predictions when available. If the
public score is not known yet, archive as pending and update score metadata
after Kaggle finishes.

### Evidence

Checkpoint-96 from `exp05_trace_occam_r4_inout` produced a valid Kaggle zip from
Drive, while only `probe_evolution.csv` had been downloaded locally at first.
The upload artifact was sufficient for Kaggle, but not sufficient for the local
dashboard/history without follow-up artifact collection.

### Consequence

Future submission turns should include explicit questions about the run id,
checkpoint, zip location, diagnostics availability, score status, and dashboard
update timing. If only `submission.zip` exists, state that the Kaggle upload can
proceed but the dashboard record is diagnostics-pending.

### Status

Active.

## Decision: Compare Only The Submitted Checkpoint Step

### Context

Some submitted run archives include local diagnostics for multiple saved
checkpoints. The S4 archives contain generated-eval rows for `checkpoint-96`,
`checkpoint-144`, and `current-193`, even though only the final/current adapter
and checkpoint-144 adapter were submitted as separate Kaggle uploads.

### Decision

Dashboard local diagnostic charts should filter multi-step diagnostics to the
actual submitted or selected checkpoint step when the archive identifies one.
Public-score charts still require Kaggle scores, but generated eval, probe, and
loss charts may include pending runs when those local metrics exist.

### Evidence

The dashboard previously surfaced `04-s4-step-96`, which looked like a
submitted run. Local files show it was only a diagnostic row inside S4 archives;
the tracked S4 submitted checkpoint is `checkpoint-144`, while S4 final/current
maps to step 193.

### Consequence

Submitted S4 final is labeled `04-s4-final`, submitted S4 checkpoint 144 is
labeled `04-s4-step-144`, and exp05 trace checkpoint 96 is labeled
`05-trace-step-96` with its available local generated-eval/probe/loss metrics.

### Status

Active.

## Decision: Do Not Continue Final-Answer Boxed SFT Without Procedural Supervision

### Context

The `S4_attention_expand_r8_private_boxed_max128_drive` run trained full-data boxed targets with a private-reasoning prompt, LoRA rank 8, and expanded attention target modules.

### Decision

Do not repeat this exact final-answer boxed SFT path as the main strategy. Future serious attempts should add procedural supervision, checkpoint selection from generated eval, or category-specific data rather than just stronger boxed-format LoRA.

### Evidence

The final/current S4 adapter had clean boxed outputs and local generated eval `95/256 = 0.371`, but the public score was only `0.53`. The checkpoint-144 adapter had lower local generated eval at `90/256 = 0.351562` but scored `0.55` publicly, still below the `0.62` raw-answer baseline. S4 was strong on numerals but weak on cipher, equation, gravity, and bit manipulation. The 5-row probe also showed loss improving while reasoning correctness oscillated and final step regressed to `1/5`.

### Consequence

Boxed formatting helps extraction discipline but is not enough to improve reasoning. Checkpoint timing can matter, but the checkpoint-144 score does not change the main conclusion. Prioritize short procedural traces, solver-guided STaR bootstrapping, and better local validation over more S4-style final-answer boxed SFT.

### Status

Active.

## Decision: Treat Exp05 Trace Occam As Useful But Not Sufficient

### Context

The `exp05_trace_occam_r4_inout` run trained on canonical trace targets with completion-only loss, LoRA rank 4 on `in_proj/out_proj`, LR `1e-4`, and Mamba-backed checkpoint evaluation. Checkpoint 96 scored `0.58` publicly. Checkpoint 144 scored `0.59` publicly and has the same local generated eval aggregate as checkpoint 96, `134/256 = 0.5234375`, while gaining one numeral row and losing one bit-manipulation row.

### Decision

Do not treat more steps on this exact trace mix as the main path to `0.7`. Keep checkpoint 144 as the current trace-supervision public baseline, then move the next research effort to better validation and better hard-family supervision.

### Evidence

Trace Occam improved over full-data raw SFT (`0.54`) and S4 boxed/attention variants (`0.53` and `0.55`), but it still did not beat the smaller raw-answer baseline at about `0.62`. Local diagnostics show unit conversion, gravity, and numeral are already strong, while cipher remains `0/40` and bit/equation remain near zero. Checkpoint 144's lower train/eval loss improved the public score slightly from `0.58` to `0.59`, but did not improve aggregate local generated eval over checkpoint 96.

### Consequence

The next serious work should build `eval_v2_grouped_family_balanced` and focus solver-guided STaR or verifier-backed traces on cipher, bit manipulation, and equation. Numeral should be excluded or downweighted in future score-seeking trace experiments because it is already saturated.

### Status

Active.

## Decision: Use Brainstorm-Then-Notebook Experiment Loop

### Context

The project has already spent several notebook iterations on raw final-answer SFT and boxed/private-reasoning variants. The best observed public score is still about `0.62`, while full-data raw and final-answer boxed experiments regressed to about `0.54`, `0.53`, and `0.55`.

### Decision

Use a deliberate loop for future work: brainstorm candidate ideas together, critique them as research hypotheses, choose one simple non-trivial idea, implement it notebook-first, run it in Colab, download the run bundle/diagnostics, build the Kaggle `submission.zip` locally, update the dashboard and experiment notes, then iterate toward at least `0.7`.

### Evidence

Recent runs show that changing several knobs without enough procedural signal can improve formatting or eval loss while hurting the public score. The current strongest directions are short procedural traces and STaR-like bootstrapped SFT, but each needs visible artifacts and category-level evidence before becoming the next submitted adapter.

### Consequence

Do not jump straight into broad refactors or factorial sweeps. For each iteration, keep the selected idea legible in the notebook top cell, preserve raw completions and generated-eval outputs, and archive the resulting bundle/submission before choosing the next idea.

### Status

Active.

## Decision: Prioritize Solver-Guided STaR Over Copying Public Trajectories

### Context

External review and public references point toward reasoning trajectories as a stronger direction than answer-only SFT. The project goal, however, is to learn and build an original research path rather than simply copying a public high-scoring notebook.

### Decision

Use outside research as literature, tools, and inspiration, including papers, open-source code, blog posts, Reddit/Medium/Substack posts, YouTube talks, other-domain competition ideas, and reasoning-challenge methods that are not from this exact Nemotron Kaggle challenge. Do not inspect or copy another active Nemotron challenge competitor's high-scoring notebook, GitHub repo, reasoning-trajectory dataset, or submission path as the primary method, training source, or implementation path unless explicitly approved. The next main research direction is solver-guided STaR: generate candidate reasoning traces, verify them with family-specific solvers or structural checks, filter aggressively, and fine-tune a Kaggle-compatible LoRA adapter on compact verified traces from our own pipeline.

### Evidence

The current project evidence shows that raw full-data SFT and final-answer boxed SFT improved local diagnostics but reduced public score from about `0.62` to `0.54`, `0.53`, and `0.55`. This contradicts more answer-only training as the main path. STaR is designed for settings with many final-answer labels but no rationales, which matches this challenge. The project also has mechanically inspectable families, especially cipher, unit conversion, and bit manipulation, where verifier-guided filtering is feasible.

### Consequence

Before another broad training run, build `eval_v2` and small verified trace datasets. Start with notebook-first experiments such as `exp05_star_seed_512` or a cipher/unit/bit focused variant, keep `00-raw-1024` as the leaderboard baseline, exclude or downweight numeral, and preserve raw completions plus accepted/rejected trace diagnostics. Outside research may inform critique and design, but the submitted method should be our own research trajectory.

### Status

Active.

## Decision: Train Cipher Traces As Target-Side Character Procedures

### Context

Cipher rows do not share one global cipher alphabet. Each prompt provides its own examples, and the model must infer a temporary character mapping inside the prompt before decrypting the target text.

### Decision

For the next trace-training path, keep the training prompt equal to the original Kaggle prompt and put the compact reasoning trace only in the SFT target/completion. For cipher rows, build target-side traces that align same-length words, list the needed mappings, mark characters not determined by examples, use phrase context only for those unknowns, and end with exactly one boxed final answer. Use two deterministic templates selected from the row id: mostly compact mapping traces, plus a smaller source-rich variant that cites example word pairs. Use expanded projection LoRA targets for the next notebook: `in_proj`, `out_proj`, `q_proj`, `k_proj`, `v_proj`, and `o_proj`.

### Evidence

Answer-only SFT and boxed final-answer SFT did not improve reasoning enough. Ciphers are verifier-friendly because every target word can be aligned against the gold answer by character position, and repeated cipher characters provide a cheap consistency check. Keeping the prompt unchanged avoids training the model to rely on row-specific hints that Kaggle will not provide.

### Consequence

The local trace builder writes `data/input/traces/trace_training.csv` with the minimal model-facing schema: `id`, `question`, `trace`, and `gold_answer`. Accepted cipher rows use the deterministic compact/source-rich template mix; other families are expanded only when their traces are mechanically defensible. The next Colab notebook should load this file rather than rebuilding traces ad hoc.

### Status

Active.

## Decision: Use Completion-Only Loss For Trace SFT

### Context

Notebook `03` originally built one formatted `text` column containing `System`, `User`, `Assistant`, and the trace target. In TRL this is treated as a language-modeling dataset, so loss can be computed over the full sequence rather than only the assistant completion.

### Decision

Use a prompt/completion dataset in notebook `03`: `prompt` contains `System/User/Assistant:` and `completion` contains the trace or boxed answer. Set `completion_only_loss=True` in `SFTConfig`.

### Evidence

TRL documents that prompt-completion datasets train on completion tokens only by default when completion-only loss is enabled, while language-modeling datasets train on the full sequence. A local dry run of the notebook data-prep cell now produces dataset columns `prompt` and `completion`; the first completion begins with `Thinking:`, and the run config records `loss_masking=completion_only`. Notebook `03` also asserts the actual trainer dataloader labels before training: prompt labels must be masked and the first unmasked labels must decode to `Thinking:` or `Final answer:`.

### Consequence

The first supervised token for trace rows is the first completion token, usually `Thinking`, conditioned on the original puzzle prompt. The adapter no longer wastes loss on reproducing fixed `System:`/`User:` labels or the question text.

### Status

Active.

## Decision: Generate Gravity Traces With Formula And Rounding Awareness

### Context

Gravity prompts provide the target formula `d = 0.5*g*t^2` and several rounded observations. Answer-only SFT did not teach the model to infer the hidden `g`, apply the formula, and round the target answer reliably.

### Decision

For gravity rows, use a single compact formula trace in the SFT target: derive `g = 2*d/t^2` from the examples, show three example estimates, state that distances are rounded, use a common value near the target-compatible `g`, apply `d = 0.5*g*t^2`, and finish with exactly one boxed final answer.

### Evidence

The simple average of example-derived `g` estimates does not reproduce every gold answer because displayed distances are rounded. The generated gravity traces therefore use rounding-aware wording and a 4-decimal displayed `g` in the target calculation. Local audit verified all 1,597 gravity traces have exactly one boxed answer, boxed text equals `gold_answer`, no non-ASCII text, and the displayed target calculation rounds back to the gold answer.

### Consequence

`data/input/traces/trace_training.csv` now has gravity formula traces for all gravity rows. This is the next mechanically verified family after cipher. Unit conversion follows the same rounding-aware principle in a separate mixed-template decision; bit manipulation and equation should remain boxed-only until real verifiers exist.

### Status

Active.

## Decision: Mix Unit Conversion Trace Lengths

### Context

Unit conversion is comparatively simple and already performed better than the hardest families, but answer-only SFT still does not directly teach ratio inference, rounded-example handling, and target rounding.

### Decision

For unit conversion rows, use a deterministic `id`-based target mix instead of giving every row a long rationale: about 50% short traces, 25% full ratio traces, and 25% boxed-only answers. Short traces state `output = k*input`, the common ratio, and the target calculation. Full traces also show three example-derived ratios. Boxed-only rows preserve direct-answer behavior.

### Evidence

All 1,594 unit conversion gold answers have two decimals. A 4-decimal target-compatible `k` is enough for the displayed target calculation to round back to the gold answer on every traced unit row. Local audit verified all unit rows have exactly one boxed final answer, boxed text equals `gold_answer`, no non-ASCII text, and zero displayed-`k` rounding mismatches.

### Consequence

`data/input/traces/trace_training.csv` now teaches the unit conversion procedure without making that already-strong family dominate the model with long traces. The remaining boxed-only families are bit manipulation, equation, and numeral; bit/equation should wait for real verifiers.

### Status

Active.

## Decision: Keep Numeral Rows Mostly Boxed-Only

### Context

Numeral conversion is already one of the strongest families locally, likely because the prompts mostly use familiar numeral-system behavior and the base model already knows the procedure.

### Decision

Keep numeral targets mostly boxed-only, but include a small deterministic short-trace slice of about 10%. The short trace decomposes the target number into Roman-style place values, for example `15 = 10 + 5`, then maps those parts to `X` and `V`. Keep future capacity control focused on downweighting or subsampling numerals in the training notebook, not adding longer numeral rationales.

### Evidence

Current generated-eval history showed numeral accuracy saturated compared with the hard families, so numeral traces should stay sparse. Local audit verified the current mix: 155 short traces and 1,421 boxed-only rows across 1,576 numeral rows. Every row has exactly one boxed final answer, boxed text equals `gold_answer`, no trace contains non-ASCII text, and all short traces were generated only when the Roman-style verifier matched the gold answer.

### Consequence

The trace dataset gives numerals a small amount of procedural signal without letting an easy family consume much trace capacity. Numerals remain available for format preservation, but the next notebook should still consider sampling only a smaller numeral subset.

### Status

Active.

## Decision: Add Only Verified Bit-Manipulation DSL Traces

### Context

Bit manipulation is a hard family where fluent reasoning text can easily be fake. A temporary candidate trace file contained some DSL-like bit rules, but other bit rows only contained vague "apply the hidden rule" text.

### Decision

Audit candidate bit traces with a small 8-bit DSL verifier before using them for training. Accept only rows where the parsed rule matches every provided example and produces the gold query answer. Add a compact verified trace for accepted rows, and leave all unverified or unparseable bit rows boxed-only.

### Evidence

The verifier audited the bit rows and stores the accepted rules in `data/input/verifier/bit_candidate_trace_audit.csv`, because this file is a local builder dependency rather than a generated Colab output. After excluding the three public sanity IDs from training, the canonical trace CSV has 942 `bit_manipulation_verified_dsl_trace_boxed` rows and 658 boxed-only bit rows. Global audit still passes: every row has exactly one boxed answer, boxed text equals `gold_answer`, and traces contain no non-ASCII text.

### Consequence

Bit manipulation now receives procedural supervision only where the rule is mechanically checked. This follows the diagnostic/training separation: model-style thinking is useful for inspection, but only verified thinking becomes SFT data. Equation remains boxed-only until a comparable verifier exists.

### Status

Active.

## Decision: Keep Equation Boxed-Only After Char-Substitution Audit

### Context

Equation-symbolic rows look like hidden string transformations, but vague rationales are risky because they can teach the model to claim a rule without applying one. The first conservative verifier candidate was same-length character substitution.

### Decision

Do not add equation reasoning traces from the current v0 DSL. Keep equation rows boxed-only until a verifier handles the actual length-changing transformations.

### Evidence

`scripts/audit_equation_char_substitution.py` audited all 1,555 equation rows. Every row was classified as `length_changing_examples`, so same-length character substitution has zero coverage. The canonical trace CSV still has 1,555 boxed-only equation rows and zero equation rows starting with `Thinking:`.

### Consequence

Equation remains the only fully boxed-only family. The next equation verifier should target length-changing string rules such as deletion/filtering, subsequence extraction, position selection, or numeric subtypes, and should still accept traces only when the rule reproduces every example and the gold target.

### Status

Active.

## Decision: Use Strict Fix-Tracing V2 Before More Trace Volume

### Context

Exp05 trace supervision improved over the S4 boxed path but still scored only `0.59` at its best submitted checkpoint, below the older `00-raw-1024` baseline near `0.62`. Local diagnostics showed the trace run mainly learned unit conversion, gravity, and numerals, while cipher stayed `0/40`, bit manipulation stayed around `3/44`, and equation stayed around `2/39`.

### Decision

Run a small fix-tracing experiment before adding broader STaR data or changing LoRA capacity. Build `trace_training_v2_2500.csv` with strict cited-map cipher traces, explicit 8-bit bit-manipulation execution traces, and small rehearsal slices for equation, unit conversion, gravity, and numeral. Keep the exp05 model knobs mostly unchanged, but use two epochs because the curriculum has only 2,500 rows; choose the submitted adapter by checkpoint behavior rather than final step by default.

### Evidence

The previous cipher traces allowed unsupported phrase-context completion, which can teach plausible-looking but wrong mapping explanations. Current train data has only 605 strict high-evidence cipher candidates after excluding public sanity IDs, so the v2 file uses 600 of them rather than oversampling. The bit verifier has 942 valid DSL rows, enough to train 900 execution traces where the target rule is actually evaluated step by step. Local audit of the v2 CSV found exactly one boxed answer per row, zero official-style boxed mismatches, and no non-ASCII traces.

### Consequence

Notebook `05_colab_fix_tracing_train_and_submit.ipynb` is the next launch notebook for `exp06_fix_trace_v2_2500_ep2_r4_inout`. The submission gate is hard-family behavior: cipher should improve from `0/40`, bit manipulation should improve from `3/44`, easy families should not collapse, and extraction/max-token failures should remain controlled. After this run, return to `eval_v2_grouped_family_balanced` before drawing broader conclusions.

### Status

Active.

## Decision: Separate Fused Mamba Training From 4-Bit Training

### Context

Installing `causal-conv1d` enables Nemotron's fused Mamba training path, but the first 4-bit fused-training attempt failed inside `mamba_split_conv1d_scan_combined(...)` with a matrix-shape error. The observed shape suggested a packed bitsandbytes projection weight was passed to a kernel expecting a normal dense matrix.

### Decision

Keep the practical 4-bit trace-training path in notebook `03` away from the fused `causal-conv1d` training kernel. Use notebook `04` only as the explicit fused-Mamba test path: dense BF16 base loading, no bitsandbytes quantization, attention-only LoRA targets, and a pre-training audit of `mixer.in_proj/out_proj` weights after LoRA is applied.

### Evidence

The current error matches a kernel/weight-layout incompatibility rather than a logical Mamba dimension choice. Mamba's inference speed and training speed come from different mechanisms, so fast fused training should be tested with dense Mamba projection weights before treating it as compatible with 4-bit LoRA training.

### Consequence

Notebook `04` is now configured as the fused-Mamba BF16 path and the active run name is `exp06_mamba_fused_bf16_attention_r4_trace_v2`. If it fails at model load or first training step due to memory, that is a BF16 feasibility result, not the same packed-weight shape failure. If it passes the projection audit and trains, fused Mamba training can be considered separately from the 4-bit production path.

### Status

Active.

## Decision: Track Fast BF16 Mamba Attention Run Separately From Occam 4-Bit Run

### Context

The user modified the Mamba notebook in Colab and started a fast run on the strict v2 trace file. The run uses dense BF16 base weights with fused Mamba kernels and attention-only LoRA targets, not the 4-bit `in_proj/out_proj` Occam path. The Mamba projection audit prints `has_lora_A: False` for every `mixer.in_proj/out_proj`, which is expected for attention-only LoRA, but it does not by itself prove that LoRA adapters exist elsewhere.

### Decision

Update notebook `04_colab_mamba_trace_train_and_submit.ipynb` to the running configuration `exp06_mamba_fused_bf16_attention_r4_trace_v2`, and add a separate LoRA-module audit that fails if no adapter modules exist and prints trainable parameters. Track this run as a fast Mamba BF16 attention variant, while keeping notebook `05` as the clean 4-bit fix-tracing Occam baseline.

### Evidence

The Colab run successfully tokenized `2,244` train rows and `256` eval rows from `trace_training_v2_2500.csv`, passed the dense BF16 Mamba projection audit, passed completion-only label masking, and completed step `72/72`. The separate LoRA audit reported `466,944` trainable parameters on attention projections (`q_proj/k_proj/v_proj/o_proj`), confirming that the adapter is real even though `mixer.in_proj/out_proj` has no LoRA. Trainer metrics improved early in training: validation loss `1.442285 -> 1.173332 -> 0.920066`, and mean token accuracy `0.687889 -> 0.716699 -> 0.771873`. Final five-row probe accuracy was only `1/5`: gravity matched, but unit conversion invented the wrong rule, cipher skipped the cited-map behavior, bit manipulation produced a long wrong trace, and equation stayed wrong. The downloaded run bundle later showed full generated eval `36/256 = 0.140625`: bit `2/91`, cipher `0/65`, equation `1/30`, gravity `4/25`, numeral `11/15`, and unit conversion `18/30`.

### Consequence

This run produces valid LoRA weights for Kaggle because attention LoRA adapters are still standard PEFT adapter files, and a strict local `submission.zip` was built from the run bundle. The zip was submitted to Kaggle as an information-gathering run, with public score pending. However, the generated-answer eval is far below the exp05 trace checkpoints and the old raw baseline, so the evidence says the fast BF16 Mamba attention-LoRA path can train quickly but is not a good modeling direction in this configuration unless the public score unexpectedly contradicts local diagnostics.

### Status

Active.

## Decision: Prepare All-Hard Trace V2 Dataset With Easy Rehearsal

### Context

The smaller `trace_training_v2_2500.csv` isolates trace quality with strict cipher and executable bit traces, but it excludes many hard-family rows. The user asked for a v2 trace-training file that includes all cipher, all bit-manipulation, all equation rows, and a small-to-medium percentage of the easier families.

### Decision

Create `trace_training_v2_all_hard_p25.csv` as a separate input file rather than overwriting the 2,500-row v2 file. Include all hard-family rows after excluding public sanity IDs: cipher, bit manipulation, and equation. Add deterministic 25% rehearsal slices from gravity, numeral, and unit conversion. Keep the target schema identical: `id`, `question`, `trace`, `gold_answer`.

### Evidence

The generated file has `5,921` rows: cipher `1,575`, bit manipulation `1,600`, equation `1,555`, gravity `399`, numeral `394`, and unit conversion `398`. The conservative trace policy avoids fake reasoning: strict cited-map traces for 605 cipher rows and boxed-only for 970 unsupported cipher rows; verified bit execution traces for 942 bit rows and boxed-only for 658 unverified bit rows; equation remains boxed-only until a real verifier exists. Local audit found exactly one boxed answer per row, zero official-style boxed mismatches, and no non-ASCII traces.

### Consequence

This file is prepared for a follow-up experiment such as `exp07_trace_v2_all_hard_p25`. It tests whether broad hard-family exposure plus easy-family rehearsal helps, but it is not pure procedural supervision because many hard rows are boxed-only. Generated eval by family remains the required selection gate.

### Status

Active.

## Decision: Try Cipher-Only Position-Level Trace Probe

### Context

The fast Mamba v2-trace run completed training and produced valid attention LoRA weights, but the five-row final probe was only `1/5`. The cipher row was especially informative: the model skipped the cited-map template and guessed a plausible phrase instead of performing character alignment.

### Decision

Create a separate cipher-only dataset with a stricter trace style. Include only rows where every target character is supported by aligned example pairs. For each target word, cite each target character's source word pair and source position, then assemble the plaintext word explicitly. Do not include unsupported cipher rows with phrase-context completion.

### Evidence

`scripts/build_trace_training_v3_cipher_only.py` generated `data/input/traces/trace_training_v3_cipher_only.csv` with 605 rows. There were 1,575 cipher candidates after excluding public sanity IDs; 970 were rejected because at least one final-answer character was unsupported by examples. Local audit found exactly one boxed answer per row, zero boxed/gold mismatches, unique IDs, and no non-ASCII traces. Average trace length is about 1,517 characters, with max about 2,236 characters.

### Consequence

This is a narrow cipher-learning probe, not a balanced leaderboard candidate. Use `notebooks/06_colab_cipher_only_trace_train_and_submit.ipynb`, which keeps `EVAL_ROWS=64`, max seq `1024`, max new `512`, LoRA `r=4` on `in_proj/out_proj`, LR `1e-4`, and 3 epochs. Judge success by cipher generated eval rather than overall public score. If it teaches alignment, the style can be mixed back into a balanced curriculum; if it fails, the problem is not just trace wording.

### Status

Active.

## Decision: Run Augmented Cipher V3 Synth Dataset Through Lightweight Mamba Notebook

### Context

The strict 605-row cipher-only dataset is clean but small. The user generated an augmented cipher dataset with existing prompt structure and 2,500 rows. The prior Mamba run showed that dense BF16 fused-Mamba training can fit in memory and train quickly, but generation callbacks during training add runtime overhead and can obscure whether the SFT objective itself is learning.

### Decision

Rename the augmented CSV to `trace_cipher_v3_synth2500.csv`, validate it, and create a dedicated Mamba notebook `07_colab_mamba_cipher_synth_train_and_submit.ipynb` for `exp09_mamba_cipher_v3_synth2500_b20_ep3`. Use batch size `20`, gradient accumulation `4`, epochs `3`, and attention LoRA targets `q_proj/k_proj/v_proj/o_proj`. During training, register only a lightweight trainer-log callback on `on_log`; do not register the generation callback with `on_log`/`on_save`.

### Evidence

The CSV audit passed: 2,500 rows, columns `id/question/trace/gold_answer`, all rows classified as cipher, unique IDs, every trace starts with `Thinking:\nCategory: cipher.`, exactly one boxed final answer per row, zero boxed/gold mismatches, zero unsupported target-character rows under the aligned-example verifier, and no non-ASCII traces. The local file hash matches the downloaded file hash.

### Consequence

This run isolates whether larger cipher-only position-trace data can teach character alignment. It is still a cipher-only curriculum, so even if cipher improves it may hurt non-cipher public behavior. Final generated eval and public sanity cipher output should decide whether to mix this data back into a balanced run.

### Status

Active.
