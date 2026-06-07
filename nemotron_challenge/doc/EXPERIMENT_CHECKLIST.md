# Experiment Checklist

Use this as the submission planning board. After each Kaggle score, copy the submitted zip and source run bundle into `data/outputs/submissions/`, then update the status and notes here.

## Status Legend

| Mark | Meaning |
| --- | --- |
| `[x]` | done / already scored |
| `[ ]` | not run yet |
| `NEXT` | recommended next |
| `RISK` | high-risk or needs preflight check |

## Resume Here

Latest local preparation is `2026-06-07`. Start here before reading the lower historical notes.

| Current answer | Detail |
| --- | --- |
| Best public baseline | `00-raw-1024`, score about `0.62`. |
| Recent S4 result | S4 final scored `0.53`; S4 checkpoint-144 scored `0.55`. Checkpoint timing helped a little but did not rescue S4. |
| Recent exp05 result | Trace Occam checkpoint 144 scored `0.59`, slightly above checkpoint 96 at `0.58`; both have the same local generated eval (`134/256`). |
| Recent exp06 result | `exp06_mamba_fused_bf16_attention_r4_trace_v2` scored `0.58` on Kaggle. Local generated eval was poor (`36/256`), so treat it as a fast-path diagnostic, not a strong method. |
| Recent exp09 result | `exp09_mamba_cipher_v3_synth2500_b20_ep3` finished locally and is rejected: generated eval `0/64`, cipher `0/64`, and `63/64` max-token hits. |
| Recent exp10 result | `exp10_mamba_cipher_v4_natural2500_b12_seq1024_ep3` scored `0.49`. Local cipher generated eval improved to `26/64 = 0.40625`, but cipher-only training was not leaderboard-safe. |
| Recent exp11 result | `exp11_mamba_trace_v2_aug25k_b8_ep1` checkpoints 195 and 585 both scored `0.56`; checkpoints 390 and 780 both scored `0.54`. Checkpoint 195 had weaker backfill generated eval (`11/64 = 0.171875`) than checkpoint 390 (`17/64 = 0.265625`), so the current small generated eval remains diagnostic rather than a public-score proxy. |
| Recent exp12 result | `exp12_mamba_trace_v2_aug25k_5k_inout_vo_r4_ep1` checkpoint/final 105 scored `0.58`; checkpoint 26 scored `0.57`. Both submitted adapters report LoRA `r=9`, not the intended `r=4`, so treat them as useful 5k inout+v/o evidence but not a clean rank-4 test. |
| Current exp13 result | `exp13_trace_v2_aug25k_10k_inout_r25_ep1` checkpoint 78 and final step 313 both scored `0.57`. The final step had local generated eval `35/64 = 0.546875`, but the public score stayed below exp12 checkpoint105 (`0.58`), exp05 checkpoint144 (`0.59`), and the raw partial baseline (`0.62`). |
| Current exp14 result | `exp14_rs_hard_vllm_cipher_bit_v1` is a useful negative data-builder run, not a submission. Full sampling produced only `2` selected bit rows and `0` selected cipher rows; cipher had `2/6300` correct raw candidates and `0` accepted. The follow-up `exp18_cipher_hinted_rs_test_v1` produced `0/1600` correct cipher candidates and `1600/1600` max-token hits even with vocabulary and word-length hints. |
| Public sanity remark | Every tracked experiment so far has failed all three public `test.csv` sanity rows. Treat those rows as a hard warning signal, not as solved by any current method. |
| Current conclusion | Natural cipher alignment and v2 broad traces improved output discipline, but neither taught enough hard-family rule inference. Free-form base-model rejection sampling also failed, especially for cipher. Because exp18 found zero accepted cipher rows, pure RFT has no data and should stop for cipher. |
| Current exp19 probe | First Colab attempt loaded vLLM and wrote raw outputs, but length control failed: the 16-row speed benchmark stopped by `length`, and displayed raw probe rows hit `MAX_NEW_TOKENS=256`, so no training decision should be made from that attempt. Notebook `12` has been patched to use a fixed four-line rationale prompt plus stop-after-box vLLM sampling. |
| Current exp19 v2 signal | Downloaded and archived `exp19_cipher_bit_star_rationale_probe_v2-20260607T184337Z-3-001.zip` under `data/outputs/downloaded_diagnostics/2026-06-07_exp19_cipher_bit_star_rationale_probe_v2_diagnostic/`, SHA-256 `954f4a83bb8899dc3fa2a9d92e7279aa2858e0b9ae7e9701e197f4f1a463310f`. The v2 probe selected `13/16` cipher rows and `2/16` bit rows. The cipher answers were mostly correct, but manual inspection showed weak proofs: accepted traces usually aligned one example word, left the target line as raw encrypted text, and jumped to the gold phrase. Both selected bit rows were invalid self-contradicting traces, proving exact-answer verification plus weak trace-signal gates are not sufficient for hard-family rationales. |
| Next active task | Rerun notebook `12_colab_star_rationale_data_builder.ipynb` as `exp19_cipher_bit_star_rationale_probe_v5`: answer-conditioned STaR-style rationale generation on a balanced 64-row probe, `32` cipher and `32` bit manipulation rows, using vLLM with `4` candidates per row. The v5 notebook adopts the user's fuller few-shot `Pattern` / `Target words` / `Check` prompt, but keeps a strict proof gate: cipher must include one target cipher_word -> plain_word pair per final-answer word, and bit must include a target 8-bit transformation whose result equals the boxed answer. Inspect raw/accepted rationales before scaling. If bit rationales still invent rules, stop model-written bit STaR and build an expanded bit DSL/rule-search diagnostic. Train only after coverage and trace quality are real. |
| Current input file | `data/input/traces/trace_v2_aug25k.csv`; copy for Colab upload is `C:\Users\mouak\Downloads\trace_v2_aug25k.csv`. |
| After that | Build and audit the new trace CSV before launching another notebook; do not choose by teacher-forced eval loss alone. |
| Do not resume from | The lower legacy S0-S7 checklist. It is historical context, not the active continuation path. |

## Current Rule

Do not run a full factorial grid. We cannot afford every combination of prompt, target format, rank, LR, sequence length, and epoch count. Use bundled runs where each bundle has a clear hypothesis and avoids changing unrelated risky knobs.

Originality rule: all outside research is allowed as literature, tools, and inspiration, including papers, open-source code, Reddit/Medium/Substack posts, YouTube talks, other-domain competition ideas, and reasoning-challenge methods that are not from this exact Nemotron Kaggle challenge. Do not inspect or copy another active Nemotron challenge competitor's high-scoring notebook, GitHub repo, trajectory dataset, or submission path as the main method unless the user explicitly approves that. Our trajectory should come from our own verifier-guided STaR experiments.

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

## Near-Term Research Plan

- [ ] `exp06_fix_trace_v2_2500_ep2_r4_inout` `NEXT` `RISK`
  - Purpose: test whether fixing trace quality for the hard families helps more than simply adding generic traces.
  - Input: `data/input/traces/trace_training_v2_2500.csv`, built by `scripts/build_trace_training_v2_2500.py`.
  - Data mix: 600 strict cipher traces where every target character cites an aligned example pair; 900 verified bit-manipulation DSL traces with explicit 8-bit execution; 250 boxed-only equation rows; 250 compact unit rows; 250 compact gravity rows; 250 boxed-only numeral rows.
  - Key config: same as exp05 except two epochs for the smaller 2,500-row curriculum: LoRA `r=4`, alpha `32`, dropout `0.05`, `in_proj/out_proj`, max seq `512`, max new `384`, LR `1e-4`, completion-only loss, 2 epochs.
  - Submit gate: choose by checkpoint, not final step blindly. Cipher must improve from `0/40`, bit must improve from `3/44`, easy families should not collapse, and extraction/max-token failures should stay controlled.

- [x] `exp06_mamba_fused_bf16_attention_r4_trace_v2` `RISK`
  - Purpose: test the same strict v2 trace file with fused Mamba enabled and dense BF16 base weights, using attention-only LoRA targets.
  - Input: `data/input/traces/trace_training_v2_2500.csv`; split is 2,244 train rows and 256 eval rows.
  - Key config: `USE_4BIT=False`, fused `mamba-ssm causal-conv1d`, LoRA `r=4`, alpha `32`, dropout `0.1`, targets `q_proj/k_proj/v_proj/o_proj`, batch `16`, grad accumulation `4`, max seq `512`, max new `384`, LR `1e-4`, completion-only loss, 2 epochs.
  - Training signal from Colab: Mamba projection audit passed with dense BF16 `mixer.in_proj/out_proj` and no accidental LoRA on Mamba projections; LoRA audit found real attention adapters with `466,944` trainable parameters on `q_proj/k_proj/v_proj/o_proj`; training reached step `72/72`.
  - Final probe signal: `1/5` on the five-row probe at step 72. Gravity matched; unit conversion invented an incorrect rule, cipher skipped the cited-map template and guessed a phrase, bit manipulation generated a long but wrong execution trace, and equation remained wrong.
  - Final generated eval from downloaded run bundle: `36/256 = 0.140625`; bit `2/91`, cipher `0/65`, equation `1/30`, gravity `4/25`, numeral `11/15`, unit conversion `18/30`; `13` max-token hits and no empty answers.
  - Archive: `data/outputs/submissions/2026-05-25_exp06_mamba_fused_bf16_attention_r4_trace_v2_submitted_pending/`. A strict `submission.zip` was submitted to Kaggle; public score returned `0.58`.
  - Interpretation: technically valid and fast, but behaviorally failed locally. Treat the upload as information gathering, not a serious candidate. The attention-only BF16 Mamba path learned the trace text under teacher forcing while damaging generated-answer behavior.

- [ ] `exp08_cipher_only_v3_position_trace` `NEXT` `RISK`
  - Purpose: isolate whether cipher can be taught by position-level aligned evidence, after the Mamba probe showed the model skipped the cited-map behavior and guessed a phrase.
  - Input: `data/input/traces/trace_training_v3_cipher_only.csv`, built by `scripts/build_trace_training_v3_cipher_only.py`.
  - Notebook: `notebooks/06_colab_cipher_only_trace_train_and_submit.ipynb`.
  - Data mix: 605 cipher rows only; every row has full target-character support from aligned example pairs. The other 970 cipher rows are excluded because at least one final-answer character is unsupported by examples.
  - Trace policy: one target word at a time, one character position at a time, with source word pair and source character position cited before assembling the plaintext word.
  - Notebook config: `EVAL_ROWS=64`, LoRA `r=4`, `in_proj/out_proj`, LR `1e-4`, max seq `1024`, max new `512`, 3 epochs. Treat this as a cipher-learning probe, not a balanced leaderboard candidate.
  - Submit gate: submit only if cipher generated eval meaningfully improves; this run may hurt non-cipher families because it contains no rehearsal.

- [x] `exp09_mamba_cipher_v3_synth2500_b20_ep3` `RISK`
  - Purpose: test augmented cipher-only position traces at 2,500 rows using the fast dense-BF16 fused-Mamba path.
  - Input: `data/input/traces/trace_cipher_v3_synth2500.csv`, renamed from `trace_training_v3_cipher_only_2500_synth_existing_prompts.csv`.
  - Data audit: 2,500 rows; columns `id/question/trace/gold_answer`; all rows infer as cipher; unique IDs; every trace starts with `Thinking:\nCategory: cipher.`; exactly one boxed final answer; zero boxed/gold mismatches; zero unsupported target-character rows under the aligned-example verifier; no non-ASCII traces.
  - Notebook: `notebooks/07_colab_mamba_cipher_synth_train_and_submit.ipynb`.
  - Key config: `USE_4BIT=False`, fused Mamba, LoRA `r=4`, targets `q_proj/k_proj/v_proj/o_proj`, batch `20`, grad accumulation `4`, epochs `3`, LR `1e-4`, max seq `512`, max new `384`, final generated eval enabled, checkpoint generated eval disabled.
  - Callback policy: only lightweight `TrainerLogFlushCallback.on_log` is registered. The `GenerationCallback` with `on_log`/`on_save` generation is not registered, so training should not pause for probe/checkpoint generation.
  - In-flight signal from Colab: split is `2,436` train rows and `64` eval rows. Adapter audit found `24` LoRA modules on `q_proj/k_proj/v_proj/o_proj` across layers `5,12,19,26,33,42`, with `466,944` trainable parameters (`0.0015%`). Completion-only mask starts at `Thinking:\nCategory: cipher.`. Through step `33/93`, loss and token accuracy are improving: train loss `3.7269 -> 2.9216 -> 2.0024`, validation loss `0.8091 -> 0.6056 -> 0.4055`, entropy `0.7544 -> 0.6911 -> 0.5306`, mean token accuracy `0.8280 -> 0.8552 -> 0.8811`.
  - Probe-evolution signal after training: the only downloaded probe row (`d300a576`) failed both before and after training. Both outputs hit `384/384` generated tokens. Before training, the model reasoned naturally over example word alignments but did not reach the target. After training, it followed the position-citation template and started from the target phrase, but produced invalid source citations and still did not reach a boxed final answer.
  - Final generated-eval signal: `0/64 = 0.0`, all cipher, with `63/64` max-token hits and average `382.8/384` generated tokens. Public sanity outputs also maxed out and repeated the cipher template even on non-cipher binary rows.
  - Archive: `data/outputs/submissions/2026-05-25_exp09_mamba_cipher_v3_synth2500_b20_ep3_local_rejected/`. A strict `submission.zip` was built for completeness but this adapter is not recommended for Kaggle submission.
  - Conclusion: teacher-forced loss was excellent, but generation failed. The long position-citation template is too verbose and too easy to imitate without real verification. Next cipher traces should be shorter, target-only, and closer to natural alignment reasoning.

- [x] `exp10_mamba_cipher_v4_natural2500_b12_seq1024_ep3` `RISK`
  - Purpose: test whether the base model's natural cipher reasoning style can be lightly reinforced without the long v3 position-citation failure.
  - Input: `data/input/traces/trace_cipher_v4_natural2500.csv`, built by `scripts/build_trace_training_v4_cipher_natural.py` from the same 2,500 augmented cipher prompts as exp09.
  - Notebook: `notebooks/08_colab_mamba_cipher_v4_natural_train_and_submit.ipynb`.
  - Trace policy: mimic the before-training probe pattern closely: `We need to find mapping from cipher to plaintext. Given examples:`, then `Cipher: ... -> plaintext: ...`, then `Let's align words. Cipher words... Plain words...`, then `So mapping per word? Let's map letters.`, one fully walked example, a short note that other examples add consistency checks, target application, and one boxed final answer. It removes position citations and `Thinking:/Category:` boilerplate.
  - Data audit: 2,500 rows; unique IDs/questions/traces; zero boxed/gold mismatches; zero missing boxed answers; zero non-ASCII traces; zero leftover `pos` or `source position` language. Trace length averages about `1,912` characters, still below the worst v3 rows while preserving the model-native pattern.
  - Key config actually run: fast dense-BF16 fused-Mamba setup, LoRA `r=14`, alpha `64`, targets `q_proj/k_proj/v_proj/o_proj`, batch `8`, grad accumulation `4`, epochs configured `3` but stopped at step `92`, LR `1e-4`, max seq `2048`, max new `1024`, final generated eval enabled, checkpoint generated eval disabled.
  - Preflight note: if the notebook assertion says the first trained labels start with `We need to find mapping...`, that is correct for v4. Notebook `08` was patched to accept this natural-prefix completion.
  - Local evidence: generated cipher eval `26/64 = 0.40625`, empty answers `0`, max-token hits `6/64`, average generated tokens `787`. The fixed probe `7f8f89aa` reached a boxed answer but failed by mapping `ynz -> the` in the wrong order, producing `hatter creates chase` instead of `teacher creates castle`.
  - Public score: `0.49`.
  - Archive: `data/outputs/submissions/2026-05-26_exp10_mamba_cipher_v4_natural2500_b12_seq1024_ep3_ready/`. A strict `submission.zip` was built and submitted. Use as evidence that cipher-only natural alignment is not leaderboard-safe, not as proof that natural alignment is bad.

- [ ] `exp07_trace_v2_all_hard_p25` `RISK`
  - Purpose: train on every hard-family row while keeping a small/medium rehearsal slice from easy families.
  - Input: `data/input/traces/trace_training_v2_all_hard_p25.csv`, built by `scripts/build_trace_training_v2_all_hard.py`.
  - Data mix: all cipher rows (`1,575`), all bit-manipulation rows (`1,600`), all equation rows (`1,555`), plus about 25% each of gravity (`399`), numeral (`394`), and unit conversion (`398`), for `5,921` rows total.
  - Trace policy: strict cited-map traces for 605 cipher rows and boxed-only for 970 unsupported cipher rows; verified bit execution traces for 942 bit rows and boxed-only for 658 unverified bit rows; all equation rows boxed-only; compact unit/gravity traces; numeral boxed-only.
  - Risk: many hard rows are boxed-only, so this tests hard-family exposure and rehearsal more than complete procedural supervision. Do not interpret lower loss as hard-family reasoning unless generated eval improves cipher/bit/equation.

- [x] `exp11_mamba_trace_v2_aug25k_b8_ep1` `RISK`
  - Purpose: test whether scaling the verified v2-style hard-family curriculum to 25k rows improves public score while keeping broad family coverage.
  - Input: `data/input/traces/trace_v2_aug25k.csv`, copied from audited bundle `trace_training_v2_augmented_25000_bundle.zip`; upload as `/content/trace_v2_aug25k.csv` in Colab.
  - Data mix: cipher `10,000`, bit manipulation `10,000`, equation `2,000`, gravity `1,000`, numeral `1,000`, unit conversion `1,000`.
  - Audit: 25,000 rows, unique IDs and questions, no public sanity test ID leakage, exactly one boxed answer per row, zero boxed/gold mismatches, zero non-ASCII traces, zero cipher verifier failures, zero bit verifier failures.
  - Notebook: `notebooks/04_colab_mamba_trace_train_and_submit.ipynb`.
  - Key config currently set: dense BF16 fused-Mamba path, LoRA attention targets `q_proj/k_proj/v_proj/o_proj`, batch `8`, grad accumulation `8`, 1 epoch, LR `1e-4`, final generated eval `64` rows, checkpoint generated eval disabled, training-time generation callback disabled.
  - Step-205 diagnostic: generated eval `21/64 = 0.328125`, bit `0/20`, cipher `8/26`, equation `4/8`, gravity `2/3`, numeral `3/3`, unit conversion `4/4`, with zero max-token hits.
  - Checkpoint-390 backfill diagnostic: generated eval `17/64 = 0.265625`, bit `4/20`, cipher `3/26`, equation `1/8`, gravity `3/3`, numeral `3/3`, unit conversion `3/4`, with one max-token hit.
  - Public score: checkpoints 195 and 585 returned `0.56`; checkpoints 390 and 780 returned `0.54`.
  - Conclusion: this is mostly v2-style cited maps/execution, not the v5 ordered-alignment fix. Scaling v2 improved stopping/format but did not beat exp05 checkpoint144 or the `0.62` raw baseline.

- [x] `exp12_mamba_trace_v2_aug25k_inout_qkvo_r4_ep1`
  - Purpose: test whether adding the Mamba `in_proj/out_proj` LoRA surface to the fused BF16 Mamba attention target set improves the broad 25k v2 trace run.
  - Input: `data/input/traces/trace_v2_aug25k.csv`, uploaded as `/content/trace_v2_aug25k.csv` in Colab.
  - Key config: dense BF16 fused Mamba, LoRA `r=4`, alpha `32`, dropout `0.05`, targets `in_proj/out_proj/q_proj/k_proj/v_proj/o_proj`, batch `12`, grad accumulation `4`, one epoch, LR `7.5e-5`, max seq `512`, max new `384`.
  - Evidence: completed step `516/516`; final generated eval `36/64 = 0.5625`, with bit `3/20`, cipher `18/26`, equation `5/8`, gravity `3/3`, numeral `3/3`, unit `4/4`, one max-token hit, and probe `5/5`.
  - Public score: `0.54`.
  - Archive: `data/outputs/submissions/2026-05-27_exp12_mamba_trace_v2_aug25k_inout_qkvo_r4_ep1_score_0_54/`.
  - Interpretation: this was the strongest recent broad local generated-eval signal, but the public score did not translate. Treat it as another local/public mismatch and do not assume all-six projection targeting solves the v2 trace issue.

- [ ] `exp13_mamba_trace_v2_aug25k_qplusc_qkvo_r4_ep1` `RISK` `INACTIVE`
  - Purpose: test whether completion-only loss made teacher-forced metrics too easy by masking the puzzle/question tokens.
  - Notebook: not currently present locally. The checklist referenced `notebooks/10_colab_mamba_trace_v2_aug25k_question_loss_train_and_submit.ipynb`, but the file is absent as of 2026-06-03, so do not treat this as a runnable next experiment without recreating or finding the notebook first.
  - Input: `data/input/traces/trace_v2_aug25k.csv`; upload as `/content/trace_v2_aug25k.csv` in Colab.
  - Loss mask: active labels on the original question plus trace/answer; masked labels on fixed `System:`, system prompt, `User:`, and `Assistant:` wrappers.
  - Key config: fused dense-BF16 Mamba, LoRA `r=4`, alpha `32`, dropout `0.05`, targets `q_proj/k_proj/v_proj/o_proj`, batch `8`, grad accumulation `8`, one epoch, LR `7.5e-5`, max seq `512`, max new `384`, final generated eval limited to 64 rows.
  - Submit gate: ignore teacher-forced loss as a cross-run proxy; judge by generated eval, probe outputs, sanity raw outputs, and public score.

- [ ] `eval_v2_grouped_family_balanced` `NEXT`
  - Purpose: replace the current 256-row generated eval as the main local diagnostic.
  - Design: 1,000-2,000 rows if runtime allows, family-balanced, grouped by prompt/rule pattern where feasible, with overall accuracy, hard-family accuracy excluding numeral, per-family accuracy, extraction failures, max-token hits, and raw completions.
  - Baselines to score: base model, `00-raw-1024`, `02-raw-full`, `04-s4`, and future STaR adapters.
  - Risk: larger eval is slower; keep it diagnostic, not a perfect leaderboard proxy.

- [ ] `exp05_star_seed_512` `NEXT` `RISK`
  - Hypothesis: a small set of compact verified reasoning traces can teach procedure without damaging the base model behavior that gave the `0.62` public score.
  - Method: start from `data/input/traces/trace_training.csv`; use original questions with target-side cipher character-position traces plus boxed answers, with non-cipher rows boxed-only unless a family-specific trace exists.
  - Key config candidate: LoRA `r=8`, targets `in_proj/out_proj/q_proj/k_proj/v_proj/o_proj`, LR `1e-4`, 1 epoch, max seq `2048` if memory allows.
  - Success signal: `eval_v2` hard-family accuracy improves and public score stays near or above `0.62`.

- [ ] `exp06_star_cipher_unit_bit_1500` `RISK`
  - Hypothesis: verifier-friendly families can improve first: cipher, unit conversion, and bit manipulation.
  - Method: about 500 verified traces per family, using family-specific solvers/verifiers before training.
  - Key config candidate: LoRA `r=8`, `in_proj/out_proj` first, LR `1e-4`, max seq `2048` or `4096` if memory allows, 1 epoch.
  - Risk: traces can be correct-final-answer but bad reasoning; rejected traces and verifier failure reasons must be saved.

## Submission Runs

| Done | Run | Public score / status | Hypothesis | Key config | Main evidence | Archive / next |
| --- | --- | --- | --- | --- | --- | --- |
| [x] | `ACTIVE_02_raw_full_r4` | `0.54` scored | Full official data improves the known raw-answer baseline without changing prompt/format behavior. | Raw target, direct prompt, full train minus 256 eval rows, LoRA `r=4`/alpha `32`, `in_proj/out_proj`, seq `512`, max new `64`, effective batch `48`, LR `3e-4`, 1 epoch. | Eval loss improved to about `0.7933`, but probe stayed about `1/5`; public score dropped below the `0.62` partial baseline. Full-data raw final-answer SFT is not sufficient. | Archive: `data/outputs/submissions/2026-05-17_colab_raw_full_r4_score_0_54/`. |
| [ ] | `ACTIVE_03_private_reasoning_boxed_r8` | Interrupted / artifacts likely lost | Private-reasoning prompt plus boxed targets improves final-answer discipline and gives rank 8 enough capacity to learn transformations. | Boxed target, private-reasoning boxed prompt, LoRA `r=8`/alpha `64`, `in_proj/out_proj`, seq `512`, full data, 256 eval rows, LR `3e-4`, 1 epoch. | Runtime-local artifacts appear lost after Colab disconnect unless a Drive/manual download copy exists. This run also combines rank and output-format changes, so it is hard to interpret. | Recover only if a Drive/manual copy exists; otherwise do not revive as a priority. |
| [x] | `ACTIVE_04_S4_attention_expand_r8_private_boxed_max128` | Final `0.53`; checkpoint-144 `0.55` | Expanded attention LoRA improves transformation learning while keeping private-reasoning boxed setup. | Boxed target, private-reasoning boxed prompt, LoRA `r=8`/alpha `64`, `in_proj/out_proj/q_proj/k_proj/v_proj/o_proj`, seq `512`, max new `128`, effective batch `48`, LR `3e-4`, 1 epoch. | Final step 193 had best local generated eval `95/256 = 0.371094` but scored `0.53`; checkpoint-144 had lower local generated eval `90/256 = 0.351562` but scored `0.55`. Checkpoint timing matters, but S4 still did not beat the `0.62` raw baseline. | Archives: `data/outputs/submissions/2026-05-17_colab_s4_attention_boxed_r8_final_score_0_53/`, `data/outputs/submissions/2026-05-17_colab_s4_checkpoint144_score_0_55/`. Shift to `eval_v2` and solver-guided STaR unless checkpoint 192 has a strong new signal. |
| [x] | `exp05_trace_occam_r4_inout` checkpoint 96 | `0.58` scored | Canonical verified trace targets plus completion-only loss can improve hard-family procedure without expanded attention LoRA. | Trace/boxed target column, original questions, LoRA `r=4`/alpha `32`/dropout `0.05`, `in_proj/out_proj`, seq `512`, max new `384`, LR `1e-4`, Mamba generation backfill. | Local generated eval `134/256 = 0.5234375`, probe `3/5`, public score `0.58`. Better than raw-full and S4, still below `00-raw-1024` at about `0.62`. Strong unit/gravity/numeral; cipher `0/40`, bit/equation weak. | Archive: `data/outputs/submissions/2026-05-25_exp05_trace_occam_checkpoint96_mamba_score_0_58/`. |
| [x] | `exp05_trace_occam_r4_inout` checkpoint 144 | `0.59` scored | Later checkpoint might benefit from lower train/eval loss while preserving the trace behavior from checkpoint 96. | Same as checkpoint 96; selected checkpoint step 144. | Public score improved from checkpoint 96's `0.58` to `0.59`, despite unchanged local generated eval at `134/256 = 0.5234375`. Only two local eval rows changed versus checkpoint 96: one numeral gain and one bit-manipulation loss. Still below `00-raw-1024` at about `0.62`. | Archive: `data/outputs/submissions/2026-05-25_exp05_trace_occam_checkpoint144_mamba_score_0_59/`. |
| [x] | `exp06_mamba_fused_bf16_attention_r4_trace_v2` | `0.58` scored | Fast dense-BF16 fused-Mamba attention LoRA might provide a cheaper training/eval path for v2 trace data. | Trace v2 2,500-row curriculum, dense BF16 fused Mamba, LoRA `r=4` on `q_proj/k_proj/v_proj/o_proj`, batch `16`, grad accumulation `4`, LR `1e-4`, 2 epochs. | Public score returned `0.58`, near exp05 checkpoint96, despite weak local generated eval `36/256 = 0.140625` and cipher `0/65`. It is useful as a runtime-path diagnostic, not as a model-quality win. | Archive: `data/outputs/submissions/2026-05-25_exp06_mamba_fused_bf16_attention_r4_trace_v2_submitted_pending/`; metadata status is now scored. |
| [x] | `exp10_mamba_cipher_v4_natural2500_b12_seq1024_ep3` | `0.49` scored | Natural alignment traces might teach cipher decoding better than rigid position-citation traces while still reaching final boxed answers. | Cipher-only v4 natural traces, dense BF16 fused Mamba, LoRA `r=14` on `q_proj/k_proj/v_proj/o_proj`, batch `8`, grad accumulation `4`, LR `1e-4`, max seq `2048`, max new `1024`, stopped/evaluated at step `92`. | Local cipher generated eval `26/64 = 0.40625`; max-token hits dropped to `6/64` from exp09's `63/64`. Public score fell to `0.49`, so cipher-only training damaged full-leaderboard behavior. | Archive: `data/outputs/submissions/2026-05-26_exp10_mamba_cipher_v4_natural2500_b12_seq1024_ep3_ready/`. Keep natural alignment as an ingredient in a broad dataset, not as a standalone submission path. |
| [x] | `exp11_mamba_trace_v2_aug25k_b8_ep1` checkpoint 195 | `0.56` scored | Earlier checkpoint might preserve more base-model behavior than later exp11 checkpoints. | 25k v2 augmented traces, dense BF16 fused Mamba, LoRA `r=14` on `q_proj/k_proj/v_proj/o_proj`, batch `8`, grad accumulation `4`, LR `1e-4`, max seq `512`, max new `384`, checkpoint 195. | Backfill generated eval is `11/64 = 0.171875`, with bit `0/20`, cipher `3/26`, equation `1/8`, gravity `3/3`, numeral `1/3`, unit `3/4`, and one max-token hit. This is worse than checkpoint 390 locally, but public score is better (`0.56` vs `0.54`). | Archive: `data/outputs/submissions/2026-05-26_exp11_mamba_trace_v2_aug25k_checkpoint195_pending/`. |
| [x] | `exp11_mamba_trace_v2_aug25k_b8_ep1` checkpoint 390 | `0.54` scored | Scaling broad v2 trace data to 25k might improve public score while preserving all-family coverage. | 25k v2 augmented traces, dense BF16 fused Mamba, LoRA `r=14` on `q_proj/k_proj/v_proj/o_proj`, batch `8`, grad accumulation `4`, LR `1e-4`, max seq `512`, max new `384`, checkpoint 390. | Backfill generated eval is `17/64 = 0.265625`, with bit `4/20`, cipher `3/26`, equation `1/8`, gravity/numeral perfect on tiny samples, and one max-token hit. Public score returned `0.54`, below exp05 checkpoint144 and the `0.62` raw baseline. | Archive: `data/outputs/submissions/2026-05-26_exp11_mamba_trace_v2_aug25k_checkpoint390_ready/`. Do not keep scaling v2 as-is; redesign hard-family traces. |
| [x] | `exp11_mamba_trace_v2_aug25k_b8_ep1` checkpoint 585 | `0.56` scored | Later checkpoint might recover from the checkpoint-390 public drop while using the same broad v2 trace curriculum. | 25k v2 augmented traces, dense BF16 fused Mamba, LoRA `r=14` on `q_proj/k_proj/v_proj/o_proj`, batch `8`, grad accumulation `4`, LR `1e-4`, max seq `512`, max new `384`, checkpoint 585. | Public score returned `0.56`, matching checkpoint 195 and beating checkpoint 390. No checkpoint-585 generated-eval diagnostics are archived yet, so it should be backfilled before using local comparisons. | Archive: `data/outputs/submissions/2026-05-27_exp11_mamba_trace_v2_aug25k_checkpoint585_ready/`. |
| [x] | `exp11_mamba_trace_v2_aug25k_b8_ep1` checkpoint 780 | `0.54` scored | Final checkpoint might recover enough from the checkpoint-390 drop to justify using the completed epoch adapter. | 25k v2 augmented traces, dense BF16 fused Mamba, LoRA `r=14` on `q_proj/k_proj/v_proj/o_proj`, batch `8`, grad accumulation `4`, LR `1e-4`, max seq `512`, max new `384`, checkpoint 780. | Public score returned `0.54`, matching checkpoint 390 and below checkpoints 195 and 585 at `0.56`. No checkpoint-780 generated-eval diagnostics are archived yet, so it should be backfilled only if we need a complete local sweep. | Archive: `data/outputs/submissions/2026-05-27_exp11_mamba_trace_v2_aug25k_checkpoint780_ready/`. |
| [x] | `exp12_mamba_trace_v2_aug25k_inout_qkvo_r4_ep1` | `0.54` scored | Adding `in_proj/out_proj` to the fused BF16 attention target set may improve broad v2 trace behavior without the 4-bit non-fused fallback. | 25k v2 augmented traces, dense BF16 fused Mamba, LoRA `r=4` on `in_proj/out_proj/q_proj/k_proj/v_proj/o_proj`, batch `12`, grad accumulation `4`, LR `7.5e-5`, max seq `512`, max new `384`, one epoch. | Local generated eval `36/64 = 0.5625`; cipher `18/26`, bit `3/20`, equation `5/8`, gravity/numeral/unit perfect on tiny samples, probe `5/5`, one max-token hit. Public score returned only `0.54`, so the strong local diagnostic did not predict leaderboard gain. | Archive: `data/outputs/submissions/2026-05-27_exp12_mamba_trace_v2_aug25k_inout_qkvo_r4_ep1_score_0_54/`. |
| [x] | `exp12_mamba_trace_v2_aug25k_5k_inout_vo_r4_ep1` checkpoint 26 | `0.57` scored | Smaller 5k subset and reduced target set might preserve more base behavior while touching Mamba plus value/output attention paths. | Intended config: 5k rows, max seq `1024`, max new `256`, targets `in_proj/out_proj/v_proj/o_proj`, batch `12`, grad accumulation `4`, LR `7.5e-5`, one epoch. Archived zip reports LoRA `r=9`, alpha `32`, dropout `0.05`, targets `in_proj/v_proj/o_proj/out_proj`, so it is not exactly the intended `r=4` config. | Public score returned `0.57`, better than exp12 full 25k all-six at `0.54` and exp11 checkpoint390/780 at `0.54`, but below exp05 checkpoint144 at `0.59`. No exact checkpoint26 generated-eval diagnostics are archived. | Archive: `data/outputs/submissions/2026-05-27_exp12_mamba_trace_v2_aug25k_5k_inout_vo_checkpoint26_score_0_57/`. |
| [x] | `exp12_mamba_trace_v2_aug25k_5k_inout_vo_r4_ep1` checkpoint/final 105 | `0.58` scored | Later checkpoint of the same 5k diagnostic might differ from checkpoint 26 while preserving the smaller-row hypothesis. | Same observed adapter config shape as checkpoint 26: LoRA `r=9`, alpha `32`, dropout `0.05`, targets `in_proj/v_proj/o_proj/out_proj`; intended thread config was `r=4`. | Public score returned `0.58`, beating checkpoint 26 at `0.57` and exp12 full all-six at `0.54`, matching exp05 checkpoint96, but still below exp05 checkpoint144 at `0.59`. The final run bundle is attached to the scored archive as related last-step diagnostics for simpler tracking. | Archive: `data/outputs/submissions/2026-05-27_exp12_mamba_trace_v2_aug25k_5k_inout_vo_checkpoint105_score_0_58/`. |
| [x] | `exp12_mamba_trace_v2_aug25k_5k_inout_vo_r4_ep1` final step 105 diagnostics | local-only | Final run bundle diagnostics are retained for inspection but are not shown in the dashboard. | Run config reports LoRA `r=9`, alpha `32`, dropout `0.05`, targets `in_proj/out_proj/v_proj/o_proj`, 5k rows, max seq `1024`, max new `256`, batch `12`, grad accumulation `4`, LR `7.5e-5`. | Final generated eval is poor: `8/64 = 0.125`, cipher `0/26`, bit `0/20`, equation `2/8`, gravity `1/3`, numeral `3/3`, unit `2/4`, and `21` max-token hits. Kept as local evidence only. | Archive: `data/outputs/submissions/2026-05-27_exp12_mamba_trace_v2_aug25k_5k_inout_vo_final105_local_rejected/`. |
| [x] | `exp13_trace_v2_aug25k_10k_inout_r25_ep1` checkpoint 78 | `0.57` scored | Early checkpoint might preserve more base-model behavior than the final rank-25 adapter. | Same run config as final exp13: dense BF16 fused Mamba, 10k rows, LoRA `r=25`, alpha `128`, dropout `0.05`, targets `in_proj/out_proj`, batch `8`, grad accumulation `4`, LR `7.5e-5`, max seq `1024`, max new `324`. | Public score returned `0.57`. The strict checkpoint zip was found in Downloads and archived locally; SHA-256 `8be6648cd722f2c91ad3c7407bc1fe57812e261dcadad6c2fdabe511f4b6004c`. Trainer eval at step 78 was `eval_loss=0.068001` and token accuracy `0.976612`. | Archive: `data/outputs/submissions/2026-05-28_exp13_trace_v2_aug25k_10k_inout_r25_checkpoint78_score_0_57_reported/`. |
| [x] | `exp13_trace_v2_aug25k_10k_inout_r25_ep1` final step 313 | `0.57` scored | Test whether a larger clean LoRA rank on Mamba `in_proj/out_proj` only improves the 10k-row v2 trace path without adding attention targets or 4-bit ambiguity. | Dense BF16 fused Mamba, 10k rows from `trace_v2_aug25k.csv`, LoRA `r=25`, alpha `128`, dropout `0.05`, targets `in_proj/out_proj`, batch `8`, grad accumulation `4`, LR `7.5e-5`, max seq `1024`, max new `324`, one epoch. | Adapter audit passed: `46` LoRA modules and `11,371,200` trainable params. Local final generated eval is `35/64 = 0.546875`: cipher `18/26`, bit `4/20`, equation `3/8`, gravity/numeral/unit perfect on tiny samples, probe `4/5`, and `3` max-token hits. Public score returned `0.57`, matching checkpoint 78 and confirming that capacity alone does not rescue the current v2 trace format. | Archive: `data/outputs/submissions/2026-05-28_exp13_trace_v2_aug25k_10k_inout_r25_ep1_submitted_pending/`. |

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

## Legacy Experiment Notes

This table preserves older ideas for context only. It is not the active continuation plan. Use `Resume Here` and `Near-Term Research Plan` first.

| Status | Candidate | Outcome / role | Keep / superseded reason |
| --- | --- | --- | --- |
| [x] | `S0_colab_raw_1024_r4` | Scored about `0.62`; best known public baseline. | Keep as leaderboard baseline and do-no-harm reference. |
| [x] | `S1_raw_full_r4` | Scored `0.54`; full-data raw-answer control. | Superseded as a main path because more raw final-answer SFT hurt public score. |
| [ ] | `S2_raw_full_r8` | Capacity-only raw full-data idea. | Not active; raw full-data direction is weak without procedural supervision. |
| [ ] | `S2_private_reasoning_boxed_r8` | Rank + boxed/private prompt idea. | Not active; combines too many axes and the related S4 path underperformed. |
| [ ] | `S3_boxed_full_r4_or_r8` | Isolate boxed target/output format. | Not active; boxed final-answer SFT has not improved reasoning enough. |
| [ ] | `S4_attention_expand_r8_private_boxed` | Expanded attention modules with boxed/private setup. | Effectively tested by S4 final/checkpoint-144; still below `0.62`. |
| [ ] | `S5_private_reasoning_boxed` | Prompt encourages internal reasoning but trains only final boxed answers. | Not active; procedural supervision is the stronger next direction. |
| [ ] | `S6_short_cot_synthetic` | Generic short rationales for selected categories. | Fold into solver-guided STaR only after trace quality checks. |
| [ ] | `S6_short_trace_boxed_r8_attention_drive` | Notebook exists at `notebooks/03_colab_short_trace_train_and_submit.ipynb`; cipher short traces for about `1,576 / 9,500` rows. | Useful source notebook, but do not run directly before `eval_v2` and trace audit. |
| [ ] | `S7_solver_guided_star_bootstrap` | Generate, verify, filter, and train on compact traces. | Active research direction, but the immediate first task is `eval_v2_grouped_family_balanced`. |
| [ ] | `S7_best_config_eval0_final` | Use all rows after a winning config is known. | Later polish only; not exploration. |

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
