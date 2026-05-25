# Nemotron Kaggle Challenge - Round 2 Status Review Brief

Date prepared: 2026-05-25

## Request To Reviewer

Please review this as an independent technical critic. This is the second status update after the first external review.

We want feedback on:

1. Whether our interpretation of the latest trace-training result is sound.
2. Whether the next effort should focus on cipher, bit manipulation, equation, a broader `eval_v2`, or a different route.
3. Whether our proposed cipher trace format is likely to teach the model the actual operation or merely teach a nicer-looking explanation.
4. What minimal next experiment has the best chance to beat the current best public score of about `0.62`.
5. What controls or diagnostics are missing before spending another serious Colab/Kaggle submission.

Important: do not assume local generated-eval accuracy is a leaderboard proxy. We already saw this metric diverge from public score.

## Current Bottom Line

Best public score remains:

```text
00-raw-1024: about 0.62
```

Latest serious method:

```text
exp05_trace_occam_r4_inout checkpoint 144: public score 0.59
```

Checkpoint 96 from the same run scored 0.58. Checkpoint 144's local generated-eval score is identical to checkpoint 96:

```text
134/256 = 0.5234375
```

Checkpoint 144 changed only two generated-eval rows relative to checkpoint 96:

- gained one numeral row
- lost one bit-manipulation row

The hidden/public score did move slightly in checkpoint 144's favor: 0.59 versus checkpoint 96's 0.58.

## Challenge And Submission Context

Competition: NVIDIA Nemotron Model Reasoning Challenge on Kaggle.

Submission format:

- Kaggle expects a LoRA adapter packaged as `submission.zip`.
- Required files at zip root:
  - `adapter_config.json`
  - `adapter_model.safetensors`
- This is not a prediction CSV competition.
- Local generation against `test.csv` is only a sanity check; leaderboard scoring loads/evaluates the adapter.

Local data:

- `train.csv`: 9,500 rows
- `test.csv`: 3 public sanity rows
- `trace_training.csv`: 9,497 rows after excluding the 3 public sanity IDs

Inferred families:

- `bit_manipulation`
- `cipher`
- `equation`
- `gravity`
- `numeral`
- `unit_conversion`

## Public Score History

| Label | Public score | Summary |
| --- | ---: | --- |
| `00-raw-1024` | about `0.62` | Best known score. Raw-answer SFT on 1,024 rows, LoRA r4, `in_proj/out_proj`, direct final-answer prompt. |
| `02-raw-full` | `0.54` | Full-data raw-answer control. More raw answer-only data hurt public score. |
| `04-s4-final` | `0.53` | Boxed/private prompt, LoRA r8, expanded attention targets. |
| `04-s4-step-144` | `0.55` | Same S4 run, selected earlier checkpoint. Checkpoint timing helped slightly. |
| `05-trace-step-96` | `0.58` | Exp05 trace Occam checkpoint 96. Trace/boxed targets, completion-only loss, LoRA r4, `in_proj/out_proj`, LR `1e-4`. |
| `05-trace-step-144` | `0.59` | Same exp05 run, later checkpoint. Local generated eval unchanged from step 96. |

Interpretation so far:

- Trace supervision helped versus full-data raw and S4 boxed variants.
- Trace supervision still did not beat the smaller raw `0.62` baseline.
- More training steps lowered loss and improved public score slightly, but did not improve local generated-eval aggregate.

## Latest Exp05 Trace Occam Setup

Run:

```text
exp05_trace_occam_r4_inout
```

Core config:

```text
base model: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
target format: trace or boxed answer from trace_training.csv
loss: completion-only
train rows: 9,241 after fixed eval split
eval rows: 256
max_seq_length: 512
max_new_tokens: 384
LoRA r: 4
LoRA alpha: 32
LoRA dropout: 0.05
LoRA modules: in_proj, out_proj
learning_rate: 1e-4
generation: fused Mamba path with use_cache=False
```

Trace data summary:

| Family | Supervision status |
| --- | --- |
| cipher | deterministic character-alignment traces, but current model still fails cipher locally |
| gravity | formula traces for all gravity rows |
| unit_conversion | mixed short/full/boxed traces |
| numeral | mostly boxed-only, small short-trace slice |
| bit_manipulation | verifier-gated DSL traces for a subset; unverified rows boxed-only |
| equation | boxed-only after simple char-substitution verifier had zero coverage |

## Local Generated Eval

Same fixed 256-row generated-eval split, scored by extracted final answer.

| Run/checkpoint | Matches | Rows | Accuracy |
| --- | ---: | ---: | ---: |
| `00-raw-1024` | 65 | 256 | 0.25390625 |
| `02-raw-full` | 87 | 256 | 0.33984375 |
| `04-s4-step-144` | 90 | 256 | 0.3515625 |
| `04-s4-final/current-193` | 95 | 256 | 0.37109375 |
| `05-trace-step-96` | 134 | 256 | 0.5234375 |
| `05-trace-step-144` | 134 | 256 | 0.5234375 |

Important caveat:

The raw `0.62` baseline has poor local generated eval. Therefore local generated eval is diagnostic only, not a public-score proxy.

## Exp05 Family Breakdown

| Family | `05-trace-step-96` | `05-trace-step-144` | Comment |
| --- | ---: | ---: | --- |
| bit_manipulation | 3/44 = 0.0682 | 2/44 = 0.0455 | Still mostly broken. |
| cipher | 0/40 = 0.0000 | 0/40 = 0.0000 | Completely broken locally despite trace format. |
| equation_symbolic | 2/39 = 0.0513 | 2/39 = 0.0513 | Mostly broken; equation currently boxed-only. |
| gravity | 41/44 = 0.9318 | 41/44 = 0.9318 | Strong local improvement. |
| numeral | 45/46 = 0.9783 | 46/46 = 1.0000 | Saturated/easy. |
| unit_conversion | 43/43 = 1.0000 | 43/43 = 1.0000 | Saturated/easy. |

This suggests the trace run mainly learned unit conversion, gravity, and numerals. It did not solve the hardest families.

## Public Sanity Cipher Case

Public sanity row:

```text
id: 00189f6a
family: cipher
expected answer: cat imagines book
```

Prompt:

```text
In Alice's Wonderland, secret encryption rules are used on text. Here are some examples:
ucoov pwgtfyoqg vorq yrjjoe -> queen discovers near valley
pqrsfv pqorzg wvgwpo trgbjo -> dragon dreams inside castle
gbcpovb tqorbog bxo zrswtrj pffq -> student creates the magical door
bxo sfjpov pqrsfv dfjjfig -> the golden dragon follows
nqwvtogg qorpg bxo zegboqwfcg gotqob -> princess reads the mysterious secret
Now, decrypt the following text: trb wzrswvog hffk
```

Saved answers:

| Run | Answer | Raw trace available? |
| --- | --- | --- |
| `00-raw-1024` | `queen creates the secret door` | No, only extracted/displayed answer was archived. |
| `05-trace-step-96` | `ivs daginger tool` | Yes. |
| `05-trace-step-144` | `ils immortal muddy` | Yes. |
| Correct target | `cat imagines book` | Template generated locally. |

### Raw trace from `05-trace-step-96`

```text
Thinking:
Category: cipher. Use same-length encrypted/plain word pairs.

Needed mappings from aligned examples:
t->i, r->v, b->s, w->d, z->a, s->g, v->n, o->e, g->r, f->o, k->l.

Apply to target:
trb -> i v s -> ivs.
wzrswvog -> d a g i n g e r -> daginger.
hffk -> ? o o l from known mappings.
The examples do not determine h directly; phrase context completes ?ool as tool.

Final answer:
\boxed{ivs daginger tool}
```

### Raw trace from `05-trace-step-144`

```text
Thinking:
Category: cipher. Use same-length encrypted/plain word pairs.

Needed mappings from aligned examples:
t->i, r->l, b->s, w->i, z->m, s->o, v->n, o->e, g->r, f->d, k->y.

Apply to target:
trb -> i l s -> ils.
wzrswvog -> i m l o n e r i -> immortal.
hffk -> ? d d y from known mappings.
The examples do not determine h directly; phrase context completes ?ddy as muddy.

Final answer:
\boxed{ils immortal muddy}
```

### Correct template-generated trace

This is the current proposed target style for cipher supervision:

```text
Thinking:
Category: cipher. Align same-length encrypted/plain word pairs and use the character map.

Mappings needed for the target:
trgbjo -> castle gives t->c, r->a, b->t.
zrswtrj -> magical gives z->m, s->g, w->i.
wvgwpo -> inside gives w->i, v->n, g->s, o->e.
pffq -> door gives f->o.

Apply to target:
trb -> c a t -> cat.
wzrswvog -> i m a g i n e s -> imagines.
hffk -> ? o o ?.

The examples do not determine h or k directly. Phrase context completes ?oo? as book.

Final answer:
\boxed{cat imagines book}
```

Diagnosis:

- The trace model learned the trace shape.
- It still invents or misaligns character mappings.
- The next cipher effort should force every displayed mapping to cite an aligned source word pair.
- The risk is that the model learns to write a better-looking trace without improving hidden test decryption.

Additional local file with this case:

```text
data/analysis/cipher_public_test_00189f6a_trace_comparison.md
```

## Current Working Hypothesis

If we fix cipher and do not damage the strong families, beating `0.62` may be plausible.

But this is not guaranteed because:

- public hidden distribution may not weight cipher enough
- local generated eval is not a reliable public-score proxy
- trace style improvements may not transfer to hidden ciphers
- more trace training can damage the base behavior that gave `00-raw-1024` its stronger public score

## Candidate Next Experiments

### Option A: `exp06_cipher_trace_weighted_r4`

Purpose: fix cipher without changing many other knobs.

```text
base: exp05 trace setup
LoRA: r4, in_proj/out_proj
LR: 1e-4
max_seq_length: 512 or 1024
target: compact cipher mapping traces
sampling: oversample cipher, downweight numeral/unit
eval: cipher-heavy eval plus normal 256-row eval
```

Why:

- Small change from exp05.
- Directly targets a family with 0/40 local accuracy.
- Keeps LoRA capacity and target modules constant.

Risks:

- Could overfit public/train cipher style.
- Could damage gravity/unit behavior.
- Could improve raw trace appearance without improving answer correctness.

### Option B: Build `eval_v2_grouped_family_balanced` before any new training

Purpose: avoid optimizing against misleading diagnostics.

```text
rows: 1000-2000 if runtime allows
split: family-balanced
grouping: by prompt/rule pattern where possible
metrics:
  - overall accuracy
  - hard-family accuracy excluding numeral
  - family accuracy
  - extraction failures
  - max-token hits
  - raw completions
```

Why:

- The current 256-row split is not trustworthy as a leaderboard proxy.
- A grouped hard-family eval may help compare methods before Kaggle upload.

Risks:

- It costs time and may still fail to predict public score.
- It delays the next submission.

### Option C: Cipher synthetic data plus verifier

Purpose: teach the general operation with more clean cipher examples.

```text
generate synthetic Alice-style cipher prompts
ensure deterministic mappings and target answers
train compact target-side traces
evaluate on held-out synthetic and real train ciphers
```

Why:

- Current train has only about 1,576 cipher rows.
- More clean alignment examples may teach the operation better.

Risks:

- Synthetic distribution mismatch.
- Hidden set may not match the synthetic generation process.

### Option D: Hard-family curriculum

Purpose: focus on cipher, bit manipulation, and equation together.

```text
exclude/downweight numeral
downweight unit/gravity after preserving some rehearsal
train only verified traces or boxed answers for hard families
```

Why:

- These are the families currently limiting the model.

Risks:

- Changes too much at once.
- Could lose the easy-family gains that exp05 achieved.

## Specific Questions For Reviewer

1. Is it better to run a cipher-focused experiment now, or build `eval_v2` first?
2. Is the proposed cipher trace template too verbose, too sparse, or about right?
3. Should the cipher trace include the full mapping table, only target-needed mappings, or both?
4. Should phrase-context completion for unseen letters be trained explicitly, or should we avoid it because it encourages hallucination?
5. Should we preserve exp05 settings exactly for the next run, or combine the cipher data fix with longer sequence length or expanded LoRA targets?
6. Is checkpoint 96's `0.58` evidence enough to keep trace supervision as the research baseline, or should we return to the smaller raw `0.62` style?
7. What local diagnostic would make a cipher improvement more trustworthy before submitting?
8. If you could spend one more serious Kaggle submission, what exact run would you choose?

## Artifacts To Attach If Useful

Main files:

```text
doc/external_review_pack/ROUND2_STATUS_REVIEW_BRIEF.md
data/outputs/reports/experiment_dashboard.html
data/analysis/cipher_public_test_00189f6a_trace_comparison.md
```

Historical context:

```text
doc/external_review_pack/CHATGPT_PRO_REVIEW_BRIEF.md
doc/external_review_pack/GLOBAL_REVIEW_RESPONSE.md
doc/SUBMISSION_TRACKING.md
doc/EXPERIMENT_CHECKLIST.md
```
