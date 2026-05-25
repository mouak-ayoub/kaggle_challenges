# Global Review Response

Date: 2026-05-17

Update 2026-05-25: `exp05_trace_occam_r4_inout` checkpoint 144 scored `0.59` publicly, slightly above checkpoint 96 at `0.58`. This is better than the S4 boxed path but still below the `00-raw-1024` public baseline at about `0.62`. Both exp05 checkpoints have the same local generated-eval aggregate (`134/256`).

## Summary

The external review is right about the main failure mode: the current 256-row generated eval and eval loss should not drive submissions. The strongest real baseline remains `00-raw-1024` at about `0.62` public, while `02-raw-full` and `04-s4` improved local diagnostics but hurt public score. S4 checkpoint-144 improved over the S4 final adapter (`0.55` vs `0.53`) but still did not approach the raw-answer baseline. The later exp05 trace run scored `0.59` at checkpoint 144, which supports trace supervision as useful but not sufficient.

The review is also right that the next useful direction is reasoning supervision, not more answer-only SFT. But the project goal is not to inspect or copy another active Nemotron challenge competitor's high-scoring notebook, GitHub repo, dataset, or submission path. Outside research is allowed as literature, tools, and inspiration, including papers, open-source code, Reddit/Medium/Substack posts, YouTube talks, other-domain competition ideas, and reasoning-challenge methods that are not from this exact Nemotron Kaggle challenge.

The chosen path is:

```text
Solver-Guided STaR for Hidden-Rule Reasoning
```

That means building our own loop that generates, verifies, filters, and trains on reasoning traces for each puzzle family.

Originality constraint: the submitted method should come from our own hypotheses, verifiers, data-generation loop, and notebook implementation. Outside research can shape critique and show what is possible, but direct artifacts from active competitors in this exact challenge should not define the trajectory unless explicitly approved.

## What The Current Evidence Supports

The results do not support these conclusions:

- more full-data answer-only SFT is better
- lower eval loss is enough
- larger LoRA rank or more modules is enough
- boxed output alone is enough
- the current 256-row generated eval is a reliable model selector

The results do support this conclusion:

```text
Teach the model how to infer the hidden rule, not only how to imitate final answers.
```

The full-data raw run and S4 likely taught local answer patterns and output discipline without improving hidden-rule generalization. The S4 checkpoint-144 result shows checkpoint timing can matter, but not enough to rescue the method. The `00-raw-1024` run probably preserved more of the base model's reasoning behavior by staying small and conservative.

## Why STaR Fits

The STaR paper is a good fit because this challenge has many prompts with final answers but no official rationales. STaR's basic loop is:

1. Generate rationales from a few seed rationale examples.
2. Keep rationales whose final answer is correct.
3. For failures, regenerate a rationale while showing the correct answer.
4. Fine-tune on rationales that ultimately lead to correct answers.
5. Repeat if the generated data improves.

For this competition, vanilla STaR is not enough. We need:

```text
STaR + family-specific verifiers + grouped eval + hard-family curriculum
```

The key risk is fake fluent reasoning. A trace that sounds plausible but does not follow the real rule can damage the model. Therefore every retained trace should be mechanically checked when possible, or at least pass strict family-specific structure checks.

## Baselines

Keep two baselines:

- Leaderboard baseline: `00-raw-1024`, public about `0.62`.
- Research baseline: a small solver-guided STaR seed run. Even matching `0.62` with trace training is useful because it proves the trace path does not destroy leaderboard behavior.

Do not use `04-s4-step-193` as the baseline just because it had the best current local generated eval. That proxy is currently misleading.

## Stage 1: Build `eval_v2`

Before trusting another serious run, build a better local validation setup:

```text
eval_v2:
  rows: 1000-2000 if runtime allows
  split: family-balanced
  holdout: grouped by prompt/rule pattern where feasible
  outputs:
    - overall exact extracted-answer accuracy
    - hard-family accuracy excluding numeral
    - accuracy by inferred family
    - extraction failures
    - max-token hits
    - raw completions
  adapters:
    - base model
    - 00-raw-1024
    - 02-raw-full
    - 04-s4
    - future STaR adapters
```

The grouped split matters because random rows may share near-identical templates. The hidden public score likely rewards generalization to new rule instances, not memorization of close templates.

## Stage 2: Seed Rationales

Write a small set of seed rationales before using model-generated traces:

```text
5-20 examples per hard family:
  - cipher
  - unit_conversion
  - bit_manipulation
  - equation
  - gravity after manual taxonomy
```

Keep the seed traces short and consistent. They do not need to be elegant. They need to be checkable and learnable.

## Stage 3: Family Verifiers

Start with verifiers that are easiest and highest signal.

### Cipher

Parse encrypted/plaintext examples, infer word or letter mappings, and verify query mappings. This directly targets errors such as getting the phrase skeleton right but choosing the wrong word, for example `cat explores book` instead of `cat imagines book`.

### Unit Conversion

Parse values and units, infer conversion factors from examples, and verify the final conversion. This family already showed local improvement, so it may be a fast win once traces are cleaner.

### Bit Manipulation

Build a small DSL search over operations such as rotations, shifts, XOR, AND, OR, NOT, constants, bit permutations, and shallow compositions. Keep this narrow at first; the verifier should be useful before it becomes clever.

### Equation

Use simple symbolic parsing and solving for linear, modular, or template-like equations. Do not rely on free-form model traces here without checking intermediate steps.

### Gravity

Inspect 50-100 rows before designing a solver. Gravity is currently the weakest family and probably contains several subtypes.

## Stage 4: STaR Loop

The first practical loop should be:

```text
seed rationales
-> generate candidate traces
-> extract final boxed answer
-> verify answer and trace structure
-> keep accepted traces
-> rationalize failures with the gold answer
-> verify rationalized traces more strictly
-> train LoRA with ordinary SFT
-> evaluate by family
-> repeat only if data quality improves
```

This is not an inference-time agent and not GRPO. It is bootstrapped supervised fine-tuning with correctness filters.

## First Experiments

### Experiment A: `exp05_star_seed_512`

Purpose: prove that reasoning traces do not damage the model.

```text
data:
  512 verified traces
  balanced across hard families if possible
  numeral excluded or heavily downweighted
LoRA:
  r=4 or r=8
  modules=in_proj,out_proj
  lr=1e-4
  max_seq=2048 if memory allows
  epochs=1
target:
  reasoning trace + final \boxed{answer}
```

Success condition:

```text
eval_v2 hard-family accuracy improves and public score is near or above 0.62.
```

### Experiment B: `exp06_star_cipher_unit_bit_1500`

Purpose: improve families where verification is most feasible.

```text
data:
  500 cipher traces
  500 unit conversion traces
  500 bit manipulation traces
LoRA:
  r=8
  modules=in_proj,out_proj first
  max_seq=2048 or 4096 if memory allows
  lr=1e-4
  epochs=1
```

Do not include gravity until we have reliable subtype analysis.

### Experiment C: `exp07_star_rationalized_hard`

Purpose: use STaR rationalization on failures.

```text
data:
  seed traces
  model-correct traces
  solver-verified rationalized traces
focus:
  equation and gravity only after verifiers exist
```

Success condition:

```text
hard families improve without damaging formatting or easier families.
```

## Guardrails

- Do not inspect or copy another active Nemotron challenge competitor's trajectory data as the main method.
- Do not inspect or copy another active Nemotron challenge competitor's high-scoring notebook or GitHub repo as the implementation path.
- Do inspect public trajectories to learn what short, learnable traces look like.
- Do not train on all `9,500` rows unless the generated supervision is high quality.
- Do not optimize for numeral; it is already locally solved.
- Do not let the 5-row probe choose checkpoints.
- Do not change rank, modules, sequence length, target format, and prompt style all at once.
- Preserve raw completions, extracted answers, accepted traces, rejected traces, and verifier failure reasons.

## Final Research Question

The question is not:

```text
Can we copy a public 0.72 run?
```

The question is:

```text
Can solver-guided self-training make Nemotron learn compact, verifiable procedures for hidden-rule reasoning?
```

That is the path that can improve the score and teach real LLM research skills.
