# Precedent-First Editing

This file addresses a recurring failure mode: changing a notebook or workflow by inventing a new pattern when a working local pattern already exists.

The goal is not to add one rule for every missed detail. The goal is to make Codex reuse local precedents and preserve workflow invariants.

## Trigger

Use this before editing:

- Colab notebooks;
- data builders;
- training, evaluation, checkpoint, and submission flows;
- artifact writing, archive, resume, or recovery logic;
- any new notebook or script based on an existing workflow.

## Step 1: Find The Precedent

Before editing, inspect the nearest working predecessor:

- previous notebook with the same role;
- script that already builds the same artifact;
- earlier Colab notebook that solved the same runtime concern;
- protocol doc that defines the expected output or archive contract.

If no local precedent exists, say so explicitly.

## Step 2: Same-Shape Default

Default to the same operational shape as the precedent:

- same cell ordering where practical;
- same source-of-truth config pattern;
- same path and artifact naming convention;
- same save/archive/resume behavior;
- same validation style;
- same separation between diagnostics and submission artifacts.

Do not introduce broad fallback machinery or hidden behavior just because it seems more general. Generality is allowed only when the current precedent is known to fail or the user asked for a redesign.

## Step 3: Delta Budget

Before or during the edit, state:

1. precedent used;
2. intended change;
3. invariants preserved;
4. intentional deviations and why they are safer or better.

If the deviation list grows, pause and convert the task into an explicit redesign proposal.

## Step 4: Preserve Generic Invariants

These are invariant categories, not one-off rules:

- configuration is centralized and easy to change;
- durable outputs are not silently written only to temporary local storage;
- raw generated outputs are saved before filtering or extraction;
- output paths derive from the experiment name or documented source of truth;
- run bundles, diagnostics, and Kaggle submission zips stay separate;
- existing resume/recovery behavior is not weakened;
- existing archive and tracking expectations are not weakened;
- notebook JSON remains valid;
- validation commands are run or a reason is given.

## Step 5: Post-Edit Self-Review

End notebook/workflow edits with:

- precedent used;
- same-shape behavior preserved;
- intentional deviations;
- validation run;
- remaining risk.

## Innovation Rule

Codex should still propose better methods. The restriction is on silent infrastructure reinvention, not research creativity.

If Codex sees a better workflow design, propose it as a `new proposal`, explain why it beats the precedent, and suggest a small diagnostic before replacing the working pattern.
