# Evidence Freshness Protocol

Use this before answering research-state or next-step questions.

## Problem

Codex can answer from committed notes while newer local, Colab, Drive, or uncommitted evidence exists. That creates stale recommendations.

## Rule

Strategy answers must be evidence-first, not document-first.

Use the freshest available evidence in this order:

1. explicit user message in the current conversation;
2. local uncommitted notebooks, scripts, or artifacts when available;
3. local generated outputs when available;
4. committed memory and decision logs;
5. older checklist, README, and method-ladder notes.

If fresher evidence contradicts committed docs, say so clearly and use the fresher evidence.

## Required Answer Gate

Before recommending a next step, identify:

- freshest evidence used;
- latest known experiment/result;
- current blocker;
- what should not be repeated;
- missing or uncertain evidence;
- recommended next diagnostic or experiment.

Do not recommend a run that the user says already ran unless proposing a deliberate rerun with changed conditions.

## Staleness Note

If only committed docs are visible, say that the answer may be stale if newer local or Colab runs exist.

If the user gives newer local evidence, use it and say committed docs may be stale.

## Promotion Rule

When a new local or user-reported result changes the next step, promote the concise lesson into existing project memory and decision logs. Do not create a new current-state file for every result.
