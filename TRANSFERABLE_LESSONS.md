# Transferable Lessons

This file collects reusable engineering and research lessons across challenges.

It is not a project status file.
It is not a project architecture file.
It is not an agent instruction file.

Use it for lessons that should transfer across repositories.

## How To Use This File

Put here:
- method lessons that generalize beyond one project
- debugging patterns that proved useful in practice
- negative results that should prevent repeated wasted effort
- refactoring lessons about notebooks, pipelines, and reusable code

Do not put here:
- current project status
- project-only parameter values
- one-off notes that are specific to a single notebook run

## Computer Vision

### Hough peaks are often diagnostic, not sufficient

A strong Hough peak view does not automatically give a robust boundary extraction method.

What failed as a final rule:
- relying only on the returned peak list
- assuming the strongest peaks correspond to the true outer structure

What helped:
- keep all accumulator bins above threshold
- group lines by dominant orientation
- explicitly build the perpendicular family
- select geometric extremes within each family

Transferable idea:
For structured rectangular or grid-like objects, the decisive step may be geometric grouping plus extremal selection, not peak ranking alone.

### Negative results should be preserved, not discarded

An idea can be useful for understanding the data and still be insufficient for the target task.

Example pattern:
- a method improves visualization
- a method reveals useful structure
- but it still does not produce the required stable output

This should be recorded explicitly as a negative result with value, not treated as wasted work.

### Boundary detection deserves its own reasoning layer

When downstream quality is poor, do not assume the fix belongs only in the final model.

If the input geometry is unstable, boundary detection and canonicalization may need to become explicit stages rather than minor preprocessing.

## Notebook And Refactoring Practice

### Separate stable method logic from debug logic

Keep these separate:
- stable method logic that should move into reusable code
- temporary diagnostics used to test hypotheses

A good split is:
- `src/`: stable method and reusable utilities
- notebook: plots, visual checks, temporary probes, red-line hypotheses, 3D debug views

### Keep a unitary notebook and a batch notebook

For method development, two notebook roles are useful:
- unitary notebook: one hard case, deep inspection
- batch notebook: many samples, fast visual validation

This avoids mixing exploratory depth with batch evaluation.

### Documentation should be split by role

A useful structure is:
- `AGENTS.md`: how an agent should work in the project
- `LOCAL_PROJECT_MEMORY.md`: current state and recent milestones
- `PROJECT_DECISION_LOG.md`: durable project-specific technical decisions
- `TRANSFERABLE_LESSONS.md`: reusable lessons across projects

This prevents one file from becoming an unclear mix of instructions, memory, and research conclusions.

## ML Pipeline Perspective

### Stage decomposition is often a modeling decision, not just an implementation detail

If a system fails on hard domains, the right response is sometimes to split the problem into stages.

Typical example:
- Stage A: normalize geometry or appearance
- Stage B: perform the final extraction or prediction

This is especially useful when domain shift is driven by acquisition conditions rather than label semantics.
