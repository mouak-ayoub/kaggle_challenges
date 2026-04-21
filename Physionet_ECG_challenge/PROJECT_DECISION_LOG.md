# Project Decision Log

This file records durable project-specific technical decisions.

It is not the same as:
- `LOCAL_PROJECT_MEMORY.md`: current state and recent milestones
- `AGENTS.md`: workflow rules for agents
- `README.md`: broader architecture and project direction

Use this file when a method decision is stable enough that future sessions should understand:
- what was decided
- why it was decided
- what it replaced

## Decision: Two-stage thinking remains necessary for hard ECG domains

Date: 2026-04
Status: Active

### Context

Earlier project work framed the pipeline as:
- Stage A: make the input closer to a canonical clean ECG page
- Stage B: segment leads and extract signals

At that stage, the aggregate signal quality was still not good enough on hard domains, and naive geometric cleanup alone was not sufficient.

### Decision

Keep the two-stage perspective:
- upstream page normalization and structure recovery are separate from downstream lead extraction
- boundary detection is treated as a first-class problem, not as a minor preprocessing detail

### Evidence

- Hard domains such as mobile photos, screen photos, stains, and damage are not solved by a single downstream segmentation step.
- Earlier Hough use improved visibility of structure, but did not by itself produce a stable boundary extraction method.

### Consequence

- Geometry and page-structure work remain strategic, not incidental.
- Boundary detection decisions should be evaluated by whether they produce a stable canonical page for later stages.

## Decision: Raw Hough peaks alone are not sufficient for ECG boundary selection

Date: 2026-04
Status: Superseded

### Context

Initial Hough analysis focused on returned peaks from `hough_line_peaks`, plus visual inspection of strong accumulator maxima.

### Decision

Do not treat the returned Hough peak list alone as the boundary selection method.

### Evidence

- Real boundary candidates could exist in the accumulator but not appear in the returned peaks because of local peak competition or non-maximum suppression.
- Peak ranking alone did not isolate the true ECG grid limits reliably.

### Consequence

- Hough peaks remain useful for diagnosis and visualization.
- They are not the final rule for selecting ECG boundary lines.

## Decision: Boundary selection should use threshold-qualified bins, orientation families, and family extremes

Date: 2026-04
Status: Active

### Context

The decisive improvement came when the notebook stopped depending on the `n_peaks`-capped `skimage` output and instead reasoned geometrically over the threshold-qualified Hough space.

### Decision

Use this boundary-grid selection method:
1. take all accumulator bins above the effective threshold
2. find the dominant theta family
3. find the perpendicular family
4. select extreme lines within each family using a common projected-rho reference

### Evidence

- This produced visually correct limited ECG grid boundaries in the unitary notebook.
- The perpendicular-family step was simple but decisive because it converted Hough from a peak detector into a structured geometric grouping method.
- The resulting logic was stable enough to extract into reusable `src` code.

### Consequence

- The shared method now lives in `src.pipelines.run_hough_boundary_grid_detection`.
- The unitary notebook keeps debug visualization, but the stable line-selection method should not be reimplemented there.
- Batch evaluation should reuse this same method unchanged.

## Decision: Theta-family discovery and rho-boundary selection should be separated

Date: 2026-04
Status: Active

### Context

The threshold-qualified family-extrema method improved line selection a lot, but a clear regression case remained:
- the dominant/perpendicular theta families were correctly detected
- a plausible border line still existed near the correct theta family
- that border line was discarded only because its accumulator was just below the global threshold

### Decision

Keep the global threshold for theta-family discovery, but do not use that same threshold as the final gate for rho-boundary selection.

The new shared selector:
1. uses threshold-qualified bins to estimate dominant and perpendicular theta families
2. builds a local-max rho candidate set inside each family
3. scores candidate pairs using:
   - pair accumulator strength
   - pair separation in projected rho

### Evidence

- The failing right-border example had the correct perpendicular family but lost the real border line before extrema selection because:
  - the line accumulator was slightly below the global threshold
  - the angle family itself was already correct
- Moving final rho selection to local-max candidate pairs recovered that case without changing theta-family discovery.

### Consequence

- `HoughBoundaryGridConfig` now exposes a configurable line-selection strategy.
- The old `global_threshold_extrema` method remains available as a fallback / comparison baseline.
- The shared YAML baseline now uses the new `theta_guided_rho_pair_score` selector.

## Decision: Debug logic stays notebook-local until it stabilizes

Date: 2026-04
Status: Active

### Context

The left-border Hough work introduced red hypotheses, 3D accumulator plots, slice plots, and accumulator probing. These are valuable for diagnosis but still exploratory.

### Decision

Keep temporary Hough debug logic in notebooks and helper scripts until the debug workflow itself becomes stable and reusable.

### Evidence

- The debug cells changed repeatedly during investigation.
- Most of that logic supports hypothesis testing rather than the production method.

### Consequence

- `src/` contains the stable boundary-grid method.
- red-line experiments, 3D inspection, and similar diagnostics stay notebook-side for now.
