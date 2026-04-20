# Local Project Memory

> **Access Rules:**
> - **Copilot:** READ ONLY - read at session start, never modify
> - **Codex:** READ + WRITE - read at session start, update when milestones reached

This file is a local working memory for AI agents.
It is not the project source of truth.

Related files:
- `README.md` for architecture and pipeline direction
- `PROJECT_DECISION_LOG.md` for durable technical decisions
- `AGENTS.md` for project-specific workflow rules

## Goal

Ultimate goal:

`any ECG image -> canonical clean page -> lead signals -> CSV`

This includes hard cases:
- mobile photos on a table
- laptop-screen photos
- stained or damaged pages
- different resolutions and alignments

## Current Stage

- We are still in the classical computer vision learning phase.
- We apply each new method we study in a focused notebook.
- Right now we are moving from pure ink removal toward page and structure detection.

## What We Confirmed

- In the synthetic ink-removal setup, `blackhat_only` is the strongest classical baseline.
- Wiener and histogram methods can help the corrupted region, but they often hurt the full image.
- Laplacian pyramid is useful and instructive, but still weaker than `blackhat_only`.
- Derivative filters are useful as feature maps, not as direct restoration methods.
- Active contour is better positioned for **page boundary detection** than for direct waveform extraction.

## Best Practices Learned

- Separate the problems:
  - boundary detection / alignment
  - ink removal
  - lead localization
  - waveform extraction
- When studying ink removal alone, use synthetic corruption on a clean image.
- When comparing methods, use the same ECG image when possible.
- Keep notebooks step by step.
- Use short markdown and short comments.
- Prefer framework implementations from `scikit-image`, `opencv`, or similar libraries.
- Write conclusions from actual saved outputs, not from expectation.
- Keep a summary table when several methods are compared.

## Current Notebook Organization

- `notebooks/ink_removal`
  - focused classical cleaning experiments
- `notebooks/boundary_detection`
  - page detection and structure detection before rectification
  - active-contour page detection before Hough-based refinement
  - Hough-line notebook on the same hard image for straight-line analysis

## Current Src Template

- New reusable classical CV code lives directly in `src`
- Older project code was moved to `src/to_do`
- The package is organized by semantic stage:
  - `io`
  - `preprocessing`
  - `features`
  - `contours`
  - `geometry`
  - `fitting`
  - `tuning`
  - `pipelines`
  - `core`
- We prefer:
  - small reusable functions
  - dataclass configs
  - thin pipeline functions
- We do not mix notebook-specific plotting with reusable src code

## What We Are Doing Now

- First active-contour notebook:
  - detect the ECG page boundary in a hard image
  - produce a page mask and a contour mapped back to the original image
- This is a preparation step before Hough-based line detection and page rectification.
- We are now preparing to refactor notebook code into `src`
- The current active-contour work uses the circular page contour notebook as the main experiment
- The boundary-detection notebook now reuses methods from `src` instead of keeping duplicate helper code inside the notebook
- The Optuna notebook is being refactored so the study boilerplate and page-scoring helpers can be reused in later scenarios

## Latest Milestone

- In `notebooks/boundary_detection/ecg_hough_lines_page_detection.ipynb`, Hough debugging is now organized behind one notebook toggle:
  - `DEBUG_HOUGH_LEFT_LINE`
- The left-border Hough debug flow now supports:
  - configurable green peak families
  - configurable red hypotheses
  - optional blue top-accumulator peaks
  - `is_peak=True/False` in printed debug lines
  - synchronized 2D and 3D debug overlays
- The standard Hough result now keeps:
  - `peak_accumulator` as an accumulator-shaped sparse peak map
  - `peak_values` as the flat vote values returned by `skimage`
  - `smoothed_accumulator` as an optional Gaussian-smoothed accumulator used for peak detection when enabled
- `StandardHoughConfig` now supports optional accumulator smoothing in reusable `src` code:
  - `smooth_accumulator`
  - `accumulator_gaussian_sigma_rho`
  - `accumulator_gaussian_sigma_theta`
- `StandardHoughConfig` also now supports explicit peak-picking control:
  - `peak_threshold_ratio`
  - `min_distance`
  - `min_angle`
- Current default peak threshold rule:
  - `threshold = 0.40 * max(accumulator_used_for_peak_picking)`
  - this replaces the old hidden `skimage` default of `0.50 * max(...)`
- The boundary-detection Hough notebook now defaults to smoothed peak picking:
  - `smooth_accumulator=True`
  - `accumulator_gaussian_sigma_rho=2.0`
  - `accumulator_gaussian_sigma_theta=0.8`
  - chosen as a conservative default to soften thorny local maxima without aggressively merging nearby angle families
- We confirmed an important interpretation point:
  - a visually plausible left border can exist in the accumulator without being returned by `hough_line_peaks`
  - this is often due to local peak competition / non-maximum suppression, not because the line is absent
- We added an external interactive helper:
  - `scripts/debug_hough_3d_window.py`
  - this now supports both:
    - Plotly interactive 3D in the browser or HTML
    - Matplotlib external window when a GUI backend is available
- The Hough notebook now separates stable boundary-grid logic from temporary debugging:
  - dominant-theta and perpendicular-family extraction now live in the main workflow section
  - their notebook parameters are now stable top-level config values:
    - `BOUNDARY_PRIMARY_THETA_TOLERANCE_DEG`
    - `BOUNDARY_PERPENDICULAR_THETA_TOLERANCE_DEG`
  - `DEBUG_CFG` is reserved for left-border red-line experiments, 3D inspection, and rho/theta slice views
- The stable boundary-grid workflow in the Hough notebook now uses:
  - all accumulator bins above the effective Hough threshold
  - not the `skimage` returned-peak list capped by `n_peaks`
  - this applies to:
    - the theta concentration summary
    - dominant/perpendicular family extraction
- That stable boundary-grid workflow now also estimates rho envelopes from the selected families:
  - `BOUNDARY_RHO_BOUND_TOLERANCE` is a top-level notebook config value
  - a dedicated notebook cell computes min/max rho only from the selected dominant and perpendicular families
  - the cell also plots the tolerance-expanded rho boundary lines back on the ECG image
- The threshold-qualified Hough boundary-grid method has now been extracted into reusable `src` code:
  - grouped config: `HoughBoundaryGridConfig`
  - reusable pipeline: `run_hough_boundary_grid_detection`
  - structured outputs for threshold bins, line families, and selected extreme boundary lines
- A new compact preview notebook now uses that shared method:
  - `notebooks/boundary_detection/ecg_hough_boundary_grid_preview.ipynb`
  - it shows 10 random sample pairs with the source image on the left and the selected boundary lines on the right
- The unitary Hough notebook now also reuses the shared boundary-grid method for line selection:
  - `notebooks/boundary_detection/ecg_hough_lines_page_detection.ipynb`
  - threshold bins, dominant/perpendicular families, and selected extreme lines now come from `run_hough_boundary_grid_detection`
  - the notebook keeps only the extra visualization and debug logic locally
- The two Hough notebooks are now aligned on the same main method parameters:
  - energy-builder config
  - standard Hough config
  - dominant/perpendicular family tolerances
- The Hough notebook defaults are now centralized in reusable code:
  - shared YAML file: `config/hough_notebooks.yaml`
  - loader: `src.core.load_hough_boundary_notebook_defaults`
  - both Hough notebooks load the same baseline from that YAML-backed helper
  - this removes config drift as a cause when the same image is compared across the unitary and batch notebooks
  - the YAML now stores direct concrete values, not notebook formulas derived from `MAX_DIM`
  - debug-only toggles are intentionally excluded from YAML and remain unitary-notebook-only
- The standard Hough path is now backend-switchable through shared config:
  - `StandardHoughConfig.backend` supports `skimage` and `opencv`
  - `config/hough_notebooks.yaml` now contains:
    - `shared_baseline` using `skimage`
    - `opencv_weighted_experimental` using OpenCV weighted voting
  - both Hough notebooks now select the shared profile through one variable:
    - `HOUGH_NOTEBOOK_PROFILE`
  - OpenCV weighted Hough is wired with minimal surface change:
    - `backend`
    - `rho_resolution_pixels`
    - `opencv_use_edge_values`
  - the OpenCV branch currently exposes a sparse accumulator of local maxima plus votes, so the existing boundary-grid pipeline can still run without notebook-side method duplication
- Project documentation is now split by purpose:
  - `AGENTS.md` for workflow rules
  - `LOCAL_PROJECT_MEMORY.md` for current state
  - `PROJECT_DECISION_LOG.md` for durable project decisions

## Next Ideas

- Compare active contour on `0005`, `0006`, and `0010`
- After the Hough lecture:
  - compare snake vs Hough for page detection
  - use Hough for straight edge and corner refinement
- Later:
  - lead layout detection
  - waveform extraction from each lead
  - CSV generation
