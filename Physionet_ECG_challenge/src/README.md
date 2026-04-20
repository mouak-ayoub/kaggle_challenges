# classical_cv template

This package is a new template for refactoring notebook code.
The old code was moved to `src/to_do`.

## Why this shape

The package is organized by semantic stage:

- `io`
  - image loading and sample lookup
- `preprocessing`
  - resize, normalize, border masking, intensity transforms
- `features`
  - energy maps and feature images
- `contours`
  - contour initialization and active contour evolution
- `geometry`
  - coordinate mapping and polygon masks
- `fitting`
  - Hough lines and later geometric fitting
- `tuning`
  - reusable Optuna helpers for parameter search
- `pipelines`
  - thin end-to-end orchestration
- `core`
  - dataclass configs and result objects

This is a better fit than one file per notebook because:

- the same resize logic should be reused by many experiments
- the same energy builder should be reused by page-detection variants
- the same contour code should be reused when we tune the page snake
- the same geometry helpers should be reused when we map the contour back

## Style choice

This template uses:

- dataclass configs
- plain functions grouped by stage
- small result dataclasses

It avoids heavy OOP for now.

## Refactor direction

Later, notebook code should move into this package like this:

- notebook config cell
  -> dataclass config instances
- notebook helper functions
  -> stage modules
- notebook step-by-step cells
  -> thin notebook calls to package functions

## First targets

The first obvious refactor targets are:

- page energy image construction
- closed snake initialization and evolution
- coordinate mapping back to the original image
- polygon mask generation from the page contour
- Hough line detection and line-to-segment conversion
- Optuna study setup and page-scoring helpers for notebook search

## Current Hough Notes

- `src/fitting/hough_lines.py` now returns:
  - `peak_accumulator` as a sparse accumulator-shaped map
  - `peak_values` as the raw peak heights returned by `skimage`
  - `smoothed_accumulator` when Gaussian accumulator smoothing is enabled
- `StandardHoughConfig` now supports optional accumulator smoothing before peak picking:
  - `smooth_accumulator`
  - `accumulator_gaussian_sigma_rho`
  - `accumulator_gaussian_sigma_theta`
- `StandardHoughConfig` also now exposes peak-picking parameters instead of relying on `skimage` defaults:
  - `peak_threshold_ratio`
  - `min_distance`
  - `min_angle`
- `StandardHoughConfig` now also supports backend swapping with minimal pipeline changes:
  - `backend="skimage"` for the existing full-accumulator path
  - `backend="opencv"` for an experimental OpenCV-based sparse-accumulator path
  - `rho_resolution_pixels`
  - `opencv_use_edge_values`
- The OpenCV branch is designed to preserve the same downstream boundary-grid interface:
  - it converts OpenCV returned lines plus votes into an accumulator-shaped sparse map
  - this lets the existing threshold-family-extrema pipeline run without notebook-side rewrites
- In the notebook workflow, dominant-theta and perpendicular-theta family extraction are now treated as stable boundary-grid logic, separate from the temporary left-border debug section.
- In that stable notebook workflow, the main boundary-grid grouping now uses all accumulator bins above the effective threshold rather than the `n_peaks`-limited returned-peak list.
- That stable notebook workflow now also computes rho min/max envelopes from the selected dominant and perpendicular families, with a top-level notebook tolerance parameter and a plotting cell for the expanded boundary lines.
- The threshold-qualified Hough boundary-grid logic has now been extracted into reusable `src` code:
  - grouped config: `HoughBoundaryGridConfig`
  - pipeline: `run_hough_boundary_grid_detection`
  - reusable result types for threshold-qualified bins, line families, and the selected extreme boundary lines
- The unitary and batch Hough notebooks now both consume that shared boundary-grid selection path:
  - the unitary notebook keeps only notebook-specific debug and visualization code
  - the preview notebook uses the same method on random sample pairs for batch inspection
- The Hough notebook baseline parameters are now also centralized:
  - shared YAML file: `config/hough_notebooks.yaml`
  - loader: `src.core.load_hough_boundary_notebook_defaults`
  - this keeps the two notebooks synchronized on resize, edge-map, and Hough settings
  - the YAML stores the concrete notebook values directly instead of deriving them from `MAX_DIM`
  - notebook-only debug toggles do not belong in that YAML and remain local to the unitary notebook
- The notebook-side Hough debugging remains outside `src` for now because it is still exploratory plotting logic.
- The current external debug helper lives in:
  - `scripts/debug_hough_3d_window.py`
  - it now supports Plotly browser/HTML output in addition to Matplotlib
- If the Hough debug workflow stabilizes, the next reusable extraction candidates are:
  - peak-family selection helpers
  - accumulator-bin lookup helpers
  - configurable red-hypothesis generation near the left page border
