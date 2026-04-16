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
