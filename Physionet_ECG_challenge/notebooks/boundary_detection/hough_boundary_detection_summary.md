# Hough Boundary Detection Summary

These tables are written to be easy to read later.  
For selector comparison, read first the rows built on the same image and the same family.

## Notebook Roles

| Notebook | Role | Current use |
|---|---|---|
| `ecg_hough_lines_page_detection.ipynb` | unitary notebook | one image, deep inspection, selector comparison, debug plots |
| `ecg_hough_boundary_grid_preview.ipynb` | batch notebook | compact visual check on the random sample set |
| `ecg_active_contour_page_detection.ipynb` | earlier boundary study | contour-based page detection before the Hough selector work |

## Selector Comparison On Known Regression Cases

| Selector | Image | Family | Observed result |
|---|---|---|---|
| `global` | `19030958 / 0009` | perpendicular | misses the far-right border, keeps `P_max = 518` |
| `score` | `19030958 / 0009` | perpendicular | recovers the far-right border, keeps `P_max = 965` |
| `hybrid` | `19030958 / 0009` | perpendicular | keeps the `score` family because its separation is larger |
| `global` | `10140238 / 0012` | perpendicular | keeps the outer border, `P_max = 994` |
| `score` | `10140238 / 0012` | perpendicular | prefers a stronger inner line, `P_max = 876` |
| `hybrid` | `10140238 / 0012` | perpendicular | keeps the `global` family because its separation is larger |

## Current Shared Boundary Method

| Stage | Current choice |
|---|---|
| theta-family discovery | global threshold on the Hough accumulator |
| rho selection | selector-dependent: `global`, `score`, or `hybrid` |
| perpendicular-family anchor | current shared baseline uses `dominant_family_reference_theta` |
| best working baseline | `hybrid_global_score_separation` |
| shared config source | `config/hough_notebooks.yaml` |
| shared reusable pipeline | `src.pipelines.run_hough_boundary_grid_detection` |
| shared pre-Hough ink removal | `src.preprocessing.apply_ink_removal` |
| current ink-removal method | `blackhat_inv` from `ecg_morphology_background_replacement.ipynb` |

## Current Stop Point

| Topic | Current status |
|---|---|
| fixed random-set qualitative result | about `9 / 10` acceptable boundary detections |
| best selector to keep for now | `hybrid` |
| main remaining failure | monitor / bezel double-edge ambiguity |
| practical interpretation | selector is usually right at family level, but can still choose the wrong member of a close parallel pair |

## Short Summary

`global` is strong when the true border already survives the threshold gate.  
`score` is useful when the correct theta family is known but the true border is weaker than a nearby inner line under the global threshold rule.  
`hybrid` is the best place to stop for now because it keeps the better family between `global` and `score` using separation only.  
The main remaining issue is the monitor double-edge case, where local line disambiguation is still needed after family selection.
