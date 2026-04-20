"""Thin pipelines that combine reusable classical CV stages."""

from .page_detection import (
    build_closed_page_detection_config,
    run_closed_page_detection,
    score_closed_page_detection,
)
from .hough_boundary_grid import run_hough_boundary_grid_detection

__all__ = [
    "build_closed_page_detection_config",
    "run_hough_boundary_grid_detection",
    "run_closed_page_detection",
    "score_closed_page_detection",
]
