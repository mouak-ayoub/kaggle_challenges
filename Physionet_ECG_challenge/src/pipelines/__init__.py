"""Thin pipelines that combine reusable classical CV stages."""

from .page_detection import (
    build_closed_page_detection_config,
    run_closed_page_detection,
    score_closed_page_detection,
)

__all__ = [
    "build_closed_page_detection_config",
    "run_closed_page_detection",
    "score_closed_page_detection",
]
