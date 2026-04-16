from dataclasses import dataclass, field

import numpy as np


@dataclass
class EnergyBuildResult:
    """Intermediate images created while building the snake energy."""

    resized_gray: np.ndarray
    smoothed_gray: np.ndarray
    enhanced_gray: np.ndarray
    raw_energy: np.ndarray
    final_energy: np.ndarray
    resize_scale: float


@dataclass
class SnakeEvolutionResult:
    """Initial, final, and optional debug states of the contour."""

    initial_snake: np.ndarray
    final_snake: np.ndarray
    debug_snakes: list[np.ndarray] = field(default_factory=list)
    debug_iteration_counts: list[int] = field(default_factory=list)


@dataclass
class CoordinateMappingResult:
    """Contour mapped from resized space back to original space."""

    mapped_snake: np.ndarray
    scale_y: float
    scale_x: float


@dataclass
class StandardHoughResult:
    """Accumulator and dominant peaks from the standard Hough transform."""

    accumulator: np.ndarray
    angles: np.ndarray
    distances: np.ndarray
    peak_accumulator: np.ndarray
    peak_angles: np.ndarray
    peak_distances: np.ndarray


@dataclass
class ProbabilisticHoughResult:
    """Detected straight line segments from probabilistic Hough."""

    segments: list[tuple[tuple[float, float], tuple[float, float]]]
    theta: np.ndarray


@dataclass
class PageScoreResult:
    """Heuristic score summary for one closed page-detection result."""

    score: float
    edge_support: float
    movement: float
    border_fraction: float
    bbox_fill_ratio: float
    page_area_ratio: float
    area_penalty: float

    def as_dict(self) -> dict[str, float]:
        """Return the score summary in a notebook-friendly flat dict."""

        return {
            "score": float(self.score),
            "edge_support": float(self.edge_support),
            "movement": float(self.movement),
            "border_fraction": float(self.border_fraction),
            "bbox_fill_ratio": float(self.bbox_fill_ratio),
            "page_area_ratio": float(self.page_area_ratio),
            "area_penalty": float(self.area_penalty),
        }
