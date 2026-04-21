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
    """Accumulator plus optional smoothed view and peak locations."""

    accumulator: np.ndarray
    smoothed_accumulator: np.ndarray | None
    angles: np.ndarray
    distances: np.ndarray
    peak_accumulator: np.ndarray
    peak_angles: np.ndarray
    peak_distances: np.ndarray
    peak_values: np.ndarray


@dataclass
class HoughThresholdEntry:
    """One threshold-qualified accumulator bin with its image-space line segment."""

    entry_index: int
    rho_idx: int
    theta_idx: int
    rho: float
    theta: float
    theta_deg: float
    value: float
    segment: tuple[tuple[float, float], tuple[float, float]]
    projected_rho: float | None = None
    accum_norm: float | None = None
    above_global_threshold: bool | None = None


@dataclass
class HoughLinePairScore:
    """Score of one ordered pair of candidate lines inside one theta family."""

    line_low: HoughThresholdEntry
    line_high: HoughThresholdEntry
    accum_pair: float
    separation: float
    sep_norm: float
    score: float


@dataclass
class HoughLineFamily:
    """One angle family of threshold-qualified Hough lines."""

    name: str
    center_theta_deg: float
    tolerance_deg: float
    reference_theta_deg: float
    strategy: str = "global_threshold_extrema"
    entries: list[HoughThresholdEntry] = field(default_factory=list)
    candidate_lines: list[HoughThresholdEntry] = field(default_factory=list)
    scored_pairs: list[HoughLinePairScore] = field(default_factory=list)
    projected_span: float | None = None
    min_entry: HoughThresholdEntry | None = None
    max_entry: HoughThresholdEntry | None = None


@dataclass
class HoughBoundaryGridDetectionResult:
    """Reusable outputs of the threshold-qualified Hough boundary-grid method."""

    energy: EnergyBuildResult
    edges: np.ndarray
    hough: StandardHoughResult
    threshold_reference: np.ndarray
    threshold_reference_name: str
    effective_threshold_value: float
    threshold_entries: list[HoughThresholdEntry] = field(default_factory=list)
    dominant_family: HoughLineFamily | None = None
    perpendicular_family: HoughLineFamily | None = None


@dataclass
class HoughRhoRefinementResult:
    """1D rho-profile refinement inside one theta family."""

    family_name: str
    center_theta_deg: float
    tolerance_deg: float
    reduction: str
    source_name: str
    theta_indices: np.ndarray
    theta_values_deg: np.ndarray
    rho_values: np.ndarray
    profile_values: np.ndarray
    peak_rho_indices: np.ndarray
    peak_rhos: np.ndarray
    peak_values: np.ndarray
    peak_prominences: np.ndarray


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
