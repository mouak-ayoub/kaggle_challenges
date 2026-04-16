from dataclasses import dataclass, field
from typing import Literal

EnhancementMode = Literal["none", "unsharp"]
EnergyMode = Literal["gaussian", "sobel", "sobel_binary", "canny", "laplace_abs", "laplace_abs_inv"]
PostEnergyMode = Literal["none", "gamma"]
BoundaryCondition = Literal["periodic", "free", "fixed", "free-fixed", "fixed-free"]


@dataclass(frozen=True)
class ResizeConfig:
    """Resize settings for fast notebook experiments."""

    max_dim: int = 1000


@dataclass(frozen=True)
class EnhancementConfig:
    """Optional intensity enhancement before feature extraction."""

    mode: EnhancementMode = "none"
    unsharp_radius: float = 2.0
    unsharp_amount: float = 1.5


@dataclass(frozen=True)
class EnergyConfig:
    """How the energy image is built for the snake."""

    mode: EnergyMode = "sobel"
    gaussian_sigma: float = 1.0
    sobel_binary_threshold: float = 0.16
    canny_sigma: float = 2.0
    canny_low_threshold: float = 0.10
    canny_high_threshold: float = 0.25
    post_brighten_mode: PostEnergyMode = "none"
    post_brighten_gamma: float = 0.6
    outer_black_border: int = 10


@dataclass(frozen=True)
class SnakeConfig:
    """Active-contour parameters for the circular page snake."""

    alpha: float = 0.1
    beta: float = 1.0
    gamma: float = 0.01
    w_line: float = 4.0
    w_edge: float = 1.0
    max_px_move: float = 1.0
    max_num_iter: int = 3000
    convergence: float = 0.01
    boundary_condition: BoundaryCondition = "periodic"
    debug_step: int | None = None


@dataclass(frozen=True)
class ClosedContourInitConfig:
    """Initialization of the rectangular contour around the image."""

    margin_ratio: float = 0.0
    n_points: int = 500


@dataclass(frozen=True)
class ClosedPageDetectionConfig:
    """Grouped config for page detection with one closed contour."""

    resize: ResizeConfig = field(default_factory=ResizeConfig)
    enhancement: EnhancementConfig = field(default_factory=EnhancementConfig)
    energy: EnergyConfig = field(default_factory=EnergyConfig)
    init: ClosedContourInitConfig = field(default_factory=ClosedContourInitConfig)
    snake: SnakeConfig = field(default_factory=SnakeConfig)


@dataclass(frozen=True)
class PageScoreWeights:
    """Weights of the heuristic page-detection score."""

    edge: float = 2.0
    movement: float = 0.8
    fill: float = 0.7
    border: float = 1.4
    area: float = 1.0


@dataclass(frozen=True)
class PageScorePriors:
    """Broad geometric priors used by the heuristic page score."""

    page_area_low: float = 0.55
    page_area_high: float = 0.92
    border_margin: int = 12


@dataclass(frozen=True)
class StandardHoughConfig:
    """Parameters for the standard Hough transform and peak selection."""

    theta_step_degrees: float = 0.3
    n_peaks: int = 70


@dataclass(frozen=True)
class ProbabilisticHoughConfig:
    """Parameters for probabilistic Hough line-segment detection."""

    theta_step_degrees: float = 0.3
    threshold: int = 10
    line_length: int = 80
    line_gap: int = 8
