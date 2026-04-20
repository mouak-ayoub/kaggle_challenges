"""Config and result objects for reusable CV pipelines."""

from .config import (
    ClosedContourInitConfig,
    ClosedPageDetectionConfig,
    EnergyConfig,
    EnhancementConfig,
    HoughBoundaryGridConfig,
    PageScorePriors,
    PageScoreWeights,
    ProbabilisticHoughConfig,
    ResizeConfig,
    SnakeConfig,
    StandardHoughConfig,
)
from .hough_notebook_defaults import (
    get_hough_notebook_defaults_path,
    load_hough_boundary_notebook_defaults,
    make_hough_boundary_notebook_defaults,
)
from .results import (
    CoordinateMappingResult,
    EnergyBuildResult,
    HoughBoundaryGridDetectionResult,
    HoughLineFamily,
    HoughThresholdEntry,
    PageScoreResult,
    ProbabilisticHoughResult,
    SnakeEvolutionResult,
    StandardHoughResult,
)

__all__ = [
    "ClosedContourInitConfig",
    "ClosedPageDetectionConfig",
    "CoordinateMappingResult",
    "EnergyBuildResult",
    "EnergyConfig",
    "EnhancementConfig",
    "get_hough_notebook_defaults_path",
    "HoughBoundaryGridConfig",
    "HoughBoundaryGridDetectionResult",
    "HoughLineFamily",
    "load_hough_boundary_notebook_defaults",
    "make_hough_boundary_notebook_defaults",
    "HoughThresholdEntry",
    "PageScorePriors",
    "PageScoreResult",
    "PageScoreWeights",
    "ProbabilisticHoughConfig",
    "ProbabilisticHoughResult",
    "ResizeConfig",
    "SnakeConfig",
    "SnakeEvolutionResult",
    "StandardHoughConfig",
    "StandardHoughResult",
]
