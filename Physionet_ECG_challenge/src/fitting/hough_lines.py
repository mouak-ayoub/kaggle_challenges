import numpy as np
from skimage.filters import gaussian
from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line

from ..core.config import ProbabilisticHoughConfig, StandardHoughConfig
from ..core.results import ProbabilisticHoughResult, StandardHoughResult


def build_hough_theta(theta_step_degrees: float) -> np.ndarray:
    """Build the angle grid used by Hough line detection."""

    return np.deg2rad(np.arange(-90.0, 90.0, theta_step_degrees))


def line_segment_from_rho_theta(
    rho: float,
    theta: float,
    shape: tuple[int, int],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Return two image-box intersection points for one infinite Hough line."""

    h, w = shape
    candidates: list[tuple[float, float]] = []

    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))

    if abs(sin_t) > 1e-8:
        y = (rho - 0.0 * cos_t) / sin_t
        if 0.0 <= y <= h - 1:
            candidates.append((0.0, float(y)))
        y = (rho - (w - 1) * cos_t) / sin_t
        if 0.0 <= y <= h - 1:
            candidates.append((float(w - 1), float(y)))

    if abs(cos_t) > 1e-8:
        x = (rho - 0.0 * sin_t) / cos_t
        if 0.0 <= x <= w - 1:
            candidates.append((float(x), 0.0))
        x = (rho - (h - 1) * sin_t) / cos_t
        if 0.0 <= x <= w - 1:
            candidates.append((float(x), float(h - 1)))

    unique_points: list[tuple[float, float]] = []
    for point in candidates:
        if not any(np.hypot(point[0] - q[0], point[1] - q[1]) < 1e-6 for q in unique_points):
            unique_points.append(point)

    if len(unique_points) >= 2:
        return unique_points[0], unique_points[1]
    return None


def build_peak_accumulator_map(
    accumulator: np.ndarray,
    angles: np.ndarray,
    distances: np.ndarray,
    peak_values: np.ndarray,
    peak_angles: np.ndarray,
    peak_distances: np.ndarray,
) -> np.ndarray:
    """Project flat Hough peak values back onto an accumulator-shaped map."""

    peak_accumulator = np.zeros_like(accumulator)
    if peak_values.size == 0:
        return peak_accumulator

    angle_indices = np.abs(angles[None, :] - peak_angles[:, None]).argmin(axis=1)
    distance_indices = np.abs(distances[None, :] - peak_distances[:, None]).argmin(axis=1)
    peak_accumulator[distance_indices, angle_indices] = peak_values
    return peak_accumulator


def run_standard_hough(
    edges: np.ndarray,
    cfg: StandardHoughConfig,
) -> StandardHoughResult:
    """Compute the standard Hough accumulator and keep the strongest peaks."""

    theta = build_hough_theta(cfg.theta_step_degrees)
    accumulator, angles, distances = hough_line(edges, theta=theta)
    smoothed_accumulator = None
    peak_detection_accumulator = accumulator

    if cfg.smooth_accumulator:
        smoothed_accumulator = gaussian(
            accumulator.astype(np.float64),
            sigma=(cfg.accumulator_gaussian_sigma_rho, cfg.accumulator_gaussian_sigma_theta),
            preserve_range=True,
        )
        peak_detection_accumulator = smoothed_accumulator

    peak_threshold = float(cfg.peak_threshold_ratio) * float(np.max(peak_detection_accumulator))

    peak_values, peak_angles, peak_distances = hough_line_peaks(
        peak_detection_accumulator,
        angles,
        distances,
        min_distance=cfg.min_distance,
        min_angle=cfg.min_angle,
        threshold=peak_threshold,
        num_peaks=cfg.n_peaks,
    )
    peak_accumulator = build_peak_accumulator_map(
        peak_detection_accumulator,
        angles,
        distances,
        peak_values,
        peak_angles,
        peak_distances,
    )
    return StandardHoughResult(
        accumulator=accumulator,
        smoothed_accumulator=smoothed_accumulator,
        angles=angles,
        distances=distances,
        peak_accumulator=peak_accumulator,
        peak_angles=peak_angles,
        peak_distances=peak_distances,
        peak_values=peak_values,
    )


def run_probabilistic_hough(
    edges: np.ndarray,
    cfg: ProbabilisticHoughConfig,
) -> ProbabilisticHoughResult:
    """Detect visible line segments with probabilistic Hough."""

    theta = build_hough_theta(cfg.theta_step_degrees)
    segments = probabilistic_hough_line(
        edges,
        threshold=cfg.threshold,
        line_length=cfg.line_length,
        line_gap=cfg.line_gap,
        theta=theta,
    )
    return ProbabilisticHoughResult(segments=segments, theta=theta)
