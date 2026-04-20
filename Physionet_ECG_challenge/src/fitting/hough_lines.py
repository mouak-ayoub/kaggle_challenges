import cv2
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


def _canonicalize_opencv_rho_theta(
    rho: float,
    theta: float,
) -> tuple[float, float]:
    """Map OpenCV's [0, pi) convention onto the local [-pi/2, pi/2) convention."""

    rho = float(rho)
    theta = float(theta)
    if theta >= np.pi / 2:
        return -rho, theta - np.pi
    return rho, theta


def _build_opencv_distance_axis(
    shape: tuple[int, int],
    rho_resolution_pixels: float,
) -> np.ndarray:
    """Build a symmetric rho axis compatible with the local Hough helpers."""

    h, w = shape
    max_abs_rho = int(np.ceil(np.hypot(max(h - 1, 0), max(w - 1, 0)) / rho_resolution_pixels))
    return np.arange(-max_abs_rho, max_abs_rho + 1, dtype=np.float64) * float(rho_resolution_pixels)


def _prepare_opencv_vote_image(
    vote_image: np.ndarray,
    *,
    use_edge_values: bool,
) -> np.ndarray:
    """Prepare an 8-bit image for OpenCV's standard or weighted Hough transform."""

    vote_image = np.asarray(vote_image, dtype=np.float64)
    if use_edge_values:
        scaled = vote_image
        if float(np.max(scaled)) <= 1.0:
            scaled = scaled * 255.0
        return np.clip(np.rint(scaled), 0, 255).astype(np.uint8)
    return np.where(vote_image > 0, 255, 0).astype(np.uint8)


def _build_sparse_accumulator_from_opencv_lines(
    lines: np.ndarray | None,
    angles: np.ndarray,
    distances: np.ndarray,
) -> np.ndarray:
    """Project OpenCV local maxima back onto an accumulator-shaped sparse map."""

    accumulator = np.zeros((distances.size, angles.size), dtype=np.float64)
    if lines is None:
        return accumulator

    for rho, theta, votes in np.asarray(lines).reshape(-1, 3):
        rho_local, theta_local = _canonicalize_opencv_rho_theta(rho=float(rho), theta=float(theta))
        angle_idx = int(np.abs(angles - theta_local).argmin())
        distance_idx = int(np.abs(distances - rho_local).argmin())
        accumulator[distance_idx, angle_idx] = max(
            float(accumulator[distance_idx, angle_idx]),
            float(votes),
        )
    return accumulator


def _extract_thresholded_sparse_peaks(
    accumulator: np.ndarray,
    angles: np.ndarray,
    distances: np.ndarray,
    *,
    threshold: float,
    n_peaks: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract threshold-qualified local maxima from a sparse accumulator map."""

    peak_rho_idx, peak_theta_idx = np.nonzero(accumulator >= threshold)
    if peak_rho_idx.size == 0:
        empty = np.array([], dtype=np.float64)
        return np.zeros_like(accumulator), empty, empty, empty

    peak_values = accumulator[peak_rho_idx, peak_theta_idx].astype(np.float64)
    order = np.argsort(-peak_values, kind="stable")
    if n_peaks > 0:
        order = order[:n_peaks]

    peak_rho_idx = peak_rho_idx[order]
    peak_theta_idx = peak_theta_idx[order]
    peak_values = peak_values[order]
    peak_angles = angles[peak_theta_idx].astype(np.float64)
    peak_distances = distances[peak_rho_idx].astype(np.float64)
    peak_accumulator = np.zeros_like(accumulator)
    peak_accumulator[peak_rho_idx, peak_theta_idx] = peak_values
    return peak_accumulator, peak_angles, peak_distances, peak_values


def _run_standard_hough_opencv(
    vote_image: np.ndarray,
    cfg: StandardHoughConfig,
) -> StandardHoughResult:
    """Compute a sparse accumulator from OpenCV's standard Hough local maxima."""

    if cfg.smooth_accumulator:
        raise ValueError("OpenCV Hough backend does not support accumulator smoothing.")

    if not hasattr(cv2, "HoughLinesWithAccumulator"):
        raise RuntimeError("OpenCV backend requires cv2.HoughLinesWithAccumulator.")

    angles = build_hough_theta(cfg.theta_step_degrees)
    distances = _build_opencv_distance_axis(vote_image.shape, cfg.rho_resolution_pixels)
    opencv_vote_image = _prepare_opencv_vote_image(
        vote_image,
        use_edge_values=cfg.opencv_use_edge_values,
    )
    lines = cv2.HoughLinesWithAccumulator(
        opencv_vote_image.copy(),
        rho=float(cfg.rho_resolution_pixels),
        theta=float(np.deg2rad(cfg.theta_step_degrees)),
        threshold=1,
        srn=0,
        stn=0,
        min_theta=0.0,
        max_theta=float(np.pi),
        use_edgeval=cfg.opencv_use_edge_values,
    )
    accumulator = _build_sparse_accumulator_from_opencv_lines(lines, angles, distances)
    peak_threshold = float(cfg.peak_threshold_ratio) * float(np.max(accumulator))
    peak_accumulator, peak_angles, peak_distances, peak_values = _extract_thresholded_sparse_peaks(
        accumulator,
        angles,
        distances,
        threshold=peak_threshold,
        n_peaks=cfg.n_peaks,
    )
    return StandardHoughResult(
        accumulator=accumulator,
        smoothed_accumulator=None,
        angles=angles,
        distances=distances,
        peak_accumulator=peak_accumulator,
        peak_angles=peak_angles,
        peak_distances=peak_distances,
        peak_values=peak_values,
    )


def run_standard_hough(
    vote_image: np.ndarray,
    cfg: StandardHoughConfig,
) -> StandardHoughResult:
    """Compute the standard Hough accumulator and keep the strongest peaks."""

    if cfg.backend == "opencv":
        return _run_standard_hough_opencv(vote_image, cfg)

    theta = build_hough_theta(cfg.theta_step_degrees)
    accumulator, angles, distances = hough_line(vote_image > 0, theta=theta)
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
