import numpy as np

from ..core.config import HoughBoundaryGridConfig
from ..core.results import (
    EnergyBuildResult,
    HoughBoundaryGridDetectionResult,
    HoughLineFamily,
    HoughThresholdEntry,
)
from ..features import build_energy_image
from ..fitting import line_segment_from_rho_theta, run_standard_hough
from ..preprocessing import resize_keep_aspect


def _wrap_line_theta_deg(theta_deg: float) -> float:
    return float(((theta_deg + 90.0) % 180.0) - 90.0)


def _line_theta_distance_deg(theta_a_deg: float, theta_b_deg: float) -> float:
    return abs(((theta_a_deg - theta_b_deg + 90.0) % 180.0) - 90.0)


def _build_threshold_entries(
    result: HoughBoundaryGridDetectionResult,
    image_shape: tuple[int, int],
) -> list[HoughThresholdEntry]:
    threshold_rho_idx, threshold_theta_idx = np.nonzero(
        result.threshold_reference >= result.effective_threshold_value
    )
    entries: list[HoughThresholdEntry] = []
    for entry_idx, (rho_idx, theta_idx) in enumerate(
        zip(threshold_rho_idx, threshold_theta_idx),
        start=1,
    ):
        rho = float(result.hough.distances[int(rho_idx)])
        theta = float(result.hough.angles[int(theta_idx)])
        segment = line_segment_from_rho_theta(rho, theta, image_shape)
        if segment is None:
            continue
        entries.append(
            HoughThresholdEntry(
                entry_index=entry_idx,
                rho_idx=int(rho_idx),
                theta_idx=int(theta_idx),
                rho=rho,
                theta=theta,
                theta_deg=float(np.rad2deg(theta)),
                value=float(result.threshold_reference[int(rho_idx), int(theta_idx)]),
                segment=segment,
            )
        )
    entries.sort(key=lambda entry: (abs(entry.rho), entry.theta_deg, -entry.value))
    return entries


def _dominant_theta_deg(entries: list[HoughThresholdEntry]) -> float | None:
    if not entries:
        return None
    theta_deg_values = np.array([entry.theta_deg for entry in entries], dtype=float)
    unique_theta_deg, inverse = np.unique(theta_deg_values, return_inverse=True)
    theta_bin_counts = np.bincount(inverse)
    dominant_theta_idx = int(np.argmax(theta_bin_counts))
    return float(unique_theta_deg[dominant_theta_idx])


def _select_family_entries(
    entries: list[HoughThresholdEntry],
    center_theta_deg: float,
    tolerance_deg: float,
) -> list[HoughThresholdEntry]:
    family = [
        entry
        for entry in entries
        if _line_theta_distance_deg(entry.theta_deg, center_theta_deg) <= tolerance_deg
    ]
    family.sort(key=lambda entry: (-entry.value, entry.rho))
    return family


def _reference_theta_deg(entries: list[HoughThresholdEntry]) -> float | None:
    if not entries:
        return None
    theta_deg_values = np.array([entry.theta_deg for entry in entries], dtype=float)
    theta_vote_values = np.array([entry.value for entry in entries], dtype=float)
    unique_theta_deg, inverse_theta = np.unique(theta_deg_values, return_inverse=True)
    theta_counts = np.bincount(inverse_theta)
    theta_vote_sums = np.bincount(inverse_theta, weights=theta_vote_values)
    best_idx = max(
        range(len(unique_theta_deg)),
        key=lambda idx: (theta_counts[idx], theta_vote_sums[idx]),
    )
    return float(unique_theta_deg[best_idx])


def _projected_rho(entry: HoughThresholdEntry, reference_theta_deg: float) -> float:
    reference_theta_rad = float(np.deg2rad(reference_theta_deg))
    (x0, y0), (x1, y1) = entry.segment
    xm = 0.5 * (x0 + x1)
    ym = 0.5 * (y0 + y1)
    return float(xm * np.cos(reference_theta_rad) + ym * np.sin(reference_theta_rad))


def _build_line_family(
    *,
    name: str,
    center_theta_deg: float,
    tolerance_deg: float,
    entries: list[HoughThresholdEntry],
) -> HoughLineFamily:
    reference_theta_deg = _reference_theta_deg(entries)
    if reference_theta_deg is None:
        return HoughLineFamily(
            name=name,
            center_theta_deg=center_theta_deg,
            tolerance_deg=tolerance_deg,
            reference_theta_deg=center_theta_deg,
            entries=[],
        )

    entry_with_projection = [
        (entry, _projected_rho(entry, reference_theta_deg))
        for entry in entries
    ]
    min_entry = min(entry_with_projection, key=lambda item: (item[1], -item[0].value))[0]
    max_entry = max(entry_with_projection, key=lambda item: (item[1], item[0].value))[0]
    return HoughLineFamily(
        name=name,
        center_theta_deg=center_theta_deg,
        tolerance_deg=tolerance_deg,
        reference_theta_deg=reference_theta_deg,
        entries=list(entries),
        min_entry=min_entry,
        max_entry=max_entry,
    )


def run_hough_boundary_grid_detection(
    gray_img: np.ndarray,
    cfg: HoughBoundaryGridConfig,
) -> HoughBoundaryGridDetectionResult:
    """Run the threshold-qualified Hough boundary-grid method on one grayscale image."""

    resized_gray, resize_scale = resize_keep_aspect(gray_img, cfg.resize.max_dim)
    smoothed, enhanced, raw_energy, final_energy = build_energy_image(
        resized_gray,
        cfg.enhancement,
        cfg.energy,
    )
    energy_result = EnergyBuildResult(
        resized_gray=resized_gray,
        smoothed_gray=smoothed,
        enhanced_gray=enhanced,
        raw_energy=raw_energy,
        final_energy=final_energy,
        resize_scale=resize_scale,
    )
    edges = final_energy > 0
    hough_result = run_standard_hough(edges, cfg.standard_hough)
    threshold_reference = (
        hough_result.smoothed_accumulator
        if hough_result.smoothed_accumulator is not None
        else hough_result.accumulator
    )
    threshold_reference_name = (
        "smoothed accumulator"
        if hough_result.smoothed_accumulator is not None
        else "raw accumulator"
    )
    effective_threshold_value = float(
        cfg.standard_hough.peak_threshold_ratio * np.max(threshold_reference)
    )
    result = HoughBoundaryGridDetectionResult(
        energy=energy_result,
        edges=edges,
        hough=hough_result,
        threshold_reference=threshold_reference,
        threshold_reference_name=threshold_reference_name,
        effective_threshold_value=effective_threshold_value,
    )
    threshold_entries = _build_threshold_entries(result, resized_gray.shape)
    result.threshold_entries = threshold_entries

    dominant_theta_deg = _dominant_theta_deg(threshold_entries)
    if dominant_theta_deg is None:
        return result

    perpendicular_theta_deg = _wrap_line_theta_deg(dominant_theta_deg + 90.0)
    dominant_entries = _select_family_entries(
        threshold_entries,
        dominant_theta_deg,
        cfg.primary_theta_tolerance_deg,
    )
    perpendicular_entries = _select_family_entries(
        threshold_entries,
        perpendicular_theta_deg,
        cfg.perpendicular_theta_tolerance_deg,
    )
    result.dominant_family = _build_line_family(
        name="dominant",
        center_theta_deg=dominant_theta_deg,
        tolerance_deg=cfg.primary_theta_tolerance_deg,
        entries=dominant_entries,
    )
    result.perpendicular_family = _build_line_family(
        name="perpendicular",
        center_theta_deg=perpendicular_theta_deg,
        tolerance_deg=cfg.perpendicular_theta_tolerance_deg,
        entries=perpendicular_entries,
    )
    return result
