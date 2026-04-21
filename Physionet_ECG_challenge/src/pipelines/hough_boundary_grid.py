import numpy as np

from ..core.config import HoughBoundaryGridConfig
from ..core.results import (
    EnergyBuildResult,
    HoughBoundaryGridDetectionResult,
    HoughLineFamily,
    HoughLinePairScore,
    HoughThresholdEntry,
)
from ..features import build_energy_image
from ..fitting import line_segment_from_rho_theta, run_standard_hough
from ..preprocessing import resize_keep_aspect


_LINE_SELECTION_STRATEGY_ALIASES: dict[str, str] = {
    "global": "global_threshold_extrema",
    "global_threshold_extrema": "global_threshold_extrema",
    "score": "theta_guided_rho_pair_score",
    "theta_guided_rho_pair_score": "theta_guided_rho_pair_score",
}


def _wrap_line_theta_deg(theta_deg: float) -> float:
    return float(((theta_deg + 90.0) % 180.0) - 90.0)


def _line_theta_distance_deg(theta_a_deg: float, theta_b_deg: float) -> float:
    return abs(((theta_a_deg - theta_b_deg + 90.0) % 180.0) - 90.0)


def _normalize_line_selection_strategy(strategy: str) -> str:
    normalized = _LINE_SELECTION_STRATEGY_ALIASES.get(str(strategy).strip().lower())
    if normalized is None:
        allowed = ", ".join(sorted(_LINE_SELECTION_STRATEGY_ALIASES))
        raise ValueError(
            "Unknown line_selection_strategy "
            f"'{strategy}'. Allowed values: {allowed}"
        )
    return normalized


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


def _image_projected_span(image_shape: tuple[int, int], reference_theta_deg: float) -> float:
    height, width = image_shape
    reference_theta_rad = float(np.deg2rad(reference_theta_deg))
    corner_values = [
        float(x * np.cos(reference_theta_rad) + y * np.sin(reference_theta_rad))
        for x, y in (
            (0.0, 0.0),
            (float(width - 1), 0.0),
            (0.0, float(height - 1)),
            (float(width - 1), float(height - 1)),
        )
    ]
    span = max(corner_values) - min(corner_values)
    return float(span if span > 0.0 else 1.0)


def _local_maxima_indices(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.array([], dtype=int)
    if values.size == 1:
        return np.array([0], dtype=int)

    maxima: list[int] = []
    if values[0] >= values[1]:
        maxima.append(0)
    interior = np.flatnonzero(
        (values[1:-1] >= values[:-2]) & (values[1:-1] >= values[2:])
    ) + 1
    maxima.extend(interior.tolist())
    if values[-1] >= values[-2]:
        maxima.append(values.size - 1)
    if not maxima:
        return np.array([], dtype=int)
    return np.array(sorted(set(maxima)), dtype=int)


def _build_line_family_global_threshold_extrema(
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
            strategy="global_threshold_extrema",
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
        strategy="global_threshold_extrema",
        entries=list(entries),
        min_entry=min_entry,
        max_entry=max_entry,
    )


def _build_line_family_theta_guided_rho_pair_score(
    *,
    name: str,
    center_theta_deg: float,
    tolerance_deg: float,
    entries: list[HoughThresholdEntry],
    threshold_reference: np.ndarray,
    hough_result,
    image_shape: tuple[int, int],
    effective_threshold_value: float,
    max_candidates_per_family: int,
    min_rho_spacing_bins: int,
    pair_accumulator_weight: float,
    pair_separation_weight: float,
) -> HoughLineFamily:
    reference_theta_deg = _reference_theta_deg(entries)
    if reference_theta_deg is None:
        reference_theta_deg = center_theta_deg

    theta_values_deg = np.rad2deg(hough_result.angles).astype(float)
    theta_band_indices = np.flatnonzero(
        [
            _line_theta_distance_deg(theta_deg, center_theta_deg) <= tolerance_deg
            for theta_deg in theta_values_deg
        ]
    )
    if theta_band_indices.size == 0:
        return HoughLineFamily(
            name=name,
            center_theta_deg=center_theta_deg,
            tolerance_deg=tolerance_deg,
            reference_theta_deg=reference_theta_deg,
            strategy="theta_guided_rho_pair_score",
            entries=list(entries),
        )

    profile_values = np.asarray(
        threshold_reference[:, theta_band_indices].max(axis=1),
        dtype=float,
    )
    local_max_idx = _local_maxima_indices(profile_values)
    sorted_local_max = sorted(
        local_max_idx.tolist(),
        key=lambda rho_idx: float(profile_values[rho_idx]),
        reverse=True,
    )

    candidate_lines: list[HoughThresholdEntry] = []
    for rho_idx in sorted_local_max:
        if any(
            abs(rho_idx - existing.rho_idx) < min_rho_spacing_bins
            for existing in candidate_lines
        ):
            continue
        rho = float(hough_result.distances[int(rho_idx)])
        theta_slice = threshold_reference[int(rho_idx), theta_band_indices]
        theta_band_offset = int(np.argmax(theta_slice))
        theta_idx = int(theta_band_indices[theta_band_offset])
        theta = float(hough_result.angles[theta_idx])
        theta_deg = float(np.rad2deg(theta))
        segment = line_segment_from_rho_theta(rho, theta, image_shape)
        if segment is None:
            continue
        projected_rho = _projected_rho(
            HoughThresholdEntry(
                entry_index=0,
                rho_idx=int(rho_idx),
                theta_idx=theta_idx,
                rho=rho,
                theta=theta,
                theta_deg=theta_deg,
                value=float(profile_values[int(rho_idx)]),
                segment=segment,
            ),
            reference_theta_deg,
        )
        candidate = HoughThresholdEntry(
            entry_index=len(candidate_lines) + 1,
            rho_idx=int(rho_idx),
            theta_idx=theta_idx,
            rho=rho,
            theta=theta,
            theta_deg=theta_deg,
            value=float(profile_values[int(rho_idx)]),
            segment=segment,
            projected_rho=projected_rho,
            above_global_threshold=bool(
                float(profile_values[int(rho_idx)]) >= effective_threshold_value
            ),
        )
        candidate_lines.append(candidate)
        if len(candidate_lines) >= max_candidates_per_family:
            break

    if not candidate_lines:
        return HoughLineFamily(
            name=name,
            center_theta_deg=center_theta_deg,
            tolerance_deg=tolerance_deg,
            reference_theta_deg=reference_theta_deg,
            strategy="theta_guided_rho_pair_score",
            entries=list(entries),
            candidate_lines=[],
            scored_pairs=[],
        )

    max_candidate_value = max(candidate.value for candidate in candidate_lines)
    projected_span = _image_projected_span(image_shape, reference_theta_deg)
    for candidate in candidate_lines:
        candidate.accum_norm = (
            float(candidate.value / max_candidate_value)
            if max_candidate_value > 0.0
            else 0.0
        )

    scored_pairs: list[HoughLinePairScore] = []
    for first_idx in range(len(candidate_lines)):
        for second_idx in range(first_idx + 1, len(candidate_lines)):
            line_a = candidate_lines[first_idx]
            line_b = candidate_lines[second_idx]
            line_low, line_high = sorted(
                (line_a, line_b),
                key=lambda candidate: float(candidate.projected_rho or 0.0),
            )
            accum_pair = float(
                min(line_a.accum_norm or 0.0, line_b.accum_norm or 0.0)
            )
            separation = float(
                abs((line_a.projected_rho or 0.0) - (line_b.projected_rho or 0.0))
            )
            sep_norm = float(np.clip(separation / projected_span, 0.0, 1.0))
            score = float(
                pair_accumulator_weight * accum_pair
                + pair_separation_weight * sep_norm
            )
            scored_pairs.append(
                HoughLinePairScore(
                    line_low=line_low,
                    line_high=line_high,
                    accum_pair=accum_pair,
                    separation=separation,
                    sep_norm=sep_norm,
                    score=score,
                )
            )

    scored_pairs.sort(
        key=lambda pair: (pair.score, pair.sep_norm, pair.accum_pair),
        reverse=True,
    )

    if len(candidate_lines) == 1:
        min_entry = candidate_lines[0]
        max_entry = candidate_lines[0]
    elif scored_pairs:
        min_entry = scored_pairs[0].line_low
        max_entry = scored_pairs[0].line_high
    else:
        min_entry = candidate_lines[0]
        max_entry = candidate_lines[-1]

    return HoughLineFamily(
        name=name,
        center_theta_deg=center_theta_deg,
        tolerance_deg=tolerance_deg,
        reference_theta_deg=reference_theta_deg,
        strategy="theta_guided_rho_pair_score",
        entries=list(entries),
        candidate_lines=candidate_lines,
        scored_pairs=scored_pairs,
        projected_span=projected_span,
        min_entry=min_entry,
        max_entry=max_entry,
    )


def run_hough_boundary_grid_detection(
    gray_img: np.ndarray,
    cfg: HoughBoundaryGridConfig,
) -> HoughBoundaryGridDetectionResult:
    """Run the threshold-qualified Hough boundary-grid method on one grayscale image."""

    line_selection_strategy = _normalize_line_selection_strategy(
        cfg.line_selection_strategy
    )
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
    hough_result = run_standard_hough(final_energy, cfg.standard_hough)
    threshold_reference = (
        hough_result.smoothed_accumulator
        if hough_result.smoothed_accumulator is not None
        else hough_result.accumulator
    )
    if hough_result.smoothed_accumulator is not None:
        threshold_reference_name = "smoothed accumulator"
    elif cfg.standard_hough.backend == "opencv":
        threshold_reference_name = "OpenCV sparse accumulator"
    else:
        threshold_reference_name = "raw accumulator"
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
    if line_selection_strategy == "theta_guided_rho_pair_score":
        result.dominant_family = _build_line_family_theta_guided_rho_pair_score(
            name="dominant",
            center_theta_deg=dominant_theta_deg,
            tolerance_deg=cfg.primary_theta_tolerance_deg,
            entries=dominant_entries,
            threshold_reference=threshold_reference,
            hough_result=hough_result,
            image_shape=resized_gray.shape,
            effective_threshold_value=effective_threshold_value,
            max_candidates_per_family=cfg.pair_max_candidates_per_family,
            min_rho_spacing_bins=cfg.pair_min_rho_spacing_bins,
            pair_accumulator_weight=cfg.pair_accumulator_weight,
            pair_separation_weight=cfg.pair_separation_weight,
        )
        result.perpendicular_family = _build_line_family_theta_guided_rho_pair_score(
            name="perpendicular",
            center_theta_deg=perpendicular_theta_deg,
            tolerance_deg=cfg.perpendicular_theta_tolerance_deg,
            entries=perpendicular_entries,
            threshold_reference=threshold_reference,
            hough_result=hough_result,
            image_shape=resized_gray.shape,
            effective_threshold_value=effective_threshold_value,
            max_candidates_per_family=cfg.pair_max_candidates_per_family,
            min_rho_spacing_bins=cfg.pair_min_rho_spacing_bins,
            pair_accumulator_weight=cfg.pair_accumulator_weight,
            pair_separation_weight=cfg.pair_separation_weight,
        )
    else:
        result.dominant_family = _build_line_family_global_threshold_extrema(
            name="dominant",
            center_theta_deg=dominant_theta_deg,
            tolerance_deg=cfg.primary_theta_tolerance_deg,
            entries=dominant_entries,
        )
        result.perpendicular_family = _build_line_family_global_threshold_extrema(
            name="perpendicular",
            center_theta_deg=perpendicular_theta_deg,
            tolerance_deg=cfg.perpendicular_theta_tolerance_deg,
            entries=perpendicular_entries,
        )
    return result
