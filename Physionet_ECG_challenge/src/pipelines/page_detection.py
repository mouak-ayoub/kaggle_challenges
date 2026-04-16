from dataclasses import dataclass

import numpy as np

from ..contours.active_snake import evolve_active_contour
from ..contours.initialization import build_rectangle_snake
from ..core.config import (
    ClosedContourInitConfig,
    ClosedPageDetectionConfig,
    EnergyConfig,
    EnhancementConfig,
    PageScorePriors,
    PageScoreWeights,
    ResizeConfig,
    SnakeConfig,
)
from ..core.results import (
    CoordinateMappingResult,
    EnergyBuildResult,
    PageScoreResult,
    SnakeEvolutionResult,
)
from ..features.energy import build_energy_image
from ..geometry.mapping import map_snake_to_original_coordinates
from ..geometry.polygons import polygon_mask_from_snake
from ..preprocessing.intensity import resize_keep_aspect


@dataclass
class ClosedPageDetectionResult:
    """Outputs of the closed-contour page-detection pipeline."""

    energy: EnergyBuildResult
    snake: SnakeEvolutionResult
    mapping: CoordinateMappingResult
    page_mask: np.ndarray


def run_closed_page_detection(
    gray_img: np.ndarray,
    cfg: ClosedPageDetectionConfig,
) -> ClosedPageDetectionResult:
    """Detect the ECG page with one closed active contour."""

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

    init_snake = build_rectangle_snake(
        resized_gray.shape,
        cfg.init.margin_ratio,
        cfg.init.n_points,
    )
    snake_result = evolve_active_contour(final_energy, init_snake, cfg.snake)
    page_mask = polygon_mask_from_snake(snake_result.final_snake, resized_gray.shape)
    mapping_result = map_snake_to_original_coordinates(
        snake_result.final_snake,
        resized_gray.shape,
        gray_img.shape,
    )

    return ClosedPageDetectionResult(
        energy=energy_result,
        snake=snake_result,
        mapping=mapping_result,
        page_mask=page_mask,
    )


def build_closed_page_detection_config(
    *,
    max_dim: int,
    enhancement_mode: str,
    unsharp_radius: float,
    unsharp_amount: float,
    energy_mode: str,
    gaussian_sigma: float,
    sobel_binary_threshold: float,
    canny_sigma: float,
    canny_low_threshold: float,
    canny_high_threshold: float,
    post_brighten_mode: str,
    post_brighten_gamma: float,
    outer_black_border: int,
    init_margin_ratio: float,
    snake_points: int,
    alpha: float,
    beta: float,
    gamma: float,
    w_line: float,
    w_edge: float,
    max_num_iter: int,
    convergence: float,
) -> ClosedPageDetectionConfig:
    """Build one grouped page-detection config from flat notebook parameters."""

    return ClosedPageDetectionConfig(
        resize=ResizeConfig(max_dim=max_dim),
        enhancement=EnhancementConfig(
            mode=enhancement_mode,
            unsharp_radius=unsharp_radius,
            unsharp_amount=unsharp_amount,
        ),
        energy=EnergyConfig(
            mode=energy_mode,
            gaussian_sigma=gaussian_sigma,
            sobel_binary_threshold=sobel_binary_threshold,
            canny_sigma=canny_sigma,
            canny_low_threshold=canny_low_threshold,
            canny_high_threshold=canny_high_threshold,
            post_brighten_mode=post_brighten_mode,
            post_brighten_gamma=post_brighten_gamma,
            outer_black_border=outer_black_border,
        ),
        init=ClosedContourInitConfig(
            margin_ratio=init_margin_ratio,
            n_points=snake_points,
        ),
        snake=SnakeConfig(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            w_line=w_line,
            w_edge=w_edge,
            max_num_iter=max_num_iter,
            convergence=convergence,
        ),
    )


def _sample_energy_on_snake(energy_img: np.ndarray, snake: np.ndarray) -> float:
    y = np.clip(np.round(snake[:, 0]).astype(int), 0, energy_img.shape[0] - 1)
    x = np.clip(np.round(snake[:, 1]).astype(int), 0, energy_img.shape[1] - 1)
    return float(np.mean(energy_img[y, x]))


def _normalized_movement(
    initial_snake: np.ndarray,
    final_snake: np.ndarray,
    shape: tuple[int, int],
) -> float:
    displacement = np.linalg.norm(final_snake - initial_snake, axis=1).mean()
    diagonal = float(np.hypot(shape[0], shape[1]))
    return float(displacement / max(diagonal, 1e-8))


def _border_fraction(
    snake: np.ndarray,
    shape: tuple[int, int],
    margin: int,
) -> float:
    h, w = shape
    y = snake[:, 0]
    x = snake[:, 1]
    near_border = (y < margin) | (y > h - 1 - margin) | (x < margin) | (x > w - 1 - margin)
    return float(np.mean(near_border))


def _bbox_fill_ratio(mask: np.ndarray) -> float:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return 0.0
    bbox_area = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
    return float(mask.sum() / max(bbox_area, 1))


def _area_window_penalty(area_ratio: float, low: float, high: float) -> float:
    if area_ratio < low:
        return float((low - area_ratio) / max(high - low, 1e-8))
    if area_ratio > high:
        return float((area_ratio - high) / max(high - low, 1e-8))
    return 0.0


def score_closed_page_detection(
    result: ClosedPageDetectionResult,
    weights: PageScoreWeights,
    priors: PageScorePriors,
) -> PageScoreResult:
    """Score one page-detection result with a reusable heuristic summary."""

    edge_support = _sample_energy_on_snake(result.energy.final_energy, result.snake.final_snake)
    movement = _normalized_movement(
        result.snake.initial_snake,
        result.snake.final_snake,
        result.energy.resized_gray.shape,
    )
    border = _border_fraction(
        result.snake.final_snake,
        result.energy.resized_gray.shape,
        priors.border_margin,
    )
    fill = _bbox_fill_ratio(result.page_mask)
    area = float(result.page_mask.mean())
    area_penalty = _area_window_penalty(area, priors.page_area_low, priors.page_area_high)
    score = (
        weights.edge * edge_support
        + weights.movement * movement
        + weights.fill * fill
        - weights.border * border
        - weights.area * area_penalty
    )
    return PageScoreResult(
        score=float(score),
        edge_support=float(edge_support),
        movement=float(movement),
        border_fraction=float(border),
        bbox_fill_ratio=float(fill),
        page_area_ratio=float(area),
        area_penalty=float(area_penalty),
    )
