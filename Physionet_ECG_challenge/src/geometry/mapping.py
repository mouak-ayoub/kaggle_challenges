import numpy as np

from ..core.results import CoordinateMappingResult


def map_snake_to_original_coordinates(
    snake: np.ndarray,
    resized_shape: tuple[int, int],
    original_shape: tuple[int, int],
) -> CoordinateMappingResult:
    """Map contour points from resized image space to original image space."""

    resized_h, resized_w = resized_shape
    original_h, original_w = original_shape
    scale_y = original_h / resized_h
    scale_x = original_w / resized_w

    snake_full = snake.copy().astype(np.float64)
    snake_full[:, 0] = snake_full[:, 0] * scale_y
    snake_full[:, 1] = snake_full[:, 1] * scale_x

    return CoordinateMappingResult(
        mapped_snake=snake_full,
        scale_y=scale_y,
        scale_x=scale_x,
    )
