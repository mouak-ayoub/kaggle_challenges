import numpy as np


def build_rectangle_snake(shape: tuple[int, int], margin_ratio: float, n_points: int) -> np.ndarray:
    """Build a rectangular contour sampled along the four image sides."""

    h, w = shape
    margin_y = int(margin_ratio * h)
    margin_x = int(margin_ratio * w)

    n_top = n_points // 4
    n_right = n_points // 4
    n_bottom = n_points // 4
    n_left = n_points - n_top - n_right - n_bottom

    xs_top = np.linspace(margin_x, w - 1 - margin_x, n_top, endpoint=False)
    ys_top = np.full_like(xs_top, margin_y)

    ys_right = np.linspace(margin_y, h - 1 - margin_y, n_right, endpoint=False)
    xs_right = np.full_like(ys_right, w - 1 - margin_x)

    xs_bottom = np.linspace(w - 1 - margin_x, margin_x, n_bottom, endpoint=False)
    ys_bottom = np.full_like(xs_bottom, h - 1 - margin_y)

    ys_left = np.linspace(h - 1 - margin_y, margin_y, n_left, endpoint=False)
    xs_left = np.full_like(ys_left, margin_x)

    return np.vstack([
        np.column_stack([ys_top, xs_top]),
        np.column_stack([ys_right, xs_right]),
        np.column_stack([ys_bottom, xs_bottom]),
        np.column_stack([ys_left, xs_left]),
    ]).astype(np.float64)
