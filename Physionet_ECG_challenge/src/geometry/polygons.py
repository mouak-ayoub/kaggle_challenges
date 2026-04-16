import numpy as np
from skimage import draw


def polygon_mask_from_snake(snake: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Rasterize a closed contour into a boolean mask."""

    rr, cc = draw.polygon(snake[:, 0], snake[:, 1], shape=shape)
    mask = np.zeros(shape, dtype=bool)
    mask[rr, cc] = True
    return mask
