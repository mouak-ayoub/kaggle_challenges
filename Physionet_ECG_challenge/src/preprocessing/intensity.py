import numpy as np
from skimage import filters, transform


def resize_keep_aspect(gray: np.ndarray, max_dim: int) -> tuple[np.ndarray, float]:
    """Resize an image while keeping aspect ratio."""

    scale = min(1.0, max_dim / max(gray.shape))
    if scale == 1.0:
        resized = gray.copy()
    else:
        resized = transform.rescale(
            gray,
            scale,
            anti_aliasing=True,
            channel_axis=None,
        )
    return resized.astype(np.float64), float(scale)


def normalize_unit(img: np.ndarray) -> np.ndarray:
    """Map an image to the [0, 1] range."""

    img = img.astype(np.float64)
    lo = float(img.min())
    hi = float(img.max())
    if hi - lo < 1e-8:
        return np.zeros_like(img)
    return (img - lo) / (hi - lo)


def zero_outer_border(img: np.ndarray, border_width: int) -> np.ndarray:
    """Set the outer frame to zero to avoid image-border attraction."""

    out = img.copy().astype(np.float64)
    b = int(max(0, border_width))
    if b == 0:
        return out
    out[:b, :] = 0.0
    out[-b:, :] = 0.0
    out[:, :b] = 0.0
    out[:, -b:] = 0.0
    return out


def enhance_image(gray_img: np.ndarray, mode: str, radius: float, amount: float) -> np.ndarray:
    """Apply a simple intensity enhancement before feature extraction."""

    if mode == "none":
        return gray_img.copy().astype(np.float64)
    if mode == "unsharp":
        return filters.unsharp_mask(
            gray_img,
            radius=radius,
            amount=amount,
            preserve_range=False,
        ).astype(np.float64)
    raise ValueError(f"Unknown enhancement mode: {mode}")


def brighten_energy_image(energy_img: np.ndarray, mode: str, gamma_value: float) -> np.ndarray:
    """Brighten the energy image after feature extraction."""

    energy = np.clip(energy_img.astype(np.float64), 0.0, 1.0)
    if mode == "none":
        return energy
    if mode == "gamma":
        return np.power(energy, gamma_value)
    raise ValueError(f"Unknown post-energy brighten mode: {mode}")
