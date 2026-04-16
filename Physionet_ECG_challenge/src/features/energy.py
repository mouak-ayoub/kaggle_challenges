import numpy as np
from skimage import feature, filters

from ..core.config import EnergyConfig, EnhancementConfig
from ..preprocessing.intensity import (
    brighten_energy_image,
    enhance_image,
    normalize_unit,
    zero_outer_border,
)


def build_energy_image(
    gray_img: np.ndarray,
    enhancement_cfg: EnhancementConfig,
    energy_cfg: EnergyConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the intermediate and final energy images used by the snake."""

    if energy_cfg.gaussian_sigma > 0:
        smoothed = filters.gaussian(gray_img, sigma=energy_cfg.gaussian_sigma)
    else:
        smoothed = gray_img.copy().astype(np.float64)

    enhanced = enhance_image(
        smoothed,
        enhancement_cfg.mode,
        enhancement_cfg.unsharp_radius,
        enhancement_cfg.unsharp_amount,
    )

    if energy_cfg.mode == "gaussian":
        energy = enhanced
    elif energy_cfg.mode == "sobel":
        energy = filters.sobel(enhanced)
    elif energy_cfg.mode == "sobel_binary":
        energy = filters.sobel(enhanced)
    elif energy_cfg.mode == "canny":
        energy = feature.canny(
            enhanced,
            sigma=energy_cfg.canny_sigma,
            low_threshold=energy_cfg.canny_low_threshold,
            high_threshold=energy_cfg.canny_high_threshold,
        ).astype(np.float64)
    elif energy_cfg.mode == "laplace_abs":
        energy = np.abs(filters.laplace(enhanced))
    elif energy_cfg.mode == "laplace_abs_inv":
        energy = 1.0 - normalize_unit(np.abs(filters.laplace(enhanced)))
    else:
        raise ValueError(f"Unknown energy mode: {energy_cfg.mode}")

    raw_energy = normalize_unit(energy)

    if energy_cfg.mode == "sobel_binary":
        final_energy = np.where(raw_energy >= energy_cfg.sobel_binary_threshold, 1.0, 0.0)
    else:
        final_energy = brighten_energy_image(
            raw_energy,
            energy_cfg.post_brighten_mode,
            energy_cfg.post_brighten_gamma,
        )
        final_energy = normalize_unit(final_energy)

    final_energy = zero_outer_border(final_energy, energy_cfg.outer_black_border)
    return smoothed, enhanced, raw_energy, final_energy
