from pathlib import Path

import numpy as np
from skimage import color, io


def find_sample_path(start: Path | None = None) -> Path:
    """Find the sample-data folder from the current workspace."""

    cwd = (start or Path.cwd()).resolve()
    candidates = [
        cwd / "data" / "sample",
        cwd.parent / "data" / "sample",
        cwd.parent.parent / "data" / "sample",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find data/sample from the current working directory.")


def load_rgb_image(image_path: str | Path) -> tuple[np.ndarray, Path]:
    """Load an image and return an RGB view plus its resolved path."""

    path = Path(image_path)
    img = io.imread(path)
    if img.ndim == 2:
        rgb = np.stack([img, img, img], axis=-1)
    elif img.shape[-1] == 4:
        rgb = img[..., :3]
    else:
        rgb = img[..., :3]
    return rgb, path


def load_sample_rgb_image(sample_root: Path, ecg_id: str, scan_type: str) -> tuple[np.ndarray, Path]:
    """Load one image from the sample ECG folder layout."""

    path = sample_root / ecg_id / f"{ecg_id}-{scan_type}.png"
    return load_rgb_image(path)


def rgb_to_gray_unit(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB to grayscale float image in a stable range."""

    gray = color.rgb2gray(rgb)
    return gray.astype(np.float64)
