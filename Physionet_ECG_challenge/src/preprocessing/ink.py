import cv2
import numpy as np


def odd_kernel_size(kernel_size: int) -> int:
    """Return an odd kernel size compatible with morphology operators."""

    kernel_size = int(kernel_size)
    if kernel_size < 3:
        kernel_size = 3
    if kernel_size % 2 == 0:
        kernel_size += 1
    return kernel_size


def _to_u8(img: np.ndarray) -> np.ndarray:
    img = np.clip(img.astype(np.float32), 0.0, 1.0)
    return np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)


def build_blackhat_only(
    gray_img: np.ndarray,
    closing_kernel: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Suppress dark ink using the best notebook black-hat baseline.

    Returns
    -------
    blackhat_only:
        The grayscale image reconstructed as ``1 - blackhat_response``.
    background:
        Morphological closing background estimate in ``[0, 1]``.
    blackhat_response:
        The raw black-hat response in ``[0, 1]``.
    """

    img_u8 = _to_u8(gray_img)
    kernel_size = odd_kernel_size(closing_kernel)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    background_u8 = cv2.morphologyEx(img_u8, cv2.MORPH_CLOSE, kernel)
    blackhat_u8 = cv2.subtract(background_u8, img_u8)
    blackhat_response = blackhat_u8.astype(np.float32) / 255.0
    background = background_u8.astype(np.float32) / 255.0
    blackhat_only = 1.0 - blackhat_response
    return (
        blackhat_only.astype(np.float32),
        background.astype(np.float32),
        blackhat_response.astype(np.float32),
    )


def build_blackhat_inv(
    gray_img: np.ndarray,
    closing_kernel: int,
    blur_ksize: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Refactor of the winning `blackhat_inv` path from the morphology notebook.

    The notebook's saved best result came from:
    1. optional small Gaussian blur
    2. morphological closing to estimate a local background
    3. black-hat response = background - image
    4. blackhat_inv = 1 - blackhat_response

    Notes
    -----
    In the saved notebook run, the blur step was effectively bypassed
    (`blurred_u8 = ink_u8`), so the default here is `blur_ksize=None`.
    """

    img_u8 = _to_u8(gray_img)
    if blur_ksize is None or int(blur_ksize) <= 1:
        preprocessed_u8 = img_u8
    else:
        blur_ksize = odd_kernel_size(blur_ksize)
        preprocessed_u8 = cv2.GaussianBlur(
            img_u8,
            (blur_ksize, blur_ksize),
            0,
        )

    kernel_size = odd_kernel_size(closing_kernel)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    background_u8 = cv2.morphologyEx(preprocessed_u8, cv2.MORPH_CLOSE, kernel)
    blackhat_u8 = cv2.subtract(background_u8, preprocessed_u8)
    blackhat_response = blackhat_u8.astype(np.float32) / 255.0
    background = background_u8.astype(np.float32) / 255.0
    blackhat_inv = 1.0 - blackhat_response
    return (
        blackhat_inv.astype(np.float32),
        background.astype(np.float32),
        blackhat_response.astype(np.float32),
    )


def apply_ink_removal(
    gray_img: np.ndarray,
    *,
    enabled: bool,
    method: str = "blackhat_inv",
    closing_kernel: int = 25,
    blur_ksize: int | None = None,
) -> np.ndarray:
    """Apply one shared ink-removal method and return the processed grayscale image."""

    if not enabled:
        return gray_img.astype(np.float32, copy=False)

    method_name = str(method).strip().lower()
    if method_name in {"none", ""}:
        return gray_img.astype(np.float32, copy=False)
    if method_name == "blackhat_inv":
        processed, _, _ = build_blackhat_inv(
            gray_img,
            closing_kernel=closing_kernel,
            blur_ksize=blur_ksize,
        )
        return processed.astype(np.float32, copy=False)
    if method_name == "blackhat_only":
        processed, _, _ = build_blackhat_only(
            gray_img,
            closing_kernel=closing_kernel,
        )
        return processed.astype(np.float32, copy=False)

    raise ValueError(
        f"Unsupported ink-removal method '{method}'. "
        "Supported methods: blackhat_inv, blackhat_only, none."
    )


def generate_random_ink_from_clean(
    clean_img: np.ndarray,
    seed: int = 7,
    n_strokes: int = 14,
    n_blobs: int = 10,
    n_smudges: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a synthetic dark-ink corruption from a clean grayscale image."""

    rng = np.random.default_rng(seed)
    height, width = clean_img.shape

    stroke_layer = np.zeros((height, width), dtype=np.float32)
    blob_layer = np.zeros((height, width), dtype=np.float32)
    smudge_layer = np.zeros((height, width), dtype=np.float32)

    for _ in range(n_strokes):
        x1 = int(rng.integers(0, width))
        y1 = int(rng.integers(0, height))
        length = int(rng.integers(min(height, width) // 30, min(height, width) // 8))
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        x2 = int(np.clip(x1 + length * np.cos(angle), 0, width - 1))
        y2 = int(np.clip(y1 + length * np.sin(angle), 0, height - 1))
        thickness = int(rng.integers(2, 8))
        value = float(rng.uniform(0.18, 0.45))
        cv2.line(
            stroke_layer,
            (x1, y1),
            (x2, y2),
            value,
            thickness,
            lineType=cv2.LINE_AA,
        )

    for _ in range(n_blobs):
        center = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        axes = (
            int(rng.integers(max(4, width // 80), max(8, width // 25))),
            int(rng.integers(max(4, height // 80), max(8, height // 25))),
        )
        angle = float(rng.uniform(0.0, 180.0))
        value = float(rng.uniform(0.10, 0.60))
        cv2.ellipse(
            blob_layer,
            center,
            axes,
            angle,
            0,
            360,
            value,
            -1,
            lineType=cv2.LINE_AA,
        )

    for _ in range(n_smudges):
        center = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        axes = (
            int(rng.integers(max(20, width // 25), max(40, width // 10))),
            int(rng.integers(max(20, height // 25), max(40, height // 10))),
        )
        angle = float(rng.uniform(0.0, 180.0))
        value = float(rng.uniform(0.03, 0.50))
        cv2.ellipse(
            smudge_layer,
            center,
            axes,
            angle,
            0,
            360,
            value,
            -1,
            lineType=cv2.LINE_AA,
        )

    blob_layer = cv2.GaussianBlur(blob_layer, (0, 0), sigmaX=3, sigmaY=3)
    smudge_layer = cv2.GaussianBlur(smudge_layer, (0, 0), sigmaX=11, sigmaY=11)

    darkening = np.clip(stroke_layer + blob_layer + smudge_layer, 0.0, 0.99)
    synthetic_ink = np.clip(clean_img.astype(np.float32) - darkening, 0.0, 1.0)
    true_ink_mask = darkening > 0.03
    return (
        synthetic_ink.astype(np.float32),
        darkening.astype(np.float32),
        true_ink_mask,
    )
