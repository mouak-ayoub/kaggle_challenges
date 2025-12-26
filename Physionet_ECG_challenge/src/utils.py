
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from mask_generation import dataframe_to_masks_layout


def remove_red_grid(img_bgr: np.ndarray) -> np.ndarray:
    """Mask red pixels in HSV and paint them white."""
    print("Removing red grid from image...")
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    lower1 = np.array([0, 50, 50])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([170, 50, 50])
    upper2 = np.array([180, 255, 255])

    mask_red = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    out = img_bgr.copy()
    out[mask_red > 0] = (255, 255, 255)
    return out


def waveform_binary_mask(img_bgr: np.ndarray) -> np.ndarray:
    """Return a binary mask (uint8 0/255) of dark strokes (waveform-ish)."""
    img_no_red = remove_red_grid(img_bgr)
    gray = cv2.cvtColor(img_no_red, cv2.COLOR_BGR2GRAY)

    # Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Black-hat highlights dark thin structures
    bh_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, bh_kernel)
    blackhat = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)
    blackhat = cv2.GaussianBlur(blackhat, (3, 3), 0)

    # Otsu threshold
    _, bw = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Clean small noise
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    # Remove long grid remnants (horizontal/vertical)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    h_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel, iterations=1)
    v_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kernel, iterations=1)
    grid_lines = cv2.bitwise_or(h_lines, v_lines)
    bw = cv2.bitwise_and(bw, cv2.bitwise_not(grid_lines))

    # Slight close to connect broken waveform segments
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    return bw




def plot_all_masks_overlay(masks):
    """
    masks: (H, W, K) float32 in {0,1}
    """
    combined = np.max(masks, axis=2)  # OR union of all leads

    plt.figure(figsize=(8, 12))
    plt.imshow(combined, cmap="gray")
    plt.title("All ECG masks (overlay)")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    import pandas as pd

    data_path = "../data/train/36494663/36494663.csv"
    df_lead_data = pd.read_csv(data_path)
    layout = [
        ["I", "aVR", "V1", "V4"],
        ["II", "aVL", "V2", "V5"],
        ["III", "aVF", "V3", "V6"]
    ]
    height, width = (1700, 2200)
    masks = dataframe_to_masks_layout(df_lead_data, height, len(df_lead_data), layout)
    combined = np.max(masks, axis=2)  # OR union of all leads

    if combined.dtype != np.uint8:
        max_val = np.nanmax(combined)
        min_val = np.nanmin(combined)
        if max_val <= 1.0:  # likely binary or 0-1 mask
            scaled = (combined * 255.0).astype(np.uint8)
        else:
            # scale linearly between min and max to full 0-255 range (handles arbitrary ranges)
            if max_val == min_val:
                scaled = np.clip(combined, 0, 255).astype(np.uint8)
            else:
                scaled = (((combined - min_val) / (max_val - min_val)) * 255.0).astype(np.uint8)
    else:
        scaled = combined

    # Create PIL image (grayscale)
    im_pil = Image.fromarray(scaled, mode="L")

    # Save and show
    im_pil.save('../data/sample/combined_mask.png')
    im_pil.show()
