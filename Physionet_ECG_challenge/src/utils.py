import os
import glob
import cv2
import numpy as np
from sklearn.cluster import KMeans
import numpy as np
from PIL import Image


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


LAYOUT_3x4 = [
    ["I", "aVR", "V1", "V4"],
    ["II", "aVL", "V2", "V5"],
    ["III", "aVF", "V3", "V6"],
]


def dataframe_to_masks_layout(
        df,
        H: int,
        W: int,
        layout_3x4=LAYOUT_3x4,
        rhythm_name: str = "II_rhythm",
        # Baseline offsets (cm) from the top y0 of each row (1..4)
        baseline_offsets_cm={1: 1.0, 2: 2.0, 3: 1.5, 4: 1.0},
        # Padding kept but default 0 via builder
        pad_x_units: float = 0.0,
        pad_y_cm: float = 0.0,
        thickness: int = 2,

):
    """
    df values are in mV.
    Uses physical template + per-row baseline offsets in cm to convert mV -> pixel y.
    Returns masks (H, W, K) float32 in {0,1}, channel order = df.columns.
    """
    lead_rois, lead_x_ranges, lead_row_index, px_per_cm_y, px_per_mm_y = build_physical_template(
        H, W,
        layout_3x4=layout_3x4,
        rhythm_name=rhythm_name,
        pad_x_units=pad_x_units,
        pad_y_cm=pad_y_cm,
    )

    leads = list(df.columns) + [rhythm_name]
    K = len(leads)
    masks = np.zeros((H, W, K), dtype=np.float32)

    # ECG calibration: 10 mm = 1 mV
    px_per_mV_y = px_per_mm_y * 10.0

    for k, lead in enumerate(leads):
        if lead not in lead_rois or lead not in lead_x_ranges:
            raise ValueError(f"Lead {lead} not found in layout ROIs.")

        y_top, y_bot = lead_rois[lead]
        x0, x1 = lead_x_ranges[lead]

        # Row-specific baseline: y0_of_row + offset_cm
        row_idx = lead_row_index.get(lead, None)
        offset_cm = baseline_offsets_cm.get(row_idx, None)

        # IMPORTANT: use the *row top* (before padding). We approximate with y_top - pad
        # Since pad is typically 0 here, this is fine. If pad>0, baseline shifts slightly.
        # If you want exact row-top, store row bounds separately.
        y_row_top_approx = y_top
        y_baseline = int(round(y_row_top_approx + offset_cm * px_per_cm_y))

        # If lead is rythm II take column II data
        is_rhythm = (lead == rhythm_name)
        src_lead = "II" if is_rhythm else lead

        # Extract lead data and find finite range
        ys_mV_full = df[src_lead].values.astype(np.float64)
        finite = np.isfinite(ys_mV_full)

        # One contiguous block assumption: take first..last finite indices
        idx = np.where(finite)[0]
        start = int(idx[0]) if lead != "II" else 0
        end = int(idx[-1]) + 1 if lead != "II" else int(len(ys_mV_full) / 4)

        ys_mV = ys_mV_full[start:end]
        Nb = len(ys_mV)

        # Build x positions across lead cell width for this block
        xs = np.linspace(x0, x1 - 1, num=Nb, dtype=np.float64)

        # Convert mV -> pixel y
        ys = y_baseline - ys_mV * px_per_mV_y  # positive mV goes up (smaller y)

        # Clip to image bounds
        xs = np.clip(xs, 0, W - 1)
        ys = np.clip(ys, 0, H - 1)

        x_int = np.round(xs).astype(np.int32)
        y_f = ys.astype(np.float64)

        order = np.argsort(x_int)
        x_int = x_int[order]
        y_f = y_f[order]

        ux, start_idx = np.unique(x_int, return_index=True)
        y_mean = np.array([y_f[x_int == u].mean() for u in ux])

        pts = np.stack([ux, np.round(y_mean).astype(np.int32)], axis=1).reshape(-1, 1, 2)

        ch = np.zeros((H, W), dtype=np.uint8)
        cv2.polylines(ch, [pts], isClosed=False, color=1, thickness=thickness)

        masks[..., k] = ch.astype(np.float32)

    return masks


def build_physical_template(
        H: int,
        W: int,
        layout_3x4=LAYOUT_3x4,
        # Height (cm) — physical
        total_height_cm=27.5,
        top_blank_cm=8.0,
        row_heights_cm=(2.5, 4.0, 4.0),  # rows 1..3
        rhythm_height_cm=2.0,  # row 4
        bottom_blank_cm=1.0,
        # Conceptual width breakdown (only used for fractions)
        total_width_units=28.0,
        left_blank_units=0.0,
        lead_width_units=7.0,  # each of 4 columns
        right_blank_units=0.0,
        # Padding kept, default 0
        pad_y_cm=0.0,  # physical padding in y (cm)
        pad_x_units=0.0,  # padding in x in the SAME "units" as width (default 0)
        rhythm_name="II_rhythm",
):
    """
    Returns:
      lead_rois: dict lead -> (y_top, y_bottom) in pixels
      lead_x_ranges: dict lead -> (x_left, x_right) in pixels
      lead_row_index: dict lead -> row number (1..4)
      px_per_cm_y, px_per_mm_y: for mV->px conversion on y
    """

    # ---- Y scaling: physical cm -> pixels ----
    px_per_cm_y = H / float(total_height_cm)
    px_per_mm_y = px_per_cm_y / 10.0
    pad_y = int(round(pad_y_cm * px_per_cm_y))

    # Build row bounds (pixels)
    y = int(round(top_blank_cm * px_per_cm_y))
    row_bounds = []
    for h_cm in row_heights_cm:
        y0 = y
        y1 = y + int(round(h_cm * px_per_cm_y))
        row_bounds.append((y0, y1))
        y = y1

    rhythm_y0 = y
    rhythm_y1 = y + int(round(rhythm_height_cm * px_per_cm_y))

    # (Optional) bottom blank not used further, but kept for consistency checks
    _bottom_end = rhythm_y1 + int(round(bottom_blank_cm * px_per_cm_y))

    # ---- X scaling: index space (fractions of W) ----
    # Convert "units" (conceptual cm) to pixels via fraction of total_width_units
    def units_to_x(u: float) -> int:
        return int(round((u / float(total_width_units)) * W))

    pad_x = int(round((pad_x_units / float(total_width_units)) * W))

    left_margin_px = units_to_x(left_blank_units)
    right_margin_px = units_to_x(right_blank_units)

    x_left_blank_end = left_margin_px
    x_right_blank_start = W - right_margin_px

    # 4 equal lead columns based on the conceptual units
    col_bounds = []
    for c in range(4):
        u0 = left_blank_units + c * lead_width_units
        u1 = left_blank_units + (c + 1) * lead_width_units
        x0 = units_to_x(u0)
        x1 = units_to_x(u1)
        col_bounds.append((x0, x1))

    # ---- Build dicts ----
    lead_rois = {}
    lead_x_ranges = {}
    lead_row_index = {}

    # 3x4 grid
    for r in range(3):
        y0, y1 = row_bounds[r]
        for c in range(4):
            lead = layout_3x4[r][c]
            x0, x1 = col_bounds[c]

            lead_rois[lead] = (max(0, y0 + pad_y), min(H, y1 - pad_y))
            lead_x_ranges[lead] = (max(0, x0 + pad_x), min(W, x1 - pad_x))
            lead_row_index[lead] = r + 1

    # Rhythm strip spans full usable width
    lead_rois[rhythm_name] = (max(0, rhythm_y0 + pad_y), min(H, rhythm_y1 - pad_y))
    lead_x_ranges[rhythm_name] = (max(0, x_left_blank_end + pad_x), min(W, x_right_blank_start - pad_x))
    lead_row_index[rhythm_name] = 4

    return lead_rois, lead_x_ranges, lead_row_index, px_per_cm_y, px_per_mm_y


import matplotlib.pyplot as plt
import numpy as np


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
