import os, glob
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---------------------------
# CONFIG
# ---------------------------



# Output filenames inside each ECG folder
DEBUG_PNG_NAME = "debug_overlay.png"

# Fixed channel order (IMPORTANT)
LEADS_12 = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
RHYTHM = "II_rhythm"
LEADS_13 = LEADS_12 + [RHYTHM]

# ---------------------------
# HELPERS
# ---------------------------
def find_png_csv(ecg_dir: str):
    pngs = sorted(glob.glob(os.path.join(ecg_dir, "*0001.png")))
    if not pngs:
        pngs = sorted(glob.glob(os.path.join(ecg_dir, "*.png")))
    csvs = sorted(glob.glob(os.path.join(ecg_dir, "*.csv")))
    if not pngs or not csvs:
        return None, None
    return pngs[0], csvs[0]

def resize_image_and_masks(masks_hwc: np.ndarray, H: int, W: int):
    """
    img_gray: (H0, W0) uint8
    masks_hwc: (H0, W0, K) float or uint8 in {0,1}
    Returns resized (img_gray_resized, masks_resized_hwc)
    """

    # Resize masks channel-wise with nearest neighbor to preserve binary values
    K = masks_hwc.shape[2]
    masks_res = np.zeros((H, W, K), dtype=masks_hwc.dtype)
    for k in range(K):
        masks_res[..., k] = cv2.resize(masks_hwc[..., k], (W, H), interpolation=cv2.INTER_NEAREST)
    return  masks_res

def make_debug_overlay(img_gray: np.ndarray, masks_hwc: np.ndarray, alpha: float = 0.35):
    """
    Creates an RGB overlay for human inspection.
    - img_gray: (H,W) uint8
    - masks_hwc: (H,W,K) binary
    Returns RGB uint8 image.
    """
    base = np.stack([img_gray]*3, axis=-1).astype(np.float32)

    # Union of all masks
    union = (np.max(masks_hwc, axis=2) > 0).astype(np.uint8)

    # Highlight union in red
    out = base.copy()
    red = np.array([255, 0, 0], dtype=np.float32)
    m = union.astype(bool)
    out[m] = (1 - alpha)*out[m] + alpha*red

    return np.clip(out, 0, 255).astype(np.uint8)

# ---------------------------
# MAIN: PRECOMPUTE + SAVE
# ---------------------------
def process_one_ecg_folder(ecg_dir: str, overlay: bool = False,thickness: int = 1):
    png_path, csv_path = find_png_csv(ecg_dir)
    if png_path is None:
        return False, f"Missing png/csv in {ecg_dir}"

    # Load image (keep grayscale)
    img = Image.open(png_path).convert("L")
    img_gray = np.array(img)  # (H_img, W_img) uint8
    H_img, W_img = img_gray.shape

    # Load CSV and enforce lead order
    df = pd.read_csv(csv_path)
    df = df.reindex(columns=LEADS_12)

    # Build masks at image size (pixel space)
    # Uses your function (must be defined in the notebook)
    masks = dataframe_to_masks_layout(
        df=df,
        H=H_img,
        W=len(df),
        layout_3x4=LAYOUT_3x4,
        rhythm_name=RHYTHM,
        thickness=thickness,

    )  # (H_img, len(df), 13)

    # Safety checks


    # Resize to fixed size
    masks = resize_image_and_masks( masks, H_img, W_img)

    # Save NPZ (training artifact)
    # Store masks as uint8 to reduce size
    masks_u8 = (masks > 0).astype(np.uint8)
    np.savez_compressed(
        os.path.join(ecg_dir, f"mask-{ecg_dir.split(os.sep)[-1]}.npz"),
        masks=masks_u8,                 # (H,W,13)
        leads=np.array(LEADS_13),       # channel names
        H=np.array([img_gray.shape[0]]),
        W=np.array([img_gray.shape[1]]),
        version=np.array(["v1"]),
    )

    # Save one debug overlay PNG (human check)
    if overlay:
        overlay = make_debug_overlay(img_gray, masks_u8, alpha=0.35)
        Image.fromarray(overlay).save(os.path.join(ecg_dir, DEBUG_PNG_NAME))

    return True, "ok"




def process_all_parallel(train_root_path, limit=None,print_overlay=False,thickness=1, workers=8):
    ecg_dirs = sorted([p for p in glob.glob(os.path.join(train_root_path, "*")) if os.path.isdir(p)])
    if limit is not None:
        ecg_dirs = ecg_dirs[:limit]
        # For small runs, print the directories being processed
        if len(ecg_dirs) <=10:
            print("Processing only these ECG dirs:")
            for d in ecg_dirs:
                print(" ", d)

    ok, bad = 0, 0
    fails = []

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(process_one_ecg_folder, d,overlay=print_overlay,thickness=thickness): d for d in ecg_dirs}

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing ECGs", unit="ecg"):
            d = futures[fut]
            try:
                success, msg = fut.result()
            except Exception as e:
                success, msg = False, f"EXCEPTION: {repr(e)}"

            if success:
                ok += 1
            else:
                bad += 1
                fails.append((d, msg))

    print(f"DONE | ok={ok} bad={bad} total={len(ecg_dirs)}")
    if fails:
        print("Some failures (first 10):")
        for d, msg in fails[:10]:
            print("FAIL:", d, msg)


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
        baseline_offsets_cm={1: 1.0, 2: 1.5, 3: 1.0, 4: 1.0},
        # Padding kept but default 0 via builder
        pad_x_units: float = 0.0,
        pad_y_cm: float = 0.0,
        thickness: int = 1,

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

def compute_baseline_rows(
        H,
        total_height_cm=21.5,
        top_blank_cm=8.0,
        row_heights_cm=(3.0, 4.0, 3.5),
        rhythm_height_cm=2.0,
        baseline_offsets_cm={1: 1.0, 2: 1.5, 3: 1.0, 4: 1.0},
):
    """
    Returns: dict row_index (1..4) -> baseline_y_px
    """
    px_per_cm_y = H / float(total_height_cm)

    # Row top positions (cm)
    row1_top = top_blank_cm
    row2_top = top_blank_cm + row_heights_cm[0]
    row3_top = top_blank_cm + row_heights_cm[0] + row_heights_cm[1]
    row4_top = top_blank_cm + sum(row_heights_cm)  # rhythm

    row_top_cm = {
        1: row1_top,
        2: row2_top,
        3: row3_top,
        4: row4_top,
    }

    baseline_y = {
        r: int(round((row_top_cm[r] + baseline_offsets_cm[r]) * px_per_cm_y))
        for r in row_top_cm
    }

    return baseline_y


def build_physical_template(
        H: int,
        W: int,
        layout_3x4=LAYOUT_3x4,
        # Height (cm) — physical
        total_height_cm=21.5,
        top_blank_cm=8.0,
        row_heights_cm=(3, 4.0, 3.5),  # rows 1..3
        rhythm_height_cm=2.0,  # row 4
        bottom_blank_cm=1.0,
        # Conceptual width breakdown (only used for fractions)
        total_width_units=28.0,
        left_blank_units=1.5,
        lead_width_units=6.25,  # each of 4 columns
        right_blank_units=1.5,
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





