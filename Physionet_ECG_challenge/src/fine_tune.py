import os, glob
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils import dataframe_to_masks_layout, LAYOUT_3x4

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
def process_one_ecg_folder(ecg_dir: str):
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
        thickness=4,
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
    overlay = make_debug_overlay(img_gray, masks_u8, alpha=0.35)
    Image.fromarray(overlay).save(os.path.join(ecg_dir, DEBUG_PNG_NAME))

    return True, "ok"




def process_all_parallel(train_root_path, limit=None, workers=8):
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
        futures = {ex.submit(process_one_ecg_folder, d): d for d in ecg_dirs}

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


# Example: run on a small subset first

if __name__ == "__main__":
    train_root_path = r"../data/train"  # change if needed
    process_all_parallel(train_root_path, limit=5, workers=2)
