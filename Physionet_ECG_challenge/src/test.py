import os, glob
from typing import Tuple

import numpy as np
from PIL import Image
import cv2
import torch
import segmentation_models_pytorch as smp

from mask_generation import build_physical_template

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def save_pred_overlays(model, model_image_input: Tuple[int, int], image_paths: list[str], out_dir="/data/sample/out",
                       thr=0.2):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    H_T, W_T = model_image_input
    K = 13  # number of channels

    for image_path in image_paths:
        ecg_id = os.path.basename(image_path)

        png_path = sorted(glob.glob(os.path.join(image_path, "*0001.png")))[0]
        npz_path = sorted(glob.glob(os.path.join(image_path, "mask-*.npz")))[0]

        img0 = np.array(Image.open(png_path).convert("L"), dtype=np.uint8)
        img = cv2.resize(img0.astype(np.float32) / 255.0, (W_T, H_T), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(img[None, None, ...]).float().to(DEVICE)

        z = np.load(npz_path, allow_pickle=True)
        gt0 = z["masks"]
        gt = np.zeros((H_T, W_T, K), dtype=np.uint8)
        for k in range(K):
            gt[..., k] = cv2.resize(gt0[..., k], (W_T, H_T), interpolation=cv2.INTER_NEAREST)

        with torch.no_grad():
            logits = model(x)[0].detach().cpu().numpy()  # (K,H,W)
            prob = 1 / (1 + np.exp(-logits))

        pred_union = (prob.max(axis=0) > thr).astype(np.uint8)
        gt_union = (gt.max(axis=2) > 0).astype(np.uint8)

        base = (img * 255).astype(np.uint8)
        base_rgb = np.stack([base] * 3, axis=-1)

        # --- Compute and draw baselines ---
        baseline_rows = compute_baseline_rows(H_T)

        base_with_baseline = draw_baselines(
            base_rgb,
            baseline_rows,
            color=(0, 255, 255),  # cyan
            thickness=1
        )

        # --- Compute and draw baselines ---
        baseline_rows = compute_baseline_rows(H_T)
        base_with_baseline = draw_baselines(base_rgb, baseline_rows, color=(0, 255, 255), thickness=1)

        # --- Build template and draw lead ROIs (x regions) ---
        lead_rois, lead_x_ranges, lead_row_index, px_per_cm_y, px_per_mm_y = build_physical_template(H_T, W_T)

        base_with_template = draw_lead_regions(
            base_with_baseline,
            lead_rois,
            lead_x_ranges,
            color=(255, 255, 0),  # yellow-ish boxes
            thickness=1,
            put_text=False  # set True if you want labels
        )

        # Pred overlay in red
        out_pred = base_with_template.copy()
        m = pred_union.astype(bool)
        out_pred[m] = (0.5 * out_pred[m] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)

        # GT overlay in green
        out_gt = base_with_template.copy()
        g = gt_union.astype(bool)
        out_gt[g] = (0.5 * out_gt[g] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)




        Image.fromarray(out_pred).save(os.path.join(out_dir, f"{ecg_id}_pred.png"))
        Image.fromarray(out_gt).save(os.path.join(out_dir, f"{ecg_id}_gt.png"))

    return out_dir



def draw_lead_regions(img_rgb, lead_rois, lead_x_ranges, color=(255, 255, 0), thickness=1, put_text=False):
    """
    Draws rectangles for each lead ROI: (x0,x1) and (y0,y1)
    img_rgb: uint8 (H,W,3)
    """
    out = img_rgb.copy()
    for lead, (y0, y1) in lead_rois.items():
        x0, x1 = lead_x_ranges[lead]
        # rectangle
        cv2.rectangle(out, (int(x0), int(y0)), (int(x1)-1, int(y1)-1), color, thickness)
        if put_text:
            cv2.putText(out, str(lead), (int(x0)+3, int(y0)+15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out

def load_model_checkpoint(ckpt_path: str):
    # model

    K = 13  # number of channels

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,  # IMPORTANT: None for inference
        in_channels=1,
        classes=K,
    ).to(DEVICE)
    ckpt = torch.load(ckpt_path
                      ,
                      map_location=DEVICE
                      )
    model.load_state_dict(ckpt["model"])
    return model


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


def draw_baselines(img_rgb, baseline_rows, color=(0, 255, 255), thickness=1):
    """
    img_rgb: (H,W,3) uint8
    baseline_rows: dict row -> y_px
    """
    out = img_rgb.copy()
    H, W, _ = out.shape

    for r, y in baseline_rows.items():
        if 0 <= y < H:
            cv2.line(out, (0, y), (W - 1, y), color, thickness)

    return out


if __name__ == "__main__":
    H_T = 864
    W_T = 1120
    model_out = load_model_checkpoint("../models/best_unet_resnet34_halfres_thickness_8.pt")
    base_img_path = "../data/sample"
    image_paths = [f"{base_img_path}/1006427285"]
    out_dir = save_pred_overlays(model_out, (H_T, W_T), image_paths, out_dir="../data/sample/out")
    print("Saved to:", out_dir)
