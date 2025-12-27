import cv2
from PIL import Image
import segmentation_models_pytorch as smp
import torch
import pandas as pd
from mask_generation import build_physical_template
from metrics import  ecg_snr_db


def logits_to_prob(logits_2d):
    return 1.0 / (1.0 + np.exp(-logits_2d))


import numpy as np


def extract_centerline(p, x0, x1, min_mass=1e-4):
    """
    p: (H,W) prob map for one lead
    x0,x1: lead x-range (pixels)
    baseline_px: baseline y in global pixels
    band_px: half-band around baseline in pixels (e.g., 30mm worth)
    """
    H, W = p.shape
    x0 = int(max(0, min(W, x0)))
    x1 = int(max(0, min(W, x1)))

    y_out = np.full(W, np.nan, dtype=np.float32)

    p_roi = p[:, x0:x1]  # (h_roi, w_roi)
    h_roi, w_roi = p_roi.shape
    ys = np.arange(h_roi, dtype=np.float32)[:, None]

    mass = p_roi.sum(axis=0)
    valid = mass > min_mass

    y_local = np.zeros(w_roi, dtype=np.float32)
    y_local[valid] = (p_roi[:, valid] * ys).sum(axis=0) / (mass[valid] + 1e-8)

    # fill missing columns in ROI
    if not np.all(valid):
        xs = np.arange(w_roi)
        if valid.sum() > 2:
            y_local[~valid] = np.interp(xs[~valid], xs[valid], y_local[valid])
        else:
            y_local[:] = h_roi / 2.0

    y_out[x0:x1] = y_local
    return y_out[x0:x1]


def baseline_from_layout(lead_idx, layout):
    """
    layout must contain baseline_y for each lead
    """
    return layout["baseline_y"][lead_idx]


def baseline_from_signal(y):
    return np.median(y)


def pixel_to_mv(y_px, baseline_px, px_per_mm):
    amp_px = baseline_px - y_px  # up is positive
    amp_mm = amp_px / px_per_mm
    amp_mv = amp_mm * 0.1
    return amp_mv



def resample_signal(sig, src_fs, tgt_fs):
    N = len(sig)
    T = (N - 1) / src_fs               # duration covered by samples
    N_tgt = int(round(T * tgt_fs)) + 1 # include endpoint

    t_src = np.arange(N) / src_fs
    t_tgt = np.arange(N_tgt) / tgt_fs

    return np.interp(t_tgt, t_src, sig)


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


def decode_lead(
        logits_2d,
        lead_idx,
        layout,
        px_per_mm,
        target_fs=500,
        speed_mm_s=25.0,
):
    p = logits_to_prob(logits_2d)

    y_px = extract_centerline(p, layout["lead_x_ranges"][lead_idx][0], layout["lead_x_ranges"][lead_idx][1])

    baseline_px = layout["baseline_y"][lead_idx]

    amp_mv = pixel_to_mv(y_px, baseline_px, px_per_mm)

    px_per_sec = px_per_mm * speed_mm_s

    amp_mv_rs = resample_signal(amp_mv, px_per_sec, target_fs)

    return amp_mv_rs


def decode_all_leads(logits, layout, px_per_mm, fs=1000):
    signals = {}
    for k, name in enumerate(layout["lead_names"]):
        signals[name] = decode_lead(
            logits[k],
            k,
            layout,
            px_per_mm,
            fs
        )
    return signals


if __name__ == "__main__":
    # model

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    K = 13  # number of channels

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,  # IMPORTANT: None for inference
        in_channels=1,
        classes=K,
    ).to(DEVICE)
    ckpt = torch.load(
        "../models/best_unet_resnet34_halfres_thickness_8.pt",
        map_location=DEVICE
    )
    model.load_state_dict(ckpt["model"])
    model.eval()
    H_T = 864
    W_T = 1120

    # image
    ecg_id="11842146"
    img_path = f"../data/sample/{ecg_id}/{ecg_id}-0001.png"
    img0 = np.array(Image.open(img_path).convert("L"), dtype=np.uint8)
    img = cv2.resize(img0.astype(np.float32) / 255.0, (W_T, H_T), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(img[None, None, ...]).float().to(DEVICE)
    logits = model(x)[0].detach().cpu().numpy()  # (K,H,W)

    # Transform masks to signals
    baselines_per_row = compute_baseline_rows(H_T)
    _, lead_x_ranges, _, _, _ = build_physical_template(H_T, W_T)

    LAYOUT = {
        "lead_names": [
            "I", "II", "III",
            "aVR", "aVL", "aVF",
            "V1", "V2", "V3",
            "V4", "V5", "V6",
            "II_rhythm"
        ],
        "baseline_y": [
            baselines_per_row[1],  # I
            baselines_per_row[2],  # II
            baselines_per_row[3],  # III
            baselines_per_row[1],  # aVR
            baselines_per_row[2],  # aVL
            baselines_per_row[3],  # aVF
            baselines_per_row[1],  # V1
            baselines_per_row[2],  # V2
            baselines_per_row[3],  # V3
            baselines_per_row[1],  # V4
            baselines_per_row[2],  # V5
            baselines_per_row[3],  # V6
            baselines_per_row[4],  # II_rhythm
        ],
        "lead_x_ranges": [
            lead_x_ranges["I"],
            lead_x_ranges["II"],
            lead_x_ranges["III"],
            lead_x_ranges["aVR"],
            lead_x_ranges["aVL"],
            lead_x_ranges["aVF"],
            lead_x_ranges["V1"],
            lead_x_ranges["V2"],
            lead_x_ranges["V3"],
            lead_x_ranges["V4"],
            lead_x_ranges["V5"],
            lead_x_ranges["V6"],
            lead_x_ranges["II_rhythm"],
        ],  # not used in this code
    }
    ecg_metadata_path="../data/train.csv"
    meta = pd.read_csv(ecg_metadata_path)
    row = meta[meta['id'].astype(str) == str(ecg_id)]
    fs = int(row['fs'].iloc[0])
    total_height_mm = 215.0
    PX_PER_MM = H_T / float(total_height_mm)

    signals=decode_all_leads(logits, LAYOUT, PX_PER_MM, fs)
    truth=pd.read_csv(f"../data/sample/{ecg_id}/{ecg_id}.csv")
    # Slice truth according to lead
    truth_slicing_factors = {
        "I": (0.0, 0.25),
        "II": (0.0, 0.25),
        "III": (0.0, 0.25),
        "aVR": (0.25, 0.5),
        "aVL": (0.25, 0.5),
        "aVF": (0.25, 0.5),
        "V1": (0.5, 0.75),
        "V2": (0.5, 0.75),
        "V3": (0.5, 0.75),
        "V4": (0.75, 1.0),
        "V5": (0.75, 1.0),
        "V6": (0.75, 1.0),
        "II_rhythm": (0.0, 1.0),
    }

    # Exclude II_rhythm from SNR calculation
    lead_names=LAYOUT["lead_names"] [:-1]
    snr_ecg_db, per_lead = ecg_snr_db(
        signals_pred=signals,
        truth_df=truth,
        fs=fs,
        lead_names=lead_names,
        truth_slicing_factors=truth_slicing_factors,
        max_shift_s=0.2,
        return_details=True,
    )

    print(f"ECG SNR: {snr_ecg_db:.2f} dB")
    print(per_lead)  # diagnostics for Lead I
    for lead, d in per_lead.items():
        snr_lead = 10 * np.log10((d["sig_power"] + 1e-12) / (d["err_power"] + 1e-12))
        print(lead, snr_lead)
