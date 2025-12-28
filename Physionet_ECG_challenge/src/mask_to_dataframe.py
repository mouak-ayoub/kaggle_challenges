import cv2
from PIL import Image
import segmentation_models_pytorch as smp
import torch
import pandas as pd

import config
from metrics import ecg_snr_db, plot_leads_compare_dynamic, plot_nonresampled_signal, align_by_xcorr, fit_gain_offset

device = "cuda" if torch.cuda.is_available() else "cpu"


def logits_to_prob(logits_2d):
    return 1.0 / (1.0 + np.exp(-logits_2d))


import numpy as np


def extract_centerline(p, x0, x1, min_mass=0.5,gamma=2.0):
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

    p2 = p_roi ** gamma
    mass = p2.sum(axis=0)
    # In the normal case, each column has p value superior( to 0.5
    valid = mass > min_mass

    y_local = np.zeros(w_roi, dtype=np.float32)
    y_local[valid] = (p2[:, valid] * ys).sum(axis=0) / (mass[valid] + 1e-8)

    # fill missing columns in ROI
    if not np.all(valid):
        xs = np.arange(w_roi)
        y_local[~valid] = np.interp(xs[~valid], xs[valid], y_local[valid])

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
    src_fs = round(src_fs)
    N = len(sig)
    T = N / src_fs  # duration covered by samples
    N_tgt = int(round(T * tgt_fs))  # include endpoint

    t_src = np.arange(N) / src_fs
    t_tgt = np.arange(N_tgt) / tgt_fs

    return np.interp(t_tgt, t_src, sig)




def decode_lead(
        logits_2d,
        lead_idx,
        layout,
        px_per_mm_x,
        px_per_mm_y,
        target_fs=1000,
        speed_mm_s=25.0,
):
    p = logits_to_prob(logits_2d)

    y_px = extract_centerline(p, layout["lead_x_ranges"][lead_idx][0], layout["lead_x_ranges"][lead_idx][1])

    baseline_px = layout["baseline_y"][lead_idx]

    amp_mv = pixel_to_mv(y_px, baseline_px, px_per_mm_y)

    px_per_sec = px_per_mm_x * speed_mm_s

    amp_mv_resampled = resample_signal(amp_mv, px_per_sec, target_fs)

    return amp_mv_resampled, amp_mv


def decode_all_leads(logits, layout, px_per_mm_x, px_per_mm_y, fs=1000):
    signals = {}
    signals_nonresampled = {}
    for k, name in enumerate(layout["lead_names"]):
        signals[name], signals_nonresampled[name] = decode_lead(
            logits[k],
            k,
            layout,
            px_per_mm_x,
            px_per_mm_y,
            fs
        )
    return signals, signals_nonresampled

def get_model(model_path):
    K = 13  # number of channels

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,  # IMPORTANT: None for inference
        in_channels=1,
        classes=K,
    ).to(device)
    ckpt = torch.load(
        model_path,
        map_location=device
    )
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model

def get_logits_from_image(model, img_path):

    H_T,W_T = config.H_T, config.W_T
    img0 = np.array(Image.open(img_path).convert("L"), dtype=np.uint8)
    img = cv2.resize(img0.astype(np.float32) / 255.0, (W_T, H_T), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(img[None, None, ...]).float().to(device)
    logits = model(x)[0].detach().cpu().numpy()  # (K,H,W)

    return logits

if __name__ == "__main__":


    model_path="../models/best_unet_resnet34_halfres_thickness_8.pt"
    model = get_model(model_path)

    # image
    ecg_id = "1053922973"
    is_test_dataset = True
    if is_test_dataset:
        base_path="../data/test"
        img_path = f"{base_path}/{ecg_id}.png"
        ecg_metadata_path = "../data/test.csv"

    else:
        base_path="../data/sample"
        img_path = f"{base_path}/{ecg_id}/{ecg_id}-0001.png"
        ecg_metadata_path = "../data/train.csv"


    logits = get_logits_from_image(model, img_path)

    # Transform masks to signals

    LAYOUT = config.LAYOUT
    truth_slicing_factors = config.TRUTH_SLICING_FACTORS
    meta = pd.read_csv(ecg_metadata_path)
    row = meta[meta['id'].astype(str) == str(ecg_id)]
    fs = int(row['fs'].iloc[0])

    PX_PER_MM_X,PX_PER_MM_Y = config.PX_PER_MM_X,config.PX_PER_MM_Y


    signals, signals_non_resmapled = decode_all_leads(logits, LAYOUT, PX_PER_MM_Y, PX_PER_MM_X, fs)

    if not is_test_dataset:
        truth = pd.read_csv(f"{base_path}/{ecg_id}/{ecg_id}.csv")
        plot_leads_compare_dynamic(truth, signals, fs, LAYOUT["lead_names"], truth_slicing_factors)

        snr_ecg_db, per_lead = ecg_snr_db(
            signals_pred=signals,
            truth_df=truth,
            fs=fs,
            lead_names=LAYOUT["lead_names"],
            truth_slicing_factors=truth_slicing_factors,
            max_shift_s=0.2,
            return_details=True,
        )

        print(f"ECG SNR: {snr_ecg_db:.2f} dB")
        print(per_lead)  # diagnostics for Lead I
        for lead, d in per_lead.items():
            snr_lead = 10 * np.log10((d["sig_power"] + 1e-12) / (d["err_power"] + 1e-12))
            lead_truth_col = lead if lead != "II_rhythm" else "II"
            lo_f, hi_f = truth_slicing_factors[lead]
            N_truth = len(truth)
            lo = int(round(lo_f * N_truth))
            hi = int(round(hi_f * N_truth))
            truth_lead = np.asarray(truth[lead_truth_col].values[lo:hi], dtype=np.float64)
            pred_al, truth_al, lag = align_by_xcorr(signals[lead], truth_lead, fs, max_shift_s=0.2)
            a, b = fit_gain_offset(pred_al, truth_al)
            print(lead, snr_lead)
            print("gain a =", a, "offset b =", b)

    else:
        # plot_leads_compare_dynamic(truth, signals, fs, ["II_rhythm"], truth_slicing_factors)
        print("plot only generated signals for test data")
        plot_leads_compare_dynamic(None, signals, fs, LAYOUT["lead_names"], truth_slicing_factors)

    # Plot non resampled lead I
    plot_nonresampled_signal(signals_non_resmapled, PX_PER_MM_X, "II_rhythm")
