

import numpy as np

def logits_to_prob(logits_2d):
    return 1.0 / (1.0 + np.exp(-logits_2d))


def extract_centerline(p, min_mass=1e-4):
    """
    p: (H, W) probability map
    returns y(x): (W,) float pixel coordinates
    """
    H, W = p.shape
    ys = np.arange(H, dtype=np.float32)[:, None]

    mass = p.sum(axis=0)
    valid = mass > min_mass

    y = np.zeros(W, dtype=np.float32)
    y[valid] = (p[:, valid] * ys).sum(axis=0) / (mass[valid] + 1e-8)

    # fill missing columns by interpolation
    if not np.all(valid):
        x = np.arange(W)
        if valid.sum() > 2:
            y[~valid] = np.interp(x[~valid], x[valid], y[valid])
        else:
            y[:] = H / 2.0

    return y


def baseline_from_layout(lead_idx, layout):
    """
    layout must contain baseline_y for each lead
    """
    return layout["baseline_y"][lead_idx]

def baseline_from_signal(y):
    return np.median(y)


def pixel_to_mv(y_px, baseline_px, px_per_mm):
    amp_px = baseline_px - y_px          # up is positive
    amp_mm = amp_px / px_per_mm
    amp_mv = amp_mm * 0.1
    return amp_mv


def time_axis(W, px_per_mm, speed_mm_s=25):
    px_per_sec = px_per_mm * speed_mm_s
    dt = 1.0 / px_per_sec
    t = np.arange(W) * dt
    return t, px_per_sec

def resample_signal(sig, src_fs, tgt_fs):
    t_src = np.arange(len(sig)) / src_fs
    t_tgt = np.arange(int(len(sig) * tgt_fs / src_fs)) / tgt_fs
    return np.interp(t_tgt, t_src, sig)

def decode_lead(
    logits_2d,
    lead_idx,
    layout,
    px_per_mm,
    target_fs=500
):
    p = logits_to_prob(logits_2d)

    y_px = extract_centerline(p)

    baseline_px = layout["baseline_y"][lead_idx]

    amp_mv = pixel_to_mv(y_px, baseline_px, px_per_mm)

    t, src_fs = time_axis(len(amp_mv), px_per_mm)

    amp_mv_rs = resample_signal(amp_mv, src_fs, target_fs)

    return amp_mv_rs

def decode_all_leads(logits, layout, px_per_mm, fs=500):
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
