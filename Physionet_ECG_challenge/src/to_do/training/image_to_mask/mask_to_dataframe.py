
import torch
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"


def logits_to_prob(logits_2d):
    return 1.0 / (1.0 + np.exp(-logits_2d))


def extract_centerline(p, x0, x1, min_mass=0.5, gamma=2.0, y_fallback=None):
    H, W = p.shape
    x0 = int(max(0, min(W, x0)))
    x1 = int(max(0, min(W, x1)))
    if x1 <= x0:
        return np.zeros(0, dtype=np.float32)

    p_roi = p[:, x0:x1]
    h_roi, w_roi = p_roi.shape
    ys = np.arange(h_roi, dtype=np.float32)[:, None]

    if y_fallback is None:
        y_fallback = h_roi / 2.0

    p2 = p_roi ** gamma
    mass = p2.sum(axis=0)
    valid = mass > min_mass

    y_local = np.full(w_roi, y_fallback, dtype=np.float32)

    if valid.any():
        y_local[valid] = (p2[:, valid] * ys).sum(axis=0) / (mass[valid] + 1e-8)
        if not np.all(valid):
            xs = np.arange(w_roi)
            # with >=1 valid point, np.interp is safe
            y_local[~valid] = np.interp(xs[~valid], xs[valid], y_local[valid])

    return y_local


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

    baseline_px = layout["baseline_y"][lead_idx]

    y_px = extract_centerline(p, layout["lead_x_ranges"][lead_idx][0], layout["lead_x_ranges"][lead_idx][1],
                              y_fallback=baseline_px)

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



@torch.no_grad()
def get_logits_from_image(model, g_uint8: np.ndarray, device: str):
    """
    g_uint8: (H,W) uint8 grayscale [0..255]
    returns logits: (K,H,W) float32 numpy
    """
    if g_uint8.dtype != np.uint8:
        # allow float inputs too, but normalize if needed
        g = g_uint8.astype(np.float32)
        if g.max() > 1.5:
            g = g / 255.0
    else:
        g = g_uint8.astype(np.float32) / 255.0

    x = torch.from_numpy(g[None, None, ...]).to(device)  # (1,1,H,W)
    logits = model(x)[0].detach().cpu().numpy()          # (K,H,W)
    return logits