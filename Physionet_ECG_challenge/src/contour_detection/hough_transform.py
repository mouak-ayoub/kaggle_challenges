import os
import cv2
import numpy as np
import shutil

from PIL import Image


# ----------------------------
# geometry helpers
# ----------------------------
def order_points(pts):
    """pts: (4,2) float32 -> returns TL, TR, BR, BL"""
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.stack([tl, tr, br, bl], axis=0)


def four_point_warp(img_bgr, quad_xy, out_w, out_h):
    quad_xy = order_points(quad_xy)
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad_xy.astype(np.float32), dst)
    return cv2.warpPerspective(img_bgr, M, (out_w, out_h), flags=cv2.INTER_LINEAR)


# ----------------------------
# quad detection (HSV mask + fallback edges)
# ----------------------------
def _best_quad_from_contour(cnt, img_shape):
    h, w = img_shape[:2]
    area_img = float(h * w)

    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

    if len(approx) == 4:
        quad = approx.reshape(4, 2).astype(np.float32)
        area = abs(cv2.contourArea(approx))
        if area / area_img > 0.08:  # sanity: at least 8% of image
            return quad

    # fallback: minAreaRect -> 4 corners
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect).astype(np.float32)
    area = abs(cv2.contourArea(box))
    if area / area_img > 0.08:
        return box

    return None


def _quad_touches_border(quad, img_shape, margin=10):
    h, w = img_shape[:2]
    xs = quad[:, 0]
    ys = quad[:, 1]
    return (xs.min() <= margin or ys.min() <= margin or
            xs.max() >= (w - 1 - margin) or ys.max() >= (h - 1 - margin))


def _aspect_ok(quad, min_ar=1.10):
    x0, y0 = quad.min(axis=0)
    x1, y1 = quad.max(axis=0)
    bw = max(1.0, x1 - x0)
    bh = max(1.0, y1 - y0)
    ar = bw / bh
    return ar >= min_ar  # landscape-ish


def find_paper_quad_hsv(img_bgr, v_min=80, s_max=130, debug_dir=None, topk=10):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    mask = ((V >= v_min) & (S <= s_max)).astype(np.uint8) * 255

    # IMPORTANT: smaller close kernel to avoid merging ceiling + screen
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    mask = cv2.medianBlur(mask, 5)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, mask

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    quad_best = None
    for c in cnts[:topk]:
        quad = _best_quad_from_contour(c, img_bgr.shape)
        if quad is None:
            continue
        quad = order_points(quad)
        # Reject "whole image" / border blobs
        if _quad_touches_border(quad, img_bgr.shape, margin=10):
            continue
        # Optional: reject portrait-ish quads (helps in your example)
        if not _aspect_ok(quad, min_ar=1.10):
            continue

        quad_best = quad
        break

    # fallback to largest if nothing else (optional)
    if quad_best is None:
        quad_best = _best_quad_from_contour(cnts[0], img_bgr.shape)

    # debug draw (keep your existing block)
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, "paper_mask.png"), mask)
        vis = img_bgr.copy()
        if quad_best is not None:
            q = order_points(quad_best).astype(int)
            cv2.polylines(vis, [q.reshape(-1, 1, 2)], True, (0, 0, 255), 4)
        cv2.imwrite(os.path.join(debug_dir, "paper_mask_with_quad.png"), vis)

    return quad_best, mask


def find_paper_quad_edges(img_bgr, debug_dir=None):
    """Fallback if HSV mask fails (e.g., very dark / weird illumination)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(debug_dir, "edges.png"), edges)
        return None

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    quad = _best_quad_from_contour(cnts[0], img_bgr.shape)

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, "edges.png"), edges)
        vis = img_bgr.copy()
        if quad is not None:
            q = order_points(quad).astype(int)
            cv2.polylines(vis, [q.reshape(-1, 1, 2)], True, (0, 255, 0), 4)
        cv2.imwrite(os.path.join(debug_dir, "edges_with_quad.png"), vis)

    return quad


# ----------------------------
# post-warp cleanup
# ----------------------------
def crop_fixed_margin(img, shrink=0.03):
    """Crop a fixed margin on all sides (stable across stains/grid failures)."""
    h, w = img.shape[:2]
    dx = int(round(shrink * w))
    dy = int(round(shrink * h))
    if (w - 2 * dx) < 20 or (h - 2 * dy) < 20:
        return img
    return img[dy:h - dy, dx:w - dx]


def to_scan_like_gray(img_bgr, use_clahe=True):
    """
    Make a stable grayscale representation.
    For monitor moiré: grayscale + slight blur usually helps.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # small blur reduces moiré without killing waveform too much
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    return gray


def draw_hough_lines(gray_u8, out_path,
                     canny1=50, canny2=150,
                     thresh=120, min_line_len=120, max_line_gap=10):
    """
    Debug helper: draw HoughLinesP on top of grayscale image.
    """
    edges = cv2.Canny(gray_u8, canny1, canny2)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, thresh,
                            minLineLength=min_line_len, maxLineGap=max_line_gap)

    vis = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
    n = 0
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            n += 1
    cv2.imwrite(out_path, vis)
    return n

def crop_by_ink_bbox(warped_bgr, debug_dir=None, name="02e_crop_ink",
                     consec=12, pad=25,
                     min_w_frac=0.82, min_h_frac=0.80,
                     keep_bottom=True):
    """
    Crop by locating dense 'ink-like' pixels (dark + low saturation).
    Works when towel mimics grid patterns.
    Returns None if unreliable (so you can fallback safely).
    """
    import os, cv2, numpy as np

    h, w = warped_bgr.shape[:2]
    hsv = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    # adaptive thresholds from center (robust to exposure + coffee)
    y0, y1 = int(0.20*h), int(0.80*h)
    x0, x1 = int(0.20*w), int(0.80*w)
    S_c = S[y0:y1, x0:x1].astype(np.float32).reshape(-1)
    V_c = V[y0:y1, x0:x1].astype(np.float32).reshape(-1)

    s_thr = float(np.percentile(S_c, 60))  # "ink" tends to be low S
    v_thr = float(np.percentile(V_c, 20))  # "ink" tends to be low V

    s_thr = min(110.0, s_thr + 15.0)
    v_thr = max(40.0, v_thr + 10.0)

    ink = ((S <= s_thr) & (V <= v_thr)).astype(np.uint8) * 255

    # connect thin strokes
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, k, iterations=1)

    row = (ink > 0).mean(axis=1)
    col = (ink > 0).mean(axis=0)

    # thresholds relative to robust center statistics
    row_thr = max(0.002, 0.35 * np.percentile(row[y0:y1], 95))
    col_thr = max(0.002, 0.35 * np.percentile(col[x0:x1], 95))

    def first_run_from_start(arr, thr, consec):
        run = 0
        for i, v in enumerate(arr):
            run = run + 1 if v >= thr else 0
            if run >= consec:
                return i - consec + 1
        return None

    def first_run_from_end(arr, thr, consec):
        run = 0
        for k in range(len(arr)-1, -1, -1):
            v = arr[k]
            run = run + 1 if v >= thr else 0
            if run >= consec:
                return k + consec - 1
        return None

    top = first_run_from_start(row, row_thr, consec)
    bottom = first_run_from_end(row, row_thr, consec)
    left = first_run_from_start(col, col_thr, consec)
    right = first_run_from_end(col, col_thr, consec)

    if None in (top, bottom, left, right):
        return None

    # padding
    top = max(0, top - pad)
    left = max(0, left - pad)
    right = min(w - 1, right + pad)

    # protect rhythm strip: keep bottom if requested
    if keep_bottom:
        bottom = h - 1
    else:
        bottom = min(h - 1, bottom + pad)

    cropped = warped_bgr[top:bottom+1, left:right+1].copy()

    # regression guards
    if cropped.shape[1] < min_w_frac * w:
        return None
    if cropped.shape[0] < min_h_frac * h:
        return None

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, f"{name}_inkmask.png"), ink)
        vis = warped_bgr.copy()
        cv2.rectangle(vis, (left, top), (right, bottom), (0, 0, 255), 3)
        cv2.imwrite(os.path.join(debug_dir, f"{name}_bbox.png"), vis)
        cv2.imwrite(os.path.join(debug_dir, f"{name}.png"), cropped)

    return cropped

def trim_top_by_redgrid(img_bgr, debug_dir=None, name="02f_trim_top_redgrid",
                        a_thr=135, L_thr=60, k=31,
                        consec=8, pad_top=8,
                        max_trim_frac=0.18,
                        min_thr=0.004, thr_frac=0.25):
    """
    Trim ONLY the top based on axis-aligned red-grid density.
    Capped so it can't delete large content (regression guard).
    Returns original image if not confident.
    """
    import os, numpy as np, cv2

    h, w = img_bgr.shape[:2]
    max_trim_px = int(max_trim_frac * h)

    # axis-aligned red grid mask
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L = lab[..., 0]
    a = lab[..., 1]
    red = ((a > a_thr) & (L > L_thr)).astype(np.uint8) * 255

    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))
    kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k))
    red_h = cv2.morphologyEx(red, cv2.MORPH_OPEN, kh, iterations=1)
    red_v = cv2.morphologyEx(red, cv2.MORPH_OPEN, kv, iterations=1)
    red_hv = cv2.bitwise_or(red_h, red_v) > 0

    row_red = red_hv.mean(axis=1)

    # adaptive threshold from center rows
    y0, y1 = int(0.25*h), int(0.75*h)
    q95 = float(np.percentile(row_red[y0:y1], 95))
    thr = max(min_thr, thr_frac * q95)

    # find first run of rows with enough red-grid density, but only within the top cap
    run = 0
    top_idx = None
    for i in range(0, min(h, max_trim_px)):
        run = run + 1 if row_red[i] >= thr else 0
        if run >= consec:
            top_idx = i - consec + 1
            break

    if top_idx is None:
        return img_bgr  # no confident trim

    top = max(0, top_idx - pad_top)
    out = img_bgr[top:, :].copy()

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        vis = img_bgr.copy()
        cv2.line(vis, (0, top), (w-1, top), (0, 0, 255), 3)
        cv2.imwrite(os.path.join(debug_dir, f"{name}_line.png"), vis)
        cv2.imwrite(os.path.join(debug_dir, f"{name}.png"), out)

    return out
def micro_trim_left_safe(img_bgr, debug_dir=None, name="02g_micro_left",
                         max_frac=0.02, consec=8, pad=2):
    """
    Remove a tiny left margin (e.g., towel sliver) safely.
    Only trims within max_frac of width; otherwise no-op.
    """
    import os, numpy as np, cv2

    h, w = img_bgr.shape[:2]
    max_cut = max(1, int(max_frac * w))

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]
    chroma = np.sqrt((a - 128.0) ** 2 + (b - 128.0) ** 2)

    # adaptive thresholds from center area
    y0, y1 = int(0.25*h), int(0.75*h)
    x0, x1 = int(0.20*w), int(0.80*w)
    Lc = L[y0:y1, x0:x1].reshape(-1)
    Cc = chroma[y0:y1, x0:x1].reshape(-1)

    L_med = float(np.median(Lc))
    C_med = float(np.median(Cc))
    C_mad = float(np.median(np.abs(Cc - C_med)) + 1e-6)

    # "paper-ish" thresholds (soft)
    L_thr = max(20.0, L_med - 50.0)
    C_thr = min(30.0, C_med + 4.0*C_mad + 3.0)

    good = (L >= L_thr) & (chroma <= C_thr)
    col_good = good.mean(axis=0)

    # We only decide within the leftmost max_cut columns
    run = 0
    left_idx = None
    thr = max(0.50, 0.25 * float(np.percentile(col_good[x0:x1], 95)))  # adaptive but not too low
    for i in range(0, max_cut):
        run = run + 1 if col_good[i] >= thr else 0
        if run >= consec:
            left_idx = i - consec + 1
            break

    if left_idx is None:
        return img_bgr  # no confident trim

    left = max(0, left_idx - pad)
    # Safety: never trim more than max_cut
    if left > max_cut:
        return img_bgr

    out = img_bgr[:, left:].copy()

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        vis = img_bgr.copy()
        cv2.line(vis, (left, 0), (left, h-1), (0, 0, 255), 3)
        cv2.imwrite(os.path.join(debug_dir, f"{name}_line.png"), vis)
        cv2.imwrite(os.path.join(debug_dir, f"{name}.png"), out)

    return out

# ----------------------------
# MAIN PIPELINE
# ----------------------------
def rectify_and_clean(
        img_path,
        debug_dir=None,
        warp_big_w=2048,
        warp_big_h=1479,
        final_w=1100,
        final_h=850,
        shrink=0.03,
        v_min_detect=80,
        s_max_detect=130,
):
    """
    Returns:
      warped_final_bgr: (final_h, final_w, 3) uint8
      scan_like_gray  : (final_h, final_w) uint8

    Key idea:
      - Warp whole paper correctly
      - Only do SAFE fixed-margin crop (no grid-based crop that can break on stains)
      - If anything looks suspicious, fallback to full-page warp
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")

    if debug_dir:
        # delete folder and recreate
        if os.path.exists(debug_dir):
            shutil.rmtree(debug_dir)
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, "00_input.png"), img)

    # 1) detect quad
    quad, paper_mask = find_paper_quad_hsv(
        img, v_min=v_min_detect, s_max=s_max_detect, debug_dir=debug_dir
    )
    if quad is None:
        quad = find_paper_quad_edges(img, debug_dir=debug_dir)

    # last resort fallback: whole image
    if quad is None:
        h, w = img.shape[:2]
        quad = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)

    # 2) warp to big canvas (keeps detail)
    warped_big = four_point_warp(img, quad, warp_big_w, warp_big_h)
    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, "01_warped_big.png"), warped_big)

    # 3) try smart crop first (remove towel/bezel), fallback to safe fixed-margin crop
    warped_trim = crop_by_ink_bbox(warped_big, debug_dir=debug_dir, keep_bottom=True)

    if warped_trim is None:
        # fallback keeps your previously working behavior
        warped_trim = crop_fixed_margin(warped_big, shrink=shrink)


    # NEW: remove macOS/Windows top bar without touching bottom
    warped_trim = trim_top_by_redgrid(warped_trim, debug_dir=debug_dir)

    # MICRO FIX: tiny left shave (very low regression risk)
    warped_trim = micro_trim_left_safe(warped_trim, debug_dir=debug_dir, max_frac=0.02)
    warped_crop = warped_trim

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, "02_warped_crop.png"), warped_crop)

    # 4) resize to model input size
    warped_final = cv2.resize(warped_crop, (final_w, final_h), interpolation=cv2.INTER_AREA)
    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, "03_warped_final.png"), warped_final)

    scan_like = to_scan_like_gray(warped_final, use_clahe=True)
    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, "04_scan_like_gray.png"), scan_like)
        draw_hough_lines(scan_like, os.path.join(debug_dir, "05_hough_overlay.png"))

    return warped_final, scan_like

def fallback_gray(img_path, final_h, final_w):
    img0 = np.array(Image.open(img_path).convert("L"), dtype=np.uint8)
    return cv2.resize(img0, (final_w, final_h), interpolation=cv2.INTER_AREA)

def preprocess_for_model(img_path, final_h, final_w, stats=None):
    """
    Returns uint8 gray (final_h, final_w).
    Tries rectify_and_clean; on any exception/None output -> fallback.
    stats: dict counters only (no filenames).
    """
    try:
        out = rectify_and_clean(img_path=img_path, debug_dir=None, final_w=final_w, final_h=final_h)

        if out is None:
            if stats is not None: stats["rectify_none"] = stats.get("rectify_none", 0) + 1
            return fallback_gray(img_path, final_h, final_w)

        _, scan_like = out
        if scan_like is None:
            if stats is not None: stats["scan_like_none"] = stats.get("scan_like_none", 0) + 1
            return fallback_gray(img_path, final_h, final_w)

        g = scan_like
        if g.shape != (final_h, final_w):
            g = cv2.resize(g, (final_w, final_h), interpolation=cv2.INTER_AREA)

        if stats is not None: stats["ok"] = stats.get("ok", 0) + 1
        return g

    except Exception:
        if stats is not None: stats["exception"] = stats.get("exception", 0) + 1
        return fallback_gray(img_path, final_h, final_w)



if __name__ == "__main__":

    base_path = "../../data/sample"

    # Better: one ECG id -> many scan types
    ecg_scans = {
        "10140238": ["0005", "0006", "0009"],
        "1228736690": ["0001", "0009"]
    }

    for ecg_id, scan_types in ecg_scans.items():
        for scan_type in scan_types:
            img_path = f"{base_path}/{ecg_id}/{ecg_id}-{scan_type}.png"
            debug_dir = f"{base_path}/{ecg_id}/debug_rectify_{scan_type}"

            rectify_and_clean(
                img_path=img_path,
                debug_dir=debug_dir,
                final_w=1600,
                final_h=1100,
            )
            print("Done. Check:", debug_dir)
