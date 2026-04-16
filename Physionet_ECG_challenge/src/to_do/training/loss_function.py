import torch
import torch.nn.functional as F


def charbonnier(pred, tgt, eps=1e-3):
    return torch.mean(torch.sqrt((pred - tgt) ** 2 + eps ** 2))


def make_ink_weight(stageB, tgt, thr=0.15, dilate=13, alpha=30.0, border=16):
    with torch.no_grad():
        tp = torch.sigmoid(stageB(tgt))
        ink = tp.max(dim=1, keepdim=True).values
        m = (ink > thr).float()
        m = F.max_pool2d(m, kernel_size=dilate, stride=1, padding=dilate // 2)
        w = 1.0 + alpha * m

        if border and border > 0:
            w[..., :border, :] = 1.0
            w[..., -border:, :] = 1.0
            w[..., :, :border] = 1.0
            w[..., :, -border:] = 1.0
    return w


def weighted_charbonnier_old(pred, tgt, w, eps=1e-3):
    per = torch.sqrt((pred - tgt) ** 2 + eps ** 2)
    return (per * w).sum() / (w.sum() + 1e-6)


def weighted_charbonnier(pred, tgt, w, eps=1e-3):
    per = torch.sqrt((pred - tgt)**2 + eps**2)
    w = w / (w.mean(dim=(2,3), keepdim=True) + 1e-6)   # normalize weights
    return (per * w).mean()

def photometric_loss(pred, tgt):
    mu = (pred.mean(dim=(2,3)) - tgt.mean(dim=(2,3))).abs().mean()
    sd = (pred.std(dim=(2,3))  - tgt.std(dim=(2,3))).abs().mean()
    return mu + sd



_sobel_kx = None
_sobel_ky = None

def weighted_l1(a, b, w):
    return (torch.abs(a-b) * w).sum() / (w.sum() + 1e-6)

def sobel_grad(x):
    kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], device=x.device, dtype=x.dtype).view(1,1,3,3)
    ky = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], device=x.device, dtype=x.dtype).view(1,1,3,3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return gx, gy

def grad_loss_weighted(pred, tgt, w):
    pgx, pgy = sobel_grad(pred)
    tgx, tgy = sobel_grad(tgt)
    return weighted_l1(pgx, tgx, w) + weighted_l1(pgy, tgy, w)


def teacher_consistency_loss(stageB, pred_img, tgt_img):
    """
    stageB expects float [B,1,H,W] in [0,1]
    stageB outputs logits [B,K,H,W]
    """
    with torch.no_grad():
        teacher_prob = torch.sigmoid(stageB(tgt_img))  # [B,K,H,W] prob
    student_logits = stageB(pred_img)  # [B,K,H,W] logits (no grad to stageB params, but grad flows to pred_img)
    return F.binary_cross_entropy_with_logits(student_logits, teacher_prob)


def teacher_consistency_loss_weighted(stageB, pred_img, tgt_img, beta=10.0):
    """
    beta higher = punish 'all black' harder.
    Start with 20.
    """
    with torch.no_grad():
        tprob = torch.sigmoid(stageB(tgt_img))  # [B,K,H,W]
    slogits = stageB(pred_img)  # [B,K,H,W]

    bce = F.binary_cross_entropy_with_logits(slogits, tprob, reduction="none")
    w = 1.0 + beta * tprob  # focus where teacher sees ink
    return (bce * w).mean()


def _ensure_b1hw(x):
    # Accept [B,H,W] or [B,1,H,W]
    if x.ndim == 3:
        return x[:, None, :, :]
    return x

@torch.no_grad()
def teacher_logits(stageB, x):
    # x: [B,1,H,W] in [0,1]
    stageB.eval()
    return stageB(x)

@torch.no_grad()
def teacher_ink_map_from_logits(logits_t):
    # logits_t: [B,K,H,W]
    p = torch.sigmoid(logits_t)
    ink = p.max(dim=1).values  # [B,H,W]
    return ink

def dilate_binary_mask(m, k=13):
    """
    m: [B,H,W] float/bool in {0,1}
    dilation via maxpool
    """
    if k <= 1:
        return m
    pad = k // 2
    return F.max_pool2d(m[:, None, :, :], kernel_size=k, stride=1, padding=pad)[:, 0]

def make_weight_map_from_ink(ink_map, thr=0.15, dilate=13, alpha=30.0, border=16, eps=1e-6):
    """
    ink_map: [B,H,W] in [0,1]
    returns wmap: [B,1,H,W] normalized so per-image mean weight = 1.0
    """
    B, H, W = ink_map.shape
    ink_bin = (ink_map > thr).float()  # [B,H,W]
    ink_dil = dilate_binary_mask(ink_bin, k=dilate)  # [B,H,W]

    w = 1.0 + alpha * ink_dil  # focus around waveform

    if border and border > 0:
        b = border
        border_mask = torch.zeros((B, H, W), device=ink_map.device, dtype=w.dtype)
        border_mask[:, :b, :] = 1
        border_mask[:, -b:, :] = 1
        border_mask[:, :, :b] = 1
        border_mask[:, :, -b:] = 1
        # slightly down-weight border
        w = w * (1.0 - 0.5 * border_mask)

    # normalize by mean (NOT sum) so it stays stable across dilation sizes
    w = w / (w.mean(dim=(1, 2), keepdim=True) + eps)  # [B,H,W]
    return w[:, None, :, :]  # [B,1,H,W]

def sample_ink_guided_crop_coords(ink_small, crop_h, crop_w, p_uniform=0.20, eps=1e-6):
    """
    ink_small: [B,Hs,Ws] in [0,1]  (teacher ink map at smaller res)
    Returns list of (y0, x0) in SMALL coordinates (top-left).
    """
    B, Hs, Ws = ink_small.shape
    coords = []
    for b in range(B):
        if torch.rand(()) < p_uniform:
            y0 = torch.randint(0, max(1, Hs - crop_h + 1), (1,)).item()
            x0 = torch.randint(0, max(1, Ws - crop_w + 1), (1,)).item()
            coords.append((y0, x0))
            continue

        prob = ink_small[b].reshape(-1) + eps
        prob = prob / prob.sum()
        idx = torch.multinomial(prob, num_samples=1, replacement=True).item()
        cy = idx // Ws
        cx = idx % Ws

        y0 = int(cy - crop_h // 2)
        x0 = int(cx - crop_w // 2)
        y0 = max(0, min(y0, Hs - crop_h))
        x0 = max(0, min(x0, Ws - crop_w))
        coords.append((y0, x0))
    return coords

def crop_batch(x, coords, crop_h, crop_w):
    """
    x: [B,1,H,W]
    coords: list[(y0,x0)] in SAME coordinate system as x
    """
    crops = []
    for b, (y0, x0) in enumerate(coords):
        crops.append(x[b:b+1, :, y0:y0+crop_h, x0:x0+crop_w])
    return torch.cat(crops, dim=0)

def photo_anchor_loss(pred, tgt):
    """
    pred/tgt: [B,1,H,W] in [0,1]
    penalize global brightness/contrast drift
    """
    mean_p = pred.mean(dim=(2, 3))
    mean_t = tgt.mean(dim=(2, 3))
    std_p  = pred.std(dim=(2, 3), unbiased=False)
    std_t  = tgt.std(dim=(2, 3), unbiased=False)
    return (mean_p - mean_t).abs().mean() + (std_p - std_t).abs().mean()

def teacher_soft_bce_loss(logits_p, logits_t, wmap=None, sharp_beta=2.0, eps=1e-6):
    """
    logits_p: [B,K,H,W] (teacher on pred)  grad flows to pred
    logits_t: [B,K,H,W] (teacher on tgt)   no grad
    wmap: [B,1,H,W] optional waveform-focused weights
    sharp_beta: >1 sharpens soft targets by raising odds^beta
    """
    with torch.no_grad():
        t = torch.sigmoid(logits_t).clamp(eps, 1 - eps)  # [B,K,H,W]
        if sharp_beta and sharp_beta != 1.0:
            odds = t / (1.0 - t)
            odds = odds.pow(sharp_beta)
            t = odds / (1.0 + odds)

    bce = F.binary_cross_entropy_with_logits(logits_p, t, reduction="none")  # [B,K,H,W]
    if wmap is not None:
        bce = bce * wmap[:, None, :, :]
    return bce.mean()

def ink_mean_from_teacher_logits(logits):
    # logits: [B,K,H,W]
    return torch.sigmoid(logits).max(dim=1).values.mean(dim=(1, 2))  # [B]
