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
