import os
import numpy as np
import matplotlib.pyplot as plt
import torch

def save_pred_debug_epoch(
    model,
    sample_x, sample_y,
    epoch,
    out_dir="debug_train_evolution",
    ch_names=None,              # list of channel names length K (optional)
    ch_idxs=(1, 12),            # e.g. (Lead II index, Rhythm index)
    thr=0.3,
    device="cuda"
):
    """
    Saves a PNG showing:
      - input image
      - GT mask for selected channels
      - Pred mask for selected channels
    Assumes x shape: [1,1,H,W], y shape: [1,K,H,W]
    """
    os.makedirs(out_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        x = sample_x.to(device)
        y = sample_y.to("cpu")

        logits = model(x).detach().to("cpu")
        probs = torch.sigmoid(logits)

    # Use first item in batch
    img = x[0, 0].detach().to("cpu").numpy()  # [H,W]

    # Build figure (1 row per channel, 3 columns)
    rows = len(ch_idxs)
    fig, axs = plt.subplots(rows, 3, figsize=(12, 4 * rows))

    if rows == 1:
        axs = np.expand_dims(axs, axis=0)

    for r, c in enumerate(ch_idxs):
        gt = y[0, c].numpy()
        pr = probs[0, c].numpy()

        name = f"ch{c}"
        if ch_names is not None and c < len(ch_names):
            name = ch_names[c]

        axs[r, 0].imshow(img, cmap="gray")
        axs[r, 0].set_title("Input")
        axs[r, 0].axis("off")

        axs[r, 1].imshow(gt, cmap="gray")
        axs[r, 1].set_title(f"GT: {name}")
        axs[r, 1].axis("off")

        axs[r, 2].imshow(pr > thr, cmap="gray")
        axs[r, 2].set_title(f"Pred (thr={thr}): {name}")
        axs[r, 2].axis("off")

    plt.tight_layout()
    path = os.path.join(out_dir, f"epoch_{epoch:03d}.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)

    return path



def save_pred_debug_epoch_stageA(
    stageA,
    stageB,                  # can be None if you don't want teacher panels
    sample_src, sample_tgt,
    epoch,
    out_dir="debug_train_evolution_stageA",
    ch_names=None,
    ch_idxs=(1, 12),         # e.g. Lead II and rhythm like you used before
    thr=0.3,
    device="cuda",
    show_teacher=True,
):
    """
    Saves a PNG showing:
      Row 0: src (input) | tgt (GT clean) | pred (StageA output) | |pred-tgt|
      Rows 1..: teacher masks on tgt vs pred (optional)

    Assumes:
      sample_src shape: [1,1,H,W] or [1,H,W]
      sample_tgt shape: [1,1,H,W] or [1,H,W]
    """
    os.makedirs(out_dir, exist_ok=True)

    stageA.eval()
    if stageB is not None:
        stageB.eval()

    # normalize shapes to [1,1,H,W]
    def ensure_1chw(x):
        if x.ndim == 3:   # [B,H,W]
            return x[:, None, :, :]
        return x          # already [B,1,H,W] or [B,C,H,W]

    with torch.no_grad():
        src = ensure_1chw(sample_src).to(device)
        tgt = ensure_1chw(sample_tgt).to(device)

        pred = torch.sigmoid(stageA(src)).clamp(0, 1)

        src_np  = src[0, 0].detach().cpu().numpy()
        tgt_np  = tgt[0, 0].detach().cpu().numpy()
        pred_np = pred[0, 0].detach().cpu().numpy()
        diff_np = np.abs(pred_np - tgt_np)

        teacher_rows = 0
        tgt_probs = pred_probs = None
        if show_teacher and (stageB is not None):
            tgt_logits = stageB(tgt).detach()
            pred_logits = stageB(pred).detach()
            tgt_probs = torch.sigmoid(tgt_logits)[0].cpu().numpy()   # [K,H,W]
            pred_probs = torch.sigmoid(pred_logits)[0].cpu().numpy() # [K,H,W]
            teacher_rows = len(ch_idxs)

    rows = 1 + teacher_rows
    cols = 4 if (show_teacher and stageB is not None) else 3
    fig, axs = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))

    if rows == 1:
        axs = np.expand_dims(axs, axis=0)

    # --- Row 0: src/tgt/pred/(diff) ---
    axs[0, 0].imshow(src_np, cmap="gray")
    axs[0, 0].set_title("Input src (hard)")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(tgt_np, cmap="gray")
    axs[0, 1].set_title("GT tgt (0001 clean)")
    axs[0, 1].axis("off")

    axs[0, 2].imshow(pred_np, cmap="gray")
    axs[0, 2].set_title("StageA(src) pred")
    axs[0, 2].axis("off")

    if cols == 4:
        axs[0, 3].imshow(diff_np, cmap="gray")
        axs[0, 3].set_title("|pred - tgt|")
        axs[0, 3].axis("off")

    # --- Teacher rows: StageB(tgt) vs StageB(pred) ---
    if cols == 4:
        for r, c in enumerate(ch_idxs, start=1):
            name = f"ch{c}"
            if ch_names is not None and c < len(ch_names):
                name = ch_names[c]

            gt_mask = (tgt_probs[c] > thr)
            pr_mask = (pred_probs[c] > thr)
            d_mask = np.logical_xor(gt_mask, pr_mask)

            axs[r, 0].imshow(src_np, cmap="gray")
            axs[r, 0].set_title("Input src (ref)")
            axs[r, 0].axis("off")

            axs[r, 1].imshow(gt_mask, cmap="gray")
            axs[r, 1].set_title(f"Teacher on tgt: {name}")
            axs[r, 1].axis("off")

            axs[r, 2].imshow(pr_mask, cmap="gray")
            axs[r, 2].set_title(f"Teacher on pred: {name}")
            axs[r, 2].axis("off")

            axs[r, 3].imshow(d_mask, cmap="gray")
            axs[r, 3].set_title("XOR diff")
            axs[r, 3].axis("off")

    plt.tight_layout()
    path = os.path.join(out_dir, f"epoch_{epoch:03d}.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path

