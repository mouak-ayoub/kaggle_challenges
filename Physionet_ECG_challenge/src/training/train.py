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
