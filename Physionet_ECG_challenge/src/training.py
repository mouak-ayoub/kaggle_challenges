import os, glob, random
import numpy as np
from PIL import Image
import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import segmentation_models_pytorch as smp

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_ROOT = "../data/train"   # <-- change
H_T, W_T = 850, 1100                # half-res of (1700,2200)
K = 13

BATCH_SIZE = 4                      # try 4; if OOM, drop to 2 or 1
NUM_WORKERS = 2
LR = 3e-4
EPOCHS = 3                          # debug run first

class ECGSegDataset(Dataset):
    def __init__(self, folders, H=H_T, W=W_T):
        self.folders = folders
        self.H, self.W = H, W

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        d = self.folders[idx]
        png_path = sorted(glob.glob(os.path.join(d, "*0001.png")))[0]
        npz_path = sorted(glob.glob(os.path.join(d, "mask-*.npz")))[0]

        # image
        img = np.array(Image.open(png_path).convert("L"), dtype=np.float32) / 255.0  # (H0,W0)
        img_r = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)      # (H,W)
        x = torch.from_numpy(img_r[None, ...]).float()                               # (1,H,W)

        # masks
        z = np.load(npz_path, allow_pickle=True)
        masks = z["masks"]  # expected (H0,W0,13) uint8

        if masks.ndim != 3:
            raise ValueError(f"Unexpected masks.ndim={masks.ndim} in {npz_path}")
        if masks.shape[-1] != K:
            raise ValueError(f"Expected last dim {K}, got {masks.shape} in {npz_path}")

        masks_r = np.zeros((self.H, self.W, K), dtype=np.uint8)
        for k in range(K):
            masks_r[..., k] = cv2.resize(masks[..., k], (self.W, self.H), interpolation=cv2.INTER_NEAREST)

        y = torch.from_numpy(np.transpose(masks_r, (2,0,1))).float()  # (13,H,W)
        return x, y


folders = sorted([p for p in glob.glob(os.path.join(DATA_ROOT, "*")) if os.path.isdir(p)])
random.shuffle(folders)

val_n = max(200, int(0.1 * len(folders)))
val_f = folders[:val_n]
tr_f  = folders[val_n:]

train_loader = DataLoader(ECGSegDataset(tr_f), batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(ECGSegDataset(val_f), batch_size=1, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

len(tr_f), len(val_f)


model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=1,
    classes=K,
    activation=None,   # logits
).to(DEVICE)


bce = nn.BCEWithLogitsLoss()

def dice_loss(logits, y, eps=1e-6):
    p = torch.sigmoid(logits)
    num = 2*(p*y).sum((2,3))
    den = (p+y).sum((2,3)) + eps
    return (1 - num/den).mean()

def loss_fn(logits, y):
    return bce(logits, y) + dice_loss(logits, y)

opt = torch.optim.AdamW(model.parameters(), lr=LR)
scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE=="cuda"))


def run_epoch(loader, train=True):
    model.train(train)
    tot, n = 0.0, 0
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(DEVICE=="cuda")):
            logits = model(x)
            loss = loss_fn(logits, y)

        if train:
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        tot += loss.item()
        n += 1
    return tot / max(n, 1)

best_val = float("inf")
for ep in range(1, EPOCHS+1):
    tr = run_epoch(train_loader, True)
    va = run_epoch(val_loader, False)
    print(f"epoch {ep}/{EPOCHS} | train {tr:.4f} | val {va:.4f}")

    if va < best_val:
        best_val = va
        torch.save({"model": model.state_dict(), "val_loss": va}, "best_unet_resnet34_halfres.pt")
        print("  saved best checkpoint")



import matplotlib.pyplot as plt

def pred_overlay(sample_idx=0, thr=0.5):
    model.eval()
    d = val_f[sample_idx]
    png_path = sorted(glob.glob(os.path.join(d, "*0001.png")))[0]
    npz_path = sorted(glob.glob(os.path.join(d, "mask-*.npz")))[0]

    img0 = np.array(Image.open(png_path).convert("L"), dtype=np.uint8)
    img = cv2.resize(img0.astype(np.float32)/255.0, (W_T, H_T), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(img[None,None,...]).float().to(DEVICE)

    z = np.load(npz_path, allow_pickle=True)
    gt0 = z["masks"]
    gt = np.zeros((H_T, W_T, K), dtype=np.uint8)
    for k in range(K):
        gt[..., k] = cv2.resize(gt0[..., k], (W_T, H_T), interpolation=cv2.INTER_NEAREST)

    with torch.no_grad():
        logits = model(x)[0].detach().cpu().numpy()  # (K,H,W)
        prob = 1/(1+np.exp(-logits))

    pred_union = (prob.max(axis=0) > thr).astype(np.uint8)
    gt_union   = (gt.max(axis=2) > 0).astype(np.uint8)

    base = (img*255).astype(np.uint8)
    base_rgb = np.stack([base]*3, axis=-1)

    # overlay pred in red
    out = base_rgb.copy()
    m = pred_union.astype(bool)
    out[m] = (0.5*out[m] + 0.5*np.array([255,0,0])).astype(np.uint8)

    plt.figure(figsize=(16,5))
    plt.subplot(1,3,1); plt.title("Half-res image"); plt.imshow(base, cmap="gray"); plt.axis("off")
    plt.subplot(1,3,2); plt.title("GT union"); plt.imshow(gt_union, cmap="gray"); plt.axis("off")
    plt.subplot(1,3,3); plt.title("Pred union overlay"); plt.imshow(out); plt.axis("off")
    plt.show()

pred_overlay(0, thr=0.5)



