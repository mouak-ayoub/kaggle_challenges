import os, glob
import random
import traceback
import time

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image

from contour_detection.hough_transform import rectify_and_clean




# import your preprocessing pipeline
# from rectify_ecg import rectify_and_clean
# If you keep it in the same notebook, just ensure rectify_and_clean is defined above.

class ECGSegDataset(Dataset):
    def __init__(
            self,
            folders,
            H, W, K,
            allowed_types=None,
            use_warp=True,
            cache_dir=None,
            debug_prob=0.0,  # e.g. 0.01 to occasionally dump debug outputs
            skip_errors=True, error_log_path="../../data/sample/bad_samples.txt", return_paths=True
    ):
        self.folders = folders
        self.H, self.W = H, W
        self.K = K
        self.allowed_types = set(allowed_types) if allowed_types is not None else None
        self.use_warp = use_warp
        self.cache_dir = cache_dir
        self.debug_prob = float(debug_prob)
        self.skip_errors = skip_errors
        self.error_log_path = error_log_path
        self.return_paths = return_paths

        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.folders)

    def _pick_png(self, d):
        pngs = sorted(glob.glob(os.path.join(d, "*-????.png")))  # id-0001.png
        if not pngs:
            pngs = sorted(glob.glob(os.path.join(d, "*????.png")))  # fallback

        if self.allowed_types is not None:
            keep = []
            for p in pngs:
                t = os.path.splitext(p)[0][-4:]
                if t in self.allowed_types:
                    keep.append(p)
            pngs = keep

        if not pngs:
            raise FileNotFoundError(f"No PNGs found in {d} with allowed_types={self.allowed_types}")

        return np.random.choice(pngs)

    def _cache_key(self, png_path):
        # include type suffix in cache key (…-0006.png etc.)
        base = os.path.basename(png_path)
        # also include H,W to avoid mixing resolutions
        return f"{base}_H{self.H}_W{self.W}.png"

    def _log_error(self, msg: str):
        if not self.error_log_path:
            return
        try:
            with open(self.error_log_path, "a") as f:
                f.write(msg + "\n")
        except Exception:
            pass

    def _preprocess_gray(self, png_path):
        if not self.use_warp:
            img0 = np.array(Image.open(png_path).convert("L"), dtype=np.uint8)
            return cv2.resize(img0, (self.W, self.H), interpolation=cv2.INTER_AREA)

        ck = None
        if self.cache_dir is not None:
            ck = os.path.join(self.cache_dir, self._cache_key(png_path))
            if os.path.exists(ck):
                g = cv2.imread(ck, cv2.IMREAD_GRAYSCALE)
                if g is not None and g.shape == (self.H, self.W):
                    return g

        out = rectify_and_clean(
            img_path=png_path,
            debug_dir=None,
            final_w=self.W,
            final_h=self.H,
        )

        if out is None:
            # IMPORTANT: log path + fallback
            self._log_error(f"[RECTIFY_NONE] path={png_path}")
            img0 = np.array(Image.open(png_path).convert("L"), dtype=np.uint8)
            g = cv2.resize(img0, (self.W, self.H), interpolation=cv2.INTER_AREA)
        else:
            warped_final, scan_like = out
            g = scan_like
            if g is None:
                self._log_error(f"[SCANLIKE_NONE] path={png_path}")
                img0 = np.array(Image.open(png_path).convert("L"), dtype=np.uint8)
                g = cv2.resize(img0, (self.W, self.H), interpolation=cv2.INTER_AREA)

        if ck is not None:
            cv2.imwrite(ck, g)

        return g


    def __getitem__(self, idx):
        d = self.folders[idx]
        png_path = None
        npz_path = None
        try:
            png_path = self._pick_png(d)
            npz_path = sorted(glob.glob(os.path.join(d, "mask-*.npz")))[0]

            g = self._preprocess_gray(png_path)
            x = torch.from_numpy((g.astype(np.float32) / 255.0)[None, ...])

            z = np.load(npz_path, allow_pickle=True)
            masks = z["masks"]
            if masks.ndim != 3 or masks.shape[-1] != self.K:
                raise ValueError(f"Bad masks shape {masks.shape}")

            masks_r = np.zeros((self.H, self.W, self.K), dtype=np.uint8)
            for k in range(self.K):
                masks_r[..., k] = cv2.resize(masks[..., k], (self.W, self.H), interpolation=cv2.INTER_NEAREST)

            y = torch.from_numpy(np.transpose(masks_r, (2, 0, 1))).float()

            return (x, y, png_path) if self.return_paths else (x, y)

        except Exception as e:
            msg = (
                f"[DATASET_ERROR] idx={idx} folder={d} png={png_path} npz={npz_path} "
                f"err={type(e).__name__}: {e}"
            )
            self._log_error(msg)
            self._log_error(traceback.format_exc())

            if self.skip_errors:
                return None  # <-- key: let training continue
            raise RuntimeError(msg) from e


if __name__ == "__main__":
    # Test the ECGSegDataset
    DATA_ROOT = "../../data/sample/"  # <-- change
    folders = sorted(
        [p for p in glob.glob(os.path.join(DATA_ROOT, "*")) if os.path.isdir(p) and os.path.basename(p).isdigit()])
    random.shuffle(folders)
    H_T, W_T = 850, 1100  # half-res of (1700,2200)
    K = 13
    dataset = ECGSegDataset(
        folders=folders,
        H=H_T,
        W=W_T,
        K=K,
        use_warp=True,
        cache_dir=f"{DATA_ROOT}/cache_ecgseg",
    )

    x, y,_ = dataset[0]
    print(f"x shape: {x.shape}, y shape: {y.shape}")
