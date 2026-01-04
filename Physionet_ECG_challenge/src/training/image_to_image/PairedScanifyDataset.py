from pathlib import Path
import os, random
import numpy as np
import torch
from torch.utils.data import Dataset
from config import config
from contour_detection.hough_transform import  canon_tensor, canon_u8  # your file

H_T, W_T = config.H_T, config.W_T
HARD_TYPES = config.HARD_TYPES
TYPE_WEIGHTS = config.TYPE_WEIGHTS



class PairedScanifyDataset(Dataset):
    def __init__(self, train_root, ids, p_identity=0.10, cache_dir="/content/cache/paired_scanify"):
        self.train_root = train_root
        self.ids = ids
        self.p_identity = p_identity
        self.cache_dir = cache_dir

        self.types = HARD_TYPES
        w = np.array([TYPE_WEIGHTS[t] for t in self.types], dtype=np.float64)
        self.probs = (w / w.sum()).tolist()

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    def _path(self, ecg_id, t):
        return os.path.join(self.train_root, ecg_id, f"{ecg_id}-{t}.png")

    def _cache_path(self, ecg_id, t):
        return os.path.join(self.cache_dir, f"{ecg_id}-{t}-{H_T}x{W_T}.npy")

    def _load_canon(self, ecg_id, t):
        if not self.cache_dir:
            return canon_tensor(self._path(ecg_id, t))

        cache_dir = Path(self.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        cp = cache_dir / f"{ecg_id}-{t}-{H_T}x{W_T}.npy"
        if cp.exists():
            g = np.load(cp)
        else:
            g = canon_u8(self._path(ecg_id, t))

            tmp = cp.parent / (cp.name + f".{os.getpid()}.{random.getrandbits(32):08x}.tmp")
            with open(tmp, "wb") as f:
                np.save(f, g)

            try:
                os.replace(tmp, cp)  # atomic
            except FileNotFoundError:
                # Another worker may have won the race; just load the final file if it exists
                if cp.exists():
                    g = np.load(cp)
                else:
                    raise

        x = (g.astype(np.float32) / 255.0)
        return torch.from_numpy(x)[None, ...]

    def __len__(self): return len(self.ids)

    def __getitem__(self, i):
        ecg_id = self.ids[i]

        # Identity samples prevent "stylization" (StageA must not ruin clean scans)
        if random.random() < self.p_identity:
            tgt = self._load_canon(ecg_id, "0001")
            src = tgt.clone()
            return src, tgt, True

        # Weighted pick of hard types
        for _ in range(10):
            t = random.choices(self.types, weights=self.probs, k=1)[0]
            src_path = self._path(ecg_id, t)
            if os.path.exists(src_path):
                src = self._load_canon(ecg_id, t)
                tgt = self._load_canon(ecg_id, "0001")
                return src, tgt, False

        # Fallback identity
        tgt = self._load_canon(ecg_id, "0001")
        src = tgt.clone()
        return src, tgt, True


if __name__ == "__main__":
    # Test the ECGSegDataset
    TRAIN_ROOT = "../../../data/sample/"  # <-- change
    all_ids = sorted([d for d in os.listdir(TRAIN_ROOT) if os.path.isdir(os.path.join(TRAIN_ROOT, d))])

    dataset = PairedScanifyDataset(
        train_root=TRAIN_ROOT,
        ids=all_ids,
        cache_dir=f"{TRAIN_ROOT}/cache_ecgseg_stageA",
    )

    x, y,is_identity = dataset[0]
    print(f"x shape: {x.shape}, y shape: {y.shape}, x and y identical ? {is_identity}")
