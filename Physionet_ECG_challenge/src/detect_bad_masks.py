
import os, glob
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from mask_generation import process_one_ecg_folder

def _check_npz_crc(path: str):
    try:
        with np.load(path) as z:
            # Force full decompression/read to catch CRC errors reliably
            _ = z["masks"][:]
        return None
    except Exception as e:
        return (path, repr(e))

def get_bad_masks_parallel(train_root: str, delete: bool = False, workers: int = 8):
    npz_files = glob.glob(os.path.join(train_root, "*", "mask-*.npz"))
    bad = []

    # Tip: too many workers can hurt due to disk contention; 4–8 is usually best in Colab.
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_check_npz_crc, f) for f in npz_files]

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Scanning npz files"):
            res = fut.result()
            if res is not None:
                bad.append(res)

    print("Total npz:", len(npz_files))
    print("Bad npz:", len(bad))
    for f, e in bad[:20]:
        print("BAD:", f, e)

    print(f"The corrupted masks will be deleted? {delete}")
    if delete and bad:
        for f, _ in bad:
            try:
                os.remove(f)
            except FileNotFoundError:
                pass
        print("Deleted", len(bad), "corrupted files")

    return bad


def regenerate_missing(train_root):
    ecg_dirs = sorted([d for d in glob.glob(os.path.join(train_root, "*")) if os.path.isdir(d)])
    missing = []
    for d in ecg_dirs:
        if not glob.glob(os.path.join(d, "mask-*.npz")):
            missing.append(d)

    print("Missing masks:", len(missing))
    for d in tqdm(missing, desc="Regenerating"):
        ok, msg = process_one_ecg_folder(d)
        if not ok:
            print("FAIL:", d, msg)


