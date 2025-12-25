

import os, glob, numpy as np
from tqdm import tqdm
from src.fine_tune import process_one_ecg_folder

def get_bad_masks(train_root:str,delete=False):

  npz_files = glob.glob(os.path.join(train_root, "*", "mask-*.npz"))

  bad = []
  for f in tqdm(npz_files, desc="Scanning npz files"):
      try:
          with np.load(f) as z:
              _ = z["masks"][:]   # force full read (triggers CRC if broken)
      except Exception as e:
          bad.append((f, repr(e)))

  print("Total npz:", len(npz_files))
  print("Bad npz:", len(bad))
  for f, e in bad[:20]:
      print("BAD:", f, e)
  print(f"The corrupted masks will be deleted? {delete}")
  if delete:
    for f, _ in bad:
      os.remove(f)
    print("Deleted", len(bad), "corrupted files")

import os, glob

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


