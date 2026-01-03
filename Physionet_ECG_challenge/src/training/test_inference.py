model_path = "../../models/best_unet_resnet34_halfres_all_types.pt"
base_path = "../../data/test"
ecg_metadata_path = "../../data/test.csv"
output_path = "../../data/submission.csv"


from contour_detection.hough_transform import preprocess_for_model
from training.mask_to_dataframe import decode_all_leads, get_logits_from_image
from pathlib import Path
import pandas as pd
import csv
from config import config
import segmentation_models_pytorch as smp
import torch
import numpy as np




device = "cuda" if torch.cuda.is_available() else "cpu"


def get_model(model_path):
    K = 13  # number of channels

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,  # IMPORTANT: None for inference
        in_channels=1,
        classes=K,
    ).to(device)
    ckpt = torch.load(
        model_path,
        map_location=device
    )
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model

LAYOUT = config.LAYOUT
PX_PER_MM_X, PX_PER_MM_Y = config.PX_PER_MM_X, config.PX_PER_MM_Y

model = get_model(str(model_path))
meta = pd.read_csv(ecg_metadata_path)
meta["id"] = meta["id"].astype(str)

# build fs lookup once (faster than filtering df each time)
fs_map = dict(zip(meta["id"], meta["fs"]))

img_ids = sorted(fs_map.keys())
print(f"processing ecgs of size {len(img_ids)}")

lead_order = LAYOUT["lead_names"][:-1]  # 12 leads (exclude II_rhythm)
# You said competition wants II as the long rhythm strip:
# We'll overwrite "II" with "II_rhythm" output below, but keep lead list as 12 leads.

def force_length(arr, n):
    arr = np.asarray(arr, dtype=np.float32)
    if len(arr) >= n:
        return arr[:n]
    out = np.zeros(n, dtype=np.float32)
    out[:len(arr)] = arr
    return out

with open(output_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "value"])

    for ecg_id in img_ids:
        img_path = f"{base_path}/{ecg_id}.png"
        if not Path(img_path).exists():
            raise FileNotFoundError(f"Missing image: {img_path}")

        img_preprocessed = preprocess_for_model(img_path, final_h=config.H_T, final_w=config.W_T)

        logits = get_logits_from_image(model, img_preprocessed,device)
        fs = int(fs_map[ecg_id])

        signals, _signals_non_resampled = decode_all_leads(
            logits, LAYOUT,  PX_PER_MM_X,PX_PER_MM_Y, int(fs)
        )

        fs = float(fs_map[ecg_id])
        N_ii = int(fs * 10.0)     # floor(fs*10)
        N_short = int(fs * 2.5)   # floor(fs*2.5)

        # Lead II must be 10s rhythm
        ii = signals["II_rhythm"] if "II_rhythm" in signals else signals["II"]
        signals["II"] = force_length(ii, N_ii)

        # All other leads must be 2.5s
        for lead in lead_order:
            if lead == "II":
                continue
            signals[lead] = force_length(signals[lead], N_short)

        # Write immediately (no RAM accumulation)
        for lead in lead_order:
            series = signals[lead]
            # series might be list/np array/pd Series; iterate directly
            writer.writerows((f"{ecg_id}_{i}_{lead}", float(v)) for i, v in enumerate(series))

print("Wrote:", output_path)

