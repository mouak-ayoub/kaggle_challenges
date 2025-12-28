from pathlib import Path

import pandas as pd
import config
from mask_to_dataframe import decode_all_leads, get_model, get_logits_from_image

model_path = Path("..") / "models" / "best_unet_resnet34_halfres_thickness_8.pt"
base_path = Path("..") / "data" / "test"
ecg_metadata_path = Path("..") / "data" / "test.csv"
submission_path = Path("..") / "data" / "submission"

LAYOUT = config.LAYOUT
PX_PER_MM_X, PX_PER_MM_Y = config.PX_PER_MM_X, config.PX_PER_MM_Y

model = get_model(str(model_path))
meta = pd.read_csv(ecg_metadata_path)

img_ids = set(meta["id"].astype(str).tolist())

signals_all = {}
for ecg_id in img_ids:
    img_path = base_path / f"{ecg_id}.png"

    logits = get_logits_from_image(model, str(img_path))

    row = meta[meta["id"].astype(str) == str(ecg_id)]
    fs = int(row["fs"].iloc[0])

    signals, _signals_non_resampled = decode_all_leads(
        logits, LAYOUT, PX_PER_MM_Y, PX_PER_MM_X, fs
    )

    # Rename rhythm lead to standard Lead II
    signals["II"] = signals.pop("II_rhythm")
    signals_all[ecg_id] = signals

lead_order = LAYOUT["lead_names"][:-1]

rows = []
for base_id, df in signals_all.items():
    for lead in lead_order:
        series = df[lead].astype(float)
        for row_id, value in enumerate(series):
            rows.append({"id": f"{base_id}_{row_id}_{lead}", "value": float(value)})

submission_df = pd.DataFrame(rows, columns=["id", "value"])
submission_df.sort_values("id", inplace=True)
submission_df.to_csv(submission_path.with_suffix(".csv"), index=False)
submission_df.to_parquet(submission_path.with_suffix(".parquet"), index=False)
