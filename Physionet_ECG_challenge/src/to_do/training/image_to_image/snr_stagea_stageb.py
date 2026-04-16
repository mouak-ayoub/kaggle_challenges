import os
import pandas as pd
from tabulate import tabulate

import torch
import segmentation_models_pytorch as smp

from to_do.config import config
from to_do.contour_detection.hough_transform import preprocess_for_model
from to_do.mask_generation.mask_generation import process_all_parallel
from to_do.training.image_to_mask.mask_to_dataframe import get_logits_from_image, decode_all_leads
from to_do.training.image_to_mask.metrics import (
    plot_leads_compare_dynamic,
    ecg_snr_db,
    plot_nonresampled_signal,
)

device = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------
# Helpers: robust ckpt loading
# ----------------------------
def _extract_state_dict(ckpt_obj):
    """
    Handles checkpoints saved as:
      - {"model": state_dict, ...}
      - state_dict directly
    """
    if isinstance(ckpt_obj, dict) and "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
        return ckpt_obj["model"]
    return ckpt_obj


def get_stageA(stageA_path):
    """
    Stage A: hard scan -> clean-like scan (1 channel)
    """
    stageA = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,  # IMPORTANT: None at inference
        in_channels=1,
        classes=1,
        activation=None,
    ).to(device)

    ckpt = torch.load(stageA_path, map_location=device)
    sd = _extract_state_dict(ckpt)
    stageA.load_state_dict(sd, strict=True)
    stageA.eval()
    return stageA


def get_stageB(stageB_path, K=13):
    """
    Stage B: clean scan -> multi-channel masks (K channels)
    """
    stageB = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,  # IMPORTANT: None at inference
        in_channels=1,
        classes=K,
        activation=None,
    ).to(device)

    ckpt = torch.load(stageB_path, map_location=device)
    sd = _extract_state_dict(ckpt)
    stageB.load_state_dict(sd, strict=True)
    stageB.eval()
    return stageB


@torch.inference_mode()
def run_stageA_np(stageA, img_preprocessed_2d):
    """
    img_preprocessed_2d: numpy float array [H,W] in [0,1]
    returns: numpy float array [H,W] in [0,1]
    """
    if img_preprocessed_2d.ndim != 2:
        raise ValueError(f"Expected 2D [H,W], got {img_preprocessed_2d.shape}")

    x = torch.from_numpy(img_preprocessed_2d).float().to(device)[None, None, :, :]  # [1,1,H,W]
    y = torch.sigmoid(stageA(x)).clamp(0, 1)  # [1,1,H,W]
    return y[0, 0].float().cpu().numpy()


def main():
    # ----------------------------
    # Settings
    # ----------------------------
    preprocess = False
    inference = True
    is_test_dataset = False

    ecg_id = "10140238"
    draw_comparaison = False
    debug = False

    # If test: usually only one image per id
    allowed_types = {"0001"} if is_test_dataset else config.training_allowed_types

    # Paths (adjust if needed)
    if is_test_dataset:
        base_path = "../../../../data/test"
        ecg_metadata_path = "../../../../data/test.csv"
        img_path_tpl = os.path.join(base_path, "{ecg_id}.png")
    else:
        base_path = "../../../../data/sample"
        ecg_metadata_path = "../../../../data/train.csv"
        img_path_tpl = os.path.join(base_path, "{ecg_id}", "{ecg_id}-{scan_type}.png")

    # Optional: preprocess all masks (if you still need it)
    if preprocess and (not is_test_dataset):
        train_path = base_path
        process_all_parallel(train_path, print_overlay=True, thickness=config.THICKNESS, workers=1)

    if not inference:
        return

    # ----------------------------
    # Load models
    # ----------------------------
    stageA_path = "../../../../models/best_stageA_scanifier.pt"
    stageB_path = "../../../../models/best_unet_resnet34_halfres_all_types.pt"

    stageA = get_stageA(stageA_path)
    stageB = get_stageB(stageB_path, K=13)

    # ----------------------------
    # Metadata / constants
    # ----------------------------
    meta = pd.read_csv(ecg_metadata_path)
    row = meta[meta["id"].astype(str) == str(ecg_id)]
    if len(row) == 0:
        raise ValueError(f"ECG id {ecg_id} not found in metadata {ecg_metadata_path}")
    fs = int(row["fs"].iloc[0])

    LAYOUT = config.LAYOUT
    truth_slicing_factors = config.TRUTH_SLICING_FACTORS
    PX_PER_MM_X, PX_PER_MM_Y = config.PX_PER_MM_X, config.PX_PER_MM_Y

    snr_map = {}

    # ----------------------------
    # Loop scan types
    # ----------------------------
    for scan_type in sorted(list(allowed_types)):
        print(f"\nProcessing ECG ID: {ecg_id} Type: {scan_type}")

        if is_test_dataset:
            img_path = img_path_tpl.format(ecg_id=ecg_id)
        else:
            img_path = img_path_tpl.format(ecg_id=ecg_id, scan_type=scan_type)

        if not os.path.exists(img_path):
            print("  -> missing:", img_path)
            continue

        # 1) Preprocess/rectify (same as Stage B only)
        img_preprocessed = preprocess_for_model(
            img_path,
            final_h=config.H_T,
            final_w=config.W_T,
            rectify=True,
        )
        # img_preprocessed should be float [H,W] in [0,1]

        # 2) Stage A scanify (OPTIONAL bypass for 0001)
        # You can keep 0001 untouched to avoid any possible degradation.
        img_for_stageB = run_stageA_np(stageA, img_preprocessed)

        # 3) Stage B segmentation logits
        logits = get_logits_from_image(stageB, img_for_stageB, device)

        # 4) Decode masks -> signals
        signals, signals_non_resampled = decode_all_leads(
            logits,
            LAYOUT,
            PX_PER_MM_X,
            PX_PER_MM_Y,
            fs,
        )

        # 5) Evaluate SNR if train/val with truth
        if not is_test_dataset:
            truth_path = os.path.join(base_path, ecg_id, f"{ecg_id}.csv")
            truth = pd.read_csv(truth_path)

            if draw_comparaison:
                plot_leads_compare_dynamic(truth, signals, fs, LAYOUT["lead_names"], truth_slicing_factors)

            snr_ecg_db, per_lead = ecg_snr_db(
                signals_pred=signals,
                truth_df=truth,
                fs=fs,
                lead_names=LAYOUT["lead_names"],
                truth_slicing_factors=truth_slicing_factors,
                max_shift_s=0.2,
                return_details=True,
            )
            snr_map[(ecg_id, scan_type)] = float(snr_ecg_db)

            if debug:
                print(f"ECG SNR: {snr_ecg_db:.2f} dB")
                # print(per_lead)  # detailed diagnostics
        else:
            print("Test dataset: plotting generated signals (no SNR).")
            plot_leads_compare_dynamic(None, signals, fs, LAYOUT["lead_names"], truth_slicing_factors)

        if draw_comparaison:
            plot_nonresampled_signal(signals_non_resampled, PX_PER_MM_X, "II_rhythm")

    # ----------------------------
    # Print summary table
    # ----------------------------
    if not is_test_dataset:
        print("\nSNR summary:\n")
        rows = []
        for (ecg_id_, scan_type_), snr_val in sorted(snr_map.items(), key=lambda kv: float(kv[1]), reverse=True):
            rows.append(
                (
                    str(ecg_id_),
                    str(scan_type_),
                    str(config.libelle_type.get(scan_type_, "")),
                    float(snr_val),
                )
            )

        headers = ("ECG ID", "Type", "Label", "SNR (dB)")
        print(tabulate(rows, headers=headers, tablefmt="grid", floatfmt=".2f"))


if __name__ == "__main__":
    main()
