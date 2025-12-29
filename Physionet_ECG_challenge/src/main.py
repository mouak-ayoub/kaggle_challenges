import numpy as np
import pandas as pd

import config
from mask_generation import process_all_parallel
from mask_to_dataframe import  get_logits_from_image, decode_all_leads
from metrics import plot_leads_compare_dynamic, ecg_snr_db, fit_gain_offset, align_by_xcorr, plot_nonresampled_signal
import segmentation_models_pytorch as smp
import torch

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

if __name__ == "__main__":
    preprocess=False
    inference=True
    # image
    ecg_id = "10140238"
    is_test_dataset = False

    if preprocess:
        train_root_path = r"../data/train"  # change if needed
        process_all_parallel(train_root_path, limit=5,print_overlay=True,thickness=2, workers=2)

    if inference:
        model_path="../models/best_unet_resnet34_halfres_thickness_8.pt"
        model = get_model(model_path)


        if is_test_dataset:
            base_path="../data/test"
            img_path = f"{base_path}/{ecg_id}.png"
            ecg_metadata_path = "../data/test.csv"

        else:
            base_path="../data/sample"
            img_path = f"{base_path}/{ecg_id}/{ecg_id}-0012.png"
            ecg_metadata_path = "../data/train.csv"


        logits = get_logits_from_image(model, img_path)

        # Transform masks to signals

        LAYOUT = config.LAYOUT
        truth_slicing_factors = config.TRUTH_SLICING_FACTORS
        meta = pd.read_csv(ecg_metadata_path)
        row = meta[meta['id'].astype(str) == str(ecg_id)]
        fs = int(row['fs'].iloc[0])

        PX_PER_MM_X,PX_PER_MM_Y = config.PX_PER_MM_X,config.PX_PER_MM_Y


        signals, signals_non_resmapled = decode_all_leads(logits, LAYOUT, PX_PER_MM_X, PX_PER_MM_Y, fs)

        if not is_test_dataset:
            truth = pd.read_csv(f"{base_path}/{ecg_id}/{ecg_id}.csv")
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

            print(f"ECG SNR: {snr_ecg_db:.2f} dB")
            print(per_lead)  # diagnostics for Lead I
            for lead, d in per_lead.items():
                snr_lead = 10 * np.log10((d["sig_power"] + 1e-12) / (d["err_power"] + 1e-12))
                lead_truth_col = lead if lead != "II_rhythm" else "II"
                lo_f, hi_f = truth_slicing_factors[lead]
                N_truth = len(truth)
                lo = int(round(lo_f * N_truth))
                hi = int(round(hi_f * N_truth))
                truth_lead = np.asarray(truth[lead_truth_col].values[lo:hi], dtype=np.float64)
                pred_al, truth_al, lag = align_by_xcorr(signals[lead], truth_lead, fs, max_shift_s=0.2)
                a, b = fit_gain_offset(pred_al, truth_al)
                print(lead, snr_lead)
                print("gain a =", a, "offset b =", b)

        else:
            # plot_leads_compare_dynamic(truth, signals, fs, ["II_rhythm"], truth_slicing_factors)
            print("plot only generated signals for test data")
            plot_leads_compare_dynamic(None, signals, fs, LAYOUT["lead_names"], truth_slicing_factors)

        # Plot non resampled lead I
        plot_nonresampled_signal(signals_non_resmapled, PX_PER_MM_X, "II_rhythm")
