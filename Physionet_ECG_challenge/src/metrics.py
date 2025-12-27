import numpy as np

def align_by_xcorr(pred, truth, fs, max_shift_s=0.2):
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)

    N = min(len(pred), len(truth))
    pred = pred[:N]
    truth = truth[:N]

    pred0 = pred - np.mean(pred)
    truth0 = truth - np.mean(truth)

    max_lag = int(round(max_shift_s * fs))
    if max_lag < 1 or N < 10:
        return pred, truth, 0

    best_lag = 0
    best_score = -np.inf

    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            a = pred0[-lag:]
            b = truth0[:N + lag]
        elif lag > 0:
            a = pred0[:N - lag]
            b = truth0[lag:]
        else:
            a = pred0
            b = truth0

        if len(a) < 10:
            continue

        score = np.dot(a, b)
        if score > best_score:
            best_score = score
            best_lag = lag

    lag = best_lag
    if lag < 0:
        pred_al = pred[-lag:]
        truth_al = truth[:len(pred_al)]
    elif lag > 0:
        pred_al = pred[:N - lag]
        truth_al = truth[lag:lag + len(pred_al)]
    else:
        pred_al = pred
        truth_al = truth

    return pred_al, truth_al, lag


def ecg_snr_db(
    signals_pred,          # dict: lead -> np.array
    truth_df,              # pd.DataFrame with columns = leads
    fs,
    lead_names,
    truth_slicing_factors=None,   # dict lead -> (lo_frac, hi_frac), optional
    max_shift_s=0.2,
    eps=1e-12,
    return_details=False,
):
    """
    Returns ECG-level SNR in dB by summing signal/noise power across leads.

    signals_pred: dict[lead] = predicted 1D array (already at fs)
    truth_df: ground-truth dataframe
    truth_slicing_factors: if truth stores leads in segments of the 10s row,
                           provide (lo_frac, hi_frac) per lead.
    """
    N_truth = len(truth_df)

    total_sig_power = 0.0
    total_err_power = 0.0

    details = {}  # optional per-lead diagnostics

    for lead in lead_names:
        pred = np.asarray(signals_pred[lead], dtype=np.float64)
        truth_lead_name = lead if lead != "II_rhythm" else "II"
        # slice truth if needed
        if truth_slicing_factors is not None:
            lo_f, hi_f = truth_slicing_factors[lead]
            lo = int(round(lo_f * N_truth))
            hi = int(round(hi_f * N_truth))
            truth = np.asarray(truth_df[truth_lead_name].values[lo:hi], dtype=np.float64)
        else:
            truth = np.asarray(truth_df[truth_lead_name].values, dtype=np.float64)

        # align in time (±0.2s)
        pred_al, truth_al, lag = align_by_xcorr(pred, truth, fs, max_shift_s=max_shift_s)

        # remove constant vertical offset after alignment
        dc = np.mean(truth_al - pred_al)
        pred_dc = pred_al + dc

        err = pred_dc - truth_al

        sig_power = np.sum(truth_al ** 2)
        err_power = np.sum(err ** 2)

        total_sig_power += sig_power
        total_err_power += err_power

        if return_details:
            rmse = np.sqrt(np.mean(err ** 2)) if len(err) else np.nan
            details[lead] = {
                "lag_samples": int(lag),
                "lag_seconds": float(lag / fs),
                "dc_mV": float(dc),
                "rmse_mV": float(rmse),
                "sig_power": float(sig_power),
                "err_power": float(err_power),
                "n": int(len(truth_al)),
            }

    snr_db = 10.0 * np.log10((total_sig_power + eps) / (total_err_power + eps))

    if return_details:
        return snr_db, details
    return snr_db
