import numpy as np
import matplotlib.pyplot as plt
import math


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
        signals_pred,  # dict: lead -> np.array
        truth_df,  # pd.DataFrame with columns = leads
        fs,
        lead_names,
        truth_slicing_factors=None,  # dict lead -> (lo_frac, hi_frac), optional
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


def plot_leads_compare_dynamic(
    df_src,
    df_pred,
    fs,
    lead_names,
    truth_slicing_factors=None,
    rhythm_lead="II_rhythm",
    ylim_mv=(-3.0, 3.0),
    show_grid=True,
    title="ECG (mV)",
    max_cols=3,
    is_test=False,
):
    """
    Dynamic lead plot.
    - Train/Val (is_test=False): plots GT + Pred (requires df_src and truth_slicing_factors).
    - Test (is_test=True or df_src is None): plots Pred only.

    lead_names: list of leads to plot (can be 1, many, or all 13).
    df_pred: can be a DataFrame or dict-like mapping lead -> array/Series.
    """

    # Auto-switch if df_src missing
    if df_src is None:
        is_test = True

    leads = list(lead_names)

    # ---- helpers ----
    def get_pred_segment(lead):
        s = df_pred[lead]
        return np.asarray(s.values if hasattr(s, "values") else s, dtype=np.float64)

    if not is_test:
        if truth_slicing_factors is None:
            raise ValueError("truth_slicing_factors must be provided when is_test=False.")
        N_truth_total = len(df_src)

        def get_truth_segment(lead):
            lo_f, hi_f = truth_slicing_factors[lead]
            lo = int(round(lo_f * N_truth_total))
            hi = int(round(hi_f * N_truth_total))
            lead_column = lead if lead != "II_rhythm" else "II"
            return np.asarray(df_src[lead_column].values[lo:hi], dtype=np.float64)
    else:
        def get_truth_segment(lead):
            return None  # unused

    # ---- plotting primitive ----
    def plot_one(ax, lead, big=False):
        y_pred = get_pred_segment(lead)

        if is_test:
            n = len(y_pred)
            t = np.arange(n) / fs
            ax.plot(t, y_pred, lw=1.2 if big else 1.0, label="Pred")
        else:
            y_true = get_truth_segment(lead)
            n = min(len(y_true), len(y_pred))
            t = np.arange(n) / fs
            ax.plot(t, y_true[:n], lw=1.2 if big else 1.0, label="GT")
            ax.plot(t, y_pred[:n], lw=1.2 if big else 1.0, alpha=0.8, label="Pred")

        ax.set_title(lead if not big else f"{lead} (Rhythm strip)", fontsize=11 if big else 10)
        ax.set_ylim(*ylim_mv)
        ax.axhline(0.0, lw=0.6 if big else 0.5)

        if show_grid:
            ax.grid(True, linewidth=0.6 if big else 0.5, alpha=0.5)

        ax.set_ylabel("mV", fontsize=10 if big else 9)

    # --- Case 1: single lead ---
    if len(leads) == 1:
        lead = leads[0]
        fig, ax = plt.subplots(figsize=(14, 4))
        plot_one(ax, lead, big=True)

        ax.set_xlabel("seconds")
        ax.legend(loc="upper right", fontsize=9)
        fig.suptitle(title, fontsize=14)
        plt.show()
        return

    # --- Multi-lead cases ---
    has_rhythm = (rhythm_lead in leads)
    small_leads = [l for l in leads if l != rhythm_lead]
    n_small = len(small_leads)

    ncols = min(max_cols, max(1, n_small))
    nrows = int(math.ceil(n_small / ncols))

    # Case 2: rhythm present -> grid + big bottom rhythm
    if has_rhythm:
        fig = plt.figure(figsize=(18, 4 + 2.8 * nrows))
        gs = fig.add_gridspec(
            nrows + 1, ncols,
            height_ratios=[1] * nrows + [1.6],
            hspace=0.45, wspace=0.25
        )

        for i, lead in enumerate(small_leads):
            r = i // ncols
            c = i % ncols
            ax = fig.add_subplot(gs[r, c])
            plot_one(ax, lead, big=False)

            # cosmetics
            if r < nrows - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("s", fontsize=9)
            if c != 0:
                ax.set_yticklabels([])

            if i == 0:
                ax.legend(loc="upper right", fontsize=8)

        ax = fig.add_subplot(gs[nrows, :])
        plot_one(ax, rhythm_lead, big=True)
        ax.set_xlabel("seconds", fontsize=10)
        ax.legend(loc="upper right", fontsize=9)

        fig.suptitle(title, fontsize=14)
        plt.show()
        return

    # Case 3: no rhythm -> compact grid only
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 3.0 * nrows), squeeze=False)
    axes = axes.flatten()

    for i, lead in enumerate(small_leads):
        ax = axes[i]
        plot_one(ax, lead, big=False)

        if (i // ncols) < (nrows - 1):
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("s", fontsize=9)

        if (i % ncols) != 0:
            ax.set_yticklabels([])

        if i == 0:
            ax.legend(loc="upper right", fontsize=8)

    # Hide unused axes
    for j in range(len(small_leads), len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()



def plot_nonresampled_signal(
        signal_nonresampled,  # 1D array in mV
        px_per_mm,
        lead_name="I",
        speed_mm_s=25.0,
        ylim_mv=(-3.0, 3.0),
        title="Non-resampled ECG signal (image-time)",
):

    signal_resampled_by_lead=signal_nonresampled[lead_name]
    # image-time sampling rate
    src_fs = px_per_mm * speed_mm_s

    t = np.arange(len(signal_resampled_by_lead)) / src_fs

    plt.figure(figsize=(12, 3))
    plt.plot(t, signal_resampled_by_lead, lw=1.2)
    plt.axhline(0.0, lw=0.5)

    plt.title(f"{title} — Lead {lead_name}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude (mV)")
    plt.ylim(*ylim_mv)
    plt.grid(True, linewidth=0.5, alpha=0.5)

    plt.show()


def fit_gain_offset(pred, truth, eps=1e-12):
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)

    p = pred - pred.mean()
    t = truth - truth.mean()

    a = (p @ t) / ((p @ p) + eps)
    b = truth.mean() - a * pred.mean()
    return a, b

