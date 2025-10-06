import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit

from plot_spectrum import plot_spectrum

# -----------------------------
# Model functions
# -----------------------------
def lorentzian(H, H_res, dH, amp, offset):
    # Derivative-like Lorentzian from earlier code
    return amp * (dH * (H - H_res)) / ((H - H_res)**2 + (dH/2)**2)**2 + offset

def asymmetric_lorentzian(H, A, B, H0, dH, C, D):
    # Absorption (symmetric) + dispersion (antisymmetric) + linear baseline
    denom = (H - H0)**2 + dH**2
    symm = A * (dH**2) / denom
    asymm = B * (dH * (H - H0)) / denom
    return symm + asymm + C + D * H

def double_asymmetric_lorentzian(H, A1, B1, H1, dH1, A2, B2, H2, dH2, C, D):
    denom1 = (H - H1)**2 + dH1**2
    symm1 = A1 * (dH1**2) / denom1
    asym1 = B1 * (dH1 * (H - H1)) / denom1

    denom2 = (H - H2)**2 + dH2**2
    symm2 = A2 * (dH2**2) / denom2
    asym2 = B2 * (dH2 * (H - H2)) / denom2

    return symm1 + asym1 + symm2 + asym2 + C + D * H

# -----------------------------
# Helpers for peak detection
# -----------------------------
def _odd_window(n):
    return n if (n % 2 == 1) else (n + 1)

def estimate_main_zero_crossing(H, y):
    """
    Estimate main resonance zero-crossing for derivative-like lines.
    Uses sign changes and proximity to the largest |y|.
    """
    N = len(y)
    if N < 3:
        return H[np.argmax(np.abs(y))]

    idx_abs_max = int(np.argmax(np.abs(y)))
    sign_changes = np.where(np.signbit(y[1:]) != np.signbit(y[:-1]))[0]
    if len(sign_changes) == 0:
        return H[idx_abs_max]

    # zero-crossing pair nearest to strongest feature
    i = int(sign_changes[np.argmin(np.abs(sign_changes - idx_abs_max))])
    x0, x1 = H[i], H[i + 1]
    y0, y1 = y[i], y[i + 1]
    if (y1 - y0) == 0:
        return 0.5 * (x0 + x1)
    t = -y0 / (y1 - y0)
    return x0 + t * (x1 - x0)

def estimate_extrema_around_H0(H, y, H0):
    """
    Find prominent extrema flanking H0 (left min, right max by default).
    """
    idx_center = np.searchsorted(H, H0)
    if idx_center <= 0 or idx_center >= len(H) - 1:
        # Fallback: global min and max split by center index
        left_idx = np.argmin(y[:max(1, idx_center)])
        right_idx = np.argmax(y[idx_center:]) + idx_center
        return left_idx, right_idx

    left_slice = y[:idx_center]
    right_slice = y[idx_center:]
    if len(left_slice) == 0 or len(right_slice) == 0:
        return int(np.argmin(y)), int(np.argmax(y))
    left_idx = int(np.argmin(left_slice))           # most negative
    right_idx = int(np.argmax(right_slice)) + idx_center  # most positive
    return left_idx, right_idx

def estimate_dH_from_extrema(H, idx_left, idx_right):
    """
    For derivative of a Lorentzian: extrema at H0 ± dH/√3 => separation S ≈ 2*dH/√3
    => dH ≈ S * √3 / 2
    """
    S = np.abs(H[idx_right] - H[idx_left])
    return (np.sqrt(3.0) / 2.0) * S

def find_secondary_peak(H, y_abs, H0, mask_halfwidth):
    """
    Detect a peripheral secondary peak (in |y|) outside a central mask around H0.
    Returns index in full array or None.
    """
    mask = (H < (H0 - mask_halfwidth)) | (H > (H0 + mask_halfwidth))
    if not np.any(mask):
        return None
    peaks, props = find_peaks(y_abs[mask], prominence=0.15 * np.max(y_abs))
    if len(peaks) == 0:
        return None
    full_indices = np.where(mask)[0][peaks]
    prominences_full = props.get('prominences', np.ones_like(peaks))
    return int(full_indices[int(np.argmax(prominences_full))])

# -----------------------------
# Main fitting function
# -----------------------------
def fit_spectrum(
    processed_file=None,
    sample="AZ5",
    results_root="results_roots",
    fitting_model='asymmetric_lorentzian',     # default to asymmetric (as requested)
    aggregate='median',                        # 'median' | 'mean' | None/'none'/False
    save_plot=True,
    show_plot=True,
    return_params=True,
    plot_suffix="fit",
    # Peak detection controls
    smooth_for_peaks=True,
    smooth_frac=0.03,                          # fraction of points used for Savitzky–Golay window
    smooth_poly=2,
    prominence_frac=0.25,                      # as fraction of max(|y|)
    distance_frac=0.03,                        # as fraction of N
    detect_secondary=True,
    # Iterative: first lock ΔH near initial, then free
    iterative=True,
    width_lock_factor=0.4,                     # ±fraction around initial dH
    # Optional: restrict single-peak fit to a window around main resonance
    fit_window_factor=None,                    # e.g., 2.5 -> |H-H0| <= 2.5*dH_init
    # plotting passthrough for plot_spectrum(...)
    **plot_kwargs
):
    """
    Asymmetric FMR fitting with robust peak and width initialization:
    - Main zero-crossing anchor, ΔH from extrema spacing
    - Initial A,B,C,D via linear least-squares on asymmetric basis (very stable)
    - Optional secondary peak detection (for double-asymmetric)
    - Optional iterative fitting (lock ΔH then free)
    """
    # --- 0) Auto-detect latest processed file if needed ---
    if processed_file is None:
        if sample is None:
            raise ValueError("If processed_file is not given, sample name must be provided.")
        search_pattern = os.path.join(results_root, sample, '*_processed.csv')
        processed_files = glob.glob(search_pattern)
        if not processed_files:
            raise FileNotFoundError(f"No processed files found in {os.path.join(results_root, sample)}")
        processed_file = max(processed_files, key=os.path.getmtime)
        print(f"[fit_spectrum] No file specified. Using the latest: {os.path.basename(processed_file)}")

    # --- 1) Load + aggregate ---
    df = pd.read_csv(processed_file)
    if ("H (T)" not in df.columns) or ("dP/dH" not in df.columns):
        raise ValueError("CSV must contain columns: 'H (T)' and 'dP/dH'.")
    H_raw = df["H (T)"].values
    y_raw = df["dP/dH"].values

    agg_off = (aggregate is None) or (str(aggregate).lower() in ["none", "false"])
    if agg_off:
        H = H_raw
        y = y_raw
    else:
        agg_func = {"mean": "mean", "median": "median"}.get(str(aggregate).lower())
        if not agg_func:
            raise ValueError("aggregate must be 'mean', 'median', None, 'none' or False")
        df_agg = df.groupby("H (T)", as_index=False)["dP/dH"].agg(agg_func)
        H = df_agg["H (T)"].values
        y = df_agg["dP/dH"].values

    # sort + basic checks
    idx = np.argsort(H)
    H = H[idx]
    y = y[idx]
    if len(np.unique(H)) < 3:
        raise ValueError("Not enough unique H values for fitting.")
    H_min, H_max = float(np.min(H)), float(np.max(H))
    width_range = H_max - H_min
    if width_range <= 0:
        raise ValueError("Invalid field range: bounds would be zero or negative.")

    # --- 2) Robust detection for H0 and dH ---
    y_proc = y.copy()
    N = len(H)
    if smooth_for_peaks and N >= 7:
        win = max(5, int(np.ceil(smooth_frac * N)))
        win = _odd_window(win)
        win = min(win, N - 1 if (N % 2 == 0) else N)  # ensure <= N and odd
        if win < 5:
            win = 5
        if win >= N:
            win = N - 1 if (N % 2 == 1) else N - 2
            if win < 5:
                win = 5
        y_proc = savgol_filter(y, window_length=_odd_window(win), polyorder=min(smooth_poly, 3))

    H0_est = estimate_main_zero_crossing(H, y_proc)
    iL, iR = estimate_extrema_around_H0(H, y_proc, H0_est)
    dH_init = estimate_dH_from_extrema(H, iL, iR)
    dH_init = float(np.clip(dH_init, 1e-6, width_range))
    C_init = float(np.median(y))
    D_init = 0.0

    # Optionally narrow to a window around H0 for single-peak fits
    fit_mask = np.ones_like(H, dtype=bool)
    if (fitting_model in ["asymmetric_lorentzian", "asymmetric"]) and (fit_window_factor is not None):
        H_left = H0_est - fit_window_factor * dH_init
        H_right = H0_est + fit_window_factor * dH_init
        mask = (H >= H_left) & (H <= H_right)
        if np.sum(mask) >= 5:
            fit_mask = mask
    # Apply mask if different
    H_fit = H[fit_mask]
    y_fit = y[fit_mask]

    # --- 2b) Initialize A,B (and C,D) via linear regression on basis (given H0,dH) ---
    def basis_asym(Hv, H0v, dHv):
        denom = (Hv - H0v)**2 + dHv**2
        symm = (dHv**2) / denom
        asym = (dHv * (Hv - H0v)) / denom
        return symm, asym

    sym_b, asym_b = basis_asym(H_fit, H0_est, dH_init)
    X = np.column_stack([sym_b, asym_b, np.ones_like(H_fit), H_fit])  # [A, B, C, D]
    # Solve least squares X*beta ≈ y_fit
    beta, *_ = np.linalg.lstsq(X, y_fit, rcond=None)
    A_init, B_init, C_init_reg, D_init_reg = [float(b) for b in beta]
    # prefer regression baseline if reasonable
    C_init = C_init_reg
    D_init = D_init_reg

    # --- 2c) (Optional) detect secondary feature for double-asymmetric ---
    secondary_idx = None
    if detect_secondary:
        # define a central mask halfwidth from data spread
        central_halfwidth = max(0.02 * width_range, np.median(np.diff(H)) * 5) * 1.2
        secondary_idx = find_secondary_peak(H, np.abs(y_proc), H0_est, central_halfwidth)

    # --- 3) Build p0 / bounds for chosen model ---
    if fitting_model.lower() in ["asymmetric", "asymmetric_lorentzian"]:
        model_func = asymmetric_lorentzian
        p0 = [A_init, B_init, H0_est, dH_init, C_init, D_init]
        bounds = ([-np.inf, -np.inf, H_min, 0.0, -np.inf, -np.inf],
                  [ np.inf,  np.inf, H_max, width_range,  np.inf,  np.inf])
        param_names = ["A", "B", "H0", "dH", "C", "D"]

    elif fitting_model.lower() == "double_asymmetric_lorentzian":
        # second resonance init
        if secondary_idx is None:
            # fallback: pick strongest far-from-center or mirror side; keep amplitudes small
            idx_far = int(np.argmax(np.abs(y_proc)))
            H2 = float(H[idx_far])
            if abs(H2 - H0_est) < 0.5 * dH_init:
                # push it to an edge if it's too close
                H2 = float(H_max) if (H0_est - H_min) < (H_max - H0_est) else float(H_min)
        else:
            H2 = float(H[secondary_idx])
        dH2 = float(np.clip(dH_init, 1e-6, width_range))

        # Solve linear amps for both components given (H1,dH1,H2,dH2)
        sym1, asym1 = basis_asym(H_fit, H0_est, dH_init)
        sym2, asym2 = basis_asym(H_fit, H2, dH2)
        X2 = np.column_stack([sym1, asym1, sym2, asym2, np.ones_like(H_fit), H_fit])  # [A1,B1,A2,B2,C,D]
        beta2, *_ = np.linalg.lstsq(X2, y_fit, rcond=None)
        A1, B1, A2, B2, C2, D2 = [float(b) for b in beta2]

        model_func = double_asymmetric_lorentzian
        p0 = [A1, B1, H0_est, dH_init, A2, B2, H2, dH2, C2, D2]
        bounds = ([-np.inf, -np.inf, H_min, 0.0, -np.inf, -np.inf, H_min, 0.0, -np.inf, -np.inf],
                  [ np.inf,  np.inf, H_max, width_range,  np.inf,  np.inf, H_max, width_range,  np.inf,  np.inf])
        param_names = ["A1","B1","H1","dH1","A2","B2","H2","dH2","C","D"]

    else:
        raise NotImplementedError(f"Model '{fitting_model}' not implemented. "
                                  "Use 'asymmetric_lorentzian' or 'double_asymmetric_lorentzian'.")

    # --- 4) Curve fitting with optional iterative width lock ---
    def _fit_with(p0_local, bounds_local, maxfev=30000):
        try:
            popt_local, pcov_local = curve_fit(model_func, H_fit, y_fit, p0=p0_local, bounds=bounds_local, maxfev=maxfev)
            perr_local = np.sqrt(np.diag(pcov_local))
            return popt_local, perr_local, True, ""
        except Exception as e:
            return np.array(p0_local, dtype=float), np.full(len(p0_local), np.nan), False, str(e)

    popt, perr, fit_success, fit_err = None, None, False, ""
    if iterative:
        p0_lock = list(p0)
        bL, bU = [np.array(b, dtype=float).copy() for b in bounds]

        def lock_width(idx_dH, p0_list):
            dH0 = float(p0_list[idx_dH])
            half_span = max(1e-6, width_lock_factor * dH0)
            bL[idx_dH] = max(bL[idx_dH], dH0 - half_span)
            bU[idx_dH] = min(bU[idx_dH], dH0 + half_span)

        if fitting_model.lower() in ["asymmetric", "asymmetric_lorentzian"]:
            lock_width(param_names.index("dH"), p0_lock)
        elif fitting_model.lower() == "double_asymmetric_lorentzian":
            lock_width(param_names.index("dH1"), p0_lock)
            lock_width(param_names.index("dH2"), p0_lock)

        popt_lock, perr_lock, ok_lock, err_lock = _fit_with(p0_lock, (bL, bU))
        if ok_lock:
            popt, perr, fit_success, fit_err = _fit_with(popt_lock, bounds)
        else:
            popt, perr, fit_success, fit_err = _fit_with(p0, bounds)
    else:
        popt, perr, fit_success, fit_err = _fit_with(p0, bounds)

    if not fit_success:
        print(f"[fit_spectrum] Fit failed: {fit_err}")

    # --- 5) Fit quality ---
    if fit_success:
        # Evaluate fit over the full x-range, not just fit window!
        fit_curve_full = model_func(H, *popt)
        residuals = y - fit_curve_full
        ss_res = float(np.sum(residuals**2))
        ss_tot = float(np.sum((y - np.mean(y))**2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
        rmse = float(np.sqrt(ss_res / len(H)))
    else:
        fit_curve_full = np.full_like(H, np.nan)
        r2, rmse = np.nan, np.nan

    # --- 6) Plot via plot_spectrum, overlay fit (over full H) ---
    filename = os.path.basename(processed_file)
    pattern = r"(?P<sample>.+)_f(?P<f>[\d\.]+)GHz_m(?P<db>-?\d+)dB"
    match = re.search(pattern, filename)
    sample_name, f_txt, db_txt = "Sample", "", ""
    if match:
        sample_name = match.group("sample")
        f_txt = match.group("f")
        db_txt = match.group("db")

    fig, ax = plot_spectrum(
        processed_file=processed_file,
        sample=sample_name,
        results_root=results_root,
        show=False,
        save_png=False,
        save_svg=False,
        aggregate=aggregate,
        return_ax=True,
        **plot_kwargs
    )
    if fit_success:
        ax.plot(
            H, fit_curve_full, '-',
            color=plot_kwargs.get("color_fit", "orange"),
            linewidth=plot_kwargs.get("linewidth_fit", 2.5),
            label="Fit"
        )
    title = f"{sample_name}, f={f_txt} GHz, {db_txt} dB\n{fitting_model.replace('_',' ').title()}"
    if fit_success:
        try:
            if fitting_model.lower().startswith("double"):
                h1 = popt[param_names.index("H1")]
                h2 = popt[param_names.index("H2")]
                title += f"\n$H_1$={h1:.4f} T, $H_2$={h2:.4f} T, R²={r2:.3f}"
            else:
                h0 = popt[param_names.index("H0")]
                title += f"\n$H_0$={h0:.4f} T, R²={r2:.3f}"
        except Exception:
            title += f"\nR²={r2:.3f}"
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    basefolder = os.path.dirname(processed_file)
    out_base = filename.replace("_processed.csv", f"_{plot_suffix}")
    png_path = os.path.join(basefolder, out_base + ".png")
    svg_path = os.path.join(basefolder, out_base + ".svg")
    if save_plot:
        fig.savefig(png_path, dpi=300)
        fig.savefig(svg_path)
    print(f"Plot saved: {png_path} and {svg_path}")

    # --- 7) Save numeric results ---
    results_dict = {name: val for name, val in zip(param_names, popt)}
    results_dict.update({f"err_{name}": err for name, err in zip(param_names, perr)})
    results_dict.update({
        "R2": r2, "RMSE": rmse, "success": fit_success, "model": fitting_model
    })
    pd.DataFrame([results_dict]).to_csv(os.path.join(basefolder, out_base + "_results.csv"), index=False)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    if return_params:
        return results_dict