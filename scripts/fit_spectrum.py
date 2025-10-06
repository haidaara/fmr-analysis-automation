import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import re
import glob

from plot_spectrum import plot_spectrum  # <-- Add this import

def lorentzian(H, H_res, dH, amp, offset):
    return amp * (dH * (H - H_res)) / ((H - H_res)**2 + (dH/2)**2)**2 + offset

def asymmetric_lorentzian(H, A, B, H0, dH, C, D):
    denom = (H - H0)**2 + dH**2
    symm = A * (dH**2) / denom
    asymm = B * (dH * (H - H0)) / denom
    return symm + asymm + C + D * H

def fit_spectrum(
    processed_file,
    sample="AZ5",
    results_root= "results_roots",
    fitting_model='lorentzian',
    aggregate='median',     
    save_plot=True,
    show_plot=True,
    return_params=True,
    plot_suffix="fit",
    # plotting style options (use the same as plot_spectrum or pass through)
    **plot_kwargs
):
    """
    FMR spectrum fitting with robust aggregation (optionally off), and enhanced plotting.
    Uses plot_spectrum to generate the base plot, then overlays the fit.
    """
    # --- 1. Load data and aggregate duplicates if requested
# --- Latest processed file detection if not given ---
    if processed_file is None:
        if sample is None:
            raise ValueError("If processed_file is not given, sample name must be provided.")
        search_pattern = os.path.join(results_root, sample, '*_processed.csv')
        processed_files = glob.glob(search_pattern)
        if not processed_files:
            raise FileNotFoundError(f"No processed files found in {os.path.join(results_root, sample)}")
        processed_file = max(processed_files, key=os.path.getmtime)
        print(f"[fit_spectrum] No file specified. Using the latest: {os.path.basename(processed_file)}")


    df = pd.read_csv(processed_file)
    if ("H (T)" not in df.columns) or ("dP/dH" not in df.columns):
        raise ValueError("CSV must contain columns: 'H (T)' and 'dP/dH'.")
    H_raw = df["H (T)"].values
    dPdH_raw = df["dP/dH"].values

    agg_off = (aggregate is None) or (str(aggregate).lower() == "none") or (aggregate is False)
    if agg_off:
        H_fit = H_raw
        dPdH_fit = dPdH_raw
    else:
        agg_func = {"mean": "mean", "median": "median"}.get(str(aggregate).lower())
        if not agg_func:
            raise ValueError("aggregate must be 'mean', 'median', None, 'none' or False")
        df_agg = df.groupby("H (T)", as_index=False)["dP/dH"].agg(agg_func)
        H_fit = df_agg["H (T)"].values
        dPdH_fit = df_agg["dP/dH"].values

    # --- 1.5. Sanity checks for bounds and data ---
    sort_idx = np.argsort(H_fit)
    H_fit = H_fit[sort_idx]
    dPdH_fit = dPdH_fit[sort_idx]
    unique_H = np.unique(H_fit)
    if len(unique_H) < 3:
        raise ValueError("Not enough unique H values for fitting.")
    H_min, H_max = np.min(H_fit), np.max(H_fit)
    width_range = H_max - H_min
    if width_range <= 0:
        raise ValueError("Invalid field range: bounds would be zero or negative.")

    # --- 2. Peak detection for initial guess (use H_fit/dPdH_fit)
    peaks, _ = find_peaks(np.abs(dPdH_fit), height=np.max(np.abs(dPdH_fit)) * 0.5)
    if len(peaks) == 0:
        peak_idx = np.argmax(np.abs(dPdH_fit))
        peaks = np.array([peak_idx])
    peak = peaks[0]
    H_peak = H_fit[peak]
    dH_init = width_range / 10
    offset_init = np.median(dPdH_fit)

    # --- 3. Model selection and initial params (use H_min/H_max) ---
    if fitting_model.lower() == "lorentzian":
        model_func = lorentzian
        amp_init = dPdH_fit[peak]
        p0 = [H_peak, dH_init, amp_init, offset_init]
        bounds = ([H_min, 0, -np.inf, -np.inf], [H_max, width_range, np.inf, np.inf])
        param_names = ["H_res", "dH", "amp", "offset"]
    elif fitting_model.lower() in ["asymmetric", "asymmetric_lorentzian"]:
        model_func = asymmetric_lorentzian
        A_init = np.max(dPdH_fit)
        B_init = 0
        H0_init = H_peak
        C_init = offset_init
        D_init = 0
        p0 = [A_init, B_init, H0_init, dH_init, C_init, D_init]
        bounds = (
            [-np.inf, -np.inf, H_min, 0, -np.inf, -np.inf],
            [np.inf, np.inf, H_max, width_range, np.inf, np.inf]
        )
        param_names = ["A", "B", "H0", "dH", "C", "D"]
    else:
        raise NotImplementedError(f"Model '{fitting_model}' not implemented.")

    # --- 4. Curve fitting
    try:
        popt, pcov = curve_fit(
            model_func, H_fit, dPdH_fit, p0=p0, bounds=bounds, maxfev=10000
        )
        perr = np.sqrt(np.diag(pcov))
        fit_success = True
    except Exception as e:
        print(f"Fit failed: {e}")
        popt = [np.nan] * len(p0)
        perr = [np.nan] * len(p0)
        fit_success = False

    # --- 5. Fit quality
    if fit_success:
        fit_curve = model_func(H_fit, *popt)
        residuals = dPdH_fit - fit_curve
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((dPdH_fit - np.mean(dPdH_fit)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        rmse = np.sqrt(ss_res / len(H_fit))
    else:
        fit_curve = np.full_like(H_fit, np.nan)
        r2, rmse = np.nan, np.nan

    # === 6. Use plot_spectrum for baseline plot, then overlay fit ===
    # Parse filename for metadata
    filename = os.path.basename(processed_file)
    pattern = r"(?P<sample>.+)_f(?P<f>[\d\.]+)GHz_m(?P<db>-?\d+)dB"
    match = re.search(pattern, filename)
    sample_name, f, db = "Sample", "", ""
    if match:
        sample_name = match.group("sample")
        f = match.group("f")
        db = match.group("db")
    # Call plot_spectrum and get fig, ax
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
    # Overlay the fit curve
    if fit_success:
        ax.plot(H_fit, fit_curve, '-', color='orange', linewidth=2.5, label="Fit")
        # Optionally bring fit to front
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)
    # Update title with fit info
    title_str = f"{sample_name}, f={f} GHz, {db} dB\n{fitting_model.title()} Fit"
    if fit_success:
        if fitting_model.lower().startswith("asym"):
            title_str += f"\n$H_0$={popt[2]:.4f} T, R²={r2:.3f}"
        else:
            title_str += f"\n$H_{{res}}$={popt[0]:.4f} T, R²={r2:.3f}"
    ax.set_title(title_str)
    plt.tight_layout()
    # Save plot
    basefolder = os.path.dirname(processed_file)
    out_base = filename.replace("_processed.csv", f"_{plot_suffix}")
    png_path = os.path.join(basefolder, out_base + ".png")
    svg_path = os.path.join(basefolder, out_base + ".svg")
    if save_plot:
        fig.savefig(png_path, dpi=300)
        fig.savefig(svg_path)
    
    print(f"Plot saved: {png_path} and {svg_path}")

    # --- 7. Save fit results (CSV/TXT)
    fit_results_path = os.path.join(basefolder, out_base + "_results.csv")
    results_dict = {name: val for name, val in zip(param_names, popt)}
    results_dict.update({f"err_{name}": err for name, err in zip(param_names, perr)})
    results_dict.update({
        "R2": r2,
        "RMSE": rmse,
        "success": fit_success,
        "model": fitting_model
    })
    

    pd.DataFrame([results_dict]).to_csv(fit_results_path, index=False)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)
        
    if return_params:
        return results_dict