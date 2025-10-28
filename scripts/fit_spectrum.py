import argparse
import os
import glob
import yaml
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import logging
from lmfit import Model, Parameters

def load_config(yaml_path=None, cli_args=None):
    """
    Load YAML config, then merge CLI args honoring 'override_config_with_args' flag.
    - If override_config_with_args is True (default), CLI overrides YAML for provided keys.
    - If False, CLI only fills missing or empty values.
    """
    config = {}
    if yaml_path and os.path.isfile(yaml_path):
        with open(yaml_path, 'r') as f:
            cfg = yaml.safe_load(f) or {}
            config.update(cfg)
    override = True
    if "override_config_with_args" in config:
        override = bool(config.get("override_config_with_args"))
    if cli_args:
        for k, v in cli_args.items():
            if v in [None, "", [], {}]:
                continue
            if override or (k not in config or config.get(k) in [None, "", [], {}]):
                config[k] = v
    # Minimal defaults
    config.setdefault("aggregation", "median")
    if "plot" not in config or not isinstance(config.get("plot"), dict):
        config["plot"] = {}
    config["plot"].setdefault("plot_formats", ["png", "pdf"])
    if "output" not in config or not isinstance(config.get("output"), dict):
        config["output"] = {}
    config["output"].setdefault("save_csv", True)
    config["output"].setdefault("return_dict", True)
    config.setdefault("results_root", "results_roots")
    return config

def select_processed_file(config):
    input_file = config.get("input_file", "")
    if input_file and os.path.isfile(input_file):
        return input_file
    sample = config.get("sample", "")
    results_root = config.get("results_root", ".")
    if not sample:
        raise ValueError("Sample name must be provided if input_file is not set.")
    search_dir = os.path.join(results_root, sample)
    files = glob.glob(os.path.join(search_dir, "*_processed.csv"))
    if not files:
        raise FileNotFoundError(f"No processed files found in {search_dir}")
    return max(files, key=os.path.getmtime)

def parse_metadata_from_filename(filename):
    """
    Parse sample, frequency, and signed dB from a processed filename.
    Accepts '_m10dB' (negative) and '_10dB' (positive). Returns db as signed int.
    """
    import re
    fname = os.path.basename(filename)
    sample_match = re.match(r"(.+?)_f", fname)
    sample = sample_match.group(1) if sample_match else "Sample"
    freq_match = re.search(r"f([\d\.]+)G?Hz?", fname)
    freq = freq_match.group(1) if freq_match else ""
    # Prefer canonical '_m10dB' for negatives; allow '_10dB' positives
    db = None
    mneg = re.search(r"_m(\d+)dB", fname, flags=re.IGNORECASE)
    if mneg:
        db = -int(mneg.group(1))
    else:
        mpos = re.search(r"_(\d+)dB", fname, flags=re.IGNORECASE)
        if mpos:
            db = int(mpos.group(1))
    return {"sample": sample, "f": freq, "db": db}

def load_and_prepare_data(processed_file, config):
    df = pd.read_csv(processed_file)
    if "H (T)" not in df.columns or "dP/dH" not in df.columns:
        raise KeyError("Processed file must contain 'H (T)' and 'dP/dH' columns.")
    data = df[["H (T)", "dP/dH"]].sort_values("H (T)").reset_index(drop=True)
    H, dPdH = data["H (T)"].values, data["dP/dH"].values
    agg = str(config.get("aggregation", "none")).lower()
    if agg in ("median", "mean"):
        grouped = pd.DataFrame({"H (T)": H, "dP/dH": dPdH}).groupby("H (T)")
        dPdH = grouped["dP/dH"].median().values if agg == "median" else grouped["dP/dH"].mean().values
        H = np.sort(np.unique(H))
    preproc = config.get("preprocessing", {})
    if isinstance(preproc, dict) and preproc.get("baseline_correction", False):
        dPdH = dPdH - np.mean(dPdH)
    smoothing = preproc.get("smoothing", {}) if isinstance(preproc, dict) else {}
    if smoothing.get("method", "none") == "gaussian":
        from scipy.ndimage import gaussian_filter1d
        dPdH = gaussian_filter1d(dPdH, smoothing.get("window", 3))
    return H, dPdH

def detect_lobes(H, dPdH):
    peaks_max, _ = find_peaks(dPdH)
    peaks_min, _ = find_peaks(-dPdH)
    lobe_candidates = []
    for max_idx in peaks_max:
        if len(peaks_min) == 0:
            continue
        dist = np.abs(peaks_min - max_idx)
        min_idx = peaks_min[np.argmin(dist)]
        amp_diff = np.abs(dPdH[max_idx] - dPdH[min_idx])
        interval = np.abs(H[max_idx] - H[min_idx])
        lobe_candidates.append({
            "max_idx": max_idx,
            "min_idx": min_idx,
            "amp_diff": amp_diff,
            "interval": interval
        })
    lobes = sorted(
        lobe_candidates,
        key=lambda x: (-x["amp_diff"], x["interval"])
    )
    return lobes

def seed_lobe_parameters(H, dPdH, lobes, n_lobes):
    seeds = []
    for i in range(n_lobes):
        if i < len(lobes):
            max_idx, min_idx = lobes[i]["max_idx"], lobes[i]["min_idx"]
            center = (H[max_idx] + H[min_idx]) / 2
            # For derivative Lorentzian, extrema at x=±dH/√3 ⇒ dH = (√3/2)·|Hmax−Hmin|
            width = np.sqrt(3) / 2 * np.abs(H[max_idx] - H[min_idx])
            amplitude = dPdH[max_idx] - dPdH[min_idx]
            seeds.append({"center": center, "width": width, "amplitude": amplitude, "asymmetry": 0.0})
        else:
            center = np.mean(H)
            width = 0.5 * (np.max(H) - np.min(H))
            seeds.append({"center": center, "width": width, "amplitude": 0.0, "asymmetry": 0.0})
    return seeds

# ===== Derivative model functions (what we fit to) =====
# Notation: x = H − H0

def d_sym_lorentz_with_baseline(H, A, H0, dH, C, D):
    x = H - H0
    # d/dH of A·dH^2 / (x^2 + dH^2) plus linear baseline
    d_abs = -2.0 * A * (dH**2) * x / ((x**2 + dH**2)**2)
    return d_abs + C + D * H

def d_asym_lorentz(H, A, B, H0, dH, C, D):
    x = H - H0
    # Derivative of absorption
    d_abs = -2.0 * A * (dH**2) * x / ((x**2 + dH**2)**2)
    # Derivative of dispersion
    d_disp = B * dH * (dH**2 - x**2) / ((x**2 + dH**2)**2)
    return d_abs + d_disp + C + D * H

def d_double_asym_lorentz(H, A1, B1, H01, dH1, A2, B2, H02, dH2, C, D):
    x1 = H - H01
    x2 = H - H02
    d_abs1 = -2.0 * A1 * (dH1**2) * x1 / ((x1**2 + dH1**2)**2)
    d_disp1 = B1 * dH1 * (dH1**2 - x1**2) / ((x1**2 + dH1**2)**2)
    d_abs2 = -2.0 * A2 * (dH2**2) * x2 / ((x2**2 + dH2**2)**2)
    d_disp2 = B2 * dH2 * (dH2**2 - x2**2) / ((x2**2 + dH2**2)**2)
    return (d_abs1 + d_disp1 + d_abs2 + d_disp2) + C + D * H

def _apply_bounds(params: Parameters, bounds_cfg: dict, pname: str):
    """
    Apply bounds if provided and not null.
    Accepts [min, max] or {min:..., max:...}. Missing/null entries are ignored.
    """
    if not isinstance(bounds_cfg, dict):
        return
    if pname not in bounds_cfg:
        return
    b = bounds_cfg[pname]
    if b is None:
        return
    minb = maxb = None
    if isinstance(b, (list, tuple)) and len(b) == 2:
        minb, maxb = b
    elif isinstance(b, dict):
        minb, maxb = b.get("min"), b.get("max")
    if minb is not None:
        params[pname].set(min=float(minb))
    if maxb is not None:
        params[pname].set(max=float(maxb))

def fit_model(H, dPdH, seeds, config, model_type):
    """
    Fit dispatcher for DERIVATIVE models:
    - Lorentzain: A, H0, dH, C, D
    - aysmetric: A, B, H0, dH, C, D
    - double_lorentzain: A1, B1, H01, dH1, A2, B2, H02, dH2, C, D
    Returns:
      {
        "params": {name: value, ...},
        "errors": {name: stderr or "", ...},
        "success": bool,
        "r2": float,
        "redchi": float,
        "fit_curve": np.array
      }
    """
    model_key = str(model_type).strip().lower()
    if model_key not in ("lorentzain", "aysmetric", "double_lorentzain"):
        raise ValueError("Unknown model type. Use one of: 'Lorentzain', 'aysmetric', 'double_lorentzain'.")

    bounds_cfg = config.get("parameter_bounds", {}) or {}

    # Baseline seeds
    C0 = float(np.median(dPdH)) if len(dPdH) else 0.0
    D0 = 0.0
    # Numerical safety minimum for dH
    dH_eps = 1e-12

    if model_key == "lorentzain":
        mod = Model(d_sym_lorentz_with_baseline, nan_policy="omit")
        params = Parameters()
        H0_ = float(seeds[0]["center"])
        dH_ = max(float(seeds[0]["width"]), dH_eps)
        params.add("A", value=0.0, vary=True)
        params.add("H0", value=H0_, vary=True)
        params.add("dH", value=dH_, min=dH_eps, vary=True)
        params.add("C", value=C0, vary=True)
        params.add("D", value=D0, vary=True)
        for name in ("A", "H0", "dH", "C", "D"):
            _apply_bounds(params, bounds_cfg, name)
        result = mod.fit(dPdH, params, H=H)
        names = ["A", "H0", "dH", "C", "D"]

    elif model_key == "aysmetric":
        mod = Model(d_asym_lorentz, nan_policy="omit")
        params = Parameters()
        H0_ = float(seeds[0]["center"])
        dH_ = max(float(seeds[0]["width"]), dH_eps)
        params.add("A", value=0.0, vary=True)
        params.add("B", value=0.0, vary=True)
        params.add("H0", value=H0_, vary=True)
        params.add("dH", value=dH_, min=dH_eps, vary=True)
        params.add("C", value=C0, vary=True)
        params.add("D", value=D0, vary=True)
        for name in ("A", "B", "H0", "dH", "C", "D"):
            _apply_bounds(params, bounds_cfg, name)
        result = mod.fit(dPdH, params, H=H)
        names = ["A", "B", "H0", "dH", "C", "D"]

    else:  # double_lorentzain
        mod = Model(d_double_asym_lorentz, nan_policy="omit")
        params = Parameters()
        for i in range(2):
            H0i = float(seeds[i]["center"])
            dHi = max(float(seeds[i]["width"]), dH_eps)
            params.add(f"A{i+1}", value=0.0, vary=True)
            params.add(f"B{i+1}", value=0.0, vary=True)
            params.add(f"H0{i+1}", value=H0i, vary=True)
            params.add(f"dH{i+1}", value=dHi, min=dH_eps, vary=True)
            _apply_bounds(params, bounds_cfg, f"A{i+1}")
            _apply_bounds(params, bounds_cfg, f"B{i+1}")
            _apply_bounds(params, bounds_cfg, f"H0{i+1}")
            _apply_bounds(params, bounds_cfg, f"dH{i+1}")
        params.add("C", value=C0, vary=True)
        params.add("D", value=D0, vary=True)
        _apply_bounds(params, bounds_cfg, "C")
        _apply_bounds(params, bounds_cfg, "D")
        result = mod.fit(dPdH, params, H=H)
        names = ["A1", "B1", "H01", "dH1", "A2", "B2", "H02", "dH2", "C", "D"]

    # Collect parameter values and errors
    param_dict = {n: result.params[n].value for n in names}
    err_dict = {}
    for n in names:
        se = result.params[n].stderr
        err_dict[n] = "" if (se is None or (isinstance(se, float) and not np.isfinite(se))) else se

    # Metrics
    ss_res = np.sum((dPdH - result.best_fit) ** 2)
    ss_tot = np.sum((dPdH - np.mean(dPdH)) ** 2) if len(dPdH) else np.nan
    r2 = 1 - ss_res / ss_tot if ss_tot not in (0, np.nan) else np.nan

    fit_results = {
        "params": param_dict,
        "errors": err_dict,
        "success": bool(getattr(result, "success", True)),
        "r2": r2,
        "redchi": result.redchi,
        "fit_curve": result.best_fit,
    }
    return fit_results

def save_results_and_plots(fit_results, H, dPdH, metadata, config, model_type, processed_file):
    import matplotlib.pyplot as plt
    # Decide output directory
    out_cfg = config.get("output", {})
    custom_base = out_cfg.get("output_dir", "") or ""
    if custom_base:
        output_dir = os.path.join(custom_base, metadata["sample"])
    else:
        output_dir = os.path.dirname(processed_file)
    os.makedirs(output_dir, exist_ok=True)

    # Build names by replacing the processed suffix
    base_name = os.path.basename(processed_file)
    if base_name.endswith("_processed.csv"):
        out_base = base_name[:-len("_processed.csv")]
    else:
        out_base = os.path.splitext(base_name)[0]

    # Compose CSV row in the requested order
    model_key = str(model_type).strip().lower()
    if model_key == "double_lorentzain":
        param_order = ["A1", "B1", "H01", "dH1", "A2", "B2", "H02", "dH2", "C", "D"]
    elif model_key == "aysmetric":
        param_order = ["A", "B", "H0", "dH", "C", "D"]
    else:  # lorentzain
        param_order = ["A", "H0", "dH", "C", "D"]

    # Header: sample, f, db, index (blank), model, r2, success, redchi, params + errors, description
    row = {
        "sample": metadata.get("sample", ""),
        "f": metadata.get("f", ""),
        "db": metadata.get("db", ""),
        "index": "",  # leave blank per your instruction
        "model": model_type,
        "r2": fit_results["r2"],
        "success": fit_results["success"],
        "redchi": fit_results["redchi"],
    }
    # Add parameters with their errors immediately after each
    for p in param_order:
        row[p] = fit_results["params"].get(p, "")
        row[f"{p}_err"] = fit_results["errors"].get(p, "")
    # Final trailing description = processed file base name
    row["description"] = base_name

    # Save CSV (single-row)
    if out_cfg.get("save_csv", True):
        csv_path = os.path.join(output_dir, f"{out_base}_fit_results.csv")
        pd.DataFrame([row]).to_csv(csv_path, index=False)

    # Plot settings
    plot_cfg = config.get("plot", {})
    formats = plot_cfg.get("plot_formats", ["png"])
    show_legend = bool(plot_cfg.get("show_legend", True))
    # Title with signed dB (e.g., -10 dB)
    db_val = metadata.get('db', "")
    title_base = f"{metadata['sample']} f={metadata['f']}GHz {db_val} dB {model_type} R2={fit_results['r2']:.3f}" if (metadata.get("f") and db_val != "") else f"{model_type} R2={fit_results['r2']:.3f}"

    # 1) Data-only plot
    plt.figure()
    plt.plot(H, dPdH, 'o', label="Data", markersize=3)
    plt.title(title_base.replace(f" {model_type} R2", ""))
    if show_legend:
        plt.legend()
    for fmt in formats:
        plt.savefig(os.path.join(output_dir, f"{out_base}_plot.{fmt}"), dpi=300 if fmt.lower()=="png" else None)
    plt.close()

    # 2) Data + Fit plot
    plt.figure()
    plt.plot(H, dPdH, 'o', label="Data", markersize=3)
    plt.plot(H, fit_results["fit_curve"], '-', label="Fit", linewidth=2.0)
    plt.title(title_base)
    if show_legend:
        plt.legend()
    for fmt in formats:
        plt.savefig(os.path.join(output_dir, f"{out_base}_fit.{fmt}"), dpi=300 if fmt.lower()=="png" else None)
    plt.close()

def fit_spectrum(config_path=None, cli_args=None):
    config = load_config(config_path, cli_args)
    processed_file = select_processed_file(config)
    metadata = parse_metadata_from_filename(processed_file)
    H, dPdH = load_and_prepare_data(processed_file, config)

    # Model selection: only the three supported names; default to double derivative
    model_type = config.get("model") or "double_lorentzain"
    n_lobes = 2 if str(model_type).strip().lower() == "double_lorentzain" else 1

    lobes = detect_lobes(H, dPdH)
    seeds = seed_lobe_parameters(H, dPdH, lobes, n_lobes)

    fit_results = fit_model(H, dPdH, seeds, config, model_type)
    save_results_and_plots(fit_results, H, dPdH, metadata, config, model_type, processed_file)
    if config.get("output", {}).get("return_dict", True):
        return {
            "fit_results": fit_results,
            "metadata": metadata,
            "processed_file": processed_file
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FMR fitting pipeline (single file mode).")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML configuration file.")
    parser.add_argument("--input_file", type=str, default="", help="Path to processed spectrum CSV file.")
    parser.add_argument("--sample", type=str, default="", help="Sample name for auto-detect (if input_file not set).")
    parser.add_argument("--results_root", type=str, default="", help="Root directory for results and processed files.")
    parser.add_argument("--model", type=str, default="", help="Model type: Lorentzain, aysmetric, double_lorentzain.")
    parser.add_argument("--aggregation", type=str, default="", help="Aggregation method: mean, median, none.")
    parser.add_argument("--log_level", type=str, default="", help="Logging level: INFO, DEBUG, WARNING, ERROR.")
    parser.add_argument("--return_dict", type=bool, default=True, help="Return output dictionary (automation).")
    parser.add_argument("--output_dir", type=str, default="", help="Directory for saving results and plots.")
    parser.add_argument("--save_csv", type=bool, default=True, help="Save fit results CSV.")
    args = parser.parse_args()
    cli_args = vars(args)
    fit_spectrum(config_path=cli_args.pop("config"), cli_args=cli_args)