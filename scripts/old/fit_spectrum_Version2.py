import argparse
import os
import glob
import re
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from lmfit import Model, Parameters
import yaml

# -----------------------------
# Model functions
# -----------------------------
def lorentzian(H, H_res, dH, amp, offset):
    return amp * (dH * (H - H_res)) / ((H - H_res)**2 + (dH/2)**2)**2 + offset

def asymmetric_lorentzian(H, A, B, H0, dH, C, D):
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
# Config loader
# -----------------------------
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# -----------------------------
# Data loading & preprocessing
# -----------------------------
def load_and_preprocess(csv_file, config):
    df = pd.read_csv(csv_file)
    H = df.iloc[:, 1].values
    dP_dH = df.iloc[:, 2].values

    agg_method = config.get("aggregation", "median")
    if agg_method.lower() in ("median", "mean"):
        unique_H = np.unique(H)
        aggregated = []
        for h_val in unique_H:
            vals = dP_dH[H == h_val]
            agg = np.median(vals) if agg_method == "median" else np.mean(vals)
            aggregated.append(agg)
        H_agg = unique_H
        dP_dH_agg = np.array(aggregated)
    else:
        H_agg, dP_dH_agg = H, dP_dH

    meta = {}
    return H_agg, dP_dH_agg, dP_dH_agg, meta

# -----------------------------
# Lobe/Peak configuration
# -----------------------------
def get_n_lobes_required(config):
    model = str(config.get("model", "double")).lower()
    return {"single": 1, "double": 2, "triple": 3}.get(model, 2)

# -----------------------------
# Lobe detection
# -----------------------------
def detect_lobes(H, dP_dH):
    peaks, _ = find_peaks(dP_dH)
    mins, _ = find_peaks(-dP_dH)
    lobe_candidates = []
    for peak in peaks:
        distances = np.abs(H[mins] - H[peak])
        if len(distances) == 0:
            continue
        min_idx = np.argmin(distances)
        min_pos = mins[min_idx]
        lobe = {
            "max_idx": peak,
            "min_idx": min_pos,
            "center": float((H[peak] + H[min_pos]) / 2),
            "width": float(np.sqrt(3) / 2 * np.abs(H[peak] - H[min_pos])),
            "amplitude": float(dP_dH[peak] - dP_dH[min_pos]),
            "H_max": float(H[peak]), "H_min": float(H[min_pos]),
            "dP_dH_max": float(dP_dH[peak]), "dP_dH_min": float(dP_dH[min_pos])
        }
        lobe_candidates.append(lobe)
    lobe_candidates.sort(key=lambda x: (-abs(x["amplitude"]), abs(x["H_max"] - x["H_min"])))
    return lobe_candidates

# -----------------------------
# Peak selection
# -----------------------------
def select_lobes(lobe_candidates, n_required):
    selected = lobe_candidates[:n_required]
    neglected_count = max(0, n_required - len(selected))
    return selected, neglected_count

# -----------------------------
# Neglecting values
# -----------------------------
def neglected_lobe_params(H):
    return {
        "A": 0, "B": 0, "center": float(np.mean(H)),
        "width": float(0.5 * (np.max(H) - np.min(H)))
    }

# -----------------------------
# Model construction
# -----------------------------
def build_model_and_params(selected_lobes, neglected_count, H, config):
    model_type = str(config.get("model", "double")).lower()
    params = Parameters()
    if model_type == "single":
        model = Model(asymmetric_lorentzian)
        if len(selected_lobes) > 0:
            lobe = selected_lobes[0]
            params.add('A', value=lobe.get("amplitude", 1), min=-1e10, max=1e10)
            params.add('B', value=0, min=-1e10, max=1e10)
            params.add('H0', value=lobe.get("center", np.mean(H)))
            params.add('dH', value=lobe.get("width", 1), min=1e-6)
        else:
            nl = neglected_lobe_params(H)
            params.add('A', value=nl["A"], min=-1e10, max=1e10)
            params.add('B', value=nl["B"], min=-1e10, max=1e10)
            params.add('H0', value=nl["center"])
            params.add('dH', value=nl["width"], min=1e-6)
        params.add('C', value=0)
        params.add('D', value=0)
        return model, params

    if model_type == "double":
        model = Model(double_asymmetric_lorentzian)
        if len(selected_lobes) > 0:
            lobe = selected_lobes[0]
            params.add('A1', value=lobe.get("amplitude", 1), min=-1e10, max=1e10)
            params.add('B1', value=0, min=-1e10, max=1e10)
            params.add('H1', value=lobe.get("center", np.mean(H)))
            params.add('dH1', value=lobe.get("width", 1), min=1e-6)
        else:
            nl = neglected_lobe_params(H)
            params.add('A1', value=nl["A"], min=-1e10, max=1e10)
            params.add('B1', value=nl["B"], min=-1e10, max=1e10)
            params.add('H1', value=nl["center"])
            params.add('dH1', value=nl["width"], min=1e-6)
        if len(selected_lobes) > 1:
            lobe = selected_lobes[1]
            params.add('A2', value=lobe.get("amplitude", 1), min=-1e10, max=1e10)
            params.add('B2', value=0, min=-1e10, max=1e10)
            params.add('H2', value=lobe.get("center", np.mean(H)))
            params.add('dH2', value=lobe.get("width", 1), min=1e-6)
        else:
            nl = neglected_lobe_params(H)
            params.add('A2', value=nl["A"], min=-1e10, max=1e10)
            params.add('B2', value=nl["B"], min=-1e10, max=1e10)
            params.add('H2', value=nl["center"])
            params.add('dH2', value=nl["width"], min=1e-6)
        params.add('C', value=0)
        params.add('D', value=0)
        return model, params

    raise ValueError(f"Unknown or unsupported model type: {model_type}")

# -----------------------------
# Fitting
# -----------------------------
class FitResult:
    def __init__(self, result, model, params):
        self.result = result
        self.model = model
        self.params = params
    def save_results_csv(self, config, outbase):
        out = {k: [v.value, v.stderr] for k, v in self.result.params.items()}
        df = pd.DataFrame.from_dict(out, orient='index', columns=['value', 'stderr'])
        outdir = config.get("results_dir", "results")
        os.makedirs(outdir, exist_ok=True)
        outfile = os.path.join(outdir, f"{outbase}_fit_results.csv")
        df.to_csv(outfile)
        print(f"Fit results saved to {outfile}")
    def as_dict(self):
        return {k: v.value for k, v in self.result.params.items()}

def run_fit(model, params, H, y, config):
    result = model.fit(y, params, H=H)
    return FitResult(result, model, params)

# -----------------------------
# Plotting & reporting
# -----------------------------
def parse_sample_metadata(filename):
    base = os.path.basename(filename)
    pattern = r'(?P<sample>[A-Za-z0-9]+)_f(?P<freq>\d+\.?\d*)GHz_m(?P<db>-?\d+)dB'
    m = re.match(pattern, base)
    info = {}
    if m:
        info = m.groupdict()
    info['filename'] = base
    return info

def build_plot_title(meta, config, fit_result=None):
    title = f"{meta.get('sample','')} f={meta.get('freq','')}GHz {meta.get('db','')}dB"
    title += f" [{config.get('model','double')}]"
    if fit_result:
        r2 = getattr(fit_result.result, "rsquared", None)
        if r2 is not None:
            title += f"  R²={r2:.4f}"
    return title

def plot_fit_results(H, dP_dH, agg_curve, fit_result, config, sample_info, outbase):
    plot_cfg = config.get("plot", {})
    results_dir = config.get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)

    plt.figure(figsize=(8, 5))
    if plot_cfg.get("data_style", "points") == "points":
        plt.scatter(H, dP_dH, s=12, alpha=0.5, label="Raw Data")
    else:
        plt.plot(H, dP_dH, '.', alpha=0.5, label="Raw Data")
    plt.plot(H, agg_curve, '-', color='black', lw=2, label="Aggregated Curve")
    plt.title(build_plot_title(sample_info, config))
    plt.xlabel("H (T)")
    plt.ylabel("dP/dH")
    if plot_cfg.get("show_legend", True): plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{outbase}_plot.png"))
    if plot_cfg.get("save_svg", True): plt.savefig(os.path.join(results_dir, f"{outbase}_plot.svg"))
    plt.close()

    plt.figure(figsize=(8, 5))
    if plot_cfg.get("data_style", "points") == "points":
        plt.scatter(H, dP_dH, s=12, alpha=0.5, label="Raw Data")
    else:
        plt.plot(H, dP_dH, '.', alpha=0.5, label="Raw Data")
    plt.plot(H, agg_curve, '-', color='black', lw=2, label="Aggregated Curve")
    plt.plot(H, fit_result.result.best_fit, '-', color='red', lw=2, label="Fit Curve")
    plt.title(build_plot_title(sample_info, config, fit_result))
    plt.xlabel("H (T)")
    plt.ylabel("dP/dH")
    if plot_cfg.get("show_legend", True): plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{outbase}_fit.png"))
    if plot_cfg.get("save_svg", True): plt.savefig(os.path.join(results_dir, f"{outbase}_fit.svg"))
    plt.close()

    if plot_cfg.get("show_residuals", True):
        plt.figure(figsize=(8, 3))
        plt.plot(H, fit_result.result.residual, '.', color='purple', alpha=0.8)
        plt.axhline(0, color='black', ls='--')
        plt.xlabel("H (T)")
        plt.ylabel("Residual")
        plt.title("Fit Residuals")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"{outbase}_residuals.png"))
        if plot_cfg.get("save_svg", True): plt.savefig(os.path.join(results_dir, f"{outbase}_residuals.svg"))
        plt.close()

# -----------------------------
# Main fitting callable
# -----------------------------
def run_fit_spectrum(
    processed_file=None,
    sample=None,
    results_root="results_roots",
    save_plot=True,
    show_plot=False,
    config_path="config.yaml"
):
    """
    If processed_file is None, runs batch fitting over all *_processed.csv in sample dir.
    If processed_file is given, fits just that file and returns results dict.
    """
    config = load_config(config_path)
    if sample is not None:
        config["results_dir"] = os.path.join(results_root, sample)
    results_dir = config["results_dir"]

    if processed_file is None:
        # Batch mode: fit all processed files in sample dir
        sample_dir = results_dir
        processed_files = glob.glob(os.path.join(sample_dir, "*_processed.csv"))
        all_results = []
        for pf in processed_files:
            print(f"[fit_spectrum] Fitting: {os.path.basename(pf)}")
            result = run_fit_spectrum(
                processed_file=pf,
                sample=sample,
                results_root=results_root,
                save_plot=save_plot,
                show_plot=show_plot,
                config_path=config_path
            )
            if result is not None:
                all_results.append(result)
        if all_results:
            df = pd.DataFrame(all_results)
            excel_path = os.path.join(sample_dir, f"{sample}_fit_results.xlsx")
            df.to_excel(excel_path, index=False)
            print(f"Aggregated fit results saved to {excel_path}")
        return all_results

    # --- Single file mode ---
    # Set results_dir for this sample/file
    if sample is not None:
        config["results_dir"] = os.path.join(results_root, sample)
    else:
        config["results_dir"] = results_root

    # Outbase: use file basename without _processed.csv
    basefile = os.path.basename(processed_file)
    if basefile.endswith("_processed.csv"):
        outbase = basefile[:-len("_processed.csv")]
    else:
        outbase = os.path.splitext(basefile)[0]

    # Preprocess/load
    H, dP_dH, aggregated_curve, meta = load_and_preprocess(processed_file, config)
    lobe_candidates = detect_lobes(H, dP_dH)
    n_lobes_required = get_n_lobes_required(config)
    selected_lobes, neglected_count = select_lobes(lobe_candidates, n_lobes_required)
    model, params = build_model_and_params(selected_lobes, neglected_count, H, config)
    fit_result = run_fit(model, params, H, aggregated_curve, config)
    sample_info = parse_sample_metadata(basefile)
    if save_plot:
        plot_fit_results(H, dP_dH, aggregated_curve, fit_result, config, sample_info, outbase)
    fit_result.save_results_csv(config, outbase)
    result_dict = fit_result.as_dict()
    result_dict["file"] = basefile
    return result_dict

# -----------------------------
# CLI entry point
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="FMR Fitting Pipeline")
    parser.add_argument('--processed_file', type=str, default=None, help='Path to processed CSV file (single file mode, optional)')
    parser.add_argument('--sample', type=str, default=None, help='Sample name (for batch mode or results organization)')
    parser.add_argument('--results_root', type=str, default="results_roots", help='Root directory for results')
    parser.add_argument('--config', type=str, default="config.yaml", help='Path to config YAML')
    parser.add_argument('--show_plot', action='store_true', help='Show plot interactively')
    args = parser.parse_args()
    run_fit_spectrum(
        processed_file=args.processed_file,
        sample=args.sample,
        results_root=args.results_root,
        show_plot=args.show_plot,
        config_path=args.config
    )

if __name__ == "__main__":
    main()