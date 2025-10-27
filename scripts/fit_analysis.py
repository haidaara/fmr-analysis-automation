import argparse
import os
import glob
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lmfit import Model, Parameters

def load_config(yaml_path=None):
    cfg = {}
    path = yaml_path or "config.yaml"
    if os.path.isfile(path):
        with open(path, "r") as f:
            y = yaml.safe_load(f) or {}
            cfg.update(y)
    # Defaults
    cfg.setdefault("results_root", "results_roots")
    cfg.setdefault("samples", [])
    cfg.setdefault("sample", "")
    cfg.setdefault("analysis", {})
    a = cfg["analysis"]
    a.setdefault("input_table", "")
    a.setdefault("columns", {})
    a["columns"].setdefault("f", "f")
    a["columns"].setdefault("H_res", "H_res")
    a["columns"].setdefault("H_res_err", "H_res_err")
    a["columns"].setdefault("Delta_H", "Delta_H")
    a["columns"].setdefault("Delta_H_err", "Delta_H_err")
    a.setdefault("units", {"f": "GHz", "H": "T"})
    a.setdefault("kittel", {})
    a["kittel"].setdefault("gamma_over_2pi_init", None)  # None = estimate from data
    a["kittel"].setdefault("slope_fraction", 0.25)        # top fraction of H_res to estimate initial slope
    a["kittel"].setdefault("weighted", "auto")            # "auto" | True | False
    a.setdefault("damping", {})
    a["damping"].setdefault("use_gamma_from_kittel", True)
    a["damping"].setdefault("weighted", "auto")           # "auto" | True | False
    a.setdefault("plot", {})
    a["plot"].setdefault("plot_formats", ["png"])
    a["plot"].setdefault("show_legend", True)
    return cfg

def parse_sample_from_table(path):
    # Expect .../<sample>/<sample>_analysis.csv or any *_analysis.csv
    base = os.path.basename(path)
    if base.endswith("_analysis.csv"):
        return base[:-len("_analysis.csv")]
    return os.path.splitext(base)[0]

def find_analysis_csv_for_sample(root, sample):
    # Prefer exact <sample>_analysis.csv
    candidate = os.path.join(root, sample, f"{sample}_analysis.csv")
    if os.path.isfile(candidate):
        return candidate
    # Fallback: latest *_analysis.csv in sample folder
    folder = os.path.join(root, sample)
    files = glob.glob(os.path.join(folder, "*_analysis.csv"))
    if files:
        return max(files, key=os.path.getmtime)
    raise FileNotFoundError(f"Could not find analysis table for sample {sample} in {folder}")

def find_analysis_sources(cfg):
    a = cfg["analysis"]
    input_table = a.get("input_table", "") or ""
    results_root = cfg.get("results_root", ".")
    sources = []
    if input_table and os.path.isfile(input_table):
        sample = parse_sample_from_table(input_table)
        sources.append((sample, input_table))
        return sources
    # Else from samples list or single sample
    samples = cfg.get("samples") or []
    if not samples:
        single = cfg.get("sample", "")
        if single:
            samples = [single]
    if not samples:
        raise ValueError("No samples specified. Set analysis.input_table, or config.samples, or config.sample.")
    for s in samples:
        path = find_analysis_csv_for_sample(results_root, s)
        sources.append((s, path))
    return sources

def kittel_function(H, G, M):
    # f(H) in GHz when G = gamma/2pi in GHz/T, H and M in T
    return G * np.sqrt(H * (H + M))

def kittel_df_dH(H, G, M):
    # df/dH = G * (2H + M) / (2*sqrt(H(H+M)))
    denom = 2.0 * np.sqrt(H * (H + M))
    denom = np.where(denom == 0, np.finfo(float).eps, denom)
    return G * (2.0 * H + M) / denom

def estimate_initials(f, H, cfg):
    frac = float(cfg["analysis"]["kittel"].get("slope_fraction", 0.25) or 0.25)
    n = len(H)
    if n < 2 or not np.isfinite(H).all() or not np.isfinite(f).all():
        return 28.0, 1.0
    idx = np.argsort(H)
    top_k = max(2, int(np.ceil(frac * n)))
    sel = idx[-top_k:]
    H_sel = H[sel]
    f_sel = f[sel]
    if np.allclose(H_sel.min(), H_sel.max()):
        G0 = 28.0
    else:
        s, b = np.polyfit(H_sel, f_sel, 1)
        G0 = float(s) if np.isfinite(s) and s > 0 else 28.0
    with np.errstate(divide="ignore", invalid="ignore"):
        M_i = (f / G0) ** 2 / H - H
    M_i = M_i[np.isfinite(M_i)]
    if len(M_i) == 0:
        M0 = 1.0
    else:
        M0 = float(np.median(M_i))
        if not np.isfinite(M0):
            M0 = 1.0
    return G0, M0

def fit_kittel(f, H, H_err, cfg):
    G_init_cfg = cfg["analysis"]["kittel"].get("gamma_over_2pi_init", None)
    if G_init_cfg is None:
        G0, M0 = estimate_initials(f, H, cfg)
    else:
        G0 = float(G_init_cfg)
        with np.errstate(divide="ignore", invalid="ignore"):
            Mi = (f / G0) ** 2 / H - H
        Mi = Mi[np.isfinite(Mi)]
        M0 = float(np.median(Mi)) if len(Mi) else 1.0

    mod = Model(kittel_function, nan_policy="omit")
    params = Parameters()
    params.add("G", value=G0, min=0, vary=True)   # gamma_over_2pi [GHz/T]
    params.add("M", value=M0, min=0, vary=True)   # Meff [T]

    weighted = cfg["analysis"]["kittel"].get("weighted", "auto")
    use_w = False
    if weighted == "auto":
        use_w = H_err is not None and np.any(np.isfinite(H_err)) and np.nanmax(H_err) > 0
    elif isinstance(weighted, bool):
        use_w = weighted

    weights = None
    if use_w and H_err is not None:
        df_dH0 = kittel_df_dH(H, G0, M0)
        sigma_f = np.abs(df_dH0) * H_err
        with np.errstate(divide="ignore", invalid="ignore"):
            w = 1.0 / sigma_f
        w[~np.isfinite(w)] = 0.0
        weights = w

    result = mod.fit(f, params, H=H, weights=weights)

    ss_res = np.sum((f - result.best_fit) ** 2)
    ss_tot = np.sum((f - np.nanmean(f)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

    out = {
        "G": result.params["G"].value,
        "G_err": result.params["G"].stderr if result.params["G"].stderr is not None else "",
        "M": result.params["M"].value,
        "M_err": result.params["M"].stderr if result.params["M"].stderr is not None else "",
        "success": bool(getattr(result, "success", True)),
        "r2": r2,
        "model": result,
    }
    return out

def fit_damping(f, dH, dH_err, G, G_err, cfg):
    def line(x, m, b):
        return m * x + b

    mod = Model(line, nan_policy="omit")
    # Simple initial slope guess
    denom = (np.nanmax(f) - np.nanmin(f)) if np.isfinite(f).all() else 1.0
    denom = denom if denom != 0 else 1.0
    params = Parameters()
    params.add("m", value=(np.nanmax(dH) - np.nanmin(dH)) / denom)
    params.add("b", value=np.nanmedian(dH))

    weighted = cfg["analysis"]["damping"].get("weighted", "auto")
    use_w = False
    if weighted == "auto":
        use_w = dH_err is not None and np.any(np.isfinite(dH_err)) and np.nanmax(dH_err) > 0
    elif isinstance(weighted, bool):
        use_w = weighted

    weights = None
    if use_w and dH_err is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            w = 1.0 / dH_err
        w[~np.isfinite(w)] = 0.0
        weights = w

    result = mod.fit(dH, params, x=f, weights=weights)

    m = result.params["m"].value
    m_err = result.params["m"].stderr if result.params["m"].stderr is not None else None
    b = result.params["b"].value
    b_err = result.params["b"].stderr if result.params["b"].stderr is not None else None

    alpha = 0.5 * m * G
    if m_err is None and (G_err in (None, "", np.nan)):
        alpha_err = ""
    else:
        m_err_val = float(m_err) if m_err is not None else 0.0
        G_err_val = float(G_err) if (G_err not in ("", None)) else 0.0
        alpha_err = 0.5 * np.sqrt((G * m_err_val) ** 2 + (m * G_err_val) ** 2)

    y = dH
    yhat = result.best_fit
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.nanmean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

    return {
        "m": m, "m_err": ("" if m_err is None else m_err),
        "b": b, "b_err": ("" if b_err is None else b_err),
        "alpha": alpha, "alpha_err": ("" if alpha_err == "" else alpha_err),
        "success": bool(getattr(result, "success", True)),
        "r2": r2,
        "model": result,
    }

def plot_kittel(sample, H, H_err, f, kittel_res, out_dir, formats, show_legend=True):
    base = f"{sample}_kittel"
    # Data
    plt.figure()
    if H_err is not None and np.any(np.isfinite(H_err)) and np.nanmax(H_err) > 0:
        plt.errorbar(H, f, xerr=H_err, fmt='o', label="Data", markersize=4)
    else:
        plt.plot(H, f, 'o', label="Data", markersize=4)
    plt.xlabel("H_res (T)")
    plt.ylabel("f (GHz)")
    plt.title("Kittel Data")
    if show_legend:
        plt.legend()
    for fmt in formats:
        plt.savefig(os.path.join(out_dir, f"{base}_data.{fmt}"), dpi=300 if fmt.lower() in ("png", "jpg") else None)
    plt.close()

    # Fit overlay
    Hgrid = np.linspace(np.nanmin(H), np.nanmax(H), 400)
    f_fit = kittel_function(Hgrid, kittel_res["G"], kittel_res["M"])
    plt.figure()
    if H_err is not None and np.any(np.isfinite(H_err)) and np.nanmax(H_err) > 0:
        plt.errorbar(H, f, xerr=H_err, fmt='o', label="Data", markersize=4)
    else:
        plt.plot(H, f, 'o', label="Data", markersize=4)
    plt.plot(Hgrid, f_fit, '-', label=f"Fit (R2={kittel_res['r2']:.3f})", linewidth=2)
    plt.xlabel("H_res (T)")
    plt.ylabel("f (GHz)")
    plt.title("Kittel Fit")
    if show_legend:
        plt.legend()
    for fmt in formats:
        plt.savefig(os.path.join(out_dir, f"{base}_fit.{fmt}"), dpi=300 if fmt.lower() in ("png", "jpg") else None)
    plt.close()

def plot_damping(sample, f, dH, dH_err, damp_res, out_dir, formats, show_legend=True):
    base = f"{sample}_damping"
    # Data
    plt.figure()
    if dH_err is not None and np.any(np.isfinite(dH_err)) and np.nanmax(dH_err) > 0:
        plt.errorbar(f, dH, yerr=dH_err, fmt='o', label="Data", markersize=4)
    else:
        plt.plot(f, dH, 'o', label="Data", markersize=4)
    plt.xlabel("f (GHz)")
    plt.ylabel("ΔH (T)")
    plt.title("Damping Data")
    if show_legend:
        plt.legend()
    for fmt in formats:
        plt.savefig(os.path.join(out_dir, f"{base}_data.{fmt}"), dpi=300 if fmt.lower() in ("png", "jpg") else None)
    plt.close()

    # Fit overlay
    fgrid = np.linspace(np.nanmin(f), np.nanmax(f), 400)
    yfit = damp_res["m"] * fgrid + damp_res["b"]
    plt.figure()
    if dH_err is not None and np.any(np.isfinite(dH_err)) and np.nanmax(dH_err) > 0:
        plt.errorbar(f, dH, yerr=dH_err, fmt='o', label="Data", markersize=4)
    else:
        plt.plot(f, dH, 'o', label="Data", markersize=4)
    plt.plot(fgrid, yfit, '-', label=f"Fit (R2={damp_res['r2']:.3f})", linewidth=2)
    plt.xlabel("f (GHz)")
    plt.ylabel("ΔH (T)")
    plt.title("Damping Fit")
    if show_legend:
        plt.legend()
    for fmt in formats:
        plt.savefig(os.path.join(out_dir, f"{base}_fit.{fmt}"), dpi=300 if fmt.lower() in ("png", "jpg") else None)
    plt.close()

def run_single_sample(cfg, sample, table_path):
    analysis = cfg["analysis"]
    cols = analysis["columns"]
    out_dir = os.path.join(cfg.get("results_root", "."), sample)
    os.makedirs(out_dir, exist_ok=True)

    # Load data
    df = pd.read_csv(table_path)
    try:
        f = np.asarray(df[cols["f"]].values, dtype=float)
        H = np.asarray(df[cols["H_res"]].values, dtype=float)
        dH = np.asarray(df[cols["Delta_H"]].values, dtype=float)
    except KeyError as e:
        raise KeyError(f"[{sample}] Missing required column in analysis table: {e}")

    H_err = None
    dH_err = None
    if cols["H_res_err"] in df.columns:
        H_err = np.asarray(df[cols["H_res_err"]].values, dtype=float)
    if cols["Delta_H_err"] in df.columns:
        dH_err = np.asarray(df[cols["Delta_H_err"]].values, dtype=float)

    # Clean rows
    mask = np.isfinite(f) & np.isfinite(H) & np.isfinite(dH)
    f, H, dH = f[mask], H[mask], dH[mask]
    if H_err is not None:
        H_err = H_err[mask]
    if dH_err is not None:
        dH_err = dH_err[mask]
    if len(f) < 2:
        raise ValueError(f"[{sample}] Not enough valid rows after cleaning.")

    # Kittel and damping
    kittel_res = fit_kittel(f, H, H_err, cfg)
    G = kittel_res["G"]
    G_err = kittel_res["G_err"]
    damp_res = fit_damping(f, dH, dH_err, G, G_err, cfg)

    # Plots
    formats = analysis.get("plot", {}).get("plot_formats", ["png"])
    show_legend = bool(analysis.get("plot", {}).get("show_legend", True))
    plot_kittel(sample, H, H_err, f, kittel_res, out_dir, formats, show_legend)
    plot_damping(sample, f, dH, dH_err, damp_res, out_dir, formats, show_legend)

    # Save Excel
    used_cols = {"f": f, "H_res": H, "Delta_H": dH}
    if H_err is not None:
        used_cols["H_res_err"] = H_err
    if dH_err is not None:
        used_cols["Delta_H_err"] = dH_err
    df_used = pd.DataFrame(used_cols)

    df_kittel = pd.DataFrame([{
        "gamma_over_2pi (GHz/T)": kittel_res["G"],
        "gamma_over_2pi_err": kittel_res["G_err"],
        "M_eff (T)": kittel_res["M"],
        "M_eff_err": kittel_res["M_err"],
        "success": kittel_res["success"],
        "r2": kittel_res["r2"],
    }])

    df_damp = pd.DataFrame([{
        "alpha": damp_res["alpha"],
        "alpha_err": damp_res["alpha_err"],
        "Delta_H0 (T)": damp_res["b"],
        "Delta_H0_err": damp_res["b_err"],
        "slope (T/GHz)": damp_res["m"],
        "slope_err": damp_res["m_err"],
        "success": damp_res["success"],
        "r2": damp_res["r2"],
        "gamma_over_2pi_used (GHz/T)": G
    }])

    excel_path = os.path.join(out_dir, f"{sample}_kittel_damping.xlsx")
    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        df_used.to_excel(writer, index=False, sheet_name="Processed Table")
        df_kittel.to_excel(writer, index=False, sheet_name="Kittel Fit")
        df_damp.to_excel(writer, index=False, sheet_name="Damping Fit")

    # CLI summary
    print(f"[{sample}] Kittel: gamma/2pi = {kittel_res['G']:.4f} ± {kittel_res['G_err'] if kittel_res['G_err']!='' else 'NA'} GHz/T, "
          f"M_eff = {kittel_res['M']:.4f} ± {kittel_res['M_err'] if kittel_res['M_err']!='' else 'NA'} T, "
          f"R2 = {kittel_res['r2']:.3f}")
    print(f"[{sample}] Damping: alpha = {damp_res['alpha']:.6g} ± {damp_res['alpha_err'] if damp_res['alpha_err']!='' else 'NA'}, "
          f"Delta_H0 = {damp_res['b']:.6g} ± {damp_res['b_err'] if damp_res['b_err']!='' else 'NA'} T, "
          f"R2 = {damp_res['r2']:.3f}")

    return {
        "kittel": kittel_res,
        "damping": damp_res,
        "excel": excel_path,
        "analysis_table": table_path,
        "output_dir": out_dir
    }

def run_from_config(config_path=None):
    cfg = load_config(config_path)
    sources = find_analysis_sources(cfg)
    results = []
    for sample, table_path in sources:
        res = run_single_sample(cfg, sample, table_path)
        results.append(res)
    return results

if __name__ == "__main__":
    # No args required; will read config.yaml automatically.
    parser = argparse.ArgumentParser(description="Kittel and damping analysis (config-driven).")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML configuration file.")
    args = parser.parse_args()
    run_from_config(args.config)