# Kittel and Damping Analysis

This script uses config.yaml automatically; no CLI flags are required.

Fits:
- Kittel: f = (γ/2π) · √[H_res · (H_res + M_eff)]
- Damping: ΔH = slope · f + ΔH0, with α = 0.5 · slope · (γ/2π)

Inputs
- Auto-detected per sample: `<results_root>/<sample>/<sample>_analysis.csv`
- Or set an explicit file: `analysis.input_table`
- Required columns (defaults):
  - f (GHz)
  - H_res (T)
  - H_res_err (T) — optional; used if present
  - Delta_H (T)
  - Delta_H_err (T) — optional; used if present

How samples are selected
- If `analysis.input_table` is set, the script uses that file and infers the sample name.
- Else, it uses `samples` (list) from config; if empty, it falls back to `sample`.

Config keys (under `analysis`)
- columns: rename headers if your CSV uses different names
- kittel:
  - gamma_over_2pi_init: null means estimate from data; else use the value (GHz/T)
  - slope_fraction: fraction of highest H points for the slope-based initial guess (default 0.25)
  - weighted: "auto" (use H_res_err if present), true, or false
- damping:
  - use_gamma_from_kittel: true (recommended)
  - weighted: "auto" (use Delta_H_err if present), true, or false
- plot:
  - plot_formats: ["png"] (add "pdf" if you like)
  - show_legend: true

Outputs per sample (under its results folder)
- Plots: `<sample>_kittel_data.*`, `<sample>_kittel_fit.*`, `<sample>_damping_data.*`, `<sample>_damping_fit.*`
- Excel: `<sample>_kittel_damping.xlsx` with sheets:
  - Processed Table (clean rows used)
  - Kittel Fit (γ/2π and M_eff with uncertainties)
  - Damping Fit (α and ΔH0 with uncertainties)

Run
- Default (reads config.yaml):  
  `python fit_analysis.py`
- Or with a custom config path:  
  `python fit_analysis.py --config myconfig.yaml`