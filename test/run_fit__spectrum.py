import sys
import os

# Add scripts directory to path so we can import the fitter
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
scripts_dir = os.path.join(project_root, "scripts")
sys.path.append(scripts_dir)

# Use the improved second version (global baseline + full-range overlay)
from fit_spectrum2 import fit_spectrum

if __name__ == "__main__":
    # Required configuration
    sample = "AZ5"
    results_root = os.path.join(project_root, "results_roots")  # make sure this matches your folder layout

    # Choose model:
    # - "asymmetric_lorentzian" (recommended default)
    # - "double_asymmetric_lorentzian" (when a peripheral secondary line often appears)
    # - "lorentzian" (legacy single derivative-like)
    fitting_model = "double_asymmetric_lorentzian"

    # Optional: pick a specific frequency and/or dB by filename pattern
    # Set one or both; leave both None to auto-detect the latest processed file
    target_freq = None  # e.g., 9 for "f9GHz"
    target_db = None    # e.g., -20 for "m-20dB"

    # Try to select a processed file by pattern if filters are provided
    processed_file = None
    if target_freq is not None or target_db is not None:
        sample_dir = os.path.join(results_root, sample)
        if not os.path.isdir(sample_dir):
            raise FileNotFoundError(f"Sample directory not found: {sample_dir}")
        for fname in sorted(os.listdir(sample_dir)):
            if not fname.endswith("_processed.csv"):
                continue
            if target_freq is not None and f"f{target_freq}GHz" not in fname:
                continue
            if target_db is not None and f"m{target_db}dB" not in fname:
                continue
            processed_file = os.path.join(sample_dir, fname)
            break
        if processed_file is None:
            raise FileNotFoundError(
                f"No processed file in {sample_dir} matching "
                f"freq={target_freq} GHz, dB={target_db}"
            )

    # Run fit (if processed_file is None, the function will auto-detect latest for the sample)
    results = fit_spectrum(
        processed_file=processed_file,
        sample=sample,
        results_root=results_root,
        fitting_model=fitting_model,
        aggregate="median",  # None/'none'/False for raw; or 'median'/'mean' for aggregation

        # Peak detection / fitting behavior tuned for asymmetric signals
        iterative=True,               # lock ΔH first, then free
        detect_secondary=True,        # allow detection of peripheral secondary (used by double_asymmetric)
        fit_window_factor=2.5,        # focus single-peak fits near the main resonance

        # Baseline estimation (global, out-of-resonance)
        baseline_exclude_factor=3.0,  # exclude ±k·ΔH around H0 when estimating baseline
        baseline_min_frac=0.15,       # fallback: use ends totaling this fraction if needed

        # You can also tweak smoothing for detection if needed:
        # smooth_for_peaks=True,
        # smooth_frac=0.03,
        # smooth_poly=2,
        # width_lock_factor=0.4,

        # Plot styling (passed through to plot_spectrum; compatible options supported)
        raw_style="curve",            # 'curve' or 'points'
        raw_color="#cc7000",
        raw_alpha=0.7,
        raw_markersize=7,
        raw_linewidth=1.4,
        color_agg="#2980b9",          # accepted for compatibility by plot_spectrum
        linewidth_agg=2.0,            # accepted for compatibility by plot_spectrum
        color_fit="orange",
        linewidth_fit=2.5,
        figsize=(7, 4),
        font_scale=1.15,

        # Output behavior
        save_plot=True,
        show_plot=True,
        return_params=True,
        plot_suffix="fit"
    )

    print("Fit Results:")
    for k, v in results.items():
        print(f"{k}: {v}")

# Run this with: python test/run_fit_spectrum.py