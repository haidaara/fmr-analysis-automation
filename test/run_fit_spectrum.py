import sys
import os

# Add scripts directory to path so we can import fit_spectrum and plot_spectrum
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
scripts_dir = os.path.join(project_root, "scripts")
sys.path.append(scripts_dir)

from fit_spectrum import fit_spectrum

# if __name__ == "__main__":
#     # Set sample and results_root
sample = "AZ5"
#     results_root = os.path.join(project_root, "results_roots")

#     # Try to find a processed file in results_root/sample/
#     sample_dir = os.path.join(results_root, sample)
#     processed_file = None
#     if os.path.isdir(sample_dir):
#         for fname in os.listdir(sample_dir):
#             if fname.endswith("_processed.csv"):
#                 processed_file = os.path.join(sample_dir, fname)
#                 break

#     if processed_file is None:
#         raise FileNotFoundError(f"No processed file found in {sample_dir}")

    # Run fit_spectrum (which now calls plot_spectrum internally for unified plotting)
results = fit_spectrum(
    processed_file=None,  # Let it auto-detect the latest processed file
    sample= sample,
    fitting_model='lorentzian', # 'lorentzian', 'asymmetric_lorentzian', 'double_lorentzian'
    aggregate='median',         # Pass None for raw data, or 'median'/'mean' for aggregation
    save_plot=True,
    show_plot=True,
    return_params=True,
    plot_suffix="fit",
    raw_style='curve',      # 'curve' or 'points'
    raw_color="#cc7000",
    raw_alpha=0.7,
    raw_markersize=7,
    raw_linewidth=1.4,
    color_agg="#2980b9",
    linewidth_agg=2.0,
    color_fit="orange",
    linewidth_fit=2.5,
    figsize=(7, 4),
    font_scale=1.15
)
print("Fit Results:")
for k, v in results.items():
    print(f"{k}: {v}")

# Run this test with: python test/run_fit_spectrum.py