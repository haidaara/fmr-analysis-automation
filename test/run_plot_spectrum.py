import sys
import os

# Add scripts directory to path so we can import plot_spectrum
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
scripts_dir = os.path.join(project_root, "scripts")
sys.path.append(scripts_dir)

from plot_spectrum import plot_spectrum

if __name__ == "__main__":
    # Set test parameters
    # Find a processed file to plot, or set sample name to auto-detect the latest
    sample = "AZ5"
    results_root = os.path.join(project_root, "results_roots")

    # Option 1: Specify a processed file directly (uncomment and adjust as needed)
    # processed_file = os.path.join(results_root, sample, "AZ5_f5GHz_m20dB_1_processed.csv")
    # plot_spectrum(processed_file=processed_file, show=True, save_png=True, save_svg=True)

    # Option 2: Specify only the sample, let the function auto-detect the latest processed file
    plot_spectrum(processed_file=None, sample=sample, results_root=results_root, 
                    show=True, save_png=True, save_svg=True)


#run test with: python test/run_plot_spectrum.py