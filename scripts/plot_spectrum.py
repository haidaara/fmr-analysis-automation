import os
import matplotlib.pyplot as plt
import pandas as pd
import glob
import re

def plot_spectrum(
    processed_file=None,
    sample=None,
    results_root="results",
    show=True,
    save_png=True,
    save_svg=True,
    aggregate="median",  # 'mean' or 'median'
    color="#2980b9",
    linewidth=2.0,
    raw_color="#cc7000",  # amber/orange-brown, stands out but not too strong
    raw_alpha=0.7,
    raw_markersize=7,
    raw_linewidth=1.4,
    figsize=(7, 4),
    font_scale=1.15,
    return_ax=False,
    raw_style='curve',      # 'curve' or 'points'
    # --- Accepted for compatibility, but not used by this function: ---
    color_agg=None,
    linewidth_agg=None,
    color_fit=None,
    linewidth_fit=None,
    **kwargs  # absorb any further unused options
):
    """
    Plots H vs dP/dH from a processed data file, aggregating duplicate H values.
    - Aggregation: 'mean' or 'median' (default: 'median' for robust noise reduction).
    - Raw data can be shown as curve or points (see comments below).
    - No smoothing, just aggregation.
    - If return_ax is True, returns (fig, ax) instead of saving/showing.
    - color_agg, linewidth_agg, color_fit, linewidth_fit are accepted for interface compatibility,
      but only 'color' and 'linewidth' are actually used for the aggregated curve here.
    """
    # If processed_file not given, search for latest _processed.csv in results_root/sample/
    if processed_file is None:
        if sample is None:
            raise ValueError("If processed_file is not given, sample name must be provided.")
        search_pattern = os.path.join(results_root, sample, '*_processed.csv')
        processed_files = glob.glob(search_pattern)
        if not processed_files:
            raise FileNotFoundError(f"No processed files found in {os.path.join(results_root, sample)}")
        processed_file = max(processed_files, key=os.path.getmtime)
        print(f"[plot_spectrum] No file specified. Using the latest: {os.path.basename(processed_file)}")

    # Load data
    df = pd.read_csv(processed_file)
    if ("H (T)" not in df.columns) or ("dP/dH" not in df.columns):
        raise ValueError("CSV must contain columns: 'H (T)' and 'dP/dH'.")

    # Sort by H and aggregate duplicates
    agg_func = {"mean": "mean", "median": "median"}.get(aggregate)
    if not agg_func:
        raise ValueError("aggregate must be 'mean' or 'median'")
    df_agg = df.groupby("H (T)", as_index=False)["dP/dH"].agg(agg_func)
    H = df_agg["H (T)"].values
    dPdH = df_agg["dP/dH"].values

    # Parse filename for metadata
    filename = os.path.basename(processed_file)
    sample_name, f, db = "Sample", "", ""
    pattern = r"(?P<sample>.+)_f(?P<f>[\d\.]+)GHz_m(?P<db>-?\d+)dB"
    match = re.search(pattern, filename)
    if match:
        sample_name = match.group("sample")
        f = match.group("f")
        db = match.group("db")

    # Set font and figure size
    plt.rcParams.update({
        "font.size": 12 * font_scale,
        "axes.labelsize": 13 * font_scale,
        "axes.titlesize": 14 * font_scale,
        "xtick.labelsize": 11 * font_scale,
        "ytick.labelsize": 11 * font_scale,
        "legend.fontsize": 12 * font_scale,
    })

    fig, ax = plt.subplots(figsize=figsize)
    raw_sorted = df.sort_values("H (T)")

    # === RAW DATA STYLE OPTIONS ===
    if raw_style == 'curve':
        ax.plot(
            raw_sorted["H (T)"], raw_sorted["dP/dH"],
            color=raw_color, alpha=raw_alpha, linewidth=raw_linewidth, label="Raw data (curve)"
        )
    elif raw_style == 'points':
        ax.plot(
            raw_sorted["H (T)"], raw_sorted["dP/dH"],
            ".", color=raw_color, alpha=raw_alpha, markersize=raw_markersize, label="Raw data (points)"
        )
    else:
        raise ValueError("raw_style must be 'curve' or 'points'")
    # === END RAW DATA STYLE OPTIONS ===

    # Plot aggregated (always use 'color' and 'linewidth' for curve, not color_agg/linewidth_agg)
    ax.plot(H, dPdH, color=color, linewidth=linewidth, label=f"Aggregated ({aggregate})")
    ax.set_xlabel("H (T)")
    ax.set_ylabel("dP/dH")
    ax.set_title(f"{sample_name}, f={f} GHz, {db} dB")
    ax.legend()
    plt.tight_layout()

    # Save plot
    basefolder = os.path.dirname(processed_file)
    basefilename = filename.replace("_processed.csv", "_plot")
    png_path = os.path.join(basefolder, basefilename + ".png")
    svg_path = os.path.join(basefolder, basefilename + ".svg")
    if save_png:
        fig.savefig(png_path, dpi=300)
    if save_svg:
        fig.savefig(svg_path)
    
    print(f"Plot saved: \n\t{os.path.basename(png_path)} \n\t{os.path.basename(svg_path)}")
    
    if return_ax:
        return fig, ax
    
    if show:
        plt.show()
    else:
        plt.close(fig)

    return png_path, svg_path

# The following commented options are supported for interface compatibility / future-proofing:
# color_agg="#2980b9",    # for aggregated curve, handled by color above
# linewidth_agg=2.0,      # for aggregated curve, handled by linewidth above
# color_fit="orange",     # for fit curve, not used in plot_spectrum
# linewidth_fit=2.5,      # for fit curve, not used in plot_spectrum