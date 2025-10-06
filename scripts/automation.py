import os
import re
import glob
import csv
import yaml
import pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN

from loading_setup import loading_setup
from plot_spectrum import plot_spectrum
from fit_spectrum2 import fit_spectrum

# ========== CONFIGURATION LOADER ==========
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config.setdefault("data_folder", "data")
    config.setdefault("results_root", "results_roots")
    config.setdefault("samples", [])
    config.setdefault("plot_format", "png")
    config.setdefault("aggregate", "median")
    config.setdefault("fit_model", "lorentzian")
    return config

# ========== HELPERS ==========

def discover_raw_files(data_folder, sample_list=None):
    pattern_txt = os.path.join(data_folder, "*_f*GHz_m*dB*.txt")
    pattern_csv = os.path.join(data_folder, "*_f*GHz_m*dB*.csv")
    files = glob.glob(pattern_txt) + glob.glob(pattern_csv)
    discovered = []
    regex = r'(?P<sample>.+)_f(?P<f>[\d\.]+)GHz_m(?P<db>-?\d+)dB(?:_(?P<index>\d+))?'
    for file in files:
        fname = os.path.basename(file)
        match = re.match(regex, fname)
        if not match:
            continue
        meta = match.groupdict()
        if sample_list and meta['sample'] not in sample_list:
            continue
        meta['filename'] = fname
        meta['full_path'] = file
        meta['f_float'] = float(meta['f'])
        meta['f'] = int(float(meta['f'])) if float(meta['f']).is_integer() else meta['f']
        meta['db'] = int(meta['db'])
        meta['index'] = int(meta['index']) if meta['index'] is not None else None
        discovered.append(meta)
    # Sort files: by sample, frequency, db, index
    discovered = sorted(
        discovered,
        key=lambda d: (
            d['sample'],
            d['f_float'],
            d['db'],
            d.get('index', 0) or 0
        )
    )
    return discovered

def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

def collect_fit_results(sample, results_root):
    sample_dir = os.path.join(results_root, sample)
    fit_result_files = glob.glob(os.path.join(sample_dir, "*_fit_results.csv"))
    rows = []
    for fr in fit_result_files:
        fname = os.path.basename(fr)
        meta = parse_metadata_from_filename(fname)
        with open(fr, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)
            values = next(reader)
        row = {**meta}
        for k, v in zip(header, values):
            row[k] = v
        row["result_file"] = fname
        rows.append(row)
    # Sort results: by frequency, db, index
    rows = sorted(
        rows,
        key=lambda d: (
            float(d['f']) if 'f' in d and d['f'] is not None else 0,
            int(d['db']) if 'db' in d and d['db'] is not None else 0,
            int(d['index']) if 'index' in d and d['index'] is not None else 0,
        )
    )
    return rows

def parse_metadata_from_filename(fname):
    regex = r'(?P<sample>.+)_f(?P<f>[\d\.]+)GHz_m(?P<db>-?\d+)dB(?:_(?P<index>\d+))?'
    match = re.match(regex, fname)
    meta = match.groupdict() if match else {}
    try:
        if "f" in meta and meta["f"]:
            meta['f'] = int(float(meta["f"])) if float(meta["f"]).is_integer() else meta["f"]
    except Exception:
        meta['f'] = meta.get('f', None)
    try:
        if "db" in meta and meta["db"]:
            meta['db'] = int(meta["db"])
    except Exception:
        meta['db'] = meta.get('db', None)
    try:
        if "index" in meta and meta["index"]:
            meta['index'] = int(meta["index"])
    except Exception:
        meta['index'] = meta.get('index', None)
    return meta

def collect_plot_files(sample, results_root, plot_format="png"):
    sample_dir = os.path.join(results_root, sample)
    pattern = os.path.join(sample_dir, f"*plot*.{plot_format}")
    return sorted(glob.glob(pattern))

def collect_fit_plot_files(sample, results_root, plot_format="png"):
    sample_dir = os.path.join(results_root, sample)
    # This matches files like *_fit.png or *_fit.svg (but not *_plot.png)
    pattern = os.path.join(sample_dir, f"*fit*.{plot_format}")
    files = glob.glob(pattern)
    # Optionally, filter out files that are not actual fit plots (e.g. avoid *_fit_results.csv)
    files = [f for f in files if os.path.splitext(f)[1].lower() == f'.{plot_format}']
    return sorted(files)

def create_sample_analysis_xlsx(sample, results_root):
    rows = collect_fit_results(sample, results_root)
    if not rows:
        print(f"No fit results for sample {sample}")
        return
    meta_cols = ["sample", "f", "db", "index", "result_file"]
    data_cols = [c for c in rows[0] if c not in meta_cols]
    columns = meta_cols + data_cols
    df = pd.DataFrame(rows)
    df = df[columns]
    out_path = os.path.join(results_root, sample, f"{sample}_analysis.xlsx")
    df.to_excel(out_path, index=False)
    print(f"[{sample}] Aggregated results saved to {out_path}")

def create_sample_pptx(sample, results_root, plot_format="png"):
    plot_files = collect_plot_files(sample, results_root, plot_format)
    if not plot_files:
        print(f"No plot files found for {sample}")
        return
    prs = Presentation()
    for plot_path in plot_files:
        slide = prs.slides.add_slide(prs.slide_layouts[5])  # Blank slide
        fname = os.path.basename(plot_path)
        meta = parse_metadata_from_filename(fname)
        title_text = f"{meta.get('sample','')} f={meta.get('f','')}GHz m{meta.get('db','')}dB"
        left = Inches(0.5)
        top = Inches(0.5)
        pic = slide.shapes.add_picture(plot_path, left, top, width=Inches(8), height=Inches(5))
        txBox = slide.shapes.add_textbox(Inches(0.5), Inches(0.1), Inches(8), Inches(0.3))
        tf = txBox.text_frame
        tf.text = title_text
        tf.paragraphs[0].font.size = Pt(20)
        tf.paragraphs[0].alignment = PP_ALIGN.CENTER
    pptx_path = os.path.join(results_root, sample, f"{sample}_plots.pptx")
    prs.save(pptx_path)
    print(f"[{sample}] PowerPoint with plots saved to {pptx_path}")

def create_sample_fit_pptx(sample, results_root, plot_format="png"):
    fit_plot_files = collect_fit_plot_files(sample, results_root, plot_format)
    if not fit_plot_files:
        print(f"No fit plot files found for {sample}")
        return
    prs = Presentation()
    for plot_path in fit_plot_files:
        slide = prs.slides.add_slide(prs.slide_layouts[5])  # Blank slide
        fname = os.path.basename(plot_path)
        meta = parse_metadata_from_filename(fname)
        title_text = f"{meta.get('sample','')} f={meta.get('f','')}GHz m{meta.get('db','')}dB FIT"
        left = Inches(0.5)
        top = Inches(0.5)
        pic = slide.shapes.add_picture(plot_path, left, top, width=Inches(8), height=Inches(5))
        txBox = slide.shapes.add_textbox(Inches(0.5), Inches(0.1), Inches(8), Inches(0.3))
        tf = txBox.text_frame
        tf.text = title_text
        tf.paragraphs[0].font.size = Pt(20)
        tf.paragraphs[0].alignment = PP_ALIGN.CENTER
    pptx_path = os.path.join(results_root, sample, f"{sample}_fit.pptx")
    prs.save(pptx_path)
    print(f"[{sample}] PowerPoint with fit plots saved to {pptx_path}")

# ========== MAIN PIPELINE ==========

def main():
    # Load YAML config (relative to project root)
    config = load_config(os.path.join(os.path.dirname(__file__), "..", "config.yaml"))
    data_folder = config["data_folder"]
    results_root = config["results_root"]
    sample_list = config["samples"] if config["samples"] else None
    plot_format = config["plot_format"]

    discovered = discover_raw_files(data_folder, sample_list)
    if not discovered:
        print("No raw files discovered.")
        return

    samples = []
    for d in discovered:
        if d["sample"] not in samples:
            samples.append(d["sample"])

    for sample in samples:
        print(f"\n=== Processing sample: {sample} ===")
        sample_discovered = [d for d in discovered if d["sample"] == sample]
        ensure_dir(os.path.join(results_root, sample))

        # Sort for each sample as well (should already be sorted globally, but for safety)
        sample_discovered = sorted(
            sample_discovered,
            key=lambda d: (
                d['f_float'],
                d['db'],
                d.get('index', 0) or 0
            )
        )

        for meta in sample_discovered:
            print(f"  -> Processing: {meta['filename']}")
            # Step 1: Loading
            processed_path = loading_setup(
                data_folder,
                results_root,
                meta['sample'],
                meta['f'],
                meta['db']
            )
            # Step 2: Plotting
            plot_spectrum(
                processed_file=processed_path,
                sample=meta['sample'],
                results_root=results_root,
                fitting_model='double_asymmetric_lorentzian',
                save_png=(plot_format=="png"),
                save_svg=(plot_format=="svg"),
                show=False
            )
      
            # Step 3: Fitting
            fit_spectrum(
                processed_file=processed_path,
                sample=meta['sample'],
                results_root=results_root,

                save_plot=True,
                show_plot=False
            )

        # Aggregation: Create sample_analysis.xlsx
        create_sample_analysis_xlsx(sample, results_root)
        # Merge plots into pptx
        create_sample_pptx(sample, results_root, plot_format=plot_format)
        # Merge fit plots into fit pptx
        create_sample_fit_pptx(sample, results_root, plot_format=plot_format)

if __name__ == "__main__":
    main()