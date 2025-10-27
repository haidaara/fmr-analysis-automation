import os
import re
import glob
import csv
import logging
import shutil
from datetime import datetime
from PIL import Image

def is_likely_csv(file_path):
    """Check if file is tabular (for extensionless files)"""
    try:
        with open(file_path, "r") as f:
            first_line = f.readline()
            if "," in first_line or "\t" in first_line or len(first_line.split()) > 1:
                return True
    except Exception:
        pass
    return False

def ensure_csv_extension(file_path):
    """Add .csv extension in place if file is likely CSV and has no extension"""
    if file_path.endswith(".csv") or file_path.endswith(".txt"):
        return file_path
    if file_path.endswith(".csv.csv"):
        base = file_path[:-4]
        os.rename(file_path, base)
        return base
    if is_likely_csv(file_path):
        new_path = file_path + ".csv"
        if not os.path.exists(new_path):
            os.rename(file_path, new_path)
        return new_path
    return file_path

def simple_parse_filename(fname):
    """Parse sample, frequency, dB, and optional index from a filename core."""
    fname_core = fname.rsplit('.', 1)[0]
    parts = fname_core.split('_')
    sample = parts[0] if len(parts) > 0 else None

    freq_match = re.search(r'f(\d+\.?\d*)', fname_core)
    freq = freq_match.group(1) if freq_match else None

    db = None
    db_neg_match = re.search(r'(?:_|-)m(\d+)dB', fname_core, re.IGNORECASE)
    if db_neg_match:
        db = -int(db_neg_match.group(1))
    else:
        db_match = re.search(r'(?:_|-)(\d+)dB', fname_core, re.IGNORECASE)
        if db_match:
            db = int(db_match.group(1))
    if db is None:
        return None

    index = None
    index_match = re.search(r'_([0-9]+)$', fname_core)
    if index_match:
        index = int(index_match.group(1))

    return {
        "sample": sample,
        "f": freq,
        "db": db,
        "index": index,
    }

def discover_raw_files(data_folder, sample_list=None):
    """Recursively find all files and match to samples (exact match only)."""
    all_files = []
    for root, dirs, files in os.walk(data_folder):
        for fname in files:
            full_path = os.path.join(root, fname)
            all_files.append(full_path)
    discovered = []
    for file in all_files:
        file_with_ext = ensure_csv_extension(file)
        fname = os.path.basename(file_with_ext)
        meta = simple_parse_filename(fname)
        if meta is None:
            continue
        if sample_list:
            if meta['sample'] not in sample_list:
                continue
        meta['filename'] = fname
        meta['full_path'] = file_with_ext
        try:
            meta['f_float'] = float(meta['f']) if meta['f'] else 0
            meta['f'] = int(float(meta['f'])) if meta['f'] and float(meta['f']).is_integer() else meta['f']
        except Exception:
            meta['f_float'] = 0
        discovered.append(meta)
    discovered = sorted(
        discovered,
        key=lambda d: (
            d['sample'],
            d['f_float'],
            d['db'] if d['db'] is not None else 0,
            d.get('index', 0) or 0
        )
    )
    return discovered

def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

def latest_mtime_in_dir(path):
    """Return latest modification time among files in dir (fallback to dir mtime)."""
    latest = None
    for root, dirs, files in os.walk(path):
        for name in files:
            p = os.path.join(root, name)
            try:
                m = os.path.getmtime(p)
                latest = m if latest is None else max(latest, m)
            except Exception:
                pass
    if latest is None:
        try:
            latest = os.path.getmtime(path)
        except Exception:
            latest = None
    return latest

def archive_and_prepare_sample_dir(results_root, sample):
    """
    If a sample dir exists and has content, move it to results_root/_archive/<sample>_<timestamp>/,
    then create a fresh sample dir.
    """
    sample_dir = os.path.join(results_root, sample)
    if os.path.isdir(sample_dir):
        has_content = any(os.scandir(sample_dir))
        if has_content:
            archive_root = os.path.join(results_root, "_archive")
            os.makedirs(archive_root, exist_ok=True)
            mtime = latest_mtime_in_dir(sample_dir)
            from datetime import datetime as _dt
            ts = _dt.fromtimestamp(mtime).strftime("%Y%m%d-%H%M%S") if mtime else _dt.now().strftime("%Y%m%d-%H%M%S")
            dest = os.path.join(archive_root, f"{sample}_{ts}")
            logging.info(f"[ARCHIVE] {sample_dir} -> {dest}")
            print("\n" + "-"*60)
            print(f"[ARCHIVE] {sample_dir} -> {dest}")
            print("-"*60 + "\n")
            shutil.move(sample_dir, dest)
    os.makedirs(sample_dir, exist_ok=True)
    return sample_dir

def collect_fit_results(sample, results_root):
    sample_dir = os.path.join(results_root, sample)
    fit_result_files = glob.glob(os.path.join(sample_dir, "*_fit_results.csv"))
    rows = []
    for fr in fit_result_files:
        fname = os.path.basename(fr)
        meta = simple_parse_filename(fname)
        if meta is None:
            continue
        with open(fr, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            try:
                header = next(reader)
                values = next(reader)
            except StopIteration:
                continue
        row = {**meta}
        for k, v in zip(header, values):
            row[k] = v
        row["result_file"] = fname
        rows.append(row)
    rows = sorted(
        rows,
        key=lambda d: (
            float(d['f']) if 'f' in d and d['f'] not in [None, ""] else 0,
            int(d['db']) if 'db' in d and d['db'] not in [None, ""] else 0,
            int(d['index']) if 'index' in d and d['index'] not in [None, ""] else 0,
        )
    )
    return rows

def collect_plot_files(sample, results_root, plot_formats=["png"]):
    sample_dir = os.path.join(results_root, sample)
    plot_files = []
    for fmt in plot_formats:
        pattern = os.path.join(sample_dir, f"*plot*.{fmt}")
        plot_files.extend(glob.glob(pattern))
    return sorted(plot_files)

def collect_fit_plot_files(sample, results_root, plot_formats=["png"]):
    sample_dir = os.path.join(results_root, sample)
    fit_plot_files = []
    for fmt in plot_formats:
        pattern = os.path.join(sample_dir, f"*fit*.{fmt}")
        files = glob.glob(pattern)
        files = [f for f in files if os.path.splitext(f)[1].lower() == f'.{fmt}']
        fit_plot_files.extend(files)
    return sorted(fit_plot_files)

def filtered_pptx_formats(formats):
    """Return only raster formats PPTX/Pillow can handle."""
    allowed = {"png", "jpg", "jpeg", "bmp"}
    chosen = [str(f).lower() for f in formats if str(f).lower() in allowed]
    return chosen if chosen else ["png"]

def validate_image_or_warn(path):
    """Verify image is readable; warn and return False if not."""
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception as e:
        logging.warning(f"[PPTX] Skipping invalid image: {path} ({e})")
        print(f"\t[WARN] Skipping invalid image: {path} ({e})")
        return False