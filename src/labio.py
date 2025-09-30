
from pathlib import Path
import pandas as pd
import numpy as np
import warnings

__all__ = ["safe_load_file", "load_folder_secure", "parse_filename_fallback"]

def _read_any(path: Path) -> pd.DataFrame:
    # whitespace 3-col
    try:
        df = pd.read_csv(path, delim_whitespace=True, header=None, names=['current','comma','signal'])
        if {'current','comma','signal'}.issubset(df.columns) and df['signal'].notna().any():
            return df[['current','signal']]
    except Exception:
        pass
    # whitespace 2-col
    try:
        df = pd.read_csv(path, delim_whitespace=True, header=None, names=['current','signal'])
        if {'current','signal'}.issubset(df.columns) and df['signal'].notna().any():
            return df
    except Exception:
        pass
    # csv with/without header
    for header in [None, 'infer']:
        try:
            df = pd.read_csv(path, header=header)
            cols = [c for c in df.columns]
            low  = [str(c).strip().lower() for c in cols]
            if 'current' in low and 'signal' in low:
                df.columns = [str(c).strip().lower() for c in cols]
                return df[['current','signal']].copy()
            if len(cols) >= 2:
                sub = df.iloc[:, :2].copy()
                sub.columns = ['current','signal']
                return sub
        except Exception:
            pass
    raise ValueError(f"Could not parse file: {path}")

def parse_filename_fallback(stem: str) -> dict:
    base = Path(stem).stem
    parts = base.split('_')
    meta = {'sample': parts[0] if parts else 'sample',
            'frequency': None, 'decibel': None, 'replicate': 1}
    # freq
    for p in parts:
        if p.startswith('f') and 'GHz' in p:
            try:
                meta['frequency'] = float(p.replace('f','').replace('GHz',''))
                break
            except Exception:
                pass
    # dB
    for p in parts:
        if p.startswith('m') and 'dB' in p:
            try:
                meta['decibel'] = int(p.replace('m','').replace('dB',''))
                break
            except Exception:
                pass
    # replicate
    try:
        tokens = [p for p in parts if p.startswith('m') and 'dB' in p]
        if tokens:
            idx = parts.index(tokens[0])
            if idx+1 < len(parts):
                meta['replicate'] = int(parts[idx+1])
    except Exception:
        pass
    return meta

def safe_load_file(path: Path, slope: float, intercept: float) -> dict:
    df = _read_any(path)
    for c in ['current','signal']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    n0 = len(df)
    df = df.dropna(subset=['current','signal'])
    if len(df) < n0:
        warnings.warn(f"{Path(path).name}: dropped {n0-len(df)} NaN rows.")
    df['H'] = slope*df['current'] + intercept
    if not (df['H'].is_monotonic_increasing or df['H'].is_monotonic_decreasing):
        warnings.warn(f"{Path(path).name}: H not monotonic; OK but check hysteresis / sorting.")
    meta = parse_filename_fallback(Path(path).name)
    meta['file_path'] = str(path)
    return {'meta': meta, 'df': df}

def load_folder_secure(folder: Path, slope: float, intercept: float):
    folder = Path(folder)
    files = list(folder.glob('*.txt')) + list(folder.glob('*.csv'))
    files += list(folder.rglob('*.txt')) + list(folder.rglob('*.csv'))
    records = []
    for p in sorted(set(files)):
        try:
            rec = safe_load_file(p, slope, intercept)
            records.append(rec)
            print(f"✓ Loaded {p.name}  →  n={len(rec['df'])}")
        except Exception as e:
            print(f"❌ {p.name} → {e}")
    return records
