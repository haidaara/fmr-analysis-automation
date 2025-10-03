"""
FMR Data Loader - Load and parse FMR measurement files
"""
import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path

def parse_filename(filename):
    """
    Extract sample, frequency, decibel from filename pattern:
    {sample}_f{freq}GHz_m{db}dB_{rep}_{id}.{ext}
    """
    parts = Path(filename).stem.split('_')
    
    # Handle different filename patterns
    sample = parts[0]
    
    # Find frequency (part starting with 'f' and ending with 'GHz')
    freq_part = next((p for p in parts if p.startswith('f') and 'GHz' in p), 'f5GHz')
    frequency = float(freq_part.replace('f', '').replace('GHz', ''))
    
    # Find decibel (part starting with 'm' and ending with 'dB')  
    db_part = next((p for p in parts if p.startswith('m') and 'dB' in p), 'm10dB')
    decibel = int(db_part.replace('m', '').replace('dB', ''))
    
    # Find replicate (usually the number after dB)
    replicate = 1
    for i, part in enumerate(parts):
        if part == db_part and i + 1 < len(parts):
            try:
                replicate = int(parts[i + 1])
                break
            except:
                pass
    
    return {
        'sample': sample,
        'frequency': frequency,
        'decibel': decibel,
        'replicate': replicate
    }

def load_single_file(file_path, slope=1.0, intercept=0.0):
    """
    Load one FMR file and convert current to magnetic field
    
    Args:
        file_path: Path to data file
        slope: Current to field conversion slope (T/A)
        intercept: Current to field conversion intercept (T)
    """
    try:
        # Parse filename metadata
        metadata = parse_filename(file_path)
        metadata['file_path'] = str(file_path)
        
        # Load data - handle tab+comma format
        df = pd.read_csv(file_path, delim_whitespace=True, header=None, 
                        names=['current', 'comma', 'signal'])
        
        # If we got 2 columns instead of 3, adjust
        if len(df.columns) == 2:
            df = pd.read_csv(file_path, delim_whitespace=True, header=None,
                            names=['current', 'signal'])
        
        current = df['current'].values
        signal = df['signal'].values
        
        # Convert current to magnetic field using linear transformation
        H_field = current * slope + intercept
        
        return {
            'metadata': metadata,
            'H_field': H_field,
            'signal': signal,
            'current': current
        }
        
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        raise

def load_folder(folder_path, slope=1.0, intercept=0.0):
    """
    Load all FMR files in a folder
    
    Returns: {sample_name: [file_data1, file_data2, ...]}
    """
    folder = Path(folder_path)
    all_data = {}
    
    # Find all text and csv files
    file_patterns = ['*.txt', '*.csv']
    file_list = []
    
    for pattern in file_patterns:
        file_list.extend(folder.glob(pattern))
        file_list.extend(folder.glob(f"**/{pattern}"))
    
    for file_path in file_list:
        try:
            data = load_single_file(file_path, slope, intercept)
            sample = data['metadata']['sample']
            
            if sample not in all_data:
                all_data[sample] = []
            all_data[sample].append(data)
            
            print(f"✓ Loaded: {file_path.name}")
            
        except Exception as e:
            print(f"❌ Failed: {file_path.name} - {e}")
            response = input("Continue loading other files? (y/n): ").lower()
            if response not in ['y', 'yes']:
                print("Stopped by user.")
                break
    
    # Print summary
    print(f"\n📊 Loading Summary:")
    for sample, files in all_data.items():
        frequencies = list(set(f['metadata']['frequency'] for f in files))
        print(f"  {sample}: {len(files)} files, {len(frequencies)} frequencies")
    
    return all_data

def loading_setup(results_root, sample, f=3.0, db=-10):
    """
    Load a single spectrum from processed files.
    
    Args:
        results_root (str): The root directory containing all processed data files.
        sample (str): The name of the sample folder inside results_root.
        f (float, optional): Frequency in GHz. Defaults to 3 GHz.
        db (int, optional): Decibel value. Defaults to -10 dB.
    
    Returns:
        pd.DataFrame: Processed data with columns including Current (I), dP/dt, and Magnetic Field (H).
    
    Raises:
        FileNotFoundError: If no matching file is found.
        ValueError: If file doesn't contain required columns or data parsing fails.
    """
    # Constants for magnetic field calculation
    K = 64.4  # Conversion slope
    H0 = 7.914  # Conversion intercept
    
    # Construct path to sample directory
    sample_dir = Path(results_root) / sample
    
    if not sample_dir.exists():
        raise FileNotFoundError(f"Sample directory not found: {sample_dir}")
    
    # Find matching file based on frequency and decibel
    # Look for pattern: {sample}_f{freq}GHz_m{abs(db)}dB_*.txt or *.csv
    # Handle both f3GHz and f3.0GHz formats
    
    matching_files = []
    for ext in ['*.txt', '*.csv']:
        for file_path in sample_dir.glob(ext):
            filename = file_path.name
            # Parse the file's metadata to check if it matches
            try:
                metadata = parse_filename(filename)
                # Check if frequency matches (allowing for float comparison)
                freq_match = abs(metadata['frequency'] - f) < 0.001
                # Check if decibel matches (note: parse_filename returns positive value)
                db_match = metadata['decibel'] == abs(db)
                
                if freq_match and db_match:
                    matching_files.append(file_path)
            except:
                # If parsing fails, try string matching as fallback
                freq_str_int = f"f{int(f)}GHz"
                freq_str_float = f"f{f}GHz"
                db_str = f"m{abs(db)}dB"
                
                if (freq_str_int in filename or freq_str_float in filename) and db_str in filename:
                    matching_files.append(file_path)
    
    if not matching_files:
        raise FileNotFoundError(
            f"No file found for sample '{sample}' with frequency {f} GHz and {db} dB in {sample_dir}"
        )
    
    # Use the first matching file if multiple found
    file_path = matching_files[0]
    
    try:
        # Read the file - try different formats
        df = None
        
        # Try reading with header first
        try:
            df_temp = pd.read_csv(file_path)
            # Check if it has the expected columns (case-insensitive)
            cols_lower = [c.lower() for c in df_temp.columns]
            if any('current' in c or 'i' in c for c in cols_lower) and \
               any('signal' in c or 'dp/dt' in c or 'dpdt' in c for c in cols_lower):
                df = df_temp
        except:
            pass
        
        # Try whitespace-delimited with no header
        if df is None:
            try:
                df = pd.read_csv(file_path, sep=r'\s+', header=None)
                # Assume first column is current, second is signal
                if len(df.columns) >= 2:
                    df.columns = ['current', 'signal'] + [f'col{i}' for i in range(2, len(df.columns))]
            except:
                pass
        
        # Try comma-separated with no header
        if df is None:
            try:
                df = pd.read_csv(file_path, header=None)
                if len(df.columns) >= 2:
                    df.columns = ['current', 'signal'] + [f'col{i}' for i in range(2, len(df.columns))]
            except:
                pass
        
        if df is None:
            raise ValueError(f"Could not parse file: {file_path}")
        
        # Normalize column names (case-insensitive mapping)
        col_mapping = {}
        for col in df.columns:
            col_lower = str(col).lower().strip()
            if 'current' in col_lower or col_lower == 'i':
                col_mapping[col] = 'current'
            elif 'signal' in col_lower or 'dp/dt' in col_lower or 'dpdt' in col_lower:
                col_mapping[col] = 'signal'
        
        df = df.rename(columns=col_mapping)
        
        # Validate required columns
        if 'current' not in df.columns or 'signal' not in df.columns:
            raise ValueError(
                f"File {file_path.name} must contain 'Current (I)' and 'dP/dt' columns. "
                f"Found columns: {list(df.columns)}"
            )
        
        # Convert to numeric and handle errors
        df['current'] = pd.to_numeric(df['current'], errors='coerce')
        df['signal'] = pd.to_numeric(df['signal'], errors='coerce')
        
        # Drop rows with NaN values
        initial_rows = len(df)
        df = df.dropna(subset=['current', 'signal'])
        
        if len(df) == 0:
            raise ValueError(f"No valid data rows in file: {file_path.name}")
        
        if len(df) < initial_rows:
            print(f"Warning: Dropped {initial_rows - len(df)} invalid rows from {file_path.name}")
        
        # Normalize units: ensure current is in Amps (assume already in correct unit)
        # Compute Magnetic Field H = K * I + H0
        df['H'] = K * df['current'] + H0
        
        # Rename columns to match specification
        df = df.rename(columns={
            'current': 'Current (I)',
            'signal': 'dP/dt',
            'H': 'Magnetic Field (H)'
        })
        
        # Save the processed data back to the file
        try:
            # Determine output format based on file extension
            if file_path.suffix.lower() == '.csv':
                df.to_csv(file_path, index=False)
            else:
                # Save as space-delimited for .txt files
                df.to_csv(file_path, sep=' ', index=False)
            print(f"✓ Saved processed data to: {file_path}")
        except Exception as e:
            print(f"Warning: Could not save processed data: {e}")
        
        return df
        
    except Exception as e:
        if isinstance(e, (FileNotFoundError, ValueError)):
            raise
        raise ValueError(f"Error processing file {file_path.name}: {str(e)}")