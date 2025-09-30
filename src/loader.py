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