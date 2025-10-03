# loading_setup Function

## Overview
The `loading_setup` function is part of the FMR Analysis pipeline and provides a simple way to load and process a single spectrum from processed files.

## Function Signature
```python
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
```

## Usage Examples

### Basic Usage
```python
from loader import loading_setup

# Load a spectrum with default parameters (3 GHz, -10 dB)
df = loading_setup("/path/to/results", "MySample")

# Access the processed data
print(df.columns)  # ['Current (I)', 'dP/dt', 'Magnetic Field (H)']
print(df.head())
```

### Custom Frequency and Decibel
```python
# Load a spectrum with custom frequency and decibel
df = loading_setup("/path/to/results", "MySample", f=5.0, db=-15)
```

### Error Handling
```python
try:
    df = loading_setup("/path/to/results", "NonExistent", f=7.0)
except FileNotFoundError as e:
    print(f"File not found: {e}")
except ValueError as e:
    print(f"Invalid data: {e}")
```

## Features

### Data Processing
1. **File Discovery**: Automatically finds files matching the pattern `{sample}_f{freq}GHz_m{db}dB_*.txt` or `*.csv`
2. **Format Flexibility**: Handles multiple file formats:
   - CSV with headers
   - Space/tab-delimited with headers
   - Space/tab-delimited without headers
3. **Column Validation**: Ensures required columns (Current, Signal/dP/dt) are present
4. **Data Normalization**: Converts column names to standard format

### Magnetic Field Calculation
The function computes the Magnetic Field (H) using the formula:
```
H = K × I + H₀
```
Where:
- K = 64.4 (conversion slope)
- H₀ = 7.914 (conversion intercept)
- I = Current in Amps

### File Operations
- **Read**: Loads data from `.txt` or `.csv` files in `results_root/sample/`
- **Process**: Normalizes data and computes magnetic field
- **Save**: Writes processed data back to the same file
- **Return**: Returns a pandas DataFrame with processed data

## Output Format

The returned DataFrame contains the following columns:
- `Current (I)`: Current values in Amps
- `dP/dt`: Signal/derivative values
- `Magnetic Field (H)`: Computed magnetic field values

Example output:
```
   Current (I)  dP/dt  Magnetic Field (H)
0          0.1    0.5              14.354
1          0.2    0.6              20.794
2          0.3    0.7              27.234
```

## Directory Structure

Expected directory structure:
```
results_root/
└── sample_name/
    ├── sample_name_f3GHz_m10dB_1.txt
    ├── sample_name_f5GHz_m10dB_1.txt
    └── sample_name_f7GHz_m15dB_1.csv
```

## Error Handling

The function provides clear error messages for common issues:
- **Missing sample directory**: `FileNotFoundError: Sample directory not found: ...`
- **No matching file**: `FileNotFoundError: No file found for sample '...' with frequency ... GHz and ... dB`
- **Invalid columns**: `ValueError: File ... must contain 'Current (I)' and 'dP/dt' columns`
- **Parsing errors**: `ValueError: Error processing file ...`

## Testing

Comprehensive unit tests are available in `tests/test_loading_setup.py`. Run tests with:
```bash
python -m unittest tests.test_loading_setup -v
```
