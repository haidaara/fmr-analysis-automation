import os
import re
import glob

def loading_setup(data_folder, results_root, sample, f, db=None, k=64.404, h0=7.918):
    """
    Loads a single spectrum file (csv or txt) found in data_folder, matching sample and f (and db if specified).
    Extracts numeric current (I) and signal (dP/dH) columns, normalizes/units, inserts H column,
    and saves processed data to results_root.

    Parameters:
    - data_folder: folder to search for the data file.
    - results_root: output folder for processed files.
    - sample: sample name (used for output folder and for searching).
    - f: frequency (GHz, used for searching).
    - db: decibel value (used for searching if specified).
    - k, h0: constants for H calculation.

    Output file columns: current, H, dP/dH (comma separated)
    """
    # Build pattern for glob
    if db is not None:
        # If db is specified, only match files with that db value
        pattern = f"{sample}_f{f}GHz_m{abs(db)}dB*"
    else:
        # If db is not specified, match any dB value
        pattern = f"{sample}_f{f}GHz_m*dB*"

    # Search for matching file(s)
    search_pattern = os.path.join(data_folder, pattern + ".*")
    matches = glob.glob(search_pattern)
    if not matches:
        # Only show the pattern, not the full path
        raise FileNotFoundError(f"\n\nNo file found for pattern: {pattern}")

    # Use the first match found
    data_path = matches[0]
    print(f"[loading_setup] Auto-selected data file: {os.path.basename(data_path)}")

    processed_rows = []
    # Read lines
    with open(data_path, 'r') as f_in:
        for line in f_in:
            # Find first number (current)
            match1 = re.search(r'[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?', line)
            if not match1:
                continue
            I = float(match1.group())
            # Find first comma after the first number
            comma_idx = line.find(',', match1.end())
            if comma_idx == -1:
                continue
            # From just after that comma, find next number (dP/dH)
            match2 = re.search(r'[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?', line[comma_idx+1:])
            if not match2:
                continue
            dPdH = float(match2.group())
            # Calculate H
            H = k * I + h0
            processed_rows.append((I, H, dPdH))

    if not processed_rows:
        raise ValueError(f"No valid data lines found in file: {os.path.basename(data_path)}")

    # Output location
    out_folder = os.path.join(results_root, sample)
    os.makedirs(out_folder, exist_ok=True)
    # Output file name is based on input data file name + '_processed.csv'
    out_name = os.path.splitext(os.path.basename(data_path))[0] + '_processed.csv'
    out_path = os.path.join(out_folder, out_name)

    # Write output file
    with open(out_path, 'w') as f_out:
        f_out.write("I (A),H (T),dP/dH\n")
        for row in processed_rows:
            f_out.write(f"{row[0]},{row[1]},{row[2]}\n")

    print(f"Processed file saved: {os.path.basename(out_path)}")
    return out_path