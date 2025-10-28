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
    # Support both f{f} and f{f}GHz patterns, and handle both m{db}dB and {db}dB for negative/positive dB
    patterns = []

    # Frequency string as in filename: allow both with and without "GHz"
    f_str = str(f)
    f_str_no_trail = f_str.rstrip("GHz")
    # db string with and without m (for minus)
    db_abs = abs(db) if db is not None else None
    db_prefix = "m" if db is not None and int(db) < 0 else ""

    if db is not None:
      # Try all 4 combinations
      # 1. With GHz, with m
      patterns.append(f"{sample}_f{f_str}GHz_{db_prefix}{db_abs}dB*")
      # 2. With GHz, without m (if db positive)
      if db_prefix == "":
          patterns.append(f"{sample}_f{f_str}GHz_{db_abs}dB*")
      # 3. Without GHz, with m
      patterns.append(f"{sample}_f{f_str}_{db_prefix}{db_abs}dB*")
      # 4. Without GHz, without m (if db positive)
      if db_prefix == "":
          patterns.append(f"{sample}_f{f_str}_{db_abs}dB*")
      # NEW precise (no 'f' in name): match exactly '{sample}_{f}GHz_{db}dB*'
      patterns.append(f"{sample}_{f_str}GHz_{db_prefix}{db_abs}dB*")
      # 5. Also support filenames like "{sample}_{number}GHz_{db}dB*" (e.g., AZ1_9GHz_0dBm)
      patterns.append(f"{sample}_[0-9]*GHz_{db_prefix}{db_abs}dB*")
      # 6. Allow ANY characters before "GHz" (e.g., decimals, negatives, etc.)
      patterns.append(f"{sample}_*GHz_{db_prefix}{db_abs}dB*")

    # Search for matching file(s)
    matches = []
    for pattern in patterns:
        search_pattern = os.path.join(data_folder, pattern + ".*")
        found = glob.glob(search_pattern)
        if found:
            matches = found
            break

    if not matches:
        # Only show the patterns tried, not the full path
        tried_patterns = "\n".join(patterns)
        raise FileNotFoundError(f"\n\nNo file found for any of these patterns:\n{tried_patterns}")

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