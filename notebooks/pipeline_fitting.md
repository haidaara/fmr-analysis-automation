# FMR Fitting Pipeline (Robust Lobe Detection, Model Override, and Seeding - Explicit Neglecting Values & Data Handling)

This pipeline integrates configuration-driven model selection with robust data-driven lobe detection and parameter seeding for reliable FMR fitting.  
**Neglected lobe parameter values are specified unambiguously.**  
**Data is always loaded from columns named "H (T)" and "dP/dH" of the processed CSV file. Preprocessing (such as aggregation) is controlled by the configuration file.**

---

## 1. Configuration File & Data Loading

- **Unified Configuration Object & Merge Rules:**  
    - At the start of the pipeline, all configuration settings (from YAML file and/or input arguments) are loaded into a unified configuration object (`config`).  
    - If both YAML and CLI arguments are provided, CLI arguments override the corresponding YAML settings. Any missing settings use defaults as documented in the pipeline.  
    - All subsequent pipeline steps (data extraction, preprocessing, fitting, output) use settings exclusively from this unified configuration object, ensuring consistency and maintainability.

- **Read configuration** from a YAML file at the start.
    - **Only `model`** (e.g., `"single"`, `"double"`, `"triple"`, `"auto"`, etc.) is used for model selection.
    - If `model` is not specified, **default to `"double"`** (detect two lobes).
    - Other settings: aggregation, parameter bounds, plotting options (e.g., curve/point data display), output directories, etc.

- **Handling Processed File (Single File Only):**  
    - The pipeline always operates on a single processed CSV file per run.
    - The file can be provided either by explicit path (`processed_file` argument) OR, if not specified, the pipeline requires a `sample` argument and will automatically search for the latest `*_processed.csv` file in the directory `results_root/<sample>/`. The most recently modified file will be selected for analysis.
    - No batch or multi-file processing is performed.

- **Load Data:**  
    - From the processed CSV file, always extract:
        - **H** (field) from the column named `"H (T)"`
        - **dP/dH** (signal) from the column named `"dP/dH"`
    - Ignore other columns.
    - If the required columns are missing, raise an error.
    - Upon loading data, the pipeline sorts all entries by the field `H` in ascending order prior to any further processing or fitting.

- **Preprocessing:**  
    - Apply aggregation and any other preprocessing steps as specified in the configuration file (e.g., averaging, baseline correction, smoothing, etc.).
    - The pipeline must use the unified configuration object to determine which preprocessing steps to apply.
    - If no preprocessing is specified, no preprocessing is performed.

---

## 2. Lobe/Peak Configuration

- **Determine number of lobes (peaks) to use** based on `model` from the configuration:
    - `"single"`: 1 lobe
    - `"double"`: 2 lobes
    - `"triple"`: 3 lobes
    - `"auto"` or not specified: use default (typically 2 lobes)
- This number controls:
    - How many lobes are selected from lobe detection
    - How many lobe parameter sets are created for model construction

---

## 3. Principal Lobe Detection

- **Input:** Preprocessed arrays H (field), dP/dH (signal).
- **Find Extrema:**
    - Use `scipy.signal.find_peaks(dP/dH)` to detect all maxima (unconstrained by config/model).
    - Use `scipy.signal.find_peaks(-dP/dH)` to detect all minima (unconstrained by config/model).
- **Lobe Pairing by Proximity:**
    - For each detected maximum, find the nearest minimum (in H or index space, using absolute difference).
    - Form max–min pairs as candidate lobes.
- **Pair Ranking:**
    - For each pair, calculate amplitude difference: \(|dP/dH_{\max} - dP/dH_{\min}|\).
    - Sort all candidate pairs by amplitude difference (primary criterion).
    - In the case of ties or ambiguity, prefer the pair with the **smallest interval** \(|H_{\max} - H_{\min}|\).
- **Output:**  
    - List of detected lobes (as dictionaries or similar structure), sorted by significance.

---

## 4. Peak Selection

- **Select the number of lobes required by the configuration/model** from the sorted list of detected lobes (from Section 3).
    - For example, if model is `"double"`, select the two top-ranked lobes.
    - If fewer lobes are detected than required, record how many need to be filled with neglecting values.
- **Parameter Seeding (Indexed):**
    - For each selected lobe, compute:
        - Center: \(H_{0,i} = \frac{H_{\min,i} + H_{\max,i}}{2}\)
        - Width:  \(\Delta H_{i} = \frac{\sqrt{3}}{2}(H_{\max,i} - H_{\min,i})\)
    - **Amplitude parameters are NOT seeded**—their initial values and handling are defined by the model implementation, not by this detection.

---

## 5. Neglecting Values (EXPLICIT)

- For each required-but-undetected lobe:
    - **Amplitude:** Set to **exactly zero** (\(\text{amplitude} = 0\))
    - **Center:** Assign to the mean of the field array (\(H_0 = \text{mean}(H)\))
    - **Width:** Assign to half the span of the field array (\(\Delta H = 0.5 \times (\max(H) - \min(H))\))
    - **Asymmetry or other model-specific parameters:** Set to zero if present
    - **Do NOT** use NaN, infinity, or out-of-range values for any parameter

---

## 6. Model Construction

- **Model Selection:**
    - Use the model specified in the configuration (`model`), or default to double asymmetric Lorentzian.
- **Parameter Setup:**
    - Use initial seeds from lobe selection for as many lobes as the model requires.
    - For any additional required lobes not detected, use the explicit neglecting values from section 5.
    - Apply parameter bounds from the configuration.

---

## 7. Fitting

- **Allow all parameters to vary** (`vary=True` for all) for robustness and simplicity.
- **Run lmfit minimization** on the prepared model and data.

---

## 8. Results & Diagnostics

- **Extract fit results:**
    - Best-fit parameter values and uncertainties.
    - Fit quality metrics (R², reduced χ², AIC/BIC, success flag).
    - Save results as CSV and/or return as a dictionary.
- **Fit results CSV files:**  
    - Each `_fit_results.csv` file contains a header line (parameters, errors, metrics, success flag, model type) and **a single row of values for the fit**.
    - The header structure is **determined by the fitting model used** and may differ for Lorentzian, asymmetric Lorentzian, or other models.  
      For example, a Lorentzian fit will produce columns such as `H_res`, `dH`, `amp`, `offset`, while an asymmetric Lorentzian will produce `A`, `B`, `H0`, `dH`, `C`, `D`, etc.
    - There is **no appending or aggregation of multiple fit results** to a single file within the fitting step.
- **Returned dictionary:**  
    - The returned dictionary contains the same fields as the CSV file, for automation.
- **No aggregation or batch processing is performed; the pipeline operates on a single processed CSV file per run.**

---

## 9. Plotting and Reporting

- **Robust Filename Parsing and Metadata Handling:**  
    - The pipeline robustly parses processed filenames for metadata by searching for key terms:
        - **Sample name:** String before `"f"` and `"GHz"` (first occurrence).
        - **Frequency:** Number between `"f"` and `"GHz"` (e.g., `f9.8GHz` → `9.8`).
        - **dB value:** Number (with optional minus sign) before `"dB"`. If the substring `"_m"` is present before the dB value, the dB is interpreted as negative (e.g., `_m10dB` → `-10 dB`). If `"_m"` is absent, the dB value is positive (e.g., `_10dB` → `+10 dB`).
    - If any metadata terms are not found, the pipeline falls back to generic metadata (`sample = "Sample"`, `f = ""`, `db = ""`).

- **General Plotting Rules:**
    - Plotting options (such as plotting data as points or as a curve) are controlled by configuration.
    - **Plot titles, axes, and legends are automatically constructed to include sample name, frequency (f), dB value, model type, key fit parameter ($H_{res}$ or $H_0$), and R², matching the conventions in the fitting script.**
- **Save at least two plots for each fit:**
    1. **Data & Aggregated Curve Only:**  
        - Contains the raw data (points or curve according to config) and the aggregated line (e.g., median/mean curve).
        - No fit curve.
    2. **Data, Aggregated Curve, and Fitted Curve:**  
        - Contains the same as above, plus the fitted model curve overlaid.
    - Both plots are to be saved in publication-quality formats (e.g., PNG, SVG).
    - All plots must include axes labels and legends as appropriate.
    - Optionally, save additional plots such as residuals, as specified in the configuration.
- **Reporting:**
    - Titles, legends, and annotations should include parsed metadata (sample, f, dB, model, R², etc.) for clarity and traceability.

---

## 10. Output

- **Plots** (publication-ready, at least the two described above).
- **Fit results CSV** (parameter values, errors, quality metrics).  
    - Output files are saved in the same folder as the processed data file, with filenames automatically encoding sample name, frequency, dB, and fit type.
    - **Fit results CSV files are model-dependent in their header structure.**
    - Each fit result CSV contains only one row of results per fit.
- **Return dictionary** (for automation pipeline).
- **No batch or sample-level aggregation is performed; only single-file results and plots are produced per run.**

---