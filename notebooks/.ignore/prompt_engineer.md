We have developed a FMR (Ferromagnetic Resonance) analysis automation tool. The project is structured as a Python package that provides a complete workflow for loading, processing, visualizing, and analyzing FMR data.

*Project Structure*
The project is organized as follows:

```bash
fmr-analysis-automation/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── src/
│   ├── __init__.py
│   ├── loader.py
│   ├── processor.py
│   ├── plotter.py
│   ├── analyzer.py
│   └── analyze_fmr.py
├── docs/
│   └── basic_usage.md
├── examples/
│   ├── basic_analysis.py
│   └── custom_pipeline.py
├── data/
│   └── sample/
│       └── README.md
└── tests/
    ├── __init__.py
    ├── test_loader.py
    └── test_processor.py
```

*Key Features*
1. *Data Loading:*
    Automatically detects and loads FMR data files from a directory.

    Parses metadata (sample name, frequency, decibel, replicate) from filenames.

    Converts current values to magnetic field using a linear transformation (slope and intercept).

2. *Data Processing:*
    Sorts data by magnetic field (low to high).

    Averages repeated measurements at the same field point.

    Computes standard deviation for error analysis.

    Identifies resonance field (minimum of the derivative signal).

3. *Visualization:*

    Plots individual FMR curves with raw data and averaged curve.

    Plots multiple curves for comparison (same sample, different frequencies).

    Plots resonance dispersion (resonance field vs frequency).

4. *Automation:*

    Complete analysis workflow with one function call.

    Interactive command-line script for easy use.

    Saves plots to disk automatically.


    ---


    # FMR Analysis Automation - Simple Project Summary

## Project Goal
**Automate FMR data analysis** to replace manual work in LabView and Origin for analyzing Yttrium Iron Garnet (YIG) thin films.

## What We Built
A Python tool that automatically analyzes Ferromagnetic Resonance (FMR) data from our lab experiments.

### 🔧 **What It Does**
1. **Loads FMR files** automatically - finds all data files in a folder
2. **Processes the data** - converts current to magnetic field, finds resonance peaks
3. **Creates plots** - makes the same graphs we manually create in Origin
4. **Extracts key parameters** - resonance fields, saturation magnetization, Gilbert damping

### 📊 **Solves Our Lab's Problem**
- **Before**: Manual analysis for each frequency (2-17 GHz) in LabView + Origin
- **After**: One command processes all frequencies automatically
- **Saves time**: Hours of repetitive work → minutes of automated processing

## Technical Implementation

### **File Structure**
```
fmr-analysis-automation/
├── src/                          # Main code
│   ├── loader.py                # Reads LabView files
│   ├── processor.py             # Analyzes FMR curves
│   ├── plotter.py               # Makes publication plots
│   ├── analyzer.py              # Complete workflow
│   └── analyze_fmr.py           # Run this to analyze data
├── examples/                    # How to use the code
└── docs/                        # Instructions
```

### **Key Features**
- **Automatic file detection** - finds all `AZ5_f5GHz_m10dB_*.txt` files
- **Current-to-field conversion** - uses lab calibration parameters
- **Resonance finding** - locates FMR peaks automatically
- **Multi-frequency analysis** - processes all 17 frequencies at once
- **Publication-ready plots** - creates figures like in our poster

### **How to Use**
```bash
# Simple one-command analysis
python src/analyze_fmr.py

# Or use in Python code
from src import analyze_experiment
results = analyze_experiment("path/to/my/YIG_data")
```

## Connection to Our Research

### **Matches Poster Analysis**
This tool automates exactly what we show in the FMR Measurement section:
- **FMR signals** → Loads and processes derivative curves
- **Dispersion fitting** → Extracts γ and Mₛ from frequency vs H_res
- **Linewidth analysis** → Calculates Gilbert damping α
- **Parameter studies** → Analyzes laser energy/O₂ pressure effects

### **Supports Our Thin Film Research**
- Processes our **30nm YIG films** data
- Handles **multiple frequencies** (2-17 GHz)
- Works with **different deposition conditions** (laser energy, O₂ pressure)
- Extracts **key parameters**: M_s, α, γ/2π

## Current Status
**✅ Complete**: Basic FMR analysis pipeline
**✅ Complete**: Data loading and processing
**✅ Complete**: Plotting and visualization
**🔄 Ready for testing**: Needs real YIG film data to validate

## Simple Engineer Prompt

```
BUILD_SCIENCE_TOOL:
I need a Python tool to automate [technique] data analysis.

MY_DATA:
- Files named: [sample]_[condition]_[parameter].txt
- Contains: [raw measurements]
- Need to extract: [key parameters]

WHAT_I_DO_MANUALLY:
1. Load each file in [software]
2. [Process step 1]
3. [Process step 2] 
4. Plot graphs in [software]
5. Extract [parameters]

AUTOMATION_GOALS:
- Load all files automatically
- Process data to get [parameters]
- Create standard plots
- Save results in organized way

OUTPUT_NEEDED:
- Processed data files
- Publication-ready figures
- Summary of extracted parameters
```

This tool directly supports our YIG thin film research by automating the FMR analysis we described in the poster, making our data processing faster and more reproducible.