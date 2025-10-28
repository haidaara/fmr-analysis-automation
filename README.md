# FMR Analysis Automation

This repository provides a fully automated pipeline for Ferromagnetic Resonance (FMR) data analysis, including spectrum fitting, Kittel law extraction, damping parameter analysis, and report generation. The workflow is controlled by a single YAML configuration file.

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/haidaara/fmr-analysis-automation.git
cd fmr-analysis-automation
```

### 2. Update the Repository

To get the latest features and bug fixes:

```bash
git pull origin main
```

### 3. Prepare Your Environment


```bash
# Update
sudo apt-get update

# Update pip and setuptools
python -m pip install --upgrade pip setuptools
```


#### Install Python Requirements


```bash
#Create and Activate a Virtual Environment (Optional but recommended)
python -m venv venv
source venv/bin/activate 
```

```bash
pip install -r requirements.txt
```

> **Note:** The requirements include all scientific Python libraries needed (`numpy`, `pandas`, `matplotlib`, `scipy`, `lmfit`, `pyyaml`, `openpyxl`, `python-pptx`, etc.), with versions tested up to Python 3.12.  
---

## Running the Project

The main pipeline is controlled by the `automation.py` script and the `config.yaml` configuration file.

```bash
python scripts/automation.py --config config.yaml
# or just 
python scripts/automation.py
```

This will:
- Process raw data files for each sample
- Perform spectrum fitting (single, double, asymmetric Lorentzian derivatives)
- Aggregate fit results
- Run Kittel and damping analysis automatically
- Generate Excel summaries and PowerPoint reports for each sample

---

## Configuration: `config.yaml`

All processing options are set in `config.yaml`. The most important parameters to adjust for your workflow are:

```yaml
# General paths
data_folder: "data"                  # Where your raw data is stored
results_root: "results_roots"        # Where all results and plots are saved

# Which samples to process
samples: [AZ9, AZ3, ...]             # List of sample names (folders)

# Fitting model (for spectrum fitting)
model: "double_lorentzain"           # Options: "Lorentzain", "aysmetric", "double_lorentzain"

# Plotting options
plot:
  plot_formats: ["png", "pdf"]       # Output formats
  show_legend: true
  font_scale: 1.2
  line_width: 2.5
  marker_size: 6
  grid: true
```

---

**If you have any issues or if the pipeline does not behave as expected, please [open an issue request on GitHub and tag @haidaara](https://github.com/haidaara/fmr-analysis-automation/issues/new) for direct help.**  
Please include a short description, error message, and (if possible) your config file and a sample of your data.