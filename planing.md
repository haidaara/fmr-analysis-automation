From thinking mode: 

fmr-analysis/
│
├── .github/
│   └── workflows/          # For CI/CD (if needed)
├── config/                 # Configuration files (e.g., for fitting parameters)
├── data/                   # Example data and tests (should be in .gitignore for real data)
├── docs/                   # Documentation
├── src/
│   └── fmr_analysis/       # Main package
│       ├── __init__.py
│       ├── data_loader.py
│       ├── curve_fitting.py
│       ├── parameter_extraction.py
│       ├── visualization.py
│       └── main_pipeline.py
├── tests/                  # Unit tests
├── examples/               # Example scripts and Jupyter notebooks
├── README.md
├── requirements.txt
└── setup.py (or pyproject.toml for modern packages)

---

## Foundation Setup - Detailed Breakdown

Let's focus ONLY on the foundation setup. This is what you need to create first:

### 1. Repository Structure Creation

```bash
# Create the main project directory
mkdir fmr-analysis-automation
cd fmr-analysis-automation

# Create the complete directory structure
mkdir -p .github/workflows
mkdir -p docs
mkdir -p examples
mkdir -p src/fmr_analysis/{core,io,fitting,visualization,cli}
mkdir -p tests/{unit,integration}
mkdir -p notebooks/{exploration,production}
mkdir -p scripts
mkdir -p requirements
mkdir -p config
mkdir -p data/{raw,processed}
mkdir -p results/{figures,parameters}
```

### 2. Essential Files to Create

#### **Package Initialization Files:**
```python
# src/fmr_analysis/__init__.py
"""FMR Analysis Automation - Automated analysis of Ferromagnetic Resonance data"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@aub.edu.lb"

from .core import *
from .io import *
from .fitting import *
```

```python
# src/fmr_analysis/core/__init__.py
"""Core FMR analysis functionality"""
```

```python
# src/fmr_analysis/io/__init__.py  
"""Data loading and input/output modules"""
```

```python
# src/fmr_analysis/fitting/__init__.py
"""Curve fitting and parameter extraction"""
```

```python
# src/fmr_analysis/visualization/__init__.py
"""Plotting and visualization utilities"""
```

```python
# src/fmr_analysis/cli/__init__.py
"""Command-line interface modules"""
```

#### **Dependency Management:**
```txt
# requirements/base.txt
numpy>=1.21.0
scipy>=1.7.0
pandas>=1.3.0
matplotlib>=3.5.0
```

```txt
# requirements/dev.txt
-r base.txt
pytest>=6.0
pytest-cov
black
flake8
mypy
jupyter
ipython
```

```txt
# requirements/notebooks.txt  
-r base.txt
jupyter>=1.0.0
ipywidgets
plotly
seaborn
```

#### **Configuration Files:**
```python
# config/default_analysis.yaml
# Default analysis parameters
fitting:
  lorentzian:
    max_iterations: 1000
    tolerance: 1e-6
  kittel:
    guess_gamma: 28.0  # GHz/T
    guess_ms: 0.2      # T
visualization:
  style: "seaborn"
  dpi: 300
```

```python
# setup.py
from setuptools import setup, find_packages

setup(
    name="fmr_analysis",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0", 
        "pandas>=1.3.0",
        "matplotlib>=3.5.0",
    ],
    python_requires=">=3.8",
)
```

#### **Documentation & Guides:**
```markdown
# README.md
# FMR Analysis Automation

Automated analysis pipeline for Ferromagnetic Resonance data.

## Quick Start

```python
from fmr_analysis import load_data, analyze_sample
results = analyze_sample("path/to/your/data")
```

## Development
See [docs/development.md](docs/development.md) for setup instructions.
```

```markdown
# docs/development.md
# Development Setup

1. Clone repository
2. Create virtual environment: `python -m venv venv`
3. Activate: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
4. Install: `pip install -e .`
5. Install dev requirements: `pip install -r requirements/dev.txt`
```

#### **Git Configuration:**
```gitignore
# .gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environment
venv/
env/

# Jupyter
.ipynb_checkpoints

# Data
data/raw/
data/processed/
results/

# IDE
.vscode/
.idea/
```

### 3. Initial Testing Structure

```python
# tests/__init__.py
"""Test suite for FMR analysis"""
```

```python
# tests/test_imports.py
"""Basic import tests to verify setup"""

def test_import_package():
    """Test that the package can be imported"""
    import fmr_analysis
    assert fmr_analysis.__version__ == "0.1.0"

def test_import_submodules():
    """Test that submodules can be imported"""
    from fmr_analysis import core, io, fitting
    assert core is not None
    assert io is not None  
    assert fitting is not None
```

### 4. Initial Exploration Notebook

```python
# notebooks/exploration/01_data_structure.ipynb
"""
# FMR Data Structure Exploration

Understanding the LabView export format and data organization.
"""
# Basic imports to test the environment
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../src')  # For development

print("FMR Analysis environment ready!")
```

### 5. Git Branch Setup Commands

```bash
# Initialize git repository
git init

# Create and switch to foundation branch
git checkout -b feature/foundation-setup

# Add all files
git add .

# Initial commit
git commit -m "feat: initial foundation setup

- Create package structure with core, io, fitting, visualization modules
- Add dependency management with requirements/
- Add basic configuration files
- Add testing framework foundation
- Add documentation structure"

# Create develop branch
git checkout -b develop

# Push to remote (when you create the GitHub repo)
git remote add origin <your-repo-url>
git push -u origin develop
git push -u origin feature/foundation-setup
```

### 6. Verification Script

```python
# scripts/verify_setup.py
#!/usr/bin/env python3
"""Verify the foundation setup is correct"""

import os
import sys

def check_structure():
    """Verify the directory structure"""
    required_dirs = [
        'src/fmr_analysis/core',
        'src/fmr_analysis/io', 
        'src/fmr_analysis/fitting',
        'src/fmr_analysis/visualization',
        'src/fmr_analysis/cli',
        'tests/unit',
        'notebooks/exploration',
        'requirements'
    ]
    
    missing = []
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing.append(dir_path)
    
    return missing

def check_requirements():
    """Check if required files exist"""
    required_files = [
        'requirements/base.txt',
        'requirements/dev.txt',
        'src/fmr_analysis/__init__.py',
        'README.md'
    ]
    
    missing = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing.append(file_path)
    
    return missing

if __name__ == "__main__":
    print("Verifying foundation setup...")
    
    missing_dirs = check_structure()
    missing_files = check_requirements()
    
    if not missing_dirs and not missing_files:
        print("✅ Foundation setup complete!")
        sys.exit(0)
    else:
        print("❌ Missing components:")
        for item in missing_dirs + missing_files:
            print(f"   - {item}")
        sys.exit(1)
```

This foundation gives you:
- ✅ Clean, modular package structure
- ✅ Dependency management  
- ✅ Testing framework
- ✅ Documentation structure
- ✅ Both notebook and script workflows
- ✅ Git branch strategy ready

Once this foundation is solid, we can move to the data loader feature. Ready to proceed?
