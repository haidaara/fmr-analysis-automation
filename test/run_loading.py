import sys
import os

# Add the scripts directory to sys.path for import
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
scripts_dir = os.path.join(project_root, "scripts")
sys.path.append(scripts_dir)

from loading_setup import loading_setup

# Updated usage: no data_path, use data_folder + parameters
data_folder = os.path.join(project_root, "data")
results_root = os.path.join(project_root, "results_roots")
sample = "AZ5"
f =11
db =None

loading_setup(data_folder, results_root, sample, f, db)

# Run this test with: python test/run_loading.py