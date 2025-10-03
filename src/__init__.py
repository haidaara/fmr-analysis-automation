"""
FMR Analysis Automation Package
Simple tools for analyzing Ferromagnetic Resonance data
"""

from .loader import load_single_file, load_folder, parse_filename, loading_setup
from .processor import process_curve, find_resonance, process_all_curves
from .plotter import plot_curve, plot_multiple_curves, plot_resonance_dispersion
from .analyzer import analyze_experiment, get_conversion_parameters

__version__ = "1.0.0"
__author__ = "FMR Research Group"

__all__ = [
    'load_single_file',
    'load_folder', 
    'parse_filename',
    'loading_setup',
    'process_curve',
    'find_resonance',
    'process_all_curves',
    'plot_curve',
    'plot_multiple_curves',
    'plot_resonance_dispersion',
    'analyze_experiment',
    'get_conversion_parameters'
]