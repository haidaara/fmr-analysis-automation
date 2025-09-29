"""FMR Analysis Automation - Automated analysis of Ferromagnetic Resonance data"""

__version__ = "0.1.0"


# Import main functions to make them easily accessible
from .io import load_labview_data
from .fitting import fit_lorentzian
from .visualization import plot_resonance

# This will be available soon
__all__ = ['load_labview_data', 'fit_lorentzian', 'plot_resonance']