# New file: src/physics/damping_analyzer.py
import numpy as np
from scipy.stats import linregress

def extract_gilbert_damping(frequencies, linewidths, gamma):
    """
    Extract Gilbert damping α from linewidth vs frequency
    ΔH_pp = ΔH_0 + (2αf)/(γ)
    """
    # Convert to proper units
    f_Hz = np.array(frequencies) * 1e9  # Convert GHz to Hz
    deltaH = np.array(linewidths)
    
    # Linear fit: ΔH_pp vs f
    slope, intercept, r_value, p_value, std_err = linregress(f_Hz, deltaH)
    
    # Calculate damping: slope = (2α)/γ
    alpha = (slope * gamma) / 2
    
    return {
        'alpha': alpha,
        'deltaH_0': intercept,
        'r_squared': r_value**2,
        'slope': slope
    }