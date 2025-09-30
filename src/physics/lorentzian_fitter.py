# src/physics/lorentzian_fitter.py
import numpy as np
from scipy.optimize import curve_fit

def lorentzian_derivative(H, H_res, deltaH, A, offset):
    """
    Lorentzian derivative function for FMR signals
    dP/dH = A * (H - H_res) / [(H - H_res)² + (ΔH/2)²]² + offset
    """
    return A * (H - H_res) / ((H - H_res)**2 + (deltaH/2)**2)**2 + offset

def fit_fmr_curve(H, signal):
    """Fit experimental data to Lorentzian derivative"""
    # Initial parameter guesses
    H_res_guess = H[np.argmin(signal)]  # Current method as starting point
    deltaH_guess = (np.max(H) - np.min(H)) * 0.1  # 10% of range
    A_guess = np.min(signal) - np.max(signal)
    offset_guess = np.mean(signal)
    
    # Perform fit
    try:
        popt, pcov = curve_fit(lorentzian_derivative, H, signal, 
                              p0=[H_res_guess, deltaH_guess, A_guess, offset_guess])
        H_res, deltaH, A, offset = popt
        return H_res, deltaH, A, offset
    except:
        # Fallback to current method if fit fails
        return H_res_guess, deltaH_guess, A_guess, offset_guess