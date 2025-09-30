# New file: src/physics/kittel_fitter.py
import numpy as np
from scipy.optimize import curve_fit

def kittel_equation_in_plane(H_res, gamma, M_s):
    """
    Kittel equation for in-plane geometry:
    f = (γ/2π) * √[H_res * (H_res + M_s)]
    Note: f in GHz, H_res in Tesla
    """
    return (gamma / (2 * np.pi)) * np.sqrt(H_res * (H_res + M_s))

def fit_kittel_dispersion(frequencies, resonance_fields):
    """
    Fit frequency vs resonance field to extract γ/2π and M_s
    """
    # Initial guesses
    gamma_guess = 28.0e9  # 28 GHz/T typical for electrons
    M_s_guess = 0.2       # 0.2 T typical for YIG
    
    try:
        popt, pcov = curve_fit(kittel_equation_in_plane, resonance_fields, frequencies,
                              p0=[gamma_guess, M_s_guess])
        gamma, M_s = popt
        return gamma, M_s, pcov
    except:
        return gamma_guess, M_s_guess, None