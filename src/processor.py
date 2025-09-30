"""
FMR Data Processor - Process and analyze FMR curves
"""
import numpy as np
from .physics.lorentzian_fitter import fit_fmr_curve

def process_curve(data):
    """
    Process single FMR curve: sort and average repeated points
    """
    H = data['H_field']
    signal = data['signal']
    
    # Sort by magnetic field (low to high)
    sort_idx = np.argsort(H)
    H_sorted = H[sort_idx]
    signal_sorted = signal[sort_idx]
    
    # Find unique field points and average repeats
    H_unique, indices, counts = np.unique(H_sorted, return_inverse=True, return_counts=True)
    
    signal_mean = np.array([signal_sorted[indices == i].mean() for i in range(len(H_unique))])
    signal_std = np.array([signal_sorted[indices == i].std() for i in range(len(H_unique))])
    
    # Add processed data to original dictionary
    data['processed'] = {
        'H_sorted': H_sorted,
        'signal_sorted': signal_sorted,
        'H_unique': H_unique,
        'signal_mean': signal_mean,
        'signal_std': signal_std,
        'n_repeats': counts
    }
    
    return data




def find_resonance(data):
    """Find resonance using Lorentzian fitting"""
    if 'processed' not in data:
        data = process_curve(data)
    
    H = data['processed']['H_unique']
    signal = data['processed']['signal_mean']
    
    # Use Lorentzian fitting instead of simple minimum
    H_res, deltaH, amplitude, offset = fit_fmr_curve(H, signal)
    
    # Store all parameters
    data['metadata']['resonance_field'] = H_res
    data['metadata']['linewidth'] = deltaH  # Peak-to-peak linewidth
    data['metadata']['fmr_amplitude'] = amplitude
    
    return H_res

def process_all_curves(experiment_data):
    """
    Process all curves in an experiment
    """
    processed_data = {}
    
    for sample, curves in experiment_data.items():
        processed_data[sample] = []
        
        for curve in curves:
            processed_curve = process_curve(curve)
            resonance = find_resonance(processed_curve)
            
            # Add resonance to metadata
            processed_curve['metadata']['resonance_field'] = resonance
            processed_data[sample].append(processed_curve)
    
    return processed_data