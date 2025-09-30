"""
FMR Plotting Functions - Visualize FMR data and results
"""
import matplotlib.pyplot as plt
import numpy as np

def plot_curve(data, show_raw=False, save_path=None):
    """
    Plot single FMR curve
    """
    metadata = data['metadata']
    processed = data['processed']
    
    plt.figure(figsize=(10, 6))
    
    if show_raw:
        # Plot all raw data points
        plt.plot(data['H_field'], data['signal'], 'o', 
                alpha=0.3, label='Raw data', color='gray', markersize=3)
    
    # Plot processed curve
    plt.plot(processed['H_unique'], processed['signal_mean'], 
            'r-', linewidth=2, label='Averaged')
    
    # Show uncertainty as shaded area
    plt.fill_between(processed['H_unique'], 
                    processed['signal_mean'] - processed['signal_std'],
                    processed['signal_mean'] + processed['signal_std'],
                    alpha=0.2, color='red', label='±1 std')
    
    # Mark resonance field
    resonance = data['metadata'].get('resonance_field', 
                                    np.min(processed['H_unique']))
    plt.axvline(resonance, color='green', linestyle='--', 
               label=f'Resonance: {resonance:.3f} T')
    
    plt.xlabel('Magnetic Field (T)')
    plt.ylabel('dP/dH (arb. units)')
    plt.title(f"{metadata['sample']} - {metadata['frequency']} GHz ({metadata['decibel']} dB)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return plt.gcf()

def plot_multiple_curves(curves_list, save_path=None):
    """
    Plot multiple curves for comparison (same sample, different frequencies)
    """
    if not curves_list:
        return
    
    sample_name = curves_list[0]['metadata']['sample']
    
    plt.figure(figsize=(12, 8))
    
    for data in curves_list:
        metadata = data['metadata']
        processed = data['processed']
        
        label = f"{metadata['frequency']} GHz"
        plt.plot(processed['H_unique'], processed['signal_mean'], 
                label=label, linewidth=2)
        
        # Mark resonance
        resonance = metadata.get('resonance_field')
        if resonance:
            plt.axvline(resonance, linestyle='--', alpha=0.5,
                       color=plt.gca().lines[-1].get_color())
    
    plt.xlabel('Magnetic Field (T)')
    plt.ylabel('dP/dH (arb. units)')
    plt.title(f"{sample_name} - Multiple Frequencies")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return plt.gcf()

def plot_resonance_dispersion(experiment_data, save_path=None):
    """
    Plot resonance field vs frequency for each sample
    """
    plt.figure(figsize=(10, 6))
    
    for sample, curves in experiment_data.items():
        frequencies = []
        resonances = []
        
        for curve in curves:
            freq = curve['metadata']['frequency']
            resonance = curve['metadata'].get('resonance_field')
            
            if resonance:
                frequencies.append(freq)
                resonances.append(resonance)
        
        if frequencies:
            # Sort by frequency
            sort_idx = np.argsort(frequencies)
            frequencies = np.array(frequencies)[sort_idx]
            resonances = np.array(resonances)[sort_idx]
            
            plt.plot(frequencies, resonances, 'o-', label=sample, linewidth=2, markersize=6)
    
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Resonance Field (T)')
    plt.title('FMR Resonance Dispersion')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return plt.gcf()

# In plotter.py - ADD new functions
def plot_lorentzian_fit(data, save_path=None):
    """Plot FMR curve with Lorentzian fit"""
    H = data['processed']['H_unique']
    signal = data['processed']['signal_mean']
    H_res = data['metadata']['resonance_field']
    deltaH = data['metadata']['linewidth']
    
    plt.figure(figsize=(10, 6))
    plt.plot(H, signal, 'o-', label='Data', linewidth=2)
    
    # Plot fit (you'll need to reconstruct the fit curve)
    # ... fitting curve plotting code ...
    
    plt.axvline(H_res, color='red', linestyle='--', 
                label=f'H_res = {H_res:.3f} T')
    plt.xlabel('Magnetic Field (T)')
    plt.ylabel('dP/dH (arb. units)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

def plot_kittel_fit(kittel_results, save_path=None):
    """Plot Kittel fit with experimental data"""
    plt.figure(figsize=(10, 6))
    
    for sample, result in kittel_results.items():
        f = result['frequencies']
        H_res = result['resonance_fields']
        
        plt.plot(H_res, f, 'o', label=f'{sample} data', markersize=6)
        
        # Plot fit curve
        H_fit = np.linspace(min(H_res), max(H_res), 100)
        f_fit = (result['gamma/2pi']) * np.sqrt(H_fit * (H_fit + result['M_s']))
        plt.plot(H_fit, f_fit, '-', label=f'{sample} fit')
    
    plt.xlabel('Resonance Field (T)')
    plt.ylabel('Frequency (GHz)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')