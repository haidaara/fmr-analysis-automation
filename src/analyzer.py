"""
Main FMR Analysis Workflow - Complete analysis pipeline
"""
import os
from pathlib import Path
from .loader import load_folder
from .processor import process_all_curves, find_resonance
from .plotter import plot_curve, plot_multiple_curves, plot_resonance_dispersion

def analyze_experiment(folder_path, slope=1.0, intercept=0.0, save_plots=True):
    """
    Complete FMR analysis workflow
    
    Args:
        folder_path: Path to data folder
        slope: Current to field conversion slope
        intercept: Current to field conversion intercept  
        save_plots: Whether to save plots to disk
    """
    # Create output directory
    output_dir = Path(folder_path) / "analysis_results"
    output_dir.mkdir(exist_ok=True)
    
    print("🔬 Starting FMR Analysis")
    print("=" * 50)
    
    # 1. Load data
    print("📁 Loading data...")
    raw_data = load_folder(folder_path, slope, intercept)
    
    if not raw_data:
        print("❌ No data loaded!")
        return {}
    
    # 2. Process data with Lorentzian fitting
    processed_data = process_all_curves(raw_data)
    
    # 3. Physics analysis
    print("\n🔬 Physics Analysis...")
    
    # Kittel analysis
    kittel_results = analyze_kittel_dispersion(processed_data)
    
    # Damping analysis  
    damping_results = analyze_gilbert_damping(processed_data, kittel_results)
    
    # 4. Generate physics plots
    if save_plots:
        # Plot Kittel fits
        plot_path = output_dir / "kittel_fits.png"
        plot_kittel_fit(kittel_results, save_path=plot_path)
        
        # Plot individual Lorentzian fits
        for sample, curves in processed_data.items():
            for curve in curves:
                plot_path = output_dir / f"{sample}_{curve['metadata']['frequency']}GHz_fit.png"
                plot_lorentzian_fit(curve, save_path=plot_path)
    
    # 5. Print results
    print("\n📊 Physics Results:")
    for sample in kittel_results:
        print(f"  {sample}:")
        print(f"    γ/2π = {kittel_results[sample]['gamma/2pi']:.2f} GHz/T")
        print(f"    M_s = {kittel_results[sample]['M_s']:.3f} T")
        if sample in damping_results:
            print(f"    α = {damping_results[sample]['alpha']:.2e}")
    
    return {
        'processed_data': processed_data,
        'kittel_results': kittel_results, 
        'damping_results': damping_results
    }

def get_conversion_parameters():
    """
    Helper to get current-to-field conversion parameters from user
    """
    print("🔧 Current-to-Field Conversion Setup")
    print("Formula: H_field = slope × current + intercept")
    
    try:
        slope = float(input("Enter slope (T/A) [default: 1.0]: ") or 1.0)
        intercept = float(input("Enter intercept (T) [default: 0.0]: ") or 0.0)
        return slope, intercept
    except:
        print("Using default values: slope=1.0, intercept=0.0")
        return 1.0, 0.0


from .physics.kittel_fitter import fit_kittel_dispersion

def analyze_kittel_dispersion(processed_data):
    """Analyze frequency vs resonance field for each sample"""
    kittel_results = {}
    
    for sample, curves in processed_data.items():
        frequencies = []
        resonance_fields = []
        
        for curve in curves:
            freq = curve['metadata']['frequency']
            H_res = curve['metadata']['resonance_field']
            
            if H_res:  # Only use curves with valid resonance
                frequencies.append(freq)
                resonance_fields.append(H_res)
        
        if len(frequencies) >= 3:  # Need at least 3 points for fitting
            gamma, M_s, covariance = fit_kittel_dispersion(frequencies, resonance_fields)
            kittel_results[sample] = {
                'gamma/2pi': gamma / (2 * np.pi),  # Convert to GHz/T
                'M_s': M_s,
                'frequencies': frequencies,
                'resonance_fields': resonance_fields
            }
    
    return kittel_results

# In analyzer.py - ADD to main workflow
from .physics.damping_analyzer import extract_gilbert_damping

def analyze_gilbert_damping(processed_data, kittel_results):
    """Analyze damping for each sample"""
    damping_results = {}
    
    for sample, curves in processed_data.items():
        if sample not in kittel_results:
            continue
            
        frequencies = []
        linewidths = []
        
        for curve in curves:
            freq = curve['metadata']['frequency']
            linewidth = curve['metadata'].get('linewidth')
            
            if linewidth:
                frequencies.append(freq)
                linewidths.append(linewidth)
        
        if len(frequencies) >= 3:
            gamma = kittel_results[sample]['gamma/2pi'] * 2 * np.pi  # Convert back to rad/s
            damping = extract_gilbert_damping(frequencies, linewidths, gamma)
            damping_results[sample] = damping
    
    return damping_results