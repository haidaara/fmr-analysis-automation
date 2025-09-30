"""
Custom FMR Analysis Pipeline Example
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from loader import load_single_file
from processor import process_curve, find_resonance
from plotter import plot_curve
import matplotlib.pyplot as plt

def custom_analysis(file_path):
    """Custom analysis for single file"""
    
    # Load single file with custom conversion
    data = load_single_file(file_path, slope=1.0, intercept=0.0)
    
    # Process the curve
    processed = process_curve(data)
    
    # Find resonance
    resonance = find_resonance(processed)
    print(f"Resonance field: {resonance:.3f} T")
    
    # Create custom plot
    plt.figure(figsize=(10, 6))
    plot_curve(processed, show_raw=True)
    plt.title(f"Custom Analysis - {processed['metadata']['sample']}")
    plt.show()
    
    return processed

if __name__ == "__main__":
    # Example usage
    file_path = "../data/sample/AZ5_f5GHz_m10dB_1_t2.txt"  # Update path
    result = custom_analysis(file_path)