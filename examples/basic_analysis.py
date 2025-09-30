"""
Basic FMR Analysis Example
"""
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from loader import load_folder
from processor import process_all_curves
from plotter import plot_curve, plot_multiple_curves
import matplotlib.pyplot as plt

def main():
    """Basic analysis workflow"""
    
    # 1. Load data
    print("Loading data...")
    data = load_folder("../data/sample/")  # Update path as needed
    
    if not data:
        print("No data found!")
        return
    
    # 2. Process data
    print("Processing curves...")
    processed_data = process_all_curves(data)
    
    # 3. Analyze and plot
    for sample_name, curves in processed_data.items():
        print(f"\nSample: {sample_name}")
        
        # Plot each curve
        for curve in curves:
            freq = curve['metadata']['frequency']
            resonance = curve['metadata'].get('resonance_field', 'N/A')
            print(f"  {freq} GHz - Resonance: {resonance}")
            
            # Create plot
            plot_curve(curve)
            plt.show()
        
        # Plot all frequencies together
        if len(curves) > 1:
            plot_multiple_curves(curves)
            plt.show()

if __name__ == "__main__":
    main()