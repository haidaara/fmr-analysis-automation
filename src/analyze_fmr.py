#!/usr/bin/env python3
"""
Main FMR Analysis Script
Run this to analyze your FMR data.
"""
import os
import sys

# Add the current directory to the path so we can import the other modules
sys.path.append(os.path.dirname(__file__))

from analyzer import analyze_experiment, get_conversion_parameters

def main():
    """Main function for the analysis script"""
    print("FMR Analysis Automation")
    print("=" * 30)
    
    # Get data folder from user
    folder = input("Enter path to your FMR data folder: ").strip()
    if not folder:
        folder = "."  # Current directory
    
    # Get conversion parameters
    slope, intercept = get_conversion_parameters()
    
    # Run analysis
    results = analyze_experiment(folder, slope, intercept)
    
    if results:
        print("\n🎉 Analysis completed successfully!")
    else:
        print("\n❌ Analysis failed!")

if __name__ == "__main__":
    main()