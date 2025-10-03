"""
Example script demonstrating the usage of loading_setup function
"""
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from loader import loading_setup

def example_basic_usage():
    """
    Example 1: Basic usage with default parameters
    """
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    
    # Define paths
    results_root = "/path/to/your/results"
    sample_name = "MySample"
    
    # Load spectrum with default parameters (3 GHz, -10 dB)
    try:
        df = loading_setup(results_root, sample_name)
        
        print(f"\n✓ Successfully loaded data for {sample_name}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Number of rows: {len(df)}")
        print("\nFirst few rows:")
        print(df.head())
        
        # Show magnetic field calculation
        print("\nMagnetic Field Calculation:")
        print(f"  Formula: H = 64.4 × I + 7.914")
        print(f"  Example: H[0] = 64.4 × {df['Current (I)'].iloc[0]:.4f} + 7.914 = {df['Magnetic Field (H)'].iloc[0]:.4f}")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("   Make sure the results_root and sample_name are correct")
    except ValueError as e:
        print(f"❌ Error: {e}")


def example_custom_parameters():
    """
    Example 2: Loading with custom frequency and decibel
    """
    print("\n" + "=" * 60)
    print("Example 2: Custom Frequency and Decibel")
    print("=" * 60)
    
    results_root = "/path/to/your/results"
    sample_name = "MySample"
    
    # Load spectrum with custom parameters
    try:
        # Load data at 5 GHz and -15 dB
        df = loading_setup(results_root, sample_name, f=5.0, db=-15)
        
        print(f"\n✓ Loaded data at 5 GHz, -15 dB")
        print(f"  Data shape: {df.shape}")
        print(f"  Current range: {df['Current (I)'].min():.4f} to {df['Current (I)'].max():.4f} A")
        print(f"  H range: {df['Magnetic Field (H)'].min():.2f} to {df['Magnetic Field (H)'].max():.2f}")
        
    except FileNotFoundError as e:
        print(f"❌ No file found for these parameters: {e}")


def example_multiple_frequencies():
    """
    Example 3: Loading multiple frequencies for the same sample
    """
    print("\n" + "=" * 60)
    print("Example 3: Multiple Frequencies")
    print("=" * 60)
    
    results_root = "/path/to/your/results"
    sample_name = "MySample"
    frequencies = [3.0, 5.0, 7.0]
    
    print(f"\nLoading data for sample '{sample_name}' at multiple frequencies:")
    
    for freq in frequencies:
        try:
            df = loading_setup(results_root, sample_name, f=freq, db=-10)
            print(f"  ✓ {freq} GHz: {len(df)} data points")
        except FileNotFoundError:
            print(f"  ❌ {freq} GHz: File not found")


def example_error_handling():
    """
    Example 4: Proper error handling
    """
    print("\n" + "=" * 60)
    print("Example 4: Error Handling")
    print("=" * 60)
    
    results_root = "/path/to/your/results"
    sample_name = "NonExistentSample"
    
    try:
        df = loading_setup(results_root, sample_name, f=3.0, db=-10)
        print(f"✓ Loaded {len(df)} rows")
        
    except FileNotFoundError as e:
        print(f"❌ FileNotFoundError: {e}")
        print("   This happens when:")
        print("   - Sample directory doesn't exist")
        print("   - No file matches the frequency/decibel parameters")
        
    except ValueError as e:
        print(f"❌ ValueError: {e}")
        print("   This happens when:")
        print("   - File doesn't have required columns")
        print("   - Data cannot be parsed")


def create_test_data():
    """
    Example 5: Create test data and load it
    """
    print("\n" + "=" * 60)
    print("Example 5: Create and Load Test Data")
    print("=" * 60)
    
    import tempfile
    import shutil
    
    # Create temporary directory structure
    test_dir = tempfile.mkdtemp()
    sample_dir = Path(test_dir) / "TestSample"
    sample_dir.mkdir()
    
    # Create test data file
    test_file = sample_dir / "TestSample_f3GHz_m10dB_1.txt"
    with open(test_file, 'w') as f:
        f.write("0.1 0.5\n")
        f.write("0.2 0.6\n")
        f.write("0.3 0.7\n")
        f.write("0.4 0.8\n")
    
    print(f"\n📁 Created test directory: {test_dir}")
    print(f"📄 Created test file: {test_file.name}")
    
    # Load the data
    try:
        df = loading_setup(test_dir, "TestSample", f=3.0, db=-10)
        
        print(f"\n✓ Successfully loaded test data")
        print(f"\nData preview:")
        print(df.to_string(index=False))
        
        print(f"\n✓ File was processed and saved back to: {test_file}")
        
    finally:
        # Cleanup
        shutil.rmtree(test_dir)
        print(f"\n🧹 Cleaned up test directory")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Loading Setup Function - Usage Examples")
    print("=" * 60)
    
    # Note: Most examples will fail without actual data files
    # Run example 5 to see a working demonstration
    
    # Uncomment the examples you want to run:
    # example_basic_usage()
    # example_custom_parameters()
    # example_multiple_frequencies()
    # example_error_handling()
    
    # This one actually works because it creates test data:
    create_test_data()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
