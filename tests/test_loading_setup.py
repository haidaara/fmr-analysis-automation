"""
Tests for loading_setup function
"""
import unittest
import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from loader import loading_setup
import pandas as pd


class TestLoadingSetup(unittest.TestCase):
    
    def setUp(self):
        """Create temporary directory structure and test files"""
        self.test_dir = tempfile.mkdtemp()
        self.sample_name = "TestSample"
        self.sample_dir = Path(self.test_dir) / self.sample_name
        self.sample_dir.mkdir()
        
    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.test_dir)
    
    def create_test_file(self, filename, data):
        """Helper to create test data files"""
        file_path = self.sample_dir / filename
        with open(file_path, 'w') as f:
            f.write(data)
        return file_path
    
    def test_loading_setup_basic(self):
        """Test basic loading with default parameters"""
        # Create a test file
        test_data = """0.1 0.5
0.2 0.6
0.3 0.7
0.4 0.8"""
        self.create_test_file("TestSample_f3GHz_m10dB_1.txt", test_data)
        
        # Load the data
        df = loading_setup(self.test_dir, self.sample_name, f=3.0, db=-10)
        
        # Verify structure
        self.assertIn('Current (I)', df.columns)
        self.assertIn('dP/dt', df.columns)
        self.assertIn('Magnetic Field (H)', df.columns)
        
        # Verify data
        self.assertEqual(len(df), 4)
        
        # Verify H calculation: H = 64.4 * I + 7.914
        expected_H = 64.4 * 0.1 + 7.914
        self.assertAlmostEqual(df['Magnetic Field (H)'].iloc[0], expected_H, places=5)
    
    def test_loading_setup_with_headers(self):
        """Test loading file with column headers"""
        test_data = """Current,Signal
0.1,0.5
0.2,0.6
0.3,0.7"""
        self.create_test_file("TestSample_f5GHz_m10dB_1.csv", test_data)
        
        df = loading_setup(self.test_dir, self.sample_name, f=5.0, db=-10)
        
        self.assertEqual(len(df), 3)
        self.assertIn('Current (I)', df.columns)
        self.assertIn('dP/dt', df.columns)
    
    def test_loading_setup_custom_frequency(self):
        """Test loading with custom frequency"""
        test_data = """0.1 0.5
0.2 0.6"""
        self.create_test_file("TestSample_f7GHz_m10dB_1.txt", test_data)
        
        df = loading_setup(self.test_dir, self.sample_name, f=7.0, db=-10)
        self.assertEqual(len(df), 2)
    
    def test_loading_setup_missing_file(self):
        """Test error handling for missing file"""
        with self.assertRaises(FileNotFoundError):
            loading_setup(self.test_dir, self.sample_name, f=99.0, db=-10)
    
    def test_loading_setup_missing_sample_dir(self):
        """Test error handling for missing sample directory"""
        with self.assertRaises(FileNotFoundError):
            loading_setup(self.test_dir, "NonExistentSample", f=3.0, db=-10)
    
    def test_loading_setup_invalid_data(self):
        """Test error handling for invalid data"""
        test_data = """invalid data
not numbers"""
        self.create_test_file("TestSample_f3GHz_m10dB_1.txt", test_data)
        
        with self.assertRaises(ValueError):
            loading_setup(self.test_dir, self.sample_name, f=3.0, db=-10)
    
    def test_magnetic_field_calculation(self):
        """Test that magnetic field is calculated correctly"""
        test_data = """0.0 1.0
0.1 2.0
0.5 3.0"""
        self.create_test_file("TestSample_f3GHz_m10dB_1.txt", test_data)
        
        df = loading_setup(self.test_dir, self.sample_name, f=3.0, db=-10)
        
        # K = 64.4, H0 = 7.914
        K, H0 = 64.4, 7.914
        
        # Check each value
        self.assertAlmostEqual(df['Magnetic Field (H)'].iloc[0], K * 0.0 + H0, places=5)
        self.assertAlmostEqual(df['Magnetic Field (H)'].iloc[1], K * 0.1 + H0, places=5)
        self.assertAlmostEqual(df['Magnetic Field (H)'].iloc[2], K * 0.5 + H0, places=5)
    
    def test_file_is_saved(self):
        """Test that processed data is saved back to file"""
        test_data = """0.1 0.5
0.2 0.6"""
        file_path = self.create_test_file("TestSample_f3GHz_m10dB_1.txt", test_data)
        
        # Load and process
        df = loading_setup(self.test_dir, self.sample_name, f=3.0, db=-10)
        
        # Read the file again to verify it was saved
        df_reloaded = pd.read_csv(file_path, sep=r'\s+')
        
        # Check that the new columns exist
        self.assertTrue('Magnetic' in ' '.join(df_reloaded.columns) or 
                       'H' in df_reloaded.columns or
                       len(df_reloaded.columns) >= 3)


if __name__ == '__main__':
    unittest.main()
