"""
Tests for data loader
"""
import unittest
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from loader import parse_filename

class TestLoader(unittest.TestCase):
    
    def test_parse_filename(self):
        """Test filename parsing"""
        metadata = parse_filename("AZ5_f5GHz_m10dB_1_t2.txt")
        
        self.assertEqual(metadata['sample'], 'AZ5')
        self.assertEqual(metadata['frequency'], 5.0)
        self.assertEqual(metadata['decibel'], -10)
        self.assertEqual(metadata['replicate'], 1)

if __name__ == '__main__':
    unittest.main()