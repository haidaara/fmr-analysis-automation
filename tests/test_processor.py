"""
Tests for data processor
"""
import unittest
import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from processor import process_curve, find_resonance

class TestProcessor(unittest.TestCase):
    
    def test_process_curve(self):
        """Test curve processing"""
        # Create test data
        test_data = {
            'H_field': np.array([1.0, 1.0, 2.0, 2.0]),
            'signal': np.array([0.1, 0.2, 0.3, 0.4]),
            'metadata': {}
        }
        
        processed = process_curve(test_data)
        
        # Check that processed data was added
        self.assertIn('processed', processed)
        self.assertEqual(len(processed['processed']['H_unique']), 2)
        
if __name__ == '__main__':
    unittest.main()