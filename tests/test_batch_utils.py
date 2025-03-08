"""Tests for the batch processing utilities."""

import pytest
import numpy as np
import logging
from unittest.mock import patch, MagicMock
import sys

from kompot.batch_utils import (
    batch_process, 
    apply_batched, 
    merge_batch_results,
    is_jax_memory_error
)

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_batch_utils")


class TestBatchUtils:
    """Tests for batch processing utilities."""
    
    def test_is_jax_memory_error(self):
        """Test JAX memory error detection."""
        # Test JAX-specific memory errors
        assert is_jax_memory_error(Exception("RESOURCE_EXHAUSTED"))
        assert is_jax_memory_error(Exception("Resource Exhausted: Out of memory"))
        assert is_jax_memory_error(Exception("out of memory"))
        assert is_jax_memory_error(Exception("memory error"))
        
        # Test non-memory errors
        assert not is_jax_memory_error(Exception("KeyError"))
        assert not is_jax_memory_error(Exception("ValueError"))
        assert not is_jax_memory_error(Exception("Generic error"))
    
    def test_merge_batch_results_dict(self):
        """Test merging dictionary batch results."""
        # Test with dictionaries containing arrays
        results = [
            {'a': np.array([1, 2]), 'b': np.array([3, 4])},
            {'a': np.array([5, 6]), 'b': np.array([7, 8])}
        ]
        merged = merge_batch_results(results)
        
        assert isinstance(merged, dict)
        assert 'a' in merged and 'b' in merged
        np.testing.assert_array_equal(merged['a'], np.array([1, 2, 5, 6]))
        np.testing.assert_array_equal(merged['b'], np.array([3, 4, 7, 8]))
        
        # Test with dictionaries with missing keys
        results = [
            {'a': np.array([1, 2]), 'b': np.array([3, 4])},
            {'a': np.array([5, 6])}  # Missing 'b'
        ]
        merged = merge_batch_results(results)
        
        assert 'a' in merged
        # The merge_batch_results doesn't drop partial keys,
        # it includes them in the merged result for consistency
        assert 'b' in merged
        
        # Test with dictionaries containing lists
        results = [
            {'a': [1, 2], 'b': [3, 4]},
            {'a': [5, 6], 'b': [7, 8]}
        ]
        merged = merge_batch_results(results)
        
        assert isinstance(merged, dict)
        assert merged['a'] == [1, 2, 5, 6]
        assert merged['b'] == [3, 4, 7, 8]
        
        # Test with empty list
        assert merge_batch_results([]) == {}
    
    def test_merge_batch_results_arrays(self):
        """Test merging array batch results."""
        # Test with numpy arrays
        results = [
            np.array([[1, 2], [3, 4]]),
            np.array([[5, 6], [7, 8]])
        ]
        merged = merge_batch_results(results)
        
        assert isinstance(merged, np.ndarray)
        assert merged.shape == (4, 2)
        np.testing.assert_array_equal(merged, np.array([[1, 2], [3, 4], [5, 6], [7, 8]]))
    
    def test_batch_process_decorator(self):
        """Test the batch_process decorator."""
        
        # Create a mock class with a method to decorate
        class TestClass:
            def __init__(self, batch_size=None):
                self.batch_size = batch_size
                self.call_count = 0
            
            @batch_process(default_batch_size=10)
            def process(self, X, multiplier=1):
                self.call_count += 1
                return X * multiplier
        
        # Test with batch_size=None (process all at once)
        instance = TestClass(batch_size=None)
        X = np.array([1, 2, 3, 4, 5])
        result = instance.process(X, multiplier=2)
        
        assert instance.call_count == 1  # Should be called once
        np.testing.assert_array_equal(result, np.array([2, 4, 6, 8, 10]))
        
        # Test with batch_size larger than input size
        instance = TestClass(batch_size=10)
        X = np.array([1, 2, 3, 4, 5])
        result = instance.process(X, multiplier=2)
        
        assert instance.call_count == 1  # Should be called once
        np.testing.assert_array_equal(result, np.array([2, 4, 6, 8, 10]))
        
        # Test with batch_size smaller than input size
        instance = TestClass(batch_size=2)
        X = np.array([1, 2, 3, 4, 5])
        result = instance.process(X, multiplier=2)
        
        assert instance.call_count == 3  # Should be called for each batch
        np.testing.assert_array_equal(result, np.array([2, 4, 6, 8, 10]))
        
        # Create a separate test class with a larger batch trigger for the test
        class LargeBatchTestClass:
            def __init__(self):
                self.call_count = 0
                self.call_sizes = []
                
            @batch_process(default_batch_size=2)  # Use small batch size to force batching
            def process(self, X, multiplier=1):
                self.call_count += 1
                self.call_sizes.append(len(X))
                return X * multiplier
        
        large_instance = LargeBatchTestClass()
        X_large = np.array(list(range(10)))  # Size 10 with batch size 2 will create 5 batches
        
        result = large_instance.process(X_large, multiplier=2)
        
        # Should have been called 5 times with batch size 2
        assert large_instance.call_count == 5
        assert all(size == 2 for size in large_instance.call_sizes)
        np.testing.assert_array_equal(result, X_large * 2)
    
    def test_batch_process_with_memory_error(self):
        """Test batch_process handling of memory errors."""
        
        # Create a mock class with a method that raises memory errors for certain batch sizes
        class TestClass:
            def __init__(self, batch_size=10):
                self.batch_size = batch_size
                self.call_args = []
            
            @batch_process(default_batch_size=10)
            def process(self, X):
                self.call_args.append(len(X))
                
                # Simulate memory error for batches larger than 3
                if len(X) > 3:
                    raise Exception("RESOURCE_EXHAUSTED: Out of memory")
                    
                return X * 2
        
        # Test with batch size reduction due to memory error
        instance = TestClass(batch_size=5)
        X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        result = instance.process(X)
        
        # Should have tried batch size=5 first, then fallen back to smaller sizes
        assert 5 in instance.call_args
        assert any(size <= 3 for size in instance.call_args)
        
        # Result should still be correct
        np.testing.assert_array_equal(result, X * 2)
    
    def test_apply_batched(self):
        """Test the apply_batched function."""
        # Define a processing function
        def process_func(X):
            return X * 2
        
        # Test with batch_size=None (process all at once)
        X = np.array([1, 2, 3, 4, 5])
        result = apply_batched(process_func, X, batch_size=None)
        
        np.testing.assert_array_equal(result, np.array([2, 4, 6, 8, 10]))
        
        # Test with batch_size larger than input size
        result = apply_batched(process_func, X, batch_size=10)
        
        np.testing.assert_array_equal(result, np.array([2, 4, 6, 8, 10]))
        
        # Test with batch_size smaller than input size
        result = apply_batched(process_func, X, batch_size=2)
        
        np.testing.assert_array_equal(result, np.array([2, 4, 6, 8, 10]))
        
        # Test with 2D array
        X_2d = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        result = apply_batched(process_func, X_2d, batch_size=2)
        
        np.testing.assert_array_equal(result, X_2d * 2)
        
        # Note: Axis-based batching is more complex, and will be tested separately
        # in integration tests with the actual differential analysis classes
    
    def test_apply_batched_with_memory_error(self):
        """Test apply_batched handling of memory errors."""
        # Define a processing function that raises memory errors for certain batch sizes
        def process_func(X):
            if len(X) > 3:
                raise Exception("RESOURCE_EXHAUSTED: Out of memory")
            return X * 2
        
        # Test with batch size reduction due to memory error
        X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        result = apply_batched(process_func, X, batch_size=5, show_progress=False)
        
        # Result should still be correct
        np.testing.assert_array_equal(result, X * 2)
    
    def test_apply_batched_with_different_return_types(self):
        """Test apply_batched with different return types."""
        # Test with function that returns dictionaries
        def dict_func(X):
            return {'values': X * 2, 'sum': np.sum(X)}
        
        X = np.array([1, 2, 3, 4, 5, 6])
        result = apply_batched(dict_func, X, batch_size=2, show_progress=False)
        
        assert isinstance(result, dict)
        assert 'values' in result
        assert 'sum' in result
        np.testing.assert_array_equal(result['values'], X * 2)
        
        # Test with function that returns scalars
        def scalar_func(X):
            return np.mean(X)
        
        result = apply_batched(scalar_func, X, batch_size=2, show_progress=False)
        
        # For scalar results, we get a list of results
        assert isinstance(result, list)
        assert len(result) == 3  # 3 batches
    
    def test_partial_success(self):
        """Test behavior when some batches fail completely."""
        def sometimes_fail(X):
            # Always fail for values greater than 5
            if np.any(X > 5):
                raise Exception("RESOURCE_EXHAUSTED: Out of memory even with smallest batches")
            return X * 2
        
        X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        # This should process values 1-5 but fail on 6-10
        with patch('logging.Logger.warning') as mock_warning:
            result = apply_batched(sometimes_fail, X, batch_size=2, show_progress=False)
            
            # Check that a warning was logged
            assert mock_warning.called
        
        # Result should contain only the processed values for 1-5
        expected = np.array([2, 4, 6, 8, 10])
        np.testing.assert_array_equal(result[:len(expected)], expected)


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])