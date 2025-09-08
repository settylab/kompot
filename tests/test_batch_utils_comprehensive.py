"""Comprehensive tests for kompot.batch_utils module to improve coverage."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock
import logging
from tqdm.auto import tqdm


class TestJAXMemoryErrorDetection:
    """Test JAX memory error detection functionality."""
    
    def test_is_jax_memory_error_resource_exhausted(self):
        """Test detection of RESOURCE_EXHAUSTED errors."""
        try:
            from kompot.batch_utils import is_jax_memory_error
        except ImportError as e:
            pytest.skip(f"Could not import is_jax_memory_error: {e}")
        
        # Test various JAX memory error messages
        error1 = Exception("RESOURCE_EXHAUSTED: Out of memory")
        error2 = Exception("resource exhausted during computation")
        error3 = Exception("Resource exhausted: insufficient GPU memory")
        
        assert is_jax_memory_error(error1) == True
        assert is_jax_memory_error(error2) == True
        assert is_jax_memory_error(error3) == True
        
    def test_is_jax_memory_error_out_of_memory(self):
        """Test detection of out of memory errors."""
        try:
            from kompot.batch_utils import is_jax_memory_error
        except ImportError as e:
            pytest.skip(f"Could not import is_jax_memory_error: {e}")
        
        error1 = Exception("Out of memory during allocation")
        error2 = Exception("out of memory: GPU device")
        error3 = Exception("Memory allocation failed")
        error4 = Exception("Insufficient memory for computation")
        
        assert is_jax_memory_error(error1) == True
        assert is_jax_memory_error(error2) == True
        assert is_jax_memory_error(error3) == True
        assert is_jax_memory_error(error4) == True
        
    def test_is_jax_memory_error_non_memory_errors(self):
        """Test that non-memory errors are not detected as memory errors."""
        try:
            from kompot.batch_utils import is_jax_memory_error
        except ImportError as e:
            pytest.skip(f"Could not import is_jax_memory_error: {e}")
        
        error1 = Exception("Invalid argument")
        error2 = ValueError("Shape mismatch")
        error3 = TypeError("Expected array")
        error4 = Exception("Computation failed")
        
        assert is_jax_memory_error(error1) == False
        assert is_jax_memory_error(error2) == False
        assert is_jax_memory_error(error3) == False
        assert is_jax_memory_error(error4) == False


class TestMergeBatchResults:
    """Test batch result merging functionality."""
    
    def test_merge_batch_results_empty_list(self):
        """Test merging empty list of results."""
        try:
            from kompot.batch_utils import merge_batch_results
        except ImportError as e:
            pytest.skip(f"Could not import merge_batch_results: {e}")
        
        result = merge_batch_results([])
        assert result == {}
        
    def test_merge_batch_results_dict_numpy_arrays(self):
        """Test merging dictionaries with numpy arrays."""
        try:
            from kompot.batch_utils import merge_batch_results
        except ImportError as e:
            pytest.skip(f"Could not import merge_batch_results: {e}")
        
        batch1 = {
            'values': np.array([1, 2, 3]),
            'scores': np.array([0.1, 0.2])
        }
        batch2 = {
            'values': np.array([4, 5]),
            'scores': np.array([0.3, 0.4, 0.5])
        }
        batch3 = {
            'values': np.array([6, 7, 8, 9]),
            'scores': np.array([0.6])
        }
        
        merged = merge_batch_results([batch1, batch2, batch3])
        
        assert 'values' in merged
        assert 'scores' in merged
        np.testing.assert_array_equal(merged['values'], np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]))
        np.testing.assert_array_equal(merged['scores'], np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
        
    def test_merge_batch_results_dict_scalar_arrays(self):
        """Test merging dictionaries with scalar arrays."""
        try:
            from kompot.batch_utils import merge_batch_results
        except ImportError as e:
            pytest.skip(f"Could not import merge_batch_results: {e}")
        
        batch1 = {'loss': np.array(0.5)}
        batch2 = {'loss': np.array(0.3)}
        batch3 = {'loss': np.array(0.8)}
        
        merged = merge_batch_results([batch1, batch2, batch3])
        
        assert 'loss' in merged
        np.testing.assert_array_equal(merged['loss'], np.array([0.5, 0.3, 0.8]))
        
    def test_merge_batch_results_dict_true_scalars(self):
        """Test merging dictionaries with true scalar arrays (shape ())."""
        try:
            from kompot.batch_utils import merge_batch_results
        except ImportError as e:
            pytest.skip(f"Could not import merge_batch_results: {e}")
        
        # Create true scalar arrays
        batch1 = {'metric': np.array(1.5)}
        batch2 = {'metric': np.array(2.3)}
        batch1['metric'] = batch1['metric'].reshape(())  # Make truly scalar
        batch2['metric'] = batch2['metric'].reshape(())
        
        merged = merge_batch_results([batch1, batch2])
        
        assert 'metric' in merged
        assert merged['metric'].shape == (2,)
        np.testing.assert_array_almost_equal(merged['metric'], np.array([1.5, 2.3]))
        
    def test_merge_batch_results_dict_lists(self):
        """Test merging dictionaries with lists."""
        try:
            from kompot.batch_utils import merge_batch_results
        except ImportError as e:
            pytest.skip(f"Could not import merge_batch_results: {e}")
        
        batch1 = {'items': [1, 2, 3], 'names': ['a', 'b']}
        batch2 = {'items': [4, 5], 'names': ['c', 'd', 'e']}
        batch3 = {'items': [6], 'names': ['f']}
        
        merged = merge_batch_results([batch1, batch2, batch3])
        
        assert 'items' in merged
        assert 'names' in merged
        assert merged['items'] == [1, 2, 3, 4, 5, 6]
        assert merged['names'] == ['a', 'b', 'c', 'd', 'e', 'f']
        
    def test_merge_batch_results_dict_mixed_types(self):
        """Test merging dictionaries with mixed value types."""
        try:
            from kompot.batch_utils import merge_batch_results
        except ImportError as e:
            pytest.skip(f"Could not import merge_batch_results: {e}")
        
        batch1 = {
            'arrays': np.array([1, 2]),
            'lists': [1, 2],
            'other': 'value1'
        }
        batch2 = {
            'arrays': np.array([3, 4, 5]),
            'lists': [3, 4],
            'other': 'value2'
        }
        
        merged = merge_batch_results([batch1, batch2])
        
        assert 'arrays' in merged
        assert 'lists' in merged
        assert 'other' in merged
        
        np.testing.assert_array_equal(merged['arrays'], np.array([1, 2, 3, 4, 5]))
        assert merged['lists'] == [1, 2, 3, 4]
        assert merged['other'] == ['value1', 'value2']  # Falls back to list
        
    def test_merge_batch_results_dict_missing_keys(self):
        """Test merging dictionaries with missing keys in some batches."""
        try:
            from kompot.batch_utils import merge_batch_results
        except ImportError as e:
            pytest.skip(f"Could not import merge_batch_results: {e}")
        
        batch1 = {'a': np.array([1, 2]), 'b': np.array([10])}
        batch2 = {'a': np.array([3]), 'c': np.array([20, 21])}
        batch3 = {'b': np.array([11, 12]), 'c': np.array([22])}
        
        merged = merge_batch_results([batch1, batch2, batch3])
        
        assert 'a' in merged
        assert 'b' in merged
        assert 'c' in merged
        
        np.testing.assert_array_equal(merged['a'], np.array([1, 2, 3]))
        np.testing.assert_array_equal(merged['b'], np.array([10, 11, 12]))
        np.testing.assert_array_equal(merged['c'], np.array([20, 21, 22]))
        
    def test_merge_batch_results_numpy_arrays_direct(self):
        """Test merging numpy arrays directly (not in dictionaries)."""
        try:
            from kompot.batch_utils import merge_batch_results
        except ImportError as e:
            pytest.skip(f"Could not import merge_batch_results: {e}")
        
        batch1 = np.array([1, 2, 3])
        batch2 = np.array([4, 5])
        batch3 = np.array([6, 7, 8, 9])
        
        merged = merge_batch_results([batch1, batch2, batch3])
        
        np.testing.assert_array_equal(merged, np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]))
        
    def test_merge_batch_results_concatenation_axis(self):
        """Test merging with different concatenation axis."""
        try:
            from kompot.batch_utils import merge_batch_results
        except ImportError as e:
            pytest.skip(f"Could not import merge_batch_results: {e}")
        
        batch1 = {'data': np.array([[1, 2], [3, 4]])}
        batch2 = {'data': np.array([[5, 6]])}
        
        # Default axis=0
        merged_0 = merge_batch_results([batch1, batch2])
        expected_0 = np.array([[1, 2], [3, 4], [5, 6]])
        np.testing.assert_array_equal(merged_0['data'], expected_0)
        
        # Axis=1 (different concatenation)
        batch1_axis1 = {'data': np.array([[1], [2]])}
        batch2_axis1 = {'data': np.array([[3, 4], [5, 6]])}
        
        merged_1 = merge_batch_results([batch1_axis1, batch2_axis1], concat_axis=1)
        expected_1 = np.array([[1, 3, 4], [2, 5, 6]])
        np.testing.assert_array_equal(merged_1['data'], expected_1)
        
    def test_merge_batch_results_concatenation_error_handling(self):
        """Test error handling during concatenation."""
        try:
            from kompot.batch_utils import merge_batch_results
        except ImportError as e:
            pytest.skip(f"Could not import merge_batch_results: {e}")
        
        # Create arrays with incompatible shapes
        batch1 = {'data': np.array([[1, 2, 3]])}  # Shape (1, 3)
        batch2 = {'data': np.array([[4, 5]])}      # Shape (1, 2) - incompatible
        
        with patch('kompot.batch_utils.logger') as mock_logger:
            merged = merge_batch_results([batch1, batch2])
            
            # Should log warning and fall back to list
            mock_logger.warning.assert_called()
            warning_call = mock_logger.warning.call_args[0][0]
            assert "Failed to concatenate arrays" in warning_call
            assert 'data' in merged
            assert isinstance(merged['data'], list)
            assert len(merged['data']) == 2


class TestApplyBatched:
    """Test apply_batched functionality."""
    
    def test_apply_batched_basic(self):
        """Test basic apply_batched functionality."""
        try:
            from kompot.batch_utils import apply_batched
        except ImportError as e:
            pytest.skip(f"Could not import apply_batched: {e}")
        
        def simple_func(X):
            return X * 2
            
        X = np.array([1, 2, 3, 4, 5, 6])
        result = apply_batched(simple_func, X, batch_size=3)
        
        expected = np.array([2, 4, 6, 8, 10, 12])
        np.testing.assert_array_equal(result, expected)
        
    def test_apply_batched_no_batching(self):
        """Test apply_batched without batching (batch_size=None)."""
        try:
            from kompot.batch_utils import apply_batched
        except ImportError as e:
            pytest.skip(f"Could not import apply_batched: {e}")
        
        def square_func(X):
            return X ** 2
            
        X = np.array([1, 2, 3, 4])
        result = apply_batched(square_func, X, batch_size=None)
        
        expected = np.array([1, 4, 9, 16])
        np.testing.assert_array_equal(result, expected)
        
    def test_apply_batched_with_progress_bar(self):
        """Test apply_batched with progress bar."""
        try:
            from kompot.batch_utils import apply_batched
        except ImportError as e:
            pytest.skip(f"Could not import apply_batched: {e}")
        
        def identity_func(X):
            return X
            
        X = np.array(range(10))
        
        with patch('kompot.batch_utils.tqdm') as mock_tqdm:
            mock_tqdm.return_value.__enter__.return_value = mock_tqdm
            # Mock the iterator to return the actual batch indices
            mock_tqdm.return_value.__iter__ = lambda x: iter([0, 3, 6, 9])  # Proper batch starts for batch_size=3
            
            result = apply_batched(identity_func, X, batch_size=3, desc="Testing")
            
            # The function may call tqdm multiple times due to retry logic
            # Check that the first call has the expected description
            assert mock_tqdm.called
            first_call_args = mock_tqdm.call_args_list[0]
            assert first_call_args[1]['desc'] == "Testing"
            
    def test_apply_batched_dict_results(self):
        """Test apply_batched with function returning dictionaries."""
        try:
            from kompot.batch_utils import apply_batched
        except ImportError as e:
            pytest.skip(f"Could not import apply_batched: {e}")
        
        def dict_func(X):
            return {
                'squared': X ** 2,
                'doubled': X * 2,
                'mean': np.array([np.mean(X)])  # Single value as array
            }
            
        X = np.array([1, 2, 3, 4, 5, 6])
        result = apply_batched(dict_func, X, batch_size=2)
        
        assert 'squared' in result
        assert 'doubled' in result
        assert 'mean' in result
        
        np.testing.assert_array_equal(result['squared'], np.array([1, 4, 9, 16, 25, 36]))
        np.testing.assert_array_equal(result['doubled'], np.array([2, 4, 6, 8, 10, 12]))
        # Mean should be concatenated from all batches
        assert len(result['mean']) == 3  # 3 batches
        
    def test_apply_batched_memory_error_fallback(self):
        """Test apply_batched fallback when memory error occurs."""
        try:
            from kompot.batch_utils import apply_batched
        except ImportError as e:
            pytest.skip(f"Could not import apply_batched: {e}")
        
        call_count = 0
        def failing_func(X):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call fails with memory error
                raise Exception("RESOURCE_EXHAUSTED: Out of memory")
            else:
                # Subsequent calls succeed
                return X * 3
        
        X = np.array([1, 2, 3, 4, 5, 6])
        
        with patch('kompot.batch_utils.logger') as mock_logger:
            result = apply_batched(failing_func, X, batch_size=6)  # Start with large batch
            
            # Should log the memory error and retry
            mock_logger.warning.assert_called()
            warning_msg = mock_logger.warning.call_args[0][0]
            assert "Memory error detected" in warning_msg
            
            # Should eventually succeed with smaller batches
            expected = np.array([3, 6, 9, 12, 15, 18])
            np.testing.assert_array_equal(result, expected)
            
    def test_apply_batched_memory_error_all_fail(self):
        """Test apply_batched when all batch sizes fail with memory error."""
        try:
            from kompot.batch_utils import apply_batched
        except ImportError as e:
            pytest.skip(f"Could not import apply_batched: {e}")
        
        def always_failing_func(X):
            raise Exception("RESOURCE_EXHAUSTED: Insufficient memory")
        
        X = np.array([1, 2, 3, 4])
        
        with patch('kompot.batch_utils.logger') as mock_logger:
            with pytest.raises(Exception, match="RESOURCE_EXHAUSTED"):
                apply_batched(always_failing_func, X, batch_size=4)
            
            # Should log multiple attempts
            assert mock_logger.warning.call_count > 0
            
    def test_apply_batched_non_memory_error(self):
        """Test apply_batched with non-memory errors (should not retry)."""
        try:
            from kompot.batch_utils import apply_batched
        except ImportError as e:
            pytest.skip(f"Could not import apply_batched: {e}")
        
        def error_func(X):
            raise ValueError("Invalid input")
        
        X = np.array([1, 2, 3])
        
        # Should propagate the error without retrying
        with pytest.raises(ValueError, match="Invalid input"):
            apply_batched(error_func, X, batch_size=2)


class TestBatchedDecorator:
    """Test batched decorator functionality."""
    
    def test_batched_decorator_basic(self):
        """Test basic batched decorator."""
        try:
            from kompot.batch_utils import batched
        except ImportError as e:
            pytest.skip(f"Could not import batched decorator: {e}")
        
        @batched(batch_size=2)
        def multiply_by_3(X):
            return X * 3
        
        X = np.array([1, 2, 3, 4, 5])
        result = multiply_by_3(X)
        
        expected = np.array([3, 6, 9, 12, 15])
        np.testing.assert_array_equal(result, expected)
        
    def test_batched_decorator_with_kwargs(self):
        """Test batched decorator with keyword arguments."""
        try:
            from kompot.batch_utils import batched
        except ImportError as e:
            pytest.skip(f"Could not import batched decorator: {e}")
        
        @batched(batch_size=3, desc="Processing")
        def add_value(X, value=1):
            return X + value
        
        X = np.array([1, 2, 3, 4, 5, 6, 7])
        result = add_value(X, value=10)
        
        expected = np.array([11, 12, 13, 14, 15, 16, 17])
        np.testing.assert_array_equal(result, expected)
        
    def test_batched_decorator_override_batch_size(self):
        """Test batched decorator with runtime batch_size override."""
        try:
            from kompot.batch_utils import batched
        except ImportError as e:
            pytest.skip(f"Could not import batched decorator: {e}")
        
        @batched(batch_size=10)  # Default large batch size
        def divide_by_2(X):
            return X / 2
        
        X = np.array([2, 4, 6, 8])
        
        # Override batch_size at runtime
        result = divide_by_2(X, batch_size=1)  # Force small batches
        
        expected = np.array([1, 2, 3, 4])
        np.testing.assert_array_equal(result, expected)
        
    def test_batched_decorator_disable_progress(self):
        """Test batched decorator with progress disabled."""
        try:
            from kompot.batch_utils import batched
        except ImportError as e:
            pytest.skip(f"Could not import batched decorator: {e}")
        
        @batched(batch_size=2, desc="Should not show")
        def negate(X):
            return -X
        
        X = np.array([1, 2, 3, 4])
        
        with patch('kompot.batch_utils.tqdm') as mock_tqdm:
            result = negate(X, desc=None)  # Disable progress
            
            # tqdm should not be called when desc=None
            mock_tqdm.assert_not_called()
        
        expected = np.array([-1, -2, -3, -4])
        np.testing.assert_array_equal(result, expected)


class TestBatchUtilsEdgeCases:
    """Test edge cases and error conditions in batch_utils."""
    
    def test_merge_batch_results_jax_arrays(self):
        """Test merging JAX arrays (if available)."""
        try:
            from kompot.batch_utils import merge_batch_results
            import jax.numpy as jnp
        except ImportError:
            pytest.skip("JAX not available")
        
        batch1 = {'data': jnp.array([1, 2, 3])}
        batch2 = {'data': jnp.array([4, 5])}
        
        merged = merge_batch_results([batch1, batch2])
        
        assert 'data' in merged
        # Result should be JAX array
        assert isinstance(merged['data'], jnp.ndarray)
        np.testing.assert_array_equal(np.array(merged['data']), np.array([1, 2, 3, 4, 5]))
        
    def test_apply_batched_empty_input(self):
        """Test apply_batched with empty input."""
        try:
            from kompot.batch_utils import apply_batched
        except ImportError as e:
            pytest.skip(f"Could not import apply_batched: {e}")
        
        def dummy_func(X):
            return X
        
        X = np.array([])
        result = apply_batched(dummy_func, X, batch_size=5)
        
        np.testing.assert_array_equal(result, np.array([]))
        
    def test_apply_batched_single_element(self):
        """Test apply_batched with single element."""
        try:
            from kompot.batch_utils import apply_batched
        except ImportError as e:
            pytest.skip(f"Could not import apply_batched: {e}")
        
        def square(X):
            return X ** 2
        
        X = np.array([5])
        result = apply_batched(square, X, batch_size=10)
        
        np.testing.assert_array_equal(result, np.array([25]))
        
    def test_logger_usage_in_batch_utils(self):
        """Test logger usage in batch_utils functions."""
        try:
            from kompot.batch_utils import apply_batched, logger
        except ImportError as e:
            pytest.skip(f"Could not import batch_utils functions: {e}")
        
        def func_with_memory_error(X):
            raise Exception("RESOURCE_EXHAUSTED")
        
        X = np.array([1, 2])
        
        with patch.object(logger, 'warning') as mock_warning:
            with pytest.raises(Exception):
                apply_batched(func_with_memory_error, X, batch_size=2)
            
            # Should log memory error warnings
            mock_warning.assert_called()
            
    def test_batch_size_reduction_logic(self):
        """Test the batch size reduction logic in memory error handling."""
        try:
            from kompot.batch_utils import apply_batched
        except ImportError as e:
            pytest.skip(f"Could not import apply_batched: {e}")
        
        call_history = []
        
        def memory_sensitive_func(X):
            call_history.append(len(X))
            if len(X) > 2:
                raise Exception("RESOURCE_EXHAUSTED: Too large batch")
            return X * 2
        
        X = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        
        with patch('kompot.batch_utils.logger'):
            result = apply_batched(memory_sensitive_func, X, batch_size=8)
        
        # Should eventually succeed with smaller batches
        expected = np.array([2, 4, 6, 8, 10, 12, 14, 16])
        np.testing.assert_array_equal(result, expected)
        
        # Should have tried larger batch sizes first, then reduced
        assert 8 in call_history  # Initial large batch that failed
        assert all(size <= 2 for size in call_history[-4:])  # Final successful batches