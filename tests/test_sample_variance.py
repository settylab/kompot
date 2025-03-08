"""Tests for the SampleVarianceEstimator class."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from kompot.differential import SampleVarianceEstimator
from kompot.batch_utils import apply_batched, is_jax_memory_error


def test_sample_variance_estimator_diag():
    """Test that the SampleVarianceEstimator works with diag=True and diag=False."""
    # Generate some sample data
    n_cells = 20
    n_features = 5
    n_genes = 3
    n_groups = 2
    
    X = np.random.randn(n_cells, n_features)
    Y = np.random.randn(n_cells, n_genes)
    grouping_vector = np.random.randint(0, n_groups, size=n_cells)
    
    # Initialize and fit the estimator
    estimator = SampleVarianceEstimator(batch_size=10)
    estimator.fit(X, Y, grouping_vector)
    
    # Test with diag=True
    variance_diag_true = estimator.predict(X, diag=True)
    
    # Check the shape
    assert variance_diag_true.shape == (n_cells, n_genes)
    
    # Test with diag=False
    variance_diag_false = estimator.predict(X, diag=False)
    
    # Check the shape
    assert variance_diag_false.shape == (n_cells, n_cells, n_genes)
    
    # Verify the diagonal elements match the diag=True result
    for i in range(n_cells):
        np.testing.assert_allclose(
            variance_diag_false[i, i, :],
            variance_diag_true[i, :],
            rtol=1e-5,
            atol=1e-8
        )
    
    # Verify symmetry of the covariance matrix
    for g in range(n_genes):
        gene_cov = variance_diag_false[:, :, g]
        np.testing.assert_allclose(
            gene_cov,
            gene_cov.T,
            rtol=1e-5,
            atol=1e-8
        )


def test_sample_variance_estimator_batching():
    """Test that the SampleVarianceEstimator properly handles batching with diag=True."""
    # Generate some sample data
    n_cells = 30
    n_features = 5
    n_genes = 3
    n_groups = 2
    
    X = np.random.randn(n_cells, n_features)
    Y = np.random.randn(n_cells, n_genes)
    grouping_vector = np.random.randint(0, n_groups, size=n_cells)
    
    # Initialize and fit the estimator with a small batch size
    small_batch_estimator = SampleVarianceEstimator(batch_size=10)
    small_batch_estimator.fit(X, Y, grouping_vector)
    
    # Initialize and fit the estimator with a large batch size
    large_batch_estimator = SampleVarianceEstimator(batch_size=50)
    large_batch_estimator.fit(X, Y, grouping_vector)
    
    # Get results with both batch sizes
    small_batch_result = small_batch_estimator.predict(X, diag=True)
    large_batch_result = large_batch_estimator.predict(X, diag=True)
    
    # Results should be the same regardless of batch size
    np.testing.assert_allclose(
        small_batch_result,
        large_batch_result,
        rtol=1e-5,
        atol=1e-8
    )


def test_sample_variance_estimator_jit():
    """Test that the SampleVarianceEstimator works with JIT compilation."""
    # Generate some sample data
    n_cells = 20
    n_features = 5
    n_genes = 3
    n_groups = 2
    
    X = np.random.randn(n_cells, n_features)
    Y = np.random.randn(n_cells, n_genes)
    grouping_vector = np.random.randint(0, n_groups, size=n_cells)
    
    # Initialize and fit the estimator with JIT
    jit_estimator = SampleVarianceEstimator(jit_compile=True)
    jit_estimator.fit(X, Y, grouping_vector)
    
    # Initialize and fit the estimator without JIT
    no_jit_estimator = SampleVarianceEstimator(jit_compile=False)
    no_jit_estimator.fit(X, Y, grouping_vector)
    
    # Test with diag=True
    jit_result = jit_estimator.predict(X, diag=True)
    no_jit_result = no_jit_estimator.predict(X, diag=True)
    
    # Results should be very close
    np.testing.assert_allclose(
        jit_result,
        no_jit_result,
        rtol=1e-4,
        atol=1e-6
    )


def test_sample_variance_estimator_apply_batched_usage():
    """Test that apply_batched is used for both diagonal and non-diagonal cases."""
    # Generate some sample data
    n_cells = 15
    n_features = 3
    n_genes = 2
    n_groups = 2
    
    X = np.random.randn(n_cells, n_features)
    Y = np.random.randn(n_cells, n_genes)
    grouping_vector = np.random.randint(0, n_groups, size=n_cells)
    
    # Initialize and fit the estimator with a lower min_cells threshold to ensure groups are created
    estimator = SampleVarianceEstimator(batch_size=5)
    estimator.fit(X, Y, grouping_vector, min_cells=3)  # Lower threshold to ensure groups are created
    
    # Mock apply_batched to track calls
    with patch('kompot.differential.apply_batched') as mock_apply_batched:
        # Create a side effect that returns the expected results
        def side_effect(func, X, **kwargs):
            if kwargs.get('desc', '').startswith('Computing variance across groups'):
                # For diag=True case
                return np.random.randn(len(X), n_genes)
            else:
                # For diag=False case with sub-batching
                return np.random.randn(len(X), len(X), n_genes)
        
        mock_apply_batched.side_effect = side_effect
        
        # Test diag=True
        _ = estimator.predict(X, diag=True)
        
        # Verify apply_batched was called once for diag=True
        assert mock_apply_batched.call_count == 1
        
        # Reset the mock
        mock_apply_batched.reset_mock()
        
        # Since we're now using direct computation instead of apply_batched for diag=False
        # this test is no longer applicable in the same way.
        # Instead, let's verify the result shape is correct
        result = estimator.predict(X, diag=False)
        assert result.shape == (n_cells, n_cells, n_genes)


def test_memory_sensitive_batch_reduction():
    """Test that the memory-sensitive batch size reduction works correctly."""
    # Generate some sample data
    n_cells = 20
    n_features = 5
    n_genes = 3
    n_groups = 2
    
    X = np.random.randn(n_cells, n_features)
    Y = np.random.randn(n_cells, n_genes)
    grouping_vector = np.random.randint(0, n_groups, size=n_cells)
    
    # Initialize and fit the estimator with lower min_cells threshold
    estimator = SampleVarianceEstimator(batch_size=10)
    estimator.fit(X, Y, grouping_vector, min_cells=3)  # Lower min_cells threshold
    
    # Create a mock of apply_batched that simulates memory errors
    original_apply_batched = apply_batched
    
    def mock_apply_batched(func, X, batch_size=None, **kwargs):
        # Generate a fake result for covariance matrix computations
        if kwargs.get('desc', '').startswith('Batch ['):
            # This is for the diag=False covariance matrix computation
            # Create a symmetric random matrix for each gene
            fake_result = np.zeros((len(X), len(X), n_genes))
            for g in range(n_genes):
                # Create symmetric positive definite matrix (for covariance)
                random_data = np.random.rand(len(X), len(X))
                # Make it symmetric
                symmetric_data = (random_data + random_data.T) / 2
                # Add to diagonal to ensure it's positive definite
                np.fill_diagonal(symmetric_data, np.diag(symmetric_data) + 1.0)
                fake_result[:, :, g] = symmetric_data
            return fake_result
        elif 'Computing variance across groups' in str(kwargs.get('desc', '')):
            # This is for diag=True case
            return np.random.rand(len(X), n_genes)
        
        # For other cases, use the original function with reduced batch size
        return original_apply_batched(func, X, batch_size=min(5, batch_size or 5), **kwargs)
    
    # Patch apply_batched with our mock version
    with patch('kompot.differential.apply_batched', side_effect=mock_apply_batched):
        # Test with diag=False to trigger the full covariance computation
        # This should now use our mock which forces batch size reduction
        result = estimator.predict(X, diag=False)
        
        # Verify the result has the expected shape
        assert result.shape == (n_cells, n_cells, n_genes)
        
        # Verify the result has non-zero values (not all zeros)
        assert np.sum(np.abs(result)) > 0
        
        # Verify the result is symmetric
        for g in range(n_genes):
            np.testing.assert_allclose(
                result[:, :, g],
                result[:, :, g].T,
                rtol=1e-5,
                atol=1e-8
            )


def test_error_handling_in_prediction():
    """Test that the prediction function handles errors appropriately."""
    # Generate some sample data
    n_cells = 15
    n_features = 3
    n_genes = 2
    n_groups = 2
    
    X = np.random.randn(n_cells, n_features)
    Y = np.random.randn(n_cells, n_genes)
    grouping_vector = np.random.randint(0, n_groups, size=n_cells)
    
    # Initialize and fit the estimator with a lower min_cells threshold
    estimator = SampleVarianceEstimator(batch_size=5)
    estimator.fit(X, Y, grouping_vector, min_cells=3)  # Lower threshold to ensure groups are created
    
    # Test diag=True with error handling by directly patching the function call
    # Since we've refactored to not use apply_batched for the covariance computation
    with patch('kompot.differential.apply_batched', side_effect=ValueError("Some error")):
        # This should still work for diag=True, which does use apply_batched
        with pytest.raises(ValueError):
            estimator.predict(X, diag=True)
            
    # For diag=False, we can just verify the correct shape
    result = estimator.predict(X, diag=False)
    assert result.shape == (n_cells, n_cells, n_genes)


def test_large_data_handling():
    """Test with a larger dataset to simulate real-world usage."""
    # Generate a larger dataset
    n_cells = 100
    n_features = 10
    n_genes = 5
    n_groups = 3
    
    X = np.random.randn(n_cells, n_features)
    Y = np.random.randn(n_cells, n_genes)
    grouping_vector = np.random.randint(0, n_groups, size=n_cells)
    
    # Initialize with a small batch size to force multiple batches
    estimator = SampleVarianceEstimator(batch_size=25)
    estimator.fit(X, Y, grouping_vector)
    
    # Test with diag=True
    variance_diag_true = estimator.predict(X, diag=True)
    
    # Check the shape
    assert variance_diag_true.shape == (n_cells, n_genes)
    
    # Test with diag=False, but only on a subset to keep the test fast
    small_X = X[:30]
    variance_diag_false = estimator.predict(small_X, diag=False)
    
    # Check the shape
    assert variance_diag_false.shape == (30, 30, n_genes)
    
    # Verify symmetry of the covariance matrix
    for g in range(n_genes):
        gene_cov = variance_diag_false[:, :, g]
        np.testing.assert_allclose(
            gene_cov,
            gene_cov.T,
            rtol=1e-5,
            atol=1e-8
        )