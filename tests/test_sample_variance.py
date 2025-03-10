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
    estimator = SampleVarianceEstimator()
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
    """Test that the SampleVarianceEstimator produces consistent results without batching."""
    # Generate some sample data
    n_cells = 30
    n_features = 5
    n_genes = 3
    n_groups = 2
    
    X = np.random.randn(n_cells, n_features)
    Y = np.random.randn(n_cells, n_genes)
    grouping_vector = np.random.randint(0, n_groups, size=n_cells)
    
    # Initialize and fit multiple estimators - all should produce the same result
    # since batching has been removed
    estimator1 = SampleVarianceEstimator()
    estimator1.fit(X, Y, grouping_vector)
    
    estimator2 = SampleVarianceEstimator()
    estimator2.fit(X, Y, grouping_vector)
    
    # Get results with both estimators
    result1 = estimator1.predict(X, diag=True)
    result2 = estimator2.predict(X, diag=True)
    
    # Results should be the same
    np.testing.assert_allclose(
        result1,
        result2,
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
    """Test that batch processing is removed and replaced with direct computation."""
    # Generate some sample data
    n_cells = 15
    n_features = 3
    n_genes = 2
    n_groups = 2
    
    X = np.random.randn(n_cells, n_features)
    Y = np.random.randn(n_cells, n_genes)
    grouping_vector = np.random.randint(0, n_groups, size=n_cells)
    
    # Initialize and fit the estimator with a lower min_cells threshold to ensure groups are created
    estimator = SampleVarianceEstimator()  # No batch_size parameter needed
    estimator.fit(X, Y, grouping_vector, min_cells=3)  # Lower threshold to ensure groups are created
    
    # Verify we don't use apply_batched at all by mocking it and making sure it's not called
    with patch('kompot.differential.apply_batched') as mock_apply_batched:
        # Both calls should now use direct computation
        
        # diag=True now calls predictors directly
        result_diag = estimator.predict(X, diag=True)
        
        # Verify apply_batched was NOT called for diag=True
        assert mock_apply_batched.call_count == 0
        
        # diag=False also uses direct computation
        result_full = estimator.predict(X, diag=False)
        
        # Verify apply_batched was still not called
        assert mock_apply_batched.call_count == 0
        
    # Verify the results have the expected shapes
    assert result_diag.shape == (n_cells, n_genes)
    assert result_full.shape == (n_cells, n_cells, n_genes)


def test_memory_sensitive_batch_reduction():
    """Test the memory warning when processing many genes."""
    # Generate some sample data
    n_cells = 20
    n_features = 5
    n_genes = 600  # More than 500 to trigger warning
    n_groups = 2
    
    X = np.random.randn(n_cells, n_features)
    Y = np.random.randn(n_cells, n_genes)
    grouping_vector = np.random.randint(0, n_groups, size=n_cells)
    
    # Initialize and fit the estimator 
    estimator = SampleVarianceEstimator()
    estimator.fit(X, Y, grouping_vector, min_cells=3)  # Lower min_cells threshold
    
    # Set the flag that indicates this estimator is called from DifferentialExpression
    estimator._called_from_differential = True
    
    # Capture the logs to verify the warning
    with patch('kompot.differential.logger.warning') as mock_warning:
        # Test with a large number of genes
        result = estimator.predict(X, diag=False)
        
        # Verify the warning was logged
        mock_warning.assert_called_once()
        warning_msg = mock_warning.call_args[0][0]
        assert "genes may require significant memory" in warning_msg
        assert "max 500 genes" in warning_msg
        
        # Verify the result has the expected shape
        assert result.shape == (n_cells, n_cells, n_genes)
        
    # The result should still be valid
    # Verify it's symmetric (covariance matrices should be symmetric)
    for g in range(min(3, n_genes)):  # Just check a few genes to save time
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
    estimator = SampleVarianceEstimator()
    estimator.fit(X, Y, grouping_vector, min_cells=3)  # Lower threshold to ensure groups are created
    
    # Test error handling by patching the predictor functions
    # For diag=True, patch one of the group predictors to raise an error
    mock_predictor = MagicMock(side_effect=ValueError("Simulated predictor error"))
    
    # Store the original predictors to restore later
    original_predictors = estimator.group_predictors.copy()
    
    # Replace first predictor with our mocked one
    first_key = next(iter(estimator.group_predictors.keys()))
    estimator.group_predictors[first_key] = mock_predictor
    
    # This should raise the ValueError from our mock
    with pytest.raises(ValueError):
        estimator.predict(X, diag=True)
    
    # Restore original predictors
    estimator.group_predictors = original_predictors
    
    # For diag=False, we can just verify the correct shape
    result = estimator.predict(X, diag=False)
    assert result.shape == (n_cells, n_cells, n_genes)


def test_no_predictors_error():
    """Test that an error is raised when there are no group predictors."""
    # Generate some sample data
    n_cells = 15
    n_features = 3
    n_genes = 2
    n_groups = 2
    
    X = np.random.randn(n_cells, n_features)
    Y = np.random.randn(n_cells, n_genes)
    grouping_vector = np.random.randint(0, n_groups, size=n_cells)
    
    # Initialize and fit the estimator with an impossibly high min_cells threshold
    # This will create a fitted estimator but with no group predictors
    estimator = SampleVarianceEstimator()
    estimator.fit(X, Y, grouping_vector, min_cells=n_cells+1)  # Set threshold higher than available cells
    
    # Now test that predict raises RuntimeError
    with pytest.raises(RuntimeError, match="No group predictors available"):
        estimator.predict(X, diag=True)
        
    # Same for non-diagonal case
    with pytest.raises(RuntimeError, match="No group predictors available"):
        estimator.predict(X, diag=False)


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
    
    # Initialize estimator
    estimator = SampleVarianceEstimator()
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


def test_sample_variance_estimator_density_mode():
    """Test that the SampleVarianceEstimator works in density mode."""
    # Generate some sample data
    n_cells = 20
    n_features = 5
    n_groups = 2
    
    X = np.random.randn(n_cells, n_features)
    grouping_vector = np.random.randint(0, n_groups, size=n_cells)
    
    # Initialize and fit the estimator in density mode
    estimator = SampleVarianceEstimator(estimator_type='density')
    estimator.fit(X=X, grouping_vector=grouping_vector)
    
    # Test with diag=True
    variance_diag_true = estimator.predict(X, diag=True)
    
    # Check the shape - for density, it should be (n_cells, 1)
    assert variance_diag_true.shape == (n_cells, 1)
    
    # Test with diag=False
    variance_diag_false = estimator.predict(X, diag=False)
    
    # Check the shape - for density, it should be (n_cells, n_cells, 1)
    assert variance_diag_false.shape == (n_cells, n_cells, 1)
    
    # Verify symmetry of the covariance matrix
    cov_matrix = variance_diag_false[:, :, 0]
    np.testing.assert_allclose(
        cov_matrix,
        cov_matrix.T,
        rtol=1e-5,
        atol=1e-8
    )
    
    # Verify the diagonal elements match the diag=True result
    for i in range(n_cells):
        np.testing.assert_allclose(
            variance_diag_false[i, i, 0],
            variance_diag_true[i, 0],
            rtol=1e-5,
            atol=1e-8
        )


def test_sample_variance_estimator_invalid_type():
    """Test that the SampleVarianceEstimator raises an error for invalid estimator_type."""
    # Try to initialize with invalid estimator_type
    with pytest.raises(ValueError, match="estimator_type must be either 'function' or 'density'"):
        SampleVarianceEstimator(estimator_type='invalid')


def test_sample_variance_estimator_function_mode_without_y():
    """Test that the SampleVarianceEstimator raises an error when Y is not provided in function mode."""
    # Generate some sample data
    n_cells = 20
    n_features = 5
    n_groups = 2
    
    X = np.random.randn(n_cells, n_features)
    grouping_vector = np.random.randint(0, n_groups, size=n_cells)
    
    # Initialize in function mode (default)
    estimator = SampleVarianceEstimator()
    
    # Try to fit without providing Y
    with pytest.raises(ValueError, match="Y must be provided for function estimator type"):
        estimator.fit(X=X, grouping_vector=grouping_vector)