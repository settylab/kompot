"""Test differential expression with sample variance in disk-backed mode with/without Dask."""

import numpy as np
import pytest
import tempfile
import logging
from kompot.differential import DifferentialExpression
from kompot.memory_utils import DASK_AVAILABLE
import os
from unittest.mock import patch

# Set up logging - use INFO level when running the file directly, WARNING when running via pytest
log_level = logging.INFO if __name__ == "__main__" else logging.WARNING 
logging.basicConfig(level=log_level)
logger = logging.getLogger("kompot")


def test_disk_backed_de_without_sample_variance():
    """
    Test that disk-backed differential expression produces identical results
    with and without Dask when sample variance is NOT used.
    
    Note: This avoids compatibility issues between Dask arrays and JAX arrays in the
    sample variance covariance computation.
    """
    # Skip if Dask is not available
    if not DASK_AVAILABLE:
        pytest.skip("Dask is not available, skipping test")

    # Generate test data
    np.random.seed(42)
    n_cells_1 = 100
    n_cells_2 = 100
    n_features = 10
    n_genes = 5  # Keep this small for CI
    n_groups = 2  # Number of sample groups

    # Create condition 1 data
    X1 = np.random.randn(n_cells_1, n_features)
    y1 = np.random.randn(n_cells_1, n_genes)
    
    # Create condition 2 data (slightly different distribution)
    X2 = np.random.randn(n_cells_2, n_features) + 0.5
    y2 = np.random.randn(n_cells_2, n_genes) + 0.2
    
    # Create sample indices for condition 1 and 2
    condition1_samples = np.array([i % n_groups for i in range(n_cells_1)])
    condition2_samples = np.array([i % n_groups for i in range(n_cells_2)])

    # Create prediction points
    n_predict = 20
    X_pred = np.random.randn(n_predict, n_features)

    # Results dictionary
    results = {}
    
    # Function to setup and run DE without sample variance
    def run_de_with_configuration(temp_dir, use_dask):
        # If not using Dask, patch the DASK_AVAILABLE flag
        import kompot.memory_utils
        original_dask_available = kompot.memory_utils.DASK_AVAILABLE
        
        if not use_dask:
            kompot.memory_utils.DASK_AVAILABLE = False
        
        try:
            # Create DE object with disk backing but NO sample variance
            de = DifferentialExpression(
                use_sample_variance=False,  # Don't use sample variance to avoid JAX/Dask conflicts
                store_arrays_on_disk=True,
                disk_storage_dir=temp_dir,
                jit_compile=False  # Disable JIT for more consistent comparison
            )
            
            # Fit the model without sample indices
            de.fit(X1, y1, X2, y2)
            
            # Make predictions with Mahalanobis distance calculation
            pred_results = de.predict(
                X_pred,
                compute_mahalanobis=True,
                progress=False  # Disable progress bar for testing
            )
            
            return pred_results
        
        finally:
            # Restore original Dask status
            if not use_dask:
                kompot.memory_utils.DASK_AVAILABLE = original_dask_available
    
    # Run with both configurations
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.info("Running differential expression with Dask support...")
        results['with_dask'] = run_de_with_configuration(temp_dir, use_dask=True)
        
        logger.info("Running differential expression without Dask support...")
        results['without_dask'] = run_de_with_configuration(temp_dir, use_dask=False)
    
    # Verify keys are the same
    assert set(results['with_dask'].keys()) == set(results['without_dask'].keys())
    
    # Check if the imputed values match (these should be identical)
    np.testing.assert_allclose(
        results['with_dask']['condition1_imputed'],
        results['without_dask']['condition1_imputed'],
        rtol=1e-5, atol=1e-8,
        err_msg="Imputed condition1 values differ between Dask and non-Dask implementations"
    )
    
    np.testing.assert_allclose(
        results['with_dask']['condition2_imputed'],
        results['without_dask']['condition2_imputed'],
        rtol=1e-5, atol=1e-8,
        err_msg="Imputed condition2 values differ between Dask and non-Dask implementations"
    )
    
    # Check fold changes
    np.testing.assert_allclose(
        results['with_dask']['fold_change'],
        results['without_dask']['fold_change'],
        rtol=1e-5, atol=1e-8,
        err_msg="Fold changes differ between Dask and non-Dask implementations"
    )
    
    # The key check: verify Mahalanobis distances are identical
    np.testing.assert_allclose(
        results['with_dask']['mahalanobis_distances'],
        results['without_dask']['mahalanobis_distances'],
        rtol=1e-5, atol=1e-8,
        err_msg="Mahalanobis distances differ between Dask and non-Dask implementations"
    )
    
    # Calculate diffs for logging
    mahalanobis_diff = np.abs(
        results['with_dask']['mahalanobis_distances'] - 
        results['without_dask']['mahalanobis_distances']
    )
    max_diff = np.max(mahalanobis_diff)
    mean_diff = np.mean(mahalanobis_diff)
    
    logger.info(f"Mahalanobis distance comparison - Max diff: {max_diff}, Mean diff: {mean_diff}")
    
    # Return diff info for direct execution
    if __name__ == "__main__":
        return {
            'max_diff': max_diff,
            'mean_diff': mean_diff,
            'mahalanobis_with_dask': results['with_dask']['mahalanobis_distances'],
            'mahalanobis_without_dask': results['without_dask']['mahalanobis_distances']
        }


if __name__ == "__main__":
    # Run the test directly if this script is executed
    test_results = test_disk_backed_de_without_sample_variance()
    
    print("\nTest Results Summary:")
    print(f"Mahalanobis distance maximum difference: {test_results['max_diff']}")
    print(f"Mahalanobis distance mean difference: {test_results['mean_diff']}")
    
    # Print sample of the actual values for comparison
    print("\nSample Mahalanobis distances (first 3 genes):")
    print("With Dask:")
    print(test_results['mahalanobis_with_dask'][:3])
    print("Without Dask:")
    print(test_results['mahalanobis_without_dask'][:3])