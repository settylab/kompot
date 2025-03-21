"""Test sample variance estimator with different disk backend configurations."""

import numpy as np
import tempfile
import logging
import pytest
from unittest.mock import patch, MagicMock
from kompot.differential.sample_variance_estimator import SampleVarianceEstimator
from kompot.memory_utils import DASK_AVAILABLE, DiskStorage

# Set up logging - use INFO level when running directly, WARNING when running via pytest
log_level = logging.INFO if __name__ == "__main__" else logging.WARNING 
logging.basicConfig(level=log_level)
logger = logging.getLogger("kompot")


def test_disk_storage_initialization():
    """Test that DiskStorage initializes correctly with and without Dask."""
    # Skip if Dask is not available
    if not DASK_AVAILABLE:
        pytest.skip("Dask is not available, skipping test")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test with Dask
        storage_with_dask = DiskStorage(temp_dir, namespace="test_dask", use_dask=True)
        assert storage_with_dask.use_dask is True
        
        # Test without Dask
        storage_without_dask = DiskStorage(temp_dir, namespace="test_no_dask", use_dask=False)
        assert storage_without_dask.use_dask is False
        
        # Verify both are functional by storing and retrieving arrays
        test_array = np.ones((10, 5))
        
        # DiskStorage.store_array has parameters (self, array, key), not (self, key, array)
        storage_with_dask.store_array(test_array, "test_array")
        retrieved_with_dask = storage_with_dask.load_array("test_array")
        
        storage_without_dask.store_array(test_array, "test_array")
        retrieved_without_dask = storage_without_dask.load_array("test_array")
        
        np.testing.assert_array_equal(test_array, retrieved_with_dask)
        np.testing.assert_array_equal(test_array, retrieved_without_dask)


def test_sample_variance_estimator_disk_storage():
    """Test SampleVarianceEstimator with different disk storage configurations."""
    # Skip if Dask is not available
    if not DASK_AVAILABLE:
        pytest.skip("Dask is not available, skipping test")
    
    # Generate simple test data
    np.random.seed(42)
    n_cells = 50
    n_features = 5
    n_genes = 3
    n_groups = 2
    
    X = np.random.randn(n_cells, n_features)
    Y = np.random.randn(n_cells, n_genes)
    grouping_vector = np.random.randint(0, n_groups, size=n_cells)
    
    results = {}
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test with both Dask and non-Dask configurations
        for use_dask in [True, False]:
            # Mock to control Dask availability
            with patch('kompot.memory_utils.DASK_AVAILABLE', use_dask):
                # Initialize estimator
                estimator = SampleVarianceEstimator(
                    store_arrays_on_disk=True,
                    disk_storage_dir=temp_dir,
                    jit_compile=False
                )
                
                # Check if disk_storage is initialized appropriately
                estimator.fit(X, Y, grouping_vector)
                                
                # Verify that get_directory_size doesn't crash
                if hasattr(estimator, 'disk_storage') and estimator.disk_storage is not None:
                    storage_size = estimator.disk_storage.get_directory_size()
                    logger.info(f"Storage size with use_dask={use_dask}: {storage_size} bytes")
                
                # Test prediction with diagonal variance (this should work in both configurations)
                diag_result = estimator.predict(X, diag=True)
                results[f'diag_{use_dask}'] = diag_result
                
                # Don't test full covariance for large matrices to avoid inefficiency
                # Only test with a small subset
                small_X = X[:10]
                cov_result = estimator.predict(small_X, diag=False)
                results[f'cov_{use_dask}'] = cov_result
    
    # Verify that results are identical between configurations
    np.testing.assert_allclose(
        results['diag_True'],
        results['diag_False'],
        rtol=1e-5, atol=1e-8,
        err_msg="Diagonal variance differs between Dask and non-Dask configurations"
    )
    
    np.testing.assert_allclose(
        results['cov_True'],
        results['cov_False'],
        rtol=1e-5, atol=1e-8,
        err_msg="Covariance matrix differs between Dask and non-Dask configurations"
    )
    
    # Calculate differences for reporting
    diag_diff = np.abs(results['diag_True'] - results['diag_False'])
    cov_diff = np.abs(results['cov_True'] - results['cov_False'])
    
    max_diag_diff = np.max(diag_diff)
    mean_diag_diff = np.mean(diag_diff)
    max_cov_diff = np.max(cov_diff)
    mean_cov_diff = np.mean(cov_diff)
    
    logger.info(f"Diagonal variance diffs - Max: {max_diag_diff}, Mean: {mean_diag_diff}")
    logger.info(f"Covariance matrix diffs - Max: {max_cov_diff}, Mean: {mean_cov_diff}")
    
    # Return result info when run directly
    if __name__ == "__main__":
        return {
            'diag_diffs': {'max': max_diag_diff, 'mean': mean_diag_diff},
            'cov_diffs': {'max': max_cov_diff, 'mean': mean_cov_diff}
        }


if __name__ == "__main__":
    # Run the tests directly
    results = test_sample_variance_estimator_disk_storage()
    
    print("\nTest Results Summary:")
    print(f"Diagonal variance differences:")
    print(f"  - Max: {results['diag_diffs']['max']}")
    print(f"  - Mean: {results['diag_diffs']['mean']}")
    
    print(f"Covariance matrix differences:")
    print(f"  - Max: {results['cov_diffs']['max']}")
    print(f"  - Mean: {results['cov_diffs']['mean']}")