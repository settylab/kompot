"""Test to compare disk-backed sample variance with and without Dask support."""

import numpy as np
import pytest
import tempfile
import time
import logging
from kompot.differential import SampleVarianceEstimator
import os

# Set up logging - use INFO level when running the file directly, WARNING when running via pytest
log_level = logging.INFO if __name__ == "__main__" else logging.WARNING 
logging.basicConfig(level=log_level)
logger = logging.getLogger("kompot")


def test_dask_vs_no_dask_comparison():
    """Compare disk-backed sample variance computation with and without Dask."""
    # Generate a moderate-sized dataset
    n_cells = 100  # Smaller dataset for faster CI runs
    n_features = 10
    n_genes = 5
    n_groups = 3
    
    # Set a fixed random seed for reproducibility
    np.random.seed(42)
    
    X = np.random.randn(n_cells, n_features)
    Y = np.random.randn(n_cells, n_genes)
    grouping_vector = np.random.randint(0, n_groups, size=n_cells)
    
    # Test metrics
    timings = {}
    results = {}
    
    # Check Dask availability
    from kompot.memory_utils import DASK_AVAILABLE
    
    if not DASK_AVAILABLE:
        pytest.skip("Dask is not available, skipping test")
    
    # Function to fit and predict
    def run_with_disk_backing(temp_dir, use_dask):
        # Set environment variable to control Dask usage
        if not use_dask:
            # Temporarily patch the module to disable Dask
            import kompot.memory_utils
            original_dask_available = kompot.memory_utils.DASK_AVAILABLE
            kompot.memory_utils.DASK_AVAILABLE = False
        
        try:
            # Create estimator
            estimator = SampleVarianceEstimator(
                store_arrays_on_disk=True,
                disk_storage_dir=temp_dir,
                jit_compile=False  # Disable JIT to focus on disk operations
            )
            
            # Time the fit operation
            fit_start = time.time()
            estimator.fit(X, Y, grouping_vector)
            fit_end = time.time()
            
            # Time the predict operation
            predict_start = time.time()
            diag_result = estimator.predict(X, diag=True)
            predict_end = time.time()
            
            # Also get a smaller covariance matrix for comparison
            small_X = X[:20]  # Use only 20 cells for full covariance to speed up CI tests
            cov_predict_start = time.time()
            cov_result = estimator.predict(small_X, diag=False)
            cov_predict_end = time.time()
            
            return {
                'fit_time': fit_end - fit_start,
                'predict_time': predict_end - predict_start,
                'cov_predict_time': cov_predict_end - cov_predict_start,
                'diag_result': diag_result,
                'cov_result': cov_result
            }
        finally:
            # Restore original Dask status if we modified it
            if not use_dask:
                kompot.memory_utils.DASK_AVAILABLE = original_dask_available
    
    # Run with disk-backing using Dask
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.info("Running with Dask support...")
        result_with_dask = run_with_disk_backing(temp_dir, use_dask=True)
        
        # Run with disk-backing without Dask
        logger.info("Running without Dask support...")
        result_without_dask = run_with_disk_backing(temp_dir, use_dask=False)
        
        # Store results
        timings['with_dask'] = {
            'fit': result_with_dask['fit_time'],
            'predict': result_with_dask['predict_time'],
            'cov_predict': result_with_dask['cov_predict_time']
        }
        
        timings['without_dask'] = {
            'fit': result_without_dask['fit_time'],
            'predict': result_without_dask['predict_time'],
            'cov_predict': result_without_dask['cov_predict_time']
        }
        
        results['with_dask'] = {
            'diag': result_with_dask['diag_result'],
            'cov': result_with_dask['cov_result']
        }
        
        results['without_dask'] = {
            'diag': result_without_dask['diag_result'],
            'cov': result_without_dask['cov_result']
        }
        
        # Print timing information
        logger.info(f"Timings with Dask: {timings['with_dask']}")
        logger.info(f"Timings without Dask: {timings['without_dask']}")
        
        # Calculate differences in results
        diag_abs_diff = np.abs(results['with_dask']['diag'] - results['without_dask']['diag'])
        diag_max_diff = np.max(diag_abs_diff)
        diag_mean_diff = np.mean(diag_abs_diff)
        
        # Convert Dask arrays to numpy arrays if needed
        if hasattr(results['with_dask']['cov'], 'compute'):
            cov_with_dask = results['with_dask']['cov'].compute()
        else:
            cov_with_dask = results['with_dask']['cov']
            
        if hasattr(results['without_dask']['cov'], 'compute'):
            cov_without_dask = results['without_dask']['cov'].compute()
        else:
            cov_without_dask = results['without_dask']['cov']
            
        cov_abs_diff = np.abs(cov_with_dask - cov_without_dask)
        cov_max_diff = np.max(cov_abs_diff)
        cov_mean_diff = np.mean(cov_abs_diff)
        
        logger.info(f"Diagonal result differences - Max: {diag_max_diff}, Mean: {diag_mean_diff}")
        logger.info(f"Covariance result differences - Max: {cov_max_diff}, Mean: {cov_mean_diff}")
        
        # Test that the results are almost identical
        assert np.all(np.isfinite(results['with_dask']['diag'])), "Dask-based diagonal results contain NaN or Inf"
        assert np.all(np.isfinite(results['without_dask']['diag'])), "Non-Dask diagonal results contain NaN or Inf"
        
        # Check if the results are very close (should be identical or extremely close)
        np.testing.assert_allclose(
            results['with_dask']['diag'],
            results['without_dask']['diag'],
            rtol=1e-5,
            atol=1e-8,
            err_msg="Diagonal variance results differ significantly between Dask and non-Dask implementations"
        )
        
        # For covariance matrices, also check they're close
        np.testing.assert_allclose(
            results['with_dask']['cov'],
            results['without_dask']['cov'],
            rtol=1e-5,
            atol=1e-8,
            err_msg="Covariance matrix results differ significantly between Dask and non-Dask implementations"
        )
        
        # Store results for direct execution case
        if __name__ == "__main__":
            return {
                'timings': timings,
                'result_diffs': {
                    'diag_max_diff': diag_max_diff,
                    'diag_mean_diff': diag_mean_diff,
                    'cov_max_diff': cov_max_diff,
                    'cov_mean_diff': cov_mean_diff
                }
            }


if __name__ == "__main__":
    # Run the test directly if this script is executed
    test_results = test_dask_vs_no_dask_comparison()
    print("\nTest Results Summary:")
    
    # Print timing information
    dask_timing = test_results['timings']['with_dask']
    no_dask_timing = test_results['timings']['without_dask']
    
    print(f"Timings with Dask: fit={dask_timing['fit']:.4f}s, predict={dask_timing['predict']:.4f}s, cov={dask_timing['cov_predict']:.4f}s")
    print(f"Timings without Dask: fit={no_dask_timing['fit']:.4f}s, predict={no_dask_timing['predict']:.4f}s, cov={no_dask_timing['cov_predict']:.4f}s")
    
    # Calculate speedup/slowdown
    fit_ratio = dask_timing['fit'] / no_dask_timing['fit']
    predict_ratio = dask_timing['predict'] / no_dask_timing['predict']
    cov_ratio = dask_timing['cov_predict'] / no_dask_timing['cov_predict']
    
    print(f"Performance comparison (Dask vs. no Dask):")
    print(f"  - Fit: {fit_ratio:.2f}x {'slower' if fit_ratio > 1 else 'faster'}")
    print(f"  - Predict: {predict_ratio:.2f}x {'slower' if predict_ratio > 1 else 'faster'}")
    print(f"  - Covariance: {cov_ratio:.2f}x {'slower' if cov_ratio > 1 else 'faster'}")
    
    # Print result differences
    print(f"Result differences:")
    print(f"  - Diagonal max diff: {test_results['result_diffs']['diag_max_diff']}")
    print(f"  - Diagonal mean diff: {test_results['result_diffs']['diag_mean_diff']}")
    print(f"  - Covariance max diff: {test_results['result_diffs']['cov_max_diff']}")
    print(f"  - Covariance mean diff: {test_results['result_diffs']['cov_mean_diff']}")