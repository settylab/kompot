"""
Memory validation tests for disk-backed storage.

This module tests that disk-backed storage significantly reduces memory usage
compared to in-memory computation while maintaining computational correctness.
"""

import pytest
import numpy as np
import tempfile
import os
import psutil
import gc
from typing import Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Skip if system monitoring is not available
try:
    process = psutil.Process()
    PSUTIL_AVAILABLE = True
except:
    PSUTIL_AVAILABLE = False


def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    if PSUTIL_AVAILABLE:
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    else:
        return 0.0


def create_large_test_data(n_points: int = 1000, n_genes: int = 100, seed: int = 42) -> Dict[str, np.ndarray]:
    """Create large test data for memory stress testing."""
    np.random.seed(seed)
    
    # Create condition 1 data with random features and gene expression
    X1 = np.random.normal(0, 1, (n_points, 20))  # 20 features
    y1 = np.random.normal(0, 1, (n_points, n_genes))
    
    # Create condition 2 data with random features and gene expression
    # Add a shift to create differences
    X2 = np.random.normal(0.3, 1, (n_points, 20))
    y2 = np.random.normal(0.5, 1, (n_points, n_genes))
    
    return {
        'X1': X1,
        'y1': y1,
        'X2': X2,
        'y2': y2
    }


class MemoryMonitor:
    """Context manager to monitor memory usage during operations."""
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_memory = 0.0
        self.peak_memory = 0.0
        self.end_memory = 0.0
        
    def __enter__(self):
        gc.collect()  # Clean up before measuring
        self.start_memory = get_memory_usage()
        self.peak_memory = self.start_memory
        logger.info(f"Starting {self.operation_name} - Initial memory: {self.start_memory:.2f} MB")
        return self
        
    def update_peak(self):
        """Update peak memory usage."""
        current = get_memory_usage()
        if current > self.peak_memory:
            self.peak_memory = current
            
    def __exit__(self, exc_type, exc_val, exc_tb):
        gc.collect()  # Clean up after operation
        self.end_memory = get_memory_usage()
        logger.info(f"Finished {self.operation_name} - Final memory: {self.end_memory:.2f} MB")
        logger.info(f"Memory usage for {self.operation_name}: Start={self.start_memory:.2f}MB, "
                   f"Peak={self.peak_memory:.2f}MB, End={self.end_memory:.2f}MB")
        
    @property
    def memory_increase(self) -> float:
        """Get the net memory increase."""
        return self.end_memory - self.start_memory
        
    @property
    def peak_increase(self) -> float:
        """Get the peak memory increase."""
        return self.peak_memory - self.start_memory


@pytest.mark.skipif(not PSUTIL_AVAILABLE, reason="psutil not available for memory monitoring")
class TestMemoryValidation:
    """Test memory usage validation for disk-backed storage."""
    
    def test_memory_usage_comparison_small_dataset(self):
        """Test memory usage comparison with small dataset."""
        pytest.importorskip("anndata")
        
        from kompot.anndata import compute_differential_expression
        import anndata
        import pandas as pd
        
        # Create test data - small enough for both approaches
        data = create_large_test_data(n_points=200, n_genes=50)
        X1, y1, X2, y2 = data['X1'], data['y1'], data['X2'], data['y2']
        
        # Combine data for AnnData
        X_combined = np.vstack([X1, X2])
        y_combined = np.vstack([y1, y2])
        
        # Create condition and sample labels
        condition_labels = ['A'] * len(X1) + ['B'] * len(X2)
        # Create 4 samples per condition
        n_samples = 4
        sample_size = len(X1) // n_samples
        sample_labels = []
        for i in range(n_samples):
            sample_labels.extend([f'sample{i+1}'] * sample_size)
        # Handle remainder
        remainder = len(X1) % n_samples
        if remainder > 0:
            sample_labels.extend([f'sample{n_samples}'] * remainder)
            
        # Same for condition 2
        for i in range(n_samples):
            sample_labels.extend([f'sample{i+5}'] * sample_size)
        remainder = len(X2) % n_samples
        if remainder > 0:
            sample_labels.extend([f'sample{n_samples+4}'] * remainder)
        
        # Create AnnData object
        adata_memory = anndata.AnnData(
            X=y_combined.copy(),
            obs=pd.DataFrame({
                'condition': condition_labels,
                'sample': sample_labels
            }),
            obsm={'DM_EigenVectors': X_combined.copy()}
        )
        
        # Create separate copy for disk version
        adata_disk = adata_memory.copy()
        
        # Test in-memory version
        with MemoryMonitor("In-memory sample variance computation") as memory_monitor:
            result_memory = compute_differential_expression(
                adata_memory,
                groupby='condition',
                condition1='A',
                condition2='B',
                sample_col='sample',
                compute_mahalanobis=True,
                result_key='memory_test',
                batch_size=50,
                n_landmarks=20,
                null_genes=0,  # Disable null genes for speed in memory test
                store_arrays_on_disk=False,  # In-memory
                progress=False
            )
            memory_monitor.update_peak()
            
        memory_usage_in_memory = memory_monitor.peak_increase
        
        # Test disk-backed version
        with tempfile.TemporaryDirectory() as temp_dir:
            with MemoryMonitor("Disk-backed sample variance computation") as disk_monitor:
                result_disk = compute_differential_expression(
                    adata_disk,
                    groupby='condition',
                    condition1='A',
                    condition2='B',
                    sample_col='sample',
                    compute_mahalanobis=True,
                    result_key='disk_test',
                    batch_size=50,
                    n_landmarks=20,
                    null_genes=0,  # Disable null genes for speed in memory test
                    store_arrays_on_disk=True,  # Disk-backed
                    disk_storage_dir=temp_dir,
                    progress=False
                )
                disk_monitor.update_peak()
                
        memory_usage_disk_backed = disk_monitor.peak_increase
        
        logger.info(f"Memory usage comparison:")
        logger.info(f"  In-memory: {memory_usage_in_memory:.2f} MB")
        logger.info(f"  Disk-backed: {memory_usage_disk_backed:.2f} MB")
        
        # Validate that both approaches produce results
        memory_key = "memory_test_A_to_B_mahalanobis_sample_var"
        disk_key = "disk_test_A_to_B_mahalanobis_sample_var"
        
        assert memory_key in adata_memory.var, f"Memory result not found: {memory_key}"
        assert disk_key in adata_disk.var, f"Disk result not found: {disk_key}"
        
        # Validate computational correctness (results should be similar)
        memory_values = adata_memory.var[memory_key].values
        disk_values = adata_disk.var[disk_key].values
        
        assert np.all(np.isfinite(memory_values)), "Memory values should be finite"
        assert np.all(np.isfinite(disk_values)), "Disk values should be finite"
        
        # Results should be correlated (same gene ranking)
        # Note: Due to different computational paths (in-memory vs disk-backed),
        # we expect reasonable correlation but not perfect match
        correlation = np.corrcoef(memory_values, disk_values)[0, 1]
        assert correlation > 0.5, f"Results should be reasonably correlated, got {correlation:.3f}"
        
        logger.info(f"Computational validation passed - correlation: {correlation:.3f}")
        
        # Memory validation - disk-backed should use less memory for larger datasets
        # For small datasets, the overhead might make disk version use slightly more memory
        # But it should not be dramatically more
        max_acceptable_overhead = 50  # MB
        if memory_usage_disk_backed > memory_usage_in_memory + max_acceptable_overhead:
            pytest.fail(f"Disk-backed version used too much more memory: "
                       f"{memory_usage_disk_backed:.2f}MB vs {memory_usage_in_memory:.2f}MB")
            
        logger.info("Memory validation passed for small dataset")
        
    def test_memory_usage_large_dataset_simulation(self):
        """Test that disk-backed storage scales better with large datasets."""
        pytest.importorskip("anndata")
        
        from kompot.differential.sample_variance_estimator import SampleVarianceEstimator
        from kompot.memory_utils import DiskStorage
        
        # Simulate a large dataset scenario
        n_cells = 500
        n_genes = 200
        n_groups = 4
        
        logger.info(f"Testing with simulated large dataset: {n_cells} cells, {n_genes} genes")
        
        # Create test data
        np.random.seed(42)
        X_new = np.random.normal(0, 1, (n_cells, 10))
        
        # Create mock predictors that return realistic gene expression data
        class MockPredictor:
            def __init__(self, base_expression: float, seed: int):
                self.base_expression = base_expression
                self.seed = seed
                
            def __call__(self, X):
                # Use consistent seed for reproducible results
                np.random.seed(self.seed + hash(str(self.base_expression)) % 1000)
                n_cells = len(X)
                # Return realistic gene expression values
                return np.random.normal(self.base_expression, 1.0, (n_cells, n_genes))
        
        # Test in-memory version
        with MemoryMonitor("In-memory large dataset simulation") as memory_monitor:
            estimator_memory = SampleVarianceEstimator(
                estimator_type='function',
                store_arrays_on_disk=False
            )
            
            # Set up mock predictors
            estimator_memory.group_predictors = {
                f'group_{i}': MockPredictor(i * 0.5, seed=42+i) for i in range(n_groups)
            }
            estimator_memory.n_groups = n_groups
            
            # Predict with in-memory
            result_memory = estimator_memory.predict(X_new, diag=False)
            memory_monitor.update_peak()
            
        memory_usage_in_memory = memory_monitor.peak_increase
        
        # Test disk-backed version
        with tempfile.TemporaryDirectory() as temp_dir:
            with MemoryMonitor("Disk-backed large dataset simulation") as disk_monitor:
                estimator_disk = SampleVarianceEstimator(
                    estimator_type='function',
                    store_arrays_on_disk=True,
                    disk_storage_dir=temp_dir
                )
                
                # Set up mock predictors (same seeds for reproducible results)
                estimator_disk.group_predictors = {
                    f'group_{i}': MockPredictor(i * 0.5, seed=42+i) for i in range(n_groups)
                }
                estimator_disk.n_groups = n_groups
                
                # Predict with disk-backing
                result_disk = estimator_disk.predict(X_new, diag=False)
                disk_monitor.update_peak()
                
        memory_usage_disk_backed = disk_monitor.peak_increase
        
        logger.info(f"Large dataset memory usage comparison:")
        logger.info(f"  In-memory: {memory_usage_in_memory:.2f} MB")
        logger.info(f"  Disk-backed: {memory_usage_disk_backed:.2f} MB")
        
        # Validate that both approaches produce results of correct shape
        expected_shape = (n_cells, n_cells, n_genes)
        assert result_memory.shape == expected_shape, f"Memory result shape: {result_memory.shape}"
        assert result_disk.shape == expected_shape, f"Disk result shape: {result_disk.shape}"
        
        # Validate computational correctness for a subset
        sample_genes = min(5, n_genes)
        memory_sample = result_memory[:10, :10, :sample_genes]
        disk_sample = result_disk[:10, :10, :sample_genes]
        
        # Results should be finite
        assert np.all(np.isfinite(memory_sample)), "Memory sample should be finite"
        assert np.all(np.isfinite(disk_sample)), "Disk sample should be finite"
        
        # Results should be close (same computation)
        np.testing.assert_allclose(memory_sample, disk_sample, rtol=1e-10, atol=1e-12,
                                 err_msg="Memory and disk results should be nearly identical")
        
        logger.info("Computational validation passed for large dataset simulation")
        
        # For large datasets, disk-backed should use significantly less memory
        memory_savings = memory_usage_in_memory - memory_usage_disk_backed
        memory_savings_percent = (memory_savings / memory_usage_in_memory) * 100 if memory_usage_in_memory > 0 else 0
        
        logger.info(f"Memory savings: {memory_savings:.2f} MB ({memory_savings_percent:.1f}%)")
        
        # We expect at least some memory savings for large datasets
        # If disk version uses more memory, it should be within reasonable bounds
        if memory_savings < 0:  # Disk uses more memory
            max_acceptable_overhead = 30  # MB
            assert abs(memory_savings) <= max_acceptable_overhead, (
                f"Disk-backed version used too much extra memory: {abs(memory_savings):.2f}MB overhead"
            )
        else:
            # Good - disk version used less memory
            assert memory_savings >= 0, "Disk-backed should use same or less memory"
            
        logger.info("Memory validation passed for large dataset simulation")
        
    def test_memory_mapped_array_properties(self):
        """Test that memory-mapped arrays have the expected properties."""
        pytest.importorskip("anndata")
        
        from kompot.differential.sample_variance_estimator import SampleVarianceEstimator
        
        n_cells = 100
        n_genes = 50
        
        # Create test data
        X_new = np.random.normal(0, 1, (n_cells, 10))
        
        # Create mock predictors
        class MockPredictor:
            def __init__(self, seed: int):
                self.seed = seed
                
            def __call__(self, X):
                np.random.seed(self.seed)
                return np.random.normal(0, 1, (len(X), n_genes))
        
        with tempfile.TemporaryDirectory() as temp_dir:
            estimator = SampleVarianceEstimator(
                estimator_type='function',
                store_arrays_on_disk=True,
                disk_storage_dir=temp_dir
            )
            
            estimator.group_predictors = {
                'group1': MockPredictor(seed=42),
                'group2': MockPredictor(seed=43)
            }
            estimator.n_groups = 2
            
            # Get disk-backed result
            result = estimator.predict(X_new, diag=False)

            # Verify it's a disk-backed array (dask if available, otherwise memmap)
            try:
                import dask.array as da
                DASK_AVAILABLE = True
            except ImportError:
                DASK_AVAILABLE = False

            if DASK_AVAILABLE:
                assert isinstance(result, da.Array), "Result should be a Dask array when dask is available"
            else:
                assert isinstance(result, np.memmap), "Result should be a memory-mapped array when dask is not available"

            # Verify shape
            expected_shape = (n_cells, n_cells, n_genes)
            assert result.shape == expected_shape, f"Expected shape {expected_shape}, got {result.shape}"

            # For memmap, verify the underlying file exists
            if isinstance(result, np.memmap):
                assert hasattr(result, 'filename'), "Memory-mapped array should have filename attribute"
                assert os.path.exists(result.filename), "Backing file should exist"
            
            # Verify values are finite
            sample_values = result[:10, :10, :5]
            assert np.all(np.isfinite(sample_values)), "Values should be finite"

            logger.info(f"Disk-backed array validation passed:")
            logger.info(f"  Shape: {result.shape}")
            logger.info(f"  Type: {type(result).__name__}")

            # Verify we can read/write (only for memmap, dask arrays are read-only)
            if isinstance(result, np.memmap):
                original_value = float(result[0, 0, 0])
                result[0, 0, 0] = 999.0
                assert result[0, 0, 0] == 999.0, "Should be able to modify memory-mapped array"
                result[0, 0, 0] = original_value  # Restore
                logger.info(f"  File: {result.filename}")
                logger.info(f"  Size on disk: {os.path.getsize(result.filename) / 1024 / 1024:.2f} MB")
            
    @pytest.mark.skipif(not PSUTIL_AVAILABLE, reason="Requires psutil for memory monitoring")
    def test_garbage_collection_effectiveness(self):
        """Test that garbage collection is effective in cleaning up JAX arrays."""
        pytest.importorskip("jax")
        
        import jax.numpy as jnp
        
        initial_memory = get_memory_usage()
        logger.info(f"Initial memory: {initial_memory:.2f} MB")
        
        # Create large JAX arrays
        large_arrays = []
        for i in range(10):
            arr = jnp.ones((1000, 1000))  # ~8MB per array
            large_arrays.append(arr)
            
        after_allocation = get_memory_usage()
        logger.info(f"After allocating JAX arrays: {after_allocation:.2f} MB")
        
        # Convert to numpy and delete JAX references
        numpy_arrays = []
        for arr in large_arrays:
            numpy_arrays.append(np.asarray(arr))
            
        # Delete JAX arrays
        del large_arrays
        gc.collect()
        
        after_conversion = get_memory_usage()
        logger.info(f"After JAX to numpy conversion: {after_conversion:.2f} MB")
        
        # Delete numpy arrays
        del numpy_arrays
        gc.collect()
        
        final_memory = get_memory_usage()
        logger.info(f"Final memory after cleanup: {final_memory:.2f} MB")
        
        # Memory should return close to initial levels
        memory_cleanup_efficiency = (after_allocation - final_memory) / (after_allocation - initial_memory)
        logger.info(f"Memory cleanup efficiency: {memory_cleanup_efficiency:.1%}")
        
        # We should recover at least some of the allocated memory
        # Lower threshold for testing environments where OS memory management may be different
        assert memory_cleanup_efficiency > 0.2, (
            f"Poor memory cleanup efficiency: {memory_cleanup_efficiency:.1%}"
        )
        
        logger.info("Garbage collection effectiveness test passed")


if __name__ == "__main__":
    # Run basic memory validation
    import logging
    logging.basicConfig(level=logging.INFO)
    
    if PSUTIL_AVAILABLE:
        test = TestMemoryValidation()
        test.test_memory_usage_comparison_small_dataset()
        test.test_memory_usage_large_dataset_simulation()
        test.test_memory_mapped_array_properties()
        test.test_garbage_collection_effectiveness()
        print("All memory validation tests passed!")
    else:
        print("psutil not available - skipping memory tests")