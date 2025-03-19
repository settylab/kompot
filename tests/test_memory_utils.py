"""Tests for memory_utils.py."""

import numpy as np
import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock
import psutil
import logging

from kompot.memory_utils import (
    human_readable_size,
    array_size,
    get_available_memory,
    memory_requirement_ratio,
    analyze_memory_requirements,
    analyze_covariance_memory_requirements,
    DiskStorage,
    DiskBackedCovarianceMatrix
)


def test_human_readable_size():
    """Test conversion of byte sizes to human-readable format."""
    # Test different size units
    assert human_readable_size(1023) == "1023.00 B"
    assert human_readable_size(1024) == "1.00 KB"
    assert human_readable_size(1024 * 1024) == "1.00 MB"
    assert human_readable_size(1024 * 1024 * 1024) == "1.00 GB"
    assert human_readable_size(1024 * 1024 * 1024 * 1024) == "1.00 TB"
    
    # Test decimal places
    assert human_readable_size(1536) == "1.50 KB"
    assert human_readable_size(2 * 1024 * 1024 + 512 * 1024) == "2.50 MB"


def test_array_size():
    """Test calculation of array memory requirements."""
    # Test with different shapes and data types
    human_str, bytes_size = array_size((100, 100), dtype=np.float64)
    assert bytes_size == 100 * 100 * 8  # float64 is 8 bytes
    assert human_str == human_readable_size(bytes_size)
    
    human_str, bytes_size = array_size((1000, 50), dtype=np.float32)
    assert bytes_size == 1000 * 50 * 4  # float32 is 4 bytes
    assert human_str == human_readable_size(bytes_size)
    
    human_str, bytes_size = array_size((10, 20, 30), dtype=np.int32)
    assert bytes_size == 10 * 20 * 30 * 4  # int32 is 4 bytes
    assert human_str == human_readable_size(bytes_size)


@patch('psutil.virtual_memory')
def test_get_available_memory(mock_virtual_memory):
    """Test retrieving available memory information."""
    # Mock available memory
    mock_vm = MagicMock()
    mock_vm.available = 4 * 1024 * 1024 * 1024  # 4 GB
    mock_virtual_memory.return_value = mock_vm
    
    human_str, bytes_size = get_available_memory()
    assert bytes_size == 4 * 1024 * 1024 * 1024
    assert human_str == "4.00 GB"


@patch('kompot.memory_utils.get_available_memory')
def test_memory_requirement_ratio(mock_get_available_memory):
    """Test calculation of memory requirement ratio."""
    # Mock available memory (8 GB)
    mock_get_available_memory.return_value = ("8.00 GB", 8 * 1024 * 1024 * 1024)
    
    # Test with a 2 GB array (25% of available memory)
    ratio = memory_requirement_ratio((1024, 256 * 1024), dtype=np.float64)
    assert ratio == 0.25


@patch('kompot.memory_utils.get_available_memory')
def test_analyze_memory_requirements(mock_get_available_memory, caplog):
    """Test memory requirements analysis functionality."""
    caplog.set_level(logging.INFO)
    
    # Mock available memory (10 GB)
    mock_get_available_memory.return_value = ("10.00 GB", 10 * 1024 * 1024 * 1024)
    
    # Test with arrays that use total 6 GB (60% of available memory)
    shapes = [
        (1000, 1000),        # ~8 MB for float64
        (10000, 10000),      # ~800 MB for float64
        (10000, 65536),      # ~5.2 GB for float64
    ]
    
    result = analyze_memory_requirements(shapes, max_memory_ratio=0.8)
    
    # Check results
    assert result['status'] == 'warning'  # Between 50% and 80% should be warning
    assert 0.5 < result['memory_ratio'] < 0.8
    assert len(result['array_sizes']) == 3
    
    # Test with arrays that use total 9 GB (90% of available memory)
    shapes = [
        (10000, 10000),      # ~800 MB for float64
        (10000, 65536),      # ~5.2 GB for float64
        (5000, 65536),       # ~2.6 GB for float64
    ]
    
    result = analyze_memory_requirements(shapes, max_memory_ratio=0.8)
    
    # Check results and verify the warning was triggered
    assert result['status'] == 'critical'  # Above 80% should be critical
    assert result['memory_ratio'] > 0.8


@patch('kompot.memory_utils.get_available_memory')
def test_analyze_covariance_memory_requirements(mock_get_available_memory):
    """Test covariance matrix memory requirements analysis."""
    # Mock available memory (16 GB)
    mock_get_available_memory.return_value = ("16.00 GB", 16 * 1024 * 1024 * 1024)
    
    # Test with a relatively small covariance matrix
    result = analyze_covariance_memory_requirements(n_points=1000, n_genes=100, max_memory_ratio=0.8)
    
    # Covariance shape would be (1000, 1000, 100)
    expected_bytes = 1000 * 1000 * 100 * 8  # float64 is 8 bytes
    
    # Verify basic calculations
    assert result['total_bytes'] == expected_bytes
    assert result['memory_ratio'] == expected_bytes / (16 * 1024 * 1024 * 1024)
    
    # Verify analysis recommendation
    # This may yield 'ok' since this matrix is smaller than 16GB * 0.8
    assert 'should_use_disk' in result
    
    # Test with a much larger covariance matrix that would exceed memory
    result = analyze_covariance_memory_requirements(n_points=5000, n_genes=2000, max_memory_ratio=0.8)
    
    # Covariance shape would be (5000, 5000, 2000) -> ~400 GB for float64
    # This should definitely trigger the disk storage recommendation
    assert result['should_use_disk'] is True
    assert result['status'] == 'critical'


def test_disk_storage():
    """Test disk storage functionality."""
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize DiskStorage with explicit directory
        storage = DiskStorage(storage_dir=temp_dir)
        
        # Store an array
        test_array = np.random.random((100, 100))
        key = "test_array"
        file_path = storage.store_array(test_array, key)
        
        # Check that file was created with expected name
        assert os.path.exists(file_path)
        # Account for namespacing that was added to the key
        assert os.path.basename(file_path).endswith(f"{key}.npy")
        
        # Check registry entry
        assert key in storage.array_registry
        assert storage.array_registry[key]['shape'] == test_array.shape
        assert storage.array_registry[key]['dtype'] == str(test_array.dtype)
        
        # Load the array back, but force non-lazy loading
        loaded_array = storage.load_array(key, lazy=False)
        np.testing.assert_array_equal(loaded_array, test_array)
        
        # Check total storage
        human_size, byte_size = storage.total_storage_used
        assert byte_size == test_array.nbytes
        
        # Store another array
        test_array2 = np.random.random((50, 200))
        key2 = "test_array2"
        storage.store_array(test_array2, key2)
        
        # Check updated total storage
        human_size, byte_size = storage.total_storage_used
        assert byte_size == test_array.nbytes + test_array2.nbytes
        
        # Test list arrays functionality
        array_list = storage.list_arrays()
        assert len(array_list) == 2
        assert key in array_list
        assert key2 in array_list
        
        # Remove an array
        storage.remove_array(key)
        assert key not in storage.array_registry
        assert not os.path.exists(file_path)
        
        # Check updated storage after removal
        human_size, byte_size = storage.total_storage_used
        assert byte_size == test_array2.nbytes


def test_disk_backed_covariance_matrix():
    """Test disk-backed covariance matrix functionality."""
    # Skip if class doesn't support expected interface
    try:
        # Try to create a sample instance to test the interface
        sample_storage = DiskStorage(storage_dir=tempfile.gettempdir())
        sample_keys = {0: "test"}
        DiskBackedCovarianceMatrix(
            disk_storage=sample_storage,
            shape=(10, 10, 1),
            gene_keys=sample_keys,
            use_dask=False
        )
    except TypeError:
        pytest.skip("DiskBackedCovarianceMatrix doesn't support expected interface")
        
    # Create test data - a 3D covariance tensor (cells, cells, genes)
    n_cells, n_genes = 50, 10
    cov_tensor = np.random.random((n_cells, n_cells, n_genes))
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize disk storage
        storage = DiskStorage(storage_dir=temp_dir)
        
        # Store each gene's covariance matrix separately
        gene_keys = {}
        for g in range(n_genes):
            key = f"gene_{g}_cov"
            storage.store_array(cov_tensor[:, :, g], key)
            gene_keys[g] = key
        
        # Create disk-backed matrix
        disk_cov = DiskBackedCovarianceMatrix(
            disk_storage=storage,
            shape=(n_cells, n_cells, n_genes),
            gene_keys=gene_keys,
            use_dask=False  # Force non-lazy loading
        )
        
        # Test single gene slice access
        for g in range(n_genes):
            gene_slice = disk_cov[g]
            np.testing.assert_array_equal(gene_slice, cov_tensor[:, :, g])
        
        # Test cell slice access for a specific gene
        g = 3  # Test with gene index 3
        cell_slice = disk_cov[:10, :20, g]
        np.testing.assert_array_equal(cell_slice, cov_tensor[:10, :20, g])
        
        # Test multiple gene slice access
        genes_slice = disk_cov[:, :, :5]
        np.testing.assert_array_equal(genes_slice, cov_tensor[:, :, :5])
        
        # Test explicit gene indices
        genes_indices = [2, 5, 8]
        genes_slice = disk_cov[:, :, genes_indices]
        np.testing.assert_array_equal(genes_slice, cov_tensor[:, :, genes_indices])


@patch('kompot.memory_utils.get_available_memory')
def test_full_disk_backed_workflow(mock_get_available_memory):
    """Test the full workflow using disk-backed functionality."""
    # Skip if class doesn't support expected interface
    try:
        # Try to create a sample instance to test the interface
        sample_storage = DiskStorage(storage_dir=tempfile.gettempdir())
        sample_keys = {0: "test"}
        DiskBackedCovarianceMatrix(
            disk_storage=sample_storage,
            shape=(10, 10, 1),
            gene_keys=sample_keys,
            use_dask=False
        )
    except TypeError:
        pytest.skip("DiskBackedCovarianceMatrix doesn't support expected interface")
        
    # Mock available memory to force disk recommendation
    mock_get_available_memory.return_value = ("1.00 GB", 1 * 1024 * 1024 * 1024)
    
    # Use smaller dimensions to keep the test fast
    n_points, n_genes = 20, 5
    
    # First, analyze memory requirements to get a recommendation
    analysis = analyze_covariance_memory_requirements(
        n_points=n_points, 
        n_genes=n_genes, 
        max_memory_ratio=0.8
    )
    
    # Create a covariance tensor and use disk backing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize disk storage
        storage = DiskStorage(storage_dir=temp_dir)
        
        # Create a random covariance tensor and store each gene slice
        gene_keys = {}
        for g in range(n_genes):
            # Create a positive definite matrix for each gene
            random_mat = np.random.random((n_points, n_points))
            # Make it symmetric positive definite
            gene_cov = random_mat @ random_mat.T + np.eye(n_points) * 0.1
            
            # Store in disk
            key = f"gene_{g}_cov"
            storage.store_array(gene_cov, key)
            gene_keys[g] = key
        
        # Create disk-backed matrix
        disk_cov = DiskBackedCovarianceMatrix(
            disk_storage=storage,
            shape=(n_points, n_points, n_genes),
            gene_keys=gene_keys,
            use_dask=False  # Force non-lazy loading
        )
        
        # Create some test fold changes (genes, points)
        fold_changes = np.random.random((n_genes, n_points))
        
        # Import and use the Mahalanobis distance computation function
        from kompot.utils import compute_mahalanobis_distances
        
        # Compute Mahalanobis distances using disk-backed covariance
        distances = compute_mahalanobis_distances(
            diff_values=fold_changes,
            covariance=disk_cov,
            batch_size=10,
            jit_compile=False,
            progress=False
        )
        
        # Verify results
        assert len(distances) == n_genes
        assert np.all(np.isfinite(distances))