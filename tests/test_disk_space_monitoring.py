"""Tests for disk space monitoring in DiskStorage."""

import numpy as np
import pytest
import tempfile
import os


def test_disk_space_utility_functions():
    """Test that disk space utility functions work."""
    from kompot.memory_utils import get_disk_space, estimate_disk_requirement

    # Test get_disk_space on /tmp
    total_h, total_bytes, used_h, used_bytes, free_h, free_bytes = get_disk_space("/tmp")

    # Should have valid values
    assert total_bytes > 0
    assert free_bytes > 0
    assert used_bytes >= 0
    assert total_bytes >= used_bytes + free_bytes  # Allow for some rounding

    # Human readable strings should have units
    assert any(unit in total_h for unit in ['B', 'KB', 'MB', 'GB', 'TB'])

    # Test estimate_disk_requirement
    n_cells = 100
    n_genes = 50
    size_h, size_bytes = estimate_disk_requirement(n_cells, n_genes)

    # Should be reasonable size for 100x100x50 float64 array
    expected_min = 100 * 100 * 50 * 8  # Minimum without overhead
    expected_max = expected_min * 1.5  # Maximum with overhead

    assert size_bytes >= expected_min
    assert size_bytes <= expected_max


def test_disk_storage_space_check():
    """Test that DiskStorage checks disk space on initialization."""
    from kompot.memory_utils import DiskStorage

    with tempfile.TemporaryDirectory() as tmpdir:
        # This should succeed - we're not requesting huge amounts of space
        storage = DiskStorage(
            storage_dir=tmpdir,
            n_cells=10,
            n_genes=5
        )

        assert storage.storage_dir == tmpdir
        assert len(storage.array_registry) == 0

        # Clean up
        storage.cleanup()


def test_disk_storage_insufficient_space_error():
    """Test that DiskStorage raises error when disk space is insufficient."""
    from kompot.memory_utils import DiskStorage

    with tempfile.TemporaryDirectory() as tmpdir:
        # Try to request an impossibly large amount of storage
        # This should raise an IOError
        with pytest.raises(IOError, match="Insufficient disk space"):
            storage = DiskStorage(
                storage_dir=tmpdir,
                expected_size_bytes=10**15  # 1 PB - should exceed available space
            )


def test_disk_storage_monitoring_during_write():
    """Test that DiskStorage monitors space during array writes."""
    from kompot.memory_utils import DiskStorage

    with tempfile.TemporaryDirectory() as tmpdir:
        storage = DiskStorage(storage_dir=tmpdir, n_cells=10, n_genes=5)

        # Create a small array and store it
        small_array = np.random.randn(10, 10)
        path = storage.store_array(small_array, "test_array")

        # Verify it was stored
        assert os.path.exists(path)
        assert "test_array" in storage.array_registry

        # Verify we can load it back
        loaded = storage.load_array("test_array", lazy=False)
        np.testing.assert_array_almost_equal(small_array, loaded)

        # Clean up
        storage.cleanup()


def test_disk_storage_total_storage_used():
    """Test that total_storage_used property works correctly."""
    from kompot.memory_utils import DiskStorage

    with tempfile.TemporaryDirectory() as tmpdir:
        storage = DiskStorage(storage_dir=tmpdir)

        # Initially should be zero
        total_h, total_bytes = storage.total_storage_used
        assert total_bytes == 0

        # Store some arrays
        arr1 = np.random.randn(5, 5)
        arr2 = np.random.randn(10, 10)

        storage.store_array(arr1, "arr1")
        storage.store_array(arr2, "arr2")

        # Total should match sum of array sizes
        total_h, total_bytes = storage.total_storage_used
        expected_total = arr1.nbytes + arr2.nbytes

        assert total_bytes == expected_total

        # Clean up
        storage.cleanup()


def test_suggest_alternative_dirs():
    """Test that alternative directory suggestion works."""
    from kompot.memory_utils import DiskStorage

    with tempfile.TemporaryDirectory() as tmpdir:
        storage = DiskStorage(storage_dir=tmpdir)

        alternatives = storage._suggest_alternative_dirs()

        # Should return a list
        assert isinstance(alternatives, list)

        # Each item should be a tuple with 3 elements
        for alt in alternatives:
            assert len(alt) == 3
            path, free_h, free_bytes = alt
            assert isinstance(path, str)
            assert isinstance(free_h, str)
            assert isinstance(free_bytes, int)
            assert free_bytes > 0

        # Should be sorted by free space descending
        if len(alternatives) > 1:
            for i in range(len(alternatives) - 1):
                assert alternatives[i][2] >= alternatives[i+1][2]

        # Clean up
        storage.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
