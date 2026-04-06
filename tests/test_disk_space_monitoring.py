"""Tests for disk space monitoring in DiskStorage."""

import numpy as np
import pytest
import tempfile
import os


def test_disk_space_utility_functions():
    """Test that disk space utility functions work."""
    from kompot.memory_utils import get_disk_space, estimate_disk_requirement

    # Test get_disk_space on /tmp
    total_h, total_bytes, used_h, used_bytes, free_h, free_bytes = get_disk_space(
        "/tmp"
    )

    # Should have valid values
    assert total_bytes > 0
    assert free_bytes > 0
    assert used_bytes >= 0
    assert total_bytes >= used_bytes + free_bytes  # Allow for some rounding

    # Human readable strings should have units
    assert any(unit in total_h for unit in ["B", "KB", "MB", "GB", "TB"])

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
        storage = DiskStorage(storage_dir=tmpdir, n_cells=10, n_genes=5)

        # storage_dir should be a unique subdirectory inside tmpdir
        assert storage.storage_dir.startswith(tmpdir)
        assert storage.storage_dir != tmpdir  # Should be a subdirectory, not the same
        assert os.path.exists(storage.storage_dir)
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
                expected_size_bytes=10**15,  # 1 PB - should exceed available space
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
                assert alternatives[i][2] >= alternatives[i + 1][2]

        # Clean up
        storage.cleanup()


def test_concurrent_storage_no_collision():
    """Test that multiple DiskStorage instances in same base dir don't collide."""
    from kompot.memory_utils import DiskStorage

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create two storage instances with same base directory
        storage1 = DiskStorage(storage_dir=tmpdir, n_cells=10, n_genes=5)
        storage2 = DiskStorage(storage_dir=tmpdir, n_cells=10, n_genes=5)

        # Should have different storage directories
        assert storage1.storage_dir != storage2.storage_dir
        assert storage1.storage_dir.startswith(tmpdir)
        assert storage2.storage_dir.startswith(tmpdir)

        # Store arrays with same key in both - should not collide
        arr1 = np.random.randn(5, 5)
        arr2 = np.random.randn(10, 10)

        path1 = storage1.store_array(arr1, "test_array")
        path2 = storage2.store_array(arr2, "test_array")

        # Paths should be different
        assert path1 != path2

        # Each should load their own array
        loaded1 = storage1.load_array("test_array", lazy=False)
        loaded2 = storage2.load_array("test_array", lazy=False)

        np.testing.assert_array_almost_equal(arr1, loaded1)
        np.testing.assert_array_almost_equal(arr2, loaded2)

        # Different shapes prove they didn't collide
        assert loaded1.shape == (5, 5)
        assert loaded2.shape == (10, 10)

        # Clean up
        storage1.cleanup()
        storage2.cleanup()


def test_tmpdir_env_variable_respected():
    """Test that TMPDIR environment variable is respected."""
    from kompot.memory_utils import DiskStorage

    # Create two different temporary directories
    with tempfile.TemporaryDirectory() as custom_tmpdir1:
        with tempfile.TemporaryDirectory() as custom_tmpdir2:
            # Ensure they are different
            assert custom_tmpdir1 != custom_tmpdir2

            # Save original state
            old_tmpdir = os.environ.get("TMPDIR")
            old_tempdir = tempfile.tempdir

            try:
                # Test with first custom TMPDIR
                os.environ["TMPDIR"] = custom_tmpdir1
                tempfile.tempdir = None  # Reset cache

                storage1 = DiskStorage(n_cells=10, n_genes=5)

                # Should be in custom_tmpdir1, not custom_tmpdir2
                assert storage1.storage_dir.startswith(custom_tmpdir1), (
                    f"Expected storage in {custom_tmpdir1}, got {storage1.storage_dir}"
                )
                assert not storage1.storage_dir.startswith(custom_tmpdir2), (
                    f"Should NOT be in {custom_tmpdir2}, but got {storage1.storage_dir}"
                )

                storage1.cleanup()

                # Now change TMPDIR to second directory
                os.environ["TMPDIR"] = custom_tmpdir2
                tempfile.tempdir = None  # Reset cache again

                storage2 = DiskStorage(n_cells=10, n_genes=5)

                # Should be in custom_tmpdir2, not custom_tmpdir1
                assert storage2.storage_dir.startswith(custom_tmpdir2), (
                    f"Expected storage in {custom_tmpdir2}, got {storage2.storage_dir}"
                )
                assert not storage2.storage_dir.startswith(custom_tmpdir1), (
                    f"Should NOT be in {custom_tmpdir1}, but got {storage2.storage_dir}"
                )

                storage2.cleanup()

            finally:
                # Restore original TMPDIR and tempdir cache
                if old_tmpdir is None:
                    os.environ.pop("TMPDIR", None)
                else:
                    os.environ["TMPDIR"] = old_tmpdir
                tempfile.tempdir = old_tempdir


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
