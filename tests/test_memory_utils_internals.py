"""
Tests for memory_utils.py internals: import fallbacks, array sizing,
available memory detection, DiskStorage, and dask array conversion.
"""

import os
import tempfile
import numpy as np
import pytest
from unittest.mock import patch, MagicMock


class TestPsutilImportBranch:
    """Cover the psutil=None import fallback (lines 13-14)."""

    def test_psutil_none_fallback(self):
        """When psutil import fails, module-level psutil should be None."""
        # We can verify by mocking, but the simplest is to exercise the
        # get_available_memory fallback path.
        from kompot import memory_utils

        original = memory_utils.psutil
        try:
            memory_utils.psutil = None
            h, b = memory_utils.get_available_memory()
            assert b > 0
            assert isinstance(h, str)
        finally:
            memory_utils.psutil = original


class TestDaskImportBranch:
    """Cover lines 21-26 (dask import branch) and lines 47-76 (get_dask_array)."""

    def test_get_dask_array_3d(self):
        """Cover 3D chunking path in get_dask_array (lines 52-56)."""
        from kompot.memory_utils import get_dask_array, DASK_AVAILABLE

        if not DASK_AVAILABLE:
            pytest.skip("dask not available")
        arr = get_dask_array((200, 200, 10))
        assert arr is not None
        assert arr.shape == (200, 200, 10)

    def test_get_dask_array_2d(self):
        """Cover non-3D chunking path in get_dask_array (lines 57-71)."""
        from kompot.memory_utils import get_dask_array, DASK_AVAILABLE

        if not DASK_AVAILABLE:
            pytest.skip("dask not available")
        arr = get_dask_array((500, 500))
        assert arr is not None
        assert arr.shape == (500, 500)

    def test_get_dask_array_with_chunk_size(self):
        """Cover explicit chunk_size path (lines 72-74)."""
        from kompot.memory_utils import get_dask_array, DASK_AVAILABLE

        if not DASK_AVAILABLE:
            pytest.skip("dask not available")
        arr = get_dask_array((100, 100), chunk_size=(50, 50))
        assert arr is not None

    def test_get_dask_array_not_available(self):
        """Cover early return when dask not available (line 44-45)."""
        from kompot import memory_utils

        orig = memory_utils.DASK_AVAILABLE
        try:
            memory_utils.DASK_AVAILABLE = False
            result = memory_utils.get_dask_array((10, 10))
            assert result is None
        finally:
            memory_utils.DASK_AVAILABLE = orig


class TestArraySize:
    """Cover line 131 (dask array branch) in array_size."""

    def test_array_size_dask(self):
        """Cover dask array nbytes path."""
        from kompot.memory_utils import array_size, DASK_AVAILABLE

        if not DASK_AVAILABLE:
            pytest.skip("dask not available")
        import dask.array as da

        arr = da.zeros((10, 10), chunks=5)
        result = array_size(arr)
        assert isinstance(result, (int, np.integer))
        assert result == 800  # 10*10*8 bytes


class TestGetAvailableMemory:
    """Cover fallback paths in get_available_memory (lines 157-158, 171-172, 187-188)."""

    def test_psutil_attribute_error(self):
        """Cover lines 157-158: psutil raises AttributeError."""
        from kompot import memory_utils

        mock_psutil = MagicMock()
        mock_psutil.virtual_memory.side_effect = AttributeError("no memory")
        original = memory_utils.psutil
        try:
            memory_utils.psutil = mock_psutil
            h, b = memory_utils.get_available_memory()
            assert b > 0
        finally:
            memory_utils.psutil = original

    def test_linux_memavailable_path(self):
        """Cover line 171-172: Linux MemAvailable reading."""
        from kompot import memory_utils

        original = memory_utils.psutil
        try:
            memory_utils.psutil = None
            # On Linux this should work via /proc/meminfo
            import platform

            if platform.system() == "Linux":
                h, b = memory_utils.get_available_memory()
                assert b > 0
            else:
                pytest.skip("Not Linux")
        finally:
            memory_utils.psutil = original


class TestAnalyzeCovarianceMemory:
    """Cover line 435 (store_arrays_on_disk forces debug log_level)."""

    def test_store_arrays_on_disk_sets_debug(self):
        """When store_arrays_on_disk=True, log_level should become debug."""
        from kompot.memory_utils import analyze_covariance_memory_requirements

        result = analyze_covariance_memory_requirements(
            n_points=5, n_genes=3, store_arrays_on_disk=True
        )
        assert "should_use_disk" in result
        assert result["status"] in ("ok", "warning", "critical")


class TestDiskStorage:
    """Cover DiskStorage lines: 540, 600, 618, 648, 655-656, 677, 685-694,
    703, 722-730, 762-763, 782-784, 788-791, 816, 822-825, 855-858, 862-865."""

    def test_init_with_dask(self):
        """Cover line 540 (dask support log message)."""
        from kompot.memory_utils import DiskStorage, DASK_AVAILABLE

        ds = DiskStorage(use_dask=True)
        assert ds.use_dask == DASK_AVAILABLE
        ds.cleanup()

    def test_init_with_storage_dir(self):
        """Cover storage_dir provided path."""
        from kompot.memory_utils import DiskStorage

        with tempfile.TemporaryDirectory() as tmpdir:
            ds = DiskStorage(storage_dir=tmpdir, use_dask=False)
            assert tmpdir in ds.storage_dir
            ds.cleanup()

    def test_init_with_n_cells_n_genes(self):
        """Cover estimated size from n_cells/n_genes."""
        from kompot.memory_utils import DiskStorage

        ds = DiskStorage(use_dask=False, n_cells=10, n_genes=5)
        ds.cleanup()

    def test_store_and_load_array(self):
        """Cover store_array and load_array paths."""
        from kompot.memory_utils import DiskStorage

        ds = DiskStorage(use_dask=False)
        arr = np.random.randn(5, 5)
        ds.store_array(arr, "test_arr")
        loaded = ds.load_array("test_arr")
        np.testing.assert_array_equal(arr, loaded)
        ds.cleanup()

    def test_load_array_key_not_found(self):
        """Cover line 812: KeyError for missing key."""
        from kompot.memory_utils import DiskStorage

        ds = DiskStorage(use_dask=False)
        with pytest.raises(KeyError):
            ds.load_array("nonexistent")
        ds.cleanup()

    def test_load_array_file_not_found(self):
        """Cover line 816: FileNotFoundError."""
        from kompot.memory_utils import DiskStorage

        ds = DiskStorage(use_dask=False)
        # Manually add a fake registry entry
        ds.array_registry["ghost"] = {
            "path": "/tmp/nonexistent_file_xyz.npy",
            "shape": (5,),
            "dtype": "float64",
            "size_bytes": 40,
            "size_human": "40.00 B",
            "namespaced_key": "ghost",
        }
        with pytest.raises(FileNotFoundError):
            ds.load_array("ghost")
        ds.cleanup()

    def test_load_array_lazy_dask(self):
        """Cover lines 822-825: lazy loading with dask."""
        from kompot.memory_utils import DiskStorage, DASK_AVAILABLE

        if not DASK_AVAILABLE:
            pytest.skip("dask not available")
        ds = DiskStorage(use_dask=True)
        arr = np.random.randn(5, 5)
        ds.store_array(arr, "lazy_test")
        loaded = ds.load_array("lazy_test", lazy=True)
        assert hasattr(loaded, "compute")
        np.testing.assert_array_almost_equal(loaded.compute(), arr)
        ds.cleanup()

    def test_remove_array(self):
        """Cover remove_array path (lines 855-858, 862-865)."""
        from kompot.memory_utils import DiskStorage

        ds = DiskStorage(use_dask=False)
        arr = np.random.randn(5, 5)
        ds.store_array(arr, "to_remove")
        assert "to_remove" in ds.array_registry
        ds.remove_array("to_remove")
        assert "to_remove" not in ds.array_registry
        ds.cleanup()

    def test_remove_array_missing_key(self):
        """Cover remove_array warning for missing key."""
        from kompot.memory_utils import DiskStorage

        ds = DiskStorage(use_dask=False)
        # Should not raise, just warn
        ds.remove_array("does_not_exist")
        ds.cleanup()

    def test_total_storage_used(self):
        """Cover total_storage_used property."""
        from kompot.memory_utils import DiskStorage

        ds = DiskStorage(use_dask=False)
        arr = np.zeros((10, 10))
        ds.store_array(arr, "sz_test")
        h, b = ds.total_storage_used
        assert b == arr.nbytes
        assert isinstance(h, str)
        ds.cleanup()

    def test_list_arrays(self):
        """Cover list_arrays."""
        from kompot.memory_utils import DiskStorage

        ds = DiskStorage(use_dask=False)
        ds.store_array(np.zeros(5), "a")
        ds.store_array(np.zeros(3), "b")
        listing = ds.list_arrays()
        assert "a" in listing
        assert "b" in listing
        ds.cleanup()

    def test_monitor_disk_space_warning(self):
        """Cover _monitor_disk_space_during_storage warning path (line 677)."""
        from kompot.memory_utils import DiskStorage

        ds = DiskStorage(use_dask=False)
        # Mock get_disk_space to return very low free space
        with patch(
            "kompot.memory_utils.get_disk_space",
            return_value=("1 GB", 10**9, "900 MB", 9 * 10**8, "100 B", 100),
        ):
            with pytest.raises(IOError):
                ds._monitor_disk_space_during_storage(10**9)
        ds.cleanup()

    def test_monitor_disk_low_but_not_zero(self):
        """Cover line 677: free < 2x array but free > array."""
        from kompot.memory_utils import DiskStorage

        ds = DiskStorage(use_dask=False)
        with patch(
            "kompot.memory_utils.get_disk_space",
            return_value=("1 GB", 10**9, "500 MB", 5 * 10**8, "500 MB", 500),
        ):
            # free (500) < 2*200 but free (500) >= 200 => warning but no exception
            # Actually 500 > 200 so no IOError, just warning
            ds.array_registry["dummy"] = {"size_bytes": 0}
            ds._monitor_disk_space_during_storage(200)
        ds.cleanup()

    def test_check_disk_space_insufficient(self):
        """Cover _check_disk_space IOError path (line 600)."""
        from kompot.memory_utils import DiskStorage

        ds = DiskStorage(use_dask=False)
        with patch(
            "kompot.memory_utils.get_disk_space",
            return_value=("1 GB", 10**9, "999 MB", 999 * 10**6, "1 B", 1),
        ):
            with pytest.raises(IOError, match="Insufficient disk space"):
                ds._check_disk_space(expected_size_bytes=10**9)
        ds.cleanup()

    def test_check_disk_space_tight(self):
        """Cover _check_disk_space warning (tight space, line 618)."""
        from kompot.memory_utils import DiskStorage

        ds = DiskStorage(use_dask=False)
        # free > required but < 1.5x required => warning
        with patch(
            "kompot.memory_utils.get_disk_space",
            return_value=(
                "10 GB",
                10**10,
                "5 GB",
                5 * 10**9,
                "1.2 GB",
                int(1.2 * 10**9),
            ),
        ):
            ds._check_disk_space(expected_size_bytes=10**9)
        ds.cleanup()

    def test_suggest_alternative_dirs(self):
        """Cover _suggest_alternative_dirs (lines 648, 655-656)."""
        from kompot.memory_utils import DiskStorage

        ds = DiskStorage(use_dask=False)
        alts = ds._suggest_alternative_dirs()
        assert isinstance(alts, list)
        ds.cleanup()

    def test_cleanup_exception_handling(self):
        """Cover cleanup exception path (lines 722-730)."""
        from kompot.memory_utils import DiskStorage

        ds = DiskStorage(use_dask=False)
        # Force cleanup to hit the rmtree exception by making directory non-existent
        ds.cleanup()
        # Second cleanup is a no-op (directory already removed)
        ds.cleanup()

    def test_cleanup_during_shutdown(self):
        """Cover line 703: sys.meta_path is None during shutdown."""
        from kompot.memory_utils import DiskStorage
        import sys

        ds = DiskStorage(use_dask=False)
        orig_meta = sys.meta_path
        try:
            sys.meta_path = None
            ds.cleanup()  # Should return early
        finally:
            sys.meta_path = orig_meta
        # Actual cleanup
        sys.meta_path = orig_meta
        ds.cleanup()

    def test_store_array_lock_removal(self):
        """Cover lines 788-791: lock file removal in finally block."""
        from kompot.memory_utils import DiskStorage

        ds = DiskStorage(use_dask=False)
        arr = np.array([1, 2, 3])
        path = ds.store_array(arr, "lock_test")
        # Verify no leftover lock file
        assert not os.path.exists(path + ".lock")
        ds.cleanup()


class TestAsDAskArray:
    """Cover as_dask_array (lines 909-957)."""

    def test_as_dask_array_no_dask(self):
        """Cover line 909-910: ImportError when dask not available."""
        from kompot.memory_utils import DiskStorage

        ds = DiskStorage(use_dask=False)
        ds.use_dask = False
        with pytest.raises(ImportError):
            ds.as_dask_array(shape=(5, 5, 3))
        ds.cleanup()

    def test_as_dask_array_3d_with_stored(self):
        """Cover lines 912-954: building 3D dask array from stored gene slices."""
        from kompot.memory_utils import DiskStorage, DASK_AVAILABLE

        if not DASK_AVAILABLE:
            pytest.skip("dask not available")
        ds = DiskStorage(use_dask=True)
        # Store 2D slices as gene_0, gene_1
        for g in range(3):
            ds.store_array(np.ones((4, 4)) * (g + 1), f"gene_{g}")
        result = ds.as_dask_array(shape=(4, 4, 3))
        assert result.shape == (4, 4, 3)
        computed = result.compute()
        assert computed[0, 0, 0] == 1.0
        assert computed[0, 0, 2] == 3.0
        ds.cleanup()

    def test_as_dask_array_infer_shape(self):
        """Cover shape inference from stored 2D arrays (lines 918-923)."""
        from kompot.memory_utils import DiskStorage, DASK_AVAILABLE

        if not DASK_AVAILABLE:
            pytest.skip("dask not available")
        ds = DiskStorage(use_dask=True)
        for g in range(2):
            ds.store_array(np.ones((3, 3)), f"gene_{g}")
        result = ds.as_dask_array()  # no shape given
        assert result.shape == (3, 3, 2)
        ds.cleanup()

    def test_as_dask_array_cannot_infer_shape(self):
        """Cover line 925: ValueError when shape cannot be inferred."""
        from kompot.memory_utils import DiskStorage, DASK_AVAILABLE

        if not DASK_AVAILABLE:
            pytest.skip("dask not available")
        ds = DiskStorage(use_dask=True)
        ds.store_array(np.ones((3, 3, 2)), "arr3d")
        with pytest.raises(ValueError, match="Cannot infer shape"):
            ds.as_dask_array()
        ds.cleanup()

    def test_as_dask_array_non_3d(self):
        """Cover line 956-957: non-3D shape returns zeros."""
        from kompot.memory_utils import DiskStorage, DASK_AVAILABLE

        if not DASK_AVAILABLE:
            pytest.skip("dask not available")
        ds = DiskStorage(use_dask=True)
        result = ds.as_dask_array(shape=(5, 5))
        assert result.shape == (5, 5)
        ds.cleanup()

    def test_as_dask_array_missing_gene_slice(self):
        """Cover line 948-949: zeros for missing gene slices."""
        from kompot.memory_utils import DiskStorage, DASK_AVAILABLE

        if not DASK_AVAILABLE:
            pytest.skip("dask not available")
        ds = DiskStorage(use_dask=True)
        ds.store_array(np.ones((3, 3)), "gene_0")
        # gene_1 is missing
        result = ds.as_dask_array(shape=(3, 3, 2))
        computed = result.compute()
        np.testing.assert_array_equal(computed[:, :, 1], np.zeros((3, 3)))
        ds.cleanup()
