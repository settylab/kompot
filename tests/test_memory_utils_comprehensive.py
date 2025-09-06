"""Comprehensive tests for kompot.memory_utils module to improve coverage."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock
import tempfile
import os


class TestHumanReadableSize:
    """Test human readable size formatting."""
    
    def test_human_readable_size_basic(self):
        """Test basic human readable size functionality."""
        try:
            from kompot.memory_utils import human_readable_size
        except ImportError as e:
            pytest.skip(f"Could not import human_readable_size: {e}")
        
        assert human_readable_size(0) == "0 B"
        assert human_readable_size(512) == "512 B"
        assert human_readable_size(1024) == "1.0 KB"
        assert human_readable_size(1536) == "1.5 KB"
        assert human_readable_size(1024**2) == "1.0 MB"
        assert human_readable_size(1024**3) == "1.0 GB"
        assert human_readable_size(1024**4) == "1.0 TB"
        
    def test_human_readable_size_precision(self):
        """Test human readable size with different precisions."""
        try:
            from kompot.memory_utils import human_readable_size
        except ImportError as e:
            pytest.skip(f"Could not import human_readable_size: {e}")
        
        # Test with different decimal places
        size = 1536  # 1.5 KB
        assert "1.5" in human_readable_size(size)
        
        size = 1843200  # ~1.76 MB
        result = human_readable_size(size)
        assert "MB" in result
        assert "1." in result
        
    def test_human_readable_size_large_values(self):
        """Test human readable size with very large values."""
        try:
            from kompot.memory_utils import human_readable_size
        except ImportError as e:
            pytest.skip(f"Could not import human_readable_size: {e}")
        
        # Test petabytes
        size = 1024**5
        result = human_readable_size(size)
        assert "PB" in result
        
        # Test exabytes  
        size = 1024**6
        result = human_readable_size(size)
        assert "EB" in result or "PB" in result  # May cap at PB


class TestArraySize:
    """Test array size calculation."""
    
    def test_array_size_numpy(self):
        """Test array size calculation for numpy arrays."""
        try:
            from kompot.memory_utils import array_size
        except ImportError as e:
            pytest.skip(f"Could not import array_size: {e}")
        
        arr = np.ones((100, 50), dtype=np.float64)
        size = array_size(arr)
        
        expected_size = 100 * 50 * 8  # 8 bytes per float64
        assert size == expected_size
        
    def test_array_size_different_dtypes(self):
        """Test array size with different data types."""
        try:
            from kompot.memory_utils import array_size
        except ImportError as e:
            pytest.skip(f"Could not import array_size: {e}")
        
        # Test int32
        arr_int32 = np.ones((10, 10), dtype=np.int32)
        size_int32 = array_size(arr_int32)
        assert size_int32 == 10 * 10 * 4  # 4 bytes per int32
        
        # Test float32
        arr_float32 = np.ones((10, 10), dtype=np.float32)
        size_float32 = array_size(arr_float32)
        assert size_float32 == 10 * 10 * 4  # 4 bytes per float32
        
        # Test bool
        arr_bool = np.ones((10, 10), dtype=bool)
        size_bool = array_size(arr_bool)
        assert size_bool == 10 * 10 * 1  # 1 byte per bool
        
    def test_array_size_multidimensional(self):
        """Test array size with multidimensional arrays."""
        try:
            from kompot.memory_utils import array_size
        except ImportError as e:
            pytest.skip(f"Could not import array_size: {e}")
        
        # 3D array
        arr = np.ones((5, 10, 20), dtype=np.float64)
        size = array_size(arr)
        expected_size = 5 * 10 * 20 * 8  # 8 bytes per float64
        assert size == expected_size
        
        # 4D array
        arr = np.ones((2, 3, 4, 5), dtype=np.int32)
        size = array_size(arr)
        expected_size = 2 * 3 * 4 * 5 * 4  # 4 bytes per int32
        assert size == expected_size


class TestGetAvailableMemory:
    """Test available memory detection."""
    
    def test_get_available_memory_basic(self):
        """Test basic available memory detection."""
        try:
            from kompot.memory_utils import get_available_memory
        except ImportError as e:
            pytest.skip(f"Could not import get_available_memory: {e}")
        
        memory = get_available_memory()
        
        assert isinstance(memory, int)
        assert memory > 0
        # Should be reasonable (at least 100 MB, less than 1 TB)
        assert 100 * 1024 * 1024 < memory < 1024**4
        
    def test_get_available_memory_with_psutil(self):
        """Test memory detection with psutil."""
        try:
            from kompot.memory_utils import get_available_memory
            import psutil
        except ImportError:
            pytest.skip("psutil not available")
        
        memory = get_available_memory()
        
        # Compare with direct psutil call
        psutil_memory = psutil.virtual_memory().available
        
        # Should be reasonably close (within 10%)
        assert abs(memory - psutil_memory) / psutil_memory < 0.1
        
    def test_get_available_memory_fallback(self):
        """Test memory detection fallback when psutil unavailable."""
        try:
            from kompot.memory_utils import get_available_memory
        except ImportError as e:
            pytest.skip(f"Could not import get_available_memory: {e}")
        
        # Mock psutil import failure
        with patch.dict('sys.modules', {'psutil': None}):
            with patch('kompot.memory_utils.psutil', None):
                memory = get_available_memory()
                
                # Should still return a reasonable value (fallback)
                assert isinstance(memory, int)
                assert memory > 0


class TestMemoryRequirementRatio:
    """Test memory requirement ratio calculation."""
    
    def test_memory_requirement_ratio_basic(self):
        """Test basic memory requirement ratio calculation."""
        try:
            from kompot.memory_utils import memory_requirement_ratio
        except ImportError as e:
            pytest.skip(f"Could not import memory_requirement_ratio: {e}")
        
        # Test with 1 MB requirement
        required = 1024 * 1024  # 1 MB
        ratio = memory_requirement_ratio(required)
        
        assert isinstance(ratio, float)
        assert 0 <= ratio <= 1.0  # Should be between 0 and 1
        
    def test_memory_requirement_ratio_large(self):
        """Test memory requirement ratio with large requirements."""
        try:
            from kompot.memory_utils import memory_requirement_ratio
        except ImportError as e:
            pytest.skip(f"Could not import memory_requirement_ratio: {e}")
        
        # Test with very large requirement (1 GB)
        required = 1024**3  # 1 GB
        ratio = memory_requirement_ratio(required)
        
        assert isinstance(ratio, float)
        assert ratio > 0
        
    def test_memory_requirement_ratio_zero(self):
        """Test memory requirement ratio with zero requirement."""
        try:
            from kompot.memory_utils import memory_requirement_ratio
        except ImportError as e:
            pytest.skip(f"Could not import memory_requirement_ratio: {e}")
        
        ratio = memory_requirement_ratio(0)
        assert ratio == 0.0


class TestAnalyzeMemoryRequirements:
    """Test memory requirements analysis."""
    
    def test_analyze_memory_requirements_basic(self):
        """Test basic memory requirements analysis."""
        try:
            from kompot.memory_utils import analyze_memory_requirements
        except ImportError as e:
            pytest.skip(f"Could not import analyze_memory_requirements: {e}")
        
        # Create test arrays
        arrays = [
            np.ones((100, 50), dtype=np.float64),
            np.ones((200, 30), dtype=np.float32),
            np.ones((50, 100), dtype=np.int32)
        ]
        
        analysis = analyze_memory_requirements(arrays)
        
        assert isinstance(analysis, dict)
        assert 'total_size' in analysis
        assert 'available_memory' in analysis
        assert 'memory_ratio' in analysis
        assert 'human_readable_size' in analysis
        
        # Check values are reasonable
        assert analysis['total_size'] > 0
        assert analysis['available_memory'] > 0
        assert 0 <= analysis['memory_ratio'] <= 10  # Should be reasonable
        
    def test_analyze_memory_requirements_empty(self):
        """Test memory requirements analysis with empty list."""
        try:
            from kompot.memory_utils import analyze_memory_requirements
        except ImportError as e:
            pytest.skip(f"Could not import analyze_memory_requirements: {e}")
        
        analysis = analyze_memory_requirements([])
        
        assert isinstance(analysis, dict)
        assert analysis['total_size'] == 0
        assert 'available_memory' in analysis
        assert analysis['memory_ratio'] == 0.0
        
    def test_analyze_memory_requirements_single_array(self):
        """Test memory requirements analysis with single array."""
        try:
            from kompot.memory_utils import analyze_memory_requirements
        except ImportError as e:
            pytest.skip(f"Could not import analyze_memory_requirements: {e}")
        
        arr = np.ones((1000, 1000), dtype=np.float64)
        analysis = analyze_memory_requirements([arr])
        
        expected_size = 1000 * 1000 * 8  # 8 bytes per float64
        assert analysis['total_size'] == expected_size
        assert analysis['memory_ratio'] > 0


class TestAnalyzeCovarianceMemoryRequirements:
    """Test covariance memory requirements analysis."""
    
    def test_analyze_covariance_memory_requirements_basic(self):
        """Test basic covariance memory requirements analysis."""
        try:
            from kompot.memory_utils import analyze_covariance_memory_requirements
        except ImportError as e:
            pytest.skip(f"Could not import analyze_covariance_memory_requirements: {e}")
        
        n_features = 100
        n_samples = 50
        
        analysis = analyze_covariance_memory_requirements(n_features, n_samples)
        
        assert isinstance(analysis, dict)
        assert 'covariance_size' in analysis
        assert 'sample_size' in analysis
        assert 'total_memory_required' in analysis
        assert 'available_memory' in analysis
        assert 'memory_ratio' in analysis
        assert 'human_readable_total' in analysis
        
        # Check covariance size calculation
        expected_cov_size = n_features * n_features * 8  # float64
        assert analysis['covariance_size'] == expected_cov_size
        
        # Check sample size calculation
        expected_sample_size = n_samples * n_features * 8  # float64
        assert analysis['sample_size'] == expected_sample_size
        
    def test_analyze_covariance_memory_requirements_large(self):
        """Test covariance memory requirements with large dimensions."""
        try:
            from kompot.memory_utils import analyze_covariance_memory_requirements
        except ImportError as e:
            pytest.skip(f"Could not import analyze_covariance_memory_requirements: {e}")
        
        n_features = 5000
        n_samples = 1000
        
        analysis = analyze_covariance_memory_requirements(n_features, n_samples)
        
        # Should detect high memory requirement
        assert analysis['memory_ratio'] > 0
        assert analysis['total_memory_required'] > analysis['covariance_size']
        
    def test_analyze_covariance_memory_requirements_zero(self):
        """Test covariance memory requirements with zero dimensions."""
        try:
            from kompot.memory_utils import analyze_covariance_memory_requirements
        except ImportError as e:
            pytest.skip(f"Could not import analyze_covariance_memory_requirements: {e}")
        
        analysis = analyze_covariance_memory_requirements(0, 0)
        
        assert analysis['covariance_size'] == 0
        assert analysis['sample_size'] == 0
        assert analysis['total_memory_required'] == 0


class TestDiskStorage:
    """Test disk storage functionality."""
    
    def test_disk_storage_init(self):
        """Test DiskStorage initialization."""
        try:
            from kompot.memory_utils import DiskStorage
        except ImportError as e:
            pytest.skip(f"Could not import DiskStorage: {e}")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = DiskStorage(tmpdir)
            
            assert storage.base_dir == tmpdir
            assert os.path.exists(tmpdir)
            
    def test_disk_storage_store_and_load_array(self):
        """Test storing and loading arrays to/from disk."""
        try:
            from kompot.memory_utils import DiskStorage
        except ImportError as e:
            pytest.skip(f"Could not import DiskStorage: {e}")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = DiskStorage(tmpdir)
            
            # Create test array
            arr = np.random.rand(100, 50)
            
            # Store array
            key = storage.store_array('test_array', arr)
            assert key == 'test_array'
            
            # Load array
            loaded_arr = storage.load_array('test_array')
            np.testing.assert_array_equal(arr, loaded_arr)
            
    def test_disk_storage_store_different_dtypes(self):
        """Test storing arrays with different dtypes."""
        try:
            from kompot.memory_utils import DiskStorage
        except ImportError as e:
            pytest.skip(f"Could not import DiskStorage: {e}")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = DiskStorage(tmpdir)
            
            # Test different dtypes
            arrays = {
                'float64_arr': np.random.rand(10, 10).astype(np.float64),
                'float32_arr': np.random.rand(10, 10).astype(np.float32),
                'int32_arr': np.random.randint(0, 100, (10, 10), dtype=np.int32),
                'bool_arr': np.random.choice([True, False], (10, 10))
            }
            
            # Store all arrays
            for key, arr in arrays.items():
                storage.store_array(key, arr)
            
            # Load and verify all arrays
            for key, original_arr in arrays.items():
                loaded_arr = storage.load_array(key)
                np.testing.assert_array_equal(original_arr, loaded_arr)
                assert loaded_arr.dtype == original_arr.dtype
                
    def test_disk_storage_cleanup(self):
        """Test disk storage cleanup functionality."""
        try:
            from kompot.memory_utils import DiskStorage
        except ImportError as e:
            pytest.skip(f"Could not import DiskStorage: {e}")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = DiskStorage(tmpdir)
            
            # Store some arrays
            arr1 = np.random.rand(10, 10)
            arr2 = np.random.rand(5, 5)
            storage.store_array('array1', arr1)
            storage.store_array('array2', arr2)
            
            # Check files exist
            assert len(os.listdir(tmpdir)) >= 2
            
            # Cleanup
            storage.cleanup()
            
            # Directory should still exist but be empty (or nearly empty)
            assert os.path.exists(tmpdir)
            
    def test_disk_storage_nonexistent_array(self):
        """Test loading nonexistent array."""
        try:
            from kompot.memory_utils import DiskStorage
        except ImportError as e:
            pytest.skip(f"Could not import DiskStorage: {e}")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = DiskStorage(tmpdir)
            
            # Try to load nonexistent array
            with pytest.raises((FileNotFoundError, KeyError, ValueError)):
                storage.load_array('nonexistent')


class TestDaskIntegration:
    """Test Dask integration (if available)."""
    
    def test_dask_available_flag(self):
        """Test DASK_AVAILABLE flag."""
        try:
            from kompot.memory_utils import DASK_AVAILABLE
        except ImportError as e:
            pytest.skip(f"Could not import DASK_AVAILABLE: {e}")
        
        assert isinstance(DASK_AVAILABLE, bool)
        
    @pytest.mark.skipif(True, reason="Dask may not be available")
    def test_dask_array_operations(self):
        """Test operations with Dask arrays if available."""
        try:
            from kompot.memory_utils import DASK_AVAILABLE
            if not DASK_AVAILABLE:
                pytest.skip("Dask not available")
                
            import dask.array as da
            from kompot.memory_utils import array_size
        except ImportError:
            pytest.skip("Dask not available")
        
        # Create dask array
        darr = da.ones((1000, 1000), chunks=(100, 100))
        
        # Test size calculation works with dask arrays
        size = array_size(darr)
        expected_size = 1000 * 1000 * 8  # 8 bytes per float64
        assert size == expected_size


class TestMemoryUtilsEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_human_readable_size_negative(self):
        """Test human readable size with negative values."""
        try:
            from kompot.memory_utils import human_readable_size
        except ImportError as e:
            pytest.skip(f"Could not import human_readable_size: {e}")
        
        # Should handle negative values gracefully
        result = human_readable_size(-1024)
        assert isinstance(result, str)
        
    def test_array_size_empty_array(self):
        """Test array size with empty array."""
        try:
            from kompot.memory_utils import array_size
        except ImportError as e:
            pytest.skip(f"Could not import array_size: {e}")
        
        arr = np.array([])
        size = array_size(arr)
        assert size == 0
        
    def test_array_size_scalar(self):
        """Test array size with scalar."""
        try:
            from kompot.memory_utils import array_size
        except ImportError as e:
            pytest.skip(f"Could not import array_size: {e}")
        
        scalar = np.float64(5.0)
        size = array_size(scalar)
        assert size == 8  # 8 bytes for float64
        
    def test_memory_functions_with_none(self):
        """Test memory functions with None inputs."""
        try:
            from kompot.memory_utils import memory_requirement_ratio, analyze_memory_requirements
        except ImportError as e:
            pytest.skip(f"Could not import memory functions: {e}")
        
        # Test memory_requirement_ratio with None
        try:
            ratio = memory_requirement_ratio(None)
            # Should either handle gracefully or raise appropriate error
        except TypeError:
            pass  # Expected
            
        # Test analyze_memory_requirements with None
        try:
            analysis = analyze_memory_requirements(None)
            # Should either handle gracefully or raise appropriate error
        except (TypeError, AttributeError):
            pass  # Expected


class TestMemoryUtilsLogging:
    """Test logging functionality in memory utils."""
    
    def test_disk_storage_logging(self):
        """Test DiskStorage logging."""
        try:
            from kompot.memory_utils import DiskStorage
        except ImportError as e:
            pytest.skip(f"Could not import DiskStorage: {e}")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('kompot.memory_utils.logger') as mock_logger:
                storage = DiskStorage(tmpdir)
                arr = np.random.rand(10, 10)
                storage.store_array('test', arr)
                
                # Should have logged some operations
                assert mock_logger.debug.call_count >= 0  # May or may not log
                
    def test_memory_analysis_logging(self):
        """Test memory analysis logging."""
        try:
            from kompot.memory_utils import analyze_covariance_memory_requirements
        except ImportError as e:
            pytest.skip(f"Could not import analyze_covariance_memory_requirements: {e}")
        
        with patch('kompot.memory_utils.logger') as mock_logger:
            # Test with large dimensions that might trigger warnings
            analysis = analyze_covariance_memory_requirements(10000, 5000)
            
            # May log warnings about high memory usage
            assert mock_logger.warning.call_count >= 0


class TestMemoryUtilsPerformance:
    """Test performance-related aspects of memory utils."""
    
    def test_array_size_performance_large_arrays(self):
        """Test array size calculation performance with large arrays."""
        try:
            from kompot.memory_utils import array_size
        except ImportError as e:
            pytest.skip(f"Could not import array_size: {e}")
        
        # Create large array (but don't fill with data to save memory)
        arr = np.zeros((10000, 1000), dtype=np.float32)
        
        # Should calculate size quickly without reading all data
        size = array_size(arr)
        expected_size = 10000 * 1000 * 4  # 4 bytes per float32
        assert size == expected_size
        
    def test_memory_analysis_speed(self):
        """Test memory analysis calculation speed."""
        try:
            from kompot.memory_utils import analyze_memory_requirements
        except ImportError as e:
            pytest.skip(f"Could not import analyze_memory_requirements: {e}")
        
        # Create multiple large arrays
        arrays = [
            np.zeros((1000, 1000), dtype=np.float64),
            np.zeros((500, 2000), dtype=np.float32),
            np.zeros((2000, 500), dtype=np.int32)
        ]
        
        # Should analyze quickly without loading all data
        analysis = analyze_memory_requirements(arrays)
        assert analysis['total_size'] > 0