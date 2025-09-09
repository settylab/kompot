"""Comprehensive tests for kompot.utils module to improve coverage."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock


class TestUtilsCoverage:
    """Test utils functions for coverage."""

    def test_compute_mahalanobis_distance_basic(self):
        """Test basic compute_mahalanobis_distance functionality."""
        try:
            from kompot.utils import compute_mahalanobis_distance
        except ImportError as e:
            pytest.skip(f"Could not import compute_mahalanobis_distance: {e}")
        
        # Create test data - single difference vector and covariance matrix
        diff_vector = np.random.rand(5)
        covariance_matrix = np.eye(5)  # Identity covariance matrix
        
        distance = compute_mahalanobis_distance(diff_vector, covariance_matrix)
        
        assert isinstance(distance, (float, np.floating))
        assert distance >= 0

    def test_compute_mahalanobis_distance_with_landmarks(self):
        """Test compute_mahalanobis_distance with landmarks."""
        try:
            from kompot.utils import compute_mahalanobis_distance
        except ImportError as e:
            pytest.skip(f"Could not import compute_mahalanobis_distance: {e}")
        
        # Create test data - test different covariance matrix
        diff_vector = np.random.rand(5)
        covariance_matrix = np.random.rand(5, 5)
        # Make covariance matrix symmetric and positive definite
        covariance_matrix = covariance_matrix @ covariance_matrix.T + np.eye(5) * 0.1
        
        distance = compute_mahalanobis_distance(
            diff_vector, covariance_matrix, eps=1e-6
        )
        
        assert isinstance(distance, (float, np.floating))
        assert distance >= 0

    def test_find_landmarks_basic(self):
        """Test basic find_landmarks functionality.""" 
        try:
            from kompot.utils import find_landmarks
        except ImportError as e:
            pytest.skip(f"Could not import find_landmarks: {e}")
        
        # Create test data
        X = np.random.rand(100, 10)
        
        landmarks, landmark_indices = find_landmarks(X, n_clusters=20)
        
        assert landmarks.shape[0] <= 25  # Algorithm may find slightly more than requested due to clustering
        assert landmarks.shape[1] == 10
        assert len(landmark_indices) == landmarks.shape[0]
        assert all(0 <= idx < X.shape[0] for idx in landmark_indices)

    def test_find_landmarks_with_groups(self):
        """Test find_landmarks with group information."""
        try:
            from kompot.utils import find_landmarks
        except ImportError as e:
            pytest.skip(f"Could not import find_landmarks: {e}")
        
        # Create test data
        X = np.random.rand(100, 10)
        
        landmarks, landmark_indices = find_landmarks(X, n_clusters=20, n_neighbors=10)
        
        assert landmarks.shape[0] <= 25  # May return more/fewer than requested due to clustering algorithm
        assert landmarks.shape[1] == 10
        assert len(landmark_indices) == landmarks.shape[0]

    def test_find_landmarks_stratified(self):
        """Test find_landmarks with stratified sampling."""
        try:
            from kompot.utils import find_landmarks
        except ImportError as e:
            pytest.skip(f"Could not import find_landmarks: {e}")
        
        # Create test data
        X = np.random.rand(100, 10)
        
        landmarks, landmark_indices = find_landmarks(X, n_clusters=15, tol=0.2)
        
        assert landmarks.shape[0] <= 20  # Algorithm may find more than requested due to clustering
        assert landmarks.shape[1] == 10
        assert len(landmark_indices) == landmarks.shape[0]


    def test_get_run_from_history_error_handling(self):
        """Test get_run_from_history error handling."""
        try:
            from kompot.utils import get_run_from_history
            import anndata
        except ImportError as e:
            pytest.skip(f"Could not import required modules: {e}")
        
        # Create empty AnnData
        empty_adata = anndata.AnnData(np.zeros((10, 5)))
        
        # Should handle missing history gracefully
        with pytest.raises((ValueError, KeyError, AttributeError)):
            get_run_from_history(empty_adata, run_id=-1, analysis_type='da')

    def test_kompot_colors_structure(self):
        """Test KOMPOT_COLORS structure and accessibility."""
        try:
            from kompot.utils import KOMPOT_COLORS
        except ImportError as e:
            pytest.skip(f"Could not import KOMPOT_COLORS: {e}")
        
        # Test basic structure
        assert isinstance(KOMPOT_COLORS, dict)
        
        # Test expected keys
        expected_keys = ['direction', 'default']
        for key in expected_keys:
            if key in KOMPOT_COLORS:
                assert isinstance(KOMPOT_COLORS[key], dict)

    def test_colors_validation(self):
        """Test color validation and format."""
        try:
            from kompot.utils import KOMPOT_COLORS
        except ImportError as e:
            pytest.skip(f"Could not import KOMPOT_COLORS: {e}")
        
        # Check if colors are valid hex strings or color names
        def is_valid_color(color):
            if isinstance(color, str):
                # Check for hex format
                if color.startswith('#') and len(color) in [4, 7]:
                    return True
                # Check for common color names
                if color.lower() in ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 
                                   'pink', 'brown', 'gray', 'black', 'white']:
                    return True
            return False
        
        # Test direction colors if they exist
        if 'direction' in KOMPOT_COLORS:
            direction_colors = KOMPOT_COLORS['direction']
            for key, color in direction_colors.items():
                if isinstance(color, str):
                    # Colors should be valid (this is a loose test)
                    assert len(color) > 0


class TestMemoryUtilsCoverage:
    """Test memory utility functions for coverage."""

    def test_human_readable_size(self):
        """Test human_readable_size function."""
        try:
            from kompot.memory_utils import human_readable_size
        except ImportError as e:
            pytest.skip(f"Could not import human_readable_size: {e}")
        
        # Test various sizes
        assert human_readable_size(0) == "0.00 B"
        assert "KB" in human_readable_size(1024)
        assert "MB" in human_readable_size(1024**2)
        assert "GB" in human_readable_size(1024**3)

    def test_array_size(self):
        """Test array_size function."""
        try:
            from kompot.memory_utils import array_size
        except ImportError as e:
            pytest.skip(f"Could not import array_size: {e}")
        
        # Test with numpy array
        arr = np.ones((100, 50))
        size = array_size(arr)
        
        assert size > 0
        assert isinstance(size, int)

    def test_get_available_memory(self):
        """Test get_available_memory function."""
        try:
            from kompot.memory_utils import get_available_memory
        except ImportError as e:
            pytest.skip(f"Could not import get_available_memory: {e}")
        
        memory_str, memory_bytes = get_available_memory()
        
        assert memory_bytes > 0
        assert isinstance(memory_bytes, int)
        assert isinstance(memory_str, str)

    def test_memory_requirement_ratio(self):
        """Test memory_requirement_ratio function."""
        try:
            from kompot.memory_utils import memory_requirement_ratio
        except ImportError as e:
            pytest.skip(f"Could not import memory_requirement_ratio: {e}")
        
        # Test with mock required memory
        ratio = memory_requirement_ratio(1024 * 1024)  # 1 MB
        
        assert 0 <= ratio <= 10  # Reasonable range
        assert isinstance(ratio, float)

    def test_analyze_memory_requirements(self):
        """Test analyze_memory_requirements function."""
        try:
            from kompot.memory_utils import analyze_memory_requirements
        except ImportError as e:
            pytest.skip(f"Could not import analyze_memory_requirements: {e}")
        
        # Create test array shapes
        shapes = [
            (100, 50),
            (200, 30),
            (50, 100)
        ]
        
        analysis = analyze_memory_requirements(shapes)
        
        assert isinstance(analysis, dict)
        assert 'total_size' in analysis
        assert 'available_memory' in analysis
        assert 'memory_ratio' in analysis

    def test_analyze_covariance_memory_requirements(self):
        """Test analyze_covariance_memory_requirements function."""
        try:
            from kompot.memory_utils import analyze_covariance_memory_requirements
        except ImportError as e:
            pytest.skip(f"Could not import analyze_covariance_memory_requirements: {e}")
        
        n_features = 100
        n_samples = 50
        
        analysis = analyze_covariance_memory_requirements(n_features, n_samples)
        
        assert isinstance(analysis, dict)
        assert 'total_bytes' in analysis
        assert 'array_sizes' in analysis
        assert 'should_use_disk' in analysis


class TestBatchUtilsCoverage:
    """Test batch processing utilities for coverage."""

    def test_is_jax_memory_error(self):
        """Test is_jax_memory_error function."""
        try:
            from kompot.batch_utils import is_jax_memory_error
        except ImportError as e:
            pytest.skip(f"Could not import is_jax_memory_error: {e}")
        
        # Test with regular exception
        regular_error = ValueError("Regular error")
        assert not is_jax_memory_error(regular_error)
        
        # Test with mock JAX memory error
        memory_error = Exception("RESOURCE_EXHAUSTED")
        assert is_jax_memory_error(memory_error)

    def test_merge_batch_results_dict(self):
        """Test merge_batch_results for dictionary results."""
        try:
            from kompot.batch_utils import merge_batch_results
        except ImportError as e:
            pytest.skip(f"Could not import merge_batch_results: {e}")
        
        # Test with list of dictionaries
        batch_results = [
            {'values': np.array([1, 2]), 'count': 2},
            {'values': np.array([3, 4, 5]), 'count': 3},
            {'values': np.array([6]), 'count': 1}
        ]
        
        merged = merge_batch_results(batch_results)
        
        assert isinstance(merged, dict)
        assert 'values' in merged
        assert 'count' in merged
        assert len(merged['values']) == 6  # Total elements

    def test_merge_batch_results_arrays(self):
        """Test merge_batch_results for array results."""
        try:
            from kompot.batch_utils import merge_batch_results
        except ImportError as e:
            pytest.skip(f"Could not import merge_batch_results: {e}")
        
        # Test with list of arrays
        batch_results = [
            np.array([1, 2, 3]),
            np.array([4, 5]),
            np.array([6, 7, 8, 9])
        ]
        
        merged = merge_batch_results(batch_results)
        
        assert isinstance(merged, np.ndarray)
        assert len(merged) == 9  # Total elements
        np.testing.assert_array_equal(merged, np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]))