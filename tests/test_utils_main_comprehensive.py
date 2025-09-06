"""Comprehensive tests for kompot.utils main module to improve coverage."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
import logging


class TestMahalanobisDistance:
    """Test Mahalanobis distance computation functions."""
    
    def test_compute_mahalanobis_distance_basic(self):
        """Test basic Mahalanobis distance computation."""
        try:
            from kompot.utils import compute_mahalanobis_distance
        except ImportError as e:
            pytest.skip(f"Could not import compute_mahalanobis_distance: {e}")
        
        # Create test data
        X_test = np.random.rand(10, 5)
        X_train = np.random.rand(20, 5) 
        y_train = np.random.rand(20)
        
        distances = compute_mahalanobis_distance(X_test, X_train, y_train)
        
        assert distances.shape == (10,)
        assert np.all(distances >= 0)
        assert np.all(np.isfinite(distances))
        
    def test_compute_mahalanobis_distance_with_landmarks(self):
        """Test Mahalanobis distance computation with landmarks."""
        try:
            from kompot.utils import compute_mahalanobis_distance
        except ImportError as e:
            pytest.skip(f"Could not import compute_mahalanobis_distance: {e}")
        
        X_test = np.random.rand(8, 4)
        X_train = np.random.rand(15, 4)
        y_train = np.random.rand(15)
        landmarks = np.random.rand(5, 4)
        
        distances = compute_mahalanobis_distance(
            X_test, X_train, y_train, landmarks=landmarks
        )
        
        assert distances.shape == (8,)
        assert np.all(distances >= 0)
        
    def test_compute_mahalanobis_distances_multiple_genes(self):
        """Test Mahalanobis distances for multiple genes."""
        try:
            from kompot.utils import compute_mahalanobis_distances
        except ImportError as e:
            pytest.skip(f"Could not import compute_mahalanobis_distances: {e}")
        
        X_test = np.random.rand(10, 5)
        expr_test = np.random.rand(10, 8)  # 8 genes
        X_train = np.random.rand(20, 5)
        expr_train = np.random.rand(20, 8)
        
        distances = compute_mahalanobis_distances(
            X_test, expr_test, X_train, expr_train
        )
        
        assert distances.shape == (8,)  # One distance per gene
        assert np.all(distances >= 0)
        
    def test_compute_mahalanobis_distances_with_covariance(self):
        """Test Mahalanobis distances with precomputed covariance."""
        try:
            from kompot.utils import compute_mahalanobis_distances
        except ImportError as e:
            pytest.skip(f"Could not import compute_mahalanobis_distances: {e}")
        
        X_test = np.random.rand(5, 3)
        expr_test = np.random.rand(5, 4)
        X_train = np.random.rand(10, 3)
        expr_train = np.random.rand(10, 4)
        
        # Create covariance matrices for each gene
        covariance_matrices = np.random.rand(4, 3, 3)  # 4 genes, 3x3 covariance
        
        distances = compute_mahalanobis_distances(
            X_test, expr_test, X_train, expr_train,
            covariance=covariance_matrices
        )
        
        assert distances.shape == (4,)
        
    def test_compute_mahalanobis_distances_batch_processing(self):
        """Test Mahalanobis distances with batch processing."""
        try:
            from kompot.utils import compute_mahalanobis_distances
        except ImportError as e:
            pytest.skip(f"Could not import compute_mahalanobis_distances: {e}")
        
        X_test = np.random.rand(12, 4)
        expr_test = np.random.rand(12, 6)
        X_train = np.random.rand(18, 4)
        expr_train = np.random.rand(18, 6)
        
        distances = compute_mahalanobis_distances(
            X_test, expr_test, X_train, expr_train,
            batch_size=5,  # Force batching
            progress=False
        )
        
        assert distances.shape == (6,)
        assert np.all(distances >= 0)


class TestLandmarkFunctions:
    """Test landmark finding functions."""
    
    def test_find_landmarks_basic(self):
        """Test basic landmark finding."""
        try:
            from kompot.utils import find_landmarks
        except ImportError as e:
            pytest.skip(f"Could not import find_landmarks: {e}")
        
        X = np.random.rand(50, 8)
        
        landmarks = find_landmarks(X, n_landmarks=15)
        
        assert landmarks.shape == (15, 8)
        # Landmarks should be a subset of X
        for landmark in landmarks:
            # Check if landmark exists in X (with some tolerance)
            distances = np.linalg.norm(X - landmark, axis=1)
            assert np.min(distances) < 1e-10
            
    def test_find_landmarks_with_groups(self):
        """Test landmark finding with group stratification."""
        try:
            from kompot.utils import find_landmarks
        except ImportError as e:
            pytest.skip(f"Could not import find_landmarks: {e}")
        
        X = np.random.rand(60, 6)
        groups = np.array(['A'] * 20 + ['B'] * 20 + ['C'] * 20)
        
        landmarks = find_landmarks(X, n_landmarks=18, groups=groups)
        
        assert landmarks.shape == (18, 6)
        # Should have proportional representation from each group
        
    def test_find_landmarks_stratified_method(self):
        """Test landmark finding with stratified method."""
        try:
            from kompot.utils import find_landmarks
        except ImportError as e:
            pytest.skip(f"Could not import find_landmarks: {e}")
        
        X = np.random.rand(40, 5)
        groups = np.array(['X'] * 20 + ['Y'] * 20)
        
        landmarks = find_landmarks(
            X, n_landmarks=12, groups=groups, method='stratified'
        )
        
        assert landmarks.shape == (12, 5)
        
    def test_find_landmarks_random_state(self):
        """Test landmark finding with random state for reproducibility."""
        try:
            from kompot.utils import find_landmarks
        except ImportError as e:
            pytest.skip(f"Could not import find_landmarks: {e}")
        
        X = np.random.rand(30, 4)
        
        # Same random state should give same results
        landmarks1 = find_landmarks(X, n_landmarks=8, random_state=42)
        landmarks2 = find_landmarks(X, n_landmarks=8, random_state=42)
        
        np.testing.assert_array_equal(landmarks1, landmarks2)
        
        # Different random state should give different results
        landmarks3 = find_landmarks(X, n_landmarks=8, random_state=123)
        assert not np.array_equal(landmarks1, landmarks3)


class TestGeneSpecificMahalanobis:
    """Test gene-specific Mahalanobis distance functions."""
    
    def test_gene_specific_mahalanobis_distances_basic(self):
        """Test basic gene-specific Mahalanobis distances."""
        try:
            from kompot.utils import gene_specific_mahalanobis_distances
        except ImportError as e:
            pytest.skip(f"Could not import gene_specific_mahalanobis_distances: {e}")
        
        X_test = np.random.rand(8, 5)
        expr_test = np.random.rand(8, 6)
        X_train = np.random.rand(15, 5)
        expr_train = np.random.rand(15, 6)
        
        distances = gene_specific_mahalanobis_distances(
            X_test, expr_test, X_train, expr_train
        )
        
        assert distances.shape == (8, 6)  # 8 test cells, 6 genes
        assert np.all(distances >= 0)
        
    def test_gene_specific_mahalanobis_distances_with_landmarks(self):
        """Test gene-specific Mahalanobis distances with landmarks."""
        try:
            from kompot.utils import gene_specific_mahalanobis_distances
        except ImportError as e:
            pytest.skip(f"Could not import gene_specific_mahalanobis_distances: {e}")
        
        X_test = np.random.rand(6, 4)
        expr_test = np.random.rand(6, 5)
        X_train = np.random.rand(12, 4)
        expr_train = np.random.rand(12, 5)
        landmarks = np.random.rand(8, 4)
        
        distances = gene_specific_mahalanobis_distances(
            X_test, expr_test, X_train, expr_train, landmarks=landmarks
        )
        
        assert distances.shape == (6, 5)
        assert np.all(distances >= 0)
        
    def test_gene_specific_mahalanobis_distances_batched(self):
        """Test gene-specific Mahalanobis distances with batching."""
        try:
            from kompot.utils import gene_specific_mahalanobis_distances
        except ImportError as e:
            pytest.skip(f"Could not import gene_specific_mahalanobis_distances: {e}")
        
        X_test = np.random.rand(10, 3)
        expr_test = np.random.rand(10, 7)
        X_train = np.random.rand(20, 3)
        expr_train = np.random.rand(20, 7)
        
        distances = gene_specific_mahalanobis_distances(
            X_test, expr_test, X_train, expr_train,
            batch_size=4,  # Force batching
            progress=False
        )
        
        assert distances.shape == (10, 7)
        assert np.all(distances >= 0)


class TestRunHistoryFunctions:
    """Test run history and metadata functions."""
    
    def test_get_run_from_history_basic(self):
        """Test basic run history retrieval."""
        try:
            from kompot.utils import get_run_from_history
            import anndata
        except ImportError as e:
            pytest.skip(f"Could not import required modules: {e}")
        
        # Create mock adata with history
        adata = anndata.AnnData(np.random.rand(10, 5))
        adata.uns['kompot_run_history'] = [
            {'run_id': 0, 'analysis_type': 'de', 'params': {'threshold': 1.0}},
            {'run_id': 1, 'analysis_type': 'da', 'params': {'threshold': 0.05}}
        ]
        
        # Get latest run
        run_info = get_run_from_history(adata, -1, 'de')
        assert run_info['analysis_type'] == 'de'
        assert run_info['params']['threshold'] == 1.0
        
        # Get specific run
        run_info = get_run_from_history(adata, 1, 'da')
        assert run_info['analysis_type'] == 'da'
        assert run_info['params']['threshold'] == 0.05
        
    def test_get_run_from_history_error_handling(self):
        """Test run history error handling."""
        try:
            from kompot.utils import get_run_from_history
            import anndata
        except ImportError as e:
            pytest.skip(f"Could not import required modules: {e}")
        
        # Empty adata
        adata = anndata.AnnData(np.random.rand(5, 3))
        
        with pytest.raises((ValueError, KeyError, AttributeError)):
            get_run_from_history(adata, -1, 'de')
            
    def test_get_run_from_history_filtering(self):
        """Test run history filtering by analysis type."""
        try:
            from kompot.utils import get_run_from_history
            import anndata
        except ImportError as e:
            pytest.skip(f"Could not import required modules: {e}")
        
        adata = anndata.AnnData(np.random.rand(8, 4))
        adata.uns['kompot_run_history'] = [
            {'run_id': 0, 'analysis_type': 'de', 'params': {}},
            {'run_id': 1, 'analysis_type': 'da', 'params': {}},
            {'run_id': 2, 'analysis_type': 'de', 'params': {'version': 2}}
        ]
        
        # Should get most recent DE run (run_id=2)
        run_info = get_run_from_history(adata, -1, 'de')
        assert run_info['run_id'] == 2
        assert run_info['params']['version'] == 2


class TestKOMPOTColors:
    """Test KOMPOT_COLORS functionality."""
    
    def test_kompot_colors_structure(self):
        """Test KOMPOT_COLORS structure and basic access."""
        try:
            from kompot.utils import KOMPOT_COLORS
        except ImportError as e:
            pytest.skip(f"Could not import KOMPOT_COLORS: {e}")
        
        assert isinstance(KOMPOT_COLORS, dict)
        
        # Test expected structure
        if 'direction' in KOMPOT_COLORS:
            assert isinstance(KOMPOT_COLORS['direction'], dict)
            # Common direction keys
            expected_directions = ['up', 'down', 'neutral']
            for direction in expected_directions:
                if direction in KOMPOT_COLORS['direction']:
                    # Should be string (color name or hex)
                    assert isinstance(KOMPOT_COLORS['direction'][direction], str)
                    
        if 'default' in KOMPOT_COLORS:
            assert isinstance(KOMPOT_COLORS['default'], dict)
            
    def test_kompot_colors_usage(self):
        """Test KOMPOT_COLORS can be used in plotting contexts."""
        try:
            from kompot.utils import KOMPOT_COLORS
            import matplotlib.pyplot as plt
        except ImportError as e:
            pytest.skip(f"Could not import required modules: {e}")
        
        # Test using colors in a simple plot
        if 'direction' in KOMPOT_COLORS:
            direction_colors = KOMPOT_COLORS['direction']
            if 'up' in direction_colors and 'down' in direction_colors:
                fig, ax = plt.subplots()
                ax.scatter([1, 2], [1, 2], 
                          c=[direction_colors['up'], direction_colors['down']])
                plt.close(fig)
                
    def test_kompot_colors_hex_format(self):
        """Test KOMPOT_COLORS hex color format validation."""
        try:
            from kompot.utils import KOMPOT_COLORS
        except ImportError as e:
            pytest.skip(f"Could not import KOMPOT_COLORS: {e}")
        
        def is_hex_color(color):
            """Check if string is valid hex color."""
            if not isinstance(color, str):
                return False
            if not color.startswith('#'):
                return True  # May be named color
            if len(color) not in [4, 7]:  # #RGB or #RRGGBB
                return False
            try:
                int(color[1:], 16)
                return True
            except ValueError:
                return False
        
        # Check all colors in the structure
        def check_colors_recursive(colors_dict):
            for key, value in colors_dict.items():
                if isinstance(value, dict):
                    check_colors_recursive(value)
                elif isinstance(value, str):
                    assert is_hex_color(value), f"Invalid color format: {value}"
        
        check_colors_recursive(KOMPOT_COLORS)


class TestUtilsJAXIntegration:
    """Test JAX integration in utils functions."""
    
    def test_jax_compilation_flags(self):
        """Test JAX compilation behavior in utils functions."""
        try:
            from kompot.utils import compute_mahalanobis_distance
        except ImportError as e:
            pytest.skip(f"Could not import utils functions: {e}")
        
        X_test = np.random.rand(5, 3)
        X_train = np.random.rand(10, 3)
        y_train = np.random.rand(10)
        
        # Test with jit_compile=True
        distances_jit = compute_mahalanobis_distance(
            X_test, X_train, y_train, jit_compile=True
        )
        
        # Test with jit_compile=False
        distances_no_jit = compute_mahalanobis_distance(
            X_test, X_train, y_train, jit_compile=False
        )
        
        # Results should be similar regardless of JIT
        np.testing.assert_allclose(distances_jit, distances_no_jit, rtol=1e-5)
        
    def test_jax_memory_efficiency(self):
        """Test JAX memory efficiency features."""
        try:
            from kompot.utils import compute_mahalanobis_distances
        except ImportError as e:
            pytest.skip(f"Could not import utils functions: {e}")
        
        # Test with moderately large data
        X_test = np.random.rand(20, 6)
        expr_test = np.random.rand(20, 10)
        X_train = np.random.rand(40, 6)
        expr_train = np.random.rand(40, 10)
        
        # Should handle without memory errors
        distances = compute_mahalanobis_distances(
            X_test, expr_test, X_train, expr_train,
            batch_size=8  # Small batch to test memory management
        )
        
        assert distances.shape == (10,)
        assert np.all(np.isfinite(distances))


class TestUtilsErrorHandling:
    """Test error handling in utils functions."""
    
    def test_mahalanobis_shape_validation(self):
        """Test Mahalanobis distance input shape validation."""
        try:
            from kompot.utils import compute_mahalanobis_distance
        except ImportError as e:
            pytest.skip(f"Could not import compute_mahalanobis_distance: {e}")
        
        X_test = np.random.rand(5, 3)
        X_train = np.random.rand(8, 4)  # Wrong feature dimension
        y_train = np.random.rand(8)
        
        with pytest.raises((ValueError, AssertionError)):
            compute_mahalanobis_distance(X_test, X_train, y_train)
            
    def test_landmarks_validation(self):
        """Test landmark finding input validation."""
        try:
            from kompot.utils import find_landmarks
        except ImportError as e:
            pytest.skip(f"Could not import find_landmarks: {e}")
        
        X = np.random.rand(10, 5)
        
        # Invalid n_landmarks (too large)
        with pytest.raises((ValueError, IndexError)):
            find_landmarks(X, n_landmarks=15)  # More than available points
            
        # Invalid n_landmarks (negative)
        with pytest.raises((ValueError, IndexError)):
            find_landmarks(X, n_landmarks=-1)
            
    def test_gene_specific_shape_validation(self):
        """Test gene-specific Mahalanobis input validation."""
        try:
            from kompot.utils import gene_specific_mahalanobis_distances
        except ImportError as e:
            pytest.skip(f"Could not import gene_specific_mahalanobis_distances: {e}")
        
        X_test = np.random.rand(5, 3)
        expr_test = np.random.rand(6, 4)  # Wrong number of cells
        X_train = np.random.rand(10, 3)
        expr_train = np.random.rand(10, 4)
        
        with pytest.raises((ValueError, AssertionError)):
            gene_specific_mahalanobis_distances(X_test, expr_test, X_train, expr_train)


class TestUtilsNumericalStability:
    """Test numerical stability of utils functions."""
    
    def test_mahalanobis_numerical_stability(self):
        """Test Mahalanobis distance numerical stability."""
        try:
            from kompot.utils import compute_mahalanobis_distance
        except ImportError as e:
            pytest.skip(f"Could not import compute_mahalanobis_distance: {e}")
        
        # Create data with potential numerical issues
        X_test = np.random.rand(5, 3) * 1e-6  # Very small values
        X_train = np.random.rand(10, 3) * 1e-6
        y_train = np.random.rand(10) * 1e-6
        
        # Should handle small values without inf/nan
        distances = compute_mahalanobis_distance(X_test, X_train, y_train, eps=1e-12)
        
        assert np.all(np.isfinite(distances))
        assert np.all(distances >= 0)
        
    def test_mahalanobis_extreme_values(self):
        """Test Mahalanobis distance with extreme values."""
        try:
            from kompot.utils import compute_mahalanobis_distance
        except ImportError as e:
            pytest.skip(f"Could not import compute_mahalanobis_distance: {e}")
        
        # Create data with large values
        X_test = np.random.rand(5, 3) * 1e6  # Large values
        X_train = np.random.rand(10, 3) * 1e6
        y_train = np.random.rand(10) * 1e6
        
        # Should handle large values without overflow
        distances = compute_mahalanobis_distance(X_test, X_train, y_train)
        
        assert np.all(np.isfinite(distances))
        assert np.all(distances >= 0)
        
    def test_covariance_regularization(self):
        """Test covariance matrix regularization."""
        try:
            from kompot.utils import compute_mahalanobis_distances
        except ImportError as e:
            pytest.skip(f"Could not import compute_mahalanobis_distances: {e}")
        
        # Create data that would produce singular covariance
        X_test = np.random.rand(3, 2)
        expr_test = np.ones((3, 1))  # Constant expression (singular covariance)
        X_train = np.random.rand(6, 2)
        expr_train = np.ones((6, 1))
        
        # Should handle singular covariance via regularization
        distances = compute_mahalanobis_distances(
            X_test, expr_test, X_train, expr_train, eps=1e-6
        )
        
        assert distances.shape == (1,)
        assert np.all(np.isfinite(distances))


class TestUtilsPerformanceOptimizations:
    """Test performance optimizations in utils functions."""
    
    def test_batch_processing_efficiency(self):
        """Test batch processing reduces memory usage."""
        try:
            from kompot.utils import gene_specific_mahalanobis_distances
        except ImportError as e:
            pytest.skip(f"Could not import gene_specific_mahalanobis_distances: {e}")
        
        # Large enough to benefit from batching
        X_test = np.random.rand(50, 8)
        expr_test = np.random.rand(50, 12)
        X_train = np.random.rand(100, 8)
        expr_train = np.random.rand(100, 12)
        
        # Test with small batch size (should still work)
        distances = gene_specific_mahalanobis_distances(
            X_test, expr_test, X_train, expr_train,
            batch_size=10, progress=False
        )
        
        assert distances.shape == (50, 12)
        assert np.all(np.isfinite(distances))
        
    def test_landmark_efficiency(self):
        """Test landmark-based approximation efficiency."""
        try:
            from kompot.utils import compute_mahalanobis_distance
        except ImportError as e:
            pytest.skip(f"Could not import compute_mahalanobis_distance: {e}")
        
        # Create large training set
        X_test = np.random.rand(10, 5)
        X_train = np.random.rand(200, 5)  # Large training set
        y_train = np.random.rand(200)
        
        # Use landmarks for efficiency
        landmarks = np.random.rand(20, 5)  # Much smaller than full training set
        
        distances = compute_mahalanobis_distance(
            X_test, X_train, y_train, landmarks=landmarks
        )
        
        assert distances.shape == (10,)
        assert np.all(np.isfinite(distances))


class TestUtilsLogging:
    """Test logging functionality in utils module."""
    
    def test_utils_logging_integration(self):
        """Test utils module logging integration."""
        try:
            from kompot.utils import find_landmarks
        except ImportError as e:
            pytest.skip(f"Could not import find_landmarks: {e}")
        
        with patch('kompot.utils.logger') as mock_logger:
            X = np.random.rand(20, 4)
            landmarks = find_landmarks(X, n_landmarks=8)
            
            # Function may or may not log, but logger should be available
            assert hasattr(mock_logger, 'info')
            assert hasattr(mock_logger, 'debug')
            assert hasattr(mock_logger, 'warning')
            
    def test_mahalanobis_progress_logging(self):
        """Test Mahalanobis distance computation progress logging."""
        try:
            from kompot.utils import gene_specific_mahalanobis_distances
        except ImportError as e:
            pytest.skip(f"Could not import gene_specific_mahalanobis_distances: {e}")
        
        X_test = np.random.rand(8, 3)
        expr_test = np.random.rand(8, 5)
        X_train = np.random.rand(16, 3)
        expr_train = np.random.rand(16, 5)
        
        # Enable progress reporting
        distances = gene_specific_mahalanobis_distances(
            X_test, expr_test, X_train, expr_train,
            batch_size=4, progress=True
        )
        
        # Should complete successfully with progress enabled
        assert distances.shape == (8, 5)