"""Comprehensive tests for DifferentialExpression core functionality to improve coverage."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
import logging

def create_de_test_data(n_cells1=25, n_cells2=30, n_features=6, n_genes=8):
    """Create test data for differential expression analysis."""
    np.random.seed(42)
    
    # Create cell embeddings
    X1 = np.random.normal(0, 1, (n_cells1, n_features))
    X2 = np.random.normal(0.5, 1, (n_cells2, n_features))
    
    # Create gene expressions with some differential genes
    expr1 = np.random.normal(2, 0.5, (n_cells1, n_genes))
    expr2 = np.random.normal(2, 0.5, (n_cells2, n_genes))
    
    # Make first few genes differentially expressed
    expr2[:, :3] += 1.0  # Higher expression in condition 2
    
    return X1, X2, expr1, expr2


class TestDifferentialExpressionInit:
    """Test DifferentialExpression initialization."""
    
    def test_differential_expression_init_basic(self):
        """Test basic DifferentialExpression initialization."""
        try:
            from kompot.differential import DifferentialExpression
        except ImportError as e:
            pytest.skip(f"Could not import DifferentialExpression: {e}")
        
        de = DifferentialExpression(
            n_landmarks=15,
            eps=1e-6,
            batch_size=100,
            max_memory_ratio=0.7
        )
        
        assert de.n_landmarks == 15
        assert de.eps == 1e-6
        assert de.batch_size == 100
        assert de.max_memory_ratio == 0.7
        assert de.use_sample_variance == False  # Default when no predictors
        
    def test_differential_expression_init_with_predictors(self):
        """Test DifferentialExpression initialization with predictors."""
        try:
            from kompot.differential import DifferentialExpression
        except ImportError as e:
            pytest.skip(f"Could not import DifferentialExpression: {e}")
        
        mock_function_predictor = MagicMock()
        mock_variance_predictor = MagicMock()
        
        de = DifferentialExpression(
            function_predictor1=mock_function_predictor,
            variance_predictor1=mock_variance_predictor,
            use_sample_variance=None  # Should auto-enable
        )
        
        assert de.function_predictor1 == mock_function_predictor
        assert de.variance_predictor1 == mock_variance_predictor
        assert de.use_sample_variance == True  # Auto-enabled
        
    def test_differential_expression_init_disk_storage(self):
        """Test DifferentialExpression initialization with disk storage."""
        try:
            from kompot.differential import DifferentialExpression
        except ImportError as e:
            pytest.skip(f"Could not import DifferentialExpression: {e}")
        
        de = DifferentialExpression(
            store_arrays_on_disk=None,
            disk_storage_dir="/tmp/test"
        )
        
        assert de.store_arrays_on_disk == True  # Auto-enabled
        assert de.disk_storage_dir == "/tmp/test"
        
    def test_differential_expression_init_contradictory_variance(self):
        """Test DifferentialExpression with contradictory variance parameters."""
        try:
            from kompot.differential import DifferentialExpression
        except ImportError as e:
            pytest.skip(f"Could not import DifferentialExpression: {e}")
        
        de = DifferentialExpression(use_sample_variance=True)
        X1, X2, expr1, expr2 = create_de_test_data()
        
        # Should raise error when use_sample_variance=True but no predictors/indices
        with pytest.raises(ValueError, match="Sample variance estimation was explicitly enabled"):
            de.fit(X1, expr1, X2, expr2)


class TestDifferentialExpressionFit:
    """Test DifferentialExpression fit functionality."""
    
    def test_differential_expression_fit_basic(self):
        """Test basic DifferentialExpression fit."""
        try:
            from kompot.differential import DifferentialExpression
        except ImportError as e:
            pytest.skip(f"Could not import DifferentialExpression: {e}")
        
        X1, X2, expr1, expr2 = create_de_test_data(n_cells1=15, n_cells2=20, n_features=4, n_genes=5)
        
        de = DifferentialExpression(n_landmarks=8, batch_size=50)
        
        # Mock mellon to avoid heavy computation
        with patch('kompot.differential.differential_expression.mellon') as mock_mellon:
            mock_estimator = MagicMock()
            mock_predictor = MagicMock()
            mock_estimator.predict = mock_predictor
            mock_mellon.FunctionEstimator.return_value = mock_estimator
            mock_mellon.parameters.compute_landmarks.return_value = np.random.rand(8, 4)
            
            result = de.fit(X1, expr1, X2, expr2)
            
            assert result is de
            assert de.function_predictor1 is not None
            assert de.function_predictor2 is not None
            
    def test_differential_expression_fit_with_landmarks(self):
        """Test DifferentialExpression fit with provided landmarks."""
        try:
            from kompot.differential import DifferentialExpression
        except ImportError as e:
            pytest.skip(f"Could not import DifferentialExpression: {e}")
        
        X1, X2, expr1, expr2 = create_de_test_data()
        landmarks = np.random.rand(10, 6)
        
        de = DifferentialExpression()
        
        with patch('kompot.differential.differential_expression.mellon') as mock_mellon:
            mock_estimator = MagicMock()
            mock_predictor = MagicMock()
            mock_estimator.predict = mock_predictor
            mock_mellon.FunctionEstimator.return_value = mock_estimator
            
            de.fit(X1, expr1, X2, expr2, landmarks=landmarks)
            
            assert hasattr(de, 'computed_landmarks')
            np.testing.assert_array_equal(de.computed_landmarks, landmarks)
            
    def test_differential_expression_fit_sync_parameters(self):
        """Test DifferentialExpression fit with synchronized parameters."""
        try:
            from kompot.differential import DifferentialExpression
        except ImportError as e:
            pytest.skip(f"Could not import DifferentialExpression: {e}")
        
        X1, X2, expr1, expr2 = create_de_test_data()
        
        de = DifferentialExpression()
        
        with patch('kompot.differential.differential_expression.mellon') as mock_mellon:
            # Mock parameter computation functions
            mock_mellon.parameters.compute_d_factal.return_value = 3.2
            mock_mellon.parameters.compute_nn_distances.return_value = np.array([0.1, 0.2, 0.3])
            mock_mellon.parameters.compute_mu.return_value = 0.9
            mock_mellon.parameters.compute_ls.return_value = 0.6
            mock_mellon.parameters.compute_landmarks.return_value = np.random.rand(10, 6)
            
            mock_estimator = MagicMock()
            mock_predictor = MagicMock()
            mock_estimator.predict = mock_predictor
            mock_mellon.FunctionEstimator.return_value = mock_estimator
            
            de.fit(X1, expr1, X2, expr2, sync_parameters=True, n_landmarks=10)
            
            # Should call parameter computation functions
            mock_mellon.parameters.compute_d_factal.assert_called_once()
            mock_mellon.parameters.compute_nn_distances.assert_called_once()
            mock_mellon.parameters.compute_mu.assert_called_once()
            mock_mellon.parameters.compute_ls.assert_called_once()
            
    def test_differential_expression_fit_with_sample_indices(self):
        """Test DifferentialExpression fit with sample variance estimation."""
        try:
            from kompot.differential import DifferentialExpression
        except ImportError as e:
            pytest.skip(f"Could not import DifferentialExpression: {e}")
        
        X1, X2, expr1, expr2 = create_de_test_data(n_cells1=20, n_cells2=24)
        
        # Create sample indices
        indices1 = np.array([0] * 10 + [1] * 10)
        indices2 = np.array([0] * 12 + [1] * 12)
        
        de = DifferentialExpression(use_sample_variance=None)
        
        with patch('kompot.differential.differential_expression.mellon') as mock_mellon:
            mock_estimator = MagicMock()
            mock_predictor = MagicMock()
            mock_estimator.predict = mock_predictor
            mock_mellon.FunctionEstimator.return_value = mock_estimator
            
            with patch('kompot.differential.differential_expression.SampleVarianceEstimator') as mock_sve:
                mock_variance_estimator = MagicMock()
                mock_variance_predictor = MagicMock()
                mock_variance_estimator.predict = mock_variance_predictor
                mock_sve.return_value = mock_variance_estimator
                
                de.fit(
                    X1, expr1, X2, expr2,
                    condition1_sample_indices=indices1,
                    condition2_sample_indices=indices2
                )
                
                assert de.use_sample_variance == True
                assert de.variance_predictor1 is not None
                assert de.variance_predictor2 is not None
                
    def test_differential_expression_fit_single_condition_variance(self):
        """Test DifferentialExpression fit with single condition variance fallback."""
        try:
            from kompot.differential import DifferentialExpression
        except ImportError as e:
            pytest.skip(f"Could not import DifferentialExpression: {e}")
        
        X1, X2, expr1, expr2 = create_de_test_data()
        indices1 = np.array([0] * 12 + [1] * 13)
        indices2 = np.array([0] * 15 + [1] * 15)
        
        de = DifferentialExpression()
        
        with patch('kompot.differential.differential_expression.mellon') as mock_mellon:
            mock_estimator = MagicMock()
            mock_predictor = MagicMock()
            mock_estimator.predict = mock_predictor
            mock_mellon.FunctionEstimator.return_value = mock_estimator
            
            with patch('kompot.differential.differential_expression.SampleVarianceEstimator') as mock_sve:
                # Mock successful estimator for condition 1, failed for condition 2
                mock_variance_estimator1 = MagicMock()
                mock_variance_predictor1 = MagicMock()
                mock_variance_estimator1.predict = mock_variance_predictor1
                
                def side_effect(*args, **kwargs):
                    if not hasattr(side_effect, 'called'):
                        side_effect.called = True
                        return mock_variance_estimator1
                    else:
                        estimator = MagicMock()
                        estimator.fit.side_effect = ValueError("Not enough samples")
                        return estimator
                        
                mock_sve.side_effect = side_effect
                
                de.fit(
                    X1, expr1, X2, expr2,
                    condition1_sample_indices=indices1,
                    condition2_sample_indices=indices2,
                    allow_single_condition_variance=True
                )
                
                # Should use condition 1 variance for both conditions
                assert de.variance_predictor1 is not None
                assert de.variance_predictor2 is not None


class TestDifferentialExpressionPredict:
    """Test DifferentialExpression prediction functionality."""
    
    def test_differential_expression_predict_not_fitted(self):
        """Test DifferentialExpression predict without fitting first."""
        try:
            from kompot.differential import DifferentialExpression
        except ImportError as e:
            pytest.skip(f"Could not import DifferentialExpression: {e}")
        
        de = DifferentialExpression()
        X_test = np.random.rand(10, 5)
        
        with pytest.raises(ValueError, match="Model not fitted"):
            de.predict(X_test)
            
    def test_differential_expression_predict_basic(self):
        """Test basic DifferentialExpression prediction."""
        try:
            from kompot.differential import DifferentialExpression
        except ImportError as e:
            pytest.skip(f"Could not import DifferentialExpression: {e}")
        
        de = DifferentialExpression()
        
        # Mock function predictors
        mock_predictor1 = MagicMock()
        mock_predictor2 = MagicMock()
        mock_predictor1.return_value = np.array([[1.0, 2.0], [1.5, 2.5], [2.0, 3.0]])  # 3 cells, 2 genes
        mock_predictor2.return_value = np.array([[1.5, 3.0], [2.0, 3.5], [2.5, 4.0]])
        mock_predictor1.uncertainty.return_value = np.array([[0.1, 0.2], [0.15, 0.25], [0.2, 0.3]])
        mock_predictor2.uncertainty.return_value = np.array([[0.12, 0.22], [0.17, 0.27], [0.22, 0.32]])
        
        de.function_predictor1 = mock_predictor1
        de.function_predictor2 = mock_predictor2
        
        X_test = np.random.rand(3, 5)
        
        with patch('kompot.differential.differential_expression.apply_batched') as mock_batch:
            with patch('kompot.differential.differential_expression.compute_mahalanobis_distances') as mock_mahal:
                mock_batch.side_effect = lambda func, X, **kwargs: func(X)
                mock_mahal.return_value = np.array([0.5, 0.8])  # 2 genes
                
                results = de.predict(X_test)
                
                assert 'expression_condition1' in results
                assert 'expression_condition2' in results
                assert 'log_fold_change' in results
                assert 'log_fold_change_uncertainty' in results
                assert 'log_fold_change_zscore' in results
                assert 'neg_log10_fold_change_ptp' in results
                assert 'log_fold_change_direction' in results
                assert 'mahalanobis_distances' in results
                
                # Check shapes
                assert results['log_fold_change'].shape == (3, 2)  # 3 cells, 2 genes
                assert results['mahalanobis_distances'].shape == (2,)  # 2 genes
                
    def test_differential_expression_predict_with_sample_variance(self):
        """Test DifferentialExpression prediction with sample variance."""
        try:
            from kompot.differential import DifferentialExpression
        except ImportError as e:
            pytest.skip(f"Could not import DifferentialExpression: {e}")
        
        de = DifferentialExpression(use_sample_variance=True)
        
        # Mock all predictors
        mock_function_predictor1 = MagicMock()
        mock_function_predictor2 = MagicMock()
        mock_variance_predictor1 = MagicMock()
        mock_variance_predictor2 = MagicMock()
        
        mock_function_predictor1.return_value = np.array([[1.0, 2.0], [1.5, 2.5]])
        mock_function_predictor2.return_value = np.array([[1.5, 3.0], [2.0, 3.5]])
        mock_function_predictor1.uncertainty.return_value = np.array([[0.1, 0.2], [0.15, 0.25]])
        mock_function_predictor2.uncertainty.return_value = np.array([[0.12, 0.22], [0.17, 0.27]])
        mock_variance_predictor1.return_value = np.array([[0.05, 0.08], [0.06, 0.09]])
        mock_variance_predictor2.return_value = np.array([[0.07, 0.10], [0.08, 0.11]])
        
        de.function_predictor1 = mock_function_predictor1
        de.function_predictor2 = mock_function_predictor2
        de.variance_predictor1 = mock_variance_predictor1
        de.variance_predictor2 = mock_variance_predictor2
        
        X_test = np.random.rand(2, 4)
        
        with patch('kompot.differential.differential_expression.apply_batched') as mock_batch:
            with patch('kompot.differential.differential_expression.compute_mahalanobis_distances') as mock_mahal:
                mock_batch.side_effect = lambda func, X, **kwargs: func(X)
                mock_mahal.return_value = np.array([0.3, 0.7])
                
                results = de.predict(X_test, progress=False)
                
                assert 'log_fold_change_uncertainty' in results
                # Uncertainty should include sample variance contribution
                assert results['log_fold_change_uncertainty'] is not None
                
    def test_differential_expression_predict_custom_thresholds(self):
        """Test DifferentialExpression prediction with custom thresholds."""
        try:
            from kompot.differential import DifferentialExpression
        except ImportError as e:
            pytest.skip(f"Could not import DifferentialExpression: {e}")
        
        de = DifferentialExpression()
        
        # Mock predictors with strong signal
        mock_predictor1 = MagicMock()
        mock_predictor2 = MagicMock()
        mock_predictor1.return_value = np.array([[1.0, 1.0], [1.0, 1.0]])
        mock_predictor2.return_value = np.array([[3.0, 0.2], [3.0, 0.2]])  # Strong up/down
        mock_predictor1.uncertainty.return_value = np.array([[0.01, 0.01], [0.01, 0.01]])
        mock_predictor2.uncertainty.return_value = np.array([[0.01, 0.01], [0.01, 0.01]])
        
        de.function_predictor1 = mock_predictor1
        de.function_predictor2 = mock_predictor2
        
        X_test = np.random.rand(2, 3)
        
        with patch('kompot.differential.differential_expression.apply_batched') as mock_batch:
            with patch('kompot.differential.differential_expression.compute_mahalanobis_distances') as mock_mahal:
                mock_batch.side_effect = lambda func, X, **kwargs: func(X)
                mock_mahal.return_value = np.array([0.1, 0.1])
                
                results = de.predict(
                    X_test,
                    log_fold_change_threshold=1.5,  # Custom threshold
                    ptp_threshold=0.01,  # Custom threshold
                    progress=False
                )
                
                directions = results['log_fold_change_direction']
                assert directions.shape == (2, 2)  # 2 cells, 2 genes


class TestDifferentialExpressionMemoryManagement:
    """Test DifferentialExpression memory management features."""
    
    def test_differential_expression_memory_analysis(self):
        """Test DifferentialExpression memory requirement analysis."""
        try:
            from kompot.differential import DifferentialExpression
        except ImportError as e:
            pytest.skip(f"Could not import DifferentialExpression: {e}")
        
        de = DifferentialExpression(max_memory_ratio=0.5)
        
        with patch('kompot.differential.differential_expression.analyze_covariance_memory_requirements') as mock_analyze:
            mock_analyze.return_value = {
                'covariance_size': 1000,
                'total_memory_required': 2000,
                'available_memory': 8000,
                'memory_ratio': 0.25
            }
            
            X1, X2, expr1, expr2 = create_de_test_data()
            
            with patch('kompot.differential.differential_expression.mellon') as mock_mellon:
                mock_estimator = MagicMock()
                mock_predictor = MagicMock()
                mock_estimator.predict = mock_predictor
                mock_mellon.FunctionEstimator.return_value = mock_estimator
                
                de.fit(X1, expr1, X2, expr2)
                
                # Should call memory analysis
                mock_analyze.assert_called()
                
    def test_differential_expression_disk_storage_setup(self):
        """Test DifferentialExpression disk storage setup."""
        try:
            from kompot.differential import DifferentialExpression
        except ImportError as e:
            pytest.skip(f"Could not import DifferentialExpression: {e}")
        
        de = DifferentialExpression(
            store_arrays_on_disk=True,
            disk_storage_dir="/tmp/test"
        )
        
        with patch('kompot.differential.differential_expression.DiskStorage') as mock_disk:
            mock_storage = MagicMock()
            mock_disk.return_value = mock_storage
            
            X1, X2, expr1, expr2 = create_de_test_data()
            
            with patch('kompot.differential.differential_expression.mellon') as mock_mellon:
                mock_estimator = MagicMock()
                mock_predictor = MagicMock()
                mock_estimator.predict = mock_predictor
                mock_mellon.FunctionEstimator.return_value = mock_estimator
                
                de.fit(X1, expr1, X2, expr2)
                
                # Should create disk storage
                mock_disk.assert_called_with("/tmp/test")


class TestDifferentialExpressionErrorHandling:
    """Test DifferentialExpression error handling and edge cases."""
    
    def test_differential_expression_mismatched_shapes(self):
        """Test DifferentialExpression with mismatched input shapes."""
        try:
            from kompot.differential import DifferentialExpression
        except ImportError as e:
            pytest.skip(f"Could not import DifferentialExpression: {e}")
        
        de = DifferentialExpression()
        
        X1 = np.random.rand(20, 5)
        X2 = np.random.rand(25, 5)
        expr1 = np.random.rand(20, 10)
        expr2 = np.random.rand(30, 10)  # Wrong number of cells
        
        with pytest.raises((ValueError, AssertionError)):
            de.fit(X1, expr1, X2, expr2)
            
    def test_differential_expression_logging(self):
        """Test DifferentialExpression logging functionality."""
        try:
            from kompot.differential import DifferentialExpression
        except ImportError as e:
            pytest.skip(f"Could not import DifferentialExpression: {e}")
        
        with patch('kompot.differential.differential_expression.logger') as mock_logger:
            de = DifferentialExpression(
                variance_predictor1=MagicMock(),
                use_sample_variance=None
            )
            
            mock_logger.info.assert_called_with(
                "Sample variance estimation automatically enabled due to presence of variance predictors"
            )
            
    def test_differential_expression_jax_memory_error_handling(self):
        """Test DifferentialExpression JAX memory error handling."""
        try:
            from kompot.differential import DifferentialExpression
        except ImportError as e:
            pytest.skip(f"Could not import DifferentialExpression: {e}")
        
        de = DifferentialExpression()
        
        # Mock function predictors
        mock_predictor1 = MagicMock()
        mock_predictor2 = MagicMock()
        mock_predictor1.return_value = np.array([[1.0, 2.0]])
        mock_predictor2.return_value = np.array([[1.5, 3.0]])
        mock_predictor1.uncertainty.return_value = np.array([[0.1, 0.2]])
        mock_predictor2.uncertainty.return_value = np.array([[0.12, 0.22]])
        
        de.function_predictor1 = mock_predictor1
        de.function_predictor2 = mock_predictor2
        
        X_test = np.random.rand(1, 4)
        
        with patch('kompot.differential.differential_expression.apply_batched') as mock_batch:
            with patch('kompot.differential.differential_expression.compute_mahalanobis_distances') as mock_mahal:
                # Mock JAX memory error
                mock_mahal.side_effect = Exception("RESOURCE_EXHAUSTED")
                mock_batch.side_effect = lambda func, X, **kwargs: func(X)
                
                with patch('kompot.differential.differential_expression.is_jax_memory_error') as mock_is_mem_error:
                    mock_is_mem_error.return_value = True
                    
                    # Should handle memory error gracefully
                    with pytest.raises(Exception, match="RESOURCE_EXHAUSTED"):
                        de.predict(X_test)


class TestDifferentialExpressionBatching:
    """Test DifferentialExpression batching functionality."""
    
    def test_differential_expression_batched_prediction(self):
        """Test DifferentialExpression with explicit batching."""
        try:
            from kompot.differential import DifferentialExpression
        except ImportError as e:
            pytest.skip(f"Could not import DifferentialExpression: {e}")
        
        de = DifferentialExpression(batch_size=2)  # Small batch size
        
        # Mock predictors
        mock_predictor1 = MagicMock()
        mock_predictor2 = MagicMock()
        
        # Return different values for different batch calls
        def side_effect_1(X):
            return np.random.rand(len(X), 3)
        def side_effect_2(X):
            return np.random.rand(len(X), 3)
        def side_effect_unc1(X):
            return np.random.rand(len(X), 3) * 0.1
        def side_effect_unc2(X):
            return np.random.rand(len(X), 3) * 0.1
        
        mock_predictor1.side_effect = side_effect_1
        mock_predictor2.side_effect = side_effect_2
        mock_predictor1.uncertainty.side_effect = side_effect_unc1
        mock_predictor2.uncertainty.side_effect = side_effect_unc2
        
        de.function_predictor1 = mock_predictor1
        de.function_predictor2 = mock_predictor2
        
        X_test = np.random.rand(5, 4)  # 5 cells, should require multiple batches
        
        with patch('kompot.differential.differential_expression.compute_mahalanobis_distances') as mock_mahal:
            mock_mahal.return_value = np.array([0.2, 0.4, 0.6])  # 3 genes
            
            results = de.predict(X_test, progress=False)
            
            assert results['log_fold_change'].shape == (5, 3)  # 5 cells, 3 genes
            assert results['mahalanobis_distances'].shape == (3,)  # 3 genes