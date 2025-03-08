"""Tests for the differential module."""

import numpy as np
import pytest
from kompot.differential import DifferentialAbundance, DifferentialExpression, compute_weighted_mean_fold_change


def test_differential_abundance_fit():
    """Test fitting the DifferentialAbundance class."""
    # Generate some sample data
    X_condition1 = np.random.randn(100, 5)
    X_condition2 = np.random.randn(100, 5) + 0.5
    
    # Initialize and fit
    diff_abundance = DifferentialAbundance()
    diff_abundance.fit(X_condition1, X_condition2)
    
    # Check that the predictors are properly set
    assert diff_abundance.density_predictor1 is not None
    assert diff_abundance.density_predictor2 is not None
    
    # Now generate predictions for the combined data
    X_combined = np.vstack([X_condition1, X_condition2])
    predictions = diff_abundance.predict(X_combined)
    
    # Check that predictions dictionary contains expected keys
    assert 'log_density_condition1' in predictions
    assert 'log_density_condition2' in predictions
    assert 'log_fold_change' in predictions
    assert 'log_fold_change_uncertainty' in predictions
    assert 'log_fold_change_zscore' in predictions
    assert 'log_fold_change_pvalue' in predictions
    assert 'log_fold_change_direction' in predictions
    # mean_log_fold_change has been removed from the output
    
    # Check shapes
    expected_shape = X_condition1.shape[0] + X_condition2.shape[0]
    assert len(predictions['log_density_condition1']) == expected_shape
    assert len(predictions['log_density_condition2']) == expected_shape
    assert len(predictions['log_fold_change']) == expected_shape
    
    # Check values in range
    assert np.isfinite(predictions['log_fold_change']).all()
    assert np.isfinite(predictions['log_fold_change_zscore']).all()
    assert np.all(predictions['log_fold_change_direction'] != '')
    
    # No longer checking for class attributes - they're not updated in the new version


def test_differential_abundance_predict():
    """Test predicting with the DifferentialAbundance class."""
    # Generate sample data
    X_condition1 = np.random.randn(100, 5)
    X_condition2 = np.random.randn(100, 5) + 0.5
    
    # New points to predict
    X_new = np.random.randn(50, 5)
    
    # Fit and predict
    diff_abundance = DifferentialAbundance()
    diff_abundance.fit(X_condition1, X_condition2)
    predictions = diff_abundance.predict(X_new)
    
    # Check the output structure
    assert isinstance(predictions, dict)
    assert 'log_density_condition1' in predictions
    assert 'log_density_condition2' in predictions
    assert 'log_fold_change' in predictions
    
    # Check shapes
    assert len(predictions['log_density_condition1']) == len(X_new)
    assert len(predictions['log_fold_change']) == len(X_new)
    
    # Check values
    assert np.isfinite(predictions['log_density_condition1']).all()
    assert np.isfinite(predictions['log_fold_change']).all()


def test_differential_abundance_predict_with_thresholds():
    """Test predicting with custom thresholds in the DifferentialAbundance class."""
    # Generate sample data with clear differences
    np.random.seed(42)  # Set seed for reproducibility
    X_condition1 = np.random.randn(100, 5)
    X_condition2 = np.random.randn(100, 5) + 1.0  # Clear shift to create differences
    
    # New points to predict
    X_new = np.random.randn(50, 5)
    
    # Fit the model
    diff_abundance = DifferentialAbundance(log_fold_change_threshold=1.0, pvalue_threshold=0.05)
    diff_abundance.fit(X_condition1, X_condition2)
    
    # Predict with default thresholds (from initialization)
    default_predictions = diff_abundance.predict(X_new)
    
    # Predict with stricter thresholds
    strict_predictions = diff_abundance.predict(
        X_new, 
        log_fold_change_threshold=2.0,  # Higher threshold means fewer 'up'/'down' calls
        pvalue_threshold=0.01  # Lower p-value means stricter significance test
    )
    
    # Predict with looser thresholds
    loose_predictions = diff_abundance.predict(
        X_new, 
        log_fold_change_threshold=0.5,  # Lower threshold means more 'up'/'down' calls
        pvalue_threshold=0.1  # Higher p-value means more relaxed significance test
    )
    
    # Check that the basic metrics are identical across all predictions
    np.testing.assert_array_equal(
        default_predictions['log_fold_change'],
        strict_predictions['log_fold_change']
    )
    np.testing.assert_array_equal(
        default_predictions['log_fold_change'],
        loose_predictions['log_fold_change']
    )
    
    # Check that direction classifications are different due to thresholds
    # Count number of non-neutral classifications in each case
    default_significant = np.sum(default_predictions['log_fold_change_direction'] != 'neutral')
    strict_significant = np.sum(strict_predictions['log_fold_change_direction'] != 'neutral')
    loose_significant = np.sum(loose_predictions['log_fold_change_direction'] != 'neutral')
    
    # Stricter thresholds should result in fewer (or equal) significant results
    assert strict_significant <= default_significant
    # Looser thresholds should result in more (or equal) significant results
    assert loose_significant >= default_significant


def test_differential_expression_fit():
    """Test fitting the DifferentialExpression class."""
    # Generate sample data
    X_condition1 = np.random.randn(100, 5)
    y_condition1 = np.random.randn(100, 10)
    X_condition2 = np.random.randn(100, 5) + 0.5
    y_condition2 = np.random.randn(100, 10) + 1.0
    
    # Initialize and fit
    diff_expression = DifferentialExpression()
    diff_expression.fit(X_condition1, y_condition1, X_condition2, y_condition2)
    
    # Check that predictors are properly set
    assert diff_expression.function_predictor1 is not None
    assert diff_expression.function_predictor2 is not None
    
    # Now run prediction to get fold changes and metrics
    X_combined = np.vstack([X_condition1, X_condition2])
    predictions = diff_expression.predict(X_combined, compute_mahalanobis=True)
    
    # Check predictions dictionary contains expected keys
    assert 'condition1_imputed' in predictions
    assert 'condition2_imputed' in predictions
    assert 'fold_change' in predictions
    assert 'fold_change_zscores' in predictions
    assert 'mean_log_fold_change' in predictions
    assert 'mahalanobis_distances' in predictions
    # weighted_mean_log_fold_change is no longer automatically computed
    # it requires explicit density_predictions to be provided
    
    # Check shapes
    expected_rows = X_condition1.shape[0] + X_condition2.shape[0]
    expected_cols = y_condition1.shape[1]
    assert predictions['condition1_imputed'].shape == (expected_rows, expected_cols)
    assert predictions['fold_change'].shape == (expected_rows, expected_cols)
    assert len(predictions['mahalanobis_distances']) == expected_cols
    
    # Check values
    assert np.isfinite(predictions['fold_change']).all()
    assert np.isfinite(predictions['mahalanobis_distances']).all()
    # weighted_mean_log_fold_change is no longer automatically included
    
    # Class attributes for backward compatibility have been removed
    
    # Run another test to verify basic functionality
    another_diff_expression = DifferentialExpression()
    another_diff_expression.fit(X_condition1, y_condition1, X_condition2, y_condition2)
    
    # Run predict
    another_predictions = another_diff_expression.predict(X_combined)
    
    # Verify basic predictions match
    assert np.allclose(predictions['mean_log_fold_change'], another_predictions['mean_log_fold_change'])
    

def test_differential_expression_predict():
    """Test predicting with the DifferentialExpression class."""
    # Generate sample data
    X_condition1 = np.random.randn(100, 5)
    y_condition1 = np.random.randn(100, 10)
    X_condition2 = np.random.randn(100, 5) + 0.5
    y_condition2 = np.random.randn(100, 10) + 1.0
    
    # New points to predict
    X_new = np.random.randn(50, 5)
    
    # Fit and predict
    diff_expression = DifferentialExpression()
    diff_expression.fit(X_condition1, y_condition1, X_condition2, y_condition2)
    
    # Test prediction on the combined dataset for Mahalanobis distances
    X_combined = np.vstack([X_condition1, X_condition2])
    combined_predictions = diff_expression.predict(X_combined, compute_mahalanobis=True)
    
    # Check that Mahalanobis distances are computed
    assert 'mahalanobis_distances' in combined_predictions
    assert len(combined_predictions['mahalanobis_distances']) == y_condition1.shape[1]
    
    # Predict on new points
    predictions = diff_expression.predict(X_new)
    
    # Check output structure
    assert isinstance(predictions, dict)
    assert 'condition1_imputed' in predictions
    assert 'condition2_imputed' in predictions
    assert 'fold_change' in predictions
    assert 'fold_change_zscores' in predictions
    
    # Check shapes
    assert predictions['condition1_imputed'].shape == (len(X_new), y_condition1.shape[1])
    assert predictions['fold_change'].shape == (len(X_new), y_condition1.shape[1])
    
    # Check values
    assert np.isfinite(predictions['condition1_imputed']).all()
    assert np.isfinite(predictions['fold_change']).all()
    assert np.isfinite(predictions['fold_change_zscores']).all()
    
    # Verify mean log fold change is included
    assert 'mean_log_fold_change' in predictions
    assert np.isfinite(predictions['mean_log_fold_change']).all()
    
    # Test with precomputed function predictors
    # First train a model to get function predictors
    diff_expression_for_predictors = DifferentialExpression()
    diff_expression_for_predictors.fit(X_condition1, y_condition1, X_condition2, y_condition2)
    
    # Create a new model using the function predictors from the first model
    diff_expression_with_function_predictors = DifferentialExpression(
        function_predictor1=diff_expression_for_predictors.function_predictor1,
        function_predictor2=diff_expression_for_predictors.function_predictor2
    )
    
    # Just fit to initialize things, no need to recompute parameters
    diff_expression_with_function_predictors.fit(X_condition1, y_condition1, X_condition2, y_condition2)
    
    # Predict using precomputed function predictors
    predictions_with_function_predictors = diff_expression_with_function_predictors.predict(X_new)
    
    # Should include all basic metrics
    assert 'condition1_imputed' in predictions_with_function_predictors
    assert 'condition2_imputed' in predictions_with_function_predictors
    assert 'fold_change' in predictions_with_function_predictors
    assert 'fold_change_zscores' in predictions_with_function_predictors
    assert 'mean_log_fold_change' in predictions_with_function_predictors
    
    # Test with cell condition labels
    cell_condition_labels = np.array([0] * 25 + [1] * 25)  # Half condition1, half condition2
    predictions_with_labels = diff_expression.predict(X_new, cell_condition_labels=cell_condition_labels)
    
    # Should include condition-specific metrics
    assert 'condition1_cells_mean_log_fold_change' in predictions_with_labels
    assert 'condition2_cells_mean_log_fold_change' in predictions_with_labels


def test_compute_weighted_mean_fold_change_standalone():
    """Test the standalone compute_weighted_mean_fold_change function."""
    # Generate sample data
    n_cells, n_genes = 100, 10
    fold_change = np.random.randn(n_cells, n_genes)
    log_density_condition1 = np.random.randn(n_cells)
    log_density_condition2 = np.random.randn(n_cells) + 0.5
    
    # Calculate using log_density_condition1 and log_density_condition2
    weighted_lfc = compute_weighted_mean_fold_change(
        fold_change, 
        log_density_condition1=log_density_condition1,
        log_density_condition2=log_density_condition2
    )
    
    # Check results
    assert weighted_lfc.shape == (n_genes,)
    assert np.isfinite(weighted_lfc).all()
    
    # Calculate log_density_diff manually
    log_density_diff = log_density_condition2 - log_density_condition1
    
    # Calculate using pre-computed log_density_diff
    weighted_lfc_precomputed = compute_weighted_mean_fold_change(
        fold_change,
        log_density_diff=log_density_diff
    )
    
    # Both methods should give identical results
    np.testing.assert_allclose(weighted_lfc, weighted_lfc_precomputed)
    
    # Test with pandas Series
    try:
        import pandas as pd
        log_density_condition1_series = pd.Series(log_density_condition1)
        log_density_condition2_series = pd.Series(log_density_condition2)
        
        # Calculate using pandas Series
        weighted_lfc_series = compute_weighted_mean_fold_change(
            fold_change,
            log_density_condition1=log_density_condition1_series,
            log_density_condition2=log_density_condition2_series
        )
        
        # Results should be the same as with numpy arrays
        np.testing.assert_allclose(weighted_lfc, weighted_lfc_series)
        
        # Test with pandas Series for log_density_diff
        log_density_diff_series = pd.Series(log_density_diff)
        weighted_lfc_diff_series = compute_weighted_mean_fold_change(
            fold_change,
            log_density_diff=log_density_diff_series
        )
        
        # Results should be the same
        np.testing.assert_allclose(weighted_lfc, weighted_lfc_diff_series)
    except ImportError:
        # Skip pandas tests if pandas is not available
        pass
        
    # Test error case when neither log_density_diff nor both density conditions are provided
    with pytest.raises(ValueError):
        compute_weighted_mean_fold_change(fold_change)
        
    with pytest.raises(ValueError):
        compute_weighted_mean_fold_change(
            fold_change,
            log_density_condition1=log_density_condition1
        )
        
    with pytest.raises(ValueError):
        compute_weighted_mean_fold_change(
            fold_change,
            log_density_condition2=log_density_condition2
        )
        

def test_weighted_fold_change_standalone_detailed():
    """
    Test that the standalone weighted fold change function works correctly with different input formats.
    """
    # Create sample data
    n_cells, n_features, n_genes = 50, 5, 8
    X_condition1 = np.random.randn(n_cells, n_features)
    y_condition1 = np.random.randn(n_cells, n_genes)
    X_condition2 = np.random.randn(n_cells, n_features) + 0.5
    y_condition2 = np.random.randn(n_cells, n_genes) + 1.0
    
    # Compute fold change
    diff_expression = DifferentialExpression()
    diff_expression.fit(X_condition1, y_condition1, X_condition2, y_condition2)
    
    # Create a full dataset for prediction
    X_combined = np.vstack([X_condition1, X_condition2])
    expression_results = diff_expression.predict(X_combined)
    fold_change = expression_results['fold_change']
    
    # Compute log density for both conditions
    diff_abundance = DifferentialAbundance()
    diff_abundance.fit(X_condition1, X_condition2)
    abundance_results = diff_abundance.predict(X_combined)
    
    # Method 1: Using log_density_condition1 and log_density_condition2
    method1_weighted_lfc = compute_weighted_mean_fold_change(
        fold_change,
        log_density_condition1=abundance_results['log_density_condition1'],
        log_density_condition2=abundance_results['log_density_condition2']
    )
    
    # Method 2: Using pre-computed log_density_diff
    log_density_diff = abundance_results['log_density_condition2'] - abundance_results['log_density_condition1']
    method2_weighted_lfc = compute_weighted_mean_fold_change(
        fold_change,
        log_density_diff=log_density_diff
    )
    
    # Verify results from both methods are identical
    np.testing.assert_allclose(method1_weighted_lfc, method2_weighted_lfc)