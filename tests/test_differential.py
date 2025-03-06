"""Tests for the differential module."""

import numpy as np
import pytest
from kompot.differential import DifferentialAbundance, DifferentialExpression


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
    assert 'mean_log_fold_change' in predictions
    
    # Check shapes
    expected_shape = X_condition1.shape[0] + X_condition2.shape[0]
    assert len(predictions['log_density_condition1']) == expected_shape
    assert len(predictions['log_density_condition2']) == expected_shape
    assert len(predictions['log_fold_change']) == expected_shape
    
    # Check values in range
    assert np.isfinite(predictions['log_fold_change']).all()
    assert np.isfinite(predictions['log_fold_change_zscore']).all()
    assert np.all(predictions['log_fold_change_direction'] != '')
    
    # Check that class attributes are updated for backward compatibility
    assert diff_abundance.log_density_condition1 is not None
    assert diff_abundance.log_density_condition2 is not None
    assert diff_abundance.log_fold_change is not None


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


def test_differential_expression_fit():
    """Test fitting the DifferentialExpression class."""
    # Generate sample data
    X_condition1 = np.random.randn(100, 5)
    y_condition1 = np.random.randn(100, 10)
    X_condition2 = np.random.randn(100, 5) + 0.5
    y_condition2 = np.random.randn(100, 10) + 1.0
    
    # Initialize and fit (explicitly enable weighted fold change)
    diff_expression = DifferentialExpression(compute_weighted_fold_change=True)
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
    assert 'weighted_mean_log_fold_change' in predictions
    
    # Check shapes
    expected_rows = X_condition1.shape[0] + X_condition2.shape[0]
    expected_cols = y_condition1.shape[1]
    assert predictions['condition1_imputed'].shape == (expected_rows, expected_cols)
    assert predictions['fold_change'].shape == (expected_rows, expected_cols)
    assert len(predictions['mahalanobis_distances']) == expected_cols
    
    # Check values
    assert np.isfinite(predictions['fold_change']).all()
    assert np.isfinite(predictions['mahalanobis_distances']).all()
    assert np.isfinite(predictions['weighted_mean_log_fold_change']).all()
    
    # Check class attributes for backward compatibility
    assert diff_expression.condition1_imputed is not None
    assert diff_expression.condition2_imputed is not None
    assert diff_expression.fold_change is not None
    assert diff_expression.fold_change_zscores is not None
    assert diff_expression.mahalanobis_distances is not None
    assert diff_expression.mean_log_fold_change is not None
    assert diff_expression.weighted_mean_log_fold_change is not None
    
    # Test with disabled weighted fold change (must be explicitly disabled)
    diff_expression_no_weight = DifferentialExpression(compute_weighted_fold_change=False)
    diff_expression_no_weight.fit(X_condition1, y_condition1, X_condition2, y_condition2)
    
    # Run predict
    predictions_no_weight = diff_expression_no_weight.predict(X_combined)
    
    # Should not have weighted fold change
    assert 'weighted_mean_log_fold_change' not in predictions_no_weight
    
    # Test with pre-computed differential abundance
    diff_abundance = DifferentialAbundance()
    diff_abundance.fit(X_condition1, X_condition2)
    
    # Get density predictions
    abundance_predictions = diff_abundance.predict(X_combined)
    
    diff_expression_precomputed = DifferentialExpression(
        compute_weighted_fold_change=True,  # Now need to explicitly enable it
        differential_abundance=diff_abundance
    )
    diff_expression_precomputed.fit(X_condition1, y_condition1, X_condition2, y_condition2)
    
    # Run predict
    predictions_precomputed = diff_expression_precomputed.predict(X_combined)
    
    # Should have weighted fold change using precomputed density
    assert 'weighted_mean_log_fold_change' in predictions_precomputed
    
    # Test with pre-computed densities
    precomputed_densities = {
        'log_density_condition1': abundance_predictions['log_density_condition1'],
        'log_density_condition2': abundance_predictions['log_density_condition2']
    }
    
    diff_expression_custom_density = DifferentialExpression(
        compute_weighted_fold_change=True,  # Now need to explicitly enable it
        precomputed_densities=precomputed_densities
    )
    
    diff_expression_custom_density.fit(
        X_condition1, y_condition1, X_condition2, y_condition2,
        compute_differential_abundance=False  # Explicitly disable computing differential abundance
    )
    
    # Run predict
    predictions_custom_density = diff_expression_custom_density.predict(X_combined)
    
    # Should have weighted fold change using custom density
    assert 'weighted_mean_log_fold_change' in predictions_custom_density
    
    # Test with pre-computed density predictors
    diff_expression_with_predictors = DifferentialExpression(
        compute_weighted_fold_change=True,  # Now need to explicitly enable it
        density_predictor1=diff_abundance.density_predictor1,
        density_predictor2=diff_abundance.density_predictor2
    )
    
    diff_expression_with_predictors.fit(
        X_condition1, y_condition1, X_condition2, y_condition2
    )
    
    # Run predict
    predictions_with_predictors = diff_expression_with_predictors.predict(X_combined)
    
    # Should have weighted fold change using density predictors
    assert 'weighted_mean_log_fold_change' in predictions_with_predictors
    

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
    diff_expression = DifferentialExpression(compute_weighted_fold_change=True)
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
    
    # Test with density predictions
    diff_abundance = DifferentialAbundance()
    diff_abundance.fit(X_condition1, X_condition2)
    density_predictions = diff_abundance.predict(X_new)
    
    # Test explicit density predictions parameter
    predictions_with_density = diff_expression.predict(X_new, density_predictions=density_predictions)
    assert 'weighted_mean_log_fold_change' in predictions_with_density
    
    # Test without weighted fold change computation
    diff_expression_no_weight = DifferentialExpression(compute_weighted_fold_change=False)
    diff_expression_no_weight.fit(X_condition1, y_condition1, X_condition2, y_condition2)
    
    predictions_no_weight = diff_expression_no_weight.predict(X_new)
    
    # Should not include weighted fold change even when density predictions are provided
    assert 'weighted_mean_log_fold_change' not in predictions_no_weight
    
    # Test with precomputed density predictors
    diff_expression_with_predictors = DifferentialExpression(
        compute_weighted_fold_change=True,
        density_predictor1=diff_abundance.density_predictor1,
        density_predictor2=diff_abundance.density_predictor2
    )
    diff_expression_with_predictors.fit(X_condition1, y_condition1, X_condition2, y_condition2)
    
    # Predict using the built-in density predictors
    predictions_with_predictors = diff_expression_with_predictors.predict(X_new)
    
    # Should have weighted_mean_log_fold_change using density predictors
    assert 'weighted_mean_log_fold_change' in predictions_with_predictors
    assert np.isfinite(predictions_with_predictors['weighted_mean_log_fold_change']).all()
    
    # Test with precomputed function predictors
    # First train a model to get function predictors
    diff_expression_for_predictors = DifferentialExpression()
    diff_expression_for_predictors.fit(X_condition1, y_condition1, X_condition2, y_condition2)
    
    # Create a new model using the function predictors from the first model
    diff_expression_with_function_predictors = DifferentialExpression(
        function_predictor1=diff_expression_for_predictors.function_predictor1,
        function_predictor2=diff_expression_for_predictors.function_predictor2,
        compute_weighted_fold_change=True,
        density_predictor1=diff_abundance.density_predictor1,
        density_predictor2=diff_abundance.density_predictor2
    )
    
    # Just fit to initialize things, no need to recompute parameters
    diff_expression_with_function_predictors.fit(X_condition1, y_condition1, X_condition2, y_condition2)
    
    # Predict using precomputed function predictors
    predictions_function_predictors = diff_expression_with_function_predictors.predict(X_new)
    
    # Should include all metrics
    assert 'condition1_imputed' in predictions_function_predictors
    assert 'condition2_imputed' in predictions_function_predictors
    assert 'fold_change' in predictions_function_predictors
    assert 'fold_change_zscores' in predictions_function_predictors
    assert 'mean_log_fold_change' in predictions_function_predictors
    assert 'weighted_mean_log_fold_change' in predictions_function_predictors
    
    # Test with cell condition labels
    cell_condition_labels = np.array([0] * 25 + [1] * 25)  # Half condition1, half condition2
    predictions_with_labels = diff_expression.predict(X_new, cell_condition_labels=cell_condition_labels)
    
    # Should include condition-specific metrics
    assert 'condition1_cells_mean_log_fold_change' in predictions_with_labels
    assert 'condition2_cells_mean_log_fold_change' in predictions_with_labels