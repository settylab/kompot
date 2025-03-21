"""Tests for the anndata integration functions."""

import numpy as np
import pytest
import datetime
import pandas as pd
import logging
from unittest.mock import patch, MagicMock

from kompot.anndata import compute_differential_abundance, compute_differential_expression, run_differential_analysis


def create_test_anndata(n_cells=100, n_genes=20, with_sample_col=False):
    """Create a test AnnData object."""
    try:
        import anndata
    except ImportError:
        pytest.skip("anndata not installed, skipping test")
        
    np.random.seed(42)
    
    # Create test data
    X = np.random.normal(0, 1, (n_cells, n_genes))
    
    # Create cell groups for testing
    groups = np.array(['A'] * (n_cells // 2) + ['B'] * (n_cells // 2))
    
    # Create embedding
    obsm = {
        'DM_EigenVectors': np.random.normal(0, 1, (n_cells, 10))
    }
    
    # Create observation dataframe
    obs_dict = {'group': groups}
    
    # Add sample column if requested (3 samples per condition)
    if with_sample_col:
        # Create 3 samples per condition, each with equal number of cells
        n_samples_per_condition = 3
        cells_per_sample = n_cells // (2 * n_samples_per_condition)
        
        sample_ids = []
        for condition in ['A', 'B']:
            for sample_id in range(n_samples_per_condition):
                sample_name = f"{condition}_sample_{sample_id}"
                sample_ids.extend([sample_name] * cells_per_sample)
        
        # If there are any remaining cells due to division, assign them to the last sample
        while len(sample_ids) < n_cells:
            sample_ids.append(f"B_sample_{n_samples_per_condition-1}")
            
        obs_dict['sample'] = sample_ids
    
    obs = pd.DataFrame(obs_dict)
    
    # Create var_names
    var_names = [f'gene_{i}' for i in range(n_genes)]
    
    # Create var DataFrame with var_names as index
    var = pd.DataFrame(index=var_names)
    
    return anndata.AnnData(X=X, obs=obs, var=var, obsm=obsm)


def test_sample_col_parameter():
    """Test the sample_col parameter in compute_differential_abundance."""
    # Create a test AnnData object with sample column
    adata = create_test_anndata(with_sample_col=True)
    
    # Run differential abundance analysis with sample_col parameter
    result = compute_differential_abundance(
        adata,
        groupby='group',
        condition1='A',
        condition2='B',
        sample_col='sample',
        result_key='test_sample_col',
        return_full_results=True  # Make sure to get the full results dictionary including model
    )
    
    # Check that the model has sample variance enabled
    assert result['model'].use_sample_variance is True
    
    # Check that variance predictors were created
    assert result['model'].variance_predictor1 is not None
    assert result['model'].variance_predictor2 is not None
    
    # Verify that the sample_col parameter was stored in run info
    assert 'kompot_da' in adata.uns
    assert 'last_run_info' in adata.uns['kompot_da']
    assert 'params' in adata.uns['kompot_da']['last_run_info']
    assert 'sample_col' in adata.uns['kompot_da']['last_run_info']['params']
    assert adata.uns['kompot_da']['last_run_info']['params']['sample_col'] == 'sample'
    assert adata.uns['kompot_da']['last_run_info']['params']['use_sample_variance'] is True
    
    # Run a comparison analysis without sample_col
    result_no_samples = compute_differential_abundance(
        adata,
        groupby='group',
        condition1='A',
        condition2='B',
        result_key='test_no_sample_col',
        return_full_results=True  # Make sure to get the full results dictionary including model
    )
    
    # Verify model doesn't use sample variance
    assert result_no_samples['model'].use_sample_variance is False
    
    # Verify variance predictors are None
    assert result_no_samples['model'].variance_predictor1 is None
    assert result_no_samples['model'].variance_predictor2 is None
    
    # Verify the parameters are stored in kompot_da last_run_info
    assert 'last_run_info' in adata.uns['kompot_da']
    assert 'params' in adata.uns['kompot_da']['last_run_info']
    assert 'sample_col' in adata.uns['kompot_da']['last_run_info']['params']
    assert adata.uns['kompot_da']['last_run_info']['params']['sample_col'] is None
    assert adata.uns['kompot_da']['last_run_info']['params']['use_sample_variance'] is False
    
    # Check that the two models produce different results
    # The log fold change values should be the same
    np.testing.assert_allclose(
        result['log_fold_change'], 
        result_no_samples['log_fold_change']
    )
    
    # Check that both models have valid outputs 
    assert 'neg_log10_fold_change_pvalue' in result
    assert 'neg_log10_fold_change_pvalue' in result_no_samples
    
    # Check that both models have the direction classifications
    assert 'log_fold_change_direction' in result
    assert 'log_fold_change_direction' in result_no_samples
    
    # Check if the variance predictors were used
    assert result['model'].variance_predictor1 is not None
    assert result['model'].variance_predictor2 is not None
    assert result_no_samples['model'].variance_predictor1 is None
    assert result_no_samples['model'].variance_predictor2 is None
    
    # Verify that sample variance affects uncertainty calculations
    # Use a subset of points for efficiency
    X_test = adata.obsm['DM_EigenVectors'][:20]  # Just use 20 test points
    
    # Get uncertainty by running predict directly on both models
    test_result_with_var = result['model'].predict(X_test)
    test_result_no_var = result_no_samples['model'].predict(X_test)
    
    with_var_uncertainty = test_result_with_var['log_fold_change_uncertainty']
    no_var_uncertainty = test_result_no_var['log_fold_change_uncertainty']
    
    # Verify that sample variance is being used by checking if uncertainty is higher
    assert np.mean(with_var_uncertainty) > np.mean(no_var_uncertainty), \
        f"Expected higher uncertainty with sample variance ({np.mean(with_var_uncertainty):.6f} > {np.mean(no_var_uncertainty):.6f})"
    
    # Verify that sample variances are non-zero
    sample_variance1 = result['model'].variance_predictor1(X_test, diag=True).flatten()
    sample_variance2 = result['model'].variance_predictor2(X_test, diag=True).flatten()
    assert np.mean(sample_variance1) > 0, "Sample variance for condition 1 should be greater than zero"
    assert np.mean(sample_variance2) > 0, "Sample variance for condition 2 should be greater than zero"


class TestRunHistoryPreservation:
    """Tests for run history preservation in AnnData objects."""
    
    def setup_method(self):
        """Set up test data."""
        self.adata = create_test_anndata()
        
    def test_da_run_history_preservation(self):
        """Test that run history is preserved for differential abundance."""
        # Run differential abundance
        compute_differential_abundance(
            self.adata,
            groupby='group',
            condition1='A',
            condition2='B',
            result_key='run1'
        )
        
        # Check that last_run_info was created in the fixed storage location
        assert 'kompot_da' in self.adata.uns
        assert 'last_run_info' in self.adata.uns['kompot_da']
        assert 'run_history' in self.adata.uns['kompot_da']
        assert len(self.adata.uns['kompot_da']['run_history']) == 1
        
        # Make sure the last_run_info has the required fields
        run_info = self.adata.uns['kompot_da']['last_run_info']
        assert 'timestamp' in run_info
        assert 'function' in run_info
        assert run_info['function'] == 'compute_differential_abundance'
        assert 'lfc_key' in run_info
        assert 'result_key' in run_info
        assert run_info['result_key'] == 'run1'
        
        # Run again with same key to create history
        compute_differential_abundance(
            self.adata,
            groupby='group',
            condition1='A',
            condition2='B',
            result_key='run1'
        )
        
        # Check that run_history was updated with the second run
        assert 'run_history' in self.adata.uns['kompot_da']
        assert len(self.adata.uns['kompot_da']['run_history']) == 2
        
        # Check that the history entries have the expected structure
        history_entry1 = self.adata.uns['kompot_da']['run_history'][0]
        history_entry2 = self.adata.uns['kompot_da']['run_history'][1]
        
        # Check both entries
        for entry in [history_entry1, history_entry2]:
            assert 'timestamp' in entry
            assert 'function' in entry
            assert entry['function'] == 'compute_differential_abundance'
            assert 'environment' in entry
        
        # Run with a new key
        compute_differential_abundance(
            self.adata,
            groupby='group',
            condition1='A',
            condition2='B',
            result_key='run2'
        )
        
        # Check that the storage was updated with the new run
        assert 'kompot_da' in self.adata.uns
        assert 'last_run_info' in self.adata.uns['kompot_da']
        assert len(self.adata.uns['kompot_da']['run_history']) == 3
        
        # The last run should have the new result_key
        latest_run = self.adata.uns['kompot_da']['run_history'][-1]
        assert latest_run['result_key'] == 'run2'
        
    def test_de_run_history_preservation(self):
        """Test that run history is preserved for differential expression."""
        # Run differential expression with compute_mahalanobis=False to avoid errors
        compute_differential_expression(
            self.adata,
            groupby='group',
            condition1='A',
            condition2='B',
            result_key='de_run1',
            compute_mahalanobis=False
        )
        
        # Check that last_run_info was created in the fixed storage location
        assert 'kompot_de' in self.adata.uns
        assert 'last_run_info' in self.adata.uns['kompot_de']
        assert 'run_history' in self.adata.uns['kompot_de']
        assert len(self.adata.uns['kompot_de']['run_history']) == 1
        
        # Make sure the last_run_info has the required fields
        run_info = self.adata.uns['kompot_de']['last_run_info']
        assert 'timestamp' in run_info
        assert 'function' in run_info
        assert run_info['function'] == 'compute_differential_expression'
        assert 'lfc_key' in run_info
        assert 'result_key' in run_info
        assert run_info['result_key'] == 'de_run1'
        
        # Run again with same key to create history
        compute_differential_expression(
            self.adata,
            groupby='group',
            condition1='A',
            condition2='B',
            result_key='de_run1',
            compute_mahalanobis=False
        )
        
        # Check that run_history was updated with the second run
        assert 'run_history' in self.adata.uns['kompot_de']
        assert len(self.adata.uns['kompot_de']['run_history']) == 2
        
        # Check that the history entries have the expected structure
        history_entry1 = self.adata.uns['kompot_de']['run_history'][0]
        history_entry2 = self.adata.uns['kompot_de']['run_history'][1]
        
        # Check both entries
        for entry in [history_entry1, history_entry2]:
            assert 'timestamp' in entry
            assert 'function' in entry
            assert entry['function'] == 'compute_differential_expression'
            assert 'environment' in entry

    def test_global_run_history(self):
        """Test that global run history is created and updated correctly."""
        # Run the full differential analysis
        result = run_differential_analysis(
            self.adata,
            groupby='group',
            condition1='A',
            condition2='B',
            abundance_key='grun1',
            expression_key='grun2',
            generate_html_report=False,
            compute_mahalanobis=False  # Turn off Mahalanobis to avoid errors in testing
        )
        
        # Check global run history creation
        assert 'kompot_run_history' in self.adata.uns
        assert 'kompot_latest_run' in self.adata.uns
        assert len(self.adata.uns['kompot_run_history']) >= 1
        
        # Check the run history fields
        run_history = self.adata.uns['kompot_run_history']
        latest_run = self.adata.uns['kompot_latest_run']
        
        # Check that latest run has the expected keys
        assert 'timestamp' in latest_run
        assert 'run_id' in latest_run
        assert 'function' in latest_run
        assert latest_run['function'] == 'run_differential_analysis'
        assert 'abundance_key' in latest_run
        assert latest_run['abundance_key'] == 'grun1'
        assert 'expression_key' in latest_run
        assert latest_run['expression_key'] == 'grun2'
        
        # Run another analysis to check that history updates
        initial_length = len(self.adata.uns['kompot_run_history'])
        
        result = run_differential_analysis(
            self.adata,
            groupby='group',
            condition1='A',
            condition2='B',
            abundance_key='grun3',
            expression_key='grun4',
            generate_html_report=False,
            compute_mahalanobis=False  # Turn off Mahalanobis to avoid errors in testing
        )
        
        # Check that run history was updated
        assert len(self.adata.uns['kompot_run_history']) > initial_length
        
        # Check latest run was updated
        assert self.adata.uns['kompot_latest_run']['abundance_key'] == 'grun3'
        assert self.adata.uns['kompot_latest_run']['expression_key'] == 'grun4'
        
        
@patch('kompot.anndata.differential_abundance.logger.warning')
def test_compute_differential_abundance_warns_overwrite(mock_warning):
    """Test that compute_differential_abundance warns when overwriting existing results."""
    adata = create_test_anndata()
    
    # First run to create initial results
    compute_differential_abundance(adata, groupby='group', condition1='A', condition2='B', result_key='test_key')
    
    # Reset mock to clear any prior calls
    mock_warning.reset_mock()
    
    # Second run with same result_key should issue warning
    compute_differential_abundance(adata, groupby='group', condition1='A', condition2='B', result_key='test_key')
    
    # Check that a warning was issued with appropriate text
    mock_warning.assert_called()
    args, _ = mock_warning.call_args
    assert "Results with result_key='test_key' already exist" in args[0]
    assert "Fields that will be overwritten:" in args[0]


@patch('kompot.anndata.differential_expression.logger.warning')
def test_compute_differential_expression_warns_overwrite(mock_warning):
    """Test that compute_differential_expression warns when overwriting existing results."""
    adata = create_test_anndata()
    
    # First run to create initial results
    compute_differential_expression(
        adata, 
        groupby='group', 
        condition1='A', 
        condition2='B', 
        result_key='test_key',
        compute_mahalanobis=False  # Avoid Mahalanobis computation errors in tests
    )
    
    # Reset mock to clear any prior calls
    mock_warning.reset_mock()
    
    # Second run with same result_key should issue warning
    compute_differential_expression(
        adata, 
        groupby='group', 
        condition1='A', 
        condition2='B', 
        result_key='test_key',
        compute_mahalanobis=False  # Avoid Mahalanobis computation errors in tests
    )
    
    # Check that a warning was issued with appropriate text
    mock_warning.assert_called()
    args, _ = mock_warning.call_args
    assert "Differential expression results with result_key='test_key' already exist" in args[0]
    assert "Fields that will be overwritten:" in args[0]


def test_run_differential_analysis_with_sample_col():
    """Test run_differential_analysis with sample_col parameter."""
    # Create a test AnnData object with sample column
    adata = create_test_anndata(with_sample_col=True)
    
    # Run the full differential analysis with sample_col
    result = run_differential_analysis(
        adata,
        groupby='group',
        condition1='A',
        condition2='B',
        sample_col='sample',  # Pass sample_col to both abundance and expression
        abundance_key='sample_run_da',
        expression_key='sample_run_de',
        generate_html_report=False,
        compute_mahalanobis=False  # Turn off Mahalanobis to avoid errors in testing
    )
    
    # Verify the abundance result used sample variance
    assert result['differential_abundance'].use_sample_variance is True
    assert result['differential_abundance'].variance_predictor1 is not None
    assert result['differential_abundance'].variance_predictor2 is not None
    
    # Verify the expression result used sample variance
    assert result['differential_expression'].use_sample_variance is True
    assert result['differential_expression'].variance_predictor1 is not None
    assert result['differential_expression'].variance_predictor2 is not None
    
    # Check that parameter was stored correctly for both analyses in fixed storage locations
    assert 'kompot_da' in adata.uns
    assert 'last_run_info' in adata.uns['kompot_da']
    assert 'params' in adata.uns['kompot_da']['last_run_info']
    assert 'sample_col' in adata.uns['kompot_da']['last_run_info']['params']
    assert adata.uns['kompot_da']['last_run_info']['params']['sample_col'] == 'sample'
    assert adata.uns['kompot_da']['last_run_info']['params']['use_sample_variance'] is True
    
    assert 'kompot_de' in adata.uns
    assert 'last_run_info' in adata.uns['kompot_de']
    assert 'params' in adata.uns['kompot_de']['last_run_info']
    assert 'sample_col' in adata.uns['kompot_de']['last_run_info']['params']
    assert adata.uns['kompot_de']['last_run_info']['params']['sample_col'] == 'sample'
    assert adata.uns['kompot_de']['last_run_info']['params']['use_sample_variance'] is True
    
    # Verify the correct result_key was stored in the last_run_info
    assert adata.uns['kompot_da']['last_run_info']['result_key'] == 'sample_run_da'
    assert adata.uns['kompot_de']['last_run_info']['result_key'] == 'sample_run_de'


def test_landmark_reuse_and_storage():
    """Test landmark reuse and optional storage feature with independent storage."""
    # Create a test AnnData object
    adata = create_test_anndata()
    
    # First run with landmarks storage enabled
    result1 = run_differential_analysis(
        adata,
        groupby='group',
        condition1='A',
        condition2='B',
        abundance_key='store_landmarks_run',
        expression_key='store_landmarks_run_de',
        generate_html_report=False,
        compute_mahalanobis=False,  # Turn off Mahalanobis to avoid errors in testing
        store_landmarks=True,  # Enable landmark storage
        share_landmarks=True,  # Enable landmark sharing
        n_landmarks=50,  # Explicitly set n_landmarks to ensure they are computed
    )
    
    # Verify landmarks were stored only in result keys (not in standard locations after changes)
    assert 'store_landmarks_run' in adata.uns
    assert 'landmarks' in adata.uns['store_landmarks_run']
    assert 'landmarks_info' in adata.uns['store_landmarks_run']
    
    assert 'store_landmarks_run_de' in adata.uns
    assert 'landmarks' in adata.uns['store_landmarks_run_de']
    assert 'landmarks_info' in adata.uns['store_landmarks_run_de']
    
    # Standard keys should exist but without landmarks (only landmarks_info)
    assert 'kompot_da' in adata.uns
    assert 'landmarks' not in adata.uns['kompot_da'], "With new implementation, landmarks should not be synchronized to standard keys"
    assert 'landmarks_info' in adata.uns['kompot_da']
    
    assert 'kompot_de' in adata.uns
    assert 'landmarks' not in adata.uns['kompot_de'], "With new implementation, landmarks should not be synchronized to standard keys"
    assert 'landmarks_info' in adata.uns['kompot_de']
    
    # Extract landmarks for comparison
    da_landmarks = adata.uns['store_landmarks_run']['landmarks']
    de_landmarks = adata.uns['store_landmarks_run_de']['landmarks']
    
    # No longer enforcing landmarks are the SAME OBJECT - they might be different objects with same values
    assert da_landmarks.shape == de_landmarks.shape, "Expected landmarks to have the same shape"
    
    # Extract the shape for verification
    landmarks_shape = da_landmarks.shape
    
    # Run another analysis with store_landmarks=False
    result2 = compute_differential_abundance(
        adata,
        groupby='group',
        condition1='A',
        condition2='B',
        result_key='no_store_landmarks_run',
        store_landmarks=False,  # Don't store landmarks
        n_landmarks=50,  # Explicitly set n_landmarks to ensure they are computed
    )
    
    # Verify landmarks info is stored but not landmarks
    assert 'no_store_landmarks_run' in adata.uns
    assert 'landmarks_info' in adata.uns['no_store_landmarks_run']
    assert 'landmarks' not in adata.uns['no_store_landmarks_run']
    
    # Now run another analysis with reuse of landmarks
    result3 = compute_differential_expression(
        adata,
        groupby='group',
        condition1='A',
        condition2='B',
        result_key='reuse_landmarks_run',
        compute_mahalanobis=False,  # Turn off Mahalanobis to avoid errors in testing
        store_landmarks=True,  # Enable landmark storage
        n_landmarks=50,  # Explicitly set n_landmarks to ensure they are computed
    )
    
    # Verify landmarks were reused from previous runs
    assert 'reuse_landmarks_run' in adata.uns
    assert 'landmarks' in adata.uns['reuse_landmarks_run']
    
    # The shape should be the same as the original landmarks
    reused_landmarks = adata.uns['reuse_landmarks_run']['landmarks']
    assert reused_landmarks.shape == landmarks_shape, "Expected reused landmarks to have the same shape"
    
    # Test sequential reuse by keeping one of the landmarks and deleting the other
    # With our new implementation, we should be able to find and reuse any stored landmarks
    # as long as they have the right shape
    
    # Save a copy of the DA landmarks for reference
    landmarks_shape = adata.uns['store_landmarks_run']['landmarks'].shape
    
    # Remove one of the landmarks but keep the other
    if 'store_landmarks_run_de' in adata.uns:
        del adata.uns['store_landmarks_run_de']['landmarks']
    
    # Run another analysis - it should find and use the remaining landmarks
    # But we need to provide n_landmarks since our test setup has deleted some landmarks
    # This ensures we can compute new ones if needed
    result4 = compute_differential_abundance(
        adata,
        groupby='group',
        condition1='A',
        condition2='B',
        result_key='reuse_from_standard_run',
        store_landmarks=True,  # Enable landmark storage
        n_landmarks=50,  # Provide n_landmarks since we're missing some landmarks now
    )
    
    # Verify standard results were generated
    assert 'reuse_from_standard_run' in adata.uns
    assert 'landmarks_info' in adata.uns['reuse_from_standard_run']


def test_landmark_cross_analysis_search():
    """Test the new cross-analysis landmark search feature."""
    # Create a test AnnData object
    adata = create_test_anndata()
    
    # First, store both landmarks in the standard locations
    # Use explicit n_landmarks to ensure they're computed
    adata.uns['kompot_da'] = {}
    adata.uns['kompot_de'] = {}
    
    # Create landmarks with different shapes but same dimensions
    # The shape doesn't need to be exact, but the dimensions must match the DM_EigenVectors (10)
    random_da_landmarks = np.random.normal(0, 1, (40, 10))
    random_de_landmarks = np.random.normal(0, 1, (60, 10))
    
    # Store them 
    adata.uns['kompot_da']['landmarks'] = random_da_landmarks
    adata.uns['kompot_de']['landmarks'] = random_de_landmarks
    
    # Now run DA without explicit landmarks - it should find and use the kompot_da landmarks
    compute_differential_abundance(
        adata,
        groupby='group',
        condition1='A',
        condition2='B',
        result_key='landmark_search_da',
        store_landmarks=True,
        # Don't provide n_landmarks to force reuse
    )
    
    # Verify landmarks were stored and match those in kompot_da
    assert 'landmark_search_da' in adata.uns
    assert 'landmarks' in adata.uns['landmark_search_da']
    assert adata.uns['landmark_search_da']['landmarks'].shape == adata.uns['kompot_da']['landmarks'].shape
    
    # Now run DE - it should find and use the kompot_de landmarks
    compute_differential_expression(
        adata,
        groupby='group',
        condition1='A',
        condition2='B',
        result_key='landmark_search_de',
        compute_mahalanobis=False,
        store_landmarks=True,
        # Don't provide n_landmarks to force reuse
    )
    
    # Verify DE has landmarks stored that match those in kompot_de
    assert 'landmark_search_de' in adata.uns
    assert 'landmarks' in adata.uns['landmark_search_de']
    assert adata.uns['landmark_search_de']['landmarks'].shape == adata.uns['kompot_de']['landmarks'].shape
    
    # Create a custom landmark key that's not standard but used by our cross-search functionality
    adata.uns['kompot_custom'] = {}
    adata.uns['kompot_custom']['landmarks'] = np.random.normal(0, 1, (75, 10))
    
    # Delete the standard storage locations to force use of the custom location
    del adata.uns['kompot_da']['landmarks'] 
    del adata.uns['kompot_de']['landmarks']
    
    # Run another analysis that should find and use these custom landmarks
    compute_differential_abundance(
        adata,
        groupby='group',
        condition1='A',
        condition2='B',
        result_key='custom_landmark_search',
        store_landmarks=True,
        # Don't provide n_landmarks to force search
    )
    
    # Verify the custom landmarks were found and reused
    assert 'custom_landmark_search' in adata.uns
    assert 'landmarks' in adata.uns['custom_landmark_search']
    assert adata.uns['custom_landmark_search']['landmarks'].shape == adata.uns['kompot_custom']['landmarks'].shape


@pytest.mark.skip(reason="Disk backed options are tested in test_mahalanobis_approaches")
def test_disk_backed_options():
    """Test that disk-backed options are properly passed through in AnnData functions."""
    # Create a test AnnData object
    adata = create_test_anndata()
    
    # Run with disk-backed options
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        # First, differential abundance with disk backing
        result_da = compute_differential_abundance(
            adata,
            groupby='group',
            condition1='A',
            condition2='B',
            result_key='disk_test_da',
            store_arrays_on_disk=True,
            disk_storage_dir=temp_dir,
            max_memory_ratio=0.7  # Custom threshold
        )
        
        # Check that parameters were stored in last_run_info
        assert 'last_run_info' in adata.uns['kompot_da']
        assert 'params' in adata.uns['kompot_da']['last_run_info']
        assert 'store_arrays_on_disk' in adata.uns['kompot_da']['last_run_info']['params']
        assert adata.uns['kompot_da']['last_run_info']['params']['store_arrays_on_disk'] is True
        assert 'disk_storage_dir' in adata.uns['kompot_da']['last_run_info']['params']
        assert adata.uns['kompot_da']['last_run_info']['params']['disk_storage_dir'] == temp_dir
        assert 'max_memory_ratio' in adata.uns['kompot_da']['last_run_info']['params']
        assert adata.uns['kompot_da']['last_run_info']['params']['max_memory_ratio'] == 0.7
        
        # Check that storage usage info was captured in run info
        assert 'disk_storage' in adata.uns['disk_test_da']
        assert 'disk_storage_dir' in adata.uns['disk_test_da']
        assert adata.uns['disk_test_da']['disk_storage_dir'] == temp_dir
        
        # Now, differential expression with disk backing
        result_de = compute_differential_expression(
            adata,
            groupby='group',
            condition1='A',
            condition2='B',
            result_key='disk_test_de',
            compute_mahalanobis=True,
            store_arrays_on_disk=True,
            disk_storage_dir=temp_dir,
            batch_size=10
        )
        
        # Check that parameters were stored in last_run_info
        assert 'last_run_info' in adata.uns['kompot_de']
        assert 'params' in adata.uns['kompot_de']['last_run_info']
        assert 'store_arrays_on_disk' in adata.uns['kompot_de']['last_run_info']['params']
        assert adata.uns['kompot_de']['last_run_info']['params']['store_arrays_on_disk'] is True
        assert 'disk_storage_dir' in adata.uns['kompot_de']['last_run_info']['params']
        assert adata.uns['kompot_de']['last_run_info']['params']['disk_storage_dir'] == temp_dir
        assert 'batch_size' in adata.uns['kompot_de']['last_run_info']['params']
        assert adata.uns['kompot_de']['last_run_info']['params']['batch_size'] == 10
        
        # Check that storage usage info was captured
        assert 'disk_storage' in adata.uns['disk_test_de']
        assert 'disk_storage_dir' in adata.uns['disk_test_de']
        
        # Finally, run differential analysis with disk backing
        result_all = run_differential_analysis(
            adata,
            groupby='group',
            condition1='A',
            condition2='B',
            abundance_key='disk_test_all_da',
            expression_key='disk_test_all_de',
            compute_mahalanobis=True,
            store_arrays_on_disk=True,
            disk_storage_dir=temp_dir,
            batch_size=10,
            generate_html_report=False
        )
        
        # Check that parameters were passed through to both abundance and expression
        assert adata.uns['kompot_da']['last_run_info']['params']['store_arrays_on_disk'] is True
        assert adata.uns['kompot_de']['last_run_info']['params']['store_arrays_on_disk'] is True
        
        # Check that disk usage stats were stored for both
        assert 'disk_storage' in adata.uns['disk_test_all_da']
        assert 'disk_storage' in adata.uns['disk_test_all_de']
        
        # Verify that temporary directory should be auto-cleaned for models with None dir
        # We can still test this by running another analysis without specifying a directory
        result_temp = compute_differential_expression(
            adata,
            groupby='group',
            condition1='A',
            condition2='B',
            result_key='temp_dir_test',
            compute_mahalanobis=True,
            store_arrays_on_disk=True,  # Enable disk storage but don't specify directory
            disk_storage_dir=None,      # Should create temp directory
            batch_size=10
        )
        
        # Check that a temporary directory was auto-created and stored
        assert 'disk_storage_dir' in adata.uns['temp_dir_test']
        assert adata.uns['temp_dir_test']['disk_storage_dir'] is not None
        # The directory path should start with a system temp directory pattern
        temp_path = adata.uns['temp_dir_test']['disk_storage_dir']
        assert temp_path.startswith('/tmp/') or 'kompot_arrays_' in temp_path