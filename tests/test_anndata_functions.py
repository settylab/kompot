"""Tests for the anndata integration functions."""

import numpy as np
import pytest
import datetime

from kompot.anndata.functions import compute_differential_abundance, compute_differential_expression, run_differential_analysis


def create_test_anndata(n_cells=100, n_genes=20):
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
    import pandas as pd
    obs = pd.DataFrame({
        'group': groups
    })
    
    # Create var_names
    var_names = [f'gene_{i}' for i in range(n_genes)]
    
    # Create var DataFrame with var_names as index
    var = pd.DataFrame(index=var_names)
    
    return anndata.AnnData(X=X, obs=obs, var=var, obsm=obsm)


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
        
        # Check that run_info was created
        assert 'run1' in self.adata.uns
        assert 'run_info' in self.adata.uns['run1']
        assert 'run_history' not in self.adata.uns['run1']
        
        # Make sure the run_info has the required fields
        run_info = self.adata.uns['run1']['run_info']
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
        
        # Check that run_history was created with the first run
        assert 'run_history' in self.adata.uns['run1']
        assert len(self.adata.uns['run1']['run_history']) == 1
        
        # Check that the history entry has the same structure as run_info
        history_entry = self.adata.uns['run1']['run_history'][0]
        assert 'timestamp' in history_entry
        assert 'function' in history_entry
        assert history_entry['function'] == 'compute_differential_abundance'
        
        # Run with a new key
        compute_differential_abundance(
            self.adata,
            groupby='group',
            condition1='A',
            condition2='B',
            result_key='run2'
        )
        
        # Check that run_info was created for the new key
        assert 'run2' in self.adata.uns
        assert 'run_info' in self.adata.uns['run2']
        
        # The individual DA function calls don't register in the run_history
        # Since we only test unit functionality here, we'll skip this check
        # Run history is tested in the test_global_run_history method
        
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
        
        # Check that run_info was created
        assert 'de_run1' in self.adata.uns
        assert 'run_info' in self.adata.uns['de_run1']
        assert 'run_history' not in self.adata.uns['de_run1']
        
        # Make sure the run_info has the required fields
        run_info = self.adata.uns['de_run1']['run_info']
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
        
        # Check that run_history was created with the first run
        assert 'run_history' in self.adata.uns['de_run1']
        assert len(self.adata.uns['de_run1']['run_history']) == 1
        
        # Check that the history entry has the same structure as run_info
        history_entry = self.adata.uns['de_run1']['run_history'][0]
        assert 'timestamp' in history_entry
        assert 'function' in history_entry
        assert history_entry['function'] == 'compute_differential_expression'

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