"""Tests for AnnData field tracking in differential analysis functions."""

import numpy as np
import pytest
import pandas as pd
import anndata
import logging

from kompot.anndata.differential_expression import compute_differential_expression
from kompot.anndata.differential_abundance import compute_differential_abundance


class TestAnnDataFieldTracking:
    """Test the AnnData field tracking functionality in differential analysis functions."""
    
    @pytest.fixture
    def dummy_adata(self):
        """Create a dummy AnnData object for testing."""
        # Create a simple AnnData object
        n_cells = 100
        n_genes = 50
        X = np.random.rand(n_cells, n_genes)
        obs = pd.DataFrame({
            'group': ['group1'] * 50 + ['group2'] * 50,
            'sample': ['sample1'] * 25 + ['sample2'] * 25 + ['sample3'] * 25 + ['sample4'] * 25
        })
        var = pd.DataFrame(index=[f'gene{i}' for i in range(n_genes)])
        
        # Create AnnData object
        adata = anndata.AnnData(X=X, obs=obs, var=var)
        
        # Add cell state data (required for DE/DA)
        adata.obsm['DM_EigenVectors'] = np.random.rand(n_cells, 10)
        
        return adata

    def test_tracking_existence(self, dummy_adata, caplog):
        """Test that tracking exists for DA analysis."""
        caplog.set_level(logging.INFO)
        
        # Run differential abundance analysis (more reliable than DE for testing)
        compute_differential_abundance(
            dummy_adata,
            groupby='group',
            condition1='group1',
            condition2='group2',
            obsm_key='DM_EigenVectors',
            result_key='test_da',
            n_landmarks=10  # Specify a small number of landmarks for testing
        )
        
        # Check that the tracking structure was created
        assert 'kompot_da' in dummy_adata.uns
        assert 'anndata_fields' in dummy_adata.uns['kompot_da']
        
        # Check that we have all expected locations
        tracking = dummy_adata.uns['kompot_da']['anndata_fields']
        assert 'obs' in tracking
        assert 'uns' in tracking
        
        # Check obs fields - should include lfc, zscore, pval, direction, density
        obs_fields = tracking['obs']
        assert len(obs_fields) >= 5
        
        # Check that all obs fields are actually in adata.obs
        for field in obs_fields:
            assert field in dummy_adata.obs.columns
            
        # Check uns fields - should include at least the result_key
        uns_fields = tracking['uns']
        assert 'test_da' in uns_fields
        
        # Run a second DA analysis with sample variance to see if it correctly tracks the new fields
        compute_differential_abundance(
            dummy_adata,
            groupby='group',
            condition1='group1',
            condition2='group2',
            obsm_key='DM_EigenVectors',
            sample_col='sample',
            result_key='test_da_with_samples',
            n_landmarks=10  # Specify a small number of landmarks for testing
        )
        
        # Check that both result keys are tracked
        tracking = dummy_adata.uns['kompot_da']['anndata_fields']
        assert 'test_da' in tracking['uns']
        assert 'test_da_with_samples' in tracking['uns']
        
        # Check that the run IDs are different
        assert tracking['uns']['test_da'] != tracking['uns']['test_da_with_samples']

    def test_tracking_with_colors(self, dummy_adata, caplog):
        """Test the field tracking for direction colors in differential abundance."""
        caplog.set_level(logging.INFO)
        
        # Run differential abundance analysis
        compute_differential_abundance(
            dummy_adata,
            groupby='group',
            condition1='group1',
            condition2='group2',
            obsm_key='DM_EigenVectors',
            result_key='test_da_colors',
            n_landmarks=10  # Specify a small number of landmarks for testing
        )
        
        # Check that the tracking structure was created
        assert 'kompot_da' in dummy_adata.uns
        assert 'anndata_fields' in dummy_adata.uns['kompot_da']
        
        # Find the direction field and check that colors are tracked
        tracking = dummy_adata.uns['kompot_da']['anndata_fields']
        
        # Get the direction field
        direction_field = None
        for field in tracking['obs']:
            if 'direction' in field:
                direction_field = field
                break
                
        assert direction_field is not None
        
        # There should be a colors key for the direction field
        direction_colors = f"{direction_field}_colors"
        assert direction_colors in tracking['uns']

    def test_tracking_with_reused_key(self, dummy_adata, caplog):
        """Test tracking behavior when reusing the same result_key."""
        caplog.set_level(logging.INFO)
        
        # Run initial DA analysis
        compute_differential_abundance(
            dummy_adata,
            groupby='group',
            condition1='group1',
            condition2='group2',
            obsm_key='DM_EigenVectors',
            result_key='test_da_reused',
            n_landmarks=10  # Specify a small number of landmarks for testing
        )
        
        # Store the run_id of the first run
        tracking = dummy_adata.uns['kompot_da']['anndata_fields']
        first_run_id = tracking['uns']['test_da_reused']
        
        # Run a second DA analysis with the same result_key
        compute_differential_abundance(
            dummy_adata,
            groupby='group',
            condition1='group1',
            condition2='group2',
            obsm_key='DM_EigenVectors',
            result_key='test_da_reused',
            n_landmarks=10,  # Specify a small number of landmarks for testing
            overwrite=True
        )
        
        # Check that the run_id was updated
        tracking = dummy_adata.uns['kompot_da']['anndata_fields']
        second_run_id = tracking['uns']['test_da_reused']
        assert second_run_id != first_run_id
        
        # The run_id should be 1 more than the first
        assert second_run_id == first_run_id + 1
        
    def test_anndata_locations_tracking(self, dummy_adata, caplog):
        """Test that anndata_locations field is properly stored in run info."""
        caplog.set_level(logging.INFO)
        
        # Run differential abundance analysis 
        compute_differential_abundance(
            dummy_adata,
            groupby='group',
            condition1='group1',
            condition2='group2',
            obsm_key='DM_EigenVectors',
            result_key='test_locations_da',
            n_landmarks=10
        )
        
        # Check that field_mapping is present in run info
        assert 'kompot_da' in dummy_adata.uns
        assert 'last_run_info' in dummy_adata.uns['kompot_da']
        assert 'field_mapping' in dummy_adata.uns['kompot_da']['last_run_info']
        
        # Check specific field mappings
        field_mapping = dummy_adata.uns['kompot_da']['last_run_info']['field_mapping']
        
        # Find the log fold change field
        lfc_field = None
        for field, mapping in field_mapping.items():
            if mapping.get('type') == 'log_fold_change':
                lfc_field = field
                break
                
        assert lfc_field is not None
        assert field_mapping[lfc_field]['location'] == 'obs'
        assert 'description' in field_mapping[lfc_field]
        
        # Find the direction field
        direction_field = None
        for field, mapping in field_mapping.items():
            if mapping.get('type') == 'direction':
                direction_field = field
                break
                
        assert direction_field is not None
        assert field_mapping[direction_field]['location'] == 'obs'
        
        # Run differential expression analysis for comparison
        compute_differential_expression(
            dummy_adata,
            groupby='group',
            condition1='group1',
            condition2='group2',
            obsm_key='DM_EigenVectors',
            result_key='test_locations_de',
            n_landmarks=10,
            compute_mahalanobis=False  # For simplicity in testing
        )
        
        # Check that field_mapping is present in DE run info
        assert 'kompot_de' in dummy_adata.uns
        assert 'last_run_info' in dummy_adata.uns['kompot_de']
        assert 'field_mapping' in dummy_adata.uns['kompot_de']['last_run_info']
        
        # Check specific field mappings for DE
        de_field_mapping = dummy_adata.uns['kompot_de']['last_run_info']['field_mapping']
        
        # Find var field (mean log fold change)
        var_field = None
        for field, mapping in de_field_mapping.items():
            if mapping.get('location') == 'var' and mapping.get('type') == 'mean_log_fold_change':
                var_field = field
                break
                
        assert var_field is not None
        assert 'description' in de_field_mapping[var_field]
        
        # Find layer field (fold change)
        layer_field = None
        for field, mapping in de_field_mapping.items():
            if mapping.get('location') == 'layers' and mapping.get('type') == 'fold_change':
                layer_field = field
                break
                
        assert layer_field is not None