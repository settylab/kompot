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
        
        # Import the JSON utility function
        from kompot.anndata.utils import get_json_metadata
        
        # Check that the tracking structure was created
        assert 'kompot_da' in dummy_adata.uns
        assert 'anndata_fields' in dummy_adata.uns['kompot_da']
        
        # Check that we have all expected locations
        tracking = get_json_metadata(dummy_adata, 'kompot_da.anndata_fields')
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
        tracking = get_json_metadata(dummy_adata, 'kompot_da.anndata_fields')
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
        
        # Import the JSON utility function
        from kompot.anndata.utils import get_json_metadata
        
        # Check that the tracking structure was created
        assert 'kompot_da' in dummy_adata.uns
        assert 'anndata_fields' in dummy_adata.uns['kompot_da']
        
        # Find the direction field and check that colors are tracked
        tracking = get_json_metadata(dummy_adata, 'kompot_da.anndata_fields')
        
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
        
        # Import the JSON utility function
        from kompot.anndata.utils import get_json_metadata
        
        # Store the run_id of the first run
        tracking = get_json_metadata(dummy_adata, 'kompot_da.anndata_fields')
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
        tracking = get_json_metadata(dummy_adata, 'kompot_da.anndata_fields')
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
        
        # Import the JSON utility function
        from kompot.anndata.utils import get_json_metadata
        
        # Check that field_mapping is present in run info
        assert 'kompot_da' in dummy_adata.uns
        assert 'last_run_info' in dummy_adata.uns['kompot_da']
        
        run_info = get_json_metadata(dummy_adata, 'kompot_da.last_run_info')
        assert 'field_mapping' in run_info
        
        # Check specific field mappings
        field_mapping = run_info['field_mapping']
        
        # Find the log fold change field
        lfc_field = None
        for field, mapping in field_mapping.items():
            # Handle case where mapping could be a string
            if isinstance(mapping, str):
                from kompot.anndata.utils import from_json_string
                mapping = from_json_string(mapping)
                
            if mapping.get('type') == 'log_fold_change':
                lfc_field = field
                break
                
        assert lfc_field is not None
        
        # Ensure we have a dictionary before accessing
        lfc_mapping = field_mapping[lfc_field]
        if isinstance(lfc_mapping, str):
            lfc_mapping = from_json_string(lfc_mapping)
            
        assert lfc_mapping['location'] == 'obs'
        assert 'description' in lfc_mapping
        
        # Find the direction field
        direction_field = None
        for field, mapping in field_mapping.items():
            # Handle case where mapping could be a string
            if isinstance(mapping, str):
                mapping = from_json_string(mapping)
                
            if mapping.get('type') == 'direction':
                direction_field = field
                break
                
        assert direction_field is not None
        
        # Ensure we have a dictionary before accessing
        direction_mapping = field_mapping[direction_field]
        if isinstance(direction_mapping, str):
            direction_mapping = from_json_string(direction_mapping)
            
        assert direction_mapping['location'] == 'obs'
        
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
        
        de_run_info = get_json_metadata(dummy_adata, 'kompot_de.last_run_info')
        assert 'field_mapping' in de_run_info
        
        # Check specific field mappings for DE
        de_field_mapping = de_run_info['field_mapping']
        
        # Find var field (mean log fold change)
        var_field = None
        for field, mapping in de_field_mapping.items():
            # Handle case where mapping could be a string
            if isinstance(mapping, str):
                mapping = from_json_string(mapping)
                
            if mapping.get('location') == 'var' and mapping.get('type') == 'mean_log_fold_change':
                var_field = field
                break
                
        assert var_field is not None
        
        # Ensure we have a dictionary before accessing
        var_mapping = de_field_mapping[var_field]
        if isinstance(var_mapping, str):
            var_mapping = from_json_string(var_mapping)
            
        assert 'description' in var_mapping
        
        # Find layer field (fold change)
        layer_field = None
        for field, mapping in de_field_mapping.items():
            # Handle case where mapping could be a string
            if isinstance(mapping, str):
                mapping = from_json_string(mapping)
                
            if mapping.get('location') == 'layers' and mapping.get('type') == 'fold_change':
                layer_field = field
                break
                
        assert layer_field is not None
        
    def test_std_keys_in_sample_variance_impacted(self, dummy_adata, caplog):
        """Test that std_key_1 and std_key_2 are properly included in sample_variance_impacted list."""
        caplog.set_level(logging.INFO)
        
        # Run differential expression analysis with sample_col to use sample variance
        compute_differential_expression(
            dummy_adata,
            groupby='group',
            condition1='group1',
            condition2='group2',
            obsm_key='DM_EigenVectors',
            result_key='test_std_keys',
            sample_col='sample',
            n_landmarks=10,
            compute_mahalanobis=False  # For simplicity in testing
        )
        
        # Import the JSON utility function
        from kompot.anndata.utils import get_json_metadata, from_json_string
        
        # Extract the field_names from run_info
        assert 'kompot_de' in dummy_adata.uns
        assert 'last_run_info' in dummy_adata.uns['kompot_de']
        
        run_info = get_json_metadata(dummy_adata, 'kompot_de.last_run_info')
        assert 'field_names' in run_info
        
        field_names = run_info['field_names']
        
        # Check that std_key_1 and std_key_2 exist in field_names
        assert 'std_key_1' in field_names
        assert 'std_key_2' in field_names
        
        # Verify these are properly formatted with the condition names
        assert 'group1_std' in field_names['std_key_1']
        assert 'group2_std' in field_names['std_key_2']
        
        # Check that sample_variance_impacted_fields include std_key_1 and std_key_2
        assert 'sample_variance_impacted_fields' in field_names
        assert 'std_key_1' in field_names['sample_variance_impacted_fields']
        assert 'std_key_2' in field_names['sample_variance_impacted_fields']
        
        # Verify std fields are present in field_mapping
        field_mapping = run_info['field_mapping']
        
        std1_field = field_names['std_key_1']
        std2_field = field_names['std_key_2']
        
        assert std1_field in field_mapping
        assert std2_field in field_mapping
        
        # Get mapping objects and handle possible JSON strings
        std1_mapping = field_mapping[std1_field]
        if isinstance(std1_mapping, str):
            std1_mapping = from_json_string(std1_mapping)
            
        std2_mapping = field_mapping[std2_field]
        if isinstance(std2_mapping, str):
            std2_mapping = from_json_string(std2_mapping)
        
        # With sample variance, std fields should be in layers
        assert std1_mapping['location'] == 'layers'
        assert std2_mapping['location'] == 'layers'
        assert std1_mapping['type'] == 'std_with_sample_var'
        assert std2_mapping['type'] == 'std_with_sample_var'
        
    def test_std_keys_without_sample_variance(self, dummy_adata, caplog):
        """Test that std_key_1 and std_key_2 are properly handled without sample variance."""
        caplog.set_level(logging.INFO)
        
        # Run differential expression analysis without sample_col
        compute_differential_expression(
            dummy_adata,
            groupby='group',
            condition1='group1',
            condition2='group2',
            obsm_key='DM_EigenVectors',
            result_key='test_std_keys_no_samples',
            n_landmarks=10,
            compute_mahalanobis=False  # For simplicity in testing
        )
        
        # Import the JSON utility function
        from kompot.anndata.utils import get_json_metadata, from_json_string
        
        # Extract the field_names from run_info
        assert 'kompot_de' in dummy_adata.uns
        assert 'last_run_info' in dummy_adata.uns['kompot_de']
        
        run_info = get_json_metadata(dummy_adata, 'kompot_de.last_run_info')
        assert 'field_names' in run_info
        
        field_names = run_info['field_names']
        
        # Check that std_key_1 and std_key_2 exist in field_names
        assert 'std_key_1' in field_names
        assert 'std_key_2' in field_names
        
        # Verify these are properly formatted with the condition names
        assert 'group1_std' in field_names['std_key_1']
        assert 'group2_std' in field_names['std_key_2']
        
        # Verify std fields are present in field_mapping
        field_mapping = run_info['field_mapping']
        
        std1_field = field_names['std_key_1']
        std2_field = field_names['std_key_2']
        
        assert std1_field in field_mapping
        assert std2_field in field_mapping
        
        # Get mapping objects and handle possible JSON strings
        std1_mapping = field_mapping[std1_field]
        if isinstance(std1_mapping, str):
            std1_mapping = from_json_string(std1_mapping)
            
        std2_mapping = field_mapping[std2_field]
        if isinstance(std2_mapping, str):
            std2_mapping = from_json_string(std2_mapping)
        
        # Without sample variance, std fields should be in obs
        assert std1_mapping['location'] == 'obs'
        assert std2_mapping['location'] == 'obs'
        assert std1_mapping['type'] == 'std'
        assert std2_mapping['type'] == 'std'
        
        # Check that fields were actually created in adata.obs
        assert std1_field in dummy_adata.obs
        assert std2_field in dummy_adata.obs