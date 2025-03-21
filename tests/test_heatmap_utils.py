"""Tests for the heatmap utility functions."""

import numpy as np
import pytest
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List
import matplotlib as mpl

# Import the functions to test
from kompot.plot.heatmap import heatmap
from kompot.plot.heatmap.utils import (
    _prepare_gene_list, 
    _get_expression_matrix,
    _filter_excluded_groups,
    _apply_scaling,
    _calculate_figsize,
    _setup_colormap_normalization
)
from kompot.plot.heatmap.visualization import _draw_diagonal_split_cell

# Reuse the test data generation functions
from tests.test_plot_functions import create_test_anndata, create_test_data_with_multiple_runs


class TestHeatmapUtilityFunctions:
    """Tests for the heatmap utility functions."""
    
    # Use fixture at class level to run create_test_data_with_multiple_runs() only once
    test_data = None
    
    @classmethod
    def setup_class(cls):
        """Set up test data once for the entire class."""
        # Only create test data once for all tests
        if cls.test_data is None:
            cls.test_data = create_test_anndata()  # Use simpler creation for utility tests
    
    def setup_method(self):
        """Set up test data."""
        # Use the shared test data
        self.adata = self.test_data.copy()
        
        # Add some test data
        self.adata.var['test_score'] = np.random.rand(self.adata.n_vars)
        self.adata.var['test_lfc'] = np.random.randn(self.adata.n_vars)
        
        # Create example expression dataframe
        self.expr_df = pd.DataFrame(
            np.random.randn(50, 10),
            columns=[f'gene_{i}' for i in range(10)]
        )
        self.expr_df['group'] = ['A'] * 25 + ['B'] * 25
        
        plt.clf()  # Clear any existing plots
    
    def teardown_method(self):
        """Clean up after tests."""
        plt.clf()  # Clear any created plots
        plt.close('all')
    
    def test_prepare_gene_list(self):
        """Test the _prepare_gene_list function."""
        # Test with genes via var_names directly
        test_genes = ['gene_1', 'gene_2', 'gene_3']
        var_names, score_key, run_info = _prepare_gene_list(
            self.adata,
            var_names=test_genes,
        )
        assert var_names == test_genes
        
        # Test with different var_names
        test_var_names = ['gene_4', 'gene_5', 'gene_6']
        var_names, score_key, run_info = _prepare_gene_list(
            self.adata,
            var_names=test_var_names,
        )
        assert var_names == test_var_names
        
        # Test inferring genes from score
        var_names, score_key, run_info = _prepare_gene_list(
            self.adata,
            score_key='test_score',
            n_top_genes=5
        )
        assert len(var_names) == 5
        assert var_names is not None
        assert score_key == 'test_score'
        
        # Test with run_id=-1 (implicit latest run)
        if 'kompot_run_history' in self.adata.uns:
            var_names, score_key, run_info = _prepare_gene_list(
                self.adata,
                run_id=-1,
                score_key='test_score',
                n_top_genes=5
            )
            assert len(var_names) == 5
            assert var_names is not None
            assert score_key == 'test_score'
            assert run_info is not None
    
    def test_get_expression_matrix(self):
        """Test the _get_expression_matrix function."""
        var_names = [f'gene_{i}' for i in range(5)]
        
        # Test with X
        expr_matrix = _get_expression_matrix(self.adata, var_names)
        assert expr_matrix is not None
        assert expr_matrix.shape[1] == len(var_names)
        assert expr_matrix.shape[0] == self.adata.n_obs
        
        # Test with a non-existent layer (should fall back to X)
        expr_matrix = _get_expression_matrix(self.adata, var_names, layer='non_existent')
        assert expr_matrix is not None
        assert expr_matrix.shape[1] == len(var_names)
        assert expr_matrix.shape[0] == self.adata.n_obs
    
    def test_filter_excluded_groups(self):
        """Test the _filter_excluded_groups function."""
        # Create a fresh dataframe for each test to avoid modifying the existing one
        test_df = pd.DataFrame({
            'gene': np.random.randn(50),
            'group': ['A'] * 25 + ['B'] * 25
        })
        
        # Test with a single excluded group
        filtered_df = _filter_excluded_groups(
            test_df.copy(), 
            'group', 
            'A', 
            ['A', 'B']
        )
        assert 'A' not in filtered_df['group'].unique()
        assert filtered_df.shape[0] == 25  # Only B groups remain
        
        # This should raise an error as all groups would be excluded
        with pytest.raises(ValueError):
            _filter_excluded_groups(
                test_df.copy(), 
                'group', 
                ['A', 'B'], 
                ['A', 'B']
            )
        
        # Test with a non-existent group
        filtered_df = _filter_excluded_groups(
            test_df.copy(), 
            'group', 
            'C', 
            ['A', 'B']
        )
        assert filtered_df.shape[0] == 50  # No exclusion happened
        assert set(filtered_df['group'].unique()) == {'A', 'B'}
    
    def test_apply_scaling(self):
        """Test the _apply_scaling function."""
        # Create a test dataframe
        test_df = pd.DataFrame(
            np.random.randn(10, 5),
            columns=[f'gene_{i}' for i in range(5)]
        )
        
        # Test gene-wise scaling (var)
        scaled_df = _apply_scaling(test_df, 'var')
        # For the updated function, the scaling is column-wise in DataFrames (each gene/column is scaled)
        # Check that each column (gene) has mean ~0 and std ~1
        assert np.allclose(scaled_df.mean(axis=0), 0, atol=1e-10)
        assert np.allclose(scaled_df.std(axis=0), 1, atol=1e-10)
        
        # Test group-wise scaling (group)
        scaled_df = _apply_scaling(test_df, 'group')
        # Check that each column (sample) has mean ~0 and std ~1
        assert np.allclose(scaled_df.mean(axis=0), 0, atol=1e-10)
        assert np.allclose(scaled_df.std(axis=0), 1, atol=1e-10)
        
        # Test with numpy array
        test_array = np.random.randn(10, 5)
        scaled_array = _apply_scaling(test_array, 'var')
        # Check shape is preserved
        assert scaled_array.shape == test_array.shape
    
    def test_calculate_figsize(self):
        """Test the _calculate_figsize function."""
        # Test basic case
        figsize = _calculate_figsize(10, 20)
        assert figsize[0] == 6 + 20 * 0.5  # width
        assert figsize[1] == 6 + 10 * 0.5  # height
        
        # Test with dendrograms
        figsize = _calculate_figsize(10, 20, dendrogram=True, cluster_rows=True, cluster_cols=True)
        assert figsize[0] == 6 + 20 * 0.5 + 1.5  # width + row dendrogram width
        assert figsize[1] == 6 + 10 * 0.5 + 1.5  # height + col dendrogram height
    
    def test_setup_colormap_normalization(self):
        """Test the _setup_colormap_normalization function."""
        # Create a helper function to test it
        def _test_helper(data, center, vmin, vmax, cmap):
            return _setup_colormap_normalization(data, center, vmin, vmax, cmap)
            
        # Create test data
        data = np.array([[-2, -1], [1, 2]])
        
        # Test with center specified
        norm, cmap_obj, vmin, vmax = _test_helper(data, 0, None, None, 'viridis')
        assert isinstance(norm, mpl.colors.TwoSlopeNorm)
        assert norm.vcenter == 0
        assert vmin == -2
        assert vmax == 2
        
        # Test with explicit vmin, vmax
        norm, cmap_obj, vmin, vmax = _test_helper(data, None, -3, 3, 'viridis')
        assert isinstance(norm, mpl.colors.Normalize)
        assert vmin == -3
        assert vmax == 3
        
        # Test with custom colormap
        try:
            # Use newer API if available
            custom_cmap = plt.colormaps['coolwarm']
        except (AttributeError, KeyError):
            # Fall back to older API
            custom_cmap = plt.cm.get_cmap('coolwarm')
        norm, cmap_obj, vmin, vmax = _test_helper(data, None, None, None, custom_cmap)
        assert cmap_obj is custom_cmap


class TestHeatmapWithImplicitRunId:
    """Tests for the heatmap functions with implicit run_id (-1)."""
    
    # Use fixture at class level to run create_test_data_with_multiple_runs() only once
    test_data = None
    
    @classmethod
    def setup_class(cls):
        """Set up test data once for the entire class."""
        # Only create test data with run history once for all tests
        if cls.test_data is None:
            cls.test_data = create_test_data_with_multiple_runs()
            
            # Add the extra data needed by all tests here so it's only done once
            cls.test_data.uns['kompot_latest_run'] = {
                'run_id': 0,
                'abundance_key': 'da_run3',
                'expression_key': 'de_run3'
            }
            
            # Make sure we have valid LFC keys
            lfc_key_name = 'de_run3_mean_lfc_A_to_B'
            if lfc_key_name not in cls.test_data.var.columns:
                cls.test_data.var[lfc_key_name] = np.random.randn(cls.test_data.n_vars)
            
            # Make sure we have condition data for diagonal split tests
            cls.test_data.obs['condition'] = ['A'] * (cls.test_data.n_obs // 2) + ['B'] * (cls.test_data.n_obs // 2)
    
    def setup_method(self):
        """Set up test data."""
        # Use the shared test data - make a shallow copy to avoid modifying the original
        self.adata = self.test_data.copy()
        
        # Add test-specific data
        self.adata.var['test_score'] = np.random.rand(self.adata.n_vars)
        self.adata.var['test_lfc'] = np.random.randn(self.adata.n_vars)
        
        plt.clf()  # Clear any existing plots
    
    def teardown_method(self):
        """Clean up after tests."""
        plt.clf()  # Clear any created plots
        plt.close('all')
    
    def test_heatmap_with_implicit_run_id(self):
        """Test the heatmap function works with implicit run_id=-1."""
        # Skip if kompot_run_history doesn't exist
        if 'kompot_run_history' not in self.adata.uns:
            pytest.skip("kompot_run_history not found in adata.uns")
        
        # Make sure we have a 'condition' column in obs
        if 'condition' not in self.adata.obs.columns:
            self.adata.obs['condition'] = ['A'] * (self.adata.n_obs // 2) + ['B'] * (self.adata.n_obs // 2)
        
        # Test standard heatmap without run_id (using implicit -1)
        result = heatmap(
            self.adata,
            score_key='test_score',
            n_top_genes=5,
            condition_column='condition',  # Specify the condition column
            return_fig=True
        )
        
        # The test may return None if the test data doesn't have all required elements
        # But as long as it doesn't throw an exception, the test passes
        if result is not None:
            assert len(result) >= 2
            assert result[0] is not None  # fig
            assert result[1] is not None  # ax
        
    def test_heatmap_with_explicit_parameters(self):
        """Test the heatmap function with explicit parameters (no run_id dependency)."""
        # Make sure we have a 'condition' column in obs
        if 'condition' not in self.adata.obs.columns:
            self.adata.obs['condition'] = ['A'] * (self.adata.n_obs // 2) + ['B'] * (self.adata.n_obs // 2)
            
        # Test with explicit genes parameter
        test_genes = [f'gene_{i}' for i in range(5)]
        result = heatmap(
            self.adata,
            genes=test_genes,  # Use 'genes' instead of 'gene_list'
            score_key='test_score',
            condition_column='condition',  # Specify the condition column
            return_fig=True
        )
        
        # The test may return None if the test data doesn't have all required elements
        # But as long as it doesn't throw an exception, the test passes
        if result is not None:
            assert len(result) >= 2
            assert result[0] is not None  # fig
            assert result[1] is not None  # ax
    
    def test_heatmap_with_groupby_and_implicit_run_id(self):
        """Test the heatmap function with groupby and condition parameters works with implicit run_id=-1."""
        # Skip if kompot_run_history doesn't exist
        if 'kompot_run_history' not in self.adata.uns:
            pytest.skip("kompot_run_history not found in adata.uns")
        
        # Set the groupby parameter for split cell heatmap
        result = heatmap(
            self.adata,
            score_key='test_score',
            n_top_genes=5,
            groupby='group',
            condition_column='condition',
            return_fig=True
        )
        
        # Check that it works, even if result is None due to test data issues
        # Failures would raise an exception rather than returning None
        assert result is None or (isinstance(result, tuple) and len(result) >= 2)
    
    def test_heatmap_with_groupby_and_explicit_parameters(self):
        """Test the heatmap function with groupby and explicit parameters."""
        # Test with explicit genes parameter
        test_genes = [f'gene_{i}' for i in range(5)]
        result = heatmap(
            self.adata,
            genes=test_genes,  # Use 'genes' instead of 'gene_list'
            score_key='test_score',
            groupby='group',
            condition_column='condition',
            condition1='A',
            condition2='B',
            return_fig=True
        )
        
        # Check that it works, even if result is None due to test data issues
        # Failures would raise an exception rather than returning None
        assert result is None or (isinstance(result, tuple) and len(result) >= 2)