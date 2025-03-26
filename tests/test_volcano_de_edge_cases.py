"""Tests for edge cases in the volcano_de plotting function."""

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent windows
import matplotlib.pyplot as plt
from typing import Dict, Any, List
import matplotlib as mpl
from matplotlib.colors import ListedColormap

from kompot.plot.volcano import volcano_de, _infer_de_keys
from kompot.utils import KOMPOT_COLORS


def create_test_anndata_edge_cases(n_cells=100, n_genes=20):
    """Create a test AnnData object with edge cases for testing volcano_de."""
    try:
        import anndata
    except ImportError:
        pytest.skip("anndata not installed, skipping test")
        
    np.random.seed(42)
    
    # Create test data
    X = np.random.normal(0, 1, (n_cells, n_genes))
    
    # Create cell groups for testing
    groups = np.array(['A'] * (n_cells // 2) + ['B'] * (n_cells // 2))
    
    # Create var_names
    var_names = [f'gene_{i}' for i in range(n_genes)]
    
    # Create var DataFrame with var_names as index
    var = pd.DataFrame(index=var_names)
    
    # Add DE metrics
    var['mean_lfc'] = np.random.normal(0, 2, n_genes)
    var['mahalanobis'] = np.abs(np.random.normal(0, 2, n_genes))
    var['pval'] = np.random.uniform(0, 1, n_genes)
    var['de_run1_mean_lfc_A_to_B'] = np.random.normal(0, 2, n_genes)
    var['de_run1_mahalanobis_A_to_B'] = np.abs(np.random.normal(0, 2, n_genes))
    
    # Add special cases
    # 1. Add missing values (NaNs)
    var.loc[var_names[0:2], 'mean_lfc'] = np.nan
    var.loc[var_names[2:4], 'mahalanobis'] = np.nan
    
    # 2. Add binary categorical data (only 2 categories)
    var['binary_category'] = ['group_A' if i % 2 == 0 else 'group_B' for i in range(n_genes)]
    
    # 3. Add multi-category data
    var['multi_category'] = pd.Categorical(['cat_' + str(i % 5) for i in range(n_genes)])
    
    # 4. Add boolean values
    var['boolean_flag'] = [i % 2 == 0 for i in range(n_genes)]
    
    # 5. Add extreme values
    var['extreme_vals'] = np.random.normal(0, 1, n_genes)
    var.loc[var_names[0], 'extreme_vals'] = 1000  # Extreme positive outlier
    var.loc[var_names[1], 'extreme_vals'] = -1000  # Extreme negative outlier
    
    # Create observation dataframe
    obs = pd.DataFrame({
        'group': groups
    })
    
    # Create AnnData object
    adata = anndata.AnnData(X=X, obs=obs, var=var)
    
    # Add run info to uns
    adata.uns['de_run1'] = {
        'run_info': {
            'field_names': {
                'mean_lfc_key': 'de_run1_mean_lfc_A_to_B',
                'mahalanobis_key': 'de_run1_mahalanobis_A_to_B'
            },
            'params': {
                'conditions': ['A', 'B']
            }
        }
    }
    
    # Setup run history
    if 'kompot_de' not in adata.uns:
        adata.uns['kompot_de'] = {}
    if 'run_history' not in adata.uns['kompot_de']:
        adata.uns['kompot_de']['run_history'] = []
        
    # Add the run to history    
    adata.uns['kompot_de']['run_history'].append({
        'run_id': 0,
        'expression_key': 'de_run1',
        'field_names': {
            'mean_lfc_key': 'de_run1_mean_lfc_A_to_B',
            'mahalanobis_key': 'de_run1_mahalanobis_A_to_B'
        },
        'params': {
            'conditions': ['A', 'B']
        }
    })
    
    return adata


class TestVolcanoDEEdgeCases:
    """Tests for edge cases in the volcano_de function."""
    
    def setup_method(self):
        """Set up test data for each test."""
        self.adata = create_test_anndata_edge_cases()
        plt.clf()  # Clear any existing plots
    
    def teardown_method(self):
        """Clean up after tests."""
        plt.clf()  # Clear any created plots
        plt.close('all')
    
    def test_binary_categorical_coloring(self):
        """Test volcano_de with binary categorical background coloring."""
        # Test with binary categorical data (special case that might need special handling)
        fig, ax = volcano_de(
            self.adata,
            lfc_key='de_run1_mean_lfc_A_to_B',
            score_key='de_run1_mahalanobis_A_to_B',
            background_color_key='binary_category',
            return_fig=True
        )
        assert fig is not None
        assert ax is not None
    
    def test_multi_categorical_pandas_categorical(self):
        """Test volcano_de with pandas Categorical type background coloring."""
        # Test with pandas Categorical data
        fig, ax = volcano_de(
            self.adata,
            lfc_key='de_run1_mean_lfc_A_to_B',
            score_key='de_run1_mahalanobis_A_to_B',
            background_color_key='multi_category',
            return_fig=True
        )
        assert fig is not None
        assert ax is not None
    
    def test_boolean_background_coloring(self):
        """Test volcano_de with boolean background coloring."""
        # Boolean values should be treated as categorical
        fig, ax = volcano_de(
            self.adata,
            lfc_key='de_run1_mean_lfc_A_to_B',
            score_key='de_run1_mahalanobis_A_to_B',
            background_color_key='boolean_flag',
            return_fig=True
        )
        assert fig is not None
        assert ax is not None
    
    def test_extreme_values_background_coloring(self):
        """Test volcano_de with extreme value background coloring."""
        # Test with extreme values, should still work with colormap normalization
        fig, ax = volcano_de(
            self.adata,
            lfc_key='de_run1_mean_lfc_A_to_B',
            score_key='de_run1_mahalanobis_A_to_B',
            background_color_key='extreme_vals',
            return_fig=True
        )
        assert fig is not None
        assert ax is not None
        
        # Test with percentile-based vmin/vmax to handle outliers
        fig, ax = volcano_de(
            self.adata,
            lfc_key='de_run1_mean_lfc_A_to_B',
            score_key='de_run1_mahalanobis_A_to_B',
            background_color_key='extreme_vals',
            vmin='p5',
            vmax='p95',
            return_fig=True
        )
        assert fig is not None
        assert ax is not None
    
    def test_missing_values_in_data(self):
        """Test volcano_de with missing values (NaNs) in the data."""
        # Data contains NaNs in the mean_lfc and mahalanobis columns
        # Function should handle these gracefully
        fig, ax = volcano_de(
            self.adata,
            lfc_key='mean_lfc',  # Contains NaNs
            score_key='mahalanobis',  # Contains NaNs
            return_fig=True
        )
        assert fig is not None
        assert ax is not None
    
    def test_nonexistent_highlight_genes(self):
        """Test volcano_de with highlight_genes that don't exist in the data."""
        # Test with completely non-existent genes
        highlight_genes = ['nonexistent_gene_1', 'nonexistent_gene_2', 'gene_0']
        fig, ax = volcano_de(
            self.adata,
            lfc_key='de_run1_mean_lfc_A_to_B',
            score_key='de_run1_mahalanobis_A_to_B',
            highlight_genes=highlight_genes,
            return_fig=True
        )
        assert fig is not None
        assert ax is not None
    
    def test_custom_colormap_with_binary_data(self):
        """Test volcano_de with custom colormap for binary categorical data."""
        # Test with custom colormap for binary data
        fig, ax = volcano_de(
            self.adata,
            lfc_key='de_run1_mean_lfc_A_to_B',
            score_key='de_run1_mahalanobis_A_to_B',
            background_color_key='binary_category',
            background_cmap='Set2',  # Different colormap
            return_fig=True
        )
        assert fig is not None
        assert ax is not None
        
        # Test with manually specified color map
        color_map = {
            'group_A': '#FF5733',
            'group_B': '#33FF57'
        }
        fig, ax = volcano_de(
            self.adata,
            lfc_key='de_run1_mean_lfc_A_to_B',
            score_key='de_run1_mahalanobis_A_to_B',
            background_color_key='binary_category',
            color_discrete_map=color_map,
            return_fig=True
        )
        assert fig is not None
        assert ax is not None
    
    def test_complex_highlight_genes(self):
        """Test volcano_de with complex highlight_genes scenarios."""
        # Test with empty dict as highlight_genes (edge case)
        fig, ax = volcano_de(
            self.adata,
            lfc_key='de_run1_mean_lfc_A_to_B',
            score_key='de_run1_mahalanobis_A_to_B',
            highlight_genes={},
            return_fig=True
        )
        assert fig is not None
        assert ax is not None
        
        # Test with list of dicts where some are malformed
        highlight_groups = [
            {
                'genes': self.adata.var_names[:3].tolist(),
                'name': 'Group 1',
                'color': '#FF5733'
            },
            {
                # Missing 'genes' key
                'name': 'Group 2',
                'color': '#33FF57'
            },
            {
                'genes': [],  # Empty genes list
                'name': 'Group 3',
                'color': '#3357FF'
            }
        ]
        fig, ax = volcano_de(
            self.adata,
            lfc_key='de_run1_mean_lfc_A_to_B',
            score_key='de_run1_mahalanobis_A_to_B',
            highlight_genes=highlight_groups,
            return_fig=True
        )
        assert fig is not None
        assert ax is not None
    
    def test_zero_top_genes(self):
        """Test volcano_de with n_top_genes=0."""
        # With n_top_genes=0, no genes should be highlighted
        fig, ax = volcano_de(
            self.adata,
            lfc_key='de_run1_mean_lfc_A_to_B',
            score_key='de_run1_mahalanobis_A_to_B',
            n_top_genes=0,
            return_fig=True
        )
        assert fig is not None
        assert ax is not None
        
    def test_no_show_names(self):
        """Test volcano_de with show_names=False."""
        # With show_names=False, no gene names should be shown
        fig, ax = volcano_de(
            self.adata,
            lfc_key='de_run1_mean_lfc_A_to_B',
            score_key='de_run1_mahalanobis_A_to_B',
            show_names=False,
            return_fig=True
        )
        assert fig is not None
        assert ax is not None
    
    def test_invalid_percentile_strings(self):
        """Test volcano_de with invalid percentile strings for vmin/vmax."""
        # Test with invalid percentile string format
        fig, ax = volcano_de(
            self.adata,
            lfc_key='de_run1_mean_lfc_A_to_B',
            score_key='de_run1_mahalanobis_A_to_B',
            background_color_key='extreme_vals',
            vmin='invalid_format',  # Should fallback to data min
            vmax='p95',
            return_fig=True
        )
        assert fig is not None
        assert ax is not None