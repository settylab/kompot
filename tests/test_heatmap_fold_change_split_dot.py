"""Tests for heatmap fold change and split dot modes."""

import numpy as np
import pytest
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List
import matplotlib as mpl

# Import the functions to test
from kompot.plot.heatmap import heatmap
from kompot.plot.heatmap.visualization import (
    _draw_diagonal_split_cell, 
    _draw_fold_change_cell,
    _draw_split_dot_cell
)

# Reuse the test data generation functions
from tests.test_plot_functions import create_test_anndata, create_test_data_with_multiple_runs


class TestHeatmapVisualizationFunctions:
    """Tests for the heatmap visualization functions."""
    
    def setup_method(self):
        """Set up test data."""
        plt.clf()  # Clear any existing plots
    
    def teardown_method(self):
        """Clean up after tests."""
        plt.clf()  # Clear any created plots
        plt.close('all')
    
    def test_draw_diagonal_split_cell(self):
        """Test the _draw_diagonal_split_cell function."""
        # Create a test figure and axes
        fig, ax = plt.subplots()
        
        # Test with valid values
        _draw_diagonal_split_cell(
            ax=ax,
            x=0,
            y=0,
            w=1,
            h=1,
            val1=1.0,
            val2=3.0,
            cmap='viridis',
            vmin=-2,
            vmax=2
        )
        
        # Test with NaN values
        _draw_diagonal_split_cell(
            ax=ax,
            x=1,
            y=0,
            w=1,
            h=1,
            val1=np.nan,
            val2=3.0,
            cmap='viridis',
            vmin=-2,
            vmax=2
        )
        
        # Test with custom colormap
        custom_cmap = plt.cm.get_cmap('coolwarm')
        _draw_diagonal_split_cell(
            ax=ax,
            x=0,
            y=1,
            w=1,
            h=1,
            val1=1.0,
            val2=3.0,
            cmap=custom_cmap,
            vmin=-2,
            vmax=2
        )
        
        # Check patches
        assert len(ax.patches) == 6  # 2 triangles per cell * 3 cells
        for patch in ax.patches:
            assert isinstance(patch, mpl.patches.Polygon)
        
        plt.close(fig)
    
    def test_draw_fold_change_cell(self):
        """Test the _draw_fold_change_cell function."""
        # Create a test figure and axes
        fig, ax = plt.subplots()
        
        # Test with valid values
        _draw_fold_change_cell(
            ax=ax,
            x=0,
            y=0,
            w=1,
            h=1,
            lfc=1.0,  # This is the fold change value
            cmap='RdBu_r',
            vmin=-2,
            vmax=2
        )
        
        # Test with NaN values
        _draw_fold_change_cell(
            ax=ax,
            x=1,
            y=0,
            w=1,
            h=1,
            lfc=np.nan,
            cmap='RdBu_r',
            vmin=-2,
            vmax=2
        )
        
        # Test with extreme values (should be clamped to vmin/vmax)
        _draw_fold_change_cell(
            ax=ax,
            x=0,
            y=1,
            w=1,
            h=1,
            lfc=5.0,  # Above vmax
            cmap='RdBu_r',
            vmin=-2,
            vmax=2
        )
        
        # Test with custom colormap
        custom_cmap = plt.cm.get_cmap('coolwarm')
        _draw_fold_change_cell(
            ax=ax,
            x=1,
            y=1,
            w=1,
            h=1,
            lfc=-5.0,  # Below vmin
            cmap=custom_cmap,
            vmin=-2,
            vmax=2
        )
        
        # Check that patches were added to the axis
        assert len(ax.patches) == 4  # 1 rectangle per cell * 4 cells
        
        # Verify that the patches are rectangles
        for patch in ax.patches:
            assert isinstance(patch, mpl.patches.Rectangle)
        
        plt.close(fig)
    
    def test_draw_split_dot_cell(self):
        """Test the _draw_split_dot_cell function."""
        # Create a test figure and axes
        fig, ax = plt.subplots()
        
        # Test with valid values and equal cell counts
        _draw_split_dot_cell(
            ax=ax,
            x=0,
            y=0,
            w=1,
            h=1,
            val1=1.0,
            val2=3.0,
            cmap='RdBu_r',
            vmin=-2,
            vmax=2,
            cell_count1=100,
            cell_count2=100,
            global_max_count=200
        )
        
        # Test with different cell counts
        _draw_split_dot_cell(
            ax=ax,
            x=1,
            y=0,
            w=1,
            h=1,
            val1=0.5,
            val2=1.5,
            cmap='RdBu_r',
            vmin=-2,
            vmax=2,
            cell_count1=50,
            cell_count2=150,
            global_max_count=200
        )
        
        # Test with NaN values
        _draw_split_dot_cell(
            ax=ax,
            x=0,
            y=1,
            w=1,
            h=1,
            val1=np.nan,
            val2=1.0,
            cmap='RdBu_r',
            vmin=-2,
            vmax=2,
            cell_count1=100,
            cell_count2=50,
            global_max_count=200
        )
        
        # Test with zero cell counts
        _draw_split_dot_cell(
            ax=ax,
            x=1,
            y=1,
            w=1,
            h=1,
            val1=0.5,
            val2=np.nan,
            cmap='RdBu_r',
            vmin=-2,
            vmax=2,
            cell_count1=0,
            cell_count2=0,
            global_max_count=200
        )
        
        # Test with custom colormap and no global_max_count
        custom_cmap = plt.cm.get_cmap('Reds')
        _draw_split_dot_cell(
            ax=ax,
            x=2,
            y=0,
            w=1,
            h=1,
            val1=1.0,
            val2=1.0,
            cmap=custom_cmap,
            vmin=0,
            vmax=3,
            cell_count1=100,
            cell_count2=50
        )
        
        # Check that wedges were added to the axis
        assert len(ax.patches) == 10  # 2 wedges per cell * 5 cells
        
        # Verify that the patches are wedges
        for patch in ax.patches:
            assert isinstance(patch, mpl.patches.Wedge)
        
        plt.close(fig)


class TestHeatmapWithFoldChangeMode:
    """Tests for the heatmap function with fold_change_mode=True."""
    
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
            
            # Make sure we have condition data for fold change tests
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
    
    def test_heatmap_fold_change_mode_basic(self):
        """Test the heatmap function with fold_change_mode=True."""
        # Skip if kompot_run_history doesn't exist
        if 'kompot_run_history' not in self.adata.uns:
            pytest.skip("kompot_run_history not found in adata.uns")
        
        # Make sure we have a 'condition' column in obs
        if 'condition' not in self.adata.obs.columns:
            self.adata.obs['condition'] = ['A'] * (self.adata.n_obs // 2) + ['B'] * (self.adata.n_obs // 2)
        
        # Test fold change heatmap with explicit genes
        test_genes = [f'gene_{i}' for i in range(5)]
        result = heatmap(
            self.adata,
            genes=test_genes,
            score_key='test_score',
            groupby='group',
            condition_column='condition',
            condition1='A',
            condition2='B',
            fold_change_mode=True,  # Use fold change mode
            cmap='RdBu_r',  # Use a diverging colormap appropriate for fold changes
            return_fig=True
        )
        
        # Check that it works
        assert result is not None
        assert isinstance(result, tuple) and len(result) >= 2
        fig, ax = result[:2]
        
        # Check that the figure and axis exist
        assert fig is not None
        assert ax is not None
        
        # Check that we have the right number of cells
        assert len(ax.patches) > 0
    
    def test_heatmap_fold_change_mode_custom_colormap(self):
        """Test the heatmap function with fold_change_mode=True and custom colormap."""
        # Skip if kompot_run_history doesn't exist
        if 'kompot_run_history' not in self.adata.uns:
            pytest.skip("kompot_run_history not found in adata.uns")
        
        # Test fold change heatmap with custom coolwarm colormap
        custom_cmap = plt.cm.get_cmap('coolwarm')
        result = heatmap(
            self.adata,
            n_top_genes=5,
            score_key='test_score',
            groupby='group',
            condition_column='condition',
            condition1='A',
            condition2='B',
            fold_change_mode=True,
            cmap=custom_cmap,
            vcenter=0,  # Explicitly set center at 0
            vmin=-1.5,  # Custom limits
            vmax=1.5,
            return_fig=True
        )
        
        # Check that it works
        assert result is not None
        assert isinstance(result, tuple) and len(result) >= 2
    
    def test_heatmap_fold_change_mode_with_standard_scale_warning(self):
        """Test that standard_scale is ignored in fold_change_mode."""
        # Skip if kompot_run_history doesn't exist
        if 'kompot_run_history' not in self.adata.uns:
            pytest.skip("kompot_run_history not found in adata.uns")
        
        # Test with fold_change_mode=True and standard_scale='var'
        # This should print a warning that standard_scale is ignored
        result = heatmap(
            self.adata,
            n_top_genes=5,
            score_key='test_score',
            groupby='group',
            condition_column='condition',
            condition1='A',
            condition2='B',
            fold_change_mode=True,
            standard_scale='var',  # This should be ignored
            return_fig=True
        )
        
        # Check that it works despite the warning
        assert result is not None
        assert isinstance(result, tuple) and len(result) >= 2
    
    def test_heatmap_fold_change_mode_with_clustering(self):
        """Test the heatmap with fold_change_mode=True and clustering."""
        # Skip if kompot_run_history doesn't exist
        if 'kompot_run_history' not in self.adata.uns:
            pytest.skip("kompot_run_history not found in adata.uns")
        
        # Test fold change heatmap with clustering
        result = heatmap(
            self.adata,
            n_top_genes=5,
            score_key='test_score',
            groupby='group',
            condition_column='condition',
            condition1='A',
            condition2='B',
            fold_change_mode=True,
            dendrogram=True,  # Show dendrograms
            cluster_rows=True,
            cluster_cols=True,
            return_fig=True
        )
        
        # Check that it works and returns dendrogram axes
        assert result is not None
        assert isinstance(result, tuple)
        # Handle unpacking based on namedtuple pattern
        fig, ax, dendrogram_axes = result[:3]
        
        # Check that dendrograms were created
        assert 'row' in dendrogram_axes or 'col' in dendrogram_axes


class TestHeatmapWithSplitDotMode:
    """Tests for the heatmap function with split_dot_mode=True."""
    
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
            
            # Make sure we have condition data for split dot tests
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
    
    def test_heatmap_split_dot_mode_basic(self):
        """Test the heatmap function with split_dot_mode=True."""
        # Skip if kompot_run_history doesn't exist
        if 'kompot_run_history' not in self.adata.uns:
            pytest.skip("kompot_run_history not found in adata.uns")
        
        # Make sure we have a 'condition' column in obs
        if 'condition' not in self.adata.obs.columns:
            self.adata.obs['condition'] = ['A'] * (self.adata.n_obs // 2) + ['B'] * (self.adata.n_obs // 2)
        
        # Test split dot heatmap with explicit genes
        test_genes = [f'gene_{i}' for i in range(5)]
        result = heatmap(
            self.adata,
            genes=test_genes,
            score_key='test_score',
            groupby='group',
            condition_column='condition',
            condition1='A',
            condition2='B',
            split_dot_mode=True,  # Use split dot mode
            cmap='Reds',  # Use a sequential colormap 
            return_fig=True
        )
        
        # Check that it works
        assert result is not None
        assert isinstance(result, tuple) and len(result) >= 2
        fig, ax = result[:2]
        
        # Check that the figure and axis exist
        assert fig is not None
        assert ax is not None
        
        # Check that we have patches (wedges for the dots)
        assert len(ax.patches) > 0
    
    def test_heatmap_split_dot_mode_with_max_cell_count(self):
        """Test the heatmap function with split_dot_mode=True and max_cell_count."""
        # Skip if kompot_run_history doesn't exist
        if 'kompot_run_history' not in self.adata.uns:
            pytest.skip("kompot_run_history not found in adata.uns")
        
        # Test split dot heatmap with max_cell_count parameter
        result = heatmap(
            self.adata,
            n_top_genes=5,
            score_key='test_score',
            groupby='group',
            condition_column='condition',
            condition1='A',
            condition2='B',
            split_dot_mode=True,
            max_cell_count=50,  # Cap max cell count for dot sizing
            return_fig=True
        )
        
        # Check that it works
        assert result is not None
        assert isinstance(result, tuple) and len(result) >= 2
    
    def test_heatmap_split_dot_mode_with_standard_scale(self):
        """Test the split dot mode with z-scoring."""
        # Skip if kompot_run_history doesn't exist
        if 'kompot_run_history' not in self.adata.uns:
            pytest.skip("kompot_run_history not found in adata.uns")
        
        # Test with split_dot_mode=True and standard_scale='var'
        # This should apply z-scoring to the expression values
        result = heatmap(
            self.adata,
            n_top_genes=5,
            score_key='test_score',
            groupby='group',
            condition_column='condition',
            condition1='A',
            condition2='B',
            split_dot_mode=True,
            standard_scale='var',  # Apply z-scoring
            cmap='RdBu_r',  # Diverging colormap for z-scored data
            return_fig=True
        )
        
        # Check that it works
        assert result is not None
        assert isinstance(result, tuple) and len(result) >= 2
    
    def test_heatmap_split_dot_mode_with_clustering(self):
        """Test the heatmap with split_dot_mode=True and clustering."""
        # Skip if kompot_run_history doesn't exist
        if 'kompot_run_history' not in self.adata.uns:
            pytest.skip("kompot_run_history not found in adata.uns")
        
        # Test split dot heatmap with clustering
        result = heatmap(
            self.adata,
            n_top_genes=5,
            score_key='test_score',
            groupby='group',
            condition_column='condition',
            condition1='A',
            condition2='B',
            split_dot_mode=True,
            dendrogram=True,  # Show dendrograms
            cluster_rows=True,
            cluster_cols=True,
            return_fig=True
        )
        
        # Check that it works and returns dendrogram axes
        assert result is not None
        assert isinstance(result, tuple) and len(result) >= 3
    
    def test_heatmap_split_dot_mode_with_uneven_groups(self):
        """Test split dot mode with uneven group sizes."""
        # Skip if kompot_run_history doesn't exist
        if 'kompot_run_history' not in self.adata.uns:
            pytest.skip("kompot_run_history not found in adata.uns")
        
        # Create uneven groups
        group_counts = {
            'A': 20,
            'B': 50,
            'C': 5,
            'D': 25
        }
        
        # Create a new group column with these counts
        groups = []
        for group, count in group_counts.items():
            groups.extend([group] * count)
        
        # Ensure the total matches n_obs
        if len(groups) < self.adata.n_obs:
            groups.extend(['D'] * (self.adata.n_obs - len(groups)))
        elif len(groups) > self.adata.n_obs:
            groups = groups[:self.adata.n_obs]
            
        self.adata.obs['uneven_groups'] = groups
        
        # Test with uneven group sizes
        result = heatmap(
            self.adata,
            n_top_genes=5,
            score_key='test_score',
            groupby='uneven_groups',
            condition_column='condition',
            condition1='A',
            condition2='B',
            split_dot_mode=True,
            return_fig=True
        )
        
        # Check that it works
        assert result is not None
        assert isinstance(result, tuple) and len(result) >= 2


class TestHeatmapEdgeCases:
    """Test edge cases and combinations of heatmap parameters."""
    
    def setup_method(self):
        """Set up test data."""
        self.adata = create_test_anndata()
        
        # Add condition column
        self.adata.obs['condition'] = ['A'] * (self.adata.n_obs // 2) + ['B'] * (self.adata.n_obs // 2)
        
        # Add test scores
        self.adata.var['test_score'] = np.random.rand(self.adata.n_vars)
        self.adata.var['test_lfc'] = np.random.randn(self.adata.n_vars)
        
        plt.clf()  # Clear any existing plots
    
    def teardown_method(self):
        """Clean up after tests."""
        plt.clf()  # Clear any created plots
        plt.close('all')
    
    def test_fold_change_with_percentile_limits(self):
        """Test fold change mode with percentile-based limits."""
        result = heatmap(
            self.adata,
            n_top_genes=5,
            score_key='test_score',
            groupby='group',
            condition_column='condition',
            condition1='A',
            condition2='B',
            fold_change_mode=True,
            vmin='p5',    # 5th percentile
            vmax='p95',   # 95th percentile
            return_fig=True
        )
        
        # Check that it works
        assert result is not None
        assert isinstance(result, tuple) and len(result) >= 2
    
    def test_fold_change_with_title_and_layout_config(self):
        """Test fold change mode with custom title and layout config."""
        layout_config = {
            'gene_label_space': 4.0,       # Increase space for gene labels
            'colorbar_height': 0.6,        # Increase colorbar height
            'legend_fontsize': 14          # Increase legend font size
        }
        
        result = heatmap(
            self.adata,
            n_top_genes=5,
            score_key='test_score',
            groupby='group',
            condition_column='condition',
            condition1='A',
            condition2='B',
            fold_change_mode=True,
            title="Custom Fold Change Heatmap",
            layout_config=layout_config,
            return_fig=True
        )
        
        # Check that it works
        assert result is not None
        assert isinstance(result, tuple) and len(result) >= 2
    
    def test_fold_change_with_colorbar_kwargs(self):
        """Test fold change mode with custom colorbar kwargs."""
        colorbar_kwargs = {
            'label_kwargs': {
                'fontsize': 14,
                'fontweight': 'bold',
                'color': 'darkblue'
            },
            'orientation': 'vertical'
        }
        
        result = heatmap(
            self.adata,
            n_top_genes=5,
            score_key='test_score',
            groupby='group',
            condition_column='condition',
            condition1='A',
            condition2='B',
            fold_change_mode=True,
            colorbar_kwargs=colorbar_kwargs,
            colorbar_title="Custom Fold Change Title",
            return_fig=True
        )
        
        # Check that it works
        assert result is not None
        assert isinstance(result, tuple) and len(result) >= 2
    
    def test_split_dot_mode_with_custom_conditions(self):
        """Test split dot mode with custom condition names."""
        result = heatmap(
            self.adata,
            n_top_genes=5,
            score_key='test_score',
            groupby='group',
            condition_column='condition',
            condition1='A',
            condition2='B',
            condition1_name='Control Group',  # Custom display name
            condition2_name='Treatment Group', # Custom display name
            split_dot_mode=True,
            return_fig=True
        )
        
        # Check that it works
        assert result is not None
        assert isinstance(result, tuple) and len(result) >= 2
    
    def test_invalid_modes_combination(self):
        """Test that fold_change_mode and split_dot_mode can't both be True."""
        # This should work with a warning (fold_change_mode takes precedence)
        result = heatmap(
            self.adata,
            n_top_genes=5,
            score_key='test_score',
            groupby='group',
            condition_column='condition',
            condition1='A',
            condition2='B',
            fold_change_mode=True,
            split_dot_mode=True,  # This will be ignored
            return_fig=True
        )
        
        # Check that it still works (should fall back to fold_change_mode)
        assert result is not None
        assert isinstance(result, tuple) and len(result) >= 2