"""Tests for the volcano_de plotting function."""

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


def create_test_anndata(n_cells=100, n_genes=20, with_categorical=False, with_continuous=False):
    """Create a test AnnData object with various data types for testing volcano_de."""
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
    
    # Add categorical column if requested
    if with_categorical:
        categories = ['category_' + str(i % 3 + 1) for i in range(n_genes)]
        var['gene_category'] = categories
        
    # Add continuous column if requested
    if with_continuous:
        var['gene_expression'] = np.random.normal(5, 2, n_genes)
    
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


class TestVolcanoDE:
    """Tests for the volcano_de function."""
    
    def setup_method(self):
        """Set up test data for each test."""
        self.adata = create_test_anndata(with_categorical=True, with_continuous=True)
        plt.clf()  # Clear any existing plots
    
    def teardown_method(self):
        """Clean up after tests."""
        plt.clf()  # Clear any created plots
        plt.close('all')
    
    def test_basic_functionality(self):
        """Test basic functionality of volcano_de."""
        fig, ax = volcano_de(
            self.adata,
            lfc_key='de_run1_mean_lfc_A_to_B',
            score_key='de_run1_mahalanobis_A_to_B',
            return_fig=True
        )
        assert fig is not None
        assert ax is not None
        
        # Test with run_id parameter
        fig, ax = volcano_de(
            self.adata,
            run_id=0,
            return_fig=True
        )
        assert fig is not None
        assert ax is not None
    
    def test_highlight_genes_parameter(self):
        """Test different formats of highlight_genes parameter."""
        # Test with list of gene names
        genes_to_highlight = self.adata.var_names[:5].tolist()
        fig, ax = volcano_de(
            self.adata,
            lfc_key='de_run1_mean_lfc_A_to_B',
            score_key='de_run1_mahalanobis_A_to_B',
            highlight_genes=genes_to_highlight,
            return_fig=True
        )
        assert fig is not None
        assert ax is not None
        
        # Test with dict mapping genes to colors
        gene_color_dict = {gene: '#FF5733' for gene in self.adata.var_names[:3]}
        fig, ax = volcano_de(
            self.adata,
            lfc_key='de_run1_mean_lfc_A_to_B',
            score_key='de_run1_mahalanobis_A_to_B',
            highlight_genes=gene_color_dict,
            return_fig=True
        )
        assert fig is not None
        assert ax is not None
        
        # Test with list of dictionaries format
        highlight_groups = [
            {
                'genes': self.adata.var_names[:3].tolist(),
                'name': 'Group 1',
                'color': '#FF5733'
            },
            {
                'genes': self.adata.var_names[5:8].tolist(),
                'name': 'Group 2',
                'color': '#33FF57'
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
        
        # Test with list of lists format
        list_of_lists = [
            self.adata.var_names[:3].tolist(),
            self.adata.var_names[5:8].tolist()
        ]
        fig, ax = volcano_de(
            self.adata,
            lfc_key='de_run1_mean_lfc_A_to_B',
            score_key='de_run1_mahalanobis_A_to_B',
            highlight_genes=list_of_lists,
            return_fig=True
        )
        assert fig is not None
        assert ax is not None
        
        # Test with single string (gene name)
        single_gene = self.adata.var_names[0]
        fig, ax = volcano_de(
            self.adata,
            lfc_key='de_run1_mean_lfc_A_to_B',
            score_key='de_run1_mahalanobis_A_to_B',
            highlight_genes=single_gene,
            return_fig=True
        )
        assert fig is not None
        assert ax is not None
    
    def test_categorical_background(self):
        """Test volcano_de with categorical background coloring."""
        fig, ax = volcano_de(
            self.adata,
            lfc_key='de_run1_mean_lfc_A_to_B',
            score_key='de_run1_mahalanobis_A_to_B',
            background_color_key='gene_category',
            return_fig=True
        )
        assert fig is not None
        assert ax is not None
        
        # Test with custom colormap
        fig, ax = volcano_de(
            self.adata,
            lfc_key='de_run1_mean_lfc_A_to_B',
            score_key='de_run1_mahalanobis_A_to_B',
            background_color_key='gene_category',
            background_cmap='Set1',
            return_fig=True
        )
        assert fig is not None
        assert ax is not None
        
        # Test with color_discrete_map
        color_map = {
            'category_1': '#FF5733',
            'category_2': '#33FF57',
            'category_3': '#3357FF'
        }
        fig, ax = volcano_de(
            self.adata,
            lfc_key='de_run1_mean_lfc_A_to_B',
            score_key='de_run1_mahalanobis_A_to_B',
            background_color_key='gene_category',
            color_discrete_map=color_map,
            return_fig=True
        )
        assert fig is not None
        assert ax is not None
        
        # Test with custom colormap object
        colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # R, G, B
        custom_cmap = ListedColormap(colors, name='custom')
        fig, ax = volcano_de(
            self.adata,
            lfc_key='de_run1_mean_lfc_A_to_B',
            score_key='de_run1_mahalanobis_A_to_B',
            background_color_key='gene_category',
            background_cmap=custom_cmap,
            return_fig=True
        )
        assert fig is not None
        assert ax is not None
    
    def test_continuous_background(self):
        """Test volcano_de with continuous background coloring."""
        fig, ax = volcano_de(
            self.adata,
            lfc_key='de_run1_mean_lfc_A_to_B',
            score_key='de_run1_mahalanobis_A_to_B',
            background_color_key='gene_expression',
            return_fig=True
        )
        assert fig is not None
        assert ax is not None
        
        # Test with custom colormap
        fig, ax = volcano_de(
            self.adata,
            lfc_key='de_run1_mean_lfc_A_to_B',
            score_key='de_run1_mahalanobis_A_to_B',
            background_color_key='gene_expression',
            background_cmap='viridis',
            return_fig=True
        )
        assert fig is not None
        assert ax is not None
        
        # Test with vmin, vmax, vcenter
        fig, ax = volcano_de(
            self.adata,
            lfc_key='de_run1_mean_lfc_A_to_B',
            score_key='de_run1_mahalanobis_A_to_B',
            background_color_key='gene_expression',
            vmin=3, vmax=7, vcenter=5,
            return_fig=True
        )
        assert fig is not None
        assert ax is not None
        
        # Test with percentile strings
        fig, ax = volcano_de(
            self.adata,
            lfc_key='de_run1_mean_lfc_A_to_B',
            score_key='de_run1_mahalanobis_A_to_B',
            background_color_key='gene_expression',
            vmin='p10', vmax='p90',
            return_fig=True
        )
        assert fig is not None
        assert ax is not None
    
    def test_layout_and_styling_options(self):
        """Test layout and styling options of volcano_de."""
        # Test with custom figure size
        # Note: The actual size may differ due to internal adjustments for legends
        fig, ax = volcano_de(
            self.adata,
            lfc_key='de_run1_mean_lfc_A_to_B',
            score_key='de_run1_mahalanobis_A_to_B',
            figsize=(12, 10),
            return_fig=True
        )
        assert fig is not None
        assert ax is not None
        # Skip exact size check as volcano_de may adjust size for legend
        
        # Test with custom title and axis labels
        fig, ax = volcano_de(
            self.adata,
            lfc_key='de_run1_mean_lfc_A_to_B',
            score_key='de_run1_mahalanobis_A_to_B',
            title='Custom Volcano Plot',
            xlabel='Custom X Label',
            ylabel='Custom Y Label',
            return_fig=True
        )
        assert fig is not None
        assert ax is not None
        assert ax.get_title() == 'Custom Volcano Plot'
        assert ax.get_xlabel() == 'Custom X Label'
        assert ax.get_ylabel() == 'Custom Y Label'
        
        # Test with custom tick settings
        fig, ax = volcano_de(
            self.adata,
            lfc_key='de_run1_mean_lfc_A_to_B',
            score_key='de_run1_mahalanobis_A_to_B',
            n_x_ticks=5,
            n_y_ticks=5,
            return_fig=True
        )
        assert fig is not None
        assert ax is not None
        
        # Test with grid off
        fig, ax = volcano_de(
            self.adata,
            lfc_key='de_run1_mean_lfc_A_to_B',
            score_key='de_run1_mahalanobis_A_to_B',
            grid=False,
            return_fig=True
        )
        assert fig is not None
        assert ax is not None
        
        # Test with custom grid settings
        fig, ax = volcano_de(
            self.adata,
            lfc_key='de_run1_mean_lfc_A_to_B',
            score_key='de_run1_mahalanobis_A_to_B',
            grid=True,
            grid_kwargs={'alpha': 0.5, 'linestyle': '--'},
            return_fig=True
        )
        assert fig is not None
        assert ax is not None
    
    def test_legend_options(self):
        """Test legend options for volcano_de."""
        # Test with legend off
        fig, ax = volcano_de(
            self.adata,
            lfc_key='de_run1_mean_lfc_A_to_B',
            score_key='de_run1_mahalanobis_A_to_B',
            show_legend=False,
            return_fig=True
        )
        assert fig is not None
        assert ax is not None
        
        # Test with legend location
        fig, ax = volcano_de(
            self.adata,
            lfc_key='de_run1_mean_lfc_A_to_B',
            score_key='de_run1_mahalanobis_A_to_B',
            legend_loc='upper right',
            return_fig=True
        )
        assert fig is not None
        assert ax is not None
        
        # Test with legend font size
        fig, ax = volcano_de(
            self.adata,
            lfc_key='de_run1_mean_lfc_A_to_B',
            score_key='de_run1_mahalanobis_A_to_B',
            legend_fontsize=14,
            legend_title_fontsize=16,
            return_fig=True
        )
        assert fig is not None
        assert ax is not None
        
        # Test with legend columns
        fig, ax = volcano_de(
            self.adata,
            lfc_key='de_run1_mean_lfc_A_to_B',
            score_key='de_run1_mahalanobis_A_to_B',
            legend_ncol=2,
            return_fig=True
        )
        assert fig is not None
        assert ax is not None
    
    def test_point_and_text_styling(self):
        """Test point and text styling options."""
        # Test with custom point sizes and colors
        fig, ax = volcano_de(
            self.adata,
            lfc_key='de_run1_mean_lfc_A_to_B',
            score_key='de_run1_mahalanobis_A_to_B',
            point_size=10,
            color_up='#FF5733',
            color_down='#3357FF',
            color_background='#CCCCCC',
            alpha_background=0.7,
            return_fig=True
        )
        assert fig is not None
        assert ax is not None
        
        # Test with text options
        fig, ax = volcano_de(
            self.adata,
            lfc_key='de_run1_mean_lfc_A_to_B',
            score_key='de_run1_mahalanobis_A_to_B',
            font_size=12,
            text_offset=(5, 5),
            text_kwargs={'fontweight': 'bold', 'color': 'red'},
            return_fig=True
        )
        assert fig is not None
        assert ax is not None
        
        # Test with show_names=False
        fig, ax = volcano_de(
            self.adata,
            lfc_key='de_run1_mean_lfc_A_to_B',
            score_key='de_run1_mahalanobis_A_to_B',
            show_names=False,
            return_fig=True
        )
        assert fig is not None
        assert ax is not None
    
    def test_show_names_parameter(self):
        """Test the show_names parameter with different input types."""
        # Test with show_names=True (default behavior - shows highlighted gene names)
        fig, ax = volcano_de(
            self.adata,
            lfc_key='de_run1_mean_lfc_A_to_B',
            score_key='de_run1_mahalanobis_A_to_B',
            show_names=True,
            n_top_genes=5,
            return_fig=True
        )
        assert fig is not None
        assert ax is not None
        
        # Check that text annotations were added (for highlighted genes)
        annotations = [child for child in ax.get_children() if hasattr(child, 'get_text')]
        # Should have some annotations since show_names=True and we have highlighted genes
        assert len(annotations) > 0
        
        # Test with show_names=False (no gene names shown)
        fig, ax = volcano_de(
            self.adata,
            lfc_key='de_run1_mean_lfc_A_to_B',
            score_key='de_run1_mahalanobis_A_to_B',
            show_names=False,
            n_top_genes=5,
            return_fig=True
        )
        assert fig is not None
        assert ax is not None
        
        # Check that no text annotations were added
        annotations = [child for child in ax.get_children() if hasattr(child, 'get_text')]
        # Should have no gene name annotations when show_names=False
        gene_annotations = [ann for ann in annotations if hasattr(ann, 'get_text') and ann.get_text() in self.adata.var_names]
        assert len(gene_annotations) == 0
        
        # Test with show_names as list of specific gene names
        genes_to_show = ['gene_0', 'gene_5', 'gene_10']
        fig, ax = volcano_de(
            self.adata,
            lfc_key='de_run1_mean_lfc_A_to_B',
            score_key='de_run1_mahalanobis_A_to_B',
            show_names=genes_to_show,
            n_top_genes=3,  # Different from genes_to_show to test independence
            return_fig=True
        )
        assert fig is not None
        assert ax is not None
        
        # Check that annotations exist for the specified genes
        annotations = [child for child in ax.get_children() if hasattr(child, 'get_text')]
        annotation_texts = [ann.get_text() for ann in annotations if hasattr(ann, 'get_text')]
        
        # All specified genes should be annotated
        for gene in genes_to_show:
            assert gene in annotation_texts
        
        # Test with show_names as list containing non-existent genes
        genes_with_invalid = ['gene_0', 'nonexistent_gene', 'gene_5']
        fig, ax = volcano_de(
            self.adata,
            lfc_key='de_run1_mean_lfc_A_to_B',
            score_key='de_run1_mahalanobis_A_to_B',
            show_names=genes_with_invalid,
            return_fig=True
        )
        assert fig is not None
        assert ax is not None
        
        # Check that valid genes are annotated, invalid ones are ignored
        annotations = [child for child in ax.get_children() if hasattr(child, 'get_text')]
        annotation_texts = [ann.get_text() for ann in annotations if hasattr(ann, 'get_text')]
        
        assert 'gene_0' in annotation_texts
        assert 'gene_5' in annotation_texts
        assert 'nonexistent_gene' not in annotation_texts
        
        # Test show_names list with highlight_genes - should show both
        highlight_genes_list = ['gene_1', 'gene_2', 'gene_3']
        show_names_list = ['gene_0', 'gene_4', 'gene_8']  # Different genes
        
        fig, ax = volcano_de(
            self.adata,
            lfc_key='de_run1_mean_lfc_A_to_B',
            score_key='de_run1_mahalanobis_A_to_B',
            highlight_genes=highlight_genes_list,
            show_names=show_names_list,
            return_fig=True
        )
        assert fig is not None
        assert ax is not None
        
        # Check that genes from show_names list are annotated (regardless of highlighting)
        annotations = [child for child in ax.get_children() if hasattr(child, 'get_text')]
        annotation_texts = [ann.get_text() for ann in annotations if hasattr(ann, 'get_text')]
        
        # All genes in show_names should be annotated
        for gene in show_names_list:
            assert gene in annotation_texts
        
        # Test with empty show_names list
        fig, ax = volcano_de(
            self.adata,
            lfc_key='de_run1_mean_lfc_A_to_B',
            score_key='de_run1_mahalanobis_A_to_B',
            show_names=[],
            n_top_genes=5,
            return_fig=True
        )
        assert fig is not None
        assert ax is not None
        
        # Should have no gene annotations with empty list
        annotations = [child for child in ax.get_children() if hasattr(child, 'get_text')]
        gene_annotations = [ann for ann in annotations if hasattr(ann, 'get_text') and ann.get_text() in self.adata.var_names]
        assert len(gene_annotations) == 0
        
        # Test show_names list with custom highlight_genes (dict format)
        highlight_dict = {'gene_1': '#FF0000', 'gene_2': '#00FF00'}
        show_names_list = ['gene_0', 'gene_1', 'gene_9']  # gene_1 overlaps with highlight
        
        fig, ax = volcano_de(
            self.adata,
            lfc_key='de_run1_mean_lfc_A_to_B',
            score_key='de_run1_mahalanobis_A_to_B',
            highlight_genes=highlight_dict,
            show_names=show_names_list,
            return_fig=True
        )
        assert fig is not None
        assert ax is not None
        
        # All genes in show_names should be annotated
        annotations = [child for child in ax.get_children() if hasattr(child, 'get_text')]
        annotation_texts = [ann.get_text() for ann in annotations if hasattr(ann, 'get_text')]
        
        for gene in show_names_list:
            assert gene in annotation_texts
    
    def test_key_inference(self):
        """Test key inference functionality."""
        # Test inference with run_id
        lfc_key, score_key = _infer_de_keys(self.adata, run_id=0)
        assert lfc_key == 'de_run1_mean_lfc_A_to_B'
        assert score_key == 'de_run1_mahalanobis_A_to_B'
        
        # Test inference with negative run_id
        lfc_key, score_key = _infer_de_keys(self.adata, run_id=-1)
        assert lfc_key == 'de_run1_mean_lfc_A_to_B'
        assert score_key == 'de_run1_mahalanobis_A_to_B'
        
        # Test direct key specification
        lfc_key, score_key = _infer_de_keys(self.adata, lfc_key='mean_lfc', score_key='mahalanobis')
        assert lfc_key == 'mean_lfc'
        assert score_key == 'mahalanobis'
    
    def test_error_handling(self):
        """Test error handling in volcano_de."""
        # Test with non-existent keys
        with pytest.raises(KeyError):
            volcano_de(
                self.adata,
                lfc_key='nonexistent_key',
                score_key='also_nonexistent',
                return_fig=True
            )
        
        # Test with invalid highlight_genes entries
        fig, ax = volcano_de(
            self.adata,
            lfc_key='de_run1_mean_lfc_A_to_B',
            score_key='de_run1_mahalanobis_A_to_B',
            highlight_genes=['nonexistent_gene_1', 'nonexistent_gene_2'],
            return_fig=True
        )
        assert fig is not None
        assert ax is not None
        
        # Test with empty highlight groups
        empty_highlight_groups = [
            {
                'genes': [],
                'name': 'Empty Group',
                'color': '#FF5733'
            }
        ]
        fig, ax = volcano_de(
            self.adata,
            lfc_key='de_run1_mean_lfc_A_to_B',
            score_key='de_run1_mahalanobis_A_to_B',
            highlight_genes=empty_highlight_groups,
            return_fig=True
        )
        assert fig is not None
        assert ax is not None
    
    def test_with_external_ax(self):
        """Test using an external axes for plotting."""
        fig, external_ax = plt.subplots(figsize=(10, 8))
        
        volcano_de(
            self.adata,
            lfc_key='de_run1_mean_lfc_A_to_B',
            score_key='de_run1_mahalanobis_A_to_B',
            ax=external_ax
        )
        
        # Check that the external ax has been used
        assert len(external_ax.get_children()) > 0
        
        # Now test with background color key
        fig, external_ax = plt.subplots(figsize=(10, 8))
        
        volcano_de(
            self.adata,
            lfc_key='de_run1_mean_lfc_A_to_B',
            score_key='de_run1_mahalanobis_A_to_B',
            background_color_key='gene_expression',
            ax=external_ax
        )
        
        # Check that the external ax has been used
        assert len(external_ax.get_children()) > 0
    
    def test_save_functionality(self):
        """Test save functionality (without actually saving)."""
        import tempfile
        import os
        
        # Create a temporary filename
        with tempfile.NamedTemporaryFile(suffix='.png') as temp:
            temp_filename = temp.name
        
        # The file should not exist yet (or anymore)
        assert not os.path.exists(temp_filename)
        
        # Test the save parameter
        volcano_de(
            self.adata,
            lfc_key='de_run1_mean_lfc_A_to_B',
            score_key='de_run1_mahalanobis_A_to_B',
            save=temp_filename
        )
        
        # Check if the file was created
        assert os.path.exists(temp_filename)
        
        # Clean up
        os.remove(temp_filename)