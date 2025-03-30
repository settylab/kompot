"""Advanced tests for the multi_volcano_da plotting function.

This file extends the basic tests in test_plot_functions.py with more detailed
tests focusing on specific features of multi_volcano_da.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List

from kompot.anndata.differential_abundance import compute_differential_abundance
from kompot.plot.volcano.multi_da import multi_volcano_da


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
    cell_types = np.random.choice(['Type1', 'Type2', 'Type3'], size=n_cells)
    
    # Create embedding
    obsm = {
        'DM_EigenVectors': np.random.normal(0, 1, (n_cells, 10)),
        'X_pca': np.random.normal(0, 1, (n_cells, 2))
    }
    
    # Create observation dataframe
    obs = pd.DataFrame({
        'group': groups,
        'cell_type': cell_types
    })
    
    # Create var_names
    var_names = [f'gene_{i}' for i in range(n_genes)]
    
    # Create var DataFrame with var_names as index
    var = pd.DataFrame(index=var_names)
    
    return anndata.AnnData(X=X, obs=obs, var=var, obsm=obsm)


def test_multi_volcano_da_background_plot_kde():
    """Test multi_volcano_da function with KDE background density plots."""
    # Create test data
    adata = create_test_anndata()
    
    # Run differential abundance analysis
    compute_differential_abundance(
        adata,
        groupby='group',
        condition1='A',
        condition2='B',
        log_fold_change_threshold=1.0,
        pvalue_threshold=0.05
    )
    
    # Test with KDE background plot
    fig, axes = multi_volcano_da(
        adata,
        groupby='cell_type',
        run_id=-1,
        background_plot="kde",
        background_alpha=0.7,
        background_color="#F0F0F0",
        return_fig=True
    )
    
    # Check results
    assert fig is not None
    assert isinstance(axes, list)
    assert len(axes) == 3  # Should have 3 axes for Type1, Type2, Type3
    
    plt.close('all')


def test_multi_volcano_da_background_plot_violin():
    """Test multi_volcano_da function with violin background density plots."""
    # Create test data
    adata = create_test_anndata()
    
    # Run differential abundance analysis
    compute_differential_abundance(
        adata,
        groupby='group',
        condition1='A',
        condition2='B',
        log_fold_change_threshold=1.0,
        pvalue_threshold=0.05
    )
    
    # Test with violin background plot
    fig, axes = multi_volcano_da(
        adata,
        groupby='cell_type',
        run_id=-1,
        background_plot="violin",
        background_alpha=0.7,
        background_color="#F0F0F0",
        return_fig=True
    )
    
    # Check results
    assert fig is not None
    assert isinstance(axes, list)
    assert len(axes) == 3  # Should have 3 axes for Type1, Type2, Type3
    
    plt.close('all')


def test_multi_volcano_da_background_plot_with_custom_kwargs():
    """Test multi_volcano_da function with custom background plot parameters."""
    # Create test data
    adata = create_test_anndata()
    
    # Run differential abundance analysis
    compute_differential_abundance(
        adata,
        groupby='group',
        condition1='A',
        condition2='B',
        log_fold_change_threshold=1.0,
        pvalue_threshold=0.05
    )
    
    # Test KDE with custom background kwargs
    kde_kwargs = {
        'bw_method': 'silverman',
        'show_2d_kde': True,
        'contour_levels': 3, 
        'contour_alpha': 0.3,
        'contour_cmap': 'Greens'
    }
    
    fig, axes = multi_volcano_da(
        adata,
        groupby='cell_type',
        run_id=-1,
        background_plot="kde",
        background_kwargs=kde_kwargs,
        return_fig=True
    )
    
    # Check results
    assert fig is not None
    assert isinstance(axes, list)
    
    plt.close('all')
    
    # Test violin with custom background kwargs
    violin_kwargs = {
        'showmeans': True,
        'showmedians': True,
        'showextrema': True
    }
    
    fig, axes = multi_volcano_da(
        adata,
        groupby='cell_type',
        run_id=-1,
        background_plot="violin",
        background_kwargs=violin_kwargs,
        return_fig=True
    )
    
    # Check results
    assert fig is not None
    assert isinstance(axes, list)
    
    plt.close('all')


def test_multi_volcano_da_numeric_color_options():
    """Test multi_volcano_da function with numeric coloring options."""
    # Create test data
    adata = create_test_anndata(n_cells=200)  # More cells for better testing
    
    # Run differential abundance analysis
    compute_differential_abundance(
        adata,
        groupby='group',
        condition1='A',
        condition2='B',
        log_fold_change_threshold=1.0,
        pvalue_threshold=0.05
    )
    
    # Add custom numeric column for coloring
    adata.obs['custom_numeric'] = np.random.randn(adata.n_obs)
    
    # Test with numeric color and diverging colormap centered at 0
    fig, axes = multi_volcano_da(
        adata,
        groupby='cell_type',
        run_id=-1,
        color='custom_numeric',
        cmap='RdBu_r',
        vcenter=0,
        vmin=-2,
        vmax=2,
        return_fig=True
    )
    
    # Check results
    assert fig is not None
    assert isinstance(axes, list)
    
    plt.close('all')


def test_multi_volcano_da_categorical_color_options():
    """Test multi_volcano_da function with categorical coloring options."""
    # Create test data
    adata = create_test_anndata(n_cells=200)  # More cells for better testing
    
    # Run differential abundance analysis
    compute_differential_abundance(
        adata,
        groupby='group',
        condition1='A',
        condition2='B',
        log_fold_change_threshold=1.0,
        pvalue_threshold=0.05
    )
    
    # Add custom categorical column for coloring
    categories = ['Cat1', 'Cat2', 'Cat3']
    adata.obs['custom_category'] = pd.Categorical(
        np.random.choice(categories, size=adata.n_obs),
        categories=categories
    )
    
    # Add color mapping to adata.uns
    adata.uns['custom_category_colors'] = ['red', 'blue', 'green']
    
    # Test with categorical color
    fig, axes = multi_volcano_da(
        adata,
        groupby='cell_type',
        run_id=-1,
        color='custom_category',
        return_fig=True
    )
    
    # Check results
    assert fig is not None
    assert isinstance(axes, list)
    
    plt.close('all')


def test_multi_volcano_da_custom_layout():
    """Test multi_volcano_da function with customized layout settings."""
    # Create test data
    adata = create_test_anndata()
    
    # Run differential abundance analysis
    compute_differential_abundance(
        adata,
        groupby='group',
        condition1='A',
        condition2='B',
        log_fold_change_threshold=1.0,
        pvalue_threshold=0.05
    )
    
    # Create a custom layout config
    custom_layout = {
        'unit_size': 0.2,
        'plot_height': 5,
        'plot_width': 50,
        'label_width': 5,
        'plot_spacing': 0.5,
        'top_margin': 2,
        'y_label_width': 3
    }
    
    # Test with custom layout
    fig, axes = multi_volcano_da(
        adata,
        groupby='cell_type',
        run_id=-1,
        plot_width_factor=15.0,
        layout_config=custom_layout,
        return_fig=True
    )
    
    # Check results
    assert fig is not None
    assert isinstance(axes, list)
    
    plt.close('all')


def test_multi_volcano_da_update_direction_with_custom_thresholds():
    """Test multi_volcano_da with direction column update using custom thresholds."""
    # Create test data
    adata = create_test_anndata()
    
    # Run differential abundance analysis with specific thresholds
    compute_differential_abundance(
        adata,
        groupby='group',
        condition1='A',
        condition2='B',
        log_fold_change_threshold=1.0,
        pvalue_threshold=0.05
    )
    
    # Test with direction column update using different thresholds
    fig, axes = multi_volcano_da(
        adata,
        groupby='cell_type',
        run_id=-1,
        update_direction=True,
        lfc_threshold=1.5,  # Different from the one used in compute_differential_abundance
        pval_threshold=0.01,  # Different from the one used in compute_differential_abundance
        return_fig=True
    )
    
    # Check results
    assert fig is not None
    assert isinstance(axes, list)
    
    # Find the direction column
    direction_columns = [col for col in adata.obs.columns if 'direction' in col]
    assert len(direction_columns) > 0, "No direction column found"
    
    # Verify that the direction column contains expected values
    direction_col = direction_columns[0]
    directions = set(adata.obs[direction_col].astype(str).unique())
    expected_values = set(['up', 'down', 'neutral'])
    
    # Just check if any of the expected values are present
    assert any(val in directions for val in expected_values), \
        f"Direction column {direction_col} does not contain expected values. Found: {directions}"
    
    plt.close('all')