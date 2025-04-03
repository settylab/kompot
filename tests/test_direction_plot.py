"""Advanced tests for the direction_barplot plotting function.

This file extends the basic tests in test_plot_functions.py with more detailed
tests focusing on specific features of direction_barplot.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List

from kompot.anndata.differential_abundance import compute_differential_abundance
from kompot.plot.heatmap.direction_plot import direction_barplot, _infer_direction_key


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


def test_infer_direction_key_multiple_columns():
    """Test _infer_direction_key function with multiple direction columns."""
    # Create test data
    adata = create_test_anndata()
    
    # Run differential abundance analysis (first run)
    compute_differential_abundance(
        adata,
        groupby='group',
        condition1='A',
        condition2='B',
        log_fold_change_threshold=1.0,
        pvalue_threshold=0.05,
        result_key='da_run1'
    )
    
    # Run differential abundance analysis (second run)
    compute_differential_abundance(
        adata,
        groupby='group',
        condition1='A',
        condition2='B',
        log_fold_change_threshold=1.5,
        pvalue_threshold=0.01,
        result_key='da_run2'
    )
    
    # We should now have two direction columns
    direction_columns = [col for col in adata.obs.columns if 'direction' in col]
    assert len(direction_columns) >= 2, "Expected at least 2 direction columns"
    
    # Add conditions to run info explicitly
    if 'kompot_da' in adata.uns and 'run_history' in adata.uns['kompot_da']:
        # Need to handle JSON string unpacking
        from kompot.anndata.utils import get_run_history, append_to_run_history
        
        # Get the run history
        runs = get_run_history(adata, 'da')
        
        # Update each run to include conditions
        for i, run in enumerate(runs):
            if 'params' not in run:
                run['params'] = {}
            run['params']['conditions'] = ['A', 'B']
    
    # Test inference without direction column (should find the most recent one)
    direction_column, condition1, condition2 = _infer_direction_key(
        adata,
        run_id=-1
    )
    
    # Check results
    assert direction_column is not None
    assert 'direction' in direction_column
    assert 'da_run2' in direction_column, "Should find the most recent run's direction column"
    assert condition1 == 'A'
    assert condition2 == 'B'


def test_direction_barplot_normalization_options():
    """Test direction_barplot function with different normalization methods."""
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
    
    # Test with index normalization (default)
    fig, ax = direction_barplot(
        adata,
        category_column='cell_type',
        run_id=-1,
        normalize="index",
        return_fig=True
    )
    
    # Check results
    assert fig is not None
    assert ax is not None
    
    plt.close('all')
    
    # Test with columns normalization
    fig, ax = direction_barplot(
        adata,
        category_column='cell_type',
        run_id=-1,
        normalize="columns",
        return_fig=True
    )
    
    # Check results
    assert fig is not None
    assert ax is not None
    
    plt.close('all')
    
    # Test with no normalization - using False instead of None
    # since pd.crosstab() doesn't accept None for normalize parameter
    fig, ax = direction_barplot(
        adata,
        category_column='cell_type',
        run_id=-1,
        normalize=False,
        return_fig=True
    )
    
    # Check results
    assert fig is not None
    assert ax is not None
    
    plt.close('all')


def test_direction_barplot_stacked_vs_grouped():
    """Test direction_barplot function with stacked and grouped bars."""
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
    
    # Test with stacked bars (default)
    fig, ax = direction_barplot(
        adata,
        category_column='cell_type',
        run_id=-1,
        stacked=True,
        return_fig=True
    )
    
    # Check results
    assert fig is not None
    assert ax is not None
    
    plt.close('all')
    
    # Test with grouped bars
    fig, ax = direction_barplot(
        adata,
        category_column='cell_type',
        run_id=-1,
        stacked=False,
        return_fig=True
    )
    
    # Check results
    assert fig is not None
    assert ax is not None
    
    plt.close('all')


def test_direction_barplot_sorting_options():
    """Test direction_barplot function with different sorting options."""
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
    
    # Test sorting by 'up' direction, ascending
    fig, ax = direction_barplot(
        adata,
        category_column='cell_type',
        run_id=-1,
        sort_by='up',
        ascending=True,
        return_fig=True
    )
    
    # Check results
    assert fig is not None
    assert ax is not None
    
    plt.close('all')
    
    # Test sorting by 'down' direction, descending
    fig, ax = direction_barplot(
        adata,
        category_column='cell_type',
        run_id=-1,
        sort_by='down',
        ascending=False,
        return_fig=True
    )
    
    # Check results
    assert fig is not None
    assert ax is not None
    
    plt.close('all')


def test_direction_barplot_legend_customization():
    """Test direction_barplot function with different legend options."""
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
    
    # Test with custom legend title and location
    fig, ax = direction_barplot(
        adata,
        category_column='cell_type',
        run_id=-1,
        legend_title="Custom Legend",
        legend_loc="upper right",
        return_fig=True
    )
    
    # Check results
    assert fig is not None
    assert ax is not None
    
    plt.close('all')


def test_direction_barplot_custom_colors():
    """Test direction_barplot function with custom colors."""
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
    
    # Define custom colors
    custom_colors = {
        'up': 'darkred',
        'down': 'navy',
        'neutral': 'lightgray'
    }
    
    # Test with custom colors
    fig, ax = direction_barplot(
        adata,
        category_column='cell_type',
        run_id=-1,
        colors=custom_colors,
        return_fig=True
    )
    
    # Check results
    assert fig is not None
    assert ax is not None
    
    plt.close('all')


def test_direction_barplot_custom_title_and_labels():
    """Test direction_barplot function with custom title and axis labels."""
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
    
    # Test with custom title and labels
    fig, ax = direction_barplot(
        adata,
        category_column='cell_type',
        run_id=-1,
        title="Custom Plot Title",
        xlabel="Custom X Label",
        ylabel="Custom Y Label",
        return_fig=True
    )
    
    # Check results
    assert fig is not None
    assert ax is not None
    
    plt.close('all')