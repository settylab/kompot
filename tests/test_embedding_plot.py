"""Tests for the embedding function."""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock

try:
    import scanpy as sc
    _has_scanpy = True
except ImportError:
    _has_scanpy = False

# Skip all tests in this module if scanpy is not available
pytestmark = pytest.mark.skipif(not _has_scanpy, reason="Scanpy is required for these tests")


def test_embedding_import():
    """Test that embedding can be imported from kompot.plot."""
    from kompot.plot import embedding
    assert callable(embedding)


def test_embedding_basic():
    """Test basic functionality of embedding."""
    from kompot.plot import embedding
    
    # Create a small test AnnData object
    np.random.seed(42)
    n_cells = 100
    adata = sc.AnnData(X=np.random.normal(size=(n_cells, 10)))
    adata.obsm['X_umap'] = np.random.normal(size=(n_cells, 2))
    adata.obs['cluster'] = np.random.choice(['A', 'B', 'C'], size=n_cells)
    adata.obs['condition'] = np.random.choice(['control', 'treatment'], size=n_cells)
    
    # Test with return_fig=True to avoid displaying the plot
    with patch('matplotlib.pyplot.show'):  # Mock plt.show to prevent display
        fig = embedding(adata, basis='umap', color='cluster', return_fig=True)
        
    assert fig is not None
    plt.close(fig)


# This test was removed as we're focusing on the new embedding function only


def test_embedding_with_groups():
    """Test embedding with group filtering."""
    from kompot.plot import embedding
    
    # Create a small test AnnData object
    np.random.seed(42)
    n_cells = 100
    adata = sc.AnnData(X=np.random.normal(size=(n_cells, 10)))
    adata.obsm['X_umap'] = np.random.normal(size=(n_cells, 2))
    adata.obs['cluster'] = np.random.choice(['A', 'B', 'C'], size=n_cells)
    adata.obs['condition'] = np.random.choice(['control', 'treatment'], size=n_cells)
    
    # Define groups for filtering
    groups = {'condition': ['treatment']}
    
    # Test with return_fig=True to avoid displaying the plot
    with patch('matplotlib.pyplot.show'):  # Mock plt.show to prevent display
        fig = embedding(adata, basis='umap', color='cluster', groups=groups, return_fig=True)
        
    assert fig is not None
    plt.close(fig)


def test_embedding_multi_panels():
    """Test embedding with multiple panels (color as list)."""
    from kompot.plot import embedding
    
    # Create a small test AnnData object
    np.random.seed(42)
    n_cells = 100
    adata = sc.AnnData(X=np.random.normal(size=(n_cells, 10)))
    adata.obsm['X_umap'] = np.random.normal(size=(n_cells, 2))
    adata.obs['cluster'] = np.random.choice(['A', 'B', 'C'], size=n_cells)
    adata.obs['condition'] = np.random.choice(['control', 'treatment'], size=n_cells)
    adata.obs['score'] = np.random.normal(size=n_cells)
    
    # Test with explicit ncols to avoid displaying the plot
    with patch('matplotlib.pyplot.show'):  # Mock plt.show to prevent display
        result = embedding(
            adata, 
            basis='umap',
            color=['cluster', 'condition', 'score'], 
            ncols=2,
            return_fig=True
        )
        
    assert result is not None
    plt.close(result)
    
    # Test with default ncols (which is now 4)
    with patch('matplotlib.pyplot.show'):
        result = embedding(
            adata, 
            basis='umap',
            color=['cluster', 'condition', 'score'], 
            return_fig=True
        )
        
    assert result is not None
    plt.close(result)
    
    # Test with shorter title list than color list
    with patch('matplotlib.pyplot.show'):
        result = embedding(
            adata, 
            basis='umap',
            color=['cluster', 'condition', 'score'], 
            title=['Clusters', 'Conditions'],
            return_fig=True
        )
        
    assert result is not None
    plt.close(result)


def test_embedding_with_groups_and_multi_panels():
    """Test embedding with both group filtering and multiple panels."""
    from kompot.plot import embedding
    
    # Create a small test AnnData object
    np.random.seed(42)
    n_cells = 100
    adata = sc.AnnData(X=np.random.normal(size=(n_cells, 10)))
    adata.obsm['X_umap'] = np.random.normal(size=(n_cells, 2))
    adata.obs['cluster'] = np.random.choice(['A', 'B', 'C'], size=n_cells)
    adata.obs['condition'] = np.random.choice(['control', 'treatment'], size=n_cells)
    adata.obs['batch'] = np.random.choice(['batch1', 'batch2'], size=n_cells)
    
    # Define groups for filtering
    groups = {'condition': ['treatment'], 'batch': ['batch1']}
    
    # Test with return_fig=True to avoid displaying the plot
    with patch('matplotlib.pyplot.show'):  # Mock plt.show to prevent display
        result = embedding(
            adata, 
            basis='umap',
            color=['cluster', 'batch'], 
            groups=groups,
            return_fig=True
        )
        
    assert result is not None
    plt.close(result)


def test_embedding_without_background():
    """Test embedding with show_background=False."""
    from kompot.plot import embedding
    
    # Create a small test AnnData object
    np.random.seed(42)
    n_cells = 100
    adata = sc.AnnData(X=np.random.normal(size=(n_cells, 10)))
    adata.obsm['X_umap'] = np.random.normal(size=(n_cells, 2))
    adata.obs['cluster'] = np.random.choice(['A', 'B', 'C'], size=n_cells)
    adata.obs['condition'] = np.random.choice(['control', 'treatment'], size=n_cells)
    
    # Define groups for filtering
    groups = {'condition': ['treatment']}
    
    # Test with return_fig=True to avoid displaying the plot
    with patch('matplotlib.pyplot.show'):  # Mock plt.show to prevent display
        fig = embedding(
            adata, 
            basis='umap',
            color='cluster', 
            groups=groups,
            background_color=None,
            return_fig=True
        )
        
    assert fig is not None
    plt.close(fig)


def test_embedding_invalid_basis():
    """Test embedding with an invalid basis."""
    from kompot.plot import embedding
    
    # Create a small test AnnData object
    np.random.seed(42)
    n_cells = 100
    adata = sc.AnnData(X=np.random.normal(size=(n_cells, 10)))
    # Intentionally NOT adding X_umap to test error handling
    
    # Test that the function raises a ValueError for invalid basis
    with pytest.raises(ValueError):
        with patch('matplotlib.pyplot.show'):  # Mock plt.show to prevent display
            embedding(adata, basis='X_invalid_basis')


def test_embedding_empty_group_selection():
    """Test embedding with groups that select no cells."""
    from kompot.plot import embedding
    
    # Create a small test AnnData object
    np.random.seed(42)
    n_cells = 100
    adata = sc.AnnData(X=np.random.normal(size=(n_cells, 10)))
    adata.obsm['X_umap'] = np.random.normal(size=(n_cells, 2))
    adata.obs['cluster'] = np.random.choice(['A', 'B', 'C'], size=n_cells)
    
    # Define groups that won't match any cells
    groups = {'cluster': ['D']}  # 'D' doesn't exist
    
    # Test that the function returns None when no cells match
    with patch('matplotlib.pyplot.show'):  # Mock plt.show to prevent display
        result = embedding(adata, basis='umap', color='cluster', groups=groups)
        
    assert result is None  # Should return None for no matching cells


def test_embedding_colormap_parameters():
    """Test embedding with colormap and vcenter parameters."""
    from kompot.plot import embedding
    
    # Create a small test AnnData object with numeric data
    np.random.seed(42)
    n_cells = 100
    adata = sc.AnnData(X=np.random.normal(size=(n_cells, 10)))
    adata.obsm['X_umap'] = np.random.normal(size=(n_cells, 2))
    adata.obs['score'] = np.random.normal(size=n_cells)
    
    # Test with color_map parameter
    with patch('matplotlib.pyplot.show'):
        fig = embedding(
            adata, 
            basis='umap',
            color='score',
            color_map='viridis',
            return_fig=True
        )
        
    assert fig is not None
    plt.close(fig)
    
    # Test with cmap parameter
    with patch('matplotlib.pyplot.show'):
        fig = embedding(
            adata, 
            basis='umap',
            color='score',
            cmap='plasma',
            return_fig=True
        )
        
    assert fig is not None
    plt.close(fig)
    
    # Test with vcenter parameter
    with patch('matplotlib.pyplot.show'):
        fig = embedding(
            adata, 
            basis='umap',
            color='score',
            cmap='RdBu_r',
            vcenter=0,
            return_fig=True
        )
        
    assert fig is not None
    plt.close(fig)
    
    # Test with multiple panels and colormap
    with patch('matplotlib.pyplot.show'):
        result = embedding(
            adata, 
            basis='umap',
            color=['score', 'score'],
            color_map='RdBu_r',
            vcenter=0,
            return_fig=True
        )
        
    assert result is not None
    plt.close(result)


def test_embedding_with_ax():
    """Test embedding when an ax is provided."""
    from kompot.plot import embedding
    
    # Create a small test AnnData object
    np.random.seed(42)
    n_cells = 100
    adata = sc.AnnData(X=np.random.normal(size=(n_cells, 10)))
    adata.obsm['X_umap'] = np.random.normal(size=(n_cells, 2))
    adata.obs['cluster'] = np.random.choice(['A', 'B', 'C'], size=n_cells)
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Test with ax provided and return_fig=False
    result = embedding(
        adata, 
        basis='umap',
        color='cluster', 
        ax=ax,
        return_fig=False
    )
    
    # Should return None as return_fig=False
    assert result is None
    
    # Test with ax provided and return_fig=True
    fig, ax = plt.subplots(figsize=(8, 6))
    result = embedding(
        adata, 
        basis='umap',
        color='cluster', 
        ax=ax,
        return_fig=True
    )
    
    # Should return the figure as return_fig=True
    assert result is not None
    assert result is fig  # Should be the same figure object
    
    plt.close(fig)


def test_embedding_with_mgroups():
    """Test embedding with the mgroups parameter."""
    from kompot.plot import embedding
    
    # Create a small test AnnData object
    np.random.seed(42)
    n_cells = 100
    adata = sc.AnnData(X=np.random.normal(size=(n_cells, 10)))
    adata.obsm['X_umap'] = np.random.normal(size=(n_cells, 2))
    adata.obs['cluster'] = np.random.choice(['A', 'B', 'C'], size=n_cells)
    adata.obs['condition'] = np.random.choice(['control', 'treatment'], size=n_cells)
    adata.obs['batch'] = np.random.choice(['batch1', 'batch2'], size=n_cells)
    
    # Define multiple group conditions
    mgroups = [
        {'condition': 'control'},
        {'condition': 'treatment'},
        {'batch': 'batch1'},
        {'batch': 'batch2'}
    ]
    
    # Test with mgroups and default ncols
    with patch('matplotlib.pyplot.show'):
        result = embedding(
            adata, 
            basis='umap',
            color='cluster',
            mgroups=mgroups,
            return_fig=True
        )
        
    assert result is not None
    # Should have the right number of subplots
    assert len(result.axes) >= len(mgroups)
    plt.close(result)
    
    # Test with mgroups and custom ncols
    with patch('matplotlib.pyplot.show'):
        result = embedding(
            adata, 
            basis='umap',
            color='cluster',
            mgroups=mgroups,
            ncols=2,  # Force 2 columns
            return_fig=True
        )
        
    assert result is not None
    # Should have the right number of subplots
    assert len(result.axes) >= 4  # At least 4 axes for our 4 groups
    plt.close(result)
    
    # Test with mgroups and custom titles
    titles = ['Control Group', 'Treatment Group', 'Batch 1', 'Batch 2']
    with patch('matplotlib.pyplot.show'):
        result = embedding(
            adata, 
            basis='umap',
            color='cluster',
            mgroups=mgroups,
            title=titles,
            return_fig=True
        )
        
    assert result is not None
    plt.close(result)
    
    # Test with mgroups and a single title
    with patch('matplotlib.pyplot.show'):
        result = embedding(
            adata, 
            basis='umap',
            color='cluster',
            mgroups=mgroups,
            title='Single Title',
            return_fig=True
        )
        
    assert result is not None
    plt.close(result)


def test_embedding_mgroups_with_color_list_error():
    """Test that error is raised when using mgroups with a list of colors."""
    from kompot.plot import embedding
    
    # Create a small test AnnData object
    np.random.seed(42)
    n_cells = 100
    adata = sc.AnnData(X=np.random.normal(size=(n_cells, 10)))
    adata.obsm['X_umap'] = np.random.normal(size=(n_cells, 2))
    adata.obs['cluster'] = np.random.choice(['A', 'B', 'C'], size=n_cells)
    adata.obs['condition'] = np.random.choice(['control', 'treatment'], size=n_cells)
    
    # Define multiple group conditions
    mgroups = [
        {'condition': 'control'},
        {'condition': 'treatment'}
    ]
    
    # Test that error is raised when using mgroups with a list of colors
    with pytest.raises(ValueError):
        embedding(
            adata, 
            basis='umap',
            color=['cluster', 'condition'],  # List of colors
            mgroups=mgroups
        )


def test_embedding_with_user_provided_ax_and_background():
    """Test embedding with user-provided ax and background cells."""
    from kompot.plot import embedding
    
    # Create a small test AnnData object
    np.random.seed(42)
    n_cells = 100
    adata = sc.AnnData(X=np.random.normal(size=(n_cells, 10)))
    adata.obsm['X_umap'] = np.random.normal(size=(n_cells, 2))
    adata.obs['cluster'] = np.random.choice(['A', 'B', 'C'], size=n_cells)
    adata.obs['condition'] = np.random.choice(['control', 'treatment'], size=n_cells)
    
    # Define groups for filtering
    groups = {'condition': ['treatment']}
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Mock the scatter function to verify it's called for background cells
    original_scatter = ax.scatter
    ax.scatter = MagicMock(wraps=original_scatter)
    
    # Test with ax provided, groups, and background_color
    embedding(
        adata, 
        basis='umap',
        color='cluster', 
        groups=groups,
        background_color='lightgrey',
        ax=ax
    )
    
    # Verify that the scatter function was called
    ax.scatter.assert_called()  # This just checks that scatter was called at least once
    
    plt.close(fig)


def test_embedding_complex_multi_panel():
    """Test embedding in a multi-panel figure with different configurations."""
    from kompot.plot import embedding
    
    # Create a small test AnnData object
    np.random.seed(42)
    n_cells = 100
    adata = sc.AnnData(X=np.random.normal(size=(n_cells, 10)))
    adata.obsm['X_umap'] = np.random.normal(size=(n_cells, 2))
    adata.obs['cluster'] = np.random.choice(['A', 'B', 'C'], size=n_cells)
    adata.obs['condition'] = np.random.choice(['control', 'treatment'], size=n_cells)
    adata.obs['batch'] = np.random.choice(['batch1', 'batch2'], size=n_cells)
    
    # Create a 3x1 figure manually
    fig, axs = plt.subplots(3, 1, figsize=(6, 12))
    
    # Plot with different groups in each subplot
    with patch('matplotlib.pyplot.show'):  # Prevent showing
        # First subplot: control group
        embedding(
            adata, 
            basis='umap',
            color='cluster', 
            groups={'condition': 'control'},
            ax=axs[0],
            title='Control Cells'
        )
        
        # Second subplot: treatment group
        embedding(
            adata, 
            basis='umap',
            color='cluster', 
            groups={'condition': 'treatment'},
            ax=axs[1],
            title='Treatment Cells'
        )
        
        # Third subplot: batch1 group
        embedding(
            adata, 
            basis='umap',
            color='cluster', 
            groups={'batch': 'batch1'},
            ax=axs[2],
            title='Batch 1 Cells'
        )
    
    plt.close(fig)
    
    # Test the same layout using mgroups
    mgroups = [
        {'condition': 'control'},
        {'condition': 'treatment'},
        {'batch': 'batch1'}
    ]
    
    with patch('matplotlib.pyplot.show'):
        result = embedding(
            adata, 
            basis='umap',
            color='cluster',
            mgroups=mgroups,
            ncols=1,  # Force 1 column
            title=['Control Cells', 'Treatment Cells', 'Batch 1 Cells'],
            return_fig=True
        )
        
    assert result is not None
    assert len(result.axes) >= 3  # At least 3 axes for our 3 groups
    plt.close(result)


def test_embedding_with_layer_list():
    """Test embedding with layer as a list to create multiple panels."""
    from kompot.plot import embedding
    
    # Create a small test AnnData object with multiple layers
    np.random.seed(42)
    n_cells = 100
    n_genes = 10
    adata = sc.AnnData(X=np.random.normal(size=(n_cells, n_genes)))
    adata.obsm['X_umap'] = np.random.normal(size=(n_cells, 2))
    adata.obs['cluster'] = np.random.choice(['A', 'B', 'C'], size=n_cells)
    
    # Add some layers to the AnnData object
    adata.layers['raw'] = np.random.normal(size=(n_cells, n_genes))
    adata.layers['normalized'] = np.random.normal(size=(n_cells, n_genes))
    adata.layers['scaled'] = np.random.normal(size=(n_cells, n_genes))
    
    # Add a gene name to var
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    
    # Test with layer as a list and default ncols
    with patch('matplotlib.pyplot.show'):
        result = embedding(
            adata, 
            basis='umap',
            color='cluster',
            layer=['raw', 'normalized', 'scaled'],
            return_fig=True
        )
        
    assert result is not None
    # Should have the right number of subplots
    assert len(result.axes) >= 3  # At least 3 axes for our 3 layers
    plt.close(result)
    
    # Test with layer as a list and custom ncols
    with patch('matplotlib.pyplot.show'):
        result = embedding(
            adata, 
            basis='umap',
            color='cluster',
            layer=['raw', 'normalized', 'scaled'],
            ncols=1,  # Force 1 column
            return_fig=True
        )
        
    assert result is not None
    assert len(result.axes) >= 3  # At least 3 axes for our 3 layers
    plt.close(result)
    
    # Test with layer as a list and custom titles
    titles = ['Raw Data', 'Normalized Data', 'Scaled Data']
    with patch('matplotlib.pyplot.show'):
        result = embedding(
            adata, 
            basis='umap',
            color='cluster',
            layer=['raw', 'normalized', 'scaled'],
            title=titles,
            return_fig=True
        )
        
    assert result is not None
    plt.close(result)
    
    # Test with layer as a list and a single title
    with patch('matplotlib.pyplot.show'):
        result = embedding(
            adata, 
            basis='umap',
            color='cluster',
            layer=['raw', 'normalized', 'scaled'],
            title='Single Title',
            return_fig=True
        )
        
    assert result is not None
    plt.close(result)


def test_embedding_layer_list_incompatible_with_mgroups():
    """Test that error is raised when using layer as a list with mgroups."""
    from kompot.plot import embedding
    
    # Create a small test AnnData object with multiple layers
    np.random.seed(42)
    n_cells = 100
    n_genes = 10
    adata = sc.AnnData(X=np.random.normal(size=(n_cells, n_genes)))
    adata.obsm['X_umap'] = np.random.normal(size=(n_cells, 2))
    adata.obs['cluster'] = np.random.choice(['A', 'B', 'C'], size=n_cells)
    adata.obs['condition'] = np.random.choice(['control', 'treatment'], size=n_cells)
    
    # Add some layers to the AnnData object
    adata.layers['raw'] = np.random.normal(size=(n_cells, n_genes))
    adata.layers['normalized'] = np.random.normal(size=(n_cells, n_genes))
    
    # Define multiple group conditions
    mgroups = [
        {'condition': 'control'},
        {'condition': 'treatment'}
    ]
    
    # Test that error is raised when using layer as a list with mgroups
    with pytest.raises(ValueError):
        embedding(
            adata, 
            basis='umap',
            color='cluster',
            layer=['raw', 'normalized'],
            mgroups=mgroups
        )


def test_embedding_layer_list_incompatible_with_color_list():
    """Test that error is raised when using layer as a list with a list of colors."""
    from kompot.plot import embedding
    
    # Create a small test AnnData object with multiple layers
    np.random.seed(42)
    n_cells = 100
    n_genes = 10
    adata = sc.AnnData(X=np.random.normal(size=(n_cells, n_genes)))
    adata.obsm['X_umap'] = np.random.normal(size=(n_cells, 2))
    adata.obs['cluster'] = np.random.choice(['A', 'B', 'C'], size=n_cells)
    adata.obs['condition'] = np.random.choice(['control', 'treatment'], size=n_cells)
    
    # Add some layers to the AnnData object
    adata.layers['raw'] = np.random.normal(size=(n_cells, n_genes))
    adata.layers['normalized'] = np.random.normal(size=(n_cells, n_genes))
    
    # Test that error is raised when using layer as a list with a list of colors
    with pytest.raises(ValueError):
        embedding(
            adata, 
            basis='umap',
            color=['cluster', 'condition'],  # List of colors
            layer=['raw', 'normalized']
        )


def test_embedding_with_mgroups_as_dict():
    """Test embedding with mgroups as a dictionary of dictionaries."""
    from kompot.plot import embedding
    
    # Create a small test AnnData object
    np.random.seed(42)
    n_cells = 100
    adata = sc.AnnData(X=np.random.normal(size=(n_cells, 10)))
    adata.obsm['X_umap'] = np.random.normal(size=(n_cells, 2))
    adata.obs['cluster'] = np.random.choice(['A', 'B', 'C'], size=n_cells)
    adata.obs['condition'] = np.random.choice(['control', 'treatment'], size=n_cells)
    adata.obs['batch'] = np.random.choice(['batch1', 'batch2'], size=n_cells)
    
    # Define mgroups as a dictionary of dictionaries
    mgroups = {
        'Control Cells': {'condition': 'control'},
        'Treatment Cells': {'condition': 'treatment'},
        'Batch 1': {'batch': 'batch1'},
        'Batch 2': {'batch': 'batch2'}
    }
    
    # Test with mgroups as a dictionary and default ncols
    with patch('matplotlib.pyplot.show'):
        result = embedding(
            adata, 
            basis='umap',
            color='cluster',
            mgroups=mgroups,
            return_fig=True
        )
        
    assert result is not None
    # Should have the right number of subplots
    assert len(result.axes) >= len(mgroups)
    plt.close(result)
    
    # Test with mgroups as a dictionary and custom ncols
    with patch('matplotlib.pyplot.show'):
        result = embedding(
            adata, 
            basis='umap',
            color='cluster',
            mgroups=mgroups,
            ncols=2,  # Force 2 columns
            return_fig=True
        )
        
    assert result is not None
    # Should have the right number of subplots
    assert len(result.axes) >= 4  # At least 4 axes for our 4 groups
    plt.close(result)


def test_embedding_with_mgroups_as_dict_and_titles():
    """Test embedding with mgroups as a dictionary and custom titles."""
    from kompot.plot import embedding
    import warnings
    
    # Create a small test AnnData object
    np.random.seed(42)
    n_cells = 100
    adata = sc.AnnData(X=np.random.normal(size=(n_cells, 10)))
    adata.obsm['X_umap'] = np.random.normal(size=(n_cells, 2))
    adata.obs['cluster'] = np.random.choice(['A', 'B', 'C'], size=n_cells)
    adata.obs['condition'] = np.random.choice(['control', 'treatment'], size=n_cells)
    adata.obs['batch'] = np.random.choice(['batch1', 'batch2'], size=n_cells)
    
    # Define mgroups as a dictionary of dictionaries
    mgroups = {
        'Control Default Title': {'condition': 'control'},
        'Treatment Default Title': {'condition': 'treatment'},
        'Batch1 Default Title': {'batch': 'batch1'},
        'Batch2 Default Title': {'batch': 'batch2'}
    }
    
    # Test with mgroups as a dictionary and custom titles that override some of the dict keys
    custom_titles = ['Custom Control Title', 'Custom Treatment Title']
    
    # Should issue a warning for short titles list
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with patch('matplotlib.pyplot.show'):
            result = embedding(
                adata, 
                basis='umap',
                color='cluster',
                mgroups=mgroups,
                title=custom_titles,  # Only 2 titles for 4 groups
                return_fig=True
            )
        # Check that a warning was issued
        assert any("too short" in str(warning.message) for warning in w)
        
    assert result is not None
    plt.close(result)
    
    # Test with mgroups as a dictionary and a single title (should use the dict keys for remaining)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with patch('matplotlib.pyplot.show'):
            result = embedding(
                adata, 
                basis='umap',
                color='cluster',
                mgroups=mgroups,
                title='Single Common Title',  # Only one title for all groups
                return_fig=True
            )
        # Check that a warning was issued
        assert any("too short" in str(warning.message) for warning in w)
        
    assert result is not None
    plt.close(result)