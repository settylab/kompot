"""Tests for the embedding function."""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch

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