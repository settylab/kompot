"""Lightweight tests for differential_expression utility functions to improve coverage."""

import numpy as np
import pytest
import pandas as pd
import anndata

from kompot.anndata.differential_expression import _prepare_null_genes


def test_prepare_null_genes_none():
    """Test _prepare_null_genes with None."""
    available_genes = [f'gene_{i}' for i in range(100)]

    null_gene_indices, used_replacement = _prepare_null_genes(None, available_genes, 42)
    assert null_gene_indices == []
    assert used_replacement is False


def test_prepare_null_genes_zero():
    """Test _prepare_null_genes with 0."""
    available_genes = [f'gene_{i}' for i in range(100)]

    null_gene_indices, used_replacement = _prepare_null_genes(0, available_genes, 42)
    assert null_gene_indices == []
    assert used_replacement is False


def test_prepare_null_genes_int_without_replacement():
    """Test _prepare_null_genes with int (no replacement needed)."""
    available_genes = [f'gene_{i}' for i in range(100)]

    null_gene_indices, used_replacement = _prepare_null_genes(50, available_genes, 42)
    assert len(null_gene_indices) == 50
    assert used_replacement is False
    assert all(isinstance(idx, int) for idx in null_gene_indices)
    assert all(0 <= idx < 100 for idx in null_gene_indices)
    # Should have no duplicates when replacement is not used
    assert len(set(null_gene_indices)) == 50


def test_prepare_null_genes_int_with_replacement():
    """Test _prepare_null_genes with int (replacement needed)."""
    available_genes = [f'gene_{i}' for i in range(100)]

    # Request more genes than available
    null_gene_indices, used_replacement = _prepare_null_genes(150, available_genes, 42)
    assert len(null_gene_indices) == 150
    assert used_replacement is True
    assert all(isinstance(idx, int) for idx in null_gene_indices)
    assert all(0 <= idx < 100 for idx in null_gene_indices)


def test_prepare_null_genes_list():
    """Test _prepare_null_genes with list of indices."""
    available_genes = [f'gene_{i}' for i in range(100)]

    specific_indices = [0, 5, 10, 15, 20]
    null_gene_indices, used_replacement = _prepare_null_genes(
        specific_indices, available_genes, 42
    )
    assert null_gene_indices == specific_indices
    assert used_replacement is False


def test_prepare_null_genes_negative_int():
    """Test _prepare_null_genes with negative int raises error."""
    available_genes = [f'gene_{i}' for i in range(100)]

    with pytest.raises(ValueError, match="null_genes must be positive"):
        _prepare_null_genes(-5, available_genes, 42)


def test_prepare_null_genes_list_with_invalid_indices():
    """Test _prepare_null_genes with invalid indices in list."""
    available_genes = [f'gene_{i}' for i in range(100)]

    # Indices out of range
    invalid_indices = [0, 5, 150]  # 150 is out of range
    with pytest.raises(ValueError, match="Invalid gene indices"):
        _prepare_null_genes(invalid_indices, available_genes, 42)

    # Negative indices
    invalid_indices = [0, -1, 5]
    with pytest.raises(ValueError, match="Invalid gene indices"):
        _prepare_null_genes(invalid_indices, available_genes, 42)


def test_prepare_null_genes_list_with_non_integers():
    """Test _prepare_null_genes with non-integer elements in list."""
    available_genes = [f'gene_{i}' for i in range(100)]

    invalid_list = [0, 5, 10.5]  # 10.5 is not an integer
    with pytest.raises(ValueError, match="All elements in null_genes list must be integers"):
        _prepare_null_genes(invalid_list, available_genes, 42)


def test_prepare_null_genes_invalid_type():
    """Test _prepare_null_genes with invalid type."""
    available_genes = [f'gene_{i}' for i in range(100)]

    with pytest.raises(ValueError, match="null_genes must be int, list of ints, or None"):
        _prepare_null_genes("invalid", available_genes, 42)

    with pytest.raises(ValueError, match="null_genes must be int, list of ints, or None"):
        _prepare_null_genes({"genes": [0, 1, 2]}, available_genes, 42)


def test_prepare_null_genes_reproducibility():
    """Test that _prepare_null_genes is reproducible with same seed."""
    available_genes = [f'gene_{i}' for i in range(100)]

    indices1, _ = _prepare_null_genes(50, available_genes, 42)
    indices2, _ = _prepare_null_genes(50, available_genes, 42)

    assert indices1 == indices2


def test_prepare_null_genes_different_seeds():
    """Test that different seeds give different results."""
    available_genes = [f'gene_{i}' for i in range(100)]

    indices1, _ = _prepare_null_genes(50, available_genes, 42)
    indices2, _ = _prepare_null_genes(50, available_genes, 123)

    # With high probability, different seeds should give different results
    assert indices1 != indices2


def test_compute_differential_expression_error_handling():
    """Test error handling in compute_differential_expression."""
    from kompot import compute_differential_expression

    # Create minimal test data
    np.random.seed(42)
    n_cells, n_genes = 100, 20
    X = np.random.normal(0, 1, (n_cells, n_genes))
    obs = pd.DataFrame({
        'condition': ['A'] * 50 + ['B'] * 50
    })
    obsm = {'DM_EigenVectors': np.random.normal(0, 1, (n_cells, 10))}
    var = pd.DataFrame(index=[f'gene_{i}' for i in range(n_genes)])
    adata = anndata.AnnData(X=X, obs=obs, var=var, obsm=obsm)

    # Test with non-existent groupby column
    with pytest.raises((ValueError, KeyError)):
        compute_differential_expression(
            adata,
            groupby='nonexistent',
            condition1='A',
            condition2='B'
        )

    # Test with non-existent condition
    with pytest.raises((ValueError, KeyError)):
        compute_differential_expression(
            adata,
            groupby='condition',
            condition1='A',
            condition2='nonexistent'
        )


def test_compute_differential_expression_parameter_combinations():
    """Test various parameter combinations without full computation."""
    from kompot import compute_differential_expression

    # Create minimal test data
    np.random.seed(42)
    n_cells, n_genes = 60, 10
    X = np.random.normal(0, 1, (n_cells, n_genes))
    obs = pd.DataFrame({
        'condition': ['A'] * 30 + ['B'] * 30,
        'sample': [f'sample_{i%3}' for i in range(n_cells)]
    })
    obsm = {'DM_EigenVectors': np.random.normal(0, 1, (n_cells, 5))}
    var = pd.DataFrame(index=[f'gene_{i}' for i in range(n_genes)])
    adata = anndata.AnnData(X=X, obs=obs, var=var, obsm=obsm)

    # Test with minimal parameters
    result = compute_differential_expression(
        adata,
        groupby='condition',
        condition1='A',
        condition2='B',
        n_landmarks=20,
        null_genes=None,  # Disable FDR for speed
        progress=False
    )

    assert result is not None or 'kompot_de' in adata.uns


def test_compute_differential_expression_with_copy():
    """Test compute_differential_expression with copy=True."""
    from kompot import compute_differential_expression

    # Create minimal test data
    np.random.seed(42)
    n_cells, n_genes = 60, 10
    X = np.random.normal(0, 1, (n_cells, n_genes))
    obs = pd.DataFrame({
        'condition': ['A'] * 30 + ['B'] * 30
    })
    obsm = {'DM_EigenVectors': np.random.normal(0, 1, (n_cells, 5))}
    var = pd.DataFrame(index=[f'gene_{i}' for i in range(n_genes)])
    adata = anndata.AnnData(X=X, obs=obs, var=var, obsm=obsm)

    # Test with copy=True
    result_adata = compute_differential_expression(
        adata,
        groupby='condition',
        condition1='A',
        condition2='B',
        n_landmarks=20,
        null_genes=None,
        copy=True,
        progress=False
    )

    # Original adata should not be modified
    assert 'kompot_de' not in adata.uns
    # Result should be an AnnData object
    assert isinstance(result_adata, anndata.AnnData)


def test_compute_differential_expression_sparse_matrix():
    """Test compute_differential_expression with sparse matrix."""
    from kompot import compute_differential_expression
    from scipy import sparse

    # Create minimal test data with sparse matrix
    np.random.seed(42)
    n_cells, n_genes = 60, 10
    X_dense = np.random.normal(0, 1, (n_cells, n_genes))
    X_dense[X_dense < 0] = 0  # Make sparse-like
    X_sparse = sparse.csr_matrix(X_dense)

    obs = pd.DataFrame({
        'condition': ['A'] * 30 + ['B'] * 30
    })
    obsm = {'DM_EigenVectors': np.random.normal(0, 1, (n_cells, 5))}
    var = pd.DataFrame(index=[f'gene_{i}' for i in range(n_genes)])
    adata = anndata.AnnData(X=X_sparse, obs=obs, var=var, obsm=obsm)

    # Should handle sparse matrix
    result = compute_differential_expression(
        adata,
        groupby='condition',
        condition1='A',
        condition2='B',
        n_landmarks=20,
        null_genes=None,
        progress=False
    )

    assert result is not None or 'kompot_de' in adata.uns
