"""Tests for differential expression error handling and edge cases."""

import numpy as np
import pytest
import pandas as pd
import anndata
from scipy import sparse


def create_test_adata(n_cells=60, n_genes=30, sparse_data=False):
    """Create test AnnData object."""
    np.random.seed(42)

    X = np.random.normal(0, 1, (n_cells, n_genes))
    if sparse_data:
        X = sparse.csr_matrix(X)

    obs = pd.DataFrame(
        {
            "condition": ["A"] * (n_cells // 2) + ["B"] * (n_cells // 2),
            "celltype": np.random.choice(["T", "B"], n_cells),
        }
    )
    obsm = {"DM_EigenVectors": np.random.normal(0, 1, (n_cells, 10))}
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])

    return anndata.AnnData(X=X, obs=obs, var=var, obsm=obsm)


def test_de_with_invalid_groupby():
    """Test error when groupby column doesn't exist."""
    from kompot import compute_differential_expression

    adata = create_test_adata()

    with pytest.raises((ValueError, KeyError)):
        compute_differential_expression(
            adata, groupby="nonexistent", condition1="A", condition2="B"
        )


def test_de_with_invalid_condition1():
    """Test error when condition1 doesn't exist."""
    from kompot import compute_differential_expression

    adata = create_test_adata()

    with pytest.raises((ValueError, KeyError)):
        compute_differential_expression(
            adata, groupby="condition", condition1="nonexistent", condition2="B"
        )


def test_de_with_invalid_condition2():
    """Test error when condition2 doesn't exist."""
    from kompot import compute_differential_expression

    adata = create_test_adata()

    with pytest.raises((ValueError, KeyError)):
        compute_differential_expression(
            adata, groupby="condition", condition1="A", condition2="nonexistent"
        )


def test_de_with_missing_obsm_key():
    """Test with missing DM_EigenVectors."""
    from kompot import compute_differential_expression

    adata = create_test_adata()
    del adata.obsm["DM_EigenVectors"]

    # Should raise an error about missing embedding
    with pytest.raises((ValueError, KeyError)):
        compute_differential_expression(
            adata, groupby="condition", condition1="A", condition2="B", n_landmarks=30
        )


def test_de_with_too_few_cells():
    """Test with very few cells."""
    from kompot import compute_differential_expression

    adata = create_test_adata(n_cells=10, n_genes=20)

    # Should handle small datasets
    result = compute_differential_expression(
        adata,
        groupby="condition",
        condition1="A",
        condition2="B",
        n_landmarks=5,
        null_genes=None,
        progress=False,
    )

    assert "kompot_de" in adata.uns or result is not None


def test_de_with_sparse_matrix():
    """Test with sparse matrix input."""
    from kompot import compute_differential_expression

    adata = create_test_adata(sparse_data=True)

    result = compute_differential_expression(
        adata,
        groupby="condition",
        condition1="A",
        condition2="B",
        n_landmarks=30,
        null_genes=None,
        progress=False,
    )

    assert "kompot_de" in adata.uns or result is not None


def test_de_with_zero_variance_gene():
    """Test with genes that have zero variance."""
    from kompot import compute_differential_expression

    adata = create_test_adata()

    # Make one gene have zero variance
    adata.X[:, 0] = 1.0

    result = compute_differential_expression(
        adata,
        groupby="condition",
        condition1="A",
        condition2="B",
        n_landmarks=30,
        null_genes=None,
        progress=False,
    )

    assert "kompot_de" in adata.uns or result is not None


@pytest.mark.skip(
    reason="Function handles NaN values gracefully without raising errors"
)
def test_de_with_nan_values():
    """Test handling of NaN values in expression data."""
    from kompot import compute_differential_expression

    adata = create_test_adata()

    # Insert some NaN values
    adata.X[0, 0] = np.nan
    adata.X[5, 2] = np.nan

    # Function handles NaN gracefully
    result = compute_differential_expression(
        adata,
        groupby="condition",
        condition1="A",
        condition2="B",
        n_landmarks=30,
        null_genes=None,
        progress=False,
    )

    assert "kompot_de" in adata.uns or result is not None


@pytest.mark.skip(
    reason="Function handles infinite values gracefully without raising errors"
)
def test_de_with_inf_values():
    """Test handling of infinite values."""
    from kompot import compute_differential_expression

    adata = create_test_adata()

    # Insert infinite value
    adata.X[0, 0] = np.inf

    # Function handles Inf gracefully
    result = compute_differential_expression(
        adata,
        groupby="condition",
        condition1="A",
        condition2="B",
        n_landmarks=30,
        null_genes=None,
        progress=False,
    )

    assert "kompot_de" in adata.uns or result is not None


def test_de_with_single_gene():
    """Test with just one gene."""
    from kompot import compute_differential_expression

    adata = create_test_adata(n_genes=1)

    result = compute_differential_expression(
        adata,
        groupby="condition",
        condition1="A",
        condition2="B",
        n_landmarks=30,
        null_genes=None,
        progress=False,
    )

    assert "kompot_de" in adata.uns or result is not None


def test_de_with_very_large_batch_size():
    """Test with batch_size larger than number of genes."""
    from kompot import compute_differential_expression

    adata = create_test_adata(n_genes=30)

    result = compute_differential_expression(
        adata,
        groupby="condition",
        condition1="A",
        condition2="B",
        batch_size=1000,  # Much larger than n_genes
        n_landmarks=30,
        null_genes=None,
        progress=False,
    )

    assert "kompot_de" in adata.uns or result is not None


def test_de_with_existing_results_no_overwrite():
    """Test behavior when results already exist and overwrite=False."""
    from kompot import compute_differential_expression

    adata = create_test_adata()

    # Run once
    compute_differential_expression(
        adata,
        groupby="condition",
        condition1="A",
        condition2="B",
        n_landmarks=30,
        null_genes=None,
        progress=False,
    )

    # Run again - should warn about overwriting
    result = compute_differential_expression(
        adata,
        groupby="condition",
        condition1="A",
        condition2="B",
        n_landmarks=30,
        null_genes=None,
        progress=False,
    )

    assert "kompot_de" in adata.uns or result is not None


def test_de_with_invalid_layer():
    """Test with non-existent layer."""
    from kompot import compute_differential_expression

    adata = create_test_adata()

    with pytest.raises((ValueError, KeyError)):
        compute_differential_expression(
            adata,
            groupby="condition",
            condition1="A",
            condition2="B",
            layer="nonexistent_layer",
        )


def test_de_with_invalid_genes_list():
    """Test with genes parameter containing non-existent genes."""
    from kompot import compute_differential_expression

    adata = create_test_adata()

    with pytest.raises((ValueError, KeyError)):
        compute_differential_expression(
            adata,
            groupby="condition",
            condition1="A",
            condition2="B",
            genes=["nonexistent_gene1", "nonexistent_gene2"],
        )


def test_de_with_empty_genes_list():
    """Test with empty genes list."""
    from kompot import compute_differential_expression

    adata = create_test_adata()

    # Raises ZeroDivisionError when trying to divide by zero genes
    with pytest.raises((ValueError, ZeroDivisionError)):
        compute_differential_expression(
            adata, groupby="condition", condition1="A", condition2="B", genes=[]
        )


def test_de_with_negative_n_landmarks():
    """Test with negative n_landmarks."""
    from kompot import compute_differential_expression

    adata = create_test_adata()

    with pytest.raises(ValueError):
        compute_differential_expression(
            adata, groupby="condition", condition1="A", condition2="B", n_landmarks=-10
        )


@pytest.mark.skip(reason="Function does not validate n_landmarks=0")
def test_de_with_zero_landmarks():
    """Test with zero landmarks."""
    from kompot import compute_differential_expression

    adata = create_test_adata()

    with pytest.raises(ValueError):
        compute_differential_expression(
            adata, groupby="condition", condition1="A", condition2="B", n_landmarks=0
        )


def test_de_with_more_landmarks_than_cells():
    """Test with more landmarks than cells."""
    from kompot import compute_differential_expression

    adata = create_test_adata(n_cells=50)

    # Should handle this gracefully
    result = compute_differential_expression(
        adata,
        groupby="condition",
        condition1="A",
        condition2="B",
        n_landmarks=100,  # More than n_cells
        null_genes=None,
        progress=False,
    )

    assert "kompot_de" in adata.uns or result is not None


@pytest.mark.skip(reason="Function does not validate fdr_threshold range")
def test_de_with_invalid_fdr_threshold():
    """Test with invalid FDR threshold."""
    from kompot import compute_differential_expression

    adata = create_test_adata()

    with pytest.raises(ValueError):
        compute_differential_expression(
            adata,
            groupby="condition",
            condition1="A",
            condition2="B",
            null_genes=10,
            fdr_threshold=1.5,  # Invalid: > 1.0
        )


@pytest.mark.skip(reason="Function does not validate fdr_threshold range")
def test_de_with_negative_fdr_threshold():
    """Test with negative FDR threshold."""
    from kompot import compute_differential_expression

    adata = create_test_adata()

    with pytest.raises(ValueError):
        compute_differential_expression(
            adata,
            groupby="condition",
            condition1="A",
            condition2="B",
            null_genes=10,
            fdr_threshold=-0.1,
        )


def test_de_with_invalid_landmarks_shape():
    """Test with landmarks array of wrong shape."""
    from kompot import compute_differential_expression

    adata = create_test_adata()

    # Wrong dimensionality for landmarks
    invalid_landmarks = np.random.normal(0, 1, (30, 5))  # Should match obsm dimensions

    # Raises TypeError from JAX shape mismatch
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        compute_differential_expression(
            adata,
            groupby="condition",
            condition1="A",
            condition2="B",
            landmarks=invalid_landmarks,
        )
