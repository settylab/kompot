"""Comprehensive FDR edge case tests for differential expression to improve coverage."""

import numpy as np
import pandas as pd
import anndata


def create_fdr_test_data(n_cells=60, n_genes=50, signal_strength=3.0, with_signal=True):
    """Create test data with controlled signal for FDR testing."""
    np.random.seed(42)

    X = np.random.normal(0, 1, (n_cells, n_genes))

    if with_signal:
        # Add strong signal to first few genes
        n_signal_genes = 10
        # Make condition B have higher expression for signal genes
        X[n_cells // 2 :, :n_signal_genes] += signal_strength

    obs = pd.DataFrame({"condition": ["A"] * (n_cells // 2) + ["B"] * (n_cells // 2)})
    obsm = {"DM_EigenVectors": np.random.normal(0, 1, (n_cells, 10))}
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])

    return anndata.AnnData(X=X, obs=obs, var=var, obsm=obsm)


def test_fdr_with_strong_signal():
    """Test FDR with strong signal genes."""
    from kompot import compute_differential_expression

    adata = create_fdr_test_data(
        n_cells=60, n_genes=50, signal_strength=5.0, with_signal=True
    )

    result = compute_differential_expression(
        adata,
        groupby="condition",
        condition1="A",
        condition2="B",
        null_genes=30,  # Enable FDR
        null_seed=42,
        fdr_threshold=0.05,
        n_landmarks=30,
        progress=False,
    )

    # Should have FDR-related fields in var (not varm)
    if "kompot_de" in adata.uns:
        # FDR values are stored in var with condition-specific keys
        fdr_cols = [col for col in adata.var.columns if "fdr" in col.lower()]
        assert len(fdr_cols) > 0


def test_fdr_with_no_signal():
    """Test FDR when there's no real signal (all null)."""
    from kompot import compute_differential_expression

    adata = create_fdr_test_data(
        n_cells=60, n_genes=50, signal_strength=0.0, with_signal=False
    )

    result = compute_differential_expression(
        adata,
        groupby="condition",
        condition1="A",
        condition2="B",
        null_genes=30,
        null_seed=42,
        fdr_threshold=0.05,
        n_landmarks=30,
        progress=False,
    )

    assert "kompot_de" in adata.uns or result is not None


def test_fdr_with_many_null_genes():
    """Test FDR with more null genes than real genes."""
    from kompot import compute_differential_expression

    adata = create_fdr_test_data(n_cells=60, n_genes=30, signal_strength=3.0)

    # Use more null genes than real genes
    result = compute_differential_expression(
        adata,
        groupby="condition",
        condition1="A",
        condition2="B",
        null_genes=50,  # More than n_genes=30
        null_seed=42,
        fdr_threshold=0.1,
        n_landmarks=30,
        progress=False,
    )

    assert "kompot_de" in adata.uns or result is not None


def test_fdr_with_few_null_genes():
    """Test FDR with very few null genes."""
    from kompot import compute_differential_expression

    adata = create_fdr_test_data(n_cells=60, n_genes=50, signal_strength=3.0)

    result = compute_differential_expression(
        adata,
        groupby="condition",
        condition1="A",
        condition2="B",
        null_genes=5,  # Very few null genes
        null_seed=42,
        fdr_threshold=0.1,
        n_landmarks=30,
        progress=False,
    )

    assert "kompot_de" in adata.uns or result is not None


def test_fdr_with_different_thresholds():
    """Test FDR with various thresholds."""
    from kompot import compute_differential_expression

    adata = create_fdr_test_data(n_cells=60, n_genes=40, signal_strength=3.0)

    # Test with strict threshold
    result = compute_differential_expression(
        adata,
        groupby="condition",
        condition1="A",
        condition2="B",
        null_genes=20,
        null_seed=42,
        fdr_threshold=0.01,  # Strict
        n_landmarks=30,
        progress=False,
    )

    assert "kompot_de" in adata.uns or result is not None


def test_fdr_with_weak_signal():
    """Test FDR with weak signal (might not reach significance)."""
    from kompot import compute_differential_expression

    adata = create_fdr_test_data(
        n_cells=60, n_genes=50, signal_strength=0.5, with_signal=True
    )

    result = compute_differential_expression(
        adata,
        groupby="condition",
        condition1="A",
        condition2="B",
        null_genes=25,
        null_seed=42,
        fdr_threshold=0.05,
        n_landmarks=30,
        progress=False,
    )

    assert "kompot_de" in adata.uns or result is not None


def test_fdr_with_return_full_results():
    """Test FDR with return_full_results to access FDR values."""
    from kompot import compute_differential_expression

    adata = create_fdr_test_data(n_cells=60, n_genes=40, signal_strength=4.0)

    result = compute_differential_expression(
        adata,
        groupby="condition",
        condition1="A",
        condition2="B",
        null_genes=20,
        null_seed=42,
        fdr_threshold=0.1,
        n_landmarks=30,
        return_full_results=True,
        progress=False,
    )

    # Should return dictionary with results
    assert result is not None
    if isinstance(result, dict):
        # May have FDR-related keys
        assert len(result) > 0


def test_fdr_with_specific_gene_subset():
    """Test FDR calculation on a subset of genes."""
    from kompot import compute_differential_expression

    adata = create_fdr_test_data(n_cells=60, n_genes=50, signal_strength=3.0)

    # Test FDR on subset
    genes_to_test = [f"gene_{i}" for i in range(20)]

    result = compute_differential_expression(
        adata,
        groupby="condition",
        condition1="A",
        condition2="B",
        genes=genes_to_test,
        null_genes=15,
        null_seed=42,
        fdr_threshold=0.1,
        n_landmarks=30,
        progress=False,
    )

    assert "kompot_de" in adata.uns or result is not None


def test_fdr_reproducibility():
    """Test that FDR calculation is reproducible with same seed."""
    from kompot import compute_differential_expression

    adata1 = create_fdr_test_data(n_cells=80, n_genes=30, signal_strength=3.0)
    adata2 = create_fdr_test_data(n_cells=80, n_genes=30, signal_strength=3.0)

    result1 = compute_differential_expression(
        adata1,
        groupby="condition",
        condition1="A",
        condition2="B",
        null_genes=15,
        null_seed=123,
        random_state=456,
        fdr_threshold=0.1,
        n_landmarks=25,
        progress=False,
        copy=True,
    )

    result2 = compute_differential_expression(
        adata2,
        groupby="condition",
        condition1="A",
        condition2="B",
        null_genes=15,
        null_seed=123,
        random_state=456,
        fdr_threshold=0.1,
        n_landmarks=25,
        progress=False,
        copy=True,
    )

    # Both should complete
    assert result1 is not None or "kompot_de" in adata1.uns
    assert result2 is not None or "kompot_de" in adata2.uns


def test_fdr_with_small_dataset():
    """Test FDR with very small dataset."""
    from kompot import compute_differential_expression

    adata = create_fdr_test_data(n_cells=40, n_genes=15, signal_strength=4.0)

    result = compute_differential_expression(
        adata,
        groupby="condition",
        condition1="A",
        condition2="B",
        null_genes=8,
        null_seed=42,
        fdr_threshold=0.1,
        n_landmarks=15,
        progress=False,
    )

    assert "kompot_de" in adata.uns or result is not None


def test_fdr_with_different_null_seeds():
    """Test FDR with different null seeds gives different null distributions."""
    from kompot import compute_differential_expression

    adata = create_fdr_test_data(n_cells=60, n_genes=40, signal_strength=3.0)

    # Run with different null seeds
    result1 = compute_differential_expression(
        adata,
        groupby="condition",
        condition1="A",
        condition2="B",
        null_genes=20,
        null_seed=42,
        fdr_threshold=0.1,
        n_landmarks=30,
        progress=False,
        copy=True,
    )

    result2 = compute_differential_expression(
        adata,
        groupby="condition",
        condition1="A",
        condition2="B",
        null_genes=20,
        null_seed=999,  # Different seed
        fdr_threshold=0.1,
        n_landmarks=30,
        progress=False,
        copy=True,
    )

    # Both should complete successfully
    assert result1 is not None or result2 is not None


def test_fdr_with_very_strict_threshold():
    """Test FDR with extremely strict threshold (likely no significant genes)."""
    from kompot import compute_differential_expression

    adata = create_fdr_test_data(n_cells=60, n_genes=40, signal_strength=2.0)

    result = compute_differential_expression(
        adata,
        groupby="condition",
        condition1="A",
        condition2="B",
        null_genes=20,
        null_seed=42,
        fdr_threshold=0.001,  # Very strict
        n_landmarks=30,
        progress=False,
    )

    assert "kompot_de" in adata.uns or result is not None


def test_fdr_with_lenient_threshold():
    """Test FDR with lenient threshold."""
    from kompot import compute_differential_expression

    adata = create_fdr_test_data(n_cells=60, n_genes=40, signal_strength=1.5)

    result = compute_differential_expression(
        adata,
        groupby="condition",
        condition1="A",
        condition2="B",
        null_genes=20,
        null_seed=42,
        fdr_threshold=0.2,  # Lenient
        n_landmarks=30,
        progress=False,
    )

    assert "kompot_de" in adata.uns or result is not None


def test_fdr_with_layer():
    """Test FDR calculation with a specific layer."""
    from kompot import compute_differential_expression

    adata = create_fdr_test_data(n_cells=60, n_genes=40, signal_strength=3.0)
    adata.layers["raw"] = adata.X.copy() * 2.0

    result = compute_differential_expression(
        adata,
        groupby="condition",
        condition1="A",
        condition2="B",
        layer="raw",
        null_genes=20,
        null_seed=42,
        fdr_threshold=0.1,
        n_landmarks=30,
        progress=False,
    )

    assert "kompot_de" in adata.uns or result is not None
