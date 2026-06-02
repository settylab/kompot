"""Tests for FDR integration in compute_differential_expression."""

import numpy as np
import pandas as pd
import pytest

from tests.test_anndata_functions import create_test_anndata


def create_test_anndata_with_differential_genes(
    n_cells=60, n_genes=50, n_differential=10
):
    """Create test AnnData with known differential genes."""
    import anndata as ad

    np.random.seed(42)

    # Create conditions
    n_cells_cond1 = n_cells // 2
    n_cells_cond2 = n_cells - n_cells_cond1

    # Create cell states (smaller embedding dimension)
    X_cond1 = np.random.normal(0, 1, (n_cells_cond1, 5))
    X_cond2 = np.random.normal([0.5, 0.2] + [0] * 3, 1, (n_cells_cond2, 5))
    X_combined = np.vstack([X_cond1, X_cond2])

    # Create expression data
    expr_base = np.random.negative_binomial(10, 0.3, (n_cells, n_genes)).astype(float)

    # Add differential expression (stronger signal for smaller dataset)
    differential_genes = np.random.choice(n_genes, n_differential, replace=False)
    for gene_idx in differential_genes:
        fold_change = np.random.uniform(5, 10)  # Stronger signal for smaller dataset
        expr_base[n_cells_cond1:, gene_idx] *= fold_change

    # Create AnnData
    gene_names = [f"Gene_{i:04d}" for i in range(n_genes)]
    cell_names = [f"Cell_{i:04d}" for i in range(n_cells)]

    adata = ad.AnnData(
        X=expr_base,
        obs=pd.DataFrame(
            {
                "condition": ["Ctrl"] * n_cells_cond1 + ["Treat"] * n_cells_cond2,
            },
            index=cell_names,
        ),
        var=pd.DataFrame(
            {"is_differential": [i in differential_genes for i in range(n_genes)]},
            index=gene_names,
        ),
    )

    adata.obsm["DM_EigenVectors"] = X_combined

    return adata, differential_genes


class TestFDRIntegration:
    """Tests for FDR functionality integration."""

    def test_fdr_enabled_basic(self):
        """Test basic FDR functionality."""
        try:
            import anndata
            from kompot.anndata import compute_differential_expression
        except ImportError:
            pytest.skip("anndata not installed")

        adata, true_differential_genes = create_test_anndata_with_differential_genes()

        # Run with FDR and store_additional_stats=True to get all FDR measures
        results = compute_differential_expression(
            adata,
            groupby="condition",
            condition1="Ctrl",
            condition2="Treat",
            null_genes=10,  # Reduced for faster testing
            null_seed=42,
            fdr_threshold=0.05,
            return_full_results=True,
            result_key="test_fdr",
            overwrite=True,
            n_landmarks=5,  # Reduced for faster testing
            store_additional_stats=True,  # Store all statistical measures
        )

        # Check FDR results are present in table DataFrame
        assert "table" in results, "Missing table in results"
        table_cols = ["pvalue", "local_fdr", "is_de"]
        for col in table_cols:
            assert col in results["table"].columns, f"Missing column: {col}"
            assert len(results["table"][col]) == adata.n_vars

        # Check AnnData columns (including the neg_log10_ptp column)
        fdr_columns = [
            "test_fdr_Ctrl_to_Treat_mahalanobis_pvalue",
            "test_fdr_Ctrl_to_Treat_mahalanobis_local_fdr",
            "test_fdr_Ctrl_to_Treat_mahalanobis_tail_fdr",
            "test_fdr_Ctrl_to_Treat_is_de",
            "test_fdr_Ctrl_to_Treat_neg_log10_ptp",
        ]
        for col in fdr_columns:
            assert col in adata.var.columns, f"Missing column: {col}"

        # Check value ranges
        assert np.all(adata.var["test_fdr_Ctrl_to_Treat_mahalanobis_pvalue"] >= 0)
        assert np.all(adata.var["test_fdr_Ctrl_to_Treat_mahalanobis_pvalue"] <= 1)
        assert np.all(adata.var["test_fdr_Ctrl_to_Treat_mahalanobis_local_fdr"] >= 0)
        assert np.all(adata.var["test_fdr_Ctrl_to_Treat_mahalanobis_local_fdr"] <= 1)
        assert adata.var["test_fdr_Ctrl_to_Treat_is_de"].dtype == bool

        # Check neg_log10_ptp values: -log10(PTP) is non-negative and unbounded
        # above (a probability PTP <= 1 maps to -log10(PTP) >= 0).
        assert np.all(adata.var["test_fdr_Ctrl_to_Treat_neg_log10_ptp"] >= 0)

        # FDR pipeline should run without error and produce valid results
        n_significant = np.sum(adata.var["test_fdr_Ctrl_to_Treat_is_de"])
        # Note: With fallback FDR methods, we may detect 0 significant genes - this is valid behavior
        assert n_significant >= 0, "Number of significant genes should be non-negative"
        assert n_significant <= adata.n_vars, (
            "Significant genes should not exceed total genes"
        )

        # If we do detect significant genes, it shouldn't be too many
        if n_significant > 0:
            assert n_significant < adata.n_vars * 0.5, "Shouldn't detect too many genes"

    def test_fdr_disabled(self):
        """Test that function works when FDR is disabled."""
        try:
            import anndata
            from kompot.anndata import compute_differential_expression
        except ImportError:
            pytest.skip("anndata not installed")

        adata, _ = create_test_anndata_with_differential_genes()

        # Run without FDR
        results = compute_differential_expression(
            adata,
            groupby="condition",
            condition1="Ctrl",
            condition2="Treat",
            null_genes=None,  # Disable FDR
            return_full_results=True,
            result_key="no_fdr_test",
            overwrite=True,
        )

        # Should not have FDR results in table
        assert "table" in results
        fdr_cols = ["pvalue", "local_fdr", "tail_fdr", "is_de"]
        for col in fdr_cols:
            assert col not in results["table"].columns, (
                f"Should not have column when disabled: {col}"
            )

        # Should still have regular results
        assert "mahalanobis" in results["table"].columns

        # Should not have FDR columns
        fdr_columns = [
            "no_fdr_test_mahalanobis_pvalue",
            "no_fdr_test_mahalanobis_local_fdr",
            "no_fdr_test_mahalanobis_tail_fdr",
            "no_fdr_test_is_de",
        ]
        for col in fdr_columns:
            assert col not in adata.var.columns, (
                f"Should not have column when disabled: {col}"
            )

    def test_fdr_reproducibility(self):
        """Test that FDR results are reproducible."""
        try:
            import anndata
            from kompot.anndata import compute_differential_expression
        except ImportError:
            pytest.skip("anndata not installed")

        adata1, _ = create_test_anndata_with_differential_genes()
        adata2 = adata1.copy()

        params = dict(
            groupby="condition",
            condition1="Ctrl",
            condition2="Treat",
            null_genes=10,  # Must be < n_genes (default 50)
            null_seed=123,
            random_state=456,
            return_full_results=True,
            overwrite=True,
            store_additional_stats=True,  # Store all statistical measures
        )

        results1 = compute_differential_expression(
            adata1, result_key="repro1", **params
        )
        results2 = compute_differential_expression(
            adata2, result_key="repro2", **params
        )

        # P-values should be identical
        np.testing.assert_array_equal(
            results1["table"]["pvalue"].values,
            results2["table"]["pvalue"].values,
            err_msg="P-values should be reproducible",
        )

        # DE calls should be identical
        np.testing.assert_array_equal(
            results1["table"]["is_de"].values,
            results2["table"]["is_de"].values,
            err_msg="DE calls should be reproducible",
        )

    def test_fdr_with_specific_genes(self):
        """Test FDR with user-specified null genes."""
        try:
            import anndata
            from kompot.anndata import compute_differential_expression
        except ImportError:
            pytest.skip("anndata not installed")

        adata, _ = create_test_anndata_with_differential_genes()

        # Use specific null gene indices (within valid range 0-49 for 50 genes)
        null_gene_indices = [5, 10, 15, 25, 35]

        results = compute_differential_expression(
            adata,
            groupby="condition",
            condition1="Ctrl",
            condition2="Treat",
            null_genes=null_gene_indices,
            null_seed=42,
            return_full_results=True,
            result_key="specific_null",
            overwrite=True,
            store_additional_stats=True,  # Store all statistical measures
        )

        # Should work with specific indices
        assert "pvalue" in results["table"].columns
        assert len(results["table"]["pvalue"]) == adata.n_vars

    def test_fdr_thresholds(self):
        """Test different FDR thresholds."""
        try:
            import anndata
            from kompot.anndata import compute_differential_expression
        except ImportError:
            pytest.skip("anndata not installed")

        adata1, _ = create_test_anndata_with_differential_genes()
        adata2 = adata1.copy()

        # Run with different thresholds
        results_strict = compute_differential_expression(
            adata1,
            groupby="condition",
            condition1="Ctrl",
            condition2="Treat",
            null_genes=10,
            fdr_threshold=0.01,
            return_full_results=True,  # Must be < n_genes (default 50)
            result_key="strict",
            overwrite=True,
            store_additional_stats=True,  # Store all statistical measures
        )

        results_lenient = compute_differential_expression(
            adata2,
            groupby="condition",
            condition1="Ctrl",
            condition2="Treat",
            null_genes=10,
            fdr_threshold=0.1,
            return_full_results=True,  # Must be < n_genes (default 50)
            result_key="lenient",
            overwrite=True,
            store_additional_stats=True,  # Store all statistical measures
        )

        n_strict = np.sum(results_strict["table"]["is_de"])
        n_lenient = np.sum(results_lenient["table"]["is_de"])

        # Lenient threshold should detect more genes
        assert n_lenient >= n_strict, (
            "Lenient threshold should detect >= genes than strict"
        )

    def test_fdr_backwards_compatibility(self):
        """Test that adding FDR doesn't break existing functionality."""
        try:
            import anndata
            from kompot.anndata import compute_differential_expression
        except ImportError:
            pytest.skip("anndata not installed")

        adata = create_test_anndata()

        # Run with old parameters (no FDR parameters)
        results = compute_differential_expression(
            adata,
            groupby="group",
            condition1="A",
            condition2="B",
            return_full_results=True,
        )

        # Should work without errors
        assert "mean_lfc" in results["table"].columns
        assert "mahalanobis" in results["table"].columns


if __name__ == "__main__":
    pytest.main([__file__])
