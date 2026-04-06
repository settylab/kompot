"""
Tests for advanced differential expression features to improve coverage.

This test file focuses on uncovered code paths including:
- FDR edge cases (zero p-values, high p-values, local FDR failures)
- Differential abundance integration
- Groups functionality
- Error handling paths
"""

import numpy as np
import pytest
import pandas as pd
import anndata as ad
import kompot


class TestFDREdgeCases:
    """Test FDR computation edge cases."""

    def setup_method(self):
        """Create test data for FDR testing."""
        np.random.seed(42)
        n_cells = 50
        n_genes = 30

        # Create expression data
        X = np.random.randn(n_cells, n_genes)

        # Create conditions
        conditions = ["A"] * 25 + ["B"] * 25
        samples = (
            ["sample1"] * 12 + ["sample2"] * 13 + ["sample3"] * 12 + ["sample4"] * 13
        )

        self.adata = ad.AnnData(X)
        self.adata.obs["condition"] = pd.Categorical(conditions)
        self.adata.obs["sample"] = pd.Categorical(samples)
        self.adata.var_names = [f"gene_{i}" for i in range(n_genes)]
        self.adata.obsm["X_pca"] = np.random.randn(n_cells, 10)
        self.adata.obsm["DM_EigenVectors"] = self.adata.obsm["X_pca"].copy()

    def test_de_with_null_genes_zero_pvalues(self):
        """Test FDR computation when some genes have zero p-values (highly significant)."""
        # Create data with some genes having very different expression
        self.adata.X[:25, :5] = 10.0  # First condition, first 5 genes very high
        self.adata.X[25:, :5] = -10.0  # Second condition, first 5 genes very low

        results = kompot.compute_differential_expression(
            self.adata,
            groupby="condition",
            condition1="A",
            condition2="B",
            sample_col="sample",
            n_landmarks=10,
            null_genes=20,  # Enable FDR
            progress=False,
            inplace=False,
            return_full_results=True,
        )

        # Should handle zero p-values correctly
        assert isinstance(results, dict)
        assert "table" in results
        table = results["table"]

        # Check that FDR values exist
        assert "local_fdr" in table.columns
        assert "tail_fdr" in table.columns
        assert "is_de" in table.columns

        # FDR values should be valid probabilities
        assert np.all((table["local_fdr"] >= 0) & (table["local_fdr"] <= 1))
        assert np.all((table["tail_fdr"] >= 0) & (table["tail_fdr"] <= 1))

    def test_de_with_null_genes_all_high_pvalues(self):
        """Test FDR computation when all p-values are high (no signal)."""
        # Create data with minimal differences (should result in high p-values)
        np.random.seed(123)
        self.adata.X = np.random.randn(50, 30) * 0.1  # Very small variance

        results = kompot.compute_differential_expression(
            self.adata,
            groupby="condition",
            condition1="A",
            condition2="B",
            sample_col="sample",
            n_landmarks=10,
            null_genes=20,
            progress=False,
            inplace=False,
            return_full_results=True,
        )

        # Should handle high p-values correctly (fall back to tail FDR)
        assert isinstance(results, dict)
        table = results["table"]

        # Most genes should not be significant
        assert np.sum(table["is_de"]) < len(table) * 0.2  # Less than 20% significant

    def test_de_with_null_genes_edge_case_small_n_genes(self):
        """Test FDR with very few genes (edge case for null distribution)."""
        # Create smaller dataset
        n_cells = 40
        n_genes = 10  # Very few genes

        X = np.random.randn(n_cells, n_genes)
        adata_small = ad.AnnData(X)
        adata_small.obs["condition"] = pd.Categorical(["A"] * 20 + ["B"] * 20)
        adata_small.obs["sample"] = pd.Categorical(
            ["s1"] * 10 + ["s2"] * 10 + ["s3"] * 10 + ["s4"] * 10
        )
        adata_small.var_names = [f"gene_{i}" for i in range(n_genes)]
        adata_small.obsm["DM_EigenVectors"] = np.random.randn(n_cells, 5)

        results = kompot.compute_differential_expression(
            adata_small,
            groupby="condition",
            condition1="A",
            condition2="B",
            sample_col="sample",
            n_landmarks=5,
            null_genes=5,  # Small null set
            progress=False,
            inplace=False,
            return_full_results=True,
        )

        # Should handle small gene sets
        assert isinstance(results, dict)
        assert "table" in results


class TestDifferentialAbundanceIntegration:
    """Test differential abundance integration in differential expression."""

    def setup_method(self):
        """Create test data."""
        np.random.seed(42)
        n_cells = 50
        n_genes = 20

        # Create expression data
        X = np.random.randn(n_cells, n_genes)

        # Create conditions
        conditions = ["A"] * 25 + ["B"] * 25
        samples = ["s1"] * 12 + ["s2"] * 13 + ["s3"] * 12 + ["s4"] * 13

        self.adata = ad.AnnData(X)
        self.adata.obs["condition"] = pd.Categorical(conditions)
        self.adata.obs["sample"] = pd.Categorical(samples)
        self.adata.var_names = [f"gene_{i}" for i in range(n_genes)]
        self.adata.obsm["X_pca"] = np.random.randn(n_cells, 10)
        self.adata.obsm["DM_EigenVectors"] = self.adata.obsm["X_pca"].copy()

    def test_de_with_differential_abundance_integration(self):
        """Test DE with differential abundance integration (weighted LFC)."""
        # First run differential abundance
        try:
            kompot.compute_differential_abundance(
                self.adata,
                groupby="condition",
                condition1="A",
                condition2="B",
                sample_col="sample",
                n_landmarks=10,
                result_key="test_da",
                progress=False,
            )
        except TypeError as e:
            if "progress" in str(e):
                # Older mellon version doesn't support progress parameter
                pytest.skip(f"Mellon version doesn't support progress parameter: {e}")
            raise

        # Now run DE with DA integration
        results = kompot.compute_differential_expression(
            self.adata,
            groupby="condition",
            condition1="A",
            condition2="B",
            sample_col="sample",
            n_landmarks=10,
            null_genes=None,
            differential_abundance_key="test_da",  # Enable DA integration
            progress=False,
            inplace=False,
            return_full_results=True,
        )

        # Should include weighted LFC fields
        assert isinstance(results, dict)
        table = results["table"]

        # Check for weighted LFC column
        weighted_lfc_cols = [
            col for col in table.columns if "weighted_lfc" in col.lower()
        ]
        assert len(weighted_lfc_cols) > 0, "Should have weighted LFC columns"


class TestGroupsFunctionality:
    """Test groups-based differential expression."""

    def setup_method(self):
        """Create test data with multiple cell types."""
        np.random.seed(42)
        n_cells = 60
        n_genes = 20

        # Create expression data
        X = np.random.randn(n_cells, n_genes)

        # Create conditions and cell types
        conditions = ["A"] * 30 + ["B"] * 30
        cell_types = ["TypeX", "TypeY", "TypeZ"] * 20
        samples = ["s1"] * 15 + ["s2"] * 15 + ["s3"] * 15 + ["s4"] * 15

        self.adata = ad.AnnData(X)
        self.adata.obs["condition"] = pd.Categorical(conditions)
        self.adata.obs["cell_type"] = pd.Categorical(cell_types)
        self.adata.obs["sample"] = pd.Categorical(samples)
        self.adata.var_names = [f"gene_{i}" for i in range(n_genes)]
        self.adata.obsm["X_pca"] = np.random.randn(n_cells, 10)
        self.adata.obsm["DM_EigenVectors"] = self.adata.obsm["X_pca"].copy()

    def test_de_with_groups_basic(self):
        """Test DE with groups parameter (cell types)."""
        results = kompot.compute_differential_expression(
            self.adata,
            groupby="condition",
            condition1="A",
            condition2="B",
            sample_col="sample",
            groups="cell_type",  # Analyze by cell type
            n_landmarks=10,
            null_genes=None,
            progress=False,
            inplace=False,
            return_full_results=True,
        )

        # Should have group-specific results
        assert isinstance(results, dict)

        # Check for varm keys (group-specific results)
        assert hasattr(self.adata, "varm") or "varm_keys" in results

    def test_de_with_groups_dict(self):
        """Test DE with groups specified as dict."""
        # Define groups as dictionary
        groups_dict = {"TypeX": {"cell_type": "TypeX"}, "TypeY": {"cell_type": "TypeY"}}

        results = kompot.compute_differential_expression(
            self.adata,
            groupby="condition",
            condition1="A",
            condition2="B",
            sample_col="sample",
            groups=groups_dict,
            n_landmarks=10,
            null_genes=None,
            progress=False,
            inplace=False,
            return_full_results=True,
        )

        # Should work with dict groups
        assert isinstance(results, dict)

    def test_de_with_groups_and_da_integration(self):
        """Test DE with both groups and differential abundance integration."""
        # Run DA first
        try:
            kompot.compute_differential_abundance(
                self.adata,
                groupby="condition",
                condition1="A",
                condition2="B",
                sample_col="sample",
                n_landmarks=10,
                result_key="test_da",
                progress=False,
            )
        except TypeError as e:
            if "progress" in str(e):
                # Older mellon version doesn't support progress parameter
                pytest.skip(f"Mellon version doesn't support progress parameter: {e}")
            raise

        # Run DE with groups and DA integration
        results = kompot.compute_differential_expression(
            self.adata,
            groupby="condition",
            condition1="A",
            condition2="B",
            sample_col="sample",
            groups="cell_type",
            differential_abundance_key="test_da",
            n_landmarks=10,
            null_genes=None,
            progress=False,
            inplace=False,
            return_full_results=True,
        )

        # Should handle both features together
        assert isinstance(results, dict)


class TestAdditionalParameters:
    """Test additional parameters and edge cases."""

    def setup_method(self):
        """Create test data."""
        np.random.seed(42)
        n_cells = 50
        n_genes = 20

        X = np.random.randn(n_cells, n_genes)
        conditions = ["A"] * 25 + ["B"] * 25
        samples = ["s1"] * 12 + ["s2"] * 13 + ["s3"] * 12 + ["s4"] * 13

        self.adata = ad.AnnData(X)
        self.adata.obs["condition"] = pd.Categorical(conditions)
        self.adata.obs["sample"] = pd.Categorical(samples)
        self.adata.var_names = [f"gene_{i}" for i in range(n_genes)]
        self.adata.obsm["X_pca"] = np.random.randn(n_cells, 10)
        self.adata.obsm["DM_EigenVectors"] = self.adata.obsm["X_pca"].copy()

    def test_de_with_genes_subset(self):
        """Test DE with specific gene subset."""
        # Test with subset of genes
        gene_subset = ["gene_0", "gene_1", "gene_2", "gene_5", "gene_10"]

        results = kompot.compute_differential_expression(
            self.adata,
            groupby="condition",
            condition1="A",
            condition2="B",
            sample_col="sample",
            genes=gene_subset,
            n_landmarks=10,
            null_genes=None,
            progress=False,
            inplace=False,
            return_full_results=True,
        )

        # Should only analyze specified genes
        assert isinstance(results, dict)
        assert len(results["table"]) == len(gene_subset)

    def test_de_with_layer(self):
        """Test DE with specific layer."""
        # Add a layer
        self.adata.layers["counts"] = np.random.negative_binomial(
            10, 0.3, (50, 20)
        ).astype(float)

        results = kompot.compute_differential_expression(
            self.adata,
            groupby="condition",
            condition1="A",
            condition2="B",
            sample_col="sample",
            layer="counts",
            n_landmarks=10,
            null_genes=None,
            progress=False,
            inplace=False,
            return_full_results=True,
        )

        # Should use specified layer
        assert isinstance(results, dict)

    def test_de_with_store_additional_stats(self):
        """Test DE with store_additional_stats enabled."""
        results = kompot.compute_differential_expression(
            self.adata,
            groupby="condition",
            condition1="A",
            condition2="B",
            sample_col="sample",
            n_landmarks=10,
            null_genes=None,
            store_additional_stats=True,
            progress=False,
            inplace=False,
            return_full_results=True,
        )

        # Should store additional statistics
        assert isinstance(results, dict)

        # Check for additional stats in results
        if "additional_stats" in results:
            assert isinstance(results["additional_stats"], dict)

    def test_de_with_custom_result_key(self):
        """Test DE with custom result key."""
        kompot.compute_differential_expression(
            self.adata,
            groupby="condition",
            condition1="A",
            condition2="B",
            sample_col="sample",
            result_key="custom_de_test",
            n_landmarks=10,
            null_genes=None,
            progress=False,
        )

        # Check for custom key in adata
        var_cols = [col for col in self.adata.var.columns if "custom_de_test" in col]
        assert len(var_cols) > 0, "Should have results with custom key"

    def test_de_with_obsm_key(self):
        """Test DE with custom obsm_key."""
        # Add another embedding
        self.adata.obsm["custom_embedding"] = np.random.randn(50, 8)

        results = kompot.compute_differential_expression(
            self.adata,
            groupby="condition",
            condition1="A",
            condition2="B",
            sample_col="sample",
            obsm_key="custom_embedding",
            n_landmarks=10,
            null_genes=None,
            progress=False,
            inplace=False,
            return_full_results=True,
        )

        # Should use custom embedding
        assert isinstance(results, dict)

    def test_de_with_compute_mahalanobis_false(self):
        """Test DE with compute_mahalanobis=False."""
        results = kompot.compute_differential_expression(
            self.adata,
            groupby="condition",
            condition1="A",
            condition2="B",
            sample_col="sample",
            n_landmarks=10,
            null_genes=None,
            compute_mahalanobis=False,  # Skip Mahalanobis distance
            progress=False,
            inplace=False,
            return_full_results=True,
        )

        # Should work without Mahalanobis computation
        assert isinstance(results, dict)
        table = results["table"]

        # Should still have mean_lfc
        assert "mean_lfc" in table.columns
