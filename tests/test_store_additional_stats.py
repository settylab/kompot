"""Tests for the store_additional_stats parameter in compute_differential_expression."""

import numpy as np
import pandas as pd
import pytest


def create_simple_test_data(n_cells=60, n_genes=50):
    """Create simple test AnnData for testing."""
    import anndata as ad

    np.random.seed(42)

    # Create conditions
    n_cells_cond1 = n_cells // 2
    n_cells_cond2 = n_cells - n_cells_cond1

    # Create cell states
    X_cond1 = np.random.normal(0, 1, (n_cells_cond1, 5))
    X_cond2 = np.random.normal([0.5, 0.2] + [0]*3, 1, (n_cells_cond2, 5))
    X_combined = np.vstack([X_cond1, X_cond2])

    # Create expression data
    expr_base = np.random.negative_binomial(10, 0.3, (n_cells, n_genes)).astype(float)

    # Create AnnData
    gene_names = [f"Gene_{i:04d}" for i in range(n_genes)]
    cell_names = [f"Cell_{i:04d}" for i in range(n_cells)]

    adata = ad.AnnData(
        X=expr_base,
        obs=pd.DataFrame({
            'condition': ['A'] * n_cells_cond1 + ['B'] * n_cells_cond2,
        }, index=cell_names),
        var=pd.DataFrame(index=gene_names)
    )

    adata.obsm['X_pca'] = X_combined

    return adata


class TestStoreAdditionalStats:
    """Tests for the store_additional_stats parameter."""

    def test_default_behavior_stores_minimal_fields(self):
        """Test that default (False) stores only primary significance measures."""
        try:
            from kompot.anndata import compute_differential_expression
        except ImportError:
            pytest.skip("anndata not installed")

        adata = create_simple_test_data()

        compute_differential_expression(
            adata,
            groupby='condition',
            condition1='A',
            condition2='B',
            obsm_key='X_pca',
            result_key='test_default',
            null_genes=10,
            null_seed=42,
            progress=False,
            n_landmarks=5
        )

        # Check that primary measures ARE stored
        assert 'test_default_A_to_B_mahalanobis' in adata.var.columns
        assert 'test_default_A_to_B_mean_lfc' in adata.var.columns
        assert 'test_default_A_to_B_mahalanobis_local_fdr' in adata.var.columns
        assert 'test_default_A_to_B_is_de' in adata.var.columns

        # Check that additional measures are NOT stored
        assert 'test_default_A_to_B_mahalanobis_pvalue' not in adata.var.columns
        assert 'test_default_A_to_B_mahalanobis_tail_fdr' not in adata.var.columns
        assert 'test_default_A_to_B_ptp' not in adata.var.columns
        assert 'test_default_A_to_B_fold_change_zscores' not in adata.layers

    def test_store_additional_stats_true_stores_all_fields(self):
        """Test that store_additional_stats=True stores all statistical measures."""
        try:
            from kompot.anndata import compute_differential_expression
        except ImportError:
            pytest.skip("anndata not installed")

        adata = create_simple_test_data()

        compute_differential_expression(
            adata,
            groupby='condition',
            condition1='A',
            condition2='B',
            obsm_key='X_pca',
            result_key='test_all_stats',
            null_genes=10,
            null_seed=42,
            store_additional_stats=True,
            progress=False,
            n_landmarks=5
        )

        # Check that ALL measures are stored
        assert 'test_all_stats_A_to_B_mahalanobis' in adata.var.columns
        assert 'test_all_stats_A_to_B_mean_lfc' in adata.var.columns
        assert 'test_all_stats_A_to_B_mahalanobis_local_fdr' in adata.var.columns
        assert 'test_all_stats_A_to_B_is_de' in adata.var.columns

        # Additional stats should be stored
        assert 'test_all_stats_A_to_B_mahalanobis_pvalue' in adata.var.columns
        assert 'test_all_stats_A_to_B_mahalanobis_tail_fdr' in adata.var.columns
        assert 'test_all_stats_A_to_B_ptp' in adata.var.columns
        assert 'test_all_stats_A_to_B_fold_change_zscores' in adata.layers

    def test_pvalue_ranges_when_stored(self):
        """Test that p-values have valid ranges when stored."""
        try:
            from kompot.anndata import compute_differential_expression
        except ImportError:
            pytest.skip("anndata not installed")

        adata = create_simple_test_data()

        compute_differential_expression(
            adata,
            groupby='condition',
            condition1='A',
            condition2='B',
            obsm_key='X_pca',
            result_key='test_pvalue',
            null_genes=10,
            null_seed=42,
            store_additional_stats=True,
            progress=False,
            n_landmarks=5
        )

        # Check p-values are in valid range [0, 1]
        pvalues = adata.var['test_pvalue_A_to_B_mahalanobis_pvalue']
        assert np.all(pvalues >= 0), "P-values should be >= 0"
        assert np.all(pvalues <= 1), "P-values should be <= 1"
        assert not np.any(np.isnan(pvalues)), "P-values should not be NaN"

    def test_fdr_values_consistency(self):
        """Test that local FDR and tail FDR are both stored when requested."""
        try:
            from kompot.anndata import compute_differential_expression
        except ImportError:
            pytest.skip("anndata not installed")

        adata = create_simple_test_data()

        compute_differential_expression(
            adata,
            groupby='condition',
            condition1='A',
            condition2='B',
            obsm_key='X_pca',
            result_key='test_fdr',
            null_genes=10,
            null_seed=42,
            store_additional_stats=True,
            progress=False,
            n_landmarks=5
        )

        local_fdr = adata.var['test_fdr_A_to_B_mahalanobis_local_fdr']
        tail_fdr = adata.var['test_fdr_A_to_B_mahalanobis_tail_fdr']

        # Both should be valid probabilities
        assert np.all(local_fdr >= 0) and np.all(local_fdr <= 1)
        assert np.all(tail_fdr >= 0) and np.all(tail_fdr <= 1)

        # Both should not be all NaN
        assert not np.all(np.isnan(local_fdr))
        assert not np.all(np.isnan(tail_fdr))

    def test_fold_change_zscores_stored_conditionally(self):
        """Test that fold_change_zscores layer is stored only when requested."""
        try:
            from kompot.anndata import compute_differential_expression
        except ImportError:
            pytest.skip("anndata not installed")

        adata1 = create_simple_test_data()
        adata2 = create_simple_test_data()

        # Default: should NOT store fold_change_zscores
        compute_differential_expression(
            adata1,
            groupby='condition',
            condition1='A',
            condition2='B',
            obsm_key='X_pca',
            result_key='test_no_zscores',
            null_genes=10,
            progress=False,
            n_landmarks=5
        )

        assert 'test_no_zscores_A_to_B_fold_change_zscores' not in adata1.layers
        assert 'test_no_zscores_A_to_B_fold_change' in adata1.layers  # Regular fold change should still be there

        # With store_additional_stats=True: SHOULD store fold_change_zscores
        compute_differential_expression(
            adata2,
            groupby='condition',
            condition1='A',
            condition2='B',
            obsm_key='X_pca',
            result_key='test_with_zscores',
            null_genes=10,
            store_additional_stats=True,
            progress=False,
            n_landmarks=5
        )

        assert 'test_with_zscores_A_to_B_fold_change_zscores' in adata2.layers
        assert 'test_with_zscores_A_to_B_fold_change' in adata2.layers

    def test_ptp_stored_conditionally(self):
        """Test that PTP is stored only when store_additional_stats=True."""
        try:
            from kompot.anndata import compute_differential_expression
        except ImportError:
            pytest.skip("anndata not installed")

        adata1 = create_simple_test_data()
        adata2 = create_simple_test_data()

        # Default: should NOT store PTP
        compute_differential_expression(
            adata1,
            groupby='condition',
            condition1='A',
            condition2='B',
            obsm_key='X_pca',
            result_key='test_no_ptp',
            null_genes=10,
            progress=False,
            n_landmarks=5
        )

        assert 'test_no_ptp_A_to_B_ptp' not in adata1.var.columns

        # With store_additional_stats=True: SHOULD store PTP
        compute_differential_expression(
            adata2,
            groupby='condition',
            condition1='A',
            condition2='B',
            obsm_key='X_pca',
            result_key='test_with_ptp',
            null_genes=10,
            store_additional_stats=True,
            progress=False,
            n_landmarks=5
        )

        assert 'test_with_ptp_A_to_B_ptp' in adata2.var.columns
        # Check PTP values are non-negative
        assert np.all(adata2.var['test_with_ptp_A_to_B_ptp'] >= 0)

    def test_storage_consistency_between_adata_and_results(self):
        """Test that what's stored in adata matches what's in results dictionary."""
        try:
            from kompot.anndata import compute_differential_expression
        except ImportError:
            pytest.skip("anndata not installed")

        adata = create_simple_test_data()

        # Run with store_additional_stats=True
        results = compute_differential_expression(
            adata,
            groupby='condition',
            condition1='A',
            condition2='B',
            obsm_key='X_pca',
            result_key='test_consistency',
            null_genes=10,
            store_additional_stats=True,
            return_full_results=True,
            progress=False,
            n_landmarks=5
        )

        # Check that results dictionary has all measures in table DataFrame
        assert 'table' in results
        assert 'pvalue' in results['table'].columns
        assert 'local_fdr' in results['table'].columns
        assert 'tail_fdr' in results['table'].columns

        # Check that adata has corresponding columns
        assert 'test_consistency_A_to_B_mahalanobis_pvalue' in adata.var.columns
        assert 'test_consistency_A_to_B_mahalanobis_local_fdr' in adata.var.columns
        assert 'test_consistency_A_to_B_mahalanobis_tail_fdr' in adata.var.columns

        # Check that values match
        np.testing.assert_array_equal(
            results['table']['pvalue'].values,
            adata.var['test_consistency_A_to_B_mahalanobis_pvalue'].values
        )

    def test_return_full_results_includes_all_when_requested(self):
        """Test that return_full_results includes all stats when store_additional_stats=True."""
        try:
            from kompot.anndata import compute_differential_expression
        except ImportError:
            pytest.skip("anndata not installed")

        adata = create_simple_test_data()

        results = compute_differential_expression(
            adata,
            groupby='condition',
            condition1='A',
            condition2='B',
            obsm_key='X_pca',
            result_key='test_results',
            null_genes=10,
            null_seed=42,
            store_additional_stats=True,
            return_full_results=True,
            progress=False,
            n_landmarks=5
        )

        # Check that results dictionary includes all measures in table DataFrame
        assert 'table' in results
        assert 'pvalue' in results['table'].columns
        assert 'local_fdr' in results['table'].columns
        assert 'tail_fdr' in results['table'].columns
        assert 'is_de' in results['table'].columns

        # Check shapes
        assert len(results['table']['pvalue']) == adata.n_vars
        assert len(results['table']['local_fdr']) == adata.n_vars
        assert len(results['table']['tail_fdr']) == adata.n_vars
        assert len(results['table']['is_de']) == adata.n_vars
