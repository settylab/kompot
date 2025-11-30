"""Tests for cleanup utilities."""

import numpy as np
import pandas as pd
import pytest


def create_test_adata_for_cleanup(n_cells=60, n_genes=50):
    """Create test AnnData for cleanup testing."""
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


class TestCleanupFunctionality:
    """Tests for cleanup function."""

    def test_cleanup_removes_layers_by_default(self):
        """Test that cleanup removes layers by default while keeping var/obs fields."""
        try:
            from kompot.anndata import compute_differential_expression, cleanup
        except ImportError:
            pytest.skip("anndata not installed")

        adata = create_test_adata_for_cleanup()

        # Run differential expression
        compute_differential_expression(
            adata,
            groupby='condition',
            condition1='A',
            condition2='B',
            obsm_key='X_pca',
            result_key='test_cleanup',
            null_genes=10,
            store_additional_stats=True,
            progress=False,
            n_landmarks=5
        )

        # Verify layers exist
        assert 'test_cleanup_A_imputed' in adata.layers
        assert 'test_cleanup_B_imputed' in adata.layers
        assert 'test_cleanup_A_to_B_fold_change' in adata.layers

        # Verify var fields exist
        assert 'test_cleanup_A_to_B_mahalanobis' in adata.var.columns
        assert 'test_cleanup_A_to_B_mean_lfc' in adata.var.columns

        # Run cleanup (default removes layers)
        cleanup(adata)

        # Layers should be gone
        assert 'test_cleanup_A_imputed' not in adata.layers
        assert 'test_cleanup_B_imputed' not in adata.layers
        assert 'test_cleanup_A_to_B_fold_change' not in adata.layers

        # Var fields should still be present
        assert 'test_cleanup_A_to_B_mahalanobis' in adata.var.columns
        assert 'test_cleanup_A_to_B_mean_lfc' in adata.var.columns

    def test_cleanup_keep_specific_layers(self):
        """Test keeping specific layer types."""
        try:
            from kompot.anndata import compute_differential_expression, cleanup
        except ImportError:
            pytest.skip("anndata not installed")

        adata = create_test_adata_for_cleanup()

        compute_differential_expression(
            adata,
            groupby='condition',
            condition1='A',
            condition2='B',
            obsm_key='X_pca',
            result_key='test_keep',
            null_genes=10,
            store_additional_stats=True,
            progress=False,
            n_landmarks=5
        )

        # Keep only fold_change layers
        cleanup(adata, keep_layers=['fold_change'])

        # Imputed layers should be removed
        assert 'test_keep_A_imputed' not in adata.layers
        assert 'test_keep_B_imputed' not in adata.layers

        # Fold change layer should remain
        assert 'test_keep_A_to_B_fold_change' in adata.layers

    def test_cleanup_keep_all_layers(self):
        """Test keeping all layers."""
        try:
            from kompot.anndata import compute_differential_expression, cleanup
        except ImportError:
            pytest.skip("anndata not installed")

        adata = create_test_adata_for_cleanup()

        compute_differential_expression(
            adata,
            groupby='condition',
            condition1='A',
            condition2='B',
            obsm_key='X_pca',
            result_key='test_keep_all',
            null_genes=10,
            progress=False,
            n_landmarks=5
        )

        # Keep all layers
        cleanup(adata, keep_layers=True)

        # All layers should remain
        assert 'test_keep_all_A_imputed' in adata.layers
        assert 'test_keep_all_B_imputed' in adata.layers
        assert 'test_keep_all_A_to_B_fold_change' in adata.layers

    def test_cleanup_remove_var_fields(self):
        """Test removing var fields."""
        try:
            from kompot.anndata import compute_differential_expression, cleanup
        except ImportError:
            pytest.skip("anndata not installed")

        adata = create_test_adata_for_cleanup()

        compute_differential_expression(
            adata,
            groupby='condition',
            condition1='A',
            condition2='B',
            obsm_key='X_pca',
            result_key='test_remove_var',
            null_genes=10,
            store_additional_stats=True,
            progress=False,
            n_landmarks=5
        )

        # Remove all fields
        cleanup(
            adata,
            keep_layers=False,
            keep_var_fields=False
        )

        # Var fields should be removed
        assert 'test_remove_var_A_to_B_mahalanobis' not in adata.var.columns
        assert 'test_remove_var_A_to_B_mean_lfc' not in adata.var.columns
        assert 'test_remove_var_A_to_B_mahalanobis_local_fdr' not in adata.var.columns

    def test_cleanup_keep_specific_var_fields(self):
        """Test keeping only specific var field types."""
        try:
            from kompot.anndata import compute_differential_expression, cleanup
        except ImportError:
            pytest.skip("anndata not installed")

        adata = create_test_adata_for_cleanup()

        compute_differential_expression(
            adata,
            groupby='condition',
            condition1='A',
            condition2='B',
            obsm_key='X_pca',
            result_key='test_keep_var',
            null_genes=10,
            store_additional_stats=True,
            progress=False,
            n_landmarks=5
        )

        # Keep only essential statistical fields
        cleanup(
            adata,
            keep_layers=False,
            keep_var_fields=['mahalanobis', 'mahalanobis_local_fdr', 'is_de', 'mean_log_fold_change']
        )

        # Essential fields should remain
        assert 'test_keep_var_A_to_B_mahalanobis' in adata.var.columns
        assert 'test_keep_var_A_to_B_mahalanobis_local_fdr' in adata.var.columns
        assert 'test_keep_var_A_to_B_is_de' in adata.var.columns
        assert 'test_keep_var_A_to_B_mean_lfc' in adata.var.columns

        # Additional stats should be removed
        assert 'test_keep_var_A_to_B_mahalanobis_pvalue' not in adata.var.columns
        assert 'test_keep_var_A_to_B_mahalanobis_tail_fdr' not in adata.var.columns
        assert 'test_keep_var_A_to_B_ptp' not in adata.var.columns

    def test_cleanup_not_inplace(self):
        """Test that cleanup returns a copy when inplace=False."""
        try:
            from kompot.anndata import compute_differential_expression, cleanup
        except ImportError:
            pytest.skip("anndata not installed")

        adata = create_test_adata_for_cleanup()

        compute_differential_expression(
            adata,
            groupby='condition',
            condition1='A',
            condition2='B',
            obsm_key='X_pca',
            result_key='test_copy',
            null_genes=10,
            progress=False,
            n_landmarks=5
        )

        # Run cleanup with inplace=False
        adata_cleaned = cleanup(adata, inplace=False)

        # Original should still have layers
        assert 'test_copy_A_imputed' in adata.layers

        # Copy should not have layers
        assert 'test_copy_A_imputed' not in adata_cleaned.layers

    def test_get_field_status(self):
        """Test get_field_status function."""
        try:
            from kompot.anndata import compute_differential_expression, cleanup, get_field_status
        except ImportError:
            pytest.skip("anndata not installed")

        adata = create_test_adata_for_cleanup()

        compute_differential_expression(
            adata,
            groupby='condition',
            condition1='A',
            condition2='B',
            obsm_key='X_pca',
            result_key='test_status',
            null_genes=10,
            store_additional_stats=True,
            progress=False,
            n_landmarks=5
        )

        # Get status before cleanup
        status_before = get_field_status(adata)

        # All fields should be present
        for location, types in status_before.items():
            for field_type, fields in types.items():
                for field_name, is_present in fields.items():
                    assert is_present, f"Field {field_name} should be present before cleanup"

        # Clean up layers
        cleanup(adata, keep_layers=False)

        # Get status after cleanup
        status_after = get_field_status(adata)

        # Layer fields should be missing
        if 'layers' in status_after:
            for field_type, fields in status_after['layers'].items():
                for field_name, is_present in fields.items():
                    assert not is_present, f"Layer field {field_name} should be missing after cleanup"

        # Var fields should still be present
        if 'var' in status_after:
            for field_type, fields in status_after['var'].items():
                for field_name, is_present in fields.items():
                    assert is_present, f"Var field {field_name} should still be present after cleanup"

    def test_cleanup_specific_run(self):
        """Test cleaning up a specific run."""
        try:
            from kompot.anndata import compute_differential_expression, cleanup
        except ImportError:
            pytest.skip("anndata not installed")

        adata = create_test_adata_for_cleanup()

        # Run two differential expression analyses
        compute_differential_expression(
            adata,
            groupby='condition',
            condition1='A',
            condition2='B',
            obsm_key='X_pca',
            result_key='test_run1',
            null_genes=10,
            progress=False,
            n_landmarks=5
        )

        compute_differential_expression(
            adata,
            groupby='condition',
            condition1='A',
            condition2='B',
            obsm_key='X_pca',
            result_key='test_run2',
            null_genes=10,
            progress=False,
            n_landmarks=5,
            overwrite=True
        )

        # Clean up only the first run (run_ids=0)
        cleanup(adata, run_ids=0, keep_layers=False)

        # First run layers should be removed
        assert 'test_run1_A_imputed' not in adata.layers
        assert 'test_run1_B_imputed' not in adata.layers

        # Second run layers should still exist
        assert 'test_run2_A_imputed' in adata.layers
        assert 'test_run2_B_imputed' in adata.layers


class TestRunInfoMissingFields:
    """Tests for RunInfo missing field detection."""

    def test_runinfo_detects_missing_fields(self):
        """Test that RunInfo detects missing/deleted fields."""
        try:
            from kompot.anndata import compute_differential_expression, cleanup, RunInfo
        except ImportError:
            pytest.skip("anndata not installed")

        adata = create_test_adata_for_cleanup()

        compute_differential_expression(
            adata,
            groupby='condition',
            condition1='A',
            condition2='B',
            obsm_key='X_pca',
            result_key='test_missing',
            null_genes=10,
            store_additional_stats=True,
            progress=False,
            n_landmarks=5
        )

        # Get run info before cleanup
        run_info_before = RunInfo(adata, analysis_type='de')
        assert len(run_info_before.missing_fields) == 0, "No fields should be missing before cleanup"

        # Clean up layers
        cleanup(adata, keep_layers=False)

        # Get run info after cleanup
        run_info_after = RunInfo(adata, analysis_type='de')
        assert len(run_info_after.missing_fields) > 0, "Some fields should be missing after cleanup"

        # Check that missing fields are layers
        for field_info in run_info_after.missing_fields:
            assert field_info['location'] == 'layers', "Missing fields should be layers"

    def test_runinfo_summary_includes_missing_count(self):
        """Test that RunInfo summary includes missing field count."""
        try:
            from kompot.anndata import compute_differential_expression, cleanup, RunInfo
        except ImportError:
            pytest.skip("anndata not installed")

        adata = create_test_adata_for_cleanup()

        compute_differential_expression(
            adata,
            groupby='condition',
            condition1='A',
            condition2='B',
            obsm_key='X_pca',
            result_key='test_summary',
            null_genes=10,
            progress=False,
            n_landmarks=5
        )

        # Clean up
        cleanup(adata, keep_layers=False)

        # Get run info and summary
        run_info = RunInfo(adata, analysis_type='de')
        summary = run_info.get_summary()

        assert 'missing_field_count' in summary
        assert summary['missing_field_count'] > 0
        assert 'missing_fields' in summary
        assert len(summary['missing_fields']) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
