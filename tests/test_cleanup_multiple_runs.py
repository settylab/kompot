"""Tests for cleanup utilities with multiple consecutive runs."""

import numpy as np
import pandas as pd
import pytest


def create_test_adata_for_multiple_runs(n_cells=60, n_genes=50):
    """Create test AnnData for cleanup testing with multiple runs."""
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


class TestCleanupMultipleRuns:
    """Tests for cleanup with multiple consecutive runs."""

    def test_cleanup_multiple_runs_separate_result_keys(self):
        """Test cleanup with multiple runs using different result_keys."""
        try:
            from kompot.anndata import compute_differential_expression, cleanup, RunInfo
        except ImportError:
            pytest.skip("anndata not installed")

        adata = create_test_adata_for_multiple_runs()

        # Run three consecutive analyses with different result_keys
        compute_differential_expression(
            adata, groupby='condition', condition1='A', condition2='B',
            obsm_key='X_pca', result_key='run1', null_genes=10,
            store_additional_stats=True, progress=False, n_landmarks=5
        )

        compute_differential_expression(
            adata, groupby='condition', condition1='A', condition2='B',
            obsm_key='X_pca', result_key='run2', null_genes=10,
            store_additional_stats=True, progress=False, n_landmarks=5
        )

        compute_differential_expression(
            adata, groupby='condition', condition1='A', condition2='B',
            obsm_key='X_pca', result_key='run3', null_genes=10,
            store_additional_stats=True, progress=False, n_landmarks=5
        )

        # Verify all runs created their fields
        assert 'run1_A_imputed' in adata.layers
        assert 'run2_A_imputed' in adata.layers
        assert 'run3_A_imputed' in adata.layers

        # Clean up run 1
        cleanup(adata, run_ids=0, keep_layers=False)

        # Run 1 layers should be gone
        assert 'run1_A_imputed' not in adata.layers
        assert 'run1_B_imputed' not in adata.layers

        # Run 2 and 3 layers should still exist
        assert 'run2_A_imputed' in adata.layers
        assert 'run3_A_imputed' in adata.layers

        # Verify RunInfo for run 1 shows missing fields
        run_info_1 = RunInfo(adata, run_id=0, analysis_type='de')
        assert len(run_info_1.missing_fields) > 0

        # Verify RunInfo for run 2 and 3 show no missing fields
        run_info_2 = RunInfo(adata, run_id=1, analysis_type='de')
        run_info_3 = RunInfo(adata, run_id=2, analysis_type='de')
        assert len(run_info_2.missing_fields) == 0
        assert len(run_info_3.missing_fields) == 0

        # Verify package versions are stored in run info
        assert 'package_versions' in run_info_2.environment
        pkg_versions = run_info_2.environment['package_versions']
        expected_packages = ['kompot', 'anndata', 'jax', 'jaxlib', 'numpy', 'scipy', 'pandas']
        for pkg in expected_packages:
            assert pkg in pkg_versions, f"Package {pkg} not found in package_versions"
            assert isinstance(pkg_versions[pkg], str)
            assert len(pkg_versions[pkg]) > 0

    def test_cleanup_with_overwritten_fields(self):
        """Test cleanup when fields have been overwritten by later runs."""
        try:
            from kompot.anndata import compute_differential_expression, cleanup, RunInfo
        except ImportError:
            pytest.skip("anndata not installed")

        adata = create_test_adata_for_multiple_runs()

        # Run three consecutive analyses with SAME result_key (overwriting)
        compute_differential_expression(
            adata, groupby='condition', condition1='A', condition2='B',
            obsm_key='X_pca', result_key='test', null_genes=10,
            store_additional_stats=True, progress=False, n_landmarks=5
        )

        compute_differential_expression(
            adata, groupby='condition', condition1='A', condition2='B',
            obsm_key='X_pca', result_key='test', null_genes=10,
            store_additional_stats=True, progress=False, n_landmarks=5,
            overwrite=True
        )

        compute_differential_expression(
            adata, groupby='condition', condition1='A', condition2='B',
            obsm_key='X_pca', result_key='test', null_genes=10,
            store_additional_stats=True, progress=False, n_landmarks=5,
            overwrite=True
        )

        # Check that first run shows overwritten fields
        run_info_0 = RunInfo(adata, run_id=0, analysis_type='de')
        assert len(run_info_0.overwritten_fields) > 0, "First run should have overwritten fields"

        # Clean up the latest run (run_ids=2)
        cleanup(adata, run_ids=2, keep_layers=False)

        # Now run 2 should have missing fields
        run_info_2 = RunInfo(adata, run_id=2, analysis_type='de')
        assert len(run_info_2.missing_fields) > 0, "Latest run should have missing fields after cleanup"

        # Run 0 should still show overwritten (not missing), because the fields
        # were overwritten BEFORE being deleted
        run_info_0_after = RunInfo(adata, run_id=0, analysis_type='de')
        assert len(run_info_0_after.overwritten_fields) > 0, "First run should still show overwritten"

    def test_missing_takes_precedence_over_overwritten(self):
        """Test that missing status takes precedence over overwritten status."""
        try:
            from kompot.anndata import compute_differential_expression, cleanup, RunInfo
        except ImportError:
            pytest.skip("anndata not installed")

        adata = create_test_adata_for_multiple_runs()

        # Run two analyses with same result_key
        compute_differential_expression(
            adata, groupby='condition', condition1='A', condition2='B',
            obsm_key='X_pca', result_key='test', null_genes=10,
            store_additional_stats=True, progress=False, n_landmarks=5
        )

        compute_differential_expression(
            adata, groupby='condition', condition1='A', condition2='B',
            obsm_key='X_pca', result_key='test', null_genes=10,
            store_additional_stats=True, progress=False, n_landmarks=5,
            overwrite=True
        )

        # First run's fields are overwritten
        run_info_0_before = RunInfo(adata, run_id=0, analysis_type='de')
        assert len(run_info_0_before.overwritten_fields) > 0

        # Now delete the fields
        cleanup(adata, run_ids=1, keep_layers=False)

        # First run should now show missing (not overwritten) for those fields
        # Because the actual data is MISSING, which takes precedence
        run_info_0_after = RunInfo(adata, run_id=0, analysis_type='de')

        # The fields that were overwritten should now show as missing
        # because they don't exist in the AnnData object
        assert len(run_info_0_after.missing_fields) > 0, "First run should show missing fields"

        # Verify the summary reflects this
        summary_0 = run_info_0_after.get_summary()
        assert summary_0['missing_field_count'] > 0

    def test_cleanup_all_runs(self):
        """Test cleaning up all runs in sequence."""
        try:
            from kompot.anndata import compute_differential_expression, cleanup, RunInfo
        except ImportError:
            pytest.skip("anndata not installed")

        adata = create_test_adata_for_multiple_runs()

        # Run three analyses
        for i in range(3):
            compute_differential_expression(
                adata, groupby='condition', condition1='A', condition2='B',
                obsm_key='X_pca', result_key=f'run{i}', null_genes=10,
                progress=False, n_landmarks=5
            )

        # Clean up all runs
        for i in range(3):
            cleanup(adata, run_ids=i, keep_layers=False)

        # All runs should show missing fields
        for i in range(3):
            run_info = RunInfo(adata, run_id=i, analysis_type='de')
            assert len(run_info.missing_fields) > 0, f"Run {i} should have missing fields"

        # No layers should remain
        assert len(adata.layers) == 0

    def test_cleanup_with_partial_overlap(self):
        """Test cleanup when runs have partial field overlap."""
        try:
            from kompot.anndata import compute_differential_expression, cleanup, RunInfo
        except ImportError:
            pytest.skip("anndata not installed")

        adata = create_test_adata_for_multiple_runs()

        # Run 1: with additional stats
        compute_differential_expression(
            adata, groupby='condition', condition1='A', condition2='B',
            obsm_key='X_pca', result_key='run1', null_genes=10,
            store_additional_stats=True, progress=False, n_landmarks=5
        )

        # Run 2: without additional stats (fewer fields)
        compute_differential_expression(
            adata, groupby='condition', condition1='A', condition2='B',
            obsm_key='X_pca', result_key='run2', null_genes=10,
            store_additional_stats=False, progress=False, n_landmarks=5
        )

        # Verify run 1 has more var fields than run 2
        run_info_1 = RunInfo(adata, run_id=0, analysis_type='de')
        run_info_2 = RunInfo(adata, run_id=1, analysis_type='de')

        run_1_var_count = len(run_info_1.adata_fields.get('var', []))
        run_2_var_count = len(run_info_2.adata_fields.get('var', []))
        assert run_1_var_count > run_2_var_count, "Run 1 should have more var fields"

        # Clean up both runs
        cleanup(adata, run_ids=0, keep_layers=False)
        cleanup(adata, run_ids=1, keep_layers=False)

        # Both should show missing layer fields
        run_info_1_after = RunInfo(adata, run_id=0, analysis_type='de')
        run_info_2_after = RunInfo(adata, run_id=1, analysis_type='de')

        assert len(run_info_1_after.missing_fields) > 0
        assert len(run_info_2_after.missing_fields) > 0

    def test_cleanup_only_specific_field_types_multiple_runs(self):
        """Test selective cleanup of specific field types across multiple runs."""
        try:
            from kompot.anndata import compute_differential_expression, cleanup
        except ImportError:
            pytest.skip("anndata not installed")

        adata = create_test_adata_for_multiple_runs()

        # Run two analyses
        for i in range(2):
            compute_differential_expression(
                adata, groupby='condition', condition1='A', condition2='B',
                obsm_key='X_pca', result_key=f'run{i}', null_genes=10,
                store_additional_stats=True, progress=False, n_landmarks=5
            )

        # Clean up only imputed layers from both runs, keep fold_change
        cleanup(
            adata, run_ids=0,
            keep_layers=['fold_change', 'fold_change_zscores']
        )
        cleanup(
            adata, run_ids=1,
            keep_layers=['fold_change', 'fold_change_zscores']
        )

        # Imputed layers should be gone
        assert 'run0_A_imputed' not in adata.layers
        assert 'run1_A_imputed' not in adata.layers

        # Fold change layers should remain
        assert 'run0_A_to_B_fold_change' in adata.layers
        assert 'run1_A_to_B_fold_change' in adata.layers

    def test_runinfo_html_display_with_missing_and_overwritten(self):
        """Test that RunInfo HTML display correctly shows both missing and overwritten."""
        try:
            from kompot.anndata import compute_differential_expression, cleanup, RunInfo
        except ImportError:
            pytest.skip("anndata not installed")

        adata = create_test_adata_for_multiple_runs()

        # Create scenario with both overwritten and missing fields
        compute_differential_expression(
            adata, groupby='condition', condition1='A', condition2='B',
            obsm_key='X_pca', result_key='test', null_genes=10,
            store_additional_stats=True, progress=False, n_landmarks=5
        )

        compute_differential_expression(
            adata, groupby='condition', condition1='A', condition2='B',
            obsm_key='X_pca', result_key='test', null_genes=10,
            store_additional_stats=True, progress=False, n_landmarks=5,
            overwrite=True
        )

        # Clean up second run's layers
        cleanup(adata, run_ids=1, keep_layers=False)

        # Get HTML representation of first run
        run_info_0 = RunInfo(adata, run_id=0, analysis_type='de')
        html = run_info_0._repr_html_()

        # Check that HTML contains both status indicators
        assert 'Missing/Deleted' in html or 'missing' in html.lower(), "HTML should mention missing fields"

        # Get summary
        summary = run_info_0.get_summary()
        assert summary['missing_field_count'] > 0

    def test_cleanup_preserves_run_history(self):
        """Test that cleanup doesn't modify run_history."""
        try:
            from kompot.anndata import compute_differential_expression, cleanup
        except ImportError:
            pytest.skip("anndata not installed")

        adata = create_test_adata_for_multiple_runs()

        # Run analysis
        compute_differential_expression(
            adata, groupby='condition', condition1='A', condition2='B',
            obsm_key='X_pca', result_key='test', null_genes=10,
            store_additional_stats=True, progress=False, n_landmarks=5
        )

        # Get original run history
        from kompot.anndata.utils.json_utils import from_json_string
        original_history = from_json_string(adata.uns['kompot_de']['run_history'])
        original_history_len = len(original_history)

        # Clean up
        cleanup(adata, keep_layers=False)

        # Run history should be unchanged
        after_history = from_json_string(adata.uns['kompot_de']['run_history'])
        assert len(after_history) == original_history_len, "Run history length should not change"

        # Field mapping should still exist
        assert 'field_mapping' in after_history[0], "Field mapping should still exist"

    def test_get_field_status_multiple_runs(self):
        """Test get_field_status with multiple runs."""
        try:
            from kompot.anndata import compute_differential_expression, cleanup, get_field_status
        except ImportError:
            pytest.skip("anndata not installed")

        adata = create_test_adata_for_multiple_runs()

        # Run two analyses
        compute_differential_expression(
            adata, groupby='condition', condition1='A', condition2='B',
            obsm_key='X_pca', result_key='run1', null_genes=10,
            progress=False, n_landmarks=5
        )

        compute_differential_expression(
            adata, groupby='condition', condition1='A', condition2='B',
            obsm_key='X_pca', result_key='run2', null_genes=10,
            progress=False, n_landmarks=5
        )

        # Get status for run 1 before cleanup
        status_before = get_field_status(adata, run_id=0)

        # Check that layers and primary var/obs fields are present
        # (PTP might not be present if store_additional_stats=False)
        if 'layers' in status_before:
            for field_type, fields in status_before['layers'].items():
                for field_name, is_present in fields.items():
                    assert is_present, f"Layer field {field_name} should be present before cleanup"

        if 'var' in status_before:
            for field_type, fields in status_before['var'].items():
                # Skip checking ptp if store_additional_stats was False
                if field_type == 'ptp':
                    continue
                for field_name, is_present in fields.items():
                    assert is_present, f"Var field {field_name} should be present before cleanup"

        # Clean up run 1
        cleanup(adata, run_ids=0, keep_layers=False)

        # Get status for run 1 after cleanup
        status_after = get_field_status(adata, run_id=0)

        # Layer fields should be missing
        if 'layers' in status_after:
            assert any(not is_present for fields in status_after['layers'].values()
                      for is_present in fields.values())

        # Get status for run 2 (should be unaffected)
        status_run2 = get_field_status(adata, run_id=1)

        # Check that layers and primary fields are present for run 2
        if 'layers' in status_run2:
            for field_type, fields in status_run2['layers'].items():
                for field_name, is_present in fields.items():
                    assert is_present, f"Run 2 layer field {field_name} should still be present"

        if 'var' in status_run2:
            for field_type, fields in status_run2['var'].items():
                # Skip PTP check
                if field_type == 'ptp':
                    continue
                for field_name, is_present in fields.items():
                    assert is_present, f"Run 2 var field {field_name} should still be present"


class TestCleanupEdgeCases:
    """Test edge cases and error handling."""

    def test_cleanup_nonexistent_run(self):
        """Test cleanup with non-existent run ID."""
        try:
            from kompot.anndata import compute_differential_expression, cleanup
        except ImportError:
            pytest.skip("anndata not installed")

        adata = create_test_adata_for_multiple_runs()

        compute_differential_expression(
            adata, groupby='condition', condition1='A', condition2='B',
            obsm_key='X_pca', result_key='test', null_genes=10,
            progress=False, n_landmarks=5
        )

        # Try to clean up non-existent run
        # Should handle gracefully (likely return None or raise informative error)
        result = cleanup(adata, run_ids=999, keep_layers=False)
        # Function should return None when it can't get run info

    def test_cleanup_already_cleaned(self):
        """Test cleaning up the same run twice."""
        try:
            from kompot.anndata import compute_differential_expression, cleanup
        except ImportError:
            pytest.skip("anndata not installed")

        adata = create_test_adata_for_multiple_runs()

        compute_differential_expression(
            adata, groupby='condition', condition1='A', condition2='B',
            obsm_key='X_pca', result_key='test', null_genes=10,
            progress=False, n_landmarks=5
        )

        # Clean up once
        cleanup(adata, keep_layers=False)

        # Clean up again (should be idempotent)
        cleanup(adata, keep_layers=False)

        # Should complete without errors

    def test_cleanup_with_no_fields_to_delete(self):
        """Test cleanup when keep parameters result in no deletions."""
        try:
            from kompot.anndata import compute_differential_expression, cleanup
        except ImportError:
            pytest.skip("anndata not installed")

        adata = create_test_adata_for_multiple_runs()

        compute_differential_expression(
            adata, groupby='condition', condition1='A', condition2='B',
            obsm_key='X_pca', result_key='test', null_genes=10,
            progress=False, n_landmarks=5
        )

        # Keep everything
        cleanup(
            adata,
            keep_layers=True,
            keep_var_fields=True,
            keep_obs_fields=True,
            keep_obsp_fields=True,
            keep_varm_fields=True
        )

        # Everything should still be there
        assert 'test_A_imputed' in adata.layers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
