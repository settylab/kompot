"""Tests for impute RunInfo, cleanup, get_field_status, and plot integration."""

import numpy as np
import pandas as pd
import pytest
import anndata

import kompot
from kompot.anndata.utils.runinfo import RunInfo, RunComparison
from kompot.anndata.cleanup import cleanup, get_field_status


@pytest.fixture
def adata_with_impute():
    """Create adata with two impute runs (all cells, then condition A)."""
    rng = np.random.RandomState(42)
    n_cells, n_features, n_genes = 80, 5, 8
    X_obsm = rng.randn(n_cells, n_features)
    X_expr = rng.randn(n_cells, n_genes)
    obs = pd.DataFrame({
        "condition": ["A"] * 40 + ["B"] * 40,
    })
    ad = anndata.AnnData(X=X_expr, obs=obs)
    ad.obsm["DM_EigenVectors"] = X_obsm
    ad.obsm["X_umap"] = rng.randn(n_cells, 2)
    ad.var_names = [f"gene_{i}" for i in range(n_genes)]

    # Run 0: all cells
    kompot.impute_expression(
        ad, n_landmarks=30, progress=False,
        use_empirical_variance=True,
    )
    # Run 1: condition A only
    kompot.impute_expression(
        ad, groupby="condition", condition="A",
        n_landmarks=30, progress=False,
        overwrite=True,
    )
    return ad


class TestRunInfoImpute:

    def test_auto_detect_impute(self, adata_with_impute):
        """RunInfo auto-detects impute when only impute runs exist."""
        # Remove DE/DA keys if present
        for k in list(adata_with_impute.uns.keys()):
            if k in ("kompot_de", "kompot_da"):
                del adata_with_impute.uns[k]
        ri = RunInfo(adata_with_impute)
        assert ri.analysis_type == "impute"

    def test_explicit_impute(self, adata_with_impute):
        ri = RunInfo(adata_with_impute, analysis_type="impute")
        assert ri.analysis_type == "impute"
        assert ri.storage_key == "kompot_impute"

    def test_run_id_access(self, adata_with_impute):
        ri0 = RunInfo(adata_with_impute, run_id=0, analysis_type="impute")
        ri1 = RunInfo(adata_with_impute, run_id=1, analysis_type="impute")
        assert ri0.adjusted_run_id == 0
        assert ri1.adjusted_run_id == 1

    def test_negative_run_id(self, adata_with_impute):
        ri = RunInfo(adata_with_impute, run_id=-1, analysis_type="impute")
        assert ri.adjusted_run_id == 1  # Most recent

    def test_params(self, adata_with_impute):
        ri = RunInfo(adata_with_impute, run_id=0, analysis_type="impute")
        assert ri.params["use_empirical_variance"] is True
        assert ri.params["condition"] is None

        ri1 = RunInfo(adata_with_impute, run_id=1, analysis_type="impute")
        assert ri1.params["condition"] == "A"

    def test_field_mapping_present(self, adata_with_impute):
        ri = RunInfo(adata_with_impute, run_id=0, analysis_type="impute")
        raw = ri.get_raw_data()
        assert "field_mapping" in raw
        fm = raw["field_mapping"]
        assert any("imputed" in k for k in fm)
        assert any("std" in k for k in fm)
        assert any("obs_variance" in k for k in fm)

    def test_adata_fields_populated(self, adata_with_impute):
        ri = RunInfo(adata_with_impute, run_id=0, analysis_type="impute")
        assert "layers" in ri.adata_fields
        assert len(ri.adata_fields["layers"]) >= 2

    def test_predict_on_all_cells(self, adata_with_impute):
        """Model trained on condition A should predict on all cells."""
        ad = adata_with_impute
        # Run 1 was condition A (40 cells), but layers should cover all 80
        layer = ad.layers["kompot_impute_A_imputed"]
        assert layer.shape[0] == ad.n_obs
        assert not np.any(np.isnan(layer))  # no NaN for non-condition cells

    def test_summary_conditions(self, adata_with_impute):
        ri0 = RunInfo(adata_with_impute, run_id=0, analysis_type="impute")
        assert ri0.get_summary()["conditions"] == "all cells"

        ri1 = RunInfo(adata_with_impute, run_id=1, analysis_type="impute")
        assert ri1.get_summary()["conditions"] == "A"

    def test_repr(self, adata_with_impute):
        ri = RunInfo(adata_with_impute, run_id=0, analysis_type="impute")
        r = repr(ri)
        assert "impute" in r.lower()
        assert "run_id=0" in r

    def test_repr_html(self, adata_with_impute):
        ri = RunInfo(adata_with_impute, run_id=0, analysis_type="impute")
        html = ri._repr_html_()
        assert "IMPUTE" in html
        assert "imputed" in html.lower()

    def test_compare_with(self, adata_with_impute):
        ri = RunInfo(adata_with_impute, run_id=0, analysis_type="impute")
        comp = ri.compare_with(1)
        assert isinstance(comp, RunComparison)
        s = comp.get_summary()
        assert s["run1"]["run_id"] == 0
        assert s["run2"]["run_id"] == 1

    def test_invalid_analysis_type_raises(self, adata_with_impute):
        with pytest.raises(ValueError, match="Invalid analysis_type"):
            RunInfo(adata_with_impute, analysis_type="foo")


class TestCleanupImpute:

    def test_cleanup_keeps_imputed_by_default(self, adata_with_impute):
        """Default impute cleanup keeps imputed layers, removes std/obs_variance."""
        ad = adata_with_impute
        assert any("kompot_impute" in k for k in ad.layers)

        cleanup(ad, analysis_type="impute")

        # Imputed layers should be kept
        assert any("_imputed" in k for k in ad.layers if "kompot_impute" in k)
        # Std and obs_variance layers should be removed
        assert not any("_std" in k for k in ad.layers if "kompot_impute" in k)
        assert not any("_obs_variance" in k for k in ad.layers if "kompot_impute" in k)

    def test_cleanup_removes_all_with_explicit_false(self, adata_with_impute):
        """Explicit keep_layers=False removes everything."""
        ad = adata_with_impute
        cleanup(ad, analysis_type="impute", keep_layers=False)
        remaining = [k for k in ad.layers if "kompot_impute" in k]
        assert len(remaining) == 0

    def test_cleanup_single_run(self, adata_with_impute):
        ad = adata_with_impute
        # Only clean run 0 (all cells) — default keeps imputed
        cleanup(ad, run_ids=0, analysis_type="impute")

        # Run 0: imputed kept, std/obs_variance removed
        assert "kompot_impute_all_imputed" in ad.layers
        assert "kompot_impute_all_std" not in ad.layers
        assert "kompot_impute_all_obs_variance" not in ad.layers
        # Run 1 layers (A_imputed, A_std) should remain untouched
        remaining_a = [k for k in ad.layers if "kompot_impute_A" in k]
        assert len(remaining_a) >= 2

    def test_cleanup_keep_layers_true(self, adata_with_impute):
        ad = adata_with_impute
        n_before = sum(1 for k in ad.layers if "kompot_impute" in k)
        cleanup(ad, analysis_type="impute", keep_layers=True)
        n_after = sum(1 for k in ad.layers if "kompot_impute" in k)
        assert n_after == n_before

    def test_cleanup_keep_specific_type(self, adata_with_impute):
        ad = adata_with_impute
        cleanup(ad, run_ids=0, analysis_type="impute", keep_layers=["imputed"])
        # imputed layer kept, std and obs_variance removed
        assert "kompot_impute_all_imputed" in ad.layers
        assert "kompot_impute_all_std" not in ad.layers
        assert "kompot_impute_all_obs_variance" not in ad.layers


class TestGetFieldStatusImpute:

    def test_field_status_all_present(self, adata_with_impute):
        status = get_field_status(adata_with_impute, run_id=0, analysis_type="impute")
        assert "layers" in status
        for field_type_dict in status["layers"].values():
            for is_present in field_type_dict.values():
                assert is_present is True

    def test_field_status_after_cleanup(self, adata_with_impute):
        ad = adata_with_impute
        cleanup(ad, run_ids=0, analysis_type="impute")
        status = get_field_status(ad, run_id=0, analysis_type="impute")
        assert "layers" in status
        # Default impute cleanup keeps imputed, removes others
        for field_type, field_dict in status["layers"].items():
            for name, is_present in field_dict.items():
                if field_type == "imputed":
                    assert is_present is True
                else:
                    assert is_present is False

    def test_missing_fields_detected(self, adata_with_impute):
        ad = adata_with_impute
        cleanup(ad, run_ids=0, analysis_type="impute")
        ri = RunInfo(ad, run_id=0, analysis_type="impute")
        assert len(ri.missing_fields) >= 1  # At least std (imputed is kept)


class TestPlotImputation:

    def test_plot_imputation_basic(self, adata_with_impute):
        fig = kompot.plot.plot_imputation(
            adata_with_impute,
            genes=["gene_0", "gene_1"],
            embedding_key="X_umap",
            condition="all",
            return_fig=True,
        )
        assert fig is not None
        plt.close(fig)

    def test_plot_imputation_auto_condition(self, adata_with_impute):
        # Should auto-detect "all" or "A" from available layers
        fig = kompot.plot.plot_imputation(
            adata_with_impute,
            genes=["gene_0"],
            embedding_key="X_umap",
            return_fig=True,
        )
        assert fig is not None
        plt.close(fig)

    def test_plot_imputation_with_obs_variance(self, adata_with_impute):
        fig = kompot.plot.plot_imputation(
            adata_with_impute,
            genes=["gene_0"],
            embedding_key="X_umap",
            condition="all",
            show_obs_variance=True,
            return_fig=True,
        )
        # Should have 4 rows: raw, imputed, std, obs_variance
        assert fig is not None
        assert fig._kompot_nrows == 4
        plt.close(fig)

    def test_plot_imputation_no_obs_variance(self, adata_with_impute):
        # Both runs now have obs_variance (use_empirical_variance=True default).
        # To test the "no obs_variance" path, explicitly remove the layer.
        layer_key = "kompot_impute_A_obs_variance"
        had_layer = layer_key in adata_with_impute.layers
        if had_layer:
            saved = adata_with_impute.layers[layer_key].copy()
            del adata_with_impute.layers[layer_key]
        try:
            fig = kompot.plot.plot_imputation(
                adata_with_impute,
                genes=["gene_0"],
                embedding_key="X_umap",
                condition="A",
                show_obs_variance=True,
                return_fig=True,
            )
            assert fig is not None
            assert fig._kompot_nrows == 3  # raw, imputed, std (no obs_variance)
            plt.close(fig)
        finally:
            if had_layer:
                adata_with_impute.layers[layer_key] = saved

    def test_plot_imputation_missing_embedding_raises(self, adata_with_impute):
        with pytest.raises(ValueError, match="not found in adata.obsm"):
            kompot.plot.plot_imputation(
                adata_with_impute, genes=["gene_0"],
                embedding_key="nonexistent",
            )

    def test_plot_imputation_missing_layer_raises(self, adata_with_impute):
        with pytest.raises(ValueError, match="No imputation layers found"):
            kompot.plot.plot_imputation(
                adata_with_impute, genes=["gene_0"],
                embedding_key="X_umap",
                result_key="nonexistent_key",
            )


try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
