"""Tests for smooth RunInfo, cleanup, get_field_status, and plot integration."""

import numpy as np
import pandas as pd
import pytest
import anndata

import kompot
from kompot import GPSettings, StorageSettings, OutputSettings
from kompot.anndata.utils.runinfo import RunInfo, RunComparison
from kompot.anndata.cleanup import cleanup, get_field_status


@pytest.fixture
def adata_with_smooth():
    """Create adata with two smooth runs (all cells, then condition A)."""
    rng = np.random.RandomState(42)
    n_cells, n_features, n_genes = 80, 5, 8
    X_obsm = rng.randn(n_cells, n_features)
    X_expr = rng.randn(n_cells, n_genes)
    obs = pd.DataFrame(
        {
            "condition": ["A"] * 40 + ["B"] * 40,
        }
    )
    ad = anndata.AnnData(X=X_expr, obs=obs)
    ad.obsm["DM_EigenVectors"] = X_obsm
    ad.obsm["X_umap"] = rng.randn(n_cells, 2)
    ad.var_names = [f"gene_{i}" for i in range(n_genes)]

    gp_fast = GPSettings(n_landmarks=30, use_empirical_variance=True)
    out = OutputSettings(progress=False)

    # Run 0: all cells
    kompot.smooth_expression(ad, gp=gp_fast, output=out)
    # Run 1: condition A only
    kompot.smooth_expression(
        ad,
        groupby="condition",
        condition="A",
        gp=gp_fast,
        output=out,
        storage=StorageSettings(overwrite=True),
    )
    return ad


class TestRunInfoSmooth:
    def test_auto_detect_smooth(self, adata_with_smooth):
        """RunInfo auto-detects smooth when only smooth runs exist."""
        # Remove DE/DA keys if present
        for k in list(adata_with_smooth.uns.keys()):
            if k in ("kompot_de", "kompot_da"):
                del adata_with_smooth.uns[k]
        ri = RunInfo(adata_with_smooth)
        assert ri.analysis_type == "smooth"

    def test_explicit_smooth(self, adata_with_smooth):
        ri = RunInfo(adata_with_smooth, analysis_type="smooth")
        assert ri.analysis_type == "smooth"
        assert ri.storage_key == "kompot_smooth"

    def test_run_id_access(self, adata_with_smooth):
        ri0 = RunInfo(adata_with_smooth, run_id=0, analysis_type="smooth")
        ri1 = RunInfo(adata_with_smooth, run_id=1, analysis_type="smooth")
        assert ri0.adjusted_run_id == 0
        assert ri1.adjusted_run_id == 1

    def test_negative_run_id(self, adata_with_smooth):
        ri = RunInfo(adata_with_smooth, run_id=-1, analysis_type="smooth")
        assert ri.adjusted_run_id == 1  # Most recent

    def test_params(self, adata_with_smooth):
        ri = RunInfo(adata_with_smooth, run_id=0, analysis_type="smooth")
        assert ri.params["use_empirical_variance"] is True
        assert ri.params["condition"] is None

        ri1 = RunInfo(adata_with_smooth, run_id=1, analysis_type="smooth")
        assert ri1.params["condition"] == "A"

    def test_field_mapping_present(self, adata_with_smooth):
        ri = RunInfo(adata_with_smooth, run_id=0, analysis_type="smooth")
        raw = ri.get_raw_data()
        assert "field_mapping" in raw
        fm = raw["field_mapping"]
        assert any("smoothed" in k for k in fm)
        assert any("std" in k for k in fm)
        assert any("obs_variance" in k for k in fm)

    def test_adata_fields_populated(self, adata_with_smooth):
        ri = RunInfo(adata_with_smooth, run_id=0, analysis_type="smooth")
        assert "layers" in ri.adata_fields
        assert len(ri.adata_fields["layers"]) >= 2

    def test_predict_on_all_cells(self, adata_with_smooth):
        """Model trained on condition A should predict on all cells."""
        ad = adata_with_smooth
        # Run 1 was condition A (40 cells), but layers should cover all 80
        layer = ad.layers["kompot_smooth_A_smoothed"]
        assert layer.shape[0] == ad.n_obs
        assert not np.any(np.isnan(layer))  # no NaN for non-condition cells

    def test_summary_conditions(self, adata_with_smooth):
        ri0 = RunInfo(adata_with_smooth, run_id=0, analysis_type="smooth")
        assert ri0.get_summary()["conditions"] == "all cells"

        ri1 = RunInfo(adata_with_smooth, run_id=1, analysis_type="smooth")
        assert ri1.get_summary()["conditions"] == "A"

    def test_repr(self, adata_with_smooth):
        ri = RunInfo(adata_with_smooth, run_id=0, analysis_type="smooth")
        r = repr(ri)
        assert "smooth" in r.lower()
        assert "run_id=0" in r

    def test_repr_html(self, adata_with_smooth):
        ri = RunInfo(adata_with_smooth, run_id=0, analysis_type="smooth")
        html = ri._repr_html_()
        assert "SMOOTH" in html
        assert "smoothed" in html.lower()

    def test_compare_with(self, adata_with_smooth):
        ri = RunInfo(adata_with_smooth, run_id=0, analysis_type="smooth")
        comp = ri.compare_with(1)
        assert isinstance(comp, RunComparison)
        s = comp.get_summary()
        assert s["run1"]["run_id"] == 0
        assert s["run2"]["run_id"] == 1

    def test_invalid_analysis_type_raises(self, adata_with_smooth):
        with pytest.raises(ValueError, match="Invalid analysis_type"):
            RunInfo(adata_with_smooth, analysis_type="foo")


class TestCleanupSmooth:
    def test_cleanup_keeps_smoothed_by_default(self, adata_with_smooth):
        """Default smooth cleanup keeps smoothed layers, removes std/obs_variance."""
        ad = adata_with_smooth
        assert any(k is not None and "kompot_smooth" in k for k in ad.layers)

        cleanup(ad, analysis_type="smooth")

        # Imputed layers should be kept
        assert any("_smoothed" in k for k in ad.layers if k is not None and "kompot_smooth" in k)
        # Std and obs_variance layers should be removed
        assert not any("_std" in k for k in ad.layers if k is not None and "kompot_smooth" in k)
        assert not any("_obs_variance" in k for k in ad.layers if k is not None and "kompot_smooth" in k)

    def test_cleanup_removes_all_with_explicit_false(self, adata_with_smooth):
        """Explicit keep_layers=False removes everything."""
        ad = adata_with_smooth
        cleanup(ad, analysis_type="smooth", keep_layers=False)
        remaining = [k for k in ad.layers if k is not None and "kompot_smooth" in k]
        assert len(remaining) == 0

    def test_cleanup_single_run(self, adata_with_smooth):
        ad = adata_with_smooth
        # Only clean run 0 (all cells) — default keeps smoothed
        cleanup(ad, run_ids=0, analysis_type="smooth")

        # Run 0: smoothed kept, std/obs_variance removed
        assert "kompot_smooth_all_smoothed" in ad.layers
        assert "kompot_smooth_all_std" not in ad.layers
        assert "kompot_smooth_all_obs_variance" not in ad.layers
        # Run 1 layers (A_smoothed, A_std) should remain untouched
        remaining_a = [k for k in ad.layers if k is not None and "kompot_smooth_A" in k]
        assert len(remaining_a) >= 2

    def test_cleanup_keep_layers_true(self, adata_with_smooth):
        ad = adata_with_smooth
        n_before = sum(1 for k in ad.layers if k is not None and "kompot_smooth" in k)
        cleanup(ad, analysis_type="smooth", keep_layers=True)
        n_after = sum(1 for k in ad.layers if k is not None and "kompot_smooth" in k)
        assert n_after == n_before

    def test_cleanup_keep_specific_type(self, adata_with_smooth):
        ad = adata_with_smooth
        cleanup(ad, run_ids=0, analysis_type="smooth", keep_layers=["smoothed"])
        # smoothed layer kept, std and obs_variance removed
        assert "kompot_smooth_all_smoothed" in ad.layers
        assert "kompot_smooth_all_std" not in ad.layers
        assert "kompot_smooth_all_obs_variance" not in ad.layers


class TestGetFieldStatusSmooth:
    def test_field_status_all_present(self, adata_with_smooth):
        status = get_field_status(adata_with_smooth, run_id=0, analysis_type="smooth")
        assert "layers" in status
        for field_type_dict in status["layers"].values():
            for is_present in field_type_dict.values():
                assert is_present is True

    def test_field_status_after_cleanup(self, adata_with_smooth):
        ad = adata_with_smooth
        cleanup(ad, run_ids=0, analysis_type="smooth")
        status = get_field_status(ad, run_id=0, analysis_type="smooth")
        assert "layers" in status
        # Default smooth cleanup keeps smoothed, removes others
        for field_type, field_dict in status["layers"].items():
            for name, is_present in field_dict.items():
                if field_type == "smoothed":
                    assert is_present is True
                else:
                    assert is_present is False

    def test_missing_fields_detected(self, adata_with_smooth):
        ad = adata_with_smooth
        cleanup(ad, run_ids=0, analysis_type="smooth")
        ri = RunInfo(ad, run_id=0, analysis_type="smooth")
        assert len(ri.missing_fields) >= 1  # At least std (smoothed is kept)


class TestPlotSmoothing:
    def test_plot_smoothing_basic(self, adata_with_smooth):
        fig = kompot.plot.plot_smoothing(
            adata_with_smooth,
            genes=["gene_0", "gene_1"],
            basis="X_umap",
            condition="all",
            return_fig=True,
        )
        assert fig is not None
        plt.close(fig)

    def test_plot_smoothing_auto_condition(self, adata_with_smooth):
        # Should auto-detect "all" or "A" from available layers
        fig = kompot.plot.plot_smoothing(
            adata_with_smooth,
            genes=["gene_0"],
            basis="X_umap",
            return_fig=True,
        )
        assert fig is not None
        plt.close(fig)

    def test_plot_smoothing_with_obs_variance(self, adata_with_smooth):
        fig = kompot.plot.plot_smoothing(
            adata_with_smooth,
            genes=["gene_0"],
            basis="X_umap",
            condition="all",
            show_obs_variance=True,
            return_fig=True,
        )
        # Should have 4 rows: raw, smoothed, std, obs_variance
        assert fig is not None
        assert fig._kompot_nrows == 4
        plt.close(fig)

    def test_plot_smoothing_no_obs_variance(self, adata_with_smooth):
        # Both runs now have obs_variance (use_empirical_variance=True default).
        # To test the "no obs_variance" path, explicitly remove the layer.
        layer_key = "kompot_smooth_A_obs_variance"
        had_layer = layer_key in adata_with_smooth.layers
        if had_layer:
            saved = adata_with_smooth.layers[layer_key].copy()
            del adata_with_smooth.layers[layer_key]
        try:
            fig = kompot.plot.plot_smoothing(
                adata_with_smooth,
                genes=["gene_0"],
                basis="X_umap",
                condition="A",
                show_obs_variance=True,
                return_fig=True,
            )
            assert fig is not None
            assert fig._kompot_nrows == 3  # raw, smoothed, std (no obs_variance)
            plt.close(fig)
        finally:
            if had_layer:
                adata_with_smooth.layers[layer_key] = saved

    def test_plot_smoothing_missing_embedding_raises(self, adata_with_smooth):
        with pytest.raises(ValueError, match="not found in adata.obsm"):
            kompot.plot.plot_smoothing(
                adata_with_smooth,
                genes=["gene_0"],
                basis="nonexistent",
            )

    def test_plot_smoothing_missing_layer_raises(self, adata_with_smooth):
        with pytest.raises(ValueError, match="No smoothing layers found"):
            kompot.plot.plot_smoothing(
                adata_with_smooth,
                genes=["gene_0"],
                basis="X_umap",
                result_key="nonexistent_key",
            )


try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
