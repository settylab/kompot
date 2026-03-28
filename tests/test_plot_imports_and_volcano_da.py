"""Tests to improve coverage for kompot.plot.__init__ and kompot.plot.volcano.da."""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock
import anndata
import json


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_da_adata(n_cells=80):
    """Create a minimal AnnData with DA result columns and run history."""
    np.random.seed(42)

    X = np.random.normal(0, 1, (n_cells, 5))

    lfc = np.random.normal(0, 2, n_cells)
    ptp = np.random.uniform(0.001, 1.0, n_cells)
    direction = np.where(lfc > 1, "up", np.where(lfc < -1, "down", "neutral"))

    obs = pd.DataFrame(
        {
            "kompot_da_lfc_A_to_B": lfc,
            "kompot_da_ptp_A_to_B": ptp,
            "kompot_da_direction_A_to_B": pd.Categorical(direction),
            "cell_type": pd.Categorical(
                np.random.choice(["T cell", "B cell", "Monocyte"], n_cells)
            ),
            "numeric_score": np.random.rand(n_cells),
        }
    )

    adata = anndata.AnnData(X=X, obs=obs)

    # Minimal run history so _infer_da_keys can work
    run_entry = {
        "run_id": 0,
        "params": {
            "condition1": "A",
            "condition2": "B",
            "log_fold_change_threshold": 1.0,
            "ptp_threshold": 0.05,
        },
        "field_mapping": {
            "lfc_key": "kompot_da_lfc_A_to_B",
            "ptp_key": "kompot_da_ptp_A_to_B",
            "direction_key": "kompot_da_direction_A_to_B",
        },
    }
    adata.uns["kompot_da"] = {"run_history": json.dumps([run_entry])}

    return adata


# ===========================================================================
# Tests for kompot.plot.__init__  (covers lines 13-34, 40-57, 63-77, 83-97,
#                                   103-107, 113-120, 130-131)
# ===========================================================================


class TestPlotInitImports:
    """Verify every public name in kompot.plot is importable."""

    def test_all_names_importable(self):
        """Each name listed in kompot.plot.__all__ should be accessible."""
        import kompot.plot as kp_plot

        for name in kp_plot.__all__:
            obj = getattr(kp_plot, name, None)
            assert obj is not None, f"kompot.plot.{name} is not accessible"

    def test_volcano_de_accessible(self):
        from kompot.plot import volcano_de
        assert callable(volcano_de)

    def test_volcano_da_accessible(self):
        from kompot.plot import volcano_da
        assert callable(volcano_da)

    def test_multi_volcano_da_accessible(self):
        from kompot.plot import multi_volcano_da
        assert callable(multi_volcano_da)

    def test_heatmap_accessible(self):
        from kompot.plot import heatmap
        assert callable(heatmap)

    def test_direction_barplot_accessible(self):
        from kompot.plot import direction_barplot
        assert callable(direction_barplot)

    def test_plot_gene_expression_accessible(self):
        from kompot.plot import plot_gene_expression
        assert callable(plot_gene_expression)

    def test_embedding_accessible(self):
        from kompot.plot import embedding
        assert callable(embedding)

    def test_plot_imputation_accessible(self):
        from kompot.plot import plot_imputation
        assert callable(plot_imputation)

    def test_stringdb_report_accessible(self):
        from kompot.plot import StringDBReport
        assert StringDBReport is not None

    def test_field_inference_accessible(self):
        from kompot.plot import infer_fields_from_run_info, get_comparison_specific_fields
        assert callable(infer_fields_from_run_info)
        assert callable(get_comparison_specific_fields)


# ===========================================================================
# Tests for kompot.plot.volcano.da  (covers many uncovered lines)
# ===========================================================================


class TestVolcanoDA:
    """Tests for volcano_da covering various parameter paths."""

    @pytest.fixture
    def da_adata(self):
        return _make_da_adata()

    # ---- Basic call (covers lines 164-325, 474-554) ----

    def test_basic_call_explicit_keys(self, da_adata):
        """Basic volcano_da with explicit keys, no color."""
        from kompot.plot.volcano.da import volcano_da

        fig = volcano_da(
            da_adata,
            lfc_key="kompot_da_lfc_A_to_B",
            ptp_key="kompot_da_ptp_A_to_B",
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_basic_inferred_keys(self, da_adata):
        """volcano_da inferring keys from run history."""
        from kompot.plot.volcano.da import volcano_da

        fig = volcano_da(da_adata, return_fig=True)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    # ---- Custom thresholds (covers lines 190-196, 280-305) ----

    def test_custom_lfc_threshold_only(self, da_adata):
        """Only LFC threshold, no PTP threshold (covers line 305)."""
        from kompot.plot.volcano.da import volcano_da

        # Remove auto-thresholds from run history so ptp_threshold stays None
        rh = json.loads(da_adata.uns["kompot_da"]["run_history"])
        rh[0]["params"].pop("ptp_threshold", None)
        rh[0]["params"].pop("log_fold_change_threshold", None)
        da_adata.uns["kompot_da"]["run_history"] = json.dumps(rh)

        fig = volcano_da(
            da_adata,
            lfc_key="kompot_da_lfc_A_to_B",
            ptp_key="kompot_da_ptp_A_to_B",
            lfc_threshold=0.5,
            ptp_threshold=None,
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_ptp_threshold_only(self, da_adata):
        """Only PTP threshold, no LFC threshold (covers line 302)."""
        from kompot.plot.volcano.da import volcano_da

        # Remove auto-thresholds from run history so lfc_threshold stays None
        rh = json.loads(da_adata.uns["kompot_da"]["run_history"])
        rh[0]["params"].pop("ptp_threshold", None)
        rh[0]["params"].pop("log_fold_change_threshold", None)
        da_adata.uns["kompot_da"]["run_history"] = json.dumps(rh)

        fig = volcano_da(
            da_adata,
            lfc_key="kompot_da_lfc_A_to_B",
            ptp_key="kompot_da_ptp_A_to_B",
            lfc_threshold=None,
            ptp_threshold=0.1,
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_both_thresholds(self, da_adata):
        from kompot.plot.volcano.da import volcano_da

        fig = volcano_da(
            da_adata,
            lfc_key="kompot_da_lfc_A_to_B",
            ptp_key="kompot_da_ptp_A_to_B",
            lfc_threshold=1.0,
            ptp_threshold=0.05,
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_no_thresholds(self, da_adata):
        """No thresholds at all (line 308 – all significant)."""
        from kompot.plot.volcano.da import volcano_da

        # Remove run-history thresholds so nothing is inferred
        run_history = json.loads(da_adata.uns["kompot_da"]["run_history"])
        run_history[0]["params"].pop("log_fold_change_threshold", None)
        run_history[0]["params"].pop("ptp_threshold", None)
        da_adata.uns["kompot_da"]["run_history"] = json.dumps(run_history)

        fig = volcano_da(
            da_adata,
            lfc_key="kompot_da_lfc_A_to_B",
            ptp_key="kompot_da_ptp_A_to_B",
            lfc_threshold=None,
            ptp_threshold=None,
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    # ---- Color with categorical column (covers lines 328-456) ----

    def test_color_categorical(self, da_adata):
        from kompot.plot.volcano.da import volcano_da

        fig = volcano_da(
            da_adata,
            lfc_key="kompot_da_lfc_A_to_B",
            ptp_key="kompot_da_ptp_A_to_B",
            lfc_threshold=0.5,
            ptp_threshold=0.5,
            color="cell_type",
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_color_categorical_with_palette_str(self, da_adata):
        from kompot.plot.volcano.da import volcano_da

        fig = volcano_da(
            da_adata,
            lfc_key="kompot_da_lfc_A_to_B",
            ptp_key="kompot_da_ptp_A_to_B",
            lfc_threshold=0.5,
            ptp_threshold=0.5,
            color="cell_type",
            palette="viridis",
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_color_categorical_with_palette_dict(self, da_adata):
        from kompot.plot.volcano.da import volcano_da

        fig = volcano_da(
            da_adata,
            lfc_key="kompot_da_lfc_A_to_B",
            ptp_key="kompot_da_ptp_A_to_B",
            lfc_threshold=0.5,
            ptp_threshold=0.5,
            color="cell_type",
            palette={"T cell": "red", "B cell": "blue", "Monocyte": "green"},
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_color_categorical_stored_colors(self, da_adata):
        """When adata.uns has pre-stored category colors."""
        from kompot.plot.volcano.da import volcano_da

        da_adata.uns["cell_type_colors"] = ["#ff0000", "#00ff00", "#0000ff"]

        fig = volcano_da(
            da_adata,
            lfc_key="kompot_da_lfc_A_to_B",
            ptp_key="kompot_da_ptp_A_to_B",
            lfc_threshold=0.5,
            ptp_threshold=0.5,
            color="cell_type",
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_color_categorical_legend_loc_custom(self, da_adata):
        """Legend placed explicitly (not 'best') – covers the else branch."""
        from kompot.plot.volcano.da import volcano_da

        fig = volcano_da(
            da_adata,
            lfc_key="kompot_da_lfc_A_to_B",
            ptp_key="kompot_da_ptp_A_to_B",
            lfc_threshold=0.5,
            ptp_threshold=0.5,
            color="cell_type",
            legend_loc="upper right",
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_color_categorical_legend_hidden(self, da_adata):
        """show_legend=False – covers skipping legend branch."""
        from kompot.plot.volcano.da import volcano_da

        fig = volcano_da(
            da_adata,
            lfc_key="kompot_da_lfc_A_to_B",
            ptp_key="kompot_da_ptp_A_to_B",
            lfc_threshold=0.5,
            ptp_threshold=0.5,
            color="cell_type",
            show_legend=False,
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_color_categorical_many_categories(self, da_adata):
        """More than 10 categories triggers multi-column legend."""
        from kompot.plot.volcano.da import volcano_da

        cats = [f"type_{i}" for i in range(15)]
        da_adata.obs["many_types"] = pd.Categorical(
            np.random.choice(cats, da_adata.n_obs)
        )

        fig = volcano_da(
            da_adata,
            lfc_key="kompot_da_lfc_A_to_B",
            ptp_key="kompot_da_ptp_A_to_B",
            lfc_threshold=0.0,
            ptp_threshold=1.0,
            color="many_types",
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_color_categorical_legend_ncol(self, da_adata):
        """Explicit legend_ncol parameter."""
        from kompot.plot.volcano.da import volcano_da

        fig = volcano_da(
            da_adata,
            lfc_key="kompot_da_lfc_A_to_B",
            ptp_key="kompot_da_ptp_A_to_B",
            lfc_threshold=0.5,
            ptp_threshold=0.5,
            color="cell_type",
            legend_ncol=2,
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    # ---- Color with numeric column (covers lines 458-473) ----

    def test_color_numeric(self, da_adata):
        from kompot.plot.volcano.da import volcano_da

        fig = volcano_da(
            da_adata,
            lfc_key="kompot_da_lfc_A_to_B",
            ptp_key="kompot_da_ptp_A_to_B",
            lfc_threshold=0.5,
            ptp_threshold=0.5,
            color="numeric_score",
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_color_numeric_no_colorbar(self, da_adata):
        from kompot.plot.volcano.da import volcano_da

        fig = volcano_da(
            da_adata,
            lfc_key="kompot_da_lfc_A_to_B",
            ptp_key="kompot_da_ptp_A_to_B",
            lfc_threshold=0.5,
            ptp_threshold=0.5,
            color="numeric_score",
            show_colorbar=False,
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    # ---- Color with string (non-categorical) column (covers lines 358-363) ----

    def test_color_string_column_converted(self, da_adata):
        """String obs column that is not categorical gets auto-converted."""
        from kompot.plot.volcano.da import volcano_da

        da_adata.obs["str_col"] = np.random.choice(["alpha", "beta"], da_adata.n_obs)
        # Ensure it's plain object, not Categorical
        da_adata.obs["str_col"] = da_adata.obs["str_col"].astype(str)

        fig = volcano_da(
            da_adata,
            lfc_key="kompot_da_lfc_A_to_B",
            ptp_key="kompot_da_ptp_A_to_B",
            lfc_threshold=0.5,
            ptp_threshold=0.5,
            color="str_col",
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    # ---- Color key missing (covers line 354 warning) ----

    def test_color_missing_key_warns(self, da_adata):
        from kompot.plot.volcano.da import volcano_da

        with pytest.warns(UserWarning, match="not found in adata.obs"):
            fig = volcano_da(
                da_adata,
                lfc_key="kompot_da_lfc_A_to_B",
                ptp_key="kompot_da_ptp_A_to_B",
                lfc_threshold=0.5,
                ptp_threshold=0.5,
                color="nonexistent_column",
                return_fig=True,
            )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    # ---- Scanpy not available branch (covers lines 328-340) ----

    def test_color_without_scanpy(self, da_adata):
        from kompot.plot.volcano import da as da_module
        from kompot.plot.volcano.da import volcano_da

        original = da_module._has_scanpy
        try:
            da_module._has_scanpy = False
            with pytest.warns(UserWarning, match="Scanpy is required"):
                fig = volcano_da(
                    da_adata,
                    lfc_key="kompot_da_lfc_A_to_B",
                    ptp_key="kompot_da_ptp_A_to_B",
                    lfc_threshold=0.5,
                    ptp_threshold=0.5,
                    color="cell_type",
                    return_fig=True,
                )
            assert isinstance(fig, plt.Figure)
            plt.close(fig)
        finally:
            da_module._has_scanpy = original

    # ---- highlight_subset (covers line 312) ----

    def test_highlight_subset(self, da_adata):
        from kompot.plot.volcano.da import volcano_da

        mask = np.zeros(da_adata.n_obs, dtype=bool)
        mask[:10] = True

        fig = volcano_da(
            da_adata,
            lfc_key="kompot_da_lfc_A_to_B",
            ptp_key="kompot_da_ptp_A_to_B",
            highlight_subset=mask,
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    # ---- update_direction=True (covers lines 199-211) ----

    def test_update_direction(self, da_adata):
        from kompot.plot.volcano.da import volcano_da

        fig = volcano_da(
            da_adata,
            lfc_key="kompot_da_lfc_A_to_B",
            ptp_key="kompot_da_ptp_A_to_B",
            lfc_threshold=0.5,
            ptp_threshold=0.1,
            update_direction=True,
            direction_column="kompot_da_direction_A_to_B",
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    # ---- log_transform_ptp=False (covers line 276-277) ----

    def test_log_transform_ptp_false(self, da_adata):
        from kompot.plot.volcano.da import volcano_da

        fig = volcano_da(
            da_adata,
            lfc_key="kompot_da_lfc_A_to_B",
            ptp_key="kompot_da_ptp_A_to_B",
            log_transform_ptp=False,
            ptp_threshold=0.5,
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    # ---- neg_log10 key branch (covers lines 267-271, 281-288, 491-500) ----

    def test_neg_log10_ptp_key(self, da_adata):
        """PTP key containing 'neg_log10' triggers the pre-transformed branch."""
        from kompot.plot.volcano.da import volcano_da

        da_adata.obs["neg_log10_ptp"] = -np.log10(
            da_adata.obs["kompot_da_ptp_A_to_B"].values
        )

        fig = volcano_da(
            da_adata,
            lfc_key="kompot_da_lfc_A_to_B",
            ptp_key="neg_log10_ptp",
            ptp_threshold=0.05,
            lfc_threshold=1.0,
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_neg_log10_ptp_key_threshold_already_transformed(self, da_adata):
        """neg_log10 key with ptp_threshold > 1 (already on -log10 scale)."""
        from kompot.plot.volcano.da import volcano_da

        da_adata.obs["neg_log10_ptp"] = -np.log10(
            da_adata.obs["kompot_da_ptp_A_to_B"].values
        )

        fig = volcano_da(
            da_adata,
            lfc_key="kompot_da_lfc_A_to_B",
            ptp_key="neg_log10_ptp",
            ptp_threshold=2.0,  # already on -log10 scale
            lfc_threshold=1.0,
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    # ---- show_thresholds=False (covers the skip-threshold path) ----

    def test_show_thresholds_false(self, da_adata):
        from kompot.plot.volcano.da import volcano_da

        fig = volcano_da(
            da_adata,
            lfc_key="kompot_da_lfc_A_to_B",
            ptp_key="kompot_da_ptp_A_to_B",
            lfc_threshold=1.0,
            ptp_threshold=0.05,
            show_thresholds=False,
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    # ---- Positive run_id (covers line 180) ----

    def test_positive_run_id(self, da_adata):
        from kompot.plot.volcano.da import volcano_da

        fig = volcano_da(
            da_adata,
            lfc_key="kompot_da_lfc_A_to_B",
            ptp_key="kompot_da_ptp_A_to_B",
            run_id=0,
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    # ---- No run history (covers line 178) ----

    def test_negative_run_id_no_history(self, da_adata):
        from kompot.plot.volcano.da import volcano_da

        # Remove run history
        del da_adata.uns["kompot_da"]

        fig = volcano_da(
            da_adata,
            lfc_key="kompot_da_lfc_A_to_B",
            ptp_key="kompot_da_ptp_A_to_B",
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    # ---- Provided ax (covers line 261) ----

    def test_provided_ax(self, da_adata):
        from kompot.plot.volcano.da import volcano_da

        fig_ext, ax_ext = plt.subplots()
        result = volcano_da(
            da_adata,
            lfc_key="kompot_da_lfc_A_to_B",
            ptp_key="kompot_da_ptp_A_to_B",
            ax=ax_ext,
            return_fig=True,
        )
        assert result is fig_ext
        plt.close(fig_ext)

    # ---- legend_loc not 'best' without color (covers line 535) ----

    def test_legend_loc_not_best_no_color(self, da_adata):
        from kompot.plot.volcano.da import volcano_da

        fig = volcano_da(
            da_adata,
            lfc_key="kompot_da_lfc_A_to_B",
            ptp_key="kompot_da_ptp_A_to_B",
            legend_loc="upper right",
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    # ---- legend_loc='none' (no legend drawn) ----

    def test_legend_loc_none(self, da_adata):
        from kompot.plot.volcano.da import volcano_da

        fig = volcano_da(
            da_adata,
            lfc_key="kompot_da_lfc_A_to_B",
            ptp_key="kompot_da_ptp_A_to_B",
            legend_loc="none",
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    # ---- save parameter (covers line 551) ----

    def test_save_figure(self, da_adata, tmp_path):
        from kompot.plot.volcano.da import volcano_da

        save_path = str(tmp_path / "volcano.png")
        volcano_da(
            da_adata,
            lfc_key="kompot_da_lfc_A_to_B",
            ptp_key="kompot_da_ptp_A_to_B",
            save=save_path,
        )
        import os

        assert os.path.exists(save_path)
        plt.close("all")

    # ---- return_fig=False returns None (covers line 554) ----

    def test_return_none(self, da_adata):
        from kompot.plot.volcano.da import volcano_da

        result = volcano_da(
            da_adata,
            lfc_key="kompot_da_lfc_A_to_B",
            ptp_key="kompot_da_ptp_A_to_B",
        )
        assert result is None
        plt.close("all")

    # ---- grid=False ----

    def test_no_grid(self, da_adata):
        from kompot.plot.volcano.da import volcano_da

        fig = volcano_da(
            da_adata,
            lfc_key="kompot_da_lfc_A_to_B",
            ptp_key="kompot_da_ptp_A_to_B",
            grid=False,
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    # ---- title=None (covers branch at line 519) ----

    def test_no_title(self, da_adata):
        from kompot.plot.volcano.da import volcano_da

        fig = volcano_da(
            da_adata,
            lfc_key="kompot_da_lfc_A_to_B",
            ptp_key="kompot_da_ptp_A_to_B",
            title=None,
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    # ---- legend_fontsize and legend_title_fontsize (covers line 451-452) ----

    def test_legend_font_sizes(self, da_adata):
        from kompot.plot.volcano.da import volcano_da

        fig = volcano_da(
            da_adata,
            lfc_key="kompot_da_lfc_A_to_B",
            ptp_key="kompot_da_ptp_A_to_B",
            lfc_threshold=0.5,
            ptp_threshold=0.5,
            color="cell_type",
            legend_fontsize=8,
            legend_title_fontsize=10,
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    # ---- ptp_threshold with log_transform_ptp=False (covers line 292) ----

    def test_ptp_threshold_no_log_transform(self, da_adata):
        from kompot.plot.volcano.da import volcano_da

        fig = volcano_da(
            da_adata,
            lfc_key="kompot_da_lfc_A_to_B",
            ptp_key="kompot_da_ptp_A_to_B",
            log_transform_ptp=False,
            ptp_threshold=0.5,
            lfc_threshold=1.0,
            show_thresholds=True,
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    # ---- Condition extraction from key name fallback (covers lines 225-235) ----

    def test_condition_extraction_fallback(self, da_adata):
        """When run_info has no condition params, falls back to key parsing."""
        from kompot.plot.volcano.da import volcano_da

        run_history = json.loads(da_adata.uns["kompot_da"]["run_history"])
        run_history[0]["params"].pop("condition1", None)
        run_history[0]["params"].pop("condition2", None)
        da_adata.uns["kompot_da"]["run_history"] = json.dumps(run_history)

        fig = volcano_da(
            da_adata,
            lfc_key="kompot_da_lfc_A_to_B",
            ptp_key="kompot_da_ptp_A_to_B",
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
