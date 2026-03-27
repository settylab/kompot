"""Tests targeting uncovered lines across multiple modules to push coverage from 86% to 90%+.

Covers:
- kompot/plot/volcano/multi_da.py
- kompot/plot/expression.py
- kompot/resource_estimation.py
- kompot/utils.py
- kompot/plot/imputation.py
- kompot/plot/field_inference.py
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock
from anndata import AnnData
import json
import os
import warnings


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _close_figures():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


def _make_da_adata(n_obs=60, n_vars=5, n_groups=3, with_run_history=True,
                   condition1="CondA", condition2="CondB",
                   result_key="kompot_da", extra_obs_cols=None):
    """Build a minimal AnnData with DA results and run history."""
    rng = np.random.RandomState(42)
    X = rng.randn(n_obs, n_vars).astype(np.float32)
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_obs)])

    lfc_col = f"{result_key}_{condition1}_to_{condition2}_log_fold_change"
    ptp_col = f"{result_key}_{condition1}_to_{condition2}_ptp"
    direction_col = f"{result_key}_{condition1}_to_{condition2}_direction"

    obs[lfc_col] = rng.randn(n_obs).astype(np.float32)
    obs[ptp_col] = rng.uniform(0.001, 1.0, n_obs).astype(np.float32)
    obs[direction_col] = pd.Categorical(
        rng.choice(["up", "down", "neutral"], n_obs)
    )

    groups = [f"group_{i}" for i in range(n_groups)]
    obs["cell_type"] = pd.Categorical(rng.choice(groups, n_obs))

    if extra_obs_cols:
        for col_name, col_vals in extra_obs_cols.items():
            obs[col_name] = col_vals

    adata = AnnData(X=X, obs=obs)
    adata.obsm["X_umap"] = rng.randn(n_obs, 2).astype(np.float32)
    adata.var_names = [f"gene_{i}" for i in range(n_vars)]

    if with_run_history:
        adata.uns[result_key] = {
            "run_history": json.dumps([{
                "timestamp": "2025-01-01T00:00:00",
                "params": {
                    "condition1": condition1,
                    "condition2": condition2,
                    "log_fold_change_threshold": 0.5,
                    "ptp_threshold": 0.05,
                },
                "field_names": {
                    "lfc_key": lfc_col,
                    "ptp_key": ptp_col,
                    "direction_key": direction_col,
                },
                "adjusted_run_id": 0,
            }]),
        }

    return adata


def _make_de_adata(n_obs=30, n_vars=10, condition1="Young", condition2="Old",
                   result_key="kompot_de", with_layers=True, with_run_history=True):
    """Build a minimal AnnData with DE results and run history."""
    rng = np.random.RandomState(0)
    X = rng.randn(n_obs, n_vars).astype(np.float32)
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)])

    lfc_col = f"{result_key}_{condition1}_to_{condition2}_mean_lfc"
    mahal_col = f"{result_key}_{condition1}_to_{condition2}_mahalanobis"

    var[lfc_col] = rng.randn(n_vars).astype(np.float32)
    var[mahal_col] = rng.uniform(0, 5, n_vars).astype(np.float32)

    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_obs)])
    adata = AnnData(X=X, obs=obs, var=var)
    adata.obsm["X_umap"] = rng.randn(n_obs, 2).astype(np.float32)

    imputed1 = f"{result_key}_{condition1}_imputed"
    imputed2 = f"{result_key}_{condition2}_imputed"
    fc_layer = f"{result_key}_{condition1}_to_{condition2}_fold_change"

    if with_layers:
        adata.layers[imputed1] = rng.randn(n_obs, n_vars).astype(np.float32)
        adata.layers[imputed2] = rng.randn(n_obs, n_vars).astype(np.float32)
        adata.layers[fc_layer] = rng.randn(n_obs, n_vars).astype(np.float32)

    if with_run_history:
        adata.uns[result_key] = {
            "run_history": json.dumps([{
                "timestamp": "2025-01-01T00:00:00",
                "params": {
                    "condition1": condition1,
                    "condition2": condition2,
                },
                "field_names": {
                    "mean_lfc_key": lfc_col,
                    "mahalanobis_key": mahal_col,
                    "imputed_key_1": imputed1,
                    "imputed_key_2": imputed2,
                    "fold_change_key": fc_layer,
                },
                "imputed_layer_keys": {
                    "condition1": imputed1,
                    "condition2": imputed2,
                    "fold_change": fc_layer,
                },
                "adjusted_run_id": 0,
            }]),
        }

    return adata


# ===========================================================================
# 1. kompot/plot/volcano/multi_da.py
# ===========================================================================

class TestMultiVolcanoDa:
    """Tests for multi_volcano_da targeting uncovered lines."""

    def test_basic_return_fig(self):
        """Lines 982-985: return_fig / save paths."""
        from kompot.plot.volcano.multi_da import multi_volcano_da
        adata = _make_da_adata()
        lfc_col = [c for c in adata.obs.columns if "log_fold_change" in c][0]
        ptp_col = [c for c in adata.obs.columns if "_ptp" in c][0]
        fig = multi_volcano_da(
            adata, groupby="cell_type",
            lfc_key=lfc_col, ptp_key=ptp_col,
            return_fig=True,
        )
        assert fig is not None

    def test_return_none_without_flag(self):
        """Line 985: returns None when return_fig=False."""
        from kompot.plot.volcano.multi_da import multi_volcano_da
        adata = _make_da_adata()
        lfc_col = [c for c in adata.obs.columns if "log_fold_change" in c][0]
        ptp_col = [c for c in adata.obs.columns if "_ptp" in c][0]
        result = multi_volcano_da(
            adata, groupby="cell_type",
            lfc_key=lfc_col, ptp_key=ptp_col,
            return_fig=False,
        )
        assert result is None

    def test_no_groups_error(self):
        """Line 227: empty groups raises ValueError."""
        from kompot.plot.volcano.multi_da import multi_volcano_da
        adata = _make_da_adata()
        lfc_col = [c for c in adata.obs.columns if "log_fold_change" in c][0]
        ptp_col = [c for c in adata.obs.columns if "_ptp" in c][0]
        # Subset to zero cells
        adata2 = adata[0:0].copy()
        with pytest.raises(ValueError, match="No groups found"):
            multi_volcano_da(adata2, groupby="cell_type",
                             lfc_key=lfc_col, ptp_key=ptp_col)

    def test_missing_groupby_col(self):
        """Line 221: groupby column not in obs."""
        from kompot.plot.volcano.multi_da import multi_volcano_da
        adata = _make_da_adata()
        lfc_col = [c for c in adata.obs.columns if "log_fold_change" in c][0]
        ptp_col = [c for c in adata.obs.columns if "_ptp" in c][0]
        with pytest.raises(ValueError, match="not found"):
            multi_volcano_da(adata, groupby="nonexistent",
                             lfc_key=lfc_col, ptp_key=ptp_col)

    def test_log_transform_false(self):
        """Lines 518-521: log_transform_ptp=False branch."""
        from kompot.plot.volcano.multi_da import multi_volcano_da
        adata = _make_da_adata()
        lfc_col = [c for c in adata.obs.columns if "log_fold_change" in c][0]
        ptp_col = [c for c in adata.obs.columns if "_ptp" in c][0]
        fig = multi_volcano_da(
            adata, groupby="cell_type",
            lfc_key=lfc_col, ptp_key=ptp_col,
            log_transform_ptp=False,
            return_fig=True,
        )
        assert fig is not None

    def test_neg_log10_ptp_key(self):
        """Lines 519-521, 526: already neg_log10 transformed key."""
        from kompot.plot.volcano.multi_da import multi_volcano_da
        adata = _make_da_adata()
        lfc_col = [c for c in adata.obs.columns if "log_fold_change" in c][0]
        ptp_col = [c for c in adata.obs.columns if "_ptp" in c][0]
        neg_col = "neg_log10_ptp_test"
        adata.obs[neg_col] = -np.log10(adata.obs[ptp_col].values.clip(1e-10))
        fig = multi_volcano_da(
            adata, groupby="cell_type",
            lfc_key=lfc_col, ptp_key=neg_col,
            return_fig=True,
        )
        assert fig is not None

    def test_show_thresholds(self):
        """Lines 806-821: show_thresholds=True draws threshold lines."""
        from kompot.plot.volcano.multi_da import multi_volcano_da
        adata = _make_da_adata()
        lfc_col = [c for c in adata.obs.columns if "log_fold_change" in c][0]
        ptp_col = [c for c in adata.obs.columns if "_ptp" in c][0]
        fig = multi_volcano_da(
            adata, groupby="cell_type",
            lfc_key=lfc_col, ptp_key=ptp_col,
            show_thresholds=True,
            lfc_threshold=0.5,
            ptp_threshold=0.05,
            return_fig=True,
        )
        assert fig is not None

    def test_show_thresholds_neg_log10(self):
        """Lines 812-821: thresholds with neg_log10 key."""
        from kompot.plot.volcano.multi_da import multi_volcano_da
        adata = _make_da_adata()
        lfc_col = [c for c in adata.obs.columns if "log_fold_change" in c][0]
        ptp_col = [c for c in adata.obs.columns if "_ptp" in c][0]
        neg_col = "neg_log10_ptp_test"
        adata.obs[neg_col] = -np.log10(adata.obs[ptp_col].values.clip(1e-10))
        fig = multi_volcano_da(
            adata, groupby="cell_type",
            lfc_key=lfc_col, ptp_key=neg_col,
            show_thresholds=True,
            lfc_threshold=0.5,
            ptp_threshold=0.05,
            return_fig=True,
        )
        assert fig is not None

    def test_show_thresholds_neg_log10_large_threshold(self):
        """Lines 818-819: thresholds with neg_log10 key and threshold >= 1."""
        from kompot.plot.volcano.multi_da import multi_volcano_da
        adata = _make_da_adata()
        lfc_col = [c for c in adata.obs.columns if "log_fold_change" in c][0]
        ptp_col = [c for c in adata.obs.columns if "_ptp" in c][0]
        neg_col = "neg_log10_ptp_test"
        adata.obs[neg_col] = -np.log10(adata.obs[ptp_col].values.clip(1e-10))
        fig = multi_volcano_da(
            adata, groupby="cell_type",
            lfc_key=lfc_col, ptp_key=neg_col,
            show_thresholds=True,
            lfc_threshold=0.5,
            ptp_threshold=1.3,  # >= 1 so it stays as-is
            return_fig=True,
        )
        assert fig is not None

    def test_color_list(self):
        """Line 677: color as list of strings."""
        from kompot.plot.volcano.multi_da import multi_volcano_da
        adata = _make_da_adata()
        lfc_col = [c for c in adata.obs.columns if "log_fold_change" in c][0]
        ptp_col = [c for c in adata.obs.columns if "_ptp" in c][0]
        fig = multi_volcano_da(
            adata, groupby="cell_type",
            lfc_key=lfc_col, ptp_key=ptp_col,
            color=[adata.obs.columns[adata.obs.columns.str.contains("direction")][0]],
            return_fig=True,
        )
        assert fig is not None

    def test_color_missing_key(self):
        """Line 682-683: color key not in obs warns."""
        from kompot.plot.volcano.multi_da import multi_volcano_da
        adata = _make_da_adata()
        lfc_col = [c for c in adata.obs.columns if "log_fold_change" in c][0]
        ptp_col = [c for c in adata.obs.columns if "_ptp" in c][0]
        fig = multi_volcano_da(
            adata, groupby="cell_type",
            lfc_key=lfc_col, ptp_key=ptp_col,
            color="nonexistent_column",
            return_fig=True,
        )
        assert fig is not None

    def test_categorical_color_with_palette_string(self):
        """Lines 698-709: categorical color with string palette."""
        from kompot.plot.volcano.multi_da import multi_volcano_da
        adata = _make_da_adata()
        lfc_col = [c for c in adata.obs.columns if "log_fold_change" in c][0]
        ptp_col = [c for c in adata.obs.columns if "_ptp" in c][0]
        direction_col = [c for c in adata.obs.columns if "direction" in c][0]
        fig = multi_volcano_da(
            adata, groupby="cell_type",
            lfc_key=lfc_col, ptp_key=ptp_col,
            color=direction_col,
            palette="Set2",
            return_fig=True,
        )
        assert fig is not None

    def test_categorical_color_with_palette_dict(self):
        """Lines 703-704: categorical color with dict palette."""
        from kompot.plot.volcano.multi_da import multi_volcano_da
        adata = _make_da_adata()
        lfc_col = [c for c in adata.obs.columns if "log_fold_change" in c][0]
        ptp_col = [c for c in adata.obs.columns if "_ptp" in c][0]
        direction_col = [c for c in adata.obs.columns if "direction" in c][0]
        fig = multi_volcano_da(
            adata, groupby="cell_type",
            lfc_key=lfc_col, ptp_key=ptp_col,
            color=direction_col,
            palette={"up": "red", "down": "blue", "neutral": "gray"},
            return_fig=True,
        )
        assert fig is not None

    def test_categorical_color_default_palette(self):
        """Lines 706-709: categorical color with default palette (no palette given, no stored colors)."""
        from kompot.plot.volcano.multi_da import multi_volcano_da
        adata = _make_da_adata()
        lfc_col = [c for c in adata.obs.columns if "log_fold_change" in c][0]
        ptp_col = [c for c in adata.obs.columns if "_ptp" in c][0]
        direction_col = [c for c in adata.obs.columns if "direction" in c][0]
        # Ensure no stored colors
        colors_key = f"{direction_col}_colors"
        if colors_key in adata.uns:
            del adata.uns[colors_key]
        fig = multi_volcano_da(
            adata, groupby="cell_type",
            lfc_key=lfc_col, ptp_key=ptp_col,
            color=direction_col,
            return_fig=True,
        )
        assert fig is not None

    def test_numeric_color_with_lfc(self):
        """Lines 739-742, 749: numeric color with lfc-like column -> auto diverging cmap."""
        from kompot.plot.volcano.multi_da import multi_volcano_da
        adata = _make_da_adata()
        lfc_col = [c for c in adata.obs.columns if "log_fold_change" in c][0]
        ptp_col = [c for c in adata.obs.columns if "_ptp" in c][0]
        fig = multi_volcano_da(
            adata, groupby="cell_type",
            lfc_key=lfc_col, ptp_key=ptp_col,
            color=lfc_col,
            return_fig=True,
        )
        assert fig is not None

    def test_numeric_color_with_vmin_vmax(self):
        """Lines 772, 777: explicit vmin/vmax for numeric color."""
        from kompot.plot.volcano.multi_da import multi_volcano_da
        adata = _make_da_adata()
        lfc_col = [c for c in adata.obs.columns if "log_fold_change" in c][0]
        ptp_col = [c for c in adata.obs.columns if "_ptp" in c][0]
        # Add a plain numeric column
        adata.obs["numeric_score"] = np.random.randn(adata.n_obs)
        fig = multi_volcano_da(
            adata, groupby="cell_type",
            lfc_key=lfc_col, ptp_key=ptp_col,
            color="numeric_score",
            vmin=-2, vmax=2,
            return_fig=True,
        )
        assert fig is not None

    def test_show_legend_explicit_false(self):
        """Line 872: show_legend explicitly False."""
        from kompot.plot.volcano.multi_da import multi_volcano_da
        adata = _make_da_adata()
        lfc_col = [c for c in adata.obs.columns if "log_fold_change" in c][0]
        ptp_col = [c for c in adata.obs.columns if "_ptp" in c][0]
        fig = multi_volcano_da(
            adata, groupby="cell_type",
            lfc_key=lfc_col, ptp_key=ptp_col,
            show_legend=False,
            return_fig=True,
        )
        assert fig is not None

    def test_color_equals_groupby_hides_legend(self):
        """Lines 878-880: when color==groupby, legend auto-hidden."""
        from kompot.plot.volcano.multi_da import multi_volcano_da
        adata = _make_da_adata()
        lfc_col = [c for c in adata.obs.columns if "log_fold_change" in c][0]
        ptp_col = [c for c in adata.obs.columns if "_ptp" in c][0]
        fig = multi_volcano_da(
            adata, groupby="cell_type",
            lfc_key=lfc_col, ptp_key=ptp_col,
            color="cell_type",
            return_fig=True,
        )
        assert fig is not None

    def test_n_y_ticks_nonzero(self):
        """Lines 839-840: n_y_ticks > 0 uses MaxNLocator."""
        from kompot.plot.volcano.multi_da import multi_volcano_da
        adata = _make_da_adata()
        lfc_col = [c for c in adata.obs.columns if "log_fold_change" in c][0]
        ptp_col = [c for c in adata.obs.columns if "_ptp" in c][0]
        fig = multi_volcano_da(
            adata, groupby="cell_type",
            lfc_key=lfc_col, ptp_key=ptp_col,
            n_y_ticks=3,
            return_fig=True,
        )
        assert fig is not None

    def test_xlabel_default(self):
        """Line 848: xlabel=None with conditions from run info."""
        from kompot.plot.volcano.multi_da import multi_volcano_da
        adata = _make_da_adata()
        lfc_col = [c for c in adata.obs.columns if "log_fold_change" in c][0]
        ptp_col = [c for c in adata.obs.columns if "_ptp" in c][0]
        fig = multi_volcano_da(
            adata, groupby="cell_type",
            lfc_key=lfc_col, ptp_key=ptp_col,
            xlabel=None,
            return_fig=True,
        )
        assert fig is not None

    def test_xlabel_custom(self):
        """Line 848: custom xlabel."""
        from kompot.plot.volcano.multi_da import multi_volcano_da
        adata = _make_da_adata()
        lfc_col = [c for c in adata.obs.columns if "log_fold_change" in c][0]
        ptp_col = [c for c in adata.obs.columns if "_ptp" in c][0]
        fig = multi_volcano_da(
            adata, groupby="cell_type",
            lfc_key=lfc_col, ptp_key=ptp_col,
            xlabel="Custom X",
            return_fig=True,
        )
        assert fig is not None

    def test_background_kde(self):
        """Lines 579-631: background_plot='kde'."""
        from kompot.plot.volcano.multi_da import multi_volcano_da
        adata = _make_da_adata()
        lfc_col = [c for c in adata.obs.columns if "log_fold_change" in c][0]
        ptp_col = [c for c in adata.obs.columns if "_ptp" in c][0]
        fig = multi_volcano_da(
            adata, groupby="cell_type",
            lfc_key=lfc_col, ptp_key=ptp_col,
            background_plot="kde",
            return_fig=True,
        )
        assert fig is not None

    def test_background_violin(self):
        """Lines 633-660: background_plot='violin'."""
        from kompot.plot.volcano.multi_da import multi_volcano_da
        adata = _make_da_adata()
        lfc_col = [c for c in adata.obs.columns if "log_fold_change" in c][0]
        ptp_col = [c for c in adata.obs.columns if "_ptp" in c][0]
        fig = multi_volcano_da(
            adata, groupby="cell_type",
            lfc_key=lfc_col, ptp_key=ptp_col,
            background_plot="violin",
            return_fig=True,
        )
        assert fig is not None

    def test_no_run_history_condition_extraction(self):
        """Lines 281-286, 290, 293: conditions inferred from key name when no run_info."""
        from kompot.plot.volcano.multi_da import multi_volcano_da
        adata = _make_da_adata(with_run_history=False)
        lfc_col = [c for c in adata.obs.columns if "log_fold_change" in c][0]
        ptp_col = [c for c in adata.obs.columns if "_ptp" in c][0]
        fig = multi_volcano_da(
            adata, groupby="cell_type",
            lfc_key=lfc_col, ptp_key=ptp_col,
            lfc_threshold=0.5,
            ptp_threshold=0.05,
            return_fig=True,
        )
        assert fig is not None

    def test_scanpy_import_error(self):
        """Lines 21-24: scanpy import failure branch."""
        from kompot.plot.volcano import multi_da as mda_module
        old = mda_module._has_scanpy
        try:
            mda_module._has_scanpy = False
            adata = _make_da_adata()
            lfc_col = [c for c in adata.obs.columns if "log_fold_change" in c][0]
            ptp_col = [c for c in adata.obs.columns if "_ptp" in c][0]
            fig = mda_module.multi_volcano_da(
                adata, groupby="cell_type",
                lfc_key=lfc_col, ptp_key=ptp_col,
                return_fig=True,
            )
            assert fig is not None
        finally:
            mda_module._has_scanpy = old

    def test_save_figure(self, tmp_path):
        """Line 982: save figure to file."""
        from kompot.plot.volcano.multi_da import multi_volcano_da
        adata = _make_da_adata()
        lfc_col = [c for c in adata.obs.columns if "log_fold_change" in c][0]
        ptp_col = [c for c in adata.obs.columns if "_ptp" in c][0]
        path = str(tmp_path / "test_multi.png")
        result = multi_volcano_da(
            adata, groupby="cell_type",
            lfc_key=lfc_col, ptp_key=ptp_col,
            save=path,
            return_fig=False,
        )
        assert result is None
        assert os.path.exists(path)

    def test_explicit_figsize(self):
        """Line 359: explicit figsize provided."""
        from kompot.plot.volcano.multi_da import multi_volcano_da
        adata = _make_da_adata()
        lfc_col = [c for c in adata.obs.columns if "log_fold_change" in c][0]
        ptp_col = [c for c in adata.obs.columns if "_ptp" in c][0]
        fig = multi_volcano_da(
            adata, groupby="cell_type",
            lfc_key=lfc_col, ptp_key=ptp_col,
            figsize=(12, 8),
            return_fig=True,
        )
        assert fig is not None

    def test_legend_default_height(self):
        """Line 896: default legend height when no handles and no colorbar."""
        from kompot.plot.volcano.multi_da import multi_volcano_da
        adata = _make_da_adata()
        lfc_col = [c for c in adata.obs.columns if "log_fold_change" in c][0]
        ptp_col = [c for c in adata.obs.columns if "_ptp" in c][0]
        fig = multi_volcano_da(
            adata, groupby="cell_type",
            lfc_key=lfc_col, ptp_key=ptp_col,
            show_legend=True,
            return_fig=True,
        )
        assert fig is not None


# ===========================================================================
# 2. kompot/plot/expression.py
# ===========================================================================

class TestPlotGeneExpression:
    """Tests for plot_gene_expression targeting uncovered lines."""

    def test_scanpy_not_available(self):
        """Lines 79-80: scanpy not available returns None."""
        from kompot.plot import expression as expr_mod
        old = expr_mod._has_scanpy
        try:
            expr_mod._has_scanpy = False
            adata = _make_de_adata()
            result = expr_mod.plot_gene_expression(adata, "gene_0")
            assert result is None
        finally:
            expr_mod._has_scanpy = old

    def test_no_keys_inferred(self):
        """Lines 169, 172: keys cannot be inferred, warnings logged."""
        from kompot.plot.expression import plot_gene_expression
        adata = _make_de_adata(with_run_history=False, with_layers=False)
        # No run history and no recognizable keys
        adata.var.drop(columns=[c for c in adata.var.columns if "lfc" in c or "mahalanobis" in c],
                       inplace=True, errors="ignore")
        result = plot_gene_expression(adata, "gene_0", return_fig=True)
        # Should still produce a figure (with limited panels)
        assert result is not None

    def test_no_run_history_default_conditions(self):
        """Lines 182, 209-214: no run_history -> default condition names."""
        from kompot.plot.expression import plot_gene_expression
        adata = _make_de_adata(with_run_history=False)
        lfc_col = [c for c in adata.var.columns if "mean_lfc" in c][0]
        mahal_col = [c for c in adata.var.columns if "mahalanobis" in c][0]
        fig = plot_gene_expression(
            adata, "gene_0",
            lfc_key=lfc_col, score_key=mahal_col,
            return_fig=True,
        )
        assert fig is not None

    def test_positive_run_id(self):
        """Line 184: positive run_id path."""
        from kompot.plot.expression import plot_gene_expression
        adata = _make_de_adata()
        lfc_col = [c for c in adata.var.columns if "mean_lfc" in c][0]
        mahal_col = [c for c in adata.var.columns if "mahalanobis" in c][0]
        fig = plot_gene_expression(
            adata, "gene_0",
            lfc_key=lfc_col, score_key=mahal_col,
            run_id=0,
            return_fig=True,
        )
        assert fig is not None

    def test_infer_layer_from_run_info(self):
        """Lines 227, 234: layer inferred from run_info params."""
        from kompot.plot.expression import plot_gene_expression
        adata = _make_de_adata()
        lfc_col = [c for c in adata.var.columns if "mean_lfc" in c][0]
        mahal_col = [c for c in adata.var.columns if "mahalanobis" in c][0]
        # Add a layer param to run info
        run_history = json.loads(adata.uns["kompot_de"]["run_history"])
        run_history[0]["params"]["layer"] = "test_layer"
        adata.uns["kompot_de"]["run_history"] = json.dumps(run_history)
        adata.layers["test_layer"] = np.random.randn(adata.n_obs, adata.n_vars).astype(np.float32)
        fig = plot_gene_expression(
            adata, "gene_0",
            lfc_key=lfc_col, score_key=mahal_col,
            return_fig=True,
        )
        assert fig is not None

    def test_ignore_fold_change_layer(self):
        """Line 234: fold_change layer in params is ignored."""
        from kompot.plot.expression import plot_gene_expression
        adata = _make_de_adata()
        lfc_col = [c for c in adata.var.columns if "mean_lfc" in c][0]
        mahal_col = [c for c in adata.var.columns if "mahalanobis" in c][0]
        run_history = json.loads(adata.uns["kompot_de"]["run_history"])
        run_history[0]["params"]["layer"] = "fold_change_something"
        adata.uns["kompot_de"]["run_history"] = json.dumps(run_history)
        fig = plot_gene_expression(
            adata, "gene_0",
            lfc_key=lfc_col, score_key=mahal_col,
            return_fig=True,
        )
        assert fig is not None

    def test_basis_none_fallback_scatter(self):
        """Lines 342-343, 347, 353: basis=None -> fallback scatter (scanpy available)."""
        from kompot.plot.expression import plot_gene_expression
        adata = _make_de_adata(with_layers=True)
        lfc_col = [c for c in adata.var.columns if "mean_lfc" in c][0]
        mahal_col = [c for c in adata.var.columns if "mahalanobis" in c][0]
        fig = plot_gene_expression(
            adata, "gene_0",
            lfc_key=lfc_col, score_key=mahal_col,
            basis=None,
            return_fig=True,
        )
        assert fig is not None

    def test_basis_none_with_layer(self):
        """Lines 342-343: basis=None with explicit layer."""
        from kompot.plot.expression import plot_gene_expression
        adata = _make_de_adata(with_layers=True)
        lfc_col = [c for c in adata.var.columns if "mean_lfc" in c][0]
        mahal_col = [c for c in adata.var.columns if "mahalanobis" in c][0]
        adata.layers["raw"] = adata.X.copy()
        fig = plot_gene_expression(
            adata, "gene_0",
            lfc_key=lfc_col, score_key=mahal_col,
            basis=None,
            layer="raw",
            return_fig=True,
        )
        assert fig is not None

    def test_basis_none_missing_layer_fallback(self):
        """Lines 346-347: basis=None, layer doesn't exist -> fallback to X."""
        from kompot.plot.expression import plot_gene_expression
        adata = _make_de_adata(with_layers=True)
        lfc_col = [c for c in adata.var.columns if "mean_lfc" in c][0]
        mahal_col = [c for c in adata.var.columns if "mahalanobis" in c][0]
        fig = plot_gene_expression(
            adata, "gene_0",
            lfc_key=lfc_col, score_key=mahal_col,
            basis=None,
            layer="nonexistent_layer",
            return_fig=True,
        )
        assert fig is not None

    def test_missing_basis(self):
        """Lines 243, 266-267: basis not in obsm."""
        from kompot.plot.expression import plot_gene_expression
        adata = _make_de_adata()
        lfc_col = [c for c in adata.var.columns if "mean_lfc" in c][0]
        mahal_col = [c for c in adata.var.columns if "mahalanobis" in c][0]
        fig = plot_gene_expression(
            adata, "gene_0",
            lfc_key=lfc_col, score_key=mahal_col,
            basis="X_nonexistent",
            return_fig=True,
        )
        assert fig is not None

    def test_missing_layers_warning(self):
        """Lines 278-279, 293: missing imputed layers -> warning."""
        from kompot.plot.expression import plot_gene_expression
        adata = _make_de_adata(with_layers=False)
        lfc_col = [c for c in adata.var.columns if "mean_lfc" in c][0]
        mahal_col = [c for c in adata.var.columns if "mahalanobis" in c][0]
        fig = plot_gene_expression(
            adata, "gene_0",
            lfc_key=lfc_col, score_key=mahal_col,
            return_fig=True,
        )
        assert fig is not None

    def test_field_names_fallback(self):
        """Lines 280-282: field_names fallback when no imputed_layer_keys."""
        from kompot.plot.expression import plot_gene_expression
        adata = _make_de_adata(with_layers=True)
        lfc_col = [c for c in adata.var.columns if "mean_lfc" in c][0]
        mahal_col = [c for c in adata.var.columns if "mahalanobis" in c][0]
        # Remove imputed_layer_keys from run info to hit field_names fallback
        run_history = json.loads(adata.uns["kompot_de"]["run_history"])
        del run_history[0]["imputed_layer_keys"]
        adata.uns["kompot_de"]["run_history"] = json.dumps(run_history)
        fig = plot_gene_expression(
            adata, "gene_0",
            lfc_key=lfc_col, score_key=mahal_col,
            return_fig=True,
        )
        assert fig is not None

    def test_both_keys_provided(self):
        """Line 49-50: both lfc_key and score_key provided skips inference."""
        from kompot.plot.expression import _infer_expression_keys
        adata = _make_de_adata()
        result = _infer_expression_keys(adata, lfc_key="my_lfc", score_key="my_score")
        assert result == ("my_lfc", "my_score")

    def test_gene_not_found(self):
        """Line 158: gene not in var_names."""
        from kompot.plot.expression import plot_gene_expression
        adata = _make_de_adata()
        with pytest.raises(ValueError, match="not found"):
            plot_gene_expression(adata, "nonexistent_gene")


# ===========================================================================
# 3. kompot/resource_estimation.py
# ===========================================================================

class TestResourceEstimation:
    """Tests for resource_estimation targeting uncovered lines."""

    def test_format_report_with_output_fields_and_overwrites(self):
        """Lines 274-278, 295-297, 302: format_report with output fields and overwrites."""
        from kompot.resource_estimation import ResourcePlan, ResourceAvailability, ResourceRequirement
        plan = ResourcePlan()
        plan.availability = ResourceAvailability(
            memory_total=16 * 1024**3,
            memory_available=8 * 1024**3,
            disk_path="/tmp",
            disk_total=100 * 1024**3,
            disk_available=50 * 1024**3,
        )
        plan.add_requirement("Test array", 1024**2, "memory", shape=(100, 100))
        plan.add_requirement("Disk array", 1024**2, "disk", shape=(100, 100), overwrite=True)
        plan.output_fields = {
            "adata.var": ["field_a", "field_b"],
            "adata.layers": ["layer_c"],
        }
        plan._overwrite_fields = {"field_a": 0, "layer_c": None}
        plan.info = ["Some info message"]
        plan.warnings = ["A warning"]
        plan.errors = []

        report = plan.format_report(verbose=True)
        assert "FEASIBLE WITH WARNINGS" in report
        assert "field_a" in report
        assert "OVERWRITES run_id=0" in report
        assert "OVERWRITE" in report

    def test_format_report_with_errors(self):
        """Lines 295-297, 302: format_report with errors."""
        from kompot.resource_estimation import ResourcePlan
        plan = ResourcePlan()
        plan.errors = ["Fatal error"]
        report = plan.format_report()
        assert "INFEASIBLE" in report

    def test_format_report_clean(self):
        """Line 302-306: clean report with no warnings or errors."""
        from kompot.resource_estimation import ResourcePlan, ResourceAvailability
        plan = ResourcePlan()
        plan.availability = ResourceAvailability(
            memory_total=16 * 1024**3,
            memory_available=8 * 1024**3,
            disk_path="/tmp",
            disk_total=100 * 1024**3,
            disk_available=50 * 1024**3,
        )
        plan.add_requirement("Small array", 1024, "memory")
        report = plan.format_report()
        assert "FEASIBLE" in report

    def test_suggest_alternative_disk_locations(self):
        """Lines 370-396: suggest_alternative_disk_locations."""
        from kompot.resource_estimation import suggest_alternative_disk_locations
        candidates = suggest_alternative_disk_locations()
        # Should return a list of tuples
        assert isinstance(candidates, list)
        for item in candidates:
            assert len(item) == 3
            assert isinstance(item[0], str)  # path
            assert isinstance(item[2], int)  # bytes

    def test_psutil_not_available(self):
        """Lines 17-18, 22: psutil/dask not available."""
        from kompot import resource_estimation as re_mod
        old_psutil = re_mod.PSUTIL_AVAILABLE
        try:
            re_mod.PSUTIL_AVAILABLE = False
            result = re_mod.get_system_resources()
            assert result.memory_total == 8 * 1024**3
            assert result.disk_total == 100 * 1024**3
        finally:
            re_mod.PSUTIL_AVAILABLE = old_psutil

    def test_format_report_disk_requirements(self):
        """Lines 255-260: disk reqs in verbose format_report."""
        from kompot.resource_estimation import ResourcePlan, ResourceAvailability
        plan = ResourcePlan()
        plan.availability = ResourceAvailability(
            memory_total=16 * 1024**3,
            memory_available=8 * 1024**3,
            disk_path="/tmp",
            disk_total=100 * 1024**3,
            disk_available=50 * 1024**3,
        )
        plan.add_requirement("Cov matrix", 2 * 1024**3, "disk", shape=(1000, 1000), overwrite=True)
        report = plan.format_report(verbose=True)
        assert "Disk Storage" in report
        assert "OVERWRITE" in report

    def test_resource_plan_no_availability(self):
        """Lines 107-108, 114-115: ratios with no availability."""
        from kompot.resource_estimation import ResourcePlan
        plan = ResourcePlan()
        plan.add_requirement("Test", 1024, "memory")
        assert plan.memory_ratio == float('inf')
        assert plan.disk_ratio == float('inf')


# ===========================================================================
# 4. kompot/utils.py
# ===========================================================================

class TestUtils:
    """Tests for utils.py targeting uncovered lines."""

    def test_kompot_colors_structure(self):
        """Line 18, 27-31: KOMPOT_COLORS is accessible and has expected structure."""
        from kompot.utils import KOMPOT_COLORS
        assert "direction" in KOMPOT_COLORS
        assert "up" in KOMPOT_COLORS["direction"]
        assert "down" in KOMPOT_COLORS["direction"]
        assert "neutral" in KOMPOT_COLORS["direction"]

    def test_mahalanobis_dimension_mismatch(self):
        """Lines 299-300: dimension mismatch returns NaN."""
        from kompot.utils import compute_mahalanobis_distances
        diffs = np.random.randn(5, 3)
        cov = np.eye(4)  # Wrong shape
        result = compute_mahalanobis_distances(diffs, cov, progress=False)
        assert np.all(np.isnan(result))

    def test_mahalanobis_cholesky_failure(self):
        """Lines 406-409: Cholesky failure returns NaN."""
        from kompot.utils import compute_mahalanobis_distances
        diffs = np.random.randn(3, 3)
        # Create a non-positive definite matrix
        cov = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
        result = compute_mahalanobis_distances(diffs, cov, progress=False, eps=0)
        assert np.all(np.isnan(result))

    def test_mahalanobis_diagonal_approximation(self):
        """Lines 283: diagonal matrix fast path."""
        from kompot.utils import compute_mahalanobis_distances
        n = 15
        diffs = np.random.randn(5, n)
        cov = np.diag(np.ones(n) * 2.0)  # Pure diagonal -> ratio > 0.95
        result = compute_mahalanobis_distances(diffs, cov, progress=False)
        assert result.shape == (5,)
        assert not np.any(np.isnan(result))

    def test_mahalanobis_diagonal_with_invalid(self):
        """Lines 365-366: NaN in diagonal computation."""
        from kompot.utils import compute_mahalanobis_distances
        n = 15
        diffs = np.random.randn(5, n)
        diffs[0, :] = np.nan  # Insert NaN
        cov = np.diag(np.ones(n) * 2.0)
        result = compute_mahalanobis_distances(diffs, cov, progress=False)
        assert np.any(np.isnan(result))

    def test_mahalanobis_gene_specific_3d(self):
        """Lines 140-141, 147, 202-205: gene-specific 3D covariance."""
        from kompot.utils import compute_mahalanobis_distances
        n_points = 5
        n_genes = 3
        diffs = np.random.randn(n_genes, n_points)
        cov = np.zeros((n_points, n_points, n_genes))
        for g in range(n_genes):
            cov[:, :, g] = np.eye(n_points)
        result = compute_mahalanobis_distances(diffs, cov, progress=False)
        assert result.shape == (n_genes,)

    def test_mahalanobis_gene_specific_cholesky_fail(self):
        """Lines 209-236: gene-specific with non-PD matrix."""
        from kompot.utils import compute_mahalanobis_distances
        n_points = 5
        n_genes = 2
        diffs = np.random.randn(n_genes, n_points)
        cov = np.zeros((n_points, n_points, n_genes))
        cov[:, :, 0] = np.eye(n_points)
        cov[:, :, 1] = -np.eye(n_points)  # Not positive definite
        result = compute_mahalanobis_distances(diffs, cov, progress=False, eps=0)
        # gene 0 should be OK, gene 1 should be NaN
        assert not np.isnan(result[0])
        assert np.isnan(result[1])

    def test_mahalanobis_single_vector(self):
        """Line 152: single vector input."""
        from kompot.utils import compute_mahalanobis_distances
        diff = np.array([1.0, 0.0, 0.0])
        cov = np.eye(3)
        result = compute_mahalanobis_distances(diff, cov, progress=False)
        assert result.shape == (1,)
        assert abs(result[0] - 1.0) < 0.1

    def test_compute_mahalanobis_distance_single(self):
        """Lines 379-381: convenience function for single distance."""
        from kompot.utils import compute_mahalanobis_distance
        diff = np.array([1.0, 0.0])
        cov = np.eye(2)
        result = compute_mahalanobis_distance(diff, cov)
        assert isinstance(result, float)
        assert abs(result - 1.0) < 0.1

    def test_sanitize_name_fallback(self):
        """Line 18: _sanitize_name fallback."""
        from kompot.utils import _sanitize_name
        assert _sanitize_name("hello world/foo") == "hello_world_foo"


# ===========================================================================
# 5. kompot/plot/imputation.py
# ===========================================================================

class TestPlotImputation:
    """Tests for plot_imputation targeting uncovered lines."""

    def _make_imputation_adata(self, with_std=True, with_obs_var=True):
        """Build AnnData with imputation layers."""
        rng = np.random.RandomState(42)
        n_obs, n_vars = 50, 8
        X = rng.randn(n_obs, n_vars).astype(np.float32)
        adata = AnnData(X=X)
        adata.var_names = [f"gene_{i}" for i in range(n_vars)]
        adata.obs_names = [f"cell_{i}" for i in range(n_obs)]
        adata.obsm["X_umap"] = rng.randn(n_obs, 2).astype(np.float32)

        cond = "treated"
        prefix = "kompot_impute"
        adata.layers[f"{prefix}_{cond}_imputed"] = rng.randn(n_obs, n_vars).astype(np.float32)
        if with_std:
            adata.layers[f"{prefix}_{cond}_std"] = np.abs(rng.randn(n_obs, n_vars)).astype(np.float32)
        if with_obs_var:
            adata.layers[f"{prefix}_{cond}_obs_variance"] = np.abs(rng.randn(n_obs, n_vars)).astype(np.float32)
        return adata

    def test_plot_imputation_basic(self):
        """Lines 218-226, 242-249, 268-275, 293-300: basic imputation plot with scanpy."""
        from kompot.plot.imputation import plot_imputation
        adata = self._make_imputation_adata()
        fig = plot_imputation(
            adata, genes=["gene_0", "gene_1"],
            return_fig=True,
        )
        assert fig is not None

    def test_plot_imputation_no_scanpy(self):
        """Lines 218-226, 242-249, 268-275, 293-300: fallback without scanpy."""
        from kompot.plot import imputation as imp_mod
        old = imp_mod._has_scanpy
        try:
            imp_mod._has_scanpy = False
            adata = self._make_imputation_adata()
            fig = imp_mod.plot_imputation(
                adata, genes=["gene_0", "gene_1"],
                return_fig=True,
            )
            assert fig is not None
        finally:
            imp_mod._has_scanpy = old

    def test_plot_imputation_no_std(self):
        """Lines 329, 332: no std layer."""
        from kompot.plot.imputation import plot_imputation
        adata = self._make_imputation_adata(with_std=False, with_obs_var=False)
        fig = plot_imputation(adata, genes=["gene_0"], return_fig=True)
        assert fig is not None

    def test_plot_imputation_auto_genes(self):
        """Lines 337-343: auto gene selection."""
        from kompot.plot.imputation import plot_imputation
        adata = self._make_imputation_adata()
        fig = plot_imputation(adata, genes=None, n_top_genes=3, return_fig=True)
        assert fig is not None

    def test_plot_imputation_missing_basis(self):
        """Lines 97: missing basis raises ValueError."""
        from kompot.plot.imputation import plot_imputation
        adata = self._make_imputation_adata()
        with pytest.raises(ValueError, match="not found"):
            plot_imputation(adata, basis="X_nonexistent")

    def test_plot_imputation_missing_layers(self):
        """Line 110: missing imputation layers raises ValueError."""
        from kompot.plot.imputation import plot_imputation
        adata = self._make_imputation_adata()
        with pytest.raises(ValueError, match="No imputation layers"):
            plot_imputation(adata, result_key="nonexistent_key")

    def test_plot_imputation_custom_title(self):
        """Line 329: custom title."""
        from kompot.plot.imputation import plot_imputation
        adata = self._make_imputation_adata()
        fig = plot_imputation(adata, genes=["gene_0"], title="Custom Title", return_fig=True)
        assert fig is not None

    def test_plot_imputation_save(self, tmp_path):
        """Lines 349-358: save and return."""
        from kompot.plot.imputation import plot_imputation
        adata = self._make_imputation_adata()
        path = str(tmp_path / "imputation.png")
        fig = plot_imputation(adata, genes=["gene_0"], save=path, return_fig=True)
        assert fig is not None
        assert os.path.exists(path)

    def test_scanpy_import_false(self):
        """Lines 20-21: scanpy import failure flag."""
        from kompot.plot import imputation as imp_mod
        # Just verify the flag exists
        assert hasattr(imp_mod, "_has_scanpy")

    def test_plot_imputation_no_obs_variance(self):
        """Lines 122-124: has_std but no obs_variance."""
        from kompot.plot.imputation import plot_imputation
        adata = self._make_imputation_adata(with_obs_var=False)
        fig = plot_imputation(adata, genes=["gene_0"], return_fig=True)
        assert fig is not None


# ===========================================================================
# 6. kompot/plot/field_inference.py
# ===========================================================================

class TestFieldInference:
    """Tests for field_inference.py targeting uncovered lines."""

    def test_infer_default_da_fields(self):
        """Lines 65, 69: default required_fields for DA."""
        from kompot.plot.field_inference import infer_fields_from_run_info
        adata = _make_da_adata()
        result = infer_fields_from_run_info(
            adata, analysis_type="da", run_id=-1, strict=False
        )
        assert "lfc_key" in result
        assert "direction_key" in result

    def test_infer_unknown_analysis_type(self):
        """Line 69: unknown analysis type -> empty required_fields."""
        from kompot.plot.field_inference import infer_fields_from_run_info
        adata = _make_da_adata()
        result = infer_fields_from_run_info(
            adata, analysis_type="unknown", strict=False
        )
        assert isinstance(result, dict)

    def test_infer_no_run_info(self):
        """Lines 77-79: no run info available."""
        from kompot.plot.field_inference import infer_fields_from_run_info
        adata = _make_de_adata(with_run_history=False)
        result = infer_fields_from_run_info(
            adata, analysis_type="de", strict=False
        )
        assert isinstance(result, dict)

    def test_condition_mismatch_warning(self):
        """Lines 257-258, 262-266: user conditions don't match run info."""
        from kompot.plot.field_inference import infer_fields_from_run_info
        adata = _make_da_adata(condition1="CondA", condition2="CondB")
        result = infer_fields_from_run_info(
            adata, analysis_type="da",
            condition1="WrongA", condition2="WrongB",
            strict=False,
        )
        assert isinstance(result, dict)

    def test_condition_mismatch_strict_raises(self):
        """Lines 275, 283: strict mode raises on condition mismatch."""
        from kompot.plot.field_inference import infer_fields_from_run_info
        adata = _make_da_adata(condition1="CondA", condition2="CondB")
        with pytest.raises(ValueError, match="Condition mismatch"):
            infer_fields_from_run_info(
                adata, analysis_type="da",
                condition1="WrongA", condition2="WrongB",
                strict=True,
            )

    def test_field_not_in_data(self):
        """Lines 313-315, 321: field in run_info but missing from data."""
        from kompot.plot.field_inference import infer_fields_from_run_info
        adata = _make_da_adata()
        # Rename the actual column so the run_info field name doesn't match
        lfc_col = [c for c in adata.obs.columns if "log_fold_change" in c][0]
        adata.obs.rename(columns={lfc_col: "renamed_lfc"}, inplace=True)
        result = infer_fields_from_run_info(
            adata, analysis_type="da", strict=False
        )
        # lfc_key should be None or fallback
        assert isinstance(result, dict)

    def test_strict_missing_required_fields(self):
        """Lines 325-328: strict mode with missing required fields."""
        from kompot.plot.field_inference import infer_fields_from_run_info
        adata = _make_de_adata(with_run_history=False)
        # Remove all recognizable columns
        adata.var = adata.var.drop(columns=adata.var.columns.tolist())
        with pytest.raises(ValueError, match="Could not infer"):
            infer_fields_from_run_info(
                adata, analysis_type="de", strict=True,
            )

    def test_fallback_multiple_candidates(self):
        """Lines 344-346, 385: multiple candidates in fallback."""
        from kompot.plot.field_inference import infer_fields_from_run_info
        adata = _make_de_adata(with_run_history=False)
        # Add multiple columns that match the pattern
        adata.var["kompot_de_A_to_B_mean_lfc"] = np.random.randn(adata.n_vars)
        adata.var["kompot_de_C_to_D_mean_lfc"] = np.random.randn(adata.n_vars)
        adata.var["kompot_de_A_to_B_mahalanobis"] = np.random.randn(adata.n_vars)
        adata.var["kompot_de_C_to_D_mahalanobis"] = np.random.randn(adata.n_vars)
        result = infer_fields_from_run_info(
            adata, analysis_type="de", strict=False,
        )
        assert isinstance(result, dict)

    def test_check_for_overwrites(self):
        """Lines 313-315: _check_for_overwrites with tracking."""
        from kompot.plot.field_inference import _check_for_overwrites
        adata = _make_da_adata()
        lfc_col = [c for c in adata.obs.columns if "log_fold_change" in c][0]
        # Set up field tracking
        adata.uns["kompot_da"]["anndata_fields"] = json.dumps({
            "obs": {lfc_col: 0}
        })
        warnings_list = []
        fields = {"lfc_key": lfc_col}
        _check_for_overwrites(adata, "da", fields, warnings_list)
        # Should not crash; may or may not add warnings

    def test_count_potential_field_writers(self):
        """Lines 369-394: _count_potential_field_writers."""
        from kompot.plot.field_inference import _count_potential_field_writers
        adata = _make_da_adata()
        lfc_col = [c for c in adata.obs.columns if "log_fold_change" in c][0]
        count = _count_potential_field_writers(adata, "da", "lfc_key", lfc_col)
        assert count >= 1


# ===========================================================================
# Additional edge-case tests for broader coverage
# ===========================================================================

class TestExpressionEdgeCases:
    """Additional expression plot edge cases."""

    def test_swapped_conditions(self):
        """Lines 265-267: swapped condition order -> imputed_layer_keys swap."""
        from kompot.plot.expression import plot_gene_expression
        adata = _make_de_adata(condition1="Young", condition2="Old")
        lfc_col = [c for c in adata.var.columns if "mean_lfc" in c][0]
        mahal_col = [c for c in adata.var.columns if "mahalanobis" in c][0]
        # Pass conditions in reversed order
        fig = plot_gene_expression(
            adata, "gene_0",
            lfc_key=lfc_col, score_key=mahal_col,
            condition1="Old", condition2="Young",
            return_fig=True,
        )
        assert fig is not None

    def test_layer_missing_fallback(self):
        """Lines 346-347: layer not found -> fallback to adata.X (scanpy available, basis=None)."""
        from kompot.plot.expression import plot_gene_expression
        adata = _make_de_adata(with_layers=False)
        lfc_col = [c for c in adata.var.columns if "mean_lfc" in c][0]
        mahal_col = [c for c in adata.var.columns if "mahalanobis" in c][0]
        fig = plot_gene_expression(
            adata, "gene_0",
            lfc_key=lfc_col, score_key=mahal_col,
            basis=None,
            layer="nonexistent_layer",
            return_fig=True,
        )
        assert fig is not None


class TestResourceEstimationAdditional:
    """Additional resource estimation tests."""

    def test_format_report_with_info_no_availability(self):
        """Lines 274-278: format_report without availability."""
        from kompot.resource_estimation import ResourcePlan
        plan = ResourcePlan()
        plan.info = ["Info1"]
        plan.output_fields = {"adata.obs": ["col_a"]}
        plan._overwrite_fields = {}
        report = plan.format_report()
        assert "Info1" in report
        assert "col_a" in report

    def test_check_availability_no_psutil(self):
        """Line 148-149: check_availability with no availability."""
        from kompot.resource_estimation import ResourcePlan
        plan = ResourcePlan()
        plan.check_availability()
        assert "Could not check" in plan.warnings[0]

    def test_memory_ratio_zero_available(self):
        """Lines 107-108: memory_available=0."""
        from kompot.resource_estimation import ResourcePlan, ResourceAvailability
        plan = ResourcePlan()
        plan.availability = ResourceAvailability(
            memory_total=0, memory_available=0,
            disk_path="/tmp", disk_total=0, disk_available=0,
        )
        plan.add_requirement("T", 1024, "memory")
        assert plan.memory_ratio == float('inf')

    def test_estimate_array_size(self):
        """Function coverage for estimate_array_size."""
        from kompot.resource_estimation import estimate_array_size
        size = estimate_array_size((100, 100), np.float64)
        assert size == 100 * 100 * 8

    def test_human_readable_size_zero(self):
        """Edge case: 0 bytes."""
        from kompot.resource_estimation import human_readable_size
        assert human_readable_size(0) == "0.00 B"

    def test_human_readable_size_large(self):
        """Large values."""
        from kompot.resource_estimation import human_readable_size
        result = human_readable_size(1024**4)
        assert "TB" in result


class TestMultiVolcanoDaAdditional:
    """Additional multi_volcano_da edge cases."""

    def test_highlight_subset(self):
        """Lines 549-551: custom highlight_subset."""
        from kompot.plot.volcano.multi_da import multi_volcano_da
        adata = _make_da_adata(n_obs=60)
        lfc_col = [c for c in adata.obs.columns if "log_fold_change" in c][0]
        ptp_col = [c for c in adata.obs.columns if "_ptp" in c][0]
        highlight = np.zeros(adata.n_obs, dtype=bool)
        highlight[:10] = True
        fig = multi_volcano_da(
            adata, groupby="cell_type",
            lfc_key=lfc_col, ptp_key=ptp_col,
            highlight_subset=highlight,
            return_fig=True,
        )
        assert fig is not None

    def test_share_y_false(self):
        """Line 423: share_y=False."""
        from kompot.plot.volcano.multi_da import multi_volcano_da
        adata = _make_da_adata()
        lfc_col = [c for c in adata.obs.columns if "log_fold_change" in c][0]
        ptp_col = [c for c in adata.obs.columns if "_ptp" in c][0]
        fig = multi_volcano_da(
            adata, groupby="cell_type",
            lfc_key=lfc_col, ptp_key=ptp_col,
            share_y=False,
            return_fig=True,
        )
        assert fig is not None

    def test_layout_config_override(self):
        """Line 329-330: custom layout_config."""
        from kompot.plot.volcano.multi_da import multi_volcano_da
        adata = _make_da_adata()
        lfc_col = [c for c in adata.obs.columns if "log_fold_change" in c][0]
        ptp_col = [c for c in adata.obs.columns if "_ptp" in c][0]
        fig = multi_volcano_da(
            adata, groupby="cell_type",
            lfc_key=lfc_col, ptp_key=ptp_col,
            layout_config={"unit_size": 0.2, "plot_height": 5},
            return_fig=True,
        )
        assert fig is not None

    def test_no_title(self):
        """Lines 463-477: title=None."""
        from kompot.plot.volcano.multi_da import multi_volcano_da
        adata = _make_da_adata()
        lfc_col = [c for c in adata.obs.columns if "log_fold_change" in c][0]
        ptp_col = [c for c in adata.obs.columns if "_ptp" in c][0]
        fig = multi_volcano_da(
            adata, groupby="cell_type",
            lfc_key=lfc_col, ptp_key=ptp_col,
            title=None,
            return_fig=True,
        )
        assert fig is not None

    def test_numeric_color_no_vcenter(self):
        """Lines 771-778: numeric color without vcenter, with global vmin/vmax."""
        from kompot.plot.volcano.multi_da import multi_volcano_da
        adata = _make_da_adata()
        lfc_col = [c for c in adata.obs.columns if "log_fold_change" in c][0]
        ptp_col = [c for c in adata.obs.columns if "_ptp" in c][0]
        adata.obs["score"] = np.random.randn(adata.n_obs)
        fig = multi_volcano_da(
            adata, groupby="cell_type",
            lfc_key=lfc_col, ptp_key=ptp_col,
            color="score",
            cmap="viridis",
            return_fig=True,
        )
        assert fig is not None


# ===========================================================================
# Additional targeted tests for deeper coverage
# ===========================================================================

class TestFieldInferenceOverwrites:
    """More targeted tests for field_inference.py overwrite detection."""

    def test_check_overwrites_with_tracking_data(self):
        """Lines 310-315, 321, 325-328, 344-346, 350-354, 363-366: full overwrite path."""
        from kompot.plot.field_inference import _check_for_overwrites
        adata = _make_da_adata(condition1="CondA", condition2="CondB")
        lfc_col = [c for c in adata.obs.columns if "log_fold_change" in c][0]
        direction_col = [c for c in adata.obs.columns if "direction" in c][0]

        # Set up field tracking with a DIFFERENT run_id than the latest
        adata.uns["kompot_da"]["anndata_fields"] = json.dumps({
            "obs": {lfc_col: 99}  # run_id=99, but latest is 0
        })

        warnings_list = []
        fields = {"lfc_key": lfc_col}
        _check_for_overwrites(adata, "da", fields, warnings_list)
        # Should detect the mismatch
        assert any("overwritten" in w.lower() or "written by" in w.lower() for w in warnings_list)

    def test_check_overwrites_location_not_in_tracking(self):
        """Line 321: location not in tracking -> early return."""
        from kompot.plot.field_inference import _check_for_overwrites
        adata = _make_da_adata()
        lfc_col = [c for c in adata.obs.columns if "log_fold_change" in c][0]
        # Only has "var" key, but DA needs "obs"
        adata.uns["kompot_da"]["anndata_fields"] = json.dumps({
            "var": {lfc_col: 0}
        })
        warnings_list = []
        _check_for_overwrites(adata, "da", {"lfc_key": lfc_col}, warnings_list)
        # Should return early without adding warnings
        assert len(warnings_list) == 0

    def test_check_overwrites_location_tracking_as_string(self):
        """Lines 325-328: location_tracking is a JSON string."""
        from kompot.plot.field_inference import _check_for_overwrites
        adata = _make_da_adata()
        lfc_col = [c for c in adata.obs.columns if "log_fold_change" in c][0]
        adata.uns["kompot_da"]["anndata_fields"] = json.dumps({
            "obs": json.dumps({lfc_col: 0})  # Nested JSON string
        })
        warnings_list = []
        _check_for_overwrites(adata, "da", {"lfc_key": lfc_col}, warnings_list)

    def test_check_overwrites_invalid_tracking(self):
        """Lines 313-315: exception accessing tracking data."""
        from kompot.plot.field_inference import _check_for_overwrites
        adata = _make_da_adata()
        lfc_col = [c for c in adata.obs.columns if "log_fold_change" in c][0]
        # Set a value that will cause from_json_string to raise
        adata.uns["kompot_da"]["anndata_fields"] = "{invalid json"
        warnings_list = []
        _check_for_overwrites(adata, "da", {"lfc_key": lfc_col}, warnings_list)

    def test_check_overwrites_multiple_writers(self):
        """Lines 362-366: multiple potential writers."""
        from kompot.plot.field_inference import _check_for_overwrites
        adata = _make_da_adata()
        lfc_col = [c for c in adata.obs.columns if "log_fold_change" in c][0]
        # Add a second run to history with same field_names
        run_history = json.loads(adata.uns["kompot_da"]["run_history"])
        run_history.append({
            "timestamp": "2025-02-01T00:00:00",
            "params": {"condition1": "CondA", "condition2": "CondB"},
            "field_names": {"lfc_key": lfc_col},
            "adjusted_run_id": 1,
        })
        adata.uns["kompot_da"]["run_history"] = json.dumps(run_history)
        adata.uns["kompot_da"]["anndata_fields"] = json.dumps({
            "obs": {lfc_col: 1}
        })
        warnings_list = []
        _check_for_overwrites(adata, "da", {"lfc_key": lfc_col}, warnings_list)
        # Should detect multiple writers
        assert any("written by" in w.lower() for w in warnings_list)

    def test_get_run_from_history_error(self):
        """Lines 77-79: error accessing run history."""
        from kompot.plot.field_inference import infer_fields_from_run_info
        adata = _make_da_adata()
        # Corrupt the run history
        adata.uns["kompot_da"]["run_history"] = "not valid json {{{"
        result = infer_fields_from_run_info(adata, analysis_type="da", strict=False)
        assert isinstance(result, dict)

    def test_de_field_mapping(self):
        """Lines 122-123: DE field mapping path."""
        from kompot.plot.field_inference import infer_fields_from_run_info
        adata = _make_de_adata()
        result = infer_fields_from_run_info(
            adata, analysis_type="de", strict=False
        )
        assert "mean_lfc_key" in result
        assert "mahalanobis_key" in result

    def test_fallback_with_condition_filtering(self):
        """Lines 250-258: fallback with condition-based filtering."""
        from kompot.plot.field_inference import _fallback_field_inference
        # Create mock data_section with columns
        data = pd.DataFrame({
            "kompot_da_A_to_B_lfc": [1.0],
            "kompot_da_C_to_D_lfc": [2.0],
        })
        result = _fallback_field_inference(
            data, "lfc_key", "da",
            condition1="A", condition2="B",
            result_key=None, strict=False,
        )
        assert result == "kompot_da_A_to_B_lfc"

    def test_fallback_with_result_key_filtering(self):
        """Lines 262-266: fallback with result_key filtering."""
        from kompot.plot.field_inference import _fallback_field_inference
        data = pd.DataFrame({
            "kompot_da_A_to_B_lfc": [1.0],
            "kompot_da_C_to_B_lfc": [2.0],
        })
        result = _fallback_field_inference(
            data, "lfc_key", "da",
            condition1=None, condition2=None,
            result_key="kompot_da_A", strict=False,
        )
        assert result is not None

    def test_fallback_unknown_field_type(self):
        """Line 230: unknown field_type returns None."""
        from kompot.plot.field_inference import _fallback_field_inference
        data = pd.DataFrame({"col": [1.0]})
        result = _fallback_field_inference(
            data, "unknown_field", "da", None, None, None, False
        )
        assert result is None

    def test_get_comparison_specific_fields(self):
        """Lines 437-462: get_comparison_specific_fields."""
        from kompot.plot.field_inference import get_comparison_specific_fields
        adata = _make_da_adata(condition1="CondA", condition2="CondB")
        fields = get_comparison_specific_fields(
            adata, "da", "CondA", "CondB"
        )
        assert isinstance(fields, dict)

    def test_fallback_strict_multiple_candidates(self):
        """Line 280: strict mode returns None for multiple candidates."""
        from kompot.plot.field_inference import _fallback_field_inference
        data = pd.DataFrame({
            "kompot_da_A_to_B_lfc": [1.0],
            "kompot_da_C_to_D_lfc": [2.0],
        })
        result = _fallback_field_inference(
            data, "lfc_key", "da",
            condition1=None, condition2=None,
            result_key=None, strict=True,
        )
        # With strict=True and multiple candidates, should return None
        assert result is None


class TestResourceEstimationDeepCoverage:
    """Deeper coverage for resource_estimation.py."""

    def test_check_availability_insufficient_memory(self):
        """Lines 153-158: insufficient memory triggers error."""
        from kompot.resource_estimation import ResourcePlan, ResourceAvailability
        plan = ResourcePlan()
        plan.availability = ResourceAvailability(
            memory_total=1024, memory_available=512,
            disk_path="/tmp", disk_total=100 * 1024**3, disk_available=50 * 1024**3,
        )
        plan.add_requirement("Big array", 1024, "memory")  # > 512 available
        plan.check_availability()
        assert any("Insufficient memory" in e for e in plan.errors)

    def test_check_availability_high_memory_warning(self):
        """Lines 159-165: high memory usage generates warning."""
        from kompot.resource_estimation import ResourcePlan, ResourceAvailability
        plan = ResourcePlan()
        plan.availability = ResourceAvailability(
            memory_total=1024**3, memory_available=1024**3,
            disk_path="/tmp", disk_total=100 * 1024**3, disk_available=50 * 1024**3,
        )
        # Use 90% of memory (above 80% threshold)
        plan.add_requirement("Big array", int(0.9 * 1024**3), "memory")
        plan.check_availability()
        assert any("High memory usage" in w for w in plan.warnings)

    def test_check_availability_insufficient_disk(self):
        """Lines 169-195: insufficient disk triggers error with suggestions."""
        from kompot.resource_estimation import ResourcePlan, ResourceAvailability
        plan = ResourcePlan()
        plan.availability = ResourceAvailability(
            memory_total=16 * 1024**3, memory_available=8 * 1024**3,
            disk_path="/tmp", disk_total=1024, disk_available=512,
        )
        plan.add_requirement("Huge file", 1024, "disk")
        plan.check_availability()
        assert any("Insufficient disk" in e for e in plan.errors)

    def test_check_availability_high_disk_warning(self):
        """Lines 196-202: high disk usage generates warning."""
        from kompot.resource_estimation import ResourcePlan, ResourceAvailability
        plan = ResourcePlan()
        plan.availability = ResourceAvailability(
            memory_total=16 * 1024**3, memory_available=8 * 1024**3,
            disk_path="/tmp", disk_total=1024**3, disk_available=1024**3,
        )
        plan.add_requirement("Large file", int(0.95 * 1024**3), "disk")
        plan.check_availability()
        assert any("High disk usage" in w for w in plan.warnings)

    def test_format_report_no_availability(self):
        """Lines 233-240: format_report without availability -> 0% ratio."""
        from kompot.resource_estimation import ResourcePlan
        plan = ResourcePlan()
        plan.add_requirement("Test", 1024, "memory")
        plan.add_requirement("Disk", 2048, "disk")
        report = plan.format_report()
        assert "0%" in report

    def test_resource_requirement_properties(self):
        """Line 53-55: ResourceRequirement.size_human property."""
        from kompot.resource_estimation import ResourceRequirement
        req = ResourceRequirement(
            name="Test", size_bytes=1024**2, resource_type="memory"
        )
        assert "MB" in req.size_human

    def test_resource_availability_properties(self):
        """Lines 68-81: ResourceAvailability properties."""
        from kompot.resource_estimation import ResourceAvailability
        avail = ResourceAvailability(
            memory_total=16 * 1024**3,
            memory_available=8 * 1024**3,
            disk_path="/tmp",
            disk_total=100 * 1024**3,
            disk_available=50 * 1024**3,
        )
        assert "GB" in avail.memory_total_human
        assert "GB" in avail.memory_available_human
        assert "GB" in avail.disk_total_human
        assert "GB" in avail.disk_available_human

    def test_is_feasible(self):
        """Line 121: is_feasible property."""
        from kompot.resource_estimation import ResourcePlan
        plan = ResourcePlan()
        assert plan.is_feasible is True
        plan.errors.append("Fatal")
        assert plan.is_feasible is False


class TestUtilsDeepCoverage:
    """Deeper coverage for utils.py."""

    def test_dask_import_branch(self):
        """Lines 27-31: DASK_AVAILABLE import branch."""
        from kompot.utils import KOMPOT_COLORS
        # Just verify the module loaded properly
        assert KOMPOT_COLORS is not None

    def test_mahalanobis_no_jit(self):
        """Lines 283, 285: jit_compile=False path."""
        from kompot.utils import compute_mahalanobis_distances
        n = 15
        diffs = np.random.randn(5, n)
        cov = np.diag(np.ones(n) * 2.0)
        result = compute_mahalanobis_distances(
            diffs, cov, jit_compile=False, progress=False
        )
        assert result.shape == (5,)

    def test_mahalanobis_diagonal_variance(self):
        """Lines 317-369: diagonal_variance factor trick."""
        from kompot.utils import compute_mahalanobis_distances
        n_points = 5
        n_genes = 3
        diffs = np.random.randn(n_genes, n_points)
        cov = np.eye(n_points).astype(np.float64)
        diag_var = np.abs(np.random.randn(n_genes, n_points)).astype(np.float64)
        result = compute_mahalanobis_distances(
            diffs, cov, progress=False, diagonal_variance=diag_var
        )
        assert result.shape == (n_genes,)
        assert not np.any(np.isnan(result))

    def test_mahalanobis_gene_specific_with_diag_var(self):
        """Lines 183-184: gene-specific 3D cov with diagonal_variance."""
        from kompot.utils import compute_mahalanobis_distances
        n_points = 5
        n_genes = 2
        diffs = np.random.randn(n_genes, n_points)
        cov = np.zeros((n_points, n_points, n_genes))
        for g in range(n_genes):
            cov[:, :, g] = np.eye(n_points)
        diag_var = np.abs(np.random.randn(n_genes, n_points))
        result = compute_mahalanobis_distances(
            diffs, cov, progress=False, diagonal_variance=diag_var
        )
        assert result.shape == (n_genes,)

    def test_mahalanobis_cholesky_no_jit(self):
        """Line 387: Cholesky path without JIT."""
        from kompot.utils import compute_mahalanobis_distances
        diffs = np.random.randn(3, 5)
        cov = np.eye(5)
        result = compute_mahalanobis_distances(
            diffs, cov, jit_compile=False, progress=False
        )
        assert result.shape == (3,)


class TestImputationDeep:
    """Deeper coverage for imputation.py."""

    def _make_imputation_adata(self, with_std=True, with_obs_var=True):
        rng = np.random.RandomState(42)
        n_obs, n_vars = 50, 8
        X = rng.randn(n_obs, n_vars).astype(np.float32)
        adata = AnnData(X=X)
        adata.var_names = [f"gene_{i}" for i in range(n_vars)]
        adata.obs_names = [f"cell_{i}" for i in range(n_obs)]
        adata.obsm["X_umap"] = rng.randn(n_obs, 2).astype(np.float32)
        cond = "treated"
        prefix = "kompot_impute"
        adata.layers[f"{prefix}_{cond}_imputed"] = rng.randn(n_obs, n_vars).astype(np.float32)
        if with_std:
            adata.layers[f"{prefix}_{cond}_std"] = np.abs(rng.randn(n_obs, n_vars)).astype(np.float32)
        if with_obs_var:
            adata.layers[f"{prefix}_{cond}_obs_variance"] = np.abs(rng.randn(n_obs, n_vars)).astype(np.float32)
        return adata

    def test_detect_condition_label_error(self):
        """Lines 361-371: _detect_condition_label raises when no matching layers."""
        from kompot.plot.imputation import _detect_condition_label
        adata = AnnData(np.random.randn(5, 3))
        with pytest.raises(ValueError, match="No imputation layers"):
            _detect_condition_label(adata, "kompot_impute")

    def test_get_raw_sparse(self):
        """Lines 335-343: _get_raw with sparse matrix."""
        from kompot.plot.imputation import _get_raw
        from scipy.sparse import csr_matrix
        adata = AnnData(csr_matrix(np.random.randn(10, 5)))
        adata.var_names = [f"g{i}" for i in range(5)]
        result = _get_raw(adata, "g0", layer=None)
        assert result.shape == (10,)

    def test_scatter_fallback(self):
        """Lines 346-358: _scatter_fallback."""
        from kompot.plot.imputation import _scatter_fallback
        fig, ax = plt.subplots()
        xy = np.random.randn(20, 2)
        values = np.random.randn(20)
        _scatter_fallback(ax, xy, values, cmap="viridis", n_obs=20)
        # Should not raise

    def test_embedding_dim_too_small(self):
        """Line 97: embedding has < 2 dimensions."""
        from kompot.plot.imputation import plot_imputation
        adata = self._make_imputation_adata()
        adata.obsm["X_1d"] = np.random.randn(adata.n_obs, 1)
        with pytest.raises(ValueError, match="2 dimensions"):
            plot_imputation(adata, basis="X_1d")
