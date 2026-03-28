"""Tests for kompot/plot/volcano/multi_da.py targeting uncovered lines.

Covers multi_volcano_da rendering, color handling, thresholds, legends,
background plots, and save/return paths.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from anndata import AnnData
import json
import os


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
