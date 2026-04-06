"""
Tests for condition extraction priority in volcano, multi_volcano, and direction plots.

These tests mock run_info directly (no DE/DA computation) so they run fast (~3s).
They verify the fix: run_info['params'] is used as the authoritative source for
condition names instead of the broken _extract_conditions_from_key() string splitter.

All tests use custom result_keys since the default case is trivially correct.
"""

import numpy as np
import pandas as pd
import anndata as ad
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for testing
import matplotlib.pyplot as plt

from kompot.anndata.utils.field_tracking import (
    generate_output_field_names,
    append_to_run_history,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_da_adata(condition1, condition2, result_key="custom_da", extra_run=None):
    """Build an AnnData with mocked DA run_info and obs columns (no DA needed)."""
    n_cells = 40
    adata = ad.AnnData(
        X=np.zeros((n_cells, 5)),
        obs=pd.DataFrame({"cond": ["A"] * 20 + ["B"] * 20}),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(5)]),
    )
    adata.obsm["DM_EigenVectors"] = np.random.randn(n_cells, 10)
    adata.obsm["X_pca"] = np.random.randn(n_cells, 10)

    fn = generate_output_field_names(result_key, condition1, condition2, "da")
    run_info = {
        "result_key": result_key,
        "field_names": fn,
        "params": {
            "condition1": condition1,
            "condition2": condition2,
            "result_key": result_key,
        },
    }

    # Populate obs columns
    adata.obs[fn["lfc_key"]] = np.random.randn(n_cells)
    adata.obs[fn["ptp_key"]] = np.random.uniform(0.001, 1.0, n_cells)
    adata.obs[fn["direction_key"]] = pd.Categorical(
        np.random.choice(["up", "down", "neutral"], n_cells)
    )
    if "zscore_key" in fn:
        adata.obs[fn["zscore_key"]] = np.random.randn(n_cells)
    if "density_key_1" in fn:
        adata.obs[fn["density_key_1"]] = np.random.randn(n_cells)
    if "density_key_2" in fn:
        adata.obs[fn["density_key_2"]] = np.random.randn(n_cells)

    # Add a groupby column for multi_volcano_da
    adata.obs["cell_type"] = pd.Categorical(
        np.random.choice(["TypeA", "TypeB"], n_cells)
    )

    adata.uns["kompot_da"] = {"run_history": "[]"}

    if extra_run is not None:
        fn_extra = generate_output_field_names(extra_run, condition1, condition2, "da")
        extra_info = {
            "result_key": extra_run,
            "field_names": fn_extra,
            "params": {
                "condition1": condition1,
                "condition2": condition2,
                "result_key": extra_run,
            },
        }
        adata.obs[fn_extra["lfc_key"]] = np.random.randn(n_cells) * 0.5
        adata.obs[fn_extra["ptp_key"]] = np.random.uniform(0.001, 1.0, n_cells)
        adata.obs[fn_extra["direction_key"]] = pd.Categorical(
            np.random.choice(["up", "down", "neutral"], n_cells)
        )
        append_to_run_history(adata, extra_info, "da")

    append_to_run_history(adata, run_info, "da")
    return adata, fn


def _make_de_adata(condition1, condition2, result_key="custom_de", extra_run=None):
    """Build an AnnData with mocked DE run_info and var columns (no DE needed)."""
    n_cells, n_genes = 20, 10
    adata = ad.AnnData(
        X=np.zeros((n_cells, n_genes)),
        obs=pd.DataFrame({"cond": ["A"] * 10 + ["B"] * 10}),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)]),
    )
    adata.obsm["X_umap"] = np.random.randn(n_cells, 2)

    fn = generate_output_field_names(result_key, condition1, condition2, "de")
    run_info = {
        "result_key": result_key,
        "imputed_layer_keys": {
            "condition1": fn["imputed_key_1"],
            "condition2": fn["imputed_key_2"],
            "fold_change": fn["fold_change_key"],
        },
        "field_names": fn,
        "params": {
            "condition1": condition1,
            "condition2": condition2,
            "result_key": result_key,
        },
    }

    # Populate var columns
    adata.var[fn["mean_lfc_key"]] = np.random.randn(n_genes)
    adata.var[fn["mahalanobis_key"]] = np.abs(np.random.randn(n_genes)) + 1.0

    adata.uns["kompot_de"] = {"run_history": "[]"}

    if extra_run is not None:
        fn_extra = generate_output_field_names(extra_run, condition1, condition2, "de")
        extra_info = {
            "result_key": extra_run,
            "imputed_layer_keys": {
                "condition1": fn_extra["imputed_key_1"],
                "condition2": fn_extra["imputed_key_2"],
                "fold_change": fn_extra["fold_change_key"],
            },
            "field_names": fn_extra,
            "params": {
                "condition1": condition1,
                "condition2": condition2,
                "result_key": extra_run,
            },
        }
        adata.var[fn_extra["mean_lfc_key"]] = np.random.randn(n_genes) * 0.5
        adata.var[fn_extra["mahalanobis_key"]] = np.abs(np.random.randn(n_genes))
        append_to_run_history(adata, extra_info, "de")

    append_to_run_history(adata, run_info, "de")
    return adata, fn


# ===========================================================================
# volcano_da
# ===========================================================================


class TestVolcanoDaConditionExtraction:
    """Verify volcano_da prioritizes run_info['params'] for condition names."""

    def test_simple_conditions_custom_key(self):
        """Simple conditions with custom result_key."""
        from kompot.plot.volcano.da import volcano_da

        adata, fn = _make_da_adata("Young", "Old", result_key="age_da")
        fig = volcano_da(adata, return_fig=True)
        ax = fig.axes[0]
        xlabel = ax.get_xlabel()
        assert "Young" in xlabel and "Old" in xlabel
        plt.close(fig)

    def test_multiword_conditions_custom_key(self):
        """Multi-word conditions must come from run_info, not key splitting."""
        from kompot.plot.volcano.da import volcano_da

        adata, fn = _make_da_adata(
            "Pre-treatment", "Post-treatment", result_key="treat_da"
        )
        fig = volcano_da(adata, return_fig=True)
        ax = fig.axes[0]
        xlabel = ax.get_xlabel()
        # run_info params stores the original unsanitized names
        assert "Pre-treatment" in xlabel
        assert "Post-treatment" in xlabel
        plt.close(fig)

    def test_two_runs_latest_wins(self):
        """With two runs, latest run's conditions should be used."""
        from kompot.plot.volcano.da import volcano_da

        adata, fn = _make_da_adata(
            "Healthy", "Disease", result_key="run2_da", extra_run="run1_da"
        )
        fig = volcano_da(adata, return_fig=True)
        ax = fig.axes[0]
        xlabel = ax.get_xlabel()
        assert "Healthy" in xlabel and "Disease" in xlabel
        plt.close(fig)

    def test_conditions_with_spaces(self):
        """Conditions with spaces must resolve from run_info params."""
        from kompot.plot.volcano.da import volcano_da

        adata, fn = _make_da_adata("Wild Type", "Knock Out", result_key="wt_ko_da")
        fig = volcano_da(adata, return_fig=True)
        ax = fig.axes[0]
        xlabel = ax.get_xlabel()
        assert "Wild Type" in xlabel
        assert "Knock Out" in xlabel
        plt.close(fig)


# ===========================================================================
# volcano_de
# ===========================================================================


class TestVolcanoDeConditionExtraction:
    """Verify volcano_de prioritizes run_info['params'] for condition names."""

    def test_simple_conditions_custom_key(self):
        """Simple conditions with custom result_key."""
        from kompot.plot.volcano.de import volcano_de

        adata, fn = _make_de_adata("Young", "Old", result_key="age_de")
        fig = volcano_de(adata, return_fig=True)
        ax = fig.axes[0]
        xlabel = ax.get_xlabel()
        assert "Young" in xlabel and "Old" in xlabel
        plt.close(fig)

    def test_multiword_conditions_custom_key(self):
        """Multi-word conditions must come from run_info, not key splitting."""
        from kompot.plot.volcano.de import volcano_de

        adata, fn = _make_de_adata(
            "Pre-treatment", "Post-treatment", result_key="treat_de"
        )
        fig = volcano_de(adata, return_fig=True)
        ax = fig.axes[0]
        xlabel = ax.get_xlabel()
        assert "Pre-treatment" in xlabel
        assert "Post-treatment" in xlabel
        plt.close(fig)

    def test_two_runs_latest_wins(self):
        """With two runs, latest run's conditions should be used."""
        from kompot.plot.volcano.de import volcano_de

        adata, fn = _make_de_adata(
            "Healthy", "Disease", result_key="run2_de", extra_run="run1_de"
        )
        fig = volcano_de(adata, return_fig=True)
        ax = fig.axes[0]
        xlabel = ax.get_xlabel()
        assert "Healthy" in xlabel and "Disease" in xlabel
        plt.close(fig)

    def test_conditions_with_spaces(self):
        """Conditions with spaces must resolve from run_info params."""
        from kompot.plot.volcano.de import volcano_de

        adata, fn = _make_de_adata("Wild Type", "Knock Out", result_key="wt_ko_de")
        fig = volcano_de(adata, return_fig=True)
        ax = fig.axes[0]
        xlabel = ax.get_xlabel()
        assert "Wild Type" in xlabel
        assert "Knock Out" in xlabel
        plt.close(fig)


# ===========================================================================
# multi_volcano_da
# ===========================================================================


class TestMultiVolcanoDaConditionExtraction:
    """Verify multi_volcano_da prioritizes run_info['params'] for condition names."""

    def test_simple_conditions_custom_key(self):
        """Simple conditions with custom result_key."""
        from kompot.plot.volcano.multi_da import multi_volcano_da

        adata, fn = _make_da_adata("Young", "Old", result_key="age_da")
        fig = multi_volcano_da(adata, groupby="cell_type", return_fig=True)
        # Find the axis that has the xlabel with conditions
        xlabels = [ax.get_xlabel() for ax in fig.axes]
        xlabel = next((x for x in xlabels if "Log Fold Change" in x), "")
        assert "Young" in xlabel and "Old" in xlabel
        plt.close(fig)

    def test_multiword_conditions_custom_key(self):
        """Multi-word conditions must come from run_info, not key splitting."""
        from kompot.plot.volcano.multi_da import multi_volcano_da

        adata, fn = _make_da_adata(
            "Pre-treatment", "Post-treatment", result_key="treat_da"
        )
        fig = multi_volcano_da(adata, groupby="cell_type", return_fig=True)
        xlabels = [ax.get_xlabel() for ax in fig.axes]
        xlabel = next((x for x in xlabels if "Log Fold Change" in x), "")
        assert "Pre-treatment" in xlabel
        assert "Post-treatment" in xlabel
        plt.close(fig)

    def test_conditions_with_spaces(self):
        """Conditions with spaces must resolve from run_info params."""
        from kompot.plot.volcano.multi_da import multi_volcano_da

        adata, fn = _make_da_adata("Wild Type", "Knock Out", result_key="wt_ko_da")
        fig = multi_volcano_da(adata, groupby="cell_type", return_fig=True)
        xlabels = [ax.get_xlabel() for ax in fig.axes]
        xlabel = next((x for x in xlabels if "Log Fold Change" in x), "")
        assert "Wild Type" in xlabel
        assert "Knock Out" in xlabel
        plt.close(fig)


# ===========================================================================
# direction_barplot
# ===========================================================================


class TestDirectionBarplotConditionExtraction:
    """Verify direction_barplot prioritizes run_info['params'] for condition names."""

    def test_simple_conditions_custom_key(self):
        """Simple conditions with custom result_key."""
        from kompot.plot.heatmap.direction_plot import direction_barplot

        adata, fn = _make_da_adata("Young", "Old", result_key="age_da")
        fig = direction_barplot(
            adata,
            category_column="cell_type",
            direction_column=fn["direction_key"],
            return_fig=True,
        )
        ax = fig.axes[0]
        title = ax.get_title()
        assert "Young" in title and "Old" in title
        plt.close(fig)

    def test_multiword_conditions_custom_key(self):
        """Multi-word conditions must come from run_info, not key splitting."""
        from kompot.plot.heatmap.direction_plot import direction_barplot

        adata, fn = _make_da_adata(
            "Pre-treatment", "Post-treatment", result_key="treat_da"
        )
        fig = direction_barplot(
            adata,
            category_column="cell_type",
            direction_column=fn["direction_key"],
            return_fig=True,
        )
        ax = fig.axes[0]
        title = ax.get_title()
        assert "Pre-treatment" in title
        assert "Post-treatment" in title
        plt.close(fig)

    def test_conditions_with_spaces(self):
        """Conditions with spaces must resolve from run_info params."""
        from kompot.plot.heatmap.direction_plot import direction_barplot

        adata, fn = _make_da_adata("Wild Type", "Knock Out", result_key="wt_ko_da")
        fig = direction_barplot(
            adata,
            category_column="cell_type",
            direction_column=fn["direction_key"],
            return_fig=True,
        )
        ax = fig.axes[0]
        title = ax.get_title()
        assert "Wild Type" in title
        assert "Knock Out" in title
        plt.close(fig)

    def test_inferred_direction_column_custom_key(self):
        """Direction column inferred from run_info (not provided explicitly)."""
        from kompot.plot.heatmap.direction_plot import direction_barplot

        adata, fn = _make_da_adata(
            "Pre-treatment", "Post-treatment", result_key="infer_da"
        )
        fig = direction_barplot(
            adata,
            category_column="cell_type",
            # direction_column NOT provided - should be inferred from run_info
            return_fig=True,
        )
        ax = fig.axes[0]
        title = ax.get_title()
        assert "Pre-treatment" in title
        assert "Post-treatment" in title
        plt.close(fig)
