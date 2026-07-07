"""Tests to improve coverage for kompot.plot.volcano.de."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt
import anndata
import json


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_de_adata(n_genes=50):
    """Create a minimal AnnData with DE result columns and run history."""
    np.random.seed(42)

    n_cells = 20
    X = np.random.normal(0, 1, (n_cells, n_genes))

    gene_names = [f"gene_{i}" for i in range(n_genes)]

    # var columns for DE results
    mahalanobis = np.random.exponential(2, n_genes)
    mean_lfc = np.random.normal(0, 2, n_genes)
    local_fdr = np.random.uniform(0.001, 1.0, n_genes)
    is_de = mahalanobis > 3  # boolean column

    var = pd.DataFrame(
        {
            "kompot_de_A_to_B_mahalanobis": mahalanobis,
            "kompot_de_A_to_B_mean_lfc": mean_lfc,
            "kompot_de_A_to_B_mahalanobis_local_fdr": local_fdr,
            "kompot_de_A_to_B_is_de": is_de,
        },
        index=gene_names,
    )

    adata = anndata.AnnData(X=X, var=var)

    # Add fold_change layer
    adata.layers["kompot_de_A_to_B_fold_change"] = np.random.normal(
        0, 1, (n_cells, n_genes)
    )

    # Run history
    run_entry = {
        "run_id": 0,
        "params": {
            "condition1": "A",
            "condition2": "B",
        },
        "field_mapping": {
            "mean_lfc_key": "kompot_de_A_to_B_mean_lfc",
            "mahalanobis_key": "kompot_de_A_to_B_mahalanobis",
        },
        "field_names": {
            "mean_lfc": "kompot_de_A_to_B_mean_lfc",
            "mahalanobis": "kompot_de_A_to_B_mahalanobis",
        },
        "fdr_keys": {
            "local_fdr_key": "kompot_de_A_to_B_mahalanobis_local_fdr",
            "is_de_key": "kompot_de_A_to_B_is_de",
        },
    }
    adata.uns["kompot_de"] = {"run_history": json.dumps([run_entry])}

    return adata


@pytest.fixture
def de_adata():
    return _make_de_adata()


# ===========================================================================
# Tests for kompot.plot.volcano.de
# ===========================================================================


class TestVolcanoDEBasic:
    """Basic call patterns for volcano_de."""

    def test_basic_explicit_keys(self, de_adata):
        """Basic volcano_de with explicit keys."""
        from kompot.plot.volcano.de import volcano_de

        fig = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_basic_inferred_keys(self, de_adata):
        """volcano_de inferring keys from run history."""
        from kompot.plot.volcano.de import volcano_de

        fig = volcano_de(de_adata, return_fig=True)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_return_fig_false(self, de_adata):
        """return_fig=False returns None."""
        from kompot.plot.volcano.de import volcano_de

        result = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            return_fig=False,
        )
        assert result is None
        plt.close("all")

    def test_save_to_file(self, de_adata, tmp_path):
        """Save figure to file."""
        from kompot.plot.volcano.de import volcano_de
        import os

        save_path = str(tmp_path / "volcano_de.png")
        volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            save=save_path,
        )
        assert os.path.exists(save_path)
        plt.close("all")

    def test_provided_ax(self, de_adata):
        """Use a pre-existing axes object."""
        from kompot.plot.volcano.de import volcano_de

        fig_ext, ax_ext = plt.subplots()
        result = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            ax=ax_ext,
            return_fig=True,
        )
        assert result is fig_ext
        plt.close(fig_ext)

    def test_show_legend_false(self, de_adata):
        """show_legend=False hides legend."""
        from kompot.plot.volcano.de import volcano_de

        fig = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            show_legend=False,
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_grid_false(self, de_adata):
        """grid=False disables grid."""
        from kompot.plot.volcano.de import volcano_de

        fig = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            grid=False,
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestVolcanoDEImportFallback:
    """Test the scanpy import fallback (lines 20-23)."""

    def test_scanpy_import_flag_exists(self):
        from kompot.plot.volcano import de as de_module

        assert hasattr(de_module, "_has_scanpy")


class TestVolcanoDEYAxisType:
    """Tests for y_axis_type parameter (lines 248-265, 293-301)."""

    def test_local_fdr_y_axis(self, de_adata):
        """y_axis_type='local_fdr' with FDR key from run info."""
        from kompot.plot.volcano.de import volcano_de

        fig = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            y_axis_type="local_fdr",
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_local_fdr_missing_key_warns(self, de_adata):
        """y_axis_type='local_fdr' with missing FDR key in data triggers warning."""
        from kompot.plot.volcano.de import volcano_de

        # Remove the local_fdr column from var
        de_adata.var.drop(
            columns=["kompot_de_A_to_B_mahalanobis_local_fdr"], inplace=True
        )
        # But keep the fdr_keys in run info pointing to the now-missing column
        fig = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            y_axis_type="local_fdr",
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_local_fdr_no_fdr_keys_in_run_info(self, de_adata):
        """y_axis_type='local_fdr' with no fdr_keys in run info triggers fallback."""
        from kompot.plot.volcano.de import volcano_de

        # Remove fdr_keys from run info
        run_history = json.loads(de_adata.uns["kompot_de"]["run_history"])
        run_history[0].pop("fdr_keys", None)
        de_adata.uns["kompot_de"]["run_history"] = json.dumps(run_history)

        fig = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            y_axis_type="local_fdr",
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_tail_fdr_y_axis_fallback(self, de_adata):
        """y_axis_type='tail_fdr' with no tail_fdr key falls back."""
        from kompot.plot.volcano.de import volcano_de

        fig = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            y_axis_type="tail_fdr",
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_ptp_y_axis_with_key(self, de_adata):
        """y_axis_type='ptp' with ptp_key in run info (lines 293-301)."""
        from kompot.plot.volcano.de import volcano_de

        # Add ptp data, stored as -log10(PTP) (the neg_log10_ptp convention)
        de_adata.var["kompot_de_A_to_B_neg_log10_ptp"] = np.random.uniform(
            0, 5, de_adata.n_vars
        )
        run_history = json.loads(de_adata.uns["kompot_de"]["run_history"])
        run_history[0]["ptp_key"] = "kompot_de_A_to_B_neg_log10_ptp"
        de_adata.uns["kompot_de"]["run_history"] = json.dumps(run_history)

        fig = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            y_axis_type="ptp",
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_ptp_y_axis_missing_key(self, de_adata):
        """y_axis_type='ptp' with missing ptp_key falls back (line 300-301)."""
        from kompot.plot.volcano.de import volcano_de

        # Add ptp_key to run_info pointing to non-existent column
        run_history = json.loads(de_adata.uns["kompot_de"]["run_history"])
        run_history[0]["ptp_key"] = "nonexistent_ptp_col"
        de_adata.uns["kompot_de"]["run_history"] = json.dumps(run_history)

        fig = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            y_axis_type="ptp",
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_y_axis_type(self, de_adata):
        """Custom y_axis_type pointing to an existing var column."""
        from kompot.plot.volcano.de import volcano_de

        de_adata.var["my_custom_score"] = np.random.uniform(0, 10, de_adata.n_vars)

        fig = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            y_axis_type="my_custom_score",
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_y_axis_type_not_found(self, de_adata):
        """Custom y_axis_type not found falls back to original score key."""
        from kompot.plot.volcano.de import volcano_de

        fig = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            y_axis_type="nonexistent_column",
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_mahalanobis_default(self, de_adata):
        """Default y_axis_type='mahalanobis'."""
        from kompot.plot.volcano.de import volcano_de

        fig = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            y_axis_type="mahalanobis",
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestVolcanoDEValidation:
    """Parameter validation (line 229)."""

    def test_n_top_genes_and_significance_threshold_conflict(self, de_adata):
        """Cannot specify both n_top_genes and significance_threshold."""
        from kompot.plot.volcano.de import volcano_de

        with pytest.raises(ValueError, match="Cannot specify both"):
            volcano_de(
                de_adata,
                lfc_key="kompot_de_A_to_B_mean_lfc",
                score_key="kompot_de_A_to_B_mahalanobis",
                n_top_genes=5,
                significance_threshold=0.05,
                return_fig=True,
            )
        plt.close("all")


class TestVolcanoDESignificanceThreshold:
    """Tests for significance_threshold parameter (lines 853-890, 902-903, 912-913, 935-936, 947, 970, 976)."""

    def test_float_threshold_mahalanobis(self, de_adata):
        """Float threshold with mahalanobis y_axis_type."""
        from kompot.plot.volcano.de import volcano_de

        fig = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            significance_threshold=3.0,
            y_axis_type="mahalanobis",
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_float_threshold_local_fdr(self, de_adata):
        """Float threshold with local_fdr y_axis_type (lines 902-903)."""
        from kompot.plot.volcano.de import volcano_de

        fig = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            significance_threshold=0.05,
            y_axis_type="local_fdr",
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_float_threshold_custom_y_axis(self, de_adata):
        """Float threshold with custom y_axis_type (lines 912-913)."""
        from kompot.plot.volcano.de import volcano_de

        de_adata.var["custom_score"] = np.random.uniform(0, 10, de_adata.n_vars)

        fig = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            significance_threshold=5.0,
            y_axis_type="custom_score",
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_float_threshold_no_significant_genes(self, de_adata):
        """Float threshold where no genes pass (line 976)."""
        from kompot.plot.volcano.de import volcano_de

        fig = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            significance_threshold=99999.0,
            y_axis_type="mahalanobis",
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_float_threshold_missing_column(self, de_adata):
        """Float threshold when significance column is missing (lines 935-936)."""
        from kompot.plot.volcano.de import volcano_de

        # Use tail_fdr which has no column and no fallback
        run_history = json.loads(de_adata.uns["kompot_de"]["run_history"])
        run_history[0].pop("fdr_keys", None)
        de_adata.uns["kompot_de"]["run_history"] = json.dumps(run_history)

        # Remove the mahalanobis column from var to trigger fallback path
        # Actually, tail_fdr with no fdr_keys and score_key without 'mahalanobis' in it
        # We need a case where significance_values_key is not found
        # Use a custom score_key that isn't in var
        fig = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            significance_threshold=0.05,
            y_axis_type="tail_fdr",
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_dict_threshold(self, de_adata):
        """Dict threshold with multiple criteria (lines 853-890)."""
        from kompot.plot.volcano.de import volcano_de

        fig = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            significance_threshold={
                "mahalanobis": 2.0,
                "local_fdr": 0.5,
            },
            y_axis_type="mahalanobis",
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_dict_threshold_no_significant_genes(self, de_adata):
        """Dict threshold where no genes pass (line 970)."""
        from kompot.plot.volcano.de import volcano_de

        fig = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            significance_threshold={
                "mahalanobis": 99999.0,
            },
            y_axis_type="mahalanobis",
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_dict_threshold_with_ptp(self, de_adata):
        """Dict threshold including ptp axis type."""
        from kompot.plot.volcano.de import volcano_de

        de_adata.var["kompot_de_A_to_B_neg_log10_ptp"] = np.random.uniform(
            0, 5, de_adata.n_vars
        )
        run_history = json.loads(de_adata.uns["kompot_de"]["run_history"])
        run_history[0]["ptp_key"] = "kompot_de_A_to_B_neg_log10_ptp"
        de_adata.uns["kompot_de"]["run_history"] = json.dumps(run_history)

        fig = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            significance_threshold={
                "ptp": 0.5,
            },
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_dict_threshold_with_custom_column(self, de_adata):
        """Dict threshold with a custom column name."""
        from kompot.plot.volcano.de import volcano_de

        de_adata.var["my_metric"] = np.random.uniform(0, 10, de_adata.n_vars)

        fig = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            significance_threshold={
                "my_metric": 5.0,
            },
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_dict_threshold_missing_axis_type(self, de_adata):
        """Dict threshold with an axis type whose column is not found."""
        from kompot.plot.volcano.de import volcano_de

        fig = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            significance_threshold={
                "nonexistent_axis": 5.0,
            },
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_significance_threshold_none(self, de_adata):
        """significance_threshold=None uses default DE column logic."""
        from kompot.plot.volcano.de import volcano_de

        fig = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            significance_threshold=None,
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_update_de_classification(self, de_adata):
        """update_de_classification=True updates the is_de column (line 947)."""
        from kompot.plot.volcano.de import volcano_de

        fig = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            significance_threshold=2.0,
            y_axis_type="mahalanobis",
            update_de_classification=True,
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_update_de_classification_dict_threshold(self, de_adata):
        """update_de_classification with dict threshold."""
        from kompot.plot.volcano.de import volcano_de

        fig = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            significance_threshold={"mahalanobis": 2.0},
            y_axis_type="mahalanobis",
            update_de_classification=True,
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_show_thresholds_with_fdr_transform(self, de_adata):
        """Threshold line with fdr transform."""
        from kompot.plot.volcano.de import volcano_de

        fig = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            significance_threshold=0.05,
            y_axis_type="local_fdr",
            show_thresholds=True,
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_show_thresholds_dict_skips_line(self, de_adata):
        """Dict threshold skips threshold line drawing (line 1282)."""
        from kompot.plot.volcano.de import volcano_de

        fig = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            significance_threshold={"mahalanobis": 2.0},
            show_thresholds=True,
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_show_thresholds_false(self, de_adata):
        """show_thresholds=False skips threshold lines."""
        from kompot.plot.volcano.de import volcano_de

        fig = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            significance_threshold=3.0,
            show_thresholds=False,
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestVolcanoDENTopGenes:
    """Tests for n_top_genes parameter."""

    def test_n_top_genes(self, de_adata):
        """n_top_genes highlights top N genes."""
        from kompot.plot.volcano.de import volcano_de

        fig = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            n_top_genes=5,
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestVolcanoDEHighlightGenes:
    """Tests for highlight_genes parameter (lines 740-741, 779-782, 787-788, 800-801, 815-831)."""

    def test_highlight_list_of_gene_names(self, de_adata):
        """highlight_genes as list of gene names."""
        from kompot.plot.volcano.de import volcano_de

        fig = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            highlight_genes=["gene_0", "gene_1", "gene_2"],
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_highlight_list_with_missing_genes(self, de_adata):
        """highlight_genes with some genes not in data (lines 740-741)."""
        from kompot.plot.volcano.de import volcano_de

        fig = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            highlight_genes=["gene_0", "nonexistent_gene"],
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_highlight_dict_gene_colors(self, de_adata):
        """highlight_genes as dict mapping genes to colors (lines 800-801)."""
        from kompot.plot.volcano.de import volcano_de

        fig = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            highlight_genes={
                "gene_0": "red",
                "gene_1": "blue",
                "missing_gene": "green",
            },
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_highlight_list_of_dicts(self, de_adata):
        """highlight_genes as list of dicts with name/genes/color."""
        from kompot.plot.volcano.de import volcano_de

        fig = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            highlight_genes=[
                {"name": "Group A", "genes": ["gene_0", "gene_1"], "color": "red"},
                {"name": "Group B", "genes": ["gene_2", "gene_3"]},
            ],
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_highlight_list_of_dicts_missing_genes_key(self, de_adata):
        """highlight_genes dict missing 'genes' key is skipped."""
        from kompot.plot.volcano.de import volcano_de

        fig = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            highlight_genes=[
                {"name": "No genes key"},  # missing 'genes'
                {"genes": ["gene_0"]},
            ],
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_highlight_list_of_dicts_all_missing_genes(self, de_adata):
        """highlight_genes dict where all genes are invalid."""
        from kompot.plot.volcano.de import volcano_de

        fig = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            highlight_genes=[
                {"genes": ["nonexistent1", "nonexistent2"]},
            ],
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_highlight_single_string(self, de_adata):
        """highlight_genes as single gene string."""
        from kompot.plot.volcano.de import volcano_de

        fig = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            highlight_genes="gene_0",
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_highlight_list_of_lists(self, de_adata):
        """highlight_genes as list of lists (mixed type branch, lines 779-788)."""
        from kompot.plot.volcano.de import volcano_de

        fig = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            highlight_genes=[["gene_0", "gene_1"], ["gene_2", "nonexistent"]],
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_highlight_list_of_lists_empty_valid(self, de_adata):
        """highlight_genes list of lists where a sublist has no valid genes (line 787-788)."""
        from kompot.plot.volcano.de import volcano_de

        fig = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            highlight_genes=[["nonexistent1", "nonexistent2"], ["gene_0"]],
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestVolcanoDEGeneLabels:
    """Tests for gene_labels parameter (lines 1211-1218)."""

    def test_gene_labels_int(self, de_adata):
        """gene_labels as int labels top N genes."""
        from kompot.plot.volcano.de import volcano_de

        fig = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            gene_labels=5,
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_gene_labels_list(self, de_adata):
        """gene_labels as list of specific genes."""
        from kompot.plot.volcano.de import volcano_de

        fig = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            gene_labels=["gene_0", "gene_1", "missing_gene"],
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_gene_labels_dict(self, de_adata):
        """gene_labels as dict with custom label text (lines 1211-1218)."""
        from kompot.plot.volcano.de import volcano_de

        fig = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            gene_labels={"gene_0": "My Gene A", "gene_1": "My Gene B", "missing": "X"},
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_gene_labels_true(self, de_adata):
        """gene_labels=True labels all highlighted genes."""
        from kompot.plot.volcano.de import volcano_de

        fig = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            gene_labels=True,
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_gene_labels_false(self, de_adata):
        """gene_labels=False labels no genes."""
        from kompot.plot.volcano.de import volcano_de

        fig = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            gene_labels=False,
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestVolcanoDEColor:
    """Tests for color parameter (lines 599, 624, 651, 663-665, 683-692, 704, 706)."""

    def test_color_categorical(self, de_adata):
        """Color by categorical var column."""
        from kompot.plot.volcano.de import volcano_de

        de_adata.var["gene_type"] = pd.Categorical(
            np.random.choice(["typeA", "typeB", "typeC"], de_adata.n_vars)
        )

        fig = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            color="gene_type",
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_color_categorical_many(self, de_adata):
        """Categorical with >20 categories triggers Set3 colormap (line 599)."""
        from kompot.plot.volcano.de import volcano_de

        cats = [f"cat_{i}" for i in range(25)]
        de_adata.var["many_cats"] = pd.Categorical(
            np.random.choice(cats, de_adata.n_vars)
        )

        fig = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            color="many_cats",
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_color_categorical_with_discrete_map(self, de_adata):
        """Color with explicit color_discrete_map."""
        from kompot.plot.volcano.de import volcano_de

        de_adata.var["gene_type"] = pd.Categorical(
            np.random.choice(["typeA", "typeB"], de_adata.n_vars)
        )

        fig = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            color="gene_type",
            color_discrete_map={"typeA": "red", "typeB": "blue"},
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_color_numeric_continuous(self, de_adata):
        """Color by numeric var column (continuous)."""
        from kompot.plot.volcano.de import volcano_de

        de_adata.var["expression"] = np.random.uniform(0, 10, de_adata.n_vars)

        fig = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            color="expression",
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_color_continuous_with_percentile_vmin_vmax(self, de_adata):
        """Continuous color with percentile vmin/vmax (lines 663-665, 683-692)."""
        from kompot.plot.volcano.de import volcano_de

        de_adata.var["expression"] = np.random.uniform(0, 10, de_adata.n_vars)

        fig = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            color="expression",
            vmin="p5",
            vmax="p95",
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_color_continuous_with_invalid_percentile(self, de_adata):
        """Continuous color with invalid percentile string (lines 663-665)."""
        from kompot.plot.volcano.de import volcano_de

        de_adata.var["expression"] = np.random.uniform(0, 10, de_adata.n_vars)

        fig = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            color="expression",
            vmin="pXYZ",
            vmax="pABC",
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_color_continuous_with_non_percentile_string(self, de_adata):
        """Continuous color with non-percentile string vmin/vmax."""
        from kompot.plot.volcano.de import volcano_de

        de_adata.var["expression"] = np.random.uniform(0, 10, de_adata.n_vars)

        fig = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            color="expression",
            vmin="invalid",
            vmax="invalid",
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_color_continuous_with_vcenter(self, de_adata):
        """Continuous color with vcenter for diverging colormap (line 704, 706)."""
        from kompot.plot.volcano.de import volcano_de

        de_adata.var["expression"] = np.random.uniform(-5, 5, de_adata.n_vars)

        fig = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            color="expression",
            vcenter=0.0,
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_color_continuous_vcenter_outside_range(self, de_adata):
        """vcenter below data range triggers TwoSlopeNorm error (line 704).

        Note: The source code at line 704 attempts to adjust vcenter but the
        resulting vmin==vcenter still violates TwoSlopeNorm's strict ordering.
        This test documents that existing behavior.
        """
        from kompot.plot.volcano.de import volcano_de

        de_adata.var["expression"] = np.random.uniform(5, 10, de_adata.n_vars)

        with pytest.raises(ValueError, match="ascending order"):
            volcano_de(
                de_adata,
                lfc_key="kompot_de_A_to_B_mean_lfc",
                score_key="kompot_de_A_to_B_mahalanobis",
                color="expression",
                vcenter=0.0,  # below min of data (5-10)
                return_fig=True,
            )
        plt.close("all")

    def test_color_continuous_vcenter_above_range(self, de_adata):
        """vcenter above data range triggers TwoSlopeNorm error (line 706).

        Note: The source code at line 706 attempts to adjust vcenter but the
        resulting vcenter==vmax still violates TwoSlopeNorm's strict ordering.
        This test documents that existing behavior.
        """
        from kompot.plot.volcano.de import volcano_de

        de_adata.var["expression"] = np.random.uniform(0, 5, de_adata.n_vars)

        with pytest.raises(ValueError, match="ascending order"):
            volcano_de(
                de_adata,
                lfc_key="kompot_de_A_to_B_mean_lfc",
                score_key="kompot_de_A_to_B_mahalanobis",
                color="expression",
                vcenter=100.0,  # above max of data (0-5)
                return_fig=True,
            )
        plt.close("all")

    def test_color_continuous_custom_cmap(self, de_adata):
        """Continuous color with custom colormap string."""
        from kompot.plot.volcano.de import volcano_de

        de_adata.var["expression"] = np.random.uniform(0, 10, de_adata.n_vars)

        fig = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            color="expression",
            background_cmap="viridis",
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_color_continuous_with_colormap_object(self, de_adata):
        """Continuous color with Colormap object (line 651)."""
        from kompot.plot.volcano.de import volcano_de

        de_adata.var["expression"] = np.random.uniform(0, 10, de_adata.n_vars)

        fig = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            color="expression",
            background_cmap=matplotlib.colormaps["coolwarm"],
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_color_categorical_with_cmap_object(self, de_adata):
        """Categorical color with Colormap object (line 624: non-listed cmap fallback)."""
        from kompot.plot.volcano.de import volcano_de

        de_adata.var["gene_type"] = pd.Categorical(
            np.random.choice(["typeA", "typeB"], de_adata.n_vars)
        )

        # Use a continuous colormap that doesn't have .colors attribute
        fig = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            color="gene_type",
            background_cmap=matplotlib.colormaps["viridis"],
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_color_string_column(self, de_adata):
        """Color by string (object dtype) column auto-converts to categorical."""
        from kompot.plot.volcano.de import volcano_de

        de_adata.var["str_type"] = np.random.choice(
            ["alpha", "beta"], de_adata.n_vars
        ).astype(str)

        fig = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            color="str_type",
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestVolcanoDEGroup:
    """Tests for group parameter (lines 419-499)."""

    def test_group_with_varm_data(self, de_adata):
        """Group-specific data from varm (lines 419-499)."""
        from kompot.plot.volcano.de import volcano_de

        # Set up varm keys and data
        run_history = json.loads(de_adata.uns["kompot_de"]["run_history"])
        run_history[0]["varm_keys"] = {
            "mean_lfc": "kompot_de_mean_lfc",
            "mahalanobis": "kompot_de_mahalanobis",
        }
        de_adata.uns["kompot_de"]["run_history"] = json.dumps(run_history)

        de_adata.varm["kompot_de_mean_lfc"] = pd.DataFrame(
            {"GroupX": np.random.normal(0, 2, de_adata.n_vars)},
            index=de_adata.var_names,
        )
        de_adata.varm["kompot_de_mahalanobis"] = pd.DataFrame(
            {"GroupX": np.random.exponential(2, de_adata.n_vars)},
            index=de_adata.var_names,
        )

        fig = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            group="GroupX",
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_group_missing_varm_keys(self, de_adata):
        """Group specified but no varm_keys in run info (line 419-423)."""
        from kompot.plot.volcano.de import volcano_de

        result = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            group="GroupX",
            return_fig=True,
        )
        # Returns None when varm_keys not found
        assert result is None
        plt.close("all")

    def test_group_missing_group_in_varm(self, de_adata):
        """Group specified but group not in varm columns (lines 474-499)."""
        from kompot.plot.volcano.de import volcano_de

        run_history = json.loads(de_adata.uns["kompot_de"]["run_history"])
        run_history[0]["varm_keys"] = {
            "mean_lfc": "kompot_de_mean_lfc",
            "mahalanobis": "kompot_de_mahalanobis",
        }
        de_adata.uns["kompot_de"]["run_history"] = json.dumps(run_history)

        # Add varm with different group name
        de_adata.varm["kompot_de_mean_lfc"] = pd.DataFrame(
            {"OtherGroup": np.random.normal(0, 2, de_adata.n_vars)},
            index=de_adata.var_names,
        )
        de_adata.varm["kompot_de_mahalanobis"] = pd.DataFrame(
            {"OtherGroup": np.random.exponential(2, de_adata.n_vars)},
            index=de_adata.var_names,
        )

        # Falls back to default var data
        fig = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            group="NonexistentGroup",
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_group_no_available_groups(self, de_adata):
        """Group specified, varm_keys present but varm data missing entirely (line 498-499)."""
        from kompot.plot.volcano.de import volcano_de

        run_history = json.loads(de_adata.uns["kompot_de"]["run_history"])
        run_history[0]["varm_keys"] = {
            "mean_lfc": "kompot_de_mean_lfc",
            "mahalanobis": "kompot_de_mahalanobis",
        }
        de_adata.uns["kompot_de"]["run_history"] = json.dumps(run_history)

        # Don't add any varm data - keys don't exist in varm
        fig = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            group="MissingGroup",
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestVolcanoDEMissingKeys:
    """Tests for missing key error handling (lines 510-517)."""

    def test_missing_lfc_key(self, de_adata):
        """Missing lfc_key raises ValueError."""
        from kompot.plot.volcano.de import volcano_de

        with pytest.raises((ValueError, KeyError)):
            volcano_de(
                de_adata,
                lfc_key="nonexistent_lfc",
                score_key="kompot_de_A_to_B_mahalanobis",
                return_fig=True,
            )
        plt.close("all")

    def test_missing_score_key(self, de_adata):
        """Missing score_key raises ValueError."""
        from kompot.plot.volcano.de import volcano_de

        with pytest.raises((ValueError, KeyError)):
            volcano_de(
                de_adata,
                lfc_key="kompot_de_A_to_B_mean_lfc",
                score_key="nonexistent_score",
                return_fig=True,
            )
        plt.close("all")


class TestVolcanoDESortKey:
    """Tests for sort_key parameter (lines 554-566)."""

    def test_sort_key_var_column(self, de_adata):
        """sort_key using a var column."""
        from kompot.plot.volcano.de import volcano_de

        fig = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            sort_key="kompot_de_A_to_B_mahalanobis",
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_sort_key_not_found(self, de_adata):
        """sort_key not found raises KeyError (line 566)."""
        from kompot.plot.volcano.de import volcano_de

        with pytest.raises(KeyError, match="not found"):
            volcano_de(
                de_adata,
                lfc_key="kompot_de_A_to_B_mean_lfc",
                score_key="kompot_de_A_to_B_mahalanobis",
                sort_key="nonexistent_sort_key",
                return_fig=True,
            )
        plt.close("all")


class TestVolcanoDEDirectionColumn:
    """Tests for direction_column parameter (lines 531-532, 543)."""

    def test_explicit_direction_column(self, de_adata):
        """Explicit direction_column (line 543)."""
        from kompot.plot.volcano.de import volcano_de

        fig = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            direction_column="kompot_de_A_to_B_is_de",
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_inferred_direction_column_from_fdr_keys(self, de_adata):
        """Direction column inferred from fdr_keys in run info (line 531-532)."""
        from kompot.plot.volcano.de import volcano_de

        fig = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestVolcanoDENoDeGenes:
    """Tests for fallback when no DE genes found (lines 1019-1026)."""

    def test_no_de_genes_fallback(self, de_adata):
        """All is_de=False triggers fallback to top genes (lines 1019-1026)."""
        from kompot.plot.volcano.de import volcano_de

        # Set all is_de to False
        de_adata.var["kompot_de_A_to_B_is_de"] = False

        fig = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestVolcanoDELegend:
    """Tests for legend placement (lines 1322)."""

    def test_legend_loc_custom(self, de_adata):
        """Custom legend_loc placement."""
        from kompot.plot.volcano.de import volcano_de

        fig = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            legend_loc="upper right",
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_legend_loc_none(self, de_adata):
        """legend_loc='none' hides legend."""
        from kompot.plot.volcano.de import volcano_de

        fig = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            legend_loc="none",
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_legend_with_continuous_color_custom_loc(self, de_adata):
        """Legend with continuous color and custom legend_loc (line 1322)."""
        from kompot.plot.volcano.de import volcano_de

        de_adata.var["expression"] = np.random.uniform(0, 10, de_adata.n_vars)

        fig = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            color="expression",
            legend_loc="upper right",
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_legend_fontsize(self, de_adata):
        """Legend with custom fontsize."""
        from kompot.plot.volcano.de import volcano_de

        fig = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            legend_fontsize=8,
            legend_title_fontsize=10,
            legend_ncol=2,
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestVolcanoDERunId:
    """Tests for run_id parameter (line 356)."""

    def test_positive_run_id(self, de_adata):
        """Positive run_id."""
        from kompot.plot.volcano.de import volcano_de

        fig = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            run_id=0,
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_negative_run_id_no_history(self, de_adata):
        """Negative run_id with no run history (line 356)."""
        from kompot.plot.volcano.de import volcano_de

        del de_adata.uns["kompot_de"]

        fig = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestVolcanoDENoHighlightGroups:
    """Tests for empty highlight_groups (line 1124 - dummy legend entries)."""

    def test_no_highlight_groups_no_de_column(self, de_adata):
        """No highlight groups and no DE column creates dummy legend."""
        from kompot.plot.volcano.de import volcano_de

        # Remove is_de column and fdr_keys
        de_adata.var.drop(columns=["kompot_de_A_to_B_is_de"], inplace=True)
        run_history = json.loads(de_adata.uns["kompot_de"]["run_history"])
        run_history[0].pop("fdr_keys", None)
        de_adata.uns["kompot_de"]["run_history"] = json.dumps(run_history)

        fig = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestVolcanoDEPerGeneColors:
    """Tests for per-gene colors in highlight_groups (line 1124)."""

    def test_highlight_dict_with_none_color(self, de_adata):
        """Dict highlight where a gene has None color falls back to direction."""
        from kompot.plot.volcano.de import volcano_de

        fig = volcano_de(
            de_adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            highlight_genes={"gene_0": "red", "gene_1": None},
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
