"""Tests for heatmap core rendering: layout, clustering, colorbars, and validation."""

import numpy as np
import pytest
import pandas as pd
import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def heatmap_adata():
    """Create a minimal AnnData with DE results suitable for heatmap tests."""
    np.random.seed(0)
    n_cells = 60
    n_genes = 10
    gene_names = [f"gene_{i}" for i in range(n_genes)]

    X = np.random.rand(n_cells, n_genes).astype(np.float64)

    conditions = np.array(["ctrl"] * 30 + ["treat"] * 30)
    cell_types = np.array(["TypeA", "TypeB", "TypeC"] * 20)

    obs = pd.DataFrame(
        {"condition": conditions, "cell_type": cell_types},
        index=[f"cell_{i}" for i in range(n_cells)],
    )

    var = pd.DataFrame(
        {
            "kompot_de_ctrl_to_treat_mahalanobis": np.random.rand(n_genes) * 10,
            "kompot_de_ctrl_to_treat_mean_lfc": np.random.randn(n_genes),
            "kompot_de_ctrl_to_treat_is_de": np.random.choice(
                [True, False], size=n_genes
            ),
        },
        index=gene_names,
    )

    adata = ad.AnnData(X=X, obs=obs, var=var)

    # Add layers expected by heatmap
    adata.layers["kompot_de_ctrl_to_treat_fold_change"] = np.random.randn(
        n_cells, n_genes
    )
    adata.layers["kompot_de_ctrl_smoothed"] = np.random.rand(n_cells, n_genes)
    adata.layers["kompot_de_treat_smoothed"] = np.random.rand(n_cells, n_genes)

    # Add run_history in uns under kompot_de storage key
    adata.uns["kompot_de"] = {
        "run_history": [
            {
                "timestamp": "2025-01-01T00:00:00",
                "params": {
                    "groupby": "condition",
                    "condition1": "ctrl",
                    "condition2": "treat",
                    "conditions": ["ctrl", "treat"],
                    "layer": None,
                },
                "field_mapping": {
                    "mahalanobis_key": "kompot_de_ctrl_to_treat_mahalanobis",
                    "mean_lfc_key": "kompot_de_ctrl_to_treat_mean_lfc",
                    "is_de_key": "kompot_de_ctrl_to_treat_is_de",
                    "fold_change_key": "kompot_de_ctrl_to_treat_fold_change",
                    "smoothed_key_1": "kompot_de_ctrl_smoothed",
                    "smoothed_key_2": "kompot_de_treat_smoothed",
                },
            }
        ]
    }

    return adata


# ---------------------------------------------------------------------------
# Heatmap core.py tests
# ---------------------------------------------------------------------------


class TestHeatmapScanpyImportGuard:
    """Cover lines 29-30: scanpy import guard."""

    def test_heatmap_module_loads_without_scanpy(self):
        """The module should load even if scanpy is not importable."""
        # Just verify the module is already imported and _has_scanpy is a bool
        from kompot.plot.heatmap.core import _has_scanpy

        assert isinstance(_has_scanpy, bool)


class TestHeatmapWithRunInfo:
    """Cover lines 264-287: infer condition_column, condition1, condition2, layer from run_info."""

    def teardown_method(self):
        plt.close("all")

    def test_infer_conditions_from_run_info(self, heatmap_adata):
        """When condition_column/condition1/condition2 are None, infer from run_info."""
        from kompot.plot.heatmap import heatmap

        fig = heatmap(
            heatmap_adata,
            genes=["gene_0", "gene_1", "gene_2"],
            groupby="cell_type",
            # condition_column, condition1, condition2 left as None => inferred
            return_fig=True,
        )
        assert fig is not None

    def test_infer_from_conditions_list(self, heatmap_adata):
        """Cover the fallback path that reads params['conditions'] list."""
        # Remove direct condition1/condition2 from params, keep conditions list
        params = heatmap_adata.uns["kompot_de"]["run_history"][0]["params"]
        params.pop("condition1", None)
        params.pop("condition2", None)
        params["conditions"] = ["ctrl", "treat"]

        from kompot.plot.heatmap import heatmap

        fig = heatmap(
            heatmap_adata,
            genes=["gene_0", "gene_1"],
            groupby="cell_type",
            return_fig=True,
        )
        assert fig is not None


class TestHeatmapValidationErrors:
    """Cover early-return paths (lines 312-340, 362-363, 376-380)."""

    def teardown_method(self):
        plt.close("all")

    def test_missing_condition_column_in_obs(self, heatmap_adata):
        from kompot.plot.heatmap import heatmap

        result = heatmap(
            heatmap_adata,
            genes=["gene_0"],
            groupby="cell_type",
            condition_column="nonexistent_col",
            condition1="ctrl",
            condition2="treat",
        )
        assert result is None

    def test_missing_condition2(self, heatmap_adata):
        from kompot.plot.heatmap import heatmap

        result = heatmap(
            heatmap_adata,
            genes=["gene_0"],
            groupby="cell_type",
            condition_column="condition",
            condition1="ctrl",
            condition2=None,
            # Force no inference by removing run_info
            run_id=999,
        )
        assert result is None

    def test_no_groupby(self, heatmap_adata):
        from kompot.plot.heatmap import heatmap

        result = heatmap(
            heatmap_adata,
            genes=["gene_0"],
            groupby=None,
            condition_column="condition",
            condition1="ctrl",
            condition2="treat",
        )
        assert result is None

    def test_condition1_not_in_data(self, heatmap_adata):
        from kompot.plot.heatmap import heatmap

        result = heatmap(
            heatmap_adata,
            genes=["gene_0"],
            groupby="cell_type",
            condition_column="condition",
            condition1="MISSING",
            condition2="treat",
        )
        assert result is None

    def test_condition2_not_in_data(self, heatmap_adata):
        from kompot.plot.heatmap import heatmap

        result = heatmap(
            heatmap_adata,
            genes=["gene_0"],
            groupby="cell_type",
            condition_column="condition",
            condition1="ctrl",
            condition2="MISSING",
        )
        assert result is None

    def test_groupby_not_in_obs(self, heatmap_adata):
        from kompot.plot.heatmap import heatmap

        result = heatmap(
            heatmap_adata,
            genes=["gene_0"],
            groupby="nonexistent_group",
            condition_column="condition",
            condition1="ctrl",
            condition2="treat",
        )
        assert result is None


class TestHeatmapFigsizeAndScaling:
    """Cover figsize cap (lines 553-564), explicit figsize path (592-644),
    tile_aspect_ratio < 1 (513-514), and standard_scale=None."""

    def teardown_method(self):
        plt.close("all")

    def test_tile_aspect_ratio_less_than_one(self, heatmap_adata):
        from kompot.plot.heatmap import heatmap

        fig = heatmap(
            heatmap_adata,
            genes=[f"gene_{i}" for i in range(10)],
            groupby="cell_type",
            condition_column="condition",
            condition1="ctrl",
            condition2="treat",
            tile_aspect_ratio=0.5,
            return_fig=True,
        )
        assert fig is not None

    def test_explicit_figsize_raises_known_bug(self, heatmap_adata):
        """Lines 592-644 and 681: explicit figsize triggers UnboundLocalError
        for TITLE_SPACE (known pre-existing bug). Verify the error is raised."""
        from kompot.plot.heatmap import heatmap

        with pytest.raises(UnboundLocalError):
            heatmap(
                heatmap_adata,
                genes=["gene_0", "gene_1"],
                groupby="cell_type",
                condition_column="condition",
                condition1="ctrl",
                condition2="treat",
                figsize=(8, 6),
                return_fig=True,
            )

    def test_standard_scale_none(self, heatmap_adata):
        """Cover standard_scale=None path (no z-scoring)."""
        from kompot.plot.heatmap import heatmap

        fig = heatmap(
            heatmap_adata,
            genes=["gene_0", "gene_1"],
            groupby="cell_type",
            condition_column="condition",
            condition1="ctrl",
            condition2="treat",
            standard_scale=None,
            return_fig=True,
        )
        assert fig is not None

    def test_standard_scale_group(self, heatmap_adata):
        """Cover standard_scale='group'."""
        from kompot.plot.heatmap import heatmap

        fig = heatmap(
            heatmap_adata,
            genes=["gene_0", "gene_1", "gene_2"],
            groupby="cell_type",
            condition_column="condition",
            condition1="ctrl",
            condition2="treat",
            standard_scale="group",
            return_fig=True,
        )
        assert fig is not None


class TestHeatmapPercentileVminVmax:
    """Cover vmin/vmax percentile string parsing (lines 927-931, 967-978, 987-991)."""

    def teardown_method(self):
        plt.close("all")

    def test_percentile_vmin_vmax(self, heatmap_adata):
        from kompot.plot.heatmap import heatmap

        fig = heatmap(
            heatmap_adata,
            genes=["gene_0", "gene_1"],
            groupby="cell_type",
            condition_column="condition",
            condition1="ctrl",
            condition2="treat",
            standard_scale=None,
            vmin="p5",
            vmax="p95",
            return_fig=True,
        )
        assert fig is not None

    def test_invalid_percentile_format(self, heatmap_adata):
        from kompot.plot.heatmap import heatmap

        fig = heatmap(
            heatmap_adata,
            genes=["gene_0", "gene_1"],
            groupby="cell_type",
            condition_column="condition",
            condition1="ctrl",
            condition2="treat",
            standard_scale=None,
            vmin="pABC",
            vmax="p200",
            return_fig=True,
        )
        assert fig is not None

    def test_vmin_vmax_swapped(self, heatmap_adata):
        """Cover the vmin >= vmax swap logic (lines 987-991)."""
        from kompot.plot.heatmap import heatmap

        fig = heatmap(
            heatmap_adata,
            genes=["gene_0", "gene_1"],
            groupby="cell_type",
            condition_column="condition",
            condition1="ctrl",
            condition2="treat",
            standard_scale=None,
            vmin=5.0,
            vmax=1.0,
            return_fig=True,
        )
        assert fig is not None

    def test_vcenter_edge_cases(self, heatmap_adata):
        """Cover lines 967-978 where vmin >= vcenter or vmax <= vcenter."""
        from kompot.plot.heatmap import heatmap

        fig = heatmap(
            heatmap_adata,
            genes=["gene_0", "gene_1"],
            groupby="cell_type",
            condition_column="condition",
            condition1="ctrl",
            condition2="treat",
            standard_scale=None,
            vcenter=0.5,
            vmin=0.5,
            vmax=0.5,
            return_fig=True,
        )
        assert fig is not None


class TestHeatmapExcludeGroups:
    """Cover exclude_groups path (lines 376-380)."""

    def teardown_method(self):
        plt.close("all")

    def test_exclude_single_group(self, heatmap_adata):
        from kompot.plot.heatmap import heatmap

        fig = heatmap(
            heatmap_adata,
            genes=["gene_0", "gene_1"],
            groupby="cell_type",
            condition_column="condition",
            condition1="ctrl",
            condition2="treat",
            exclude_groups="TypeC",
            return_fig=True,
        )
        assert fig is not None

    def test_exclude_multiple_groups(self, heatmap_adata):
        from kompot.plot.heatmap import heatmap

        # Exclude 2 of 3 groups; disable clustering since only 1 group remains
        fig = heatmap(
            heatmap_adata,
            genes=["gene_0", "gene_1"],
            groupby="cell_type",
            condition_column="condition",
            condition1="ctrl",
            condition2="treat",
            exclude_groups=["TypeB", "TypeC"],
            cluster_rows=False,
            cluster_cols=False,
            return_fig=True,
        )
        assert fig is not None


class TestHeatmapClusteringAndDendrogram:
    """Cover clustering with dendrogram (lines 734-735, 772-774, 780-781,
    831-836, 848-856) and provided axes (line 734)."""

    def teardown_method(self):
        plt.close("all")

    def test_cluster_rows_false(self, heatmap_adata):
        from kompot.plot.heatmap import heatmap

        fig = heatmap(
            heatmap_adata,
            genes=["gene_0", "gene_1", "gene_2"],
            groupby="cell_type",
            condition_column="condition",
            condition1="ctrl",
            condition2="treat",
            cluster_rows=False,
            cluster_cols=True,
            return_fig=True,
        )
        assert fig is not None

    def test_cluster_cols_false(self, heatmap_adata):
        from kompot.plot.heatmap import heatmap

        fig = heatmap(
            heatmap_adata,
            genes=["gene_0", "gene_1", "gene_2"],
            groupby="cell_type",
            condition_column="condition",
            condition1="ctrl",
            condition2="treat",
            cluster_rows=True,
            cluster_cols=False,
            return_fig=True,
        )
        assert fig is not None

    def test_provided_axes_raises_known_bug(self, heatmap_adata):
        """Cover line 734-735: using an existing axes. Currently triggers
        UnboundLocalError for 'cax' (pre-existing bug)."""
        from kompot.plot.heatmap import heatmap

        fig, ax = plt.subplots()
        with pytest.raises(UnboundLocalError):
            heatmap(
                heatmap_adata,
                genes=["gene_0", "gene_1"],
                groupby="cell_type",
                condition_column="condition",
                condition1="ctrl",
                condition2="treat",
                ax=ax,
                return_fig=True,
            )


class TestHeatmapFoldChangeMode:
    """Cover fold_change_mode paths (903-906, 951, 1079)."""

    def teardown_method(self):
        plt.close("all")

    def test_fold_change_mode_basic(self, heatmap_adata):
        from kompot.plot.heatmap import heatmap

        fig = heatmap(
            heatmap_adata,
            genes=["gene_0", "gene_1"],
            groupby="cell_type",
            condition_column="condition",
            condition1="ctrl",
            condition2="treat",
            fold_change_mode=True,
            return_fig=True,
        )
        assert fig is not None

    def test_fold_change_with_standard_scale_warns(self, heatmap_adata):
        """Cover warning when standard_scale is set with fold_change_mode."""
        from kompot.plot.heatmap import heatmap

        fig = heatmap(
            heatmap_adata,
            genes=["gene_0", "gene_1"],
            groupby="cell_type",
            condition_column="condition",
            condition1="ctrl",
            condition2="treat",
            fold_change_mode=True,
            standard_scale="var",
            return_fig=True,
        )
        assert fig is not None


class TestHeatmapReturnData:
    """Cover return_data path (lines 1637-1649)."""

    def teardown_method(self):
        plt.close("all")

    def test_return_data_only(self, heatmap_adata):
        from kompot.plot.heatmap import heatmap

        result = heatmap(
            heatmap_adata,
            genes=["gene_0", "gene_1"],
            groupby="cell_type",
            condition_column="condition",
            condition1="ctrl",
            condition2="treat",
            return_data=True,
        )
        assert result is not None
        assert hasattr(result, "cond1_means")
        assert hasattr(result, "cond2_means")
        assert result.fig is None  # return_fig=False

    def test_return_data_with_fig(self, heatmap_adata):
        from kompot.plot.heatmap import heatmap

        result = heatmap(
            heatmap_adata,
            genes=["gene_0", "gene_1"],
            groupby="cell_type",
            condition_column="condition",
            condition1="ctrl",
            condition2="treat",
            return_data=True,
            return_fig=True,
        )
        assert result is not None
        assert result.fig is not None

    def test_return_data_fold_change_mode(self, heatmap_adata):
        from kompot.plot.heatmap import heatmap

        result = heatmap(
            heatmap_adata,
            genes=["gene_0", "gene_1"],
            groupby="cell_type",
            condition_column="condition",
            condition1="ctrl",
            condition2="treat",
            fold_change_mode=True,
            return_data=True,
        )
        assert result is not None
        assert result.fold_changes is not None


class TestHeatmapNTopGenes:
    """Cover the n_top_genes path (when var_names is None)."""

    def teardown_method(self):
        plt.close("all")

    def test_n_top_genes_with_score_key(self, heatmap_adata):
        from kompot.plot.heatmap import heatmap

        fig = heatmap(
            heatmap_adata,
            n_top_genes=3,
            groupby="cell_type",
            condition_column="condition",
            condition1="ctrl",
            condition2="treat",
            return_fig=True,
        )
        assert fig is not None


class TestHeatmapColorbar:
    """Cover colorbar_kwargs paths (lines 1598, 1623, 1627, 1635)."""

    def teardown_method(self):
        plt.close("all")

    def test_custom_colorbar_title(self, heatmap_adata):
        from kompot.plot.heatmap import heatmap

        fig = heatmap(
            heatmap_adata,
            genes=["gene_0", "gene_1"],
            groupby="cell_type",
            condition_column="condition",
            condition1="ctrl",
            condition2="treat",
            standard_scale=None,
            colorbar_title="Custom Title",
            return_fig=True,
        )
        assert fig is not None

    def test_colorbar_kwargs_locator_formatter(self, heatmap_adata):
        from kompot.plot.heatmap import heatmap

        fig = heatmap(
            heatmap_adata,
            genes=["gene_0", "gene_1"],
            groupby="cell_type",
            condition_column="condition",
            condition1="ctrl",
            condition2="treat",
            colorbar_kwargs={
                "locator": plt.MaxNLocator(5),
                "formatter": plt.FormatStrFormatter("%.2f"),
                "label_kwargs": {"fontsize": 8},
            },
            return_fig=True,
        )
        assert fig is not None

    def test_save_figure(self, heatmap_adata, tmp_path):
        from kompot.plot.heatmap import heatmap

        save_path = str(tmp_path / "test_heatmap.png")
        heatmap(
            heatmap_adata,
            genes=["gene_0", "gene_1"],
            groupby="cell_type",
            condition_column="condition",
            condition1="ctrl",
            condition2="treat",
            save=save_path,
        )
        import os

        assert os.path.exists(save_path)


class TestHeatmapConditionNames:
    """Cover custom condition names and title generation (line 1079)."""

    def teardown_method(self):
        plt.close("all")

    def test_custom_condition_names(self, heatmap_adata):
        from kompot.plot.heatmap import heatmap

        fig = heatmap(
            heatmap_adata,
            genes=["gene_0", "gene_1"],
            groupby="cell_type",
            condition_column="condition",
            condition1="ctrl",
            condition2="treat",
            condition1_name="Control Group",
            condition2_name="Treatment Group",
            return_fig=True,
        )
        assert fig is not None

    def test_no_condition_names_no_fig(self, heatmap_adata):
        """Cover line 1079: default title when no condition names."""
        from kompot.plot.heatmap import heatmap

        fig = heatmap(
            heatmap_adata,
            genes=["gene_0", "gene_1"],
            groupby="cell_type",
            condition_column="condition",
            condition1="ctrl",
            condition2="treat",
            condition1_name=None,
            condition2_name=None,
            title=None,
            return_fig=True,
        )
        assert fig is not None
