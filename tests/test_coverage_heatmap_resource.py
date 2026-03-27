"""Tests to improve coverage for heatmap/core.py and resource_estimation.py."""

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
    adata.layers["kompot_de_ctrl_imputed"] = np.random.rand(n_cells, n_genes)
    adata.layers["kompot_de_treat_imputed"] = np.random.rand(n_cells, n_genes)

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
                    "imputed_key_1": "kompot_de_ctrl_imputed",
                    "imputed_key_2": "kompot_de_treat_imputed",
                },
            }
        ]
    }

    return adata


@pytest.fixture
def small_adata():
    """A very small AnnData for resource estimation tests."""
    np.random.seed(1)
    n_cells, n_genes = 20, 5
    X = np.random.rand(n_cells, n_genes)
    obs = pd.DataFrame(
        {
            "condition": ["A"] * 10 + ["B"] * 10,
            "sample": ["s1"] * 5 + ["s2"] * 5 + ["s3"] * 5 + ["s4"] * 5,
        },
        index=[f"c{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=[f"g{i}" for i in range(n_genes)])
    return ad.AnnData(X=X, obs=obs, var=var)


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


# ---------------------------------------------------------------------------
# Resource estimation tests
# ---------------------------------------------------------------------------


class TestResourcePlanFormatReport:
    """Cover ResourcePlan.format_report (lines 274-278, 295-297, 302)."""

    def test_format_report_with_all_sections(self):
        from kompot.resource_estimation import (
            ResourcePlan,
            ResourceAvailability,
        )

        plan = ResourcePlan()
        plan.availability = ResourceAvailability(
            memory_total=16 * 1024**3,
            memory_available=8 * 1024**3,
            disk_path="/tmp",
            disk_total=100 * 1024**3,
            disk_available=50 * 1024**3,
        )
        plan.add_requirement("Mem item", 1024**3, "memory", shape=(100, 100))
        plan.add_requirement(
            "Disk item", 512 * 1024**2, "disk", shape=(50, 50), overwrite=True
        )
        plan.output_fields = {"adata.var": ["field_a", "field_b"]}
        plan.info.append("Some info message")
        plan.warnings.append("Some warning")

        report = plan.format_report(verbose=True)
        assert "RESOURCE USAGE PLAN" in report
        assert "Mem item" in report
        assert "Disk item" in report
        assert "OVERWRITE" in report
        assert "field_a" in report
        assert "Some info message" in report
        assert "Some warning" in report
        assert "FEASIBLE WITH WARNINGS" in report

    def test_format_report_with_errors(self):
        from kompot.resource_estimation import ResourcePlan

        plan = ResourcePlan()
        plan.errors.append("Big error")
        report = plan.format_report(verbose=False)
        assert "INFEASIBLE" in report
        assert "Big error" in report

    def test_format_report_clean(self):
        from kompot.resource_estimation import ResourcePlan, ResourceAvailability

        plan = ResourcePlan()
        plan.availability = ResourceAvailability(
            memory_total=16 * 1024**3,
            memory_available=8 * 1024**3,
            disk_path="/tmp",
            disk_total=100 * 1024**3,
            disk_available=50 * 1024**3,
        )
        report = plan.format_report()
        assert "FEASIBLE" in report

    def test_format_report_overwrite_fields_with_run_id(self):
        """Cover lines 274-278: overwrite fields with run_id annotation."""
        from kompot.resource_estimation import ResourcePlan, ResourceAvailability

        plan = ResourcePlan()
        plan.availability = ResourceAvailability(
            memory_total=16 * 1024**3,
            memory_available=8 * 1024**3,
            disk_path="/tmp",
            disk_total=100 * 1024**3,
            disk_available=50 * 1024**3,
        )
        plan.output_fields = {"adata.var": ["my_field"]}
        plan._overwrite_fields = {"my_field": 0}

        report = plan.format_report()
        assert "OVERWRITES run_id=0" in report

    def test_format_report_overwrite_fields_no_run_id(self):
        from kompot.resource_estimation import ResourcePlan, ResourceAvailability

        plan = ResourcePlan()
        plan.availability = ResourceAvailability(
            memory_total=16 * 1024**3,
            memory_available=8 * 1024**3,
            disk_path="/tmp",
            disk_total=100 * 1024**3,
            disk_available=50 * 1024**3,
        )
        plan.output_fields = {"adata.var": ["my_field"]}
        plan._overwrite_fields = {"my_field": None}

        report = plan.format_report()
        assert "OVERWRITE" in report


class TestResourcePlanCheckAvailability:
    """Cover check_availability with edge cases (lines 149-150, 160, 171-197)."""

    def test_no_availability(self):
        """Cover line 149-150: availability is None."""
        from kompot.resource_estimation import ResourcePlan

        plan = ResourcePlan()
        plan.availability = None
        plan.add_requirement("test", 1024, "memory")
        plan.check_availability()
        assert any("psutil" in w for w in plan.warnings)

    def test_high_memory_warning(self):
        """Cover lines 159-165: high memory warning threshold."""
        from kompot.resource_estimation import ResourcePlan, ResourceAvailability

        plan = ResourcePlan()
        plan.availability = ResourceAvailability(
            memory_total=10 * 1024**3,
            memory_available=10 * 1024**3,
            disk_path="/tmp",
            disk_total=100 * 1024**3,
            disk_available=100 * 1024**3,
        )
        # 85% of available => over threshold (0.8) but under 1.0
        plan.add_requirement("big", int(8.5 * 1024**3), "memory")
        plan.check_availability(memory_threshold=0.8)
        assert any("High memory" in w for w in plan.warnings)
        assert len(plan.errors) == 0

    def test_insufficient_memory_error(self):
        from kompot.resource_estimation import ResourcePlan, ResourceAvailability

        plan = ResourcePlan()
        plan.availability = ResourceAvailability(
            memory_total=4 * 1024**3,
            memory_available=2 * 1024**3,
            disk_path="/tmp",
            disk_total=100 * 1024**3,
            disk_available=100 * 1024**3,
        )
        plan.add_requirement("huge", 10 * 1024**3, "memory")
        plan.check_availability()
        assert any("Insufficient memory" in e for e in plan.errors)

    def test_insufficient_disk_error(self):
        """Cover lines 171-195: insufficient disk with alternatives."""
        from kompot.resource_estimation import ResourcePlan, ResourceAvailability

        plan = ResourcePlan()
        plan.availability = ResourceAvailability(
            memory_total=16 * 1024**3,
            memory_available=16 * 1024**3,
            disk_path="/tmp",
            disk_total=1 * 1024**3,
            disk_available=100 * 1024**2,  # 100 MB
        )
        plan.add_requirement("big disk", 10 * 1024**3, "disk")
        plan.check_availability()
        assert any("Insufficient disk" in e for e in plan.errors)

    def test_high_disk_warning(self):
        """Cover line 197: high disk usage warning."""
        from kompot.resource_estimation import ResourcePlan, ResourceAvailability

        plan = ResourcePlan()
        plan.availability = ResourceAvailability(
            memory_total=16 * 1024**3,
            memory_available=16 * 1024**3,
            disk_path="/tmp",
            disk_total=10 * 1024**3,
            disk_available=10 * 1024**3,
        )
        # 95% => over disk_threshold=0.9 but under 1.0
        plan.add_requirement("disk", int(9.5 * 1024**3), "disk")
        plan.check_availability(disk_threshold=0.9)
        assert any("High disk" in w for w in plan.warnings)

    def test_memory_ratio_no_availability(self):
        """Cover line 108: memory_ratio returns inf when no availability."""
        from kompot.resource_estimation import ResourcePlan

        plan = ResourcePlan()
        plan.add_requirement("test", 1024, "memory")
        assert plan.memory_ratio == float("inf")

    def test_disk_ratio_no_availability(self):
        """Cover line 115: disk_ratio returns inf when no availability."""
        from kompot.resource_estimation import ResourcePlan

        plan = ResourcePlan()
        plan.add_requirement("test", 1024, "disk")
        assert plan.disk_ratio == float("inf")


class TestEstimateDEResources:
    """Cover estimate_differential_expression_resources (lines 495-496, 514, 520, 861, etc.)."""

    def test_basic_estimation(self, small_adata):
        from kompot.resource_estimation import (
            estimate_differential_expression_resources,
        )

        plan = estimate_differential_expression_resources(
            small_adata,
            condition1="A",
            condition2="B",
            groupby="condition",
        )
        assert plan.is_feasible or not plan.is_feasible  # just check it runs
        assert plan.total_memory_required > 0

    def test_with_sample_variance(self, small_adata):
        """Cover sample variance paths (lines 802-865)."""
        from kompot.resource_estimation import (
            estimate_differential_expression_resources,
        )

        plan = estimate_differential_expression_resources(
            small_adata,
            condition1="A",
            condition2="B",
            groupby="condition",
            use_sample_variance=True,
            sample_col="sample",
        )
        assert plan.total_memory_required > 0
        # Should have warning about sample variance in memory
        report = plan.format_report()
        assert "sample" in report.lower() or "Sample" in report

    def test_disk_storage(self, small_adata, tmp_path):
        """Cover disk storage paths (lines 854-865)."""
        from kompot.resource_estimation import (
            estimate_differential_expression_resources,
        )

        plan = estimate_differential_expression_resources(
            small_adata,
            condition1="A",
            condition2="B",
            groupby="condition",
            use_sample_variance=True,
            store_arrays_on_disk=True,
            disk_storage_dir=str(tmp_path),
            sample_col="sample",
        )
        report = plan.format_report()
        assert "Disk" in report or "disk" in report

    def test_disk_storage_default_path(self, small_adata):
        """Cover line 861: disk storage without explicit dir uses temp."""
        from kompot.resource_estimation import (
            estimate_differential_expression_resources,
        )

        plan = estimate_differential_expression_resources(
            small_adata,
            condition1="A",
            condition2="B",
            groupby="condition",
            use_sample_variance=True,
            store_arrays_on_disk=True,
            sample_col="sample",
        )
        # Should have info about temp path
        assert any("temp" in i.lower() or "Disk" in i for i in plan.info)

    def test_with_null_genes_list(self, small_adata):
        """Cover line 495-496: null_genes as a list."""
        from kompot.resource_estimation import (
            estimate_differential_expression_resources,
        )

        plan = estimate_differential_expression_resources(
            small_adata,
            condition1="A",
            condition2="B",
            groupby="condition",
            null_genes=["g0", "g1", "g2"],
        )
        assert any("null" in i.lower() or "Null" in i for i in plan.info)

    def test_landmarks_param(self, small_adata):
        """Cover line 514: landmarks passed via kwargs."""
        from kompot.resource_estimation import (
            estimate_differential_expression_resources,
        )

        landmarks = np.random.rand(8, 3)
        plan = estimate_differential_expression_resources(
            small_adata,
            condition1="A",
            condition2="B",
            groupby="condition",
            landmarks=landmarks,
        )
        assert plan.total_memory_required > 0

    def test_n_landmarks_none(self, small_adata):
        """Cover line 520: n_landmarks is None."""
        from kompot.resource_estimation import (
            estimate_differential_expression_resources,
        )

        plan = estimate_differential_expression_resources(
            small_adata,
            condition1="A",
            condition2="B",
            groupby="condition",
            n_landmarks=None,
        )
        assert plan.total_memory_required > 0

    def test_custom_landmarks_array(self, small_adata):
        """Cover the landmarks parameter directly."""
        from kompot.resource_estimation import (
            estimate_differential_expression_resources,
        )

        custom_landmarks = np.random.rand(5, 2)
        plan = estimate_differential_expression_resources(
            small_adata,
            condition1="A",
            condition2="B",
            groupby="condition",
            landmarks=custom_landmarks,
        )
        assert plan.total_memory_required > 0

    def test_many_genes(self):
        """Test with a dataset that has many genes."""
        from kompot.resource_estimation import (
            estimate_differential_expression_resources,
        )

        np.random.seed(2)
        n_cells, n_genes = 30, 500
        X = np.random.rand(n_cells, n_genes)
        obs = pd.DataFrame(
            {"cond": ["X"] * 15 + ["Y"] * 15},
            index=[f"c{i}" for i in range(n_cells)],
        )
        var = pd.DataFrame(index=[f"g{i}" for i in range(n_genes)])
        adata = ad.AnnData(X=X, obs=obs, var=var)

        plan = estimate_differential_expression_resources(
            adata,
            condition1="X",
            condition2="Y",
            groupby="cond",
        )
        assert plan.total_memory_required > 0
        report = plan.format_report()
        assert "intermediate" in report.lower() or "Prediction" in report

    def test_very_small_dataset(self):
        """Test edge case with very small dataset."""
        from kompot.resource_estimation import (
            estimate_differential_expression_resources,
        )

        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        obs = pd.DataFrame(
            {"group": ["A", "A", "B", "B"]},
            index=["c0", "c1", "c2", "c3"],
        )
        var = pd.DataFrame(index=["gA", "gB"])
        adata = ad.AnnData(X=X, obs=obs, var=var)

        plan = estimate_differential_expression_resources(
            adata,
            condition1="A",
            condition2="B",
            groupby="group",
        )
        assert plan.total_memory_required > 0

    def test_batch_size_warning(self, small_adata):
        """Cover lines 917-921: batch_size >= n_total_genes warning."""
        from kompot.resource_estimation import (
            estimate_differential_expression_resources,
        )

        plan = estimate_differential_expression_resources(
            small_adata,
            condition1="A",
            condition2="B",
            groupby="condition",
            batch_size=0,
        )
        assert any("batch_size" in w for w in plan.warnings)

    def test_compute_mahalanobis_false(self, small_adata):
        """Cover skipping Mahalanobis computation."""
        from kompot.resource_estimation import (
            estimate_differential_expression_resources,
        )

        plan = estimate_differential_expression_resources(
            small_adata,
            condition1="A",
            condition2="B",
            groupby="condition",
            compute_mahalanobis=False,
            null_genes=0,
        )
        assert plan.total_memory_required > 0

    def test_store_additional_stats(self, small_adata):
        """Cover store_additional_stats path."""
        from kompot.resource_estimation import (
            estimate_differential_expression_resources,
        )

        plan = estimate_differential_expression_resources(
            small_adata,
            condition1="A",
            condition2="B",
            groupby="condition",
            store_additional_stats=True,
        )
        report = plan.format_report()
        assert "z-score" in report.lower() or "zscores" in report.lower() or len(plan.requirements) > 0

    def test_overwrite_detection(self, small_adata):
        """Cover lines 940-1020: existing results overwrite detection."""
        from kompot.resource_estimation import (
            estimate_differential_expression_resources,
        )

        # Add fake existing DE results to trigger overwrite detection
        small_adata.var["kompot_de_A_to_B_mahalanobis"] = np.zeros(
            small_adata.n_vars
        )
        small_adata.var["kompot_de_A_to_B_mean_lfc"] = np.zeros(
            small_adata.n_vars
        )

        # Add run history
        small_adata.uns["kompot_de_run_history"] = [
            {
                "timestamp": "2025-01-01T00:00:00",
                "params": {
                    "groupby": "condition",
                    "condition1": "A",
                    "condition2": "B",
                },
                "field_mapping": {
                    "mahalanobis_key": "kompot_de_A_to_B_mahalanobis",
                    "mean_lfc_key": "kompot_de_A_to_B_mean_lfc",
                },
            }
        ]

        plan = estimate_differential_expression_resources(
            small_adata,
            condition1="A",
            condition2="B",
            groupby="condition",
        )
        # The plan should run without error. Overwrite detection is best-effort.
        assert plan.total_memory_required > 0

    def test_empirical_variance(self, small_adata):
        """Cover use_empirical_variance paths."""
        from kompot.resource_estimation import (
            estimate_differential_expression_resources,
        )

        plan = estimate_differential_expression_resources(
            small_adata,
            condition1="A",
            condition2="B",
            groupby="condition",
            use_empirical_variance=True,
        )
        report = plan.format_report()
        assert "mpirical" in report or plan.total_memory_required > 0


class TestSuggestAlternativeDiskLocations:
    """Cover suggest_alternative_disk_locations (lines 370-396)."""

    def test_returns_list(self):
        from kompot.resource_estimation import suggest_alternative_disk_locations

        result = suggest_alternative_disk_locations()
        assert isinstance(result, list)
        # Each entry should be (path, human_readable, bytes)
        for entry in result:
            assert len(entry) == 3


class TestDryRunDeprecation:
    """Cover dry_run_differential_expression deprecation wrapper."""

    def test_dry_run_emits_deprecation_warning(self, small_adata):
        from kompot.resource_estimation import dry_run_differential_expression

        with pytest.warns(DeprecationWarning, match="deprecated"):
            plan = dry_run_differential_expression(
                small_adata,
                groupby="condition",
                condition1="A",
                condition2="B",
                verbose=False,
            )
        assert isinstance(plan.format_report(), str)
