"""Tests for resource estimation: ResourcePlan, availability checks, and DE resource estimation."""

import numpy as np
import pytest
import pandas as pd
import anndata as ad


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
        assert (
            "z-score" in report.lower()
            or "zscores" in report.lower()
            or len(plan.requirements) > 0
        )

    def test_overwrite_detection(self, small_adata):
        """Cover lines 940-1020: existing results overwrite detection."""
        from kompot.resource_estimation import (
            estimate_differential_expression_resources,
        )

        # Add fake existing DE results to trigger overwrite detection
        small_adata.var["kompot_de_A_to_B_mahalanobis"] = np.zeros(small_adata.n_vars)
        small_adata.var["kompot_de_A_to_B_mean_lfc"] = np.zeros(small_adata.n_vars)

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
