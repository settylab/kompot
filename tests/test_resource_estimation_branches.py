"""Tests for kompot/resource_estimation.py targeting uncovered lines.

Covers ResourcePlan, ResourceAvailability, ResourceRequirement,
format_report, check_availability, and utility functions.
"""

import numpy as np


class TestResourceEstimation:
    """Tests for resource_estimation targeting uncovered lines."""

    def test_format_report_with_output_fields_and_overwrites(self):
        """Lines 274-278, 295-297, 302: format_report with output fields and overwrites."""
        from kompot.resource_estimation import ResourcePlan, ResourceAvailability

        plan = ResourcePlan()
        plan.availability = ResourceAvailability(
            memory_total=16 * 1024**3,
            memory_available=8 * 1024**3,
            disk_path="/tmp",
            disk_total=100 * 1024**3,
            disk_available=50 * 1024**3,
        )
        plan.add_requirement("Test array", 1024**2, "memory", shape=(100, 100))
        plan.add_requirement(
            "Disk array", 1024**2, "disk", shape=(100, 100), overwrite=True
        )
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
        plan.add_requirement(
            "Cov matrix", 2 * 1024**3, "disk", shape=(1000, 1000), overwrite=True
        )
        report = plan.format_report(verbose=True)
        assert "Disk Storage" in report
        assert "OVERWRITE" in report

    def test_resource_plan_no_availability(self):
        """Lines 107-108, 114-115: ratios with no availability."""
        from kompot.resource_estimation import ResourcePlan

        plan = ResourcePlan()
        plan.add_requirement("Test", 1024, "memory")
        assert plan.memory_ratio == float("inf")
        assert plan.disk_ratio == float("inf")


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
            memory_total=0,
            memory_available=0,
            disk_path="/tmp",
            disk_total=0,
            disk_available=0,
        )
        plan.add_requirement("T", 1024, "memory")
        assert plan.memory_ratio == float("inf")

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


class TestResourceEstimationDeepCoverage:
    """Deeper coverage for resource_estimation.py."""

    def test_check_availability_insufficient_memory(self):
        """Lines 153-158: insufficient memory triggers error."""
        from kompot.resource_estimation import ResourcePlan, ResourceAvailability

        plan = ResourcePlan()
        plan.availability = ResourceAvailability(
            memory_total=1024,
            memory_available=512,
            disk_path="/tmp",
            disk_total=100 * 1024**3,
            disk_available=50 * 1024**3,
        )
        plan.add_requirement("Big array", 1024, "memory")  # > 512 available
        plan.check_availability()
        assert any("Insufficient memory" in e for e in plan.errors)

    def test_check_availability_high_memory_warning(self):
        """Lines 159-165: high memory usage generates warning."""
        from kompot.resource_estimation import ResourcePlan, ResourceAvailability

        plan = ResourcePlan()
        plan.availability = ResourceAvailability(
            memory_total=1024**3,
            memory_available=1024**3,
            disk_path="/tmp",
            disk_total=100 * 1024**3,
            disk_available=50 * 1024**3,
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
            memory_total=16 * 1024**3,
            memory_available=8 * 1024**3,
            disk_path="/tmp",
            disk_total=1024,
            disk_available=512,
        )
        plan.add_requirement("Huge file", 1024, "disk")
        plan.check_availability()
        assert any("Insufficient disk" in e for e in plan.errors)

    def test_check_availability_high_disk_warning(self):
        """Lines 196-202: high disk usage generates warning."""
        from kompot.resource_estimation import ResourcePlan, ResourceAvailability

        plan = ResourcePlan()
        plan.availability = ResourceAvailability(
            memory_total=16 * 1024**3,
            memory_available=8 * 1024**3,
            disk_path="/tmp",
            disk_total=1024**3,
            disk_available=1024**3,
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
