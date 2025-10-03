"""Tests for resource estimation utilities."""

import numpy as np
import pytest
import anndata as ad


def test_human_readable_size():
    """Test human readable size formatting."""
    from kompot.resource_estimation import human_readable_size

    assert human_readable_size(0) == "0.00 B"
    assert human_readable_size(1023) == "1023.00 B"
    assert human_readable_size(1024) == "1.00 KB"
    assert human_readable_size(1024**2) == "1.00 MB"
    assert human_readable_size(1024**3) == "1.00 GB"
    assert human_readable_size(1024**4) == "1.00 TB"
    assert human_readable_size(1536) == "1.50 KB"  # 1.5 KB


def test_estimate_array_size():
    """Test array size estimation."""
    from kompot.resource_estimation import estimate_array_size

    # 100x50 float64 array
    size = estimate_array_size((100, 50), dtype=np.float64)
    assert size == 100 * 50 * 8  # 8 bytes per float64

    # 1000x1000x100 float64 array
    size = estimate_array_size((1000, 1000, 100), dtype=np.float64)
    assert size == 1000 * 1000 * 100 * 8


def test_resource_requirement():
    """Test ResourceRequirement dataclass."""
    from kompot.resource_estimation import ResourceRequirement

    req = ResourceRequirement(
        name="Test array",
        size_bytes=1024**2,
        resource_type='memory',
        shape=(100, 100),
        field_name='test_field'
    )

    assert req.name == "Test array"
    assert req.size_bytes == 1024**2
    assert req.size_human == "1.00 MB"
    assert req.resource_type == 'memory'
    assert req.shape == (100, 100)
    assert req.field_name == 'test_field'


def test_resource_plan_basic():
    """Test basic ResourcePlan functionality."""
    from kompot.resource_estimation import ResourcePlan, get_system_resources

    plan = ResourcePlan()
    plan.availability = get_system_resources()

    # Add memory requirement
    plan.add_requirement(
        "Test memory",
        1024**3,  # 1 GB
        'memory',
        shape=(1000, 1000, 100)
    )

    # Add disk requirement
    plan.add_requirement(
        "Test disk",
        1024**2 * 100,  # 100 MB
        'disk'
    )

    assert len(plan.requirements) == 2
    assert plan.total_memory_required == 1024**3
    assert plan.total_disk_required == 1024**2 * 100

    # Check availability
    plan.check_availability()

    # Should be feasible for these small sizes
    assert plan.is_feasible


def test_resource_plan_insufficient_memory():
    """Test resource plan with insufficient memory."""
    from kompot.resource_estimation import ResourcePlan, ResourceAvailability

    plan = ResourcePlan()

    # Create fake availability with very little memory
    plan.availability = ResourceAvailability(
        memory_total=1024**3,  # 1 GB total
        memory_available=512 * 1024**2,  # 512 MB available
        disk_path="/tmp",
        disk_total=100 * 1024**3,  # 100 GB
        disk_available=50 * 1024**3  # 50 GB
    )

    # Request more memory than available
    plan.add_requirement(
        "Huge array",
        1024**3,  # 1 GB
        'memory'
    )

    plan.check_availability()

    # Should have errors
    assert not plan.is_feasible
    assert len(plan.errors) > 0
    assert "Insufficient memory" in plan.errors[0]


def test_dry_run_differential_expression():
    """Test dry run for differential expression."""
    from kompot.resource_estimation import dry_run_differential_expression

    # Create simple test data
    np.random.seed(42)
    n_cells = 100
    n_genes = 50

    X = np.random.randn(n_cells, n_genes)  # Fixed: match X columns to n_genes
    var_names = [f"gene_{i}" for i in range(n_genes)]
    obs_data = {
        'condition': ['A'] * 50 + ['B'] * 50,
        'sample': [f's{i//10}' for i in range(n_cells)]
    }

    adata = ad.AnnData(
        X=X,
        var={'gene_ids': var_names},
        obs=obs_data
    )

    # Run dry run
    plan = dry_run_differential_expression(
        adata,
        condition1='A',
        condition2='B',
        groupby='condition',
        verbose=False
    )

    assert plan is not None
    assert plan.availability is not None
    assert len(plan.requirements) > 0
    assert plan.total_memory_required > 0


def test_dry_run_with_sample_variance():
    """Test dry run with sample variance enabled."""
    from kompot.resource_estimation import dry_run_differential_expression

    # Create test data
    np.random.seed(42)
    n_cells = 200
    n_genes = 100

    X = np.random.randn(n_cells, n_genes)  # Fixed
    var_names = [f"gene_{i}" for i in range(n_genes)]
    obs_data = {
        'condition': ['treated'] * 100 + ['control'] * 100,
        'donor_id': [f'donor{i//20}' for i in range(n_cells)]
    }

    adata = ad.AnnData(
        X=X,
        var={'gene_ids': var_names},
        obs=obs_data
    )

    # Run dry run with sample variance
    plan = dry_run_differential_expression(
        adata,
        condition1='treated',
        condition2='control',
        groupby='condition',
        use_sample_variance=True,
        sample_column='donor_id',
        verbose=False
    )

    assert plan is not None

    # Should have covariance matrix requirements
    cov_reqs = [r for r in plan.requirements if 'covariance' in r.name.lower() or 'variance' in r.name.lower()]
    assert len(cov_reqs) > 0, f"Expected covariance requirements, got: {[r.name for r in plan.requirements]}"

    # Should have sample covariance specific requirements
    sv_reqs = [r for r in plan.requirements if 'Sample covariances' in r.name]
    assert len(sv_reqs) > 0, "Expected sample covariance requirements"

    # Should warn about memory usage if not using disk storage
    assert any('disk_storage_dir' in w or 'memory' in w or 'covariance' in w.lower() for w in plan.warnings)


def test_dry_run_with_disk_storage():
    """Test dry run with disk storage enabled."""
    from kompot.resource_estimation import dry_run_differential_expression
    import tempfile

    # Create test data
    np.random.seed(42)
    n_genes = 50
    X = np.random.randn(100, n_genes)  # Fixed
    obs_data = {
        'condition': ['A'] * 50 + ['B'] * 50,
        'sample': [f's{i//10}' for i in range(100)]
    }

    adata = ad.AnnData(X=X, obs=obs_data)

    with tempfile.TemporaryDirectory() as tmpdir:
        plan = dry_run_differential_expression(
            adata,
            condition1='A',
            condition2='B',
            groupby='condition',
            use_sample_variance=True,
            sample_column='sample',
            store_arrays_on_disk=True,
            disk_storage_dir=tmpdir,
            verbose=False
        )

        # Should have disk requirements
        disk_reqs = [r for r in plan.requirements if r.resource_type == 'disk']
        assert len(disk_reqs) > 0


def test_format_report():
    """Test resource plan report formatting."""
    from kompot.resource_estimation import ResourcePlan, get_system_resources

    plan = ResourcePlan()
    plan.availability = get_system_resources()

    plan.add_requirement("Memory array", 1024**2, 'memory', shape=(100, 100))
    plan.add_requirement("Disk array", 1024**3, 'disk')
    plan.warnings.append("Test warning")

    report = plan.format_report(verbose=True)

    assert "RESOURCE USAGE PLAN" in report
    assert "System Resources:" in report
    assert "Memory array" in report
    assert "Disk array" in report
    assert "Test warning" in report
    assert "FEASIBLE" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
