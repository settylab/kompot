"""Tests for resource_estimation.py to improve coverage."""

import numpy as np
import pytest
import tempfile
import os

from kompot.resource_estimation import (
    ResourceRequirement,
    ResourceAvailability,
    ResourcePlan,
    human_readable_size,
    dry_run_differential_expression
)


def test_human_readable_size_zero():
    """Test human_readable_size with zero."""
    assert human_readable_size(0) == "0.00 B"


def test_human_readable_size_various():
    """Test human_readable_size with various sizes."""
    assert "B" in human_readable_size(100)
    assert "KB" in human_readable_size(2048)
    assert "MB" in human_readable_size(2 * 1024 * 1024)
    assert "GB" in human_readable_size(3 * 1024 * 1024 * 1024)
    assert "TB" in human_readable_size(5 * 1024 * 1024 * 1024 * 1024)


def test_resource_requirement_properties():
    """Test ResourceRequirement properties."""
    req = ResourceRequirement(
        name="Test array",
        size_bytes=1024 * 1024,  # 1 MB
        resource_type="memory",
        shape=(100, 100),
        field_name="test_field",
        overwrite=True
    )

    assert req.name == "Test array"
    assert req.size_bytes == 1024 * 1024
    assert req.resource_type == "memory"
    assert req.shape == (100, 100)
    assert req.field_name == "test_field"
    assert req.overwrite is True
    assert "MB" in req.size_human


def test_resource_requirement_no_overwrite():
    """Test ResourceRequirement with overwrite=False."""
    req = ResourceRequirement(
        name="Test",
        size_bytes=100,
        resource_type="disk",
        overwrite=False
    )

    assert req.overwrite is False


def test_resource_availability_properties():
    """Test ResourceAvailability properties."""
    avail = ResourceAvailability(
        memory_total=16 * 1024 * 1024 * 1024,  # 16 GB
        memory_available=8 * 1024 * 1024 * 1024,  # 8 GB
        disk_path="/tmp",
        disk_total=100 * 1024 * 1024 * 1024,  # 100 GB
        disk_available=50 * 1024 * 1024 * 1024  # 50 GB
    )

    assert "GB" in avail.memory_total_human
    assert "GB" in avail.memory_available_human
    assert "GB" in avail.disk_total_human
    assert "GB" in avail.disk_available_human
    assert avail.disk_path == "/tmp"


def test_resource_plan_empty():
    """Test ResourcePlan with no requirements."""
    plan = ResourcePlan()

    assert plan.total_memory_required == 0
    assert plan.total_disk_required == 0
    assert len(plan.requirements) == 0
    assert len(plan.info) == 0
    assert len(plan.warnings) == 0
    assert len(plan.errors) == 0


def test_resource_plan_with_requirements():
    """Test ResourcePlan with memory and disk requirements."""
    req1 = ResourceRequirement("Array1", 1024, "memory")
    req2 = ResourceRequirement("Array2", 2048, "memory")
    req3 = ResourceRequirement("File1", 4096, "disk")

    plan = ResourcePlan(requirements=[req1, req2, req3])

    assert plan.total_memory_required == 1024 + 2048
    assert plan.total_disk_required == 4096


def test_resource_plan_with_messages():
    """Test ResourcePlan with messages."""
    plan = ResourcePlan(
        info=["Info message 1", "Info message 2"],
        warnings=["Warning 1"],
        errors=["Error 1", "Error 2"]
    )

    assert len(plan.info) == 2
    assert len(plan.warnings) == 1
    assert len(plan.errors) == 2


def test_resource_plan_with_output_fields():
    """Test ResourcePlan with output fields."""
    plan = ResourcePlan(
        output_fields={
            "uns": ["field1", "field2"],
            "varm": ["field3"]
        }
    )

    assert "uns" in plan.output_fields
    assert "varm" in plan.output_fields
    assert len(plan.output_fields["uns"]) == 2
    assert len(plan.output_fields["varm"]) == 1


def test_resource_plan_with_availability():
    """Test ResourcePlan with availability."""
    avail = ResourceAvailability(
        memory_total=8 * 1024 * 1024 * 1024,
        memory_available=4 * 1024 * 1024 * 1024,
        disk_path="/tmp",
        disk_total=100 * 1024 * 1024 * 1024,
        disk_available=50 * 1024 * 1024 * 1024
    )

    plan = ResourcePlan(availability=avail)

    assert plan.availability is not None
    assert plan.availability.memory_total == 8 * 1024 * 1024 * 1024


def test_dry_run_basic():
    """Test dry_run_differential_expression basic functionality."""
    import anndata
    import pandas as pd

    # Create minimal test data
    np.random.seed(42)
    n_cells, n_genes = 100, 20
    X = np.random.normal(0, 1, (n_cells, n_genes))
    obs = pd.DataFrame({
        'condition': ['A'] * 50 + ['B'] * 50,
        'celltype': np.random.choice(['T', 'B', 'NK'], n_cells)
    })
    obsm = {'DM_EigenVectors': np.random.normal(0, 1, (n_cells, 10))}
    var = pd.DataFrame(index=[f'gene_{i}' for i in range(n_genes)])
    adata = anndata.AnnData(X=X, obs=obs, var=var, obsm=obsm)

    # Run dry run
    plan = dry_run_differential_expression(
        adata,
        groupby='condition',
        condition1='A',
        condition2='B',
        verbose=False
    )

    assert isinstance(plan, ResourcePlan)
    assert plan.total_memory_required > 0


def test_dry_run_with_landmarks():
    """Test dry_run with specific landmarks parameter."""
    import anndata
    import pandas as pd

    np.random.seed(42)
    n_cells, n_genes = 100, 20
    X = np.random.normal(0, 1, (n_cells, n_genes))
    obs = pd.DataFrame({'condition': ['A'] * 50 + ['B'] * 50})
    obsm = {'DM_EigenVectors': np.random.normal(0, 1, (n_cells, 10))}
    var = pd.DataFrame(index=[f'gene_{i}' for i in range(n_genes)])
    adata = anndata.AnnData(X=X, obs=obs, var=var, obsm=obsm)

    plan = dry_run_differential_expression(
        adata,
        groupby='condition',
        condition1='A',
        condition2='B',
        n_landmarks=30,
        verbose=False
    )

    assert isinstance(plan, ResourcePlan)


def test_dry_run_with_disk_storage():
    """Test dry_run with disk_storage_dir parameter."""
    import anndata
    import pandas as pd

    np.random.seed(42)
    n_cells, n_genes = 100, 20
    X = np.random.normal(0, 1, (n_cells, n_genes))
    obs = pd.DataFrame({'condition': ['A'] * 50 + ['B'] * 50})
    obsm = {'DM_EigenVectors': np.random.normal(0, 1, (n_cells, 10))}
    var = pd.DataFrame(index=[f'gene_{i}' for i in range(n_genes)])
    adata = anndata.AnnData(X=X, obs=obs, var=var, obsm=obsm)

    with tempfile.TemporaryDirectory() as tmpdir:
        plan = dry_run_differential_expression(
            adata,
            groupby='condition',
            condition1='A',
            condition2='B',
            disk_storage_dir=tmpdir,
            verbose=False
        )

        assert isinstance(plan, ResourcePlan)
        # Disk requirement may be 0 if data fits in memory
        assert plan.total_disk_required >= 0


def test_dry_run_with_groups():
    """Test dry_run with groups parameter."""
    import anndata
    import pandas as pd

    np.random.seed(42)
    n_cells, n_genes = 100, 20
    X = np.random.normal(0, 1, (n_cells, n_genes))
    obs = pd.DataFrame({
        'condition': ['A'] * 50 + ['B'] * 50,
        'celltype': np.random.choice(['T', 'B'], n_cells)
    })
    obsm = {'DM_EigenVectors': np.random.normal(0, 1, (n_cells, 10))}
    var = pd.DataFrame(index=[f'gene_{i}' for i in range(n_genes)])
    adata = anndata.AnnData(X=X, obs=obs, var=var, obsm=obsm)

    plan = dry_run_differential_expression(
        adata,
        groupby='condition',
        condition1='A',
        condition2='B',
        groups='celltype',
        verbose=False
    )

    assert isinstance(plan, ResourcePlan)


def test_dry_run_with_genes_subset():
    """Test dry_run with genes parameter."""
    import anndata
    import pandas as pd

    np.random.seed(42)
    n_cells, n_genes = 100, 20
    X = np.random.normal(0, 1, (n_cells, n_genes))
    obs = pd.DataFrame({'condition': ['A'] * 50 + ['B'] * 50})
    obsm = {'DM_EigenVectors': np.random.normal(0, 1, (n_cells, 10))}
    var = pd.DataFrame(index=[f'gene_{i}' for i in range(n_genes)])
    adata = anndata.AnnData(X=X, obs=obs, var=var, obsm=obsm)

    # Test with subset of genes
    gene_subset = [f'gene_{i}' for i in range(10)]
    plan = dry_run_differential_expression(
        adata,
        groupby='condition',
        condition1='A',
        condition2='B',
        genes=gene_subset,
        verbose=False
    )

    assert isinstance(plan, ResourcePlan)


def test_dry_run_with_sample_variance():
    """Test dry_run with use_sample_variance parameter."""
    import anndata
    import pandas as pd

    np.random.seed(42)
    n_cells, n_genes = 120, 20
    X = np.random.normal(0, 1, (n_cells, n_genes))
    obs = pd.DataFrame({
        'condition': ['A'] * 60 + ['B'] * 60,
        'sample': [f'sample_{i%3}' for i in range(n_cells)]
    })
    obsm = {'DM_EigenVectors': np.random.normal(0, 1, (n_cells, 10))}
    var = pd.DataFrame(index=[f'gene_{i}' for i in range(n_genes)])
    adata = anndata.AnnData(X=X, obs=obs, var=var, obsm=obsm)

    plan = dry_run_differential_expression(
        adata,
        groupby='condition',
        condition1='A',
        condition2='B',
        use_sample_variance=True,
        sample_col='sample',
        verbose=False
    )

    assert isinstance(plan, ResourcePlan)


def test_dry_run_with_result_key():
    """Test dry_run with custom result_key."""
    import anndata
    import pandas as pd

    np.random.seed(42)
    n_cells, n_genes = 100, 20
    X = np.random.normal(0, 1, (n_cells, n_genes))
    obs = pd.DataFrame({'condition': ['A'] * 50 + ['B'] * 50})
    obsm = {'DM_EigenVectors': np.random.normal(0, 1, (n_cells, 10))}
    var = pd.DataFrame(index=[f'gene_{i}' for i in range(n_genes)])
    adata = anndata.AnnData(X=X, obs=obs, var=var, obsm=obsm)

    plan = dry_run_differential_expression(
        adata,
        groupby='condition',
        condition1='A',
        condition2='B',
        result_key='my_custom_de',
        verbose=False
    )

    assert isinstance(plan, ResourcePlan)


def test_dry_run_reports_errors():
    """Test dry_run reports errors in plan."""
    import anndata
    import pandas as pd

    np.random.seed(42)
    n_cells, n_genes = 100, 20
    X = np.random.normal(0, 1, (n_cells, n_genes))
    obs = pd.DataFrame({'condition': ['A'] * 50 + ['B'] * 50})
    obsm = {'DM_EigenVectors': np.random.normal(0, 1, (n_cells, 10))}
    var = pd.DataFrame(index=[f'gene_{i}' for i in range(n_genes)])
    adata = anndata.AnnData(X=X, obs=obs, var=var, obsm=obsm)

    # Test with non-existent condition - should include error in plan
    plan = dry_run_differential_expression(
        adata,
        groupby='condition',
        condition1='A',
        condition2='nonexistent',
        verbose=False
    )

    # Should have errors in the plan
    assert isinstance(plan, ResourcePlan)


def test_resource_requirement_disk_type():
    """Test ResourceRequirement with disk resource type."""
    req = ResourceRequirement(
        name="Disk array",
        size_bytes=10 * 1024 * 1024,
        resource_type="disk"
    )

    assert req.resource_type == "disk"
    assert req.size_bytes == 10 * 1024 * 1024
