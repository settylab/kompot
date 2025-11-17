"""Comprehensive tests for resource_estimation.py to improve coverage."""

import numpy as np
import pytest
import pandas as pd
import anndata
import tempfile


def create_resource_test_data(n_cells=100, n_genes=50, with_layer=False, with_groups=True):
    """Create test data for resource estimation."""
    np.random.seed(42)

    X = np.random.normal(0, 1, (n_cells, n_genes))
    X[X < 0] = 0

    obs_data = {
        'condition': ['A'] * (n_cells // 2) + ['B'] * (n_cells // 2),
    }

    if with_groups:
        obs_data['celltype'] = np.random.choice(['T', 'B', 'NK'], n_cells)
        obs_data['sample'] = [f'sample_{i%3}' for i in range(n_cells)]

    obs = pd.DataFrame(obs_data)
    obsm = {'DM_EigenVectors': np.random.normal(0, 1, (n_cells, 10))}
    var = pd.DataFrame(index=[f'gene_{i}' for i in range(n_genes)])

    layers = {}
    if with_layer:
        layers['raw'] = X.copy() * 1.5

    return anndata.AnnData(X=X, obs=obs, var=var, obsm=obsm, layers=layers if with_layer else None)


def test_dry_run_verbose_mode():
    """Test dry_run with verbose=True."""
    from kompot import dry_run_differential_expression

    adata = create_resource_test_data()

    plan = dry_run_differential_expression(
        adata,
        groupby='condition',
        condition1='A',
        condition2='B',
        verbose=True  # Enable verbose output
    )

    assert plan is not None
    assert hasattr(plan, 'total_memory_required')


def test_dry_run_with_large_dataset():
    """Test dry_run with larger dataset to trigger different memory paths."""
    from kompot import dry_run_differential_expression

    adata = create_resource_test_data(n_cells=500, n_genes=200)

    plan = dry_run_differential_expression(
        adata,
        groupby='condition',
        condition1='A',
        condition2='B',
        n_landmarks=100,
        verbose=False
    )

    assert plan.total_memory_required > 0


def test_dry_run_with_fdr():
    """Test dry_run with FDR enabled."""
    from kompot import dry_run_differential_expression

    adata = create_resource_test_data(n_cells=100, n_genes=80)

    plan = dry_run_differential_expression(
        adata,
        groupby='condition',
        condition1='A',
        condition2='B',
        null_genes=40,  # Enable FDR
        fdr_threshold=0.1,
        verbose=False
    )

    assert plan is not None


def test_dry_run_with_store_options():
    """Test dry_run with various storage options."""
    from kompot import dry_run_differential_expression

    adata = create_resource_test_data()

    plan = dry_run_differential_expression(
        adata,
        groupby='condition',
        condition1='A',
        condition2='B',
        store_landmarks=True,
        store_posterior_covariance=True,
        store_additional_stats=True,
        verbose=False
    )

    assert plan.total_memory_required > 0


def test_dry_run_with_custom_result_key():
    """Test dry_run with custom result_key."""
    from kompot import dry_run_differential_expression

    adata = create_resource_test_data()

    plan = dry_run_differential_expression(
        adata,
        groupby='condition',
        condition1='A',
        condition2='B',
        result_key='my_custom_key',
        verbose=False
    )

    assert plan is not None


def test_dry_run_with_cell_filter():
    """Test dry_run with cell_filter parameter."""
    from kompot import dry_run_differential_expression

    adata = create_resource_test_data()
    adata.obs['use_cell'] = np.random.choice([True, False], adata.n_obs)

    plan = dry_run_differential_expression(
        adata,
        groupby='condition',
        condition1='A',
        condition2='B',
        cell_filter='use_cell',
        verbose=False
    )

    assert plan is not None


def test_dry_run_with_groups_column():
    """Test dry_run with groups as column name."""
    from kompot import dry_run_differential_expression

    adata = create_resource_test_data()

    plan = dry_run_differential_expression(
        adata,
        groupby='condition',
        condition1='A',
        condition2='B',
        groups='celltype',
        verbose=False
    )

    assert plan is not None


def test_dry_run_with_layer_parameter():
    """Test dry_run with layer parameter."""
    from kompot import dry_run_differential_expression

    adata = create_resource_test_data(with_layer=True)

    plan = dry_run_differential_expression(
        adata,
        groupby='condition',
        condition1='A',
        condition2='B',
        layer='raw',
        verbose=False
    )

    assert plan is not None


def test_dry_run_with_custom_landmarks_array():
    """Test dry_run with custom landmarks array."""
    from kompot import dry_run_differential_expression

    adata = create_resource_test_data()

    # Provide custom landmarks coordinates
    landmarks = adata.obsm['DM_EigenVectors'][:30]

    plan = dry_run_differential_expression(
        adata,
        groupby='condition',
        condition1='A',
        condition2='B',
        landmarks=landmarks,
        verbose=False
    )

    assert plan is not None


def test_dry_run_with_sample_variance():
    """Test dry_run with sample variance enabled."""
    from kompot import dry_run_differential_expression

    adata = create_resource_test_data(n_cells=120, with_groups=True)

    plan = dry_run_differential_expression(
        adata,
        groupby='condition',
        condition1='A',
        condition2='B',
        use_sample_variance=True,
        sample_col='sample',
        verbose=False
    )

    assert plan is not None


def test_dry_run_with_min_cells():
    """Test dry_run with min_cells filtering."""
    from kompot import dry_run_differential_expression

    adata = create_resource_test_data()

    plan = dry_run_differential_expression(
        adata,
        groupby='condition',
        condition1='A',
        condition2='B',
        min_cells=10,
        verbose=False
    )

    assert plan is not None


def test_dry_run_with_min_percentage():
    """Test dry_run with min_percentage filtering."""
    from kompot import dry_run_differential_expression

    adata = create_resource_test_data()

    plan = dry_run_differential_expression(
        adata,
        groupby='condition',
        condition1='A',
        condition2='B',
        min_percentage=0.1,
        verbose=False
    )

    assert plan is not None


def test_dry_run_with_custom_params():
    """Test dry_run with custom ls, sigma, batch_size."""
    from kompot import dry_run_differential_expression

    adata = create_resource_test_data()

    plan = dry_run_differential_expression(
        adata,
        groupby='condition',
        condition1='A',
        condition2='B',
        ls=0.5,
        sigma=2.5,
        batch_size=100,
        eps=1e-10,
        verbose=False
    )

    assert plan is not None


def test_dry_run_compute_mahalanobis_false():
    """Test dry_run with compute_mahalanobis=False."""
    from kompot import dry_run_differential_expression

    adata = create_resource_test_data()

    plan = dry_run_differential_expression(
        adata,
        groupby='condition',
        condition1='A',
        condition2='B',
        compute_mahalanobis=False,
        verbose=False
    )

    assert plan is not None


def test_dry_run_with_invalid_condition():
    """Test dry_run handles invalid conditions gracefully."""
    from kompot import dry_run_differential_expression

    adata = create_resource_test_data()

    # Should not raise, but may include errors in plan
    plan = dry_run_differential_expression(
        adata,
        groupby='condition',
        condition1='A',
        condition2='nonexistent',
        verbose=False
    )

    # Plan should be returned even with errors
    assert plan is not None


def test_dry_run_with_nonexistent_groupby():
    """Test dry_run with non-existent groupby column."""
    from kompot import dry_run_differential_expression

    adata = create_resource_test_data()

    # Should raise an error for non-existent groupby
    with pytest.raises((ValueError, KeyError)):
        plan = dry_run_differential_expression(
            adata,
            groupby='nonexistent_column',
            condition1='A',
            condition2='B',
            verbose=False
        )


def test_dry_run_checks_output_fields():
    """Test that dry_run properly identifies output fields."""
    from kompot import dry_run_differential_expression

    adata = create_resource_test_data()

    plan = dry_run_differential_expression(
        adata,
        groupby='condition',
        condition1='A',
        condition2='B',
        store_landmarks=True,
        verbose=False
    )

    # Should have output_fields information
    assert hasattr(plan, 'output_fields')


def test_dry_run_disk_storage_with_path():
    """Test dry_run with disk_storage_dir to estimate disk usage."""
    from kompot import dry_run_differential_expression

    adata = create_resource_test_data(n_cells=200, n_genes=100)

    with tempfile.TemporaryDirectory() as tmpdir:
        plan = dry_run_differential_expression(
            adata,
            groupby='condition',
            condition1='A',
            condition2='B',
            disk_storage_dir=tmpdir,
            verbose=False
        )

        # Should estimate disk requirements
        assert plan.total_disk_required >= 0


def test_dry_run_with_availability_info():
    """Test that dry_run includes resource availability information."""
    from kompot import dry_run_differential_expression

    adata = create_resource_test_data()

    plan = dry_run_differential_expression(
        adata,
        groupby='condition',
        condition1='A',
        condition2='B',
        verbose=True  # Verbose mode may include availability
    )

    # Plan should have availability attribute
    assert hasattr(plan, 'availability')


def test_dry_run_with_small_dataset():
    """Test dry_run with very small dataset."""
    from kompot import dry_run_differential_expression

    adata = create_resource_test_data(n_cells=20, n_genes=10)

    plan = dry_run_differential_expression(
        adata,
        groupby='condition',
        condition1='A',
        condition2='B',
        n_landmarks=10,
        verbose=False
    )

    assert plan.total_memory_required > 0


def test_dry_run_with_many_genes():
    """Test dry_run with many genes to see memory scaling."""
    from kompot import dry_run_differential_expression

    adata = create_resource_test_data(n_cells=100, n_genes=500)

    plan = dry_run_differential_expression(
        adata,
        groupby='condition',
        condition1='A',
        condition2='B',
        verbose=False
    )

    # Memory requirements should scale with gene count
    assert plan.total_memory_required > 0


def test_dry_run_requirement_details():
    """Test that dry_run provides detailed requirement breakdown."""
    from kompot import dry_run_differential_expression

    adata = create_resource_test_data()

    plan = dry_run_differential_expression(
        adata,
        groupby='condition',
        condition1='A',
        condition2='B',
        verbose=False
    )

    # Plan should have requirements list
    assert hasattr(plan, 'requirements')


def test_dry_run_with_inplace_false():
    """Test dry_run with inplace=False."""
    from kompot import dry_run_differential_expression

    adata = create_resource_test_data()

    plan = dry_run_differential_expression(
        adata,
        groupby='condition',
        condition1='A',
        condition2='B',
        inplace=False,
        verbose=False
    )

    assert plan is not None


def test_dry_run_with_copy_true():
    """Test dry_run with copy=True."""
    from kompot import dry_run_differential_expression

    adata = create_resource_test_data()

    plan = dry_run_differential_expression(
        adata,
        groupby='condition',
        condition1='A',
        condition2='B',
        copy=True,
        verbose=False
    )

    assert plan is not None


def test_dry_run_with_random_state():
    """Test dry_run with random_state parameter."""
    from kompot import dry_run_differential_expression

    adata = create_resource_test_data()

    plan = dry_run_differential_expression(
        adata,
        groupby='condition',
        condition1='A',
        condition2='B',
        random_state=42,
        verbose=False
    )

    assert plan is not None


def test_dry_run_with_allow_single_condition_variance():
    """Test dry_run with allow_single_condition_variance."""
    from kompot import dry_run_differential_expression

    adata = create_resource_test_data()

    plan = dry_run_differential_expression(
        adata,
        groupby='condition',
        condition1='A',
        condition2='B',
        allow_single_condition_variance=True,
        verbose=False
    )

    assert plan is not None


def test_dry_run_messages():
    """Test that dry_run can produce info, warnings, and errors."""
    from kompot import dry_run_differential_expression

    adata = create_resource_test_data()

    plan = dry_run_differential_expression(
        adata,
        groupby='condition',
        condition1='A',
        condition2='B',
        verbose=False
    )

    # Plan should have message lists
    assert hasattr(plan, 'info')
    assert hasattr(plan, 'warnings')
    assert hasattr(plan, 'errors')
