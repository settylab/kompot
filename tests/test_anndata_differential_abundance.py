"""Advanced tests for the anndata.differential_abundance module.

This file extends the basic tests in test_anndata_functions.py with more detailed
tests focusing on specific features of compute_differential_abundance.
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from kompot.anndata.differential_abundance import compute_differential_abundance


def create_test_anndata(n_cells=100, n_genes=20, with_samples=False):
    """Create a test AnnData object."""
    try:
        import anndata
    except ImportError:
        pytest.skip("anndata not installed, skipping test")
        
    np.random.seed(42)
    
    # Create test data
    X = np.random.normal(0, 1, (n_cells, n_genes))
    
    # Create cell groups for testing
    groups = np.array(['A'] * (n_cells // 2) + ['B'] * (n_cells // 2))
    
    # Create embedding
    obsm = {
        'DM_EigenVectors': np.random.normal(0, 1, (n_cells, 10)),
        'X_pca': np.random.normal(0, 1, (n_cells, 2))
    }
    
    # Create observation dataframe
    obs = pd.DataFrame({
        'group': groups
    })
    
    # Add sample IDs if requested
    if with_samples:
        samples_A = np.repeat(['S1', 'S2', 'S3'], n_cells // 6)
        samples_B = np.repeat(['S4', 'S5', 'S6'], n_cells // 6)
        obs['sample'] = np.concatenate([samples_A, samples_B])
    
    # Create var_names
    var_names = [f'gene_{i}' for i in range(n_genes)]
    
    # Create var DataFrame with var_names as index
    var = pd.DataFrame(index=var_names)
    
    return anndata.AnnData(X=X, obs=obs, var=var, obsm=obsm)


def test_compute_differential_abundance_with_landmarks():
    """Test compute_differential_abundance function with explicit n_landmarks."""
    # Create test data
    adata = create_test_anndata()
    
    # Run differential abundance analysis with landmarks
    result = compute_differential_abundance(
        adata,
        groupby='group',
        condition1='A',
        condition2='B',
        n_landmarks=20,
        store_landmarks=True,
        return_full_results=True
    )
    
    # Check that results are returned
    assert result is not None
    assert isinstance(result, dict)
    
    # Check that landmarks are stored
    assert 'kompot_da' in adata.uns
    assert 'landmarks_info' in adata.uns['kompot_da']
    assert adata.uns['kompot_da']['landmarks_info']['n_landmarks'] == 20
    
    # Check that landmarks are actually stored when store_landmarks=True
    assert 'landmarks' in result, "Landmarks should be in result dictionary"
    assert 'kompot_da' in adata.uns
    assert 'landmarks' in adata.uns['kompot_da']
    assert adata.uns['kompot_da']['landmarks'].shape[0] == 20
    
    # Test setting n_landmarks but not storing them
    adata2 = create_test_anndata()
    result2 = compute_differential_abundance(
        adata2,
        groupby='group',
        condition1='A',
        condition2='B',
        n_landmarks=15,
        store_landmarks=False,
        return_full_results=True
    )
    
    # Check that landmarks are not stored when store_landmarks=False
    assert 'landmarks' in result2, "Landmarks should be in result dictionary"
    assert 'kompot_da' in adata2.uns
    assert 'landmarks_info' in adata2.uns['kompot_da']
    assert 'landmarks' not in adata2.uns['kompot_da']


def test_compute_differential_abundance_with_custom_ls_factor():
    """Test compute_differential_abundance function with custom ls_factor."""
    # Create test data
    adata = create_test_anndata()
    
    # Run differential abundance analysis with custom ls_factor
    result = compute_differential_abundance(
        adata,
        groupby='group',
        condition1='A',
        condition2='B',
        ls_factor=20.0,  # Larger than default (10.0)
        return_full_results=True
    )
    
    # Check that results are returned
    assert result is not None
    assert isinstance(result, dict)
    
    # Check that ls_factor is stored in params
    assert 'kompot_da' in adata.uns
    
    # Get last run info using the utility function
    from kompot.anndata.utils import get_last_run_info
    last_run_info = get_last_run_info(adata, 'da')
    
    assert last_run_info is not None
    assert 'params' in last_run_info
    assert 'ls_factor' in last_run_info['params']
    assert last_run_info['params']['ls_factor'] == 20.0


def test_compute_differential_abundance_with_random_state():
    """Test compute_differential_abundance function with random_state."""
    # Create test data
    adata1 = create_test_anndata()
    adata2 = create_test_anndata()
    
    # Run differential abundance analysis with same random state
    result1 = compute_differential_abundance(
        adata1,
        groupby='group',
        condition1='A',
        condition2='B',
        n_landmarks=20,
        random_state=42,
        return_full_results=True
    )
    
    result2 = compute_differential_abundance(
        adata2,
        groupby='group',
        condition1='A',
        condition2='B',
        n_landmarks=20,
        random_state=42,
        return_full_results=True
    )
    
    # Check that landmarks are the same with same random state
    assert np.allclose(result1['landmarks'], result2['landmarks'])
    
    # Run with different random state
    adata3 = create_test_anndata()
    result3 = compute_differential_abundance(
        adata3,
        groupby='group',
        condition1='A',
        condition2='B',
        n_landmarks=20,
        random_state=43,  # Different random state
        return_full_results=True
    )
    
    # Check that landmarks are different with different random state
    assert not np.allclose(result1['landmarks'], result3['landmarks'])


def test_compute_differential_abundance_with_batch_size():
    """Test compute_differential_abundance function with batch_size parameter."""
    # Create test data
    adata = create_test_anndata(n_cells=200)  # More cells for testing batch processing
    
    # Run differential abundance analysis with batch size
    result = compute_differential_abundance(
        adata,
        groupby='group',
        condition1='A',
        condition2='B',
        batch_size=50,  # Process 50 samples at a time
        return_full_results=True
    )
    
    # Check that results are returned
    assert result is not None
    assert isinstance(result, dict)
    
    # Basic check for correct result shapes
    assert result['log_fold_change'].shape[0] == adata.n_obs
    assert result['log_fold_change_zscore'].shape[0] == adata.n_obs
    assert result['neg_log10_fold_change_pvalue'].shape[0] == adata.n_obs


def test_compute_differential_abundance_with_copy():
    """Test compute_differential_abundance function with copy parameter."""
    # Create test data
    adata = create_test_anndata()
    
    # Run differential abundance analysis with copy=True
    adata_copy = compute_differential_abundance(
        adata,
        groupby='group',
        condition1='A',
        condition2='B',
        copy=True
    )
    
    # Check that a copy was returned
    assert adata_copy is not None
    assert adata_copy is not adata
    
    # Check that results are in the copy
    expected_obs_patterns = [
        'kompot_da_log_fold_change',
        'kompot_da_log_fold_change_zscore',
        'kompot_da_neg_log10_fold_change_pvalue',
        'kompot_da_log_fold_change_direction'
    ]
    
    for pattern in expected_obs_patterns:
        matching_cols = [col for col in adata_copy.obs.columns if pattern in col]
        assert matching_cols, f"No column matching '{pattern}' found in adata_copy.obs"
    
    # Check that no results are in the original
    for pattern in expected_obs_patterns:
        matching_cols = [col for col in adata.obs.columns if pattern in col]
        assert not matching_cols, f"Column matching '{pattern}' found in original adata.obs"


def test_compute_differential_abundance_with_result_key():
    """Test compute_differential_abundance function with custom result_key."""
    # Create test data
    adata = create_test_anndata()
    
    # Run differential abundance analysis with custom result_key
    result = compute_differential_abundance(
        adata,
        groupby='group',
        condition1='A',
        condition2='B',
        result_key='custom_da',
        return_full_results=True
    )
    
    # Check that results are returned
    assert result is not None
    assert isinstance(result, dict)
    
    # Check that fields with custom prefix were added to the AnnData object
    expected_obs_patterns = [
        'custom_da_log_fold_change',
        'custom_da_log_fold_change_zscore',
        'custom_da_neg_log10_fold_change_pvalue',
        'custom_da_log_fold_change_direction'
    ]
    
    for pattern in expected_obs_patterns:
        matching_cols = [col for col in adata.obs.columns if pattern in col]
        assert matching_cols, f"No column matching '{pattern}' found in adata.obs"
    
    # Check that custom result is tracked in kompot_da
    assert 'kompot_da' in adata.uns
    
    # Get last run info using the utility function
    from kompot.anndata.utils import get_last_run_info
    last_run_info = get_last_run_info(adata, 'da')
    
    assert last_run_info is not None
    assert last_run_info['result_key'] == 'custom_da'


def test_compute_differential_abundance_overwrite_behavior():
    """Test compute_differential_abundance function's overwrite behavior."""
    # Create test data
    adata = create_test_anndata()
    
    # Run differential abundance analysis first time
    compute_differential_abundance(
        adata,
        groupby='group',
        condition1='A',
        condition2='B',
        result_key='test_da'
    )
    
    # Check if result key is tracked in field tracking
    assert 'kompot_da' in adata.uns
    
    # Get field tracking and run history using utility functions
    from kompot.anndata.utils import get_json_metadata, get_run_history
    
    field_tracking = get_json_metadata(adata, 'kompot_da.anndata_fields')
    assert field_tracking is not None
    assert 'uns' in field_tracking
    assert 'test_da' in field_tracking['uns']
    
    # Check run history
    assert 'run_history' in adata.uns['kompot_da']
    run_history = get_run_history(adata, 'da')
    initial_run_count = len(run_history)
    
    # Run again with same parameters and overwrite=True
    compute_differential_abundance(
        adata,
        groupby='group',
        condition1='A',
        condition2='B',
        result_key='test_da',
        overwrite=True
    )
    
    # Check that another run was added to history
    updated_run_history = get_run_history(adata, 'da')
    assert len(updated_run_history) == initial_run_count + 1
    
    # Verify that field tracking has been updated
    updated_field_tracking = get_json_metadata(adata, 'kompot_da.anndata_fields')
    assert updated_field_tracking is not None
    assert 'obs' in updated_field_tracking
    assert 'test_da_log_fold_change_A_to_B' in updated_field_tracking['obs']
    
    # Run again with overwrite=False - should raise a ValueError
    with pytest.raises(ValueError):
        compute_differential_abundance(
            adata,
            groupby='group',
            condition1='A',
            condition2='B',
            result_key='test_da',
            overwrite=False,
            return_full_results=True
        )