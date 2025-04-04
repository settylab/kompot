"""Tests for the posterior covariance storage functionality."""

import numpy as np
import pytest
import logging
import pandas as pd

from kompot.anndata import compute_differential_expression
from kompot.anndata.utils import get_last_run_info


def create_test_anndata(n_cells=100, n_genes=20, with_sample_col=False):
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
        'DM_EigenVectors': np.random.normal(0, 1, (n_cells, 10))
    }
    
    # Create observation dataframe
    obs_dict = {'group': groups}
    
    # Add sample column if requested (3 samples per condition)
    if with_sample_col:
        # Create 3 samples per condition, each with equal number of cells
        n_samples_per_condition = 3
        cells_per_sample = n_cells // (2 * n_samples_per_condition)
        
        sample_ids = []
        for condition in ['A', 'B']:
            for sample_id in range(n_samples_per_condition):
                sample_name = f"{condition}_sample_{sample_id}"
                sample_ids.extend([sample_name] * cells_per_sample)
        
        # If there are any remaining cells due to division, assign them to the last sample
        while len(sample_ids) < n_cells:
            sample_ids.append(f"B_sample_{n_samples_per_condition-1}")
            
        obs_dict['sample'] = sample_ids
    
    obs = pd.DataFrame(obs_dict)
    
    # Create var_names
    var_names = [f'gene_{i}' for i in range(n_genes)]
    
    # Create var DataFrame with var_names as index
    var = pd.DataFrame(index=var_names)
    
    return anndata.AnnData(X=X, obs=obs, var=var, obsm=obsm)


def test_posterior_covariance_storage():
    """Test the store_posterior_covariance parameter in compute_differential_expression."""
    # Create a test AnnData object without sample column
    adata = create_test_anndata(with_sample_col=False)
    
    # Run differential expression analysis with store_posterior_covariance=True
    result_key = 'test_posterior_cov'
    compute_differential_expression(
        adata,
        groupby='group',
        condition1='A',
        condition2='B',
        n_landmarks=None,  # No landmarks to ensure covariance computation works
        sample_col=None,  # No sample variance
        store_posterior_covariance=True,
        result_key=result_key,
        inplace=True,
        return_full_results=False
    )
    
    # Get the expected covariance matrix key
    last_run_info = get_last_run_info(adata, 'de')
    assert last_run_info is not None
    assert 'posterior_covariance_key' in last_run_info
    
    # Get the key name used for storing the covariance matrix
    cov_key = last_run_info['posterior_covariance_key']
    
    # Verify the covariance matrix was stored in obsp
    assert cov_key in adata.obsp
    
    # Verify the condition names are in the field name
    assert "A_to_B" in cov_key
    
    # Verify the shape of the covariance matrix (n_cells x n_cells)
    assert adata.obsp[cov_key].shape == (adata.n_obs, adata.n_obs)
    
    # Check that it's symmetric
    cov_matrix = adata.obsp[cov_key]
    assert np.allclose(cov_matrix, cov_matrix.T)
    
    # Check that the parameter was stored in run info
    assert 'params' in last_run_info
    assert 'store_posterior_covariance' in last_run_info['params']
    assert last_run_info['params']['store_posterior_covariance'] is True


def test_posterior_covariance_with_landmarks():
    """Test that posterior covariance can be stored when landmarks are used."""
    # Create a test AnnData object without sample column
    adata = create_test_anndata(with_sample_col=False)
    
    # Run differential expression analysis with landmarks
    result_key = 'test_with_landmarks'
    compute_differential_expression(
        adata,
        groupby='group',
        condition1='A',
        condition2='B',
        n_landmarks=10,  # Use landmarks
        sample_col=None,  # No sample variance
        store_posterior_covariance=True,  # Should now work with landmarks
        result_key=result_key,
        inplace=True,
        return_full_results=False
    )
    
    # Get the last run info
    last_run_info = get_last_run_info(adata, 'de')
    assert last_run_info is not None
    
    # Check that the parameter was stored in run info
    assert 'params' in last_run_info
    assert 'store_posterior_covariance' in last_run_info['params']
    assert last_run_info['params']['store_posterior_covariance'] is True
    
    # Verify the posterior covariance key is in the run info
    assert 'posterior_covariance_key' in last_run_info
    
    # Get the key name and verify the covariance matrix was stored
    cov_key = last_run_info['posterior_covariance_key']
    assert cov_key in adata.obsp
    
    # Verify the condition names are in the field name
    assert "A_to_B" in cov_key
    
    # Verify the shape of the covariance matrix
    assert adata.obsp[cov_key].shape == (adata.n_obs, adata.n_obs)


def test_posterior_covariance_with_sample_variance():
    """Test that posterior covariance is not stored when sample variance is used."""
    # Create a test AnnData object with sample column
    adata = create_test_anndata(with_sample_col=True)
    
    # Run differential expression analysis with sample variance
    result_key = 'test_with_sample_var'
    compute_differential_expression(
        adata,
        groupby='group',
        condition1='A',
        condition2='B',
        n_landmarks=None,  # No landmarks
        sample_col='sample',  # Use sample variance
        store_posterior_covariance=True,  # Try to store covariance even though it won't work
        result_key=result_key,
        inplace=True,
        return_full_results=False
    )
    
    # Get the last run info
    last_run_info = get_last_run_info(adata, 'de')
    assert last_run_info is not None
    
    # Check that the parameter was stored in run info
    assert 'params' in last_run_info
    assert 'store_posterior_covariance' in last_run_info['params']
    assert last_run_info['params']['store_posterior_covariance'] is True
    
    # Verify that the posterior_covariance_key is not in the run info
    assert 'posterior_covariance_key' not in last_run_info
    
    # Verify that no covariance matrix was stored in obsp
    # Get the expected key name pattern
    field_names = last_run_info.get('field_names', {})
    assert 'posterior_covariance_key' in field_names
    cov_key = field_names['posterior_covariance_key']
    
    # Check that the covariance matrix was not stored
    assert cov_key not in adata.obsp