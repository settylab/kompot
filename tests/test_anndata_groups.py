"""Tests for the 'groups' parameter in anndata functions."""

import numpy as np
import pytest
import pandas as pd
import json
import logging

from kompot.anndata import compute_differential_expression
from kompot.anndata.utils import parse_groups

def check_group_metrics_varm(adata, result_key):
    """Helper to check for expected varm fields for group metrics.
    
    Returns:
    - mean_lfc_key: The varm key for mean log fold change metrics across groups
    - mahalanobis_key: The varm key for mahalanobis distances across groups
    """
    # Don't try to access complex metadata, just look at field names directly
    
    # Get all varm keys
    varm_keys = list(adata.varm.keys())
    print(f"Available varm keys: {varm_keys}")
    
    # Find the keys we need based on pattern matching
    mean_lfc_key = None
    mahalanobis_key = None
    
    # Look for varm keys with the result_key and other identifiers
    for key in varm_keys:
        if result_key in key and "mean_lfc" in key and "_groups" in key:
            mean_lfc_key = key
        elif result_key in key and "mahalanobis" in key and "_groups" in key:
            mahalanobis_key = key
    
    # If we didn't find a mahalanobis key but found a mean key, it's ok since some tests
    # use compute_mahalanobis=False
    
    # Return the keys we found
    return mean_lfc_key, mahalanobis_key


def create_test_anndata(n_cells=100, n_genes=20, with_sample_col=False, with_multiple_groups=False):
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
    
    # Add multiple group columns if requested
    if with_multiple_groups:
        # Create a categorical column
        obs_dict['category'] = np.random.choice(['cat1', 'cat2', 'cat3'], size=n_cells)
        
        # Create a boolean column
        obs_dict['is_selected'] = np.random.choice([True, False], size=n_cells)
        
        # Create a numeric column
        obs_dict['score'] = np.random.uniform(0, 10, size=n_cells)
        
        # Create a column with some NaN values
        obs_dict['has_nan'] = np.random.uniform(0, 1, size=n_cells)
        obs_dict['has_nan'][np.random.choice(n_cells, size=n_cells//10)] = np.nan
    
    obs = pd.DataFrame(obs_dict)
    
    # Create var_names
    var_names = [f'gene_{i}' for i in range(n_genes)]
    
    # Create var DataFrame with var_names as index
    var = pd.DataFrame(index=var_names)
    
    return anndata.AnnData(X=X, obs=obs, var=var, obsm=obsm)


def test_parse_groups_string():
    """Test the parse_groups function with string input."""
    adata = create_test_anndata(with_multiple_groups=True)
    
    # Test with categorical column
    subset_masks, subset_names = parse_groups(adata, 'category')
    assert len(subset_masks) == 3  # Three unique categories
    assert len(subset_names) == 3
    for mask in subset_masks.values():
        assert mask.shape == (adata.n_obs,)
        assert mask.dtype == bool
    
    # Test with boolean column
    subset_masks, subset_names = parse_groups(adata, 'is_selected')
    assert len(subset_masks) == 1  # Only one subset (True values)
    assert len(subset_names) == 1
    assert subset_names[0] == 'True'
    assert subset_masks['True'].shape == (adata.n_obs,)
    assert subset_masks['True'].dtype == bool
    
    # Test with non-existent column (should raise ValueError)
    with pytest.raises(ValueError):
        parse_groups(adata, 'non_existent_column')
        
    # Test with numeric column (should raise ValueError)
    with pytest.raises(ValueError):
        parse_groups(adata, 'score')


def test_parse_groups_dict():
    """Test the parse_groups function with dictionary input."""
    adata = create_test_anndata(with_multiple_groups=True)
    
    # Test with single condition
    subset_masks, subset_names = parse_groups(adata, {'category': 'cat1'})
    assert len(subset_masks) == 1
    assert 'category=cat1' in subset_names[0]
    
    # Test with multiple conditions in one filter
    subset_masks, subset_names = parse_groups(adata, {'category': ['cat1', 'cat2'], 'is_selected': True})
    assert len(subset_masks) == 1
    filter_desc = subset_names[0]
    assert 'category=cat1,cat2' in filter_desc
    assert 'is_selected=True' in filter_desc
    
    # Test with non-existent column (should raise ValueError)
    with pytest.raises(ValueError):
        parse_groups(adata, {'non_existent_column': 'value'})


def test_parse_groups_dict_of_dicts():
    """Test the parse_groups function with dictionary of dictionaries input."""
    adata = create_test_anndata(with_multiple_groups=True)
    
    # Test with named filters
    filters = {
        'cat1_group': {'category': 'cat1'},
        'cat2_selected': {'category': 'cat2', 'is_selected': True},
        'cat3_not_selected': {'category': 'cat3', 'is_selected': False}
    }
    
    subset_masks, subset_names = parse_groups(adata, filters)
    assert len(subset_masks) == 3
    assert len(subset_names) == 3
    
    # Check that the provided names were used as subset names
    for name in filters.keys():
        assert name in subset_names
        assert name in subset_masks
    
    # Each mask should have different cells
    for name1, mask1 in subset_masks.items():
        for name2, mask2 in subset_masks.items():
            if name1 != name2:
                assert not np.all(mask1 == mask2)
    
    # Test empty dictionary
    subset_masks, subset_names = parse_groups(adata, {})
    assert len(subset_masks) == 0
    assert len(subset_names) == 0
    
    # Test with non-existent column
    with pytest.raises(ValueError):
        parse_groups(adata, {'test_group': {'non_existent_column': 'value'}})


def test_parse_groups_list_of_dicts():
    """Test the parse_groups function with list of dictionaries input."""
    adata = create_test_anndata(with_multiple_groups=True)
    
    # Test with multiple filters
    filters = [
        {'category': 'cat1'},
        {'category': 'cat2', 'is_selected': True},
        {'category': 'cat3', 'is_selected': False}
    ]
    
    subset_masks, subset_names = parse_groups(adata, filters)
    assert len(subset_masks) == 3
    assert len(subset_names) == 3
    
    # Each mask should have different cells
    for name1, mask1 in subset_masks.items():
        for name2, mask2 in subset_masks.items():
            if name1 != name2:
                assert not np.all(mask1 == mask2)
    
    # Test with empty list (should return empty results)
    subset_masks, subset_names = parse_groups(adata, [])
    assert len(subset_masks) == 0
    assert len(subset_names) == 0


def test_parse_groups_array():
    """Test the parse_groups function with array input."""
    adata = create_test_anndata(with_multiple_groups=True)
    
    # Test with boolean array
    bool_array = np.random.choice([True, False], size=adata.n_obs)
    subset_masks, subset_names = parse_groups(adata, bool_array)
    assert len(subset_masks) == 1
    assert 'True' in subset_names
    assert np.all(subset_masks['True'] == bool_array)
    
    # Test with categorical array
    cat_array = np.random.choice(['x', 'y', 'z'], size=adata.n_obs)
    subset_masks, subset_names = parse_groups(adata, cat_array)
    assert len(subset_masks) == 3
    assert len(subset_names) == 3
    assert 'x' in subset_names
    assert 'y' in subset_names
    assert 'z' in subset_names
    
    # Test with wrong length array (should raise ValueError)
    with pytest.raises(ValueError):
        parse_groups(adata, np.array([True, False]))


def test_parse_groups_series():
    """Test the parse_groups function with pandas Series input."""
    adata = create_test_anndata(with_multiple_groups=True)
    
    # Test with boolean Series
    bool_series = pd.Series(np.random.choice([True, False], size=adata.n_obs))
    subset_masks, subset_names = parse_groups(adata, bool_series)
    assert len(subset_masks) == 1
    assert 'True' in subset_names
    
    # Test with categorical Series
    cat_series = pd.Series(np.random.choice(['x', 'y', 'z'], size=adata.n_obs))
    subset_masks, subset_names = parse_groups(adata, cat_series)
    assert len(subset_masks) == 3
    assert len(subset_names) == 3
    
    # Test with wrong length Series (should raise ValueError)
    with pytest.raises(ValueError):
        parse_groups(adata, pd.Series([True, False]))


def test_parse_groups_boolean_mask_array():
    """Test the parse_groups function with 2D boolean mask array."""
    adata = create_test_anndata(with_multiple_groups=True)
    
    # Create 3 different boolean masks - explicitly create a 2D array
    mask1 = np.random.choice([True, False], size=adata.n_obs)
    mask2 = np.random.choice([True, False], size=adata.n_obs)
    mask3 = np.random.choice([True, False], size=adata.n_obs)
    
    # Stack them into a proper 2D array
    masks = np.vstack([mask1, mask2, mask3])
    
    # Make sure the shape is correct for our test
    assert masks.shape == (3, adata.n_obs)
    
    subset_masks, subset_names = parse_groups(adata, masks)
    assert len(subset_masks) == 3
    assert 'subset1' in subset_names
    assert 'subset2' in subset_names
    assert 'subset3' in subset_names
    
    # Each mask should match the input (comparing boolean values)
    for i, name in enumerate(subset_names):
        assert np.all(subset_masks[name] == masks[i])


def test_parse_groups_list_of_arrays():
    """Test the parse_groups function with list of arrays input."""
    adata = create_test_anndata(with_multiple_groups=True)
    
    # Create list of different arrays
    arrays = [
        np.random.choice([True, False], size=adata.n_obs),  # Boolean
        np.random.choice(['x', 'y'], size=adata.n_obs),     # Categorical
        np.ones(adata.n_obs, dtype=np.int64)                # Fixed numeric array always with 1s
    ]
    
    subset_masks, subset_names = parse_groups(adata, arrays)
    
    # Should create masks for boolean array and each category in categorical/numeric arrays
    assert len(subset_masks) > 3  # At least 1 + 2 + 3 = 6 masks
    
    # Boolean mask should be included
    assert 'subset1' in subset_names
    
    # Categories from second array should be included
    assert 'subset2_x' in subset_names
    assert 'subset2_y' in subset_names
    
    # Categories from third array should be included
    assert 'subset3_1' in subset_names
    assert 'subset3_2' in subset_names
    assert 'subset3_3' in subset_names


def test_compute_de_with_groups_string():
    """Test compute_differential_expression with string-based grouping."""
    adata = create_test_anndata(with_multiple_groups=True)
    
    # Run with a column name as groups
    result = compute_differential_expression(
        adata,
        groupby='group',
        condition1='A',
        condition2='B',
        groups='category',  # Use category column for subsetting
        result_key='de_test_groups',
        compute_mahalanobis=False,  # Disable for faster testing
        return_full_results=True
    )
    
    # Get unique categories
    categories = adata.obs['category'].unique()
    
    # Get the varm keys for group metrics using our helper
    mean_lfc_key, _ = check_group_metrics_varm(adata, 'de_test_groups')
    
    # We should have found a mean LFC key since we used groups
    assert mean_lfc_key is not None, "No mean LFC varm key found for groups"
    print(f"Found mean LFC varm key: {mean_lfc_key}")
    
    # Use this as our varm_key
    varm_key = mean_lfc_key
    
    # Check that each category has its own column in the varm matrix
    varm_df = adata.varm[varm_key]
    print(f"varm matrix columns: {list(varm_df.columns)}")
    
    for category in categories:
        # Check that the category is a column in the varm DataFrame
        assert category in varm_df.columns, f"Category '{category}' not found in varm matrix columns: {list(varm_df.columns)}"
        
        # Check that the column has values (not all NaN)
        assert not pd.isna(varm_df[category]).all(), f"All values for category '{category}' are NaN"


def test_compute_de_with_groups_dict():
    """Test compute_differential_expression with dictionary-based grouping."""
    adata = create_test_anndata(with_multiple_groups=True)
    
    # Run with a dictionary as groups
    result = compute_differential_expression(
        adata,
        groupby='group',
        condition1='A',
        condition2='B',
        groups={'is_selected': True},  # Only select cells where is_selected is True
        result_key='de_test_dict',
        compute_mahalanobis=False,  # Disable for faster testing
        return_full_results=True
    )
    
    # Get the varm keys for group metrics using our helper
    mean_lfc_key, _ = check_group_metrics_varm(adata, 'de_test_dict')
    
    # We should have found a mean LFC key since we used groups
    assert mean_lfc_key is not None, "No mean LFC varm key found for groups"
    print(f"Found mean LFC varm key: {mean_lfc_key}")
    
    # Use this as our varm_key
    varm_key = mean_lfc_key
    
    # Check that the varm matrix has the expected subset column
    varm_df = adata.varm[varm_key]
    print(f"varm matrix columns: {list(varm_df.columns)}")
    
    # The expected subset name for the dictionary filter
    expected_subset = "is_selected=True"
    
    # Check if there's a column containing the subset name (might be partial match)
    matching_cols = [col for col in varm_df.columns if "is_selected=True" in col]
    if not matching_cols:
        # If no exact match, look for any column containing "is_selected"
        matching_cols = [col for col in varm_df.columns if "is_selected" in col]
    
    assert matching_cols, f"No column for subset '{expected_subset}' found in varm matrix columns: {list(varm_df.columns)}"
    subset_col = matching_cols[0]
    
    # Check that the column has values (not all NaN)
    assert not pd.isna(varm_df[subset_col]).all(), f"All values for subset '{subset_col}' are NaN"


def test_compute_de_with_groups_array():
    """Test compute_differential_expression with array-based grouping."""
    adata = create_test_anndata(with_multiple_groups=True)
    
    # Create a random boolean mask
    mask = np.random.choice([True, False], size=adata.n_obs)
    
    # Run with a boolean array as groups
    result = compute_differential_expression(
        adata,
        groupby='group',
        condition1='A',
        condition2='B',
        groups=mask,  # Use the boolean mask
        result_key='de_test_array',
        compute_mahalanobis=False,  # Disable for faster testing
        return_full_results=True
    )
    
    # Get the varm keys for group metrics using our helper
    mean_lfc_key, _ = check_group_metrics_varm(adata, 'de_test_array')
    
    # We should have found a mean LFC key since we used groups
    assert mean_lfc_key is not None, "No mean LFC varm key found for groups"
    print(f"Found mean LFC varm key: {mean_lfc_key}")
    
    # Use this as our varm_key
    varm_key = mean_lfc_key
    
    # Check that the varm matrix has the expected subset column
    varm_df = adata.varm[varm_key]
    print(f"varm matrix columns: {list(varm_df.columns)}")
    
    # For a boolean array, the subset name should be 'True'
    expected_subset = "True"
    
    # Check if there's a column with the exact name 'True'
    matching_cols = [col for col in varm_df.columns if col == "True"]
    if not matching_cols:
        # If no exact match, look for any column containing "True"
        matching_cols = [col for col in varm_df.columns if "True" in str(col)]
    
    assert matching_cols, f"No column for subset '{expected_subset}' found in varm matrix columns: {list(varm_df.columns)}"
    subset_col = matching_cols[0]
    
    # Check that the column has values (not all NaN)
    assert not pd.isna(varm_df[subset_col]).all(), f"All values for subset '{subset_col}' are NaN"


def test_compute_de_with_named_groups():
    """Test compute_differential_expression with named groups (dict of dicts)."""
    adata = create_test_anndata(with_multiple_groups=True)
    
    # Create a dict of filters with custom names
    named_filters = {
        'category1': {'category': 'cat1'},
        'category2_selected': {'category': 'cat2', 'is_selected': True}
    }
    
    # Run with named groups
    result = compute_differential_expression(
        adata,
        groupby='group',
        condition1='A',
        condition2='B',
        groups=named_filters,  # Use the dict of filters
        result_key='de_test_named',
        compute_mahalanobis=True,  # Enable to test Mahalanobis distances
        return_full_results=True
    )
    
    # Get the varm keys for group metrics using our helper
    mean_lfc_varm_key, mahalanobis_varm_key = check_group_metrics_varm(adata, 'de_test_named')
    
    # We should have found both keys since we used groups and compute_mahalanobis=True
    assert mean_lfc_varm_key is not None, "No mean LFC varm key found for groups"
    assert mahalanobis_varm_key is not None, "No mahalanobis varm key found for groups"
    print(f"Found varm keys - mean LFC: {mean_lfc_varm_key}, mahalanobis: {mahalanobis_varm_key}")
    
    # Check that the varm matrices exist
    assert mean_lfc_varm_key in adata.varm, f"varm key {mean_lfc_varm_key} not found in adata.varm"
    assert mahalanobis_varm_key in adata.varm, f"varm key {mahalanobis_varm_key} not found in adata.varm"
    
    # Get the varm dataframes
    mean_lfc_df = adata.varm[mean_lfc_varm_key]
    mahalanobis_df = adata.varm[mahalanobis_varm_key]
    
    print(f"Mean LFC matrix columns: {list(mean_lfc_df.columns)}")
    print(f"Mahalanobis matrix columns: {list(mahalanobis_df.columns)}")
    
    # Check for custom group names
    for group_name in named_filters.keys():
        # Check in mean LFC varm
        mean_lfc_cols = [col for col in mean_lfc_df.columns if group_name in str(col)]
        assert mean_lfc_cols, f"No column for named group '{group_name}' found in mean LFC matrix: {list(mean_lfc_df.columns)}"
        mean_lfc_col = mean_lfc_cols[0]
        assert not pd.isna(mean_lfc_df[mean_lfc_col]).all(), f"All values for named group '{group_name}' in mean LFC matrix are NaN"
        
        # Check in mahalanobis varm
        mahalanobis_cols = [col for col in mahalanobis_df.columns if group_name in str(col)]
        assert mahalanobis_cols, f"No column for named group '{group_name}' found in mahalanobis matrix: {list(mahalanobis_df.columns)}"
        mahalanobis_col = mahalanobis_cols[0]
        assert not pd.isna(mahalanobis_df[mahalanobis_col]).all(), f"All values for named group '{group_name}' in mahalanobis matrix are NaN"


def test_compute_de_with_multiple_groups():
    """Test compute_differential_expression with multiple groups."""
    adata = create_test_anndata(with_multiple_groups=True)
    
    # Create a list of filters for multiple groups
    filters = [
        {'category': 'cat1'},
        {'category': 'cat2', 'is_selected': True}
    ]
    
    # Run with multiple groups
    result = compute_differential_expression(
        adata,
        groupby='group',
        condition1='A',
        condition2='B',
        groups=filters,  # Use the list of filters
        result_key='de_test_multiple',
        compute_mahalanobis=True,  # Enable to test Mahalanobis distances
        return_full_results=True
    )
    
    # Get the varm keys for group metrics using our helper
    mean_lfc_key, mahalanobis_key = check_group_metrics_varm(adata, 'de_test_multiple')
    
    # We should have found both keys since we used groups and compute_mahalanobis=True
    assert mean_lfc_key is not None, "No mean LFC varm key found for groups"
    assert mahalanobis_key is not None, "No mahalanobis varm key found for groups"
    print(f"Found varm keys - mean LFC: {mean_lfc_key}, mahalanobis: {mahalanobis_key}")
    
    # Check that the varm matrices exist
    assert mean_lfc_key in adata.varm, f"varm key {mean_lfc_key} not found in adata.varm"
    assert mahalanobis_key in adata.varm, f"varm key {mahalanobis_key} not found in adata.varm"
    
    # Get the varm dataframes
    mean_lfc_df = adata.varm[mean_lfc_key]
    mahalanobis_df = adata.varm[mahalanobis_key]
    
    print(f"Mean LFC matrix columns: {list(mean_lfc_df.columns)}")
    print(f"Mahalanobis matrix columns: {list(mahalanobis_df.columns)}")
    
    # Expected subset names or patterns
    expected_subsets = [
        "category=cat1",  # For the first filter
        "category=cat2,is_selected=True"  # For the second filter
    ]
    
    # Check first filter - cat1
    cat1_subset_cols = [col for col in mean_lfc_df.columns if "cat1" in str(col)]
    assert cat1_subset_cols, f"No column for 'cat1' found in mean LFC matrix: {list(mean_lfc_df.columns)}"
    cat1_col = cat1_subset_cols[0]
    
    # Check second filter - cat2 + is_selected
    cat2_subset_cols = [col for col in mean_lfc_df.columns 
                       if "cat2" in str(col) and "is_selected" in str(col)]
    assert cat2_subset_cols, f"No column for 'cat2,is_selected' found in mean LFC matrix: {list(mean_lfc_df.columns)}"
    cat2_col = cat2_subset_cols[0]
    
    # Check that the columns have values (not all NaN)
    assert not pd.isna(mean_lfc_df[cat1_col]).all(), f"All values for '{cat1_col}' in mean LFC matrix are NaN"
    assert not pd.isna(mean_lfc_df[cat2_col]).all(), f"All values for '{cat2_col}' in mean LFC matrix are NaN"
    
    # Check that mahalanobis columns exist and have values
    mahalanobis_cat1_cols = [col for col in mahalanobis_df.columns if "cat1" in str(col)]
    assert mahalanobis_cat1_cols, f"No column for 'cat1' found in mahalanobis matrix: {list(mahalanobis_df.columns)}"
    mahalanobis_cat1_col = mahalanobis_cat1_cols[0]
    
    mahalanobis_cat2_cols = [col for col in mahalanobis_df.columns 
                            if "cat2" in str(col) and "is_selected" in str(col)]
    assert mahalanobis_cat2_cols, f"No column for 'cat2,is_selected' found in mahalanobis matrix: {list(mahalanobis_df.columns)}"
    mahalanobis_cat2_col = mahalanobis_cat2_cols[0]
    
    # Check that the mahalanobis columns have values (not all NaN)
    assert not pd.isna(mahalanobis_df[mahalanobis_cat1_col]).all(), f"All values for '{mahalanobis_cat1_col}' in mahalanobis matrix are NaN"
    assert not pd.isna(mahalanobis_df[mahalanobis_cat2_col]).all(), f"All values for '{mahalanobis_cat2_col}' in mahalanobis matrix are NaN"
    
    # Check field tracking - make sure varm matrices are tracked
    assert 'anndata_fields' in adata.uns['kompot_de']
    
    # Get tracking data - need to deserialize
    from kompot.anndata.utils import get_json_metadata
    field_tracking = get_json_metadata(adata, 'kompot_de.anndata_fields')
    assert isinstance(field_tracking, dict), f"Expected field_tracking to be a dict, but got {type(field_tracking)}"
    assert 'varm' in field_tracking
    
    # Check that varm fields are tracked
    assert mean_lfc_key in field_tracking['varm']
    assert mahalanobis_key in field_tracking['varm']


def test_compute_de_with_landmark_handling():
    """Test compute_differential_expression landmark handling with groups."""
    adata = create_test_anndata(n_cells=300, n_genes=20, with_multiple_groups=True)
    
    # Run DE with a small number of landmarks
    n_landmarks = 50  # Small number of landmarks
    
    # First run without groups to establish landmarks
    result = compute_differential_expression(
        adata,
        groupby='group',
        condition1='A',
        condition2='B',
        result_key='de_test_landmarks',
        n_landmarks=n_landmarks,  # Use a small number of landmarks
        compute_mahalanobis=True,
        store_landmarks=True,  # Store landmarks for later
        return_full_results=True
    )
    
    # Now create a large subset with more cells than landmarks
    # Large subset with more than n_landmarks cells
    large_subset = np.random.choice([True, False], size=adata.n_obs, p=[0.7, 0.3])
    # Make sure we have enough cells in the subset
    assert np.sum(large_subset) > n_landmarks
    
    # Small subset with fewer cells than landmarks
    small_subset = np.random.choice([True, False], size=adata.n_obs, p=[0.1, 0.9])
    # Make sure we have fewer cells than landmarks
    if np.sum(small_subset) >= n_landmarks:
        # If by chance we still have too many, just reduce the subset size manually
        indices = np.where(small_subset)[0]
        keep = indices[:n_landmarks-5]  # Keep fewer than n_landmarks
        small_subset = np.zeros_like(small_subset, dtype=bool)
        small_subset[keep] = True
    
    # Verify we have fewer cells than landmarks
    assert np.sum(small_subset) < n_landmarks
    
    # Create 2D array of boolean masks with both subsets
    masks = np.vstack([large_subset, small_subset])
    
    # Run with both subsets as groups
    result = compute_differential_expression(
        adata,
        groupby='group',
        condition1='A',
        condition2='B',
        groups=masks,  # Use the 2D boolean masks array
        result_key='de_test_landmarks_with_groups',
        compute_mahalanobis=True,
        return_full_results=True
    )
    
    # Get the varm keys for group metrics using our helper
    mean_lfc_varm_key, mahalanobis_varm_key = check_group_metrics_varm(adata, 'de_test_landmarks_with_groups')
    
    # We should have found both keys since we used groups and compute_mahalanobis=True
    assert mean_lfc_varm_key is not None, "No mean LFC varm key found for groups"
    assert mahalanobis_varm_key is not None, "No mahalanobis varm key found for groups"
    print(f"Found varm keys - mean LFC: {mean_lfc_varm_key}, mahalanobis: {mahalanobis_varm_key}")
    
    # Check that the varm matrices exist
    assert mean_lfc_varm_key in adata.varm, f"varm key {mean_lfc_varm_key} not found in adata.varm"
    assert mahalanobis_varm_key in adata.varm, f"varm key {mahalanobis_varm_key} not found in adata.varm"
    
    # Get the varm dataframes
    mean_lfc_df = adata.varm[mean_lfc_varm_key]
    mahalanobis_df = adata.varm[mahalanobis_varm_key]
    
    print(f"Mean LFC matrix columns: {list(mean_lfc_df.columns)}")
    print(f"Mahalanobis matrix columns: {list(mahalanobis_df.columns)}")
    
    # The subset names should be "subset1" and "subset2" for boolean arrays in a 2D array
    # Check first subset (large)
    subset1_cols = [col for col in mean_lfc_df.columns if "subset1" in str(col)]
    assert subset1_cols, f"No column for 'subset1' found in mean LFC matrix: {list(mean_lfc_df.columns)}"
    subset1_col = subset1_cols[0]
    
    # Check second subset (small)
    subset2_cols = [col for col in mean_lfc_df.columns if "subset2" in str(col)]
    assert subset2_cols, f"No column for 'subset2' found in mean LFC matrix: {list(mean_lfc_df.columns)}"
    subset2_col = subset2_cols[0]
    
    # Check that the columns have values (not all NaN)
    assert not pd.isna(mean_lfc_df[subset1_col]).all(), f"All values for '{subset1_col}' in mean LFC matrix are NaN"
    assert not pd.isna(mean_lfc_df[subset2_col]).all(), f"All values for '{subset2_col}' in mean LFC matrix are NaN"
    
    # Check that mahalanobis columns exist and have values
    mahalanobis_subset1_cols = [col for col in mahalanobis_df.columns if "subset1" in str(col)]
    assert mahalanobis_subset1_cols, f"No column for 'subset1' found in mahalanobis matrix: {list(mahalanobis_df.columns)}"
    mahalanobis_subset1_col = mahalanobis_subset1_cols[0]
    
    mahalanobis_subset2_cols = [col for col in mahalanobis_df.columns if "subset2" in str(col)]
    assert mahalanobis_subset2_cols, f"No column for 'subset2' found in mahalanobis matrix: {list(mahalanobis_df.columns)}"
    mahalanobis_subset2_col = mahalanobis_subset2_cols[0]
    
    # Check that the mahalanobis columns have values (not all NaN)
    assert not pd.isna(mahalanobis_df[mahalanobis_subset1_col]).all(), f"All values for '{mahalanobis_subset1_col}' in mahalanobis matrix are NaN"
    assert not pd.isna(mahalanobis_df[mahalanobis_subset2_col]).all(), f"All values for '{mahalanobis_subset2_col}' in mahalanobis matrix are NaN"


def test_compute_de_with_weighted_lfc_and_groups():
    """Test compute_differential_expression with weighted log fold change and groups."""
    adata = create_test_anndata(with_multiple_groups=True)
    
    # First, run differential abundance to create log density columns
    from kompot.anndata import compute_differential_abundance
    
    compute_differential_abundance(
        adata,
        groupby='group',
        condition1='A',
        condition2='B',
        result_key='da_test'
    )
    
    # Now run differential expression with groups and differential_abundance_key
    result = compute_differential_expression(
        adata,
        groupby='group',
        condition1='A',
        condition2='B',
        groups='category',  # Use category column for subsetting
        differential_abundance_key='da_test',  # Use DA results for weighting
        result_key='de_test_weighted',
        compute_mahalanobis=False,  # Disable for faster testing
        return_full_results=True
    )
    
    # Get unique categories
    categories = adata.obs['category'].unique()
    
    # Get the varm keys for group metrics using our helper
    mean_lfc_key, _ = check_group_metrics_varm(adata, 'de_test_weighted')
    
    # We should have found a mean LFC key since we used groups
    assert mean_lfc_key is not None, "No mean LFC varm key found for groups"
    print(f"Found mean LFC varm key: {mean_lfc_key}")
    
    # Also look for a weighted LFC key
    weighted_lfc_key = None
    for key in adata.varm.keys():
        if 'de_test_weighted' in key and 'weighted_lfc' in key and '_groups' in key:
            weighted_lfc_key = key
            break
            
    assert weighted_lfc_key is not None, "No weighted LFC varm key found for groups"
    print(f"Found weighted LFC varm key: {weighted_lfc_key}")
    
    print(f"Found varm keys - mean LFC: {mean_lfc_key}, weighted LFC: {weighted_lfc_key}")
    
    # Check that the varm matrices exist
    assert mean_lfc_key in adata.varm, f"varm key {mean_lfc_key} not found in adata.varm"
    
    # Get the mean LFC dataframe
    mean_lfc_df = adata.varm[mean_lfc_key]
    
    # Get the weighted LFC dataframe if available
    weighted_lfc_df = None
    if weighted_lfc_key is not None and weighted_lfc_key in adata.varm:
        weighted_lfc_df = adata.varm[weighted_lfc_key]
    
    print(f"Mean LFC matrix columns: {list(mean_lfc_df.columns)}")
    if weighted_lfc_df is not None:
        print(f"Weighted LFC matrix columns: {list(weighted_lfc_df.columns)}")
    else:
        print("No weighted LFC matrix available")
    
    # Check the global field for regular LFC
    assert not pd.isna(adata.var['de_test_weighted_mean_lfc_A_to_B']).all(), "Global mean LFC column has all NaN values"
    
    # Check the global field for weighted LFC
    global_weighted_lfc_cols = [col for col in adata.var.columns if "weighted" in col and "_A_to_B" in col]
    assert global_weighted_lfc_cols, f"No global weighted LFC column found in: {list(adata.var.columns)}"
    global_weighted_lfc = global_weighted_lfc_cols[0]
    assert not pd.isna(adata.var[global_weighted_lfc]).all(), "Global weighted LFC column has all NaN values"
    
    # Check that each category has its own column in both varm matrices
    for category in categories:
        # Check in mean LFC matrix
        cat_cols_mean = [col for col in mean_lfc_df.columns if category in str(col)]
        assert cat_cols_mean, f"No column for category '{category}' found in mean LFC matrix: {list(mean_lfc_df.columns)}"
        cat_col_mean = cat_cols_mean[0]
        assert not pd.isna(mean_lfc_df[cat_col_mean]).all(), f"All values for category '{category}' in mean LFC matrix are NaN"
        
        # Check in weighted LFC matrix - only do this if we have a weighted matrix
        if weighted_lfc_df is not None:
            cat_cols_weighted = [col for col in weighted_lfc_df.columns if category in str(col)]
            assert cat_cols_weighted, f"No column for category '{category}' found in weighted LFC matrix: {list(weighted_lfc_df.columns)}"
            cat_col_weighted = cat_cols_weighted[0]
            assert not pd.isna(weighted_lfc_df[cat_col_weighted]).all(), f"All values for category '{category}' in weighted LFC matrix are NaN"