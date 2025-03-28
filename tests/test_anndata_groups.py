"""Tests for the 'groups' parameter in anndata functions."""

import numpy as np
import pytest
import pandas as pd

from kompot.anndata import compute_differential_expression
from kompot.anndata.utils import parse_groups


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
        np.random.choice([1, 2, 3], size=adata.n_obs)       # Numeric
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
    
    # Check that the field names include subset-specific fields
    run_info = adata.uns['kompot_de']['last_run_info']
    field_mapping = run_info['field_mapping']
    
    # Get unique categories
    categories = adata.obs['category'].unique()
    
    # Print all columns to debug
    print(f"Available columns for string test: {list(adata.var.columns)}")
    
    # Check that each category has its own mean log fold change field
    for category in categories:
        # Find columns that match this category
        matching_cols = [col for col in adata.var.columns if category in col and "mean" in col]
        assert matching_cols, f"No columns containing '{category}' found in: {list(adata.var.columns)}"
        subset_field = matching_cols[0]
        
        # Check that the field has values (not all NaN)
        assert not pd.isna(adata.var[subset_field]).all()


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
    
    # Print columns to debug
    print(f"Available columns for dict test: {list(adata.var.columns)}")
    
    # The subset name should be based on the filter description
    subset_field_long = "de_test_dict_mean_log_fold_change_A_to_B_is_selected=True"
    subset_field_short = "de_test_dict_mean_lfc_A_to_B_is_selected=True"
    
    # Use the field that exists in the data
    if subset_field_long in adata.var.columns:
        subset_field = subset_field_long
    elif subset_field_short in adata.var.columns:
        subset_field = subset_field_short
    else:
        # Look for any field with is_selected=True in the name
        matching_cols = [col for col in adata.var.columns if "is_selected=True" in col and "mean" in col]
        assert matching_cols, f"No columns containing 'is_selected=True' found in: {list(adata.var.columns)}"
        subset_field = matching_cols[0]
    
    # Check that the field exists and has values
    assert not pd.isna(adata.var[subset_field]).all()


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
    
    # Print columns to debug
    print(f"Available columns for array test: {list(adata.var.columns)}")
    
    # The subset name should be 'True' for boolean array (try both naming patterns)
    subset_field_long = "de_test_array_mean_log_fold_change_A_to_B_True"
    subset_field_short = "de_test_array_mean_lfc_A_to_B_True"
    
    # Use the field that exists in the data
    if subset_field_long in adata.var.columns:
        subset_field = subset_field_long
    elif subset_field_short in adata.var.columns:
        subset_field = subset_field_short
    else:
        # Just look for any field with the 'True' subset marker
        matching_cols = [col for col in adata.var.columns if "True" in col and "mean" in col]
        assert matching_cols, f"No columns containing 'True' subset identifier found in: {list(adata.var.columns)}"
        subset_field = matching_cols[0]
    
    # Check that the field exists and has values
    assert not pd.isna(adata.var[subset_field]).all()


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
    
    # Print all columns to debug
    print(f"Available columns for multiple groups test: {list(adata.var.columns)}")
    
    # Find column name patterns - be more flexible with the naming
    # For first subset (cat1)
    cat1_cols = [col for col in adata.var.columns if "cat1" in col and "mean" in col]
    assert cat1_cols, f"No columns containing 'cat1' found in: {list(adata.var.columns)}"
    subset1_lfc = cat1_cols[0]
    
    # For second subset (cat2 + is_selected)
    cat2_cols = [col for col in adata.var.columns if "cat2" in col and "is_selected" in col and "mean" in col]
    assert cat2_cols, f"No columns containing both 'cat2' and 'is_selected' found in: {list(adata.var.columns)}"
    subset2_lfc = cat2_cols[0]
    
    # Check that Mahalanobis distances are also computed for each subset
    # Find mahalanobis columns for cat1
    cat1_mah_cols = [col for col in adata.var.columns if "cat1" in col and "mahalanobis" in col.lower()]
    assert cat1_mah_cols, f"No mahalanobis columns containing 'cat1' found in: {list(adata.var.columns)}"
    subset1_mah = cat1_mah_cols[0]
    
    # Find mahalanobis columns for cat2 + is_selected
    cat2_mah_cols = [col for col in adata.var.columns if "cat2" in col and "is_selected" in col and "mahalanobis" in col.lower()]
    assert cat2_mah_cols, f"No mahalanobis columns containing both 'cat2' and 'is_selected' found in: {list(adata.var.columns)}"
    subset2_mah = cat2_mah_cols[0]
    
    # Check that field_mapping includes subset fields
    run_info = adata.uns['kompot_de']['last_run_info']
    assert 'field_mapping' in run_info
    assert subset1_lfc in run_info['field_mapping']
    assert subset2_lfc in run_info['field_mapping']
    assert subset1_mah in run_info['field_mapping']
    assert subset2_mah in run_info['field_mapping']
    
    # Check field tracking
    assert 'anndata_fields' in adata.uns['kompot_de']
    field_tracking = adata.uns['kompot_de']['anndata_fields']
    assert 'var' in field_tracking
    
    # Check that subset fields are tracked
    for field in [subset1_lfc, subset2_lfc, subset1_mah, subset2_mah]:
        assert field in field_tracking['var']


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
    
    # Check that results were computed successfully for both subsets
    large_subset_cols = [col for col in adata.var.columns if "subset1" in col and "mean" in col]
    assert large_subset_cols, f"No columns for large subset (subset1) found"
    large_subset_field = large_subset_cols[0]
    
    small_subset_cols = [col for col in adata.var.columns if "subset2" in col and "mean" in col]
    assert small_subset_cols, f"No columns for small subset (subset2) found"
    small_subset_field = small_subset_cols[0]
    
    # Check that mahalanobis distances were computed for both subsets
    large_subset_mah_cols = [col for col in adata.var.columns if "subset1" in col and "mahalanobis" in col.lower()]
    assert large_subset_mah_cols, f"No mahalanobis columns for large subset (subset1) found"
    small_subset_mah_cols = [col for col in adata.var.columns if "subset2" in col and "mahalanobis" in col.lower()]
    assert small_subset_mah_cols, f"No mahalanobis columns for small subset (subset2) found"
    
    # Check that the fields have values (not all NaN)
    assert not pd.isna(adata.var[large_subset_field]).all()
    assert not pd.isna(adata.var[small_subset_field]).all()


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
    
    # Print all columns to help debug
    print(f"Available columns for weighted test: {list(adata.var.columns)}")
    
    # Look for any column that contains "weighted" for the global field
    matching_cols = [col for col in adata.var.columns if "weighted" in col and "_A_to_B" in col and not any(cat in col for cat in categories)]
    assert matching_cols, f"No weighted LFC columns found in: {list(adata.var.columns)}"
    global_weighted_lfc = matching_cols[0]
    
    print(f"Using global weighted LFC column: {global_weighted_lfc}")
    
    # Check that each category has its own weighted mean log fold change field
    for category in categories:
        # Look for any field with both weighted and category in the name
        potential_matches = [col for col in adata.var.columns 
                          if "weighted" in col and category in col]
        assert potential_matches, f"No weighted column found for category {category} in: {list(adata.var.columns)}"
        subset_field = potential_matches[0]
        
        print(f"Using subset field for {category}: {subset_field}")
        
        # Check that the field has values (not all NaN)
        assert not pd.isna(adata.var[subset_field]).all()