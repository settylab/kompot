"""Core utility functions for anndata module."""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Union, Tuple
from anndata import AnnData
import logging
import pprint
import json

logger = logging.getLogger("kompot")


def generate_output_field_names(
    result_key: str,
    condition1: str,
    condition2: str,
    analysis_type: str = "da",
    with_sample_suffix: bool = False,
    sample_suffix: str = "_sample_var"
) -> Dict[str, Any]:
    """
    Generate standardized field names for analysis outputs and create AnnData field patterns.
    
    Parameters
    ----------
    result_key : str
        Base key for results (e.g., "kompot_da", "kompot_de")
    condition1 : str
        Name of the first condition
    condition2 : str
        Name of the second condition
    analysis_type : str, optional
        Type of analysis: "da" for differential abundance or "de" for differential expression
        By default "da"
    with_sample_suffix : bool, optional
        Whether to include sample variance suffix in field names, by default False
    sample_suffix : str, optional
        Suffix to add for sample variance variants, by default "_sample_var"
        
    Returns
    -------
    Dict[str, Any]
        Dictionary mapping field types to their standardized names and AnnData field patterns
    """
    # Sanitize condition names
    cond1_safe = _sanitize_name(condition1)
    cond2_safe = _sanitize_name(condition2)
    
    # Apply suffix when sample variance is used
    suffix = sample_suffix if with_sample_suffix else ""
    
    # Basic fields for both analysis types
    field_names = {"sample_variance_impacted_fields": []}
    
    if analysis_type == "da":
        # Define which fields are actually impacted by sample variance
        # Fields like log_fold_change, log_density are not affected by sample variance
        sample_variance_impacted = ["zscore_key", "pval_key", "direction_key"]
        
        # Differential abundance field names
        field_names.update({
            "lfc_key": f"{result_key}_log_fold_change_{cond1_safe}_to_{cond2_safe}",
            "zscore_key": f"{result_key}_log_fold_change_zscore_{cond1_safe}_to_{cond2_safe}{suffix}",
            "pval_key": f"{result_key}_neg_log10_fold_change_pvalue_{cond1_safe}_to_{cond2_safe}{suffix}",
            "direction_key": f"{result_key}_log_fold_change_direction_{cond1_safe}_to_{cond2_safe}{suffix}",
            "density_key_1": f"{result_key}_log_density_{cond1_safe}",
            "density_key_2": f"{result_key}_log_density_{cond2_safe}"
        })
        field_names["sample_variance_impacted_fields"] = sample_variance_impacted

        # Generate all_patterns for DA - all metrics are in obs
        field_names["all_patterns"] = {
            "obs": [
                field_names["lfc_key"],        # Not impacted by sample variance
                field_names["zscore_key"],     # Impacted by sample variance
                field_names["pval_key"],       # Impacted by sample variance
                field_names["direction_key"],  # Impacted by sample variance
                field_names["density_key_1"],  # Not impacted by sample variance
                field_names["density_key_2"]   # Not impacted by sample variance
            ]
        }
        
    elif analysis_type == "de":
        # Define which fields are actually impacted by sample variance
        # Fields like mean_lfc, bidirectionality, imputed data, fold_change are not affected by sample variance
        sample_variance_impacted = ["mahalanobis_key", "lfc_std_key", "mahalanobis_varm_key", "std_key_1", "std_key_2", "fold_change_zscores_key"]
        
        # Differential expression field names
        field_names.update({
            "mahalanobis_key": f"{result_key}_mahalanobis_{cond1_safe}_to_{cond2_safe}{suffix}",
            "mean_lfc_key": f"{result_key}_mean_lfc_{cond1_safe}_to_{cond2_safe}",
            "weighted_lfc_key": f"{result_key}_weighted_lfc_{cond1_safe}_to_{cond2_safe}",
            "lfc_std_key": f"{result_key}_lfc_std_{cond1_safe}_to_{cond2_safe}{suffix}",
            "bidirectionality_key": f"{result_key}_bidirectionality_{cond1_safe}_to_{cond2_safe}",
            "imputed_key_1": f"{result_key}_imputed_{cond1_safe}",
            "imputed_key_2": f"{result_key}_imputed_{cond2_safe}",
            "fold_change_key": f"{result_key}_fold_change_{cond1_safe}_to_{cond2_safe}",
            "fold_change_zscores_key": f"{result_key}_fold_change_zscores_{cond1_safe}_to_{cond2_safe}{suffix}",
            "std_key_1": f"{result_key}_{cond1_safe}_std",
            "std_key_2": f"{result_key}_{cond2_safe}_std",
            
            # Add varm field names for group-specific metrics
            "mean_lfc_varm_key": f"{result_key}_mean_lfc_{cond1_safe}_to_{cond2_safe}_groups",
            "mahalanobis_varm_key": f"{result_key}_mahalanobis_{cond1_safe}_to_{cond2_safe}{suffix}_groups",
            "weighted_lfc_varm_key": f"{result_key}_weighted_lfc_{cond1_safe}_to_{cond2_safe}_groups"
        })
        field_names["sample_variance_impacted_fields"] = sample_variance_impacted

        # Generate all_patterns for DE
        field_names["all_patterns"] = {
            "var": [
                field_names["mahalanobis_key"],      # Impacted by sample variance
                field_names["mean_lfc_key"],         # Not impacted by sample variance
                field_names["bidirectionality_key"], # Not impacted by sample variance
                field_names["lfc_std_key"]           # Impacted by sample variance
            ],
            "layers": [
                field_names["imputed_key_1"],        # Not impacted by sample variance
                field_names["imputed_key_2"],        # Not impacted by sample variance
                field_names["fold_change_key"],      # Not impacted by sample variance
                field_names["fold_change_zscores_key"] # Impacted by sample variance
            ]
        }
        
        # Conditionally add fields to all_patterns based on analysis details
        # For standard deviation tracking based on sample variance
        if with_sample_suffix:
            # With sample variance, track in layers
            field_names["all_patterns"]["layers"].append(field_names["std_key_1"])
            field_names["all_patterns"]["layers"].append(field_names["std_key_2"])
        else:
            # Without sample variance, track in obs
            if "obs" not in field_names["all_patterns"]:
                field_names["all_patterns"]["obs"] = []
            field_names["all_patterns"]["obs"].append(field_names["std_key_1"])
            field_names["all_patterns"]["obs"].append(field_names["std_key_2"])
        
        # For weighted log fold change (only if differential abundance integration is used)
        field_names["has_weighted_lfc"] = False  # Initialize flag
        
        # For group-specific metrics
        field_names["has_groups"] = False  # Initialize flag
        
    else:
        raise ValueError(f"Unknown analysis_type: {analysis_type}. Use 'da' or 'de'.")
    
    return field_names


def get_environment_info() -> Dict[str, str]:
    """
    Get information about the current execution environment.
    
    Returns
    -------
    Dict[str, str]
        Dictionary with environment information
    """
    from datetime import datetime
    import platform
    import getpass
    import socket
    import os
    
    try:
        hostname = socket.gethostname()
    except:
        hostname = "unknown"
        
    try:
        username = getpass.getuser()
    except:
        username = "unknown"
        
    env_info = {
        "timestamp": datetime.now().isoformat(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "hostname": hostname,
        "username": username,
        "pid": os.getpid()
    }
    
    # Try to get package version if available
    try:
        from kompot.version import __version__
        env_info["kompot_version"] = __version__
    except ImportError:
        try:
            # Alternative way to get version
            import pkg_resources
            env_info["kompot_version"] = pkg_resources.get_distribution("kompot").version
        except:
            env_info["kompot_version"] = "unknown"
        
    return env_info




def detect_output_field_overwrite(
    adata: AnnData, 
    result_key: str, 
    output_patterns: List[str],
    location: str = "obs",
    result_type: str = "results",
    with_sample_suffix: bool = False,
    sample_suffix: str = "_sample_var",
    analysis_type: str = "da"
) -> Tuple[bool, List[str], Optional[Dict[str, Any]]]:
    """
    Detects if we would overwrite existing output fields in an AnnData object.
    This function scans AnnData object for output fields that match the given patterns
    and looks through run history to find previous runs that might have created them.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object to check for existing fields
    result_key : str
        Key under which results are stored (used for field generation, not storage location)
    output_patterns : List[str]
        Patterns of output field names to check for.
    location : str, optional
        Location to check for field patterns (e.g., "obs", "var", "layers"), by default "obs"
    result_type : str, optional
        Description of the results for warning/error messages, by default "results"
    with_sample_suffix : bool, optional
        Whether to also check for patterns with sample suffix, by default False
    sample_suffix : str, optional
        Suffix to add when checking for sample variance variants, by default "_sample_var"
    analysis_type : str, optional
        Type of analysis ("da" or "de"), determines where run history is stored, by default "da"
        
    Returns
    -------
    Tuple[bool, List[str], Optional[Dict[str, Any]]]
        - Boolean indicating if any fields would be overwritten
        - List of field names that would be overwritten
        - Previous run info if found in run history, otherwise None
        
    Notes
    -----
    Run history is stored in adata.uns["kompot_da"] or adata.uns["kompot_de"] based on analysis_type.
    This function will look in those fixed locations rather than using result_key for storage location.
    """
    existing_fields = []
    
    # Get the object to check for patterns based on location
    if location == "obs":
        obj_to_check = adata.obs
    elif location == "var":
        obj_to_check = adata.var
    elif location == "layers":
        obj_to_check = adata.layers
    elif location == "varm":
        obj_to_check = adata.varm
    else:
        raise ValueError(f"Unknown location: {location}. Use 'obs', 'var', 'layers', or 'varm'")
    
    # Check for patterns in the specified location
    if hasattr(obj_to_check, 'columns'):  # DataFrame-like (obs or var)
        for pattern in output_patterns:
            for column in obj_to_check.columns:
                if column.startswith(pattern):
                    existing_fields.append(f"{location}:{column}")
                    logger.debug(f"Found existing field to be overwritten: {location}:{column}")
                    break
                    
            # Also check with sample suffix if requested
            if with_sample_suffix:
                for column in obj_to_check.columns:
                    if column.startswith(pattern + sample_suffix):
                        existing_fields.append(f"{location}:{column}")
                        logger.debug(f"Found existing field with sample suffix to be overwritten: {location}:{column}")
                        break
    
    else:  # dict-like (layers)
        for pattern in output_patterns:
            for key in obj_to_check.keys():
                if key.startswith(pattern):
                    existing_fields.append(f"{location}:{key}")
                    logger.debug(f"Found existing field to be overwritten: {location}:{key}")
                    break
                    
            # Also check with sample suffix if requested
            if with_sample_suffix:
                for key in obj_to_check.keys():
                    if key.startswith(pattern + sample_suffix):
                        existing_fields.append(f"{location}:{key}")
                        logger.debug(f"Found existing field with sample suffix to be overwritten: {location}:{key}")
                        break
    
    # Infer analysis_type from result_key if not provided
    if analysis_type is None:
        if "da" in result_key:
            analysis_type = "da"
        elif "de" in result_key:
            analysis_type = "de"
    
    # Look for matching run in run history - we'll check both specific and global locations with a single call
    previous_run = None
    
    # First try to get the run from the analysis-specific history using the analysis_type 
    if analysis_type:
        previous_run = get_run_from_history(adata, run_id=-1, analysis_type=analysis_type)
    
    # If no previous run found and we have a global history, check there as a fallback
    if previous_run is None and 'kompot_run_history' in adata.uns:
        # Look for most recent run with matching analysis_type in global history
        matching_runs = []
        for i, run in enumerate(adata.uns['kompot_run_history']):
            if run.get('analysis_type') == analysis_type:
                matching_runs.append((i, run))
        
        if matching_runs:
            # Get the most recent matching run
            previous_run = matching_runs[-1][1]
    
    # Return a tuple with detection results
    return (len(existing_fields) > 0, existing_fields, previous_run)


def _sanitize_name(name):
    """Convert a string to a valid column/key name.

    Args:
        name: String to convert.

    Returns:
        String with invalid characters replaced.
    """
    return "".join([c if c.isalnum() else "_" for c in name])


def refine_filter_for_underrepresentation(
    adata,
    filter_mask: np.ndarray,
    groupby: str,
    groups: Optional[Union[str, Dict[str, Any], List[Dict[str, Any]], pd.Series, np.ndarray, List[np.ndarray]]],
    conditions: Optional[List[str]] = None,
    min_cells: int = 10,
    min_percentage: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, Any], int]:
    """
    Check for underrepresentation in filtered data and refine the filter if needed.
    
    This function applies a filter mask to an AnnData object, checks for underrepresentation
    in the filtered data, and returns an updated filter mask that excludes any newly 
    detected underrepresented groups.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object
    filter_mask : np.ndarray
        Boolean mask of cells to include (True) or exclude (False)
    groupby : str
        Column in adata.obs containing the condition labels
    groups : str, Dict, List[Dict], pd.Series, np.ndarray, List[np.ndarray], optional
        Specification for the groups to check for underrepresentation
    conditions : List[str], optional
        List of condition values to check
    min_cells : int, optional
        Minimum number of cells required for a condition, by default 10
    min_percentage : float, optional
        Minimum percentage of cells required for a condition, by default None
        
    Returns
    -------
    Tuple[np.ndarray, Dict[str, Any], int]
        - Updated filter mask
        - Dictionary of underrepresentation data 
        - Number of additional cells excluded
    """
    
    # Create a view on the filtered data
    # This avoids making a full copy
    adata_view = adata[filter_mask]
    
    # Get original indices for mapping back
    filtered_indices = np.where(filter_mask)[0]
    
    # Check for underrepresentation on filtered data
    logger.info("Checking for underrepresentation on filtered cells")
    underrep_result = check_underrepresentation(
        adata_view, 
        groupby=groupby, 
        groups=groups,
        conditions=conditions,
        min_cells=min_cells,
        min_percentage=min_percentage,
        warn=False,  # Don't warn, we'll handle logging
        print_summary=False  # Don't print summary, we'll handle logging
    )
    
    # Extract underrepresentation data
    underrep_data = {}
    if "__underrepresentation_data" in underrep_result:
        underrep_data = underrep_result.pop("__underrepresentation_data")
    
    # If no underrepresentation found, return original filter
    if not underrep_data:
        return filter_mask, underrep_data, 0
    
    # Log what groups would be filtered
    n_groups = len(underrep_data)
    logger.info(f"Found {n_groups} groups with underrepresented conditions in filtered cells")
    for group, conditions in underrep_data.items():
        logger.info(f"  - Group '{group}': Underrepresented conditions: {conditions}")
    
    # Create a mask for the filtered data
    refined_mask = np.ones(len(adata_view), dtype=bool)
    
    # Apply the filter to get an additional filter mask
    additional_mask, additional_excluded = apply_cell_filter(adata_view, underrep_result, groups)
    refined_mask = refined_mask & additional_mask
    
    # Create updated filter mask for original data
    updated_filter = filter_mask.copy()
    
    # Map refined mask back to original indices
    excluded_count = 0
    for i, include in enumerate(additional_mask):
        if not include:
            # This filtered cell should be excluded
            orig_idx = filtered_indices[i]
            updated_filter[orig_idx] = False
            excluded_count += 1
    
    if excluded_count > 0:
        logger.info(f"Refined filter excluded additional {excluded_count:,} cells")
    
    return updated_filter, underrep_data, excluded_count

def check_underrepresentation(
    adata,
    groupby: str, 
    groups: Optional[Union[str, Dict[str, Any], List[Dict[str, Any]], pd.Series, np.ndarray, List[np.ndarray]]],
    conditions: Optional[List[str]] = None,
    min_cells: int = 3,
    min_percentage: Optional[float] = None,
    warn: bool = False,
    print_summary: bool = True
) -> Dict[str, Union[List[str], Dict[str, List[str]]]]:
    """
    Check if any condition is underrepresented within the specified groups.
    
    This function checks each unique value in groups to see if any condition has 
    too few cells or represents a small percentage of the total cells in that group.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object
    groupby : str
        Column in adata.obs containing the condition labels
    groups : str, Dict, List[Dict], pd.Series, np.ndarray, List[np.ndarray]
        Specification for the groups to check for underrepresentation:
        - str: column name in adata.obs
        - Dict: filter with keys as column names in adata.obs and values as allowed values
        - List[Dict]: list of filters for different subgroups
        - pd.Series/np.ndarray: values to divide or subset
        - np.ndarray with boolean values: each row specifies a subset
        - List of vectors/series: multiple subsetting vectors
    conditions : List[str], optional
        List of condition values to check. If None, uses all unique values in adata.obs[groupby]
    min_cells : int, optional
        Minimum number of cells required for a condition to be considered adequately represented, by default 10
    min_percentage : float, optional
        Minimum percentage of cells required for a condition, relative to total cells in the group.
        If None, uses 10% divided by the number of conditions, by default None
    warn : bool, optional
        Whether to log warnings when underrepresentation is detected, by default False
    print_summary : bool, optional
        Whether to print a summary report when underrepresentation is detected, by default True
        
    Returns
    -------
    Dict
        A dictionary structured for filtering in compute_differential_expression.
        
        The dictionary contains:
        - A special key '__underrepresentation_data' with the detailed findings,
          where keys are group values and values are lists of underrepresented conditions
        - A key matching the provided groups parameter (if it's a string),
          with the values being a list of groups that have underrepresentation
        
        This format is designed to be used directly with the cell_filter parameter in
        compute_differential_expression
        
    Notes
    -----
    If groups is specified and underrepresentation is detected, the function will log a warning 
    and suggest using the returned dictionary as a filter for differential expression analysis.
    
    Example
    -------
    >>> import anndata as ad
    >>> import pandas as pd
    >>> import numpy as np
    >>> import kompot as kp
    >>>
    >>> # Create example data
    >>> adata = ad.AnnData(X=np.random.normal(0, 1, (100, 10)))
    >>> adata.obs['condition'] = ['A'] * 70 + ['B'] * 30
    >>> adata.obs['group'] = ['group1'] * 50 + ['group2'] * 50
    >>>
    >>> # Check for underrepresentation
    >>> underrep = kp.check_underrepresentation(
    ...     adata,
    ...     groupby='condition', 
    ...     groups='group', 
    ...     min_cells=20
    ... )
    >>> print(underrep)
    {'group': ['group1']}
    >>>
    >>> # Use result as filter in differential expression
    >>> result = kp.compute_differential_expression(
    ...     adata,
    ...     groupby='condition',
    ...     condition1='A',
    ...     condition2='B',
    ...     groups='group',
    ...     cell_filter=underrep
    ... )
    """
    
    # Check if groupby column exists
    if groupby not in adata.obs:
        raise ValueError(f"Column '{groupby}' not found in adata.obs. Available columns: {list(adata.obs.columns)}")
    
    # Get conditions if not provided
    if conditions is None:
        conditions = list(adata.obs[groupby].unique())
    
    # Set default min_percentage if not provided
    if min_percentage is None:
        min_percentage = 4 / len(conditions)
    
    # Initialize result dictionary
    underrepresented = {}
    
    # If no groups specified, just check overall representation
    if groups is None:
        return {}
    
    # Parse the groups parameter to get subset masks
    subset_masks, subset_names = parse_groups(adata, groups)
    
    # Check each group for underrepresentation
    for group_name, mask in subset_masks.items():
        # Count cells in this group
        group_total = np.sum(mask)
        
        if group_total == 0:
            logger.warning(f"Group '{group_name}' has no cells, skipping.")
            continue
        
        # Check each condition
        underrep_conditions = []
        for condition in conditions:
            # Create condition mask
            condition_mask = (adata.obs[groupby] == condition).values
            
            # Count cells that are both in this group and this condition
            cells_in_condition = np.sum(mask & condition_mask)
            
            # Calculate percentage
            percentage = (cells_in_condition / group_total) * 100
            
            # Check if underrepresented
            if cells_in_condition < min_cells or percentage < min_percentage:
                underrep_conditions.append(condition)
                
                if warn:
                    logger.warning(
                        f"Condition '{condition}' is underrepresented in group '{group_name}': "
                        f"{cells_in_condition} cells ({percentage:.2f}% of group). "
                        f"Min required: {min_cells} cells or {min_percentage:.2f}%."
                    )
        
        # If any conditions are underrepresented, add to result
        if underrep_conditions:
            underrepresented[group_name] = underrep_conditions
    
    # Print summary report
    if underrepresented and print_summary:
        print(f"\n{'='*80}\nUNDERREPRESENTATION REPORT\n{'='*80}")
        print(f"Found {len(underrepresented)} groups with underrepresented conditions.")
        
        for group, conditions_list in underrepresented.items():
            print(f"\n- Group: {group}")
            print(f"  Underrepresented conditions: {', '.join(conditions_list)}")
            
            # Show detailed counts
            group_mask = subset_masks[group]
            group_total = np.sum(group_mask)
            
            # Table header
            print(f"\n  {'Condition':<20} {'Count':<10} {'Percentage':<15} {'Status':<15}")
            print(f"  {'-'*60}")
            
            # Show stats for all conditions
            for condition in conditions:
                condition_mask = (adata.obs[groupby] == condition).values
                cells_in_condition = np.sum(group_mask & condition_mask)
                percentage = (cells_in_condition / group_total) * 100
                
                # Determine status
                if condition in conditions_list:
                    status = "UNDERREPRESENTED"
                else:
                    status = "OK"
                
                print(f"  {condition:<20} {cells_in_condition:<10} {percentage:.2f}%{' '*9} {status:<15}")
        
        # Suggestion for filtering
        print(f"\n{'='*80}")
        print("RECOMMENDATION:")
        logger.info("The detected underrepresentation may affect differential expression results.")
        logger.info("Consider filtering these groups with the returned dictionary when running differential analysis:")
        logger.info("Example: adata.compute_differential_expression(..., cell_filter=underrepresented)")
        logger.info(f"{'='*80}\n")
    
    # Create a filter that's compatible with compute_differential_expression
    # For direct use as cell_filter, we need to return a dictionary where:
    # - Keys are column names in adata.obs
    # - Values are values to exclude
    filter_dict = {}
    
    if underrepresented:
        # Store the original data format for reference
        filter_dict["__underrepresentation_data"] = underrepresented
        
        # Create a filter format for compute_differential_expression
        # We'll use the groupby column and the groups parameter to create a filter
        if isinstance(groups, str):
            # If groups is a column name, filter by both group column and condition
            for group, conditions_list in underrepresented.items():
                # We need to exclude cells that match BOTH this group AND any underrepresented conditions
                filter_dict[groups] = filter_dict.get(groups, []) + [group]
        else:
            # For other group types, we'll use the group name as provided in the result
            # This is less precise but still gives a usable filter
            for group, conditions_list in underrepresented.items():
                filter_dict[groupby] = filter_dict.get(groupby, []) + conditions_list
    
    return filter_dict

def parse_groups(adata, groups, formatted_names=False):
    """
    Parse various group specifications into a dictionary of subset masks.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object
    groups : str, Dict, Dict[str, Dict], List[Dict], pd.Series, np.ndarray, List[np.ndarray]
        Group specification for subsetting. Can be:
        - str: column name in adata.obs (creates a subset for each unique value)
        - Dict: filter with keys as column names in adata.obs and values as allowed values
          Example: {'category': 'cat1', 'is_selected': True} creates a subset of cells where 
          category is 'cat1' AND is_selected is True
        - Dict[str, Dict]: dict of filters for different subgroups, where outer dict keys are 
          used as subset names. Each inner dict defines a filter as above.
          Example: {'group1': {'category': 'cat1'}, 'group2': {'category': 'cat2', 'is_selected': True}}
          creates two named subsets: 'group1' for cat1 cells and 'group2' for cat2 cells that are also selected
        - List[Dict]: list of filters for different subgroups (similar to Dict[str, Dict] but with 
          auto-generated names)
        - pd.Series/np.ndarray: values to divide or subset (creates a subset for each unique value)
        - np.ndarray with boolean values: each row specifies a subset
        - List of vectors/series: multiple subsetting vectors
    formatted_names : bool, optional
        If True, returns more human-readable names for automatically generated subset names.
        If False (default), returns the sanitized machine-friendly names.
        
    Returns
    -------
    Tuple[Dict[str, np.ndarray], List[str]]
        - Dictionary of subset masks, with keys as subset names and values as boolean masks
        - List of subset names in the order they were defined
    
    Raises
    ------
    ValueError
        If groups cannot be interpreted for subsetting, or if column does not exist
    
    Examples
    --------
    >>> # Using a column name to create subsets for each category
    >>> subset_masks, subset_names = parse_groups(adata, 'category')
    >>> 
    >>> # Using a dictionary to filter cells
    >>> subset_masks, subset_names = parse_groups(adata, {'category': 'A', 'is_selected': True})
    >>>
    >>> # Using a dictionary of dictionaries for named filters
    >>> named_filters = {
    ...     'control_group': {'treatment': 'control'},
    ...     'treated_high_dose': {'treatment': 'drug', 'dose': 'high'}
    ... }
    >>> subset_masks, subset_names = parse_groups(adata, named_filters)
    >>> # subset_names will be ['control_group', 'treated_high_dose']
    >>>
    >>> # Using formatted names for display
    >>> subset_masks, subset_names = parse_groups(adata, {'category': 'A'}, formatted_names=True)
    >>> # Will return "Category: A" instead of "category=A"
    """
    subset_masks = {}
    subset_names = []
    
    # Case 1: String (column name in adata.obs)
    if isinstance(groups, str):
        group_col = groups
        if group_col not in adata.obs:
            raise ValueError(f"Column '{group_col}' not found in adata.obs")
        
        col_dtype = adata.obs[group_col].dtype
        col_values = adata.obs[group_col]
        
        # Boolean column - single subset of True values
        if pd.api.types.is_bool_dtype(col_dtype):
            subset_masks["True"] = col_values.values
            display_name = f"{group_col.capitalize()}: True" if formatted_names else "True"
            subset_names.append(display_name)
        # Categorical or string column - subset for each category
        elif isinstance(col_dtype, pd.CategoricalDtype) or pd.api.types.is_string_dtype(col_dtype):
            for category in adata.obs[group_col].unique():
                mask = (adata.obs[group_col] == category).values
                subset_name = str(category)
                subset_masks[subset_name] = mask
                
                # Format the name if requested
                if formatted_names:
                    display_name = f"{group_col.capitalize()}: {subset_name}"
                else:
                    display_name = subset_name
                
                subset_names.append(display_name)
        # Float column - not valid for grouping
        elif pd.api.types.is_float_dtype(col_dtype):
            raise ValueError(f"Column '{group_col}' has float values which cannot be used for grouping")
        else:
            # Try to convert to categories and use those for grouping
            try:
                for category in adata.obs[group_col].unique():
                    mask = (adata.obs[group_col] == category).values
                    subset_name = str(category)
                    subset_masks[subset_name] = mask
                    
                    # Format the name if requested
                    if formatted_names:
                        display_name = f"{group_col.capitalize()}: {subset_name}"
                    else:
                        display_name = subset_name
                    
                    subset_names.append(display_name)
            except Exception as e:
                raise ValueError(f"Cannot interpret column '{group_col}' for grouping: {str(e)}")
    
    # Case 2: Dictionary (filter on obs columns)
    elif isinstance(groups, dict):
        # Check if this is a dict of dicts (Case 2b) or a regular dict filter (Case 2a)
        if all(isinstance(value, dict) for value in groups.values()):
            # Case 2b: Dict of dicts (named filters)
            for name, group_dict in groups.items():
                mask = np.ones(adata.n_obs, dtype=bool)
                filter_desc = []
                
                for col, values in group_dict.items():
                    if col not in adata.obs:
                        raise ValueError(f"Column '{col}' not found in adata.obs")
                    
                    # Convert single value to list for uniform handling
                    if not isinstance(values, (list, tuple, np.ndarray, pd.Series)):
                        values = [values]
                    
                    # Create a submask for each value
                    submask = np.zeros(adata.n_obs, dtype=bool)
                    for value in values:
                        submask |= (adata.obs[col] == value).values
                    
                    mask &= submask
                    filter_desc.append(f"{col}={','.join(map(str, values))}")
                
                # Use the provided name as the subset name
                subset_name = name
                subset_masks[subset_name] = mask
                
                # No formatting for explicitly named groups, as they're already named by the user
                subset_names.append(subset_name)
        else:
            # Case 2a: Regular dict (single filter)
            mask = np.ones(adata.n_obs, dtype=bool)
            filter_desc = []
            
            for col, values in groups.items():
                if col not in adata.obs:
                    raise ValueError(f"Column '{col}' not found in adata.obs")
                
                # Convert single value to list for uniform handling
                if not isinstance(values, (list, tuple, np.ndarray, pd.Series)):
                    values = [values]
                
                # Create a submask for each value
                submask = np.zeros(adata.n_obs, dtype=bool)
                for value in values:
                    submask |= (adata.obs[col] == value).values
                
                mask &= submask
                filter_desc.append(f"{col}={','.join(map(str, values))}")
            
            # Create sanitized key for the dictionary
            sanitized_name = "_".join(filter_desc)
            
            # Create a more readable version if formatted_names is True
            if formatted_names:
                formatted_filters = []
                for desc in filter_desc:
                    col, val = desc.split('=', 1)
                    formatted_filters.append(f"{col.capitalize()}: {val}")
                display_name = " & ".join(formatted_filters)
            else:
                display_name = sanitized_name
            
            subset_masks[sanitized_name] = mask
            subset_names.append(display_name)
    
    # Case 3: List of dictionaries (multiple filters)
    elif isinstance(groups, list) and all(isinstance(g, dict) for g in groups):
        for i, group_dict in enumerate(groups):
            mask = np.ones(adata.n_obs, dtype=bool)
            filter_desc = []
            
            for col, values in group_dict.items():
                if col not in adata.obs:
                    raise ValueError(f"Column '{col}' not found in adata.obs")
                
                # Convert single value to list for uniform handling
                if not isinstance(values, (list, tuple, np.ndarray, pd.Series)):
                    values = [values]
                
                # Create a submask for each value
                submask = np.zeros(adata.n_obs, dtype=bool)
                for value in values:
                    submask |= (adata.obs[col] == value).values
                
                mask &= submask
                filter_desc.append(f"{col}={','.join(map(str, values))}")
            
            # Create sanitized key for the dictionary
            sanitized_name = f"group{i+1}" if not filter_desc else "_".join(filter_desc)
            
            # Create a more readable version if formatted_names is True
            if formatted_names and filter_desc:
                formatted_filters = []
                for desc in filter_desc:
                    col, val = desc.split('=', 1)
                    formatted_filters.append(f"{col.capitalize()}: {val}")
                display_name = " & ".join(formatted_filters)
            elif formatted_names:
                display_name = f"Group {i+1}"
            else:
                display_name = sanitized_name
            
            subset_masks[sanitized_name] = mask
            subset_names.append(display_name)
    
    # Case 4: 2D Array of boolean masks - needs to be checked BEFORE the 1D array case
    elif isinstance(groups, np.ndarray) and groups.ndim == 2:
        # First check if it's a boolean array (which is what we're looking for)
        if pd.api.types.is_bool_dtype(groups.dtype) or np.all(np.isin(groups, [0, 1, True, False])):
            # Check shapes to ensure it's a 2D array of masks
            n_subsets, n_cells = groups.shape
            
            # If the first dimension is larger, assume it's an array of observations, not masks
            if n_subsets > n_cells:
                # This is likely not a set of masks, but an array of values (features x cells)
                raise ValueError(f"2D array with shape {groups.shape} doesn't match expected mask format. "
                               f"The first dimension ({n_subsets}) should be the number of masks and "
                               f"the second dimension ({n_cells}) should match the number of cells ({adata.n_obs}).")
                
            # Check if the second dimension matches the number of cells
            if n_cells != adata.n_obs:
                raise ValueError(f"2D array of shape {groups.shape} doesn't match the number of cells ({adata.n_obs})")
            
            # Each row is a different subset
            for i in range(n_subsets):
                sanitized_name = f"subset{i+1}"
                # Create a more readable version if formatted_names is True
                display_name = f"Subset {i+1}" if formatted_names else sanitized_name
                
                # Convert to boolean array if not already
                mask = groups[i].astype(bool)
                subset_masks[sanitized_name] = mask
                subset_names.append(display_name)
                
    # Case 5: Series or array (like a column)
    elif isinstance(groups, (pd.Series, np.ndarray)):
        # Ensure it's the right shape
        if len(groups) != adata.n_obs:
            raise ValueError(f"Length of groups ({len(groups)}) doesn't match number of cells ({adata.n_obs})")
        
        # Handle boolean mask directly
        if pd.api.types.is_bool_dtype(groups.dtype):
            subset_masks["True"] = np.array(groups)
            display_name = "Selected" if formatted_names else "True"
            subset_names.append(display_name)
        else:
            # Use unique values to create subsets
            unique_values = np.unique(groups)
            for value in unique_values:
                if isinstance(groups, pd.Series):
                    mask = (groups == value).values
                else:
                    mask = (groups == value)
                subset_name = str(value)
                
                # Create a more readable display name if requested
                if formatted_names and isinstance(groups, pd.Series) and groups.name is not None:
                    display_name = f"{groups.name.capitalize()}: {subset_name}"
                elif formatted_names:
                    display_name = f"Value: {subset_name}"
                else:
                    display_name = subset_name
                
                subset_masks[subset_name] = mask
                subset_names.append(display_name)
    
    # Case 6: List of arrays/series
    elif isinstance(groups, list) and all(isinstance(g, (np.ndarray, pd.Series)) for g in groups):
        for i, group_arr in enumerate(groups):
            # Check length
            if len(group_arr) != adata.n_obs:
                raise ValueError(f"Length of group {i+1} ({len(group_arr)}) doesn't match number of cells ({adata.n_obs})")
            
            # Handle boolean mask directly
            if pd.api.types.is_bool_dtype(group_arr.dtype):
                sanitized_name = f"subset{i+1}"
                # Create a more readable display name if requested
                display_name = f"Subset {i+1}" if formatted_names else sanitized_name
                
                subset_masks[sanitized_name] = np.array(group_arr)
                subset_names.append(display_name)
            else:
                # Use unique values to create subsets
                unique_values = np.unique(group_arr)
                for value in unique_values:
                    if isinstance(group_arr, pd.Series):
                        mask = (group_arr == value).values
                    else:
                        mask = (group_arr == value)
                    sanitized_name = f"subset{i+1}_{value}"
                    
                    # Create a more readable display name if requested
                    if formatted_names and isinstance(group_arr, pd.Series) and group_arr.name is not None:
                        display_name = f"{group_arr.name.capitalize()} {i+1}: {value}"
                    elif formatted_names:
                        display_name = f"Subset {i+1}: {value}"
                    else:
                        display_name = sanitized_name
                    
                    subset_masks[sanitized_name] = mask
                    subset_names.append(display_name)
    
    # If we couldn't interpret the groups parameter
    else:
        raise ValueError(
            "Cannot interpret 'groups' parameter. It should be a string (column name), "
            "a dictionary (filter), a dictionary of dictionaries (named filters), "
            "a list of dictionaries (multiple filters), a Series/array (like a column), "
            "an array of boolean masks, or a list of arrays/series."
        )
        
    return subset_masks, subset_names


class RunComparison:
    """
    Class to display comparison results between two runs.
    
    This class provides a convenient interface to examine differences
    between two differential analysis runs, with nice formatting for
    both terminal and Jupyter notebook environments.
    
    Attributes
    ----------
    adata : AnnData
        AnnData object containing the run information
    this_run_id : int
        Run ID for the first run
    other_run_id : int
        Run ID for the second run
    analysis_type : str
        Type of analysis ('de' or 'da')
    parameter_differences : Dict[str, Dict[str, Any]]
        Dictionary of parameter differences
    field_differences : Dict[str, Dict[str, List[str]]]
        Dictionary of field differences by location
    this_run_adjusted_id : int
        The adjusted (positive) run ID for the first run
    other_run_adjusted_id : int
        The adjusted (positive) run ID for the second run
    overwritten_fields : List[Dict[str, Any]]
        List of fields that were overwritten by one run from the other
    """
    
    def __init__(self, 
                 adata, 
                 run_id1: int, 
                 run_id2: int, 
                 analysis_type: str):
        """
        Initialize a RunComparison object.
        
        Parameters
        ----------
        adata : AnnData
            AnnData object containing run history
        run_id1 : int
            First run ID to compare
        run_id2 : int
            Second run ID to compare
        analysis_type : str
            Type of analysis: 'de' for differential expression or 
            'da' for differential abundance
        """
        self.adata = adata
        self.this_run_id = run_id1
        self.other_run_id = run_id2
        self.analysis_type = analysis_type
        self.storage_key = f"kompot_{analysis_type}"

        # Get the run info for the first run
        this_run_info = get_run_from_history(adata, run_id=run_id1, analysis_type=analysis_type)
        if this_run_info is None:
            raise ValueError(f"Run ID {run_id1} not found in {analysis_type} run history.")
        
        # Get the run info for the second run
        other_run_info = get_run_from_history(adata, run_id=run_id2, analysis_type=analysis_type)
        if other_run_info is None:
            raise ValueError(f"Run ID {run_id2} not found in {analysis_type} run history.")
        
        # Make sure we use adjusted run IDs
        self.this_run_adjusted_id = this_run_info.get('adjusted_run_id', run_id1)
        self.other_run_adjusted_id = other_run_info.get('adjusted_run_id', run_id2)
        
        # Store timestamps for display
        self.this_timestamp = this_run_info.get('timestamp', '')
        self.other_timestamp = other_run_info.get('timestamp', '')
        
        # Compare parameters
        param_comparison = {}
        this_params = this_run_info.get('params', {})
        other_params = other_run_info.get('params', {})
        all_params = set(list(this_params.keys()) + list(other_params.keys()))
        
        for param in all_params:
            this_value = this_params.get(param, None)
            other_value = other_params.get(param, None)
            
            # Special handling for array-like parameters
            values_equal = False
            
            # Check if either value is None
            if this_value is None or other_value is None:
                values_equal = (this_value is None and other_value is None)
            # If both are array-like
            elif hasattr(this_value, '__len__') and not isinstance(this_value, (str, dict)) and \
                 hasattr(other_value, '__len__') and not isinstance(other_value, (str, dict)):
                try:
                    import numpy as np
                    values_equal = np.array_equal(np.array(this_value), np.array(other_value), equal_nan=True)
                except:
                    # Fall back to list comparison if numpy is not available or comparison fails
                    try:
                        values_equal = list(this_value) == list(other_value)
                    except:
                        # If conversion to list fails, try direct comparison
                        try:
                            values_equal = (this_value == other_value)
                        except:
                            # If all else fails, assume they're different
                            values_equal = False
            else:
                # For non-array values, use direct comparison
                values_equal = (this_value == other_value)
            
            if not values_equal:
                param_comparison[param] = {
                    'this_run': this_value,
                    'other_run': other_value
                }
        
        self.parameter_differences = param_comparison
        
        # Extract field names and locations from run info
        this_field_names = this_run_info.get('field_names', {})
        other_field_names = other_run_info.get('field_names', {})
        
        # Store field mappings to know where fields were written
        self.this_field_mapping = this_run_info.get('field_mapping', {})
        self.other_field_mapping = other_run_info.get('field_mapping', {})
        
        # Get fields written by each run from tracking information
        self.this_run_fields = self._get_fields_for_run(self.this_run_adjusted_id)
        self.other_run_fields = self._get_fields_for_run(self.other_run_adjusted_id)
        
        # Calculate field differences
        self.field_differences = self._calculate_field_differences()
        
        # Find overwritten fields
        self.overwritten_fields = self._find_overwritten_fields()
    
    def _get_fields_for_run(self, run_id: int) -> Dict[str, List[str]]:
        """
        Get all fields in the AnnData object that were written by this run.
        
        Parameters
        ----------
        run_id : int
            The adjusted run ID to find fields for
            
        Returns
        -------
        Dict[str, List[str]]
            Dictionary with AnnData locations as keys and lists of field names as values
        """
        if (self.storage_key not in self.adata.uns or 
            'anndata_fields' not in self.adata.uns[self.storage_key]):
            return {}
            
        tracking = self.adata.uns[self.storage_key]['anndata_fields']
        result = {}
        
        # Initialize result for all locations
        for location in tracking.keys():
            result[location] = []
        
        # Collect fields by run ID
        for location, fields in tracking.items():
            for field, field_run_id in fields.items():
                if field_run_id == run_id:
                    result[location].append(field)
        
        return result
    
    def _calculate_field_differences(self) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """
        Calculate differences in fields between the two runs, including current field ownership.
        
        Returns
        -------
        Dict[str, Dict[str, List[Dict[str, Any]]]]
            Dictionary of field differences by location, with enhanced field information
        """
        field_differences = {}
        
        # Get all locations from both runs
        all_locations = set()
        for location in list(self.this_run_fields.keys()) + list(self.other_run_fields.keys()):
            all_locations.add(location)
        
        # Get field ownership information from anndata_fields tracking
        field_ownership = {}
        if (self.storage_key in self.adata.uns and 'anndata_fields' in self.adata.uns[self.storage_key]):
            tracking = self.adata.uns[self.storage_key]['anndata_fields']
            for location, fields in tracking.items():
                if location not in field_ownership:
                    field_ownership[location] = {}
                for field, owner_id in fields.items():
                    field_ownership[location][field] = owner_id
        
        # Compare fields for each location
        for location in all_locations:
            this_fields = set(self.this_run_fields.get(location, []))
            other_fields = set(self.other_run_fields.get(location, []))
            
            only_this = this_fields - other_fields
            only_other = other_fields - this_fields
            both = this_fields.intersection(other_fields)
            
            if only_this or only_other or both:
                field_differences[location] = {
                    'only_this_run': [],
                    'only_other_run': [],
                    'both_runs': []
                }
                
                # Process fields only in this run
                for field in sorted(list(only_this)):
                    field_info = {
                        "field": field,
                        "location": location,
                        "type": None,
                        "description": None,
                        "current_owner": None
                    }
                    
                    # Add mapping information if available
                    if field in self.this_field_mapping:
                        mapping_info = self.this_field_mapping[field]
                        field_info["type"] = mapping_info.get("type")
                        field_info["description"] = mapping_info.get("description")
                    
                    # Add current owner info
                    if location in field_ownership and field in field_ownership[location]:
                        current_owner = field_ownership[location][field]
                        field_info["current_owner"] = current_owner
                        # Is the current owner one of our compared runs?
                        if current_owner == self.this_run_adjusted_id:
                            field_info["owned_by"] = "this_run"
                        elif current_owner == self.other_run_adjusted_id:
                            field_info["owned_by"] = "other_run"
                        else:
                            field_info["owned_by"] = "different_run"
                    
                    field_differences[location]['only_this_run'].append(field_info)
                
                # Process fields only in other run
                for field in sorted(list(only_other)):
                    field_info = {
                        "field": field,
                        "location": location,
                        "type": None,
                        "description": None,
                        "current_owner": None
                    }
                    
                    # Add mapping information if available
                    if field in self.other_field_mapping:
                        mapping_info = self.other_field_mapping[field]
                        field_info["type"] = mapping_info.get("type")
                        field_info["description"] = mapping_info.get("description")
                    
                    # Add current owner info
                    if location in field_ownership and field in field_ownership[location]:
                        current_owner = field_ownership[location][field]
                        field_info["current_owner"] = current_owner
                        # Is the current owner one of our compared runs?
                        if current_owner == self.this_run_adjusted_id:
                            field_info["owned_by"] = "this_run"
                        elif current_owner == self.other_run_adjusted_id:
                            field_info["owned_by"] = "other_run"
                        else:
                            field_info["owned_by"] = "different_run"
                    
                    field_differences[location]['only_other_run'].append(field_info)
                
                # Process fields in both runs
                for field in sorted(list(both)):
                    field_info = {
                        "field": field,
                        "location": location,
                        "type": None,
                        "description": None,
                        "current_owner": None
                    }
                    
                    # Try to find info in either mapping (first this run, then other run)
                    if field in self.this_field_mapping:
                        mapping_info = self.this_field_mapping[field]
                        field_info["type"] = mapping_info.get("type")
                        field_info["description"] = mapping_info.get("description")
                    elif field in self.other_field_mapping:
                        mapping_info = self.other_field_mapping[field]
                        field_info["type"] = mapping_info.get("type")
                        field_info["description"] = mapping_info.get("description")
                    
                    # Add current owner info
                    if location in field_ownership and field in field_ownership[location]:
                        current_owner = field_ownership[location][field]
                        field_info["current_owner"] = current_owner
                        # Is the current owner one of our compared runs?
                        if current_owner == self.this_run_adjusted_id:
                            field_info["owned_by"] = "this_run"
                        elif current_owner == self.other_run_adjusted_id:
                            field_info["owned_by"] = "other_run"
                        else:
                            field_info["owned_by"] = "different_run"
                    
                    field_differences[location]['both_runs'].append(field_info)
        
        return field_differences
    
    def _find_overwritten_fields(self) -> List[Dict[str, Any]]:
        """
        Find fields that were overwritten by one run from the other.
        
        Returns
        -------
        List[Dict[str, Any]]
            List of overwritten field information, each containing:
            - field: The field name
            - location: The location in AnnData (obs, var, etc.)
            - original_run_id: The run ID that originally created the field
            - overwritten_by_run_id: The run ID that overwrote the field
        """
        if (self.storage_key not in self.adata.uns or 
            'anndata_fields' not in self.adata.uns[self.storage_key]):
            return []
            
        tracking = self.adata.uns[self.storage_key]['anndata_fields']
        overwritten = []
        
        # Get run history to check chronological order
        if 'run_history' not in self.adata.uns[self.storage_key]:
            return []
        
        run_history = self.adata.uns[self.storage_key]['run_history']
        
        # Determine which run came first
        if self.this_run_adjusted_id < self.other_run_adjusted_id:
            earlier_run_id = self.this_run_adjusted_id
            later_run_id = self.other_run_adjusted_id
        else:
            earlier_run_id = self.other_run_adjusted_id
            later_run_id = self.this_run_adjusted_id
        
        # Find fields that appear in both runs' field names
        # Go through each location and check fields
        for location in set(list(self.this_run_fields.keys()) + list(self.other_run_fields.keys())):
            if location not in tracking:
                continue
            
            # Get fields from each run at this location
            this_fields = set(self.this_run_fields.get(location, []))
            other_fields = set(self.other_run_fields.get(location, []))
            
            # Find fields in common between both runs
            common_fields = this_fields.intersection(other_fields)
            
            # Check each common field to see who currently owns it
            for field in common_fields:
                if field not in tracking[location]:
                    continue
                    
                current_owner = tracking[location][field]
                
                # Determine if it was overwritten
                if current_owner == self.this_run_adjusted_id and self.this_run_adjusted_id > self.other_run_adjusted_id:
                    # Field is owned by this_run and this_run is newer
                    overwritten.append({
                        'field': field,
                        'location': location,
                        'original_run_id': self.other_run_adjusted_id,
                        'overwritten_by_run_id': self.this_run_adjusted_id
                    })
                elif current_owner == self.other_run_adjusted_id and self.other_run_adjusted_id > self.this_run_adjusted_id:
                    # Field is owned by other_run and other_run is newer
                    overwritten.append({
                        'field': field,
                        'location': location,
                        'original_run_id': self.this_run_adjusted_id,
                        'overwritten_by_run_id': self.other_run_adjusted_id
                    })
        
        # If no overwritten fields were found, look for fields with the same name
        # This handles cases where both runs use the same result_key
        if not overwritten:
            for location in set(list(self.this_run_fields.keys()) + list(self.other_run_fields.keys())):
                if location not in tracking:
                    continue
                    
                # Get all fields in this location for both runs
                this_fields = set(self.this_run_fields.get(location, []))
                other_fields = set(self.other_run_fields.get(location, []))
                
                # Find common fields
                common_fields = this_fields.intersection(other_fields)
                
                # If both runs are in the sequence and there are common fields,
                # assume fields from the earlier run were overwritten by the later run
                if common_fields and abs(self.this_run_adjusted_id - self.other_run_adjusted_id) == 1:
                    if self.this_run_adjusted_id > self.other_run_adjusted_id:
                        # this_run is newer, it likely overwrote other_run's fields
                        for field in common_fields:
                            overwritten.append({
                                'field': field,
                                'location': location,
                                'original_run_id': self.other_run_adjusted_id,
                                'overwritten_by_run_id': self.this_run_adjusted_id
                            })
                    else:
                        # other_run is newer, it likely overwrote this_run's fields
                        for field in common_fields:
                            overwritten.append({
                                'field': field,
                                'location': location,
                                'original_run_id': self.this_run_adjusted_id,
                                'overwritten_by_run_id': self.other_run_adjusted_id
                            })
        
        return overwritten
        
    def __str__(self) -> str:
        """
        String representation of the comparison.
        
        Returns
        -------
        str
            Human-readable comparison summary
        """
        lines = [
            f"Comparison of Run {self.this_run_adjusted_id} and Run {self.other_run_adjusted_id}",
            f"Run {self.this_run_adjusted_id} timestamp: {self.this_timestamp}",
            f"Run {self.other_run_adjusted_id} timestamp: {self.other_timestamp}",
            ""
        ]
        
        # Parameter differences
        if self.parameter_differences:
            lines.append("Parameter Differences:")
            for param, values in self.parameter_differences.items():
                lines.append(f"  {param}:")
                lines.append(f"    Run {self.this_run_adjusted_id}: {values['this_run']}")
                lines.append(f"    Run {self.other_run_adjusted_id}: {values['other_run']}")
            lines.append("")
        else:
            lines.append("No parameter differences found")
            lines.append("")
        
        # Field differences with tabular format
        if self.field_differences:
            lines.append("Field Differences:")
            
            # Collect all fields for a more organized display
            all_diff_fields = []
            for location, diffs in self.field_differences.items():
                # Process fields only in this run
                for field_info in diffs.get('only_this_run', []):
                    if isinstance(field_info, dict):
                        field_info['category'] = f"Only in Run {self.this_run_adjusted_id}"
                        field_info['display_location'] = location
                        all_diff_fields.append(field_info)
                
                # Process fields only in other run
                for field_info in diffs.get('only_other_run', []):
                    if isinstance(field_info, dict):
                        field_info['category'] = f"Only in Run {self.other_run_adjusted_id}"
                        field_info['display_location'] = location
                        all_diff_fields.append(field_info)
                
                # Process fields in both runs
                for field_info in diffs.get('both_runs', []):
                    if isinstance(field_info, dict):
                        field_info['category'] = "In both runs"
                        field_info['display_location'] = location
                        all_diff_fields.append(field_info)
            
            # If we have fields to display, create a nice table
            if all_diff_fields:
                # Prepare tabular headers
                lines.append("")
                lines.append("  ┌───────────────────────────────┬──────────┬────────────────────────┬───────────────┬───────────────────┐")
                lines.append("  │ Field Name                    │ Location │ Description            │ Status        │ Current Owner     │")
                lines.append("  ├───────────────────────────────┼──────────┼────────────────────────┼───────────────┼───────────────────┤")
                
                # Sort fields by location and name
                all_diff_fields.sort(key=lambda x: (x.get('display_location', ''), x.get('field', '')))
                
                # Add each field as a row
                for info in all_diff_fields:
                    name = info.get('field', '')[:31].ljust(31)  # Wider field name column
                    location = info.get('display_location', '')[:8].ljust(8)
                    desc = (info.get('description') or "")[:24].ljust(24)  # Narrower description
                    category = info.get('category', '')[:13].ljust(13)  # Slightly narrower status
                    
                    # Determine current owner display 
                    owner_id = info.get('current_owner')
                    owned_by = info.get('owned_by')
                    
                    if owned_by == 'this_run':
                        owner_display = f"Run {self.this_run_adjusted_id} (current)"
                    elif owned_by == 'other_run':
                        owner_display = f"Run {self.other_run_adjusted_id} (current)"
                    elif owner_id is not None:
                        owner_display = f"Run {owner_id} (different)"
                    else:
                        owner_display = "Unknown"
                        
                    owner_display = owner_display[:17].ljust(17)
                    
                    lines.append(f"  │ {name} │ {location} │ {desc} │ {category} │ {owner_display} │")
                
                lines.append("  └───────────────────────────────┴──────────┴────────────────────────┴───────────────┴───────────────────┘")
            lines.append("")
        else:
            lines.append("No field differences found")
            lines.append("")
        
        # We no longer need a separate Overwritten Fields section since this info is now in the Field Differences table
        
        return "\n".join(lines)
    
    def _repr_html_(self) -> str:
        """
        HTML representation for Jupyter notebooks.
        
        Returns
        -------
        str
            HTML-formatted comparison
        """
        html = [
            "<div style='max-width:800px'>",
            f"<h3>Comparison of Run {self.this_run_adjusted_id} and Run {self.other_run_adjusted_id}</h3>",
            "<table style='width:100%; border-collapse:collapse; margin-bottom:10px'>",
            "<tr style='background-color:#f0f0f0'>",
            "<th style='text-align:left; padding:5px'>Run</th>",
            "<th style='text-align:left; padding:5px'>Timestamp</th>",
            "</tr>"
        ]
        
        # Add run info
        html.append(f"<tr><td style='padding:5px; border:1px solid #ddd'>Run {self.this_run_adjusted_id}</td>")
        html.append(f"<td style='padding:5px; border:1px solid #ddd'>{self.this_timestamp}</td></tr>")
        
        html.append(f"<tr><td style='padding:5px; border:1px solid #ddd'>Run {self.other_run_adjusted_id}</td>")
        html.append(f"<td style='padding:5px; border:1px solid #ddd'>{self.other_timestamp}</td></tr>")
        
        html.append("</table>")
        
        # Parameter differences
        if self.parameter_differences:
            html.append("<h4>Parameter Differences</h4>")
            html.append("<table style='width:100%; border-collapse:collapse'>")
            html.append("<tr style='background-color:#f0f0f0'>")
            html.append(f"<th style='text-align:left; padding:5px; width:30%'>Parameter</th>")
            html.append(f"<th style='text-align:left; padding:5px; width:35%'>Run {self.this_run_adjusted_id}</th>")
            html.append(f"<th style='text-align:left; padding:5px; width:35%'>Run {self.other_run_adjusted_id}</th>")
            html.append("</tr>")
            
            for param, values in self.parameter_differences.items():
                html.append(f"<tr><td style='padding:5px; border:1px solid #ddd'>{param}</td>")
                html.append(f"<td style='padding:5px; border:1px solid #ddd'>{values['this_run']}</td>")
                html.append(f"<td style='padding:5px; border:1px solid #ddd'>{values['other_run']}</td></tr>")
                
            html.append("</table>")
        else:
            html.append("<p><em>No parameter differences found</em></p>")
        
        # Field differences with tabular format
        if self.field_differences:
            html.append("<h4>Field Differences</h4>")
            
            # Collect all fields for a more organized display
            all_diff_fields = []
            for location, diffs in self.field_differences.items():
                # Process fields only in this run
                for field_info in diffs.get('only_this_run', []):
                    if isinstance(field_info, dict):
                        field_info['category'] = f"Only in Run {self.this_run_adjusted_id}"
                        field_info['display_location'] = location
                        all_diff_fields.append(field_info)
                
                # Process fields only in other run
                for field_info in diffs.get('only_other_run', []):
                    if isinstance(field_info, dict):
                        field_info['category'] = f"Only in Run {self.other_run_adjusted_id}"
                        field_info['display_location'] = location
                        all_diff_fields.append(field_info)
                
                # Process fields in both runs
                for field_info in diffs.get('both_runs', []):
                    if isinstance(field_info, dict):
                        field_info['category'] = "In both runs"
                        field_info['display_location'] = location
                        all_diff_fields.append(field_info)
            
            # Create a table for all field differences
            if all_diff_fields:
                # Sort fields by location and name for better organization
                all_diff_fields.sort(key=lambda x: (x.get('display_location', ''), x.get('field', '')))
                
                # Create table
                html.append("<table style='width:100%; border-collapse:collapse; margin-top:10px'>")
                html.append("<tr style='background-color:#f0f0f0'>")
                html.append("<th style='text-align:left; padding:5px; width:35%'>Field Name</th>") # Made wider
                html.append("<th style='text-align:left; padding:5px; width:8%'>Location</th>")    # Made slightly narrower
                html.append("<th style='text-align:left; padding:5px; width:25%'>Description</th>") # Made slightly narrower
                html.append("<th style='text-align:left; padding:5px; width:12%'>Status</th>")
                html.append("<th style='text-align:left; padding:5px; width:20%'>Current Owner</th>")
                html.append("</tr>")
                
                # Add rows for each field
                for info in all_diff_fields:
                    field = info.get('field', '')
                    location = info.get('display_location', '')
                    desc = info.get('description', '')
                    category = info.get('category', '')
                    
                    # Determine current owner display
                    owner_id = info.get('current_owner')
                    owned_by = info.get('owned_by')
                    
                    if owned_by == 'this_run':
                        owner_display = f"Run {self.this_run_adjusted_id} <span style='color:green'>(current)</span>"
                    elif owned_by == 'other_run':
                        owner_display = f"Run {self.other_run_adjusted_id} <span style='color:green'>(current)</span>"
                    elif owner_id is not None:
                        owner_display = f"Run {owner_id} <span style='color:orange'>(different)</span>"
                    else:
                        owner_display = "<span style='color:gray'>Unknown</span>"
                    
                    # Apply row styling based on category
                    if category == f"Only in Run {self.this_run_adjusted_id}":
                        category_style = "color:#2a6099"  # Blue for this run
                    elif category == f"Only in Run {self.other_run_adjusted_id}":
                        category_style = "color:#992a5b"  # Pink for other run
                    else:
                        category_style = "color:#666"  # Gray for both
                    
                    html.append("<tr>")
                    html.append(f"<td style='padding:5px; border:1px solid #ddd'>{field}</td>")
                    html.append(f"<td style='padding:5px; border:1px solid #ddd'>{location}</td>")
                    html.append(f"<td style='padding:5px; border:1px solid #ddd'>{desc}</td>")
                    html.append(f"<td style='padding:5px; border:1px solid #ddd; {category_style}'>{category}</td>")
                    html.append(f"<td style='padding:5px; border:1px solid #ddd'>{owner_display}</td>")
                    html.append("</tr>")
                
                html.append("</table>")
            else:
                html.append("<p><em>No field differences details available</em></p>")
        else:
            html.append("<p><em>No field differences found</em></p>")
        
        # We no longer need a separate Overwritten Fields section since this info is now in the Field Differences table
        
        html.append("</div>")
        return "\n".join(html)
    
    def as_dict(self) -> Dict[str, Any]:
        """
        Return the comparison data as a dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with comparison results
        """
        return {
            'this_run_id': self.this_run_id,
            'other_run_id': self.other_run_id,
            'this_run_adjusted_id': self.this_run_adjusted_id,
            'other_run_adjusted_id': self.other_run_adjusted_id,
            'this_timestamp': self.this_timestamp,
            'other_timestamp': self.other_timestamp,
            'parameter_differences': self.parameter_differences,
            'field_differences': self.field_differences,
            'overwritten_fields': self.overwritten_fields
        }


class RunInfo:
    """
    Class to retrieve and format run information for differential analysis.
    
    This class provides a convenient interface to examine run information
    from differential expression (de) or differential abundance (da) analyses.
    It offers various display formats for both interactive Python sessions and
    Jupyter notebooks.
    
    Attributes
    ----------
    adata : AnnData
        The AnnData object containing the run information
    run_id : int
        The run ID for this specific analysis
    analysis_type : str
        The type of analysis ('de' or 'da')
    run_info : dict
        The full run information dictionary
    storage_key : str
        The key in adata.uns where this run is stored
    field_names : dict
        The field names used by this run
    adata_fields : dict
        Dictionary tracking which fields in adata were written by this run
    params : dict
        The parameters used for this analysis
    environment : dict
        Information about the environment where the analysis was run
    overwritten_fields : list
        List of fields that were overwritten by newer runs
    """
    
    def __init__(self, 
                 adata, 
                 run_id: Optional[int] = None, 
                 analysis_type: Optional[str] = None):
        """
        Initialize a RunInfo object.
        
        Parameters
        ----------
        adata : AnnData
            AnnData object containing run history
        run_id : int, optional
            Run ID to retrieve. Negative indices count from the end.
            If None, uses the most recent run (-1).
        analysis_type : str, optional
            Type of analysis: 'de' for differential expression or 
            'da' for differential abundance. If None, attempts to detect.
        """
        self.adata = adata
        if run_id is None:
            run_id = -1  # Default to most recent run
        self.run_id = run_id
        
        # Detect analysis type if not provided
        if analysis_type is None:
            # Try to detect from uns keys
            if 'kompot_de' in adata.uns and 'run_history' in adata.uns['kompot_de']:
                analysis_type = 'de'
            elif 'kompot_da' in adata.uns and 'run_history' in adata.uns['kompot_da']:
                analysis_type = 'da'
            else:
                raise ValueError("Could not detect analysis type. Please specify 'de' or 'da'.")
                
        if analysis_type not in ['de', 'da']:
            raise ValueError(f"Invalid analysis_type: {analysis_type}. Must be 'de' or 'da'.")
            
        self.analysis_type = analysis_type
        self.storage_key = f"kompot_{analysis_type}"
        
        # Check if run history exists
        if (self.storage_key not in adata.uns or 
            'run_history' not in adata.uns[self.storage_key] or
            len(adata.uns[self.storage_key]['run_history']) == 0):
            raise ValueError(f"No run history found for {analysis_type} analysis.")
        
        # Get run info
        self.run_info = get_run_from_history(adata, run_id=run_id, analysis_type=analysis_type)
        
        if self.run_info is None:
            raise ValueError(f"Run ID {run_id} not found in {analysis_type} run history.")
            
        # Set adjusted run_id
        self.adjusted_run_id = self.run_info.get('adjusted_run_id', None)
        
        # Extract key information
        self.field_names = self.run_info.get('field_names', {})
        self.params = self.run_info.get('params', {}).copy()  # Make a copy to avoid modifying the original
        self.environment = self.run_info.get('environment', {})
        self.timestamp = self.run_info.get('timestamp', '')
        
        # Ensure result_key is included in params if missing
        if 'result_key' not in self.params and 'result_key' in self.run_info:
            self.params['result_key'] = self.run_info['result_key']
        
        # Get all fields modified by this run
        self.adata_fields = self._get_fields_for_run()
        
        # Check for fields that have been overwritten by newer runs
        self.overwritten_fields = self._check_overwritten_fields()
        
    def _get_fields_for_run(self) -> Dict[str, List[str]]:
        """
        Get all fields in the AnnData object that were written by this run.
        
        Returns
        -------
        Dict[str, List[str]]
            Dictionary with AnnData locations as keys and lists of field names as values
        """
        result = {}
        
        # Get fields from field_mapping in the run_info - this is the only source of truth
        field_mapping = self.get_raw_data().get('field_mapping', {})
        
        if not field_mapping:
            logger.warning(f"No field_mapping found for run {self.adjusted_run_id}.")
            return {}
            
        # Initialize result
        locations = set(mapping.get('location') for mapping in field_mapping.values() if mapping.get('location'))
        for location in locations:
            result[location] = []
            
        # Add fields to their locations
        for field, mapping in field_mapping.items():
            location = mapping.get('location')
            if location:
                if location not in result:
                    result[location] = []
                result[location].append(field)
        
        # Sort field lists for consistent display
        for location in result:
            result[location].sort()
            
        return result
    
    def _check_overwritten_fields(self) -> List[Dict[str, Any]]:
        """
        Check if any fields from this run have been overwritten by newer runs.
        
        Returns
        -------
        List[Dict[str, Any]]
            List of dictionaries with overwritten field information, each containing:
            - field: The field name
            - location: The location in AnnData (obs, var, etc.)
            - current_run_id: The run ID that now owns this field
            - expected_run_id: The run ID that should own this field (this run)
        """
        if (self.storage_key not in self.adata.uns or 
            'anndata_fields' not in self.adata.uns[self.storage_key]):
            return []
            
        tracking = self.adata.uns[self.storage_key]['anndata_fields']
        overwritten = []
        
        # Get fields from field_mapping as the source of truth
        field_mapping = self.get_raw_data().get('field_mapping', {})
        
        # If no field_mapping, we don't know what fields to check
        if not field_mapping:
            logger.warning(f"No field_mapping found for run {self.adjusted_run_id} to check for overwritten fields.")
            return []
        
        # Check each field from field_mapping against the tracking info
        for field, mapping in field_mapping.items():
            location = mapping.get('location')
            if not location or location not in tracking or field not in tracking[location]:
                # Skip fields not in the tracking dictionary
                continue
                
            # Check if the field is attributed to a different run
            current_run_id = tracking[location][field]
            if current_run_id != self.adjusted_run_id:
                overwritten.append({
                    'field': field,
                    'location': location,
                    'current_run_id': current_run_id,
                    'expected_run_id': self.adjusted_run_id
                })
                    
        return overwritten
    
    def compare_with(self, other_run_id: int) -> 'RunComparison':
        """
        Compare this run with another run.
        
        Parameters
        ----------
        other_run_id : int
            Run ID to compare with
            
        Returns
        -------
        RunComparison
            Object containing comparison results with nice display methods
        """
        return RunComparison(self.adata, self.run_id, other_run_id, self.analysis_type)
    
    def get_data(self) -> Dict[str, Any]:
        """
        Get all data related to this run.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with all run data
        """
        # Get field data based on adata_fields
        field_data = {}
        
        for location, fields in self.adata_fields.items():
            field_data[location] = {}
            
            if location == 'obs':
                for field in fields:
                    if field in self.adata.obs:
                        field_data[location][field] = self.adata.obs[field]
            elif location == 'var':
                for field in fields:
                    if field in self.adata.var:
                        field_data[location][field] = self.adata.var[field]
            elif location == 'uns':
                for field in fields:
                    if field in self.adata.uns:
                        field_data[location][field] = self.adata.uns[field]
            elif location == 'layers':
                for field in fields:
                    if field in self.adata.layers:
                        field_data[location][field] = self.adata.layers[field]
        
        return {
            'run_id': self.run_id,
            'adjusted_run_id': self.adjusted_run_id,
            'analysis_type': self.analysis_type,
            'field_names': self.field_names,
            'params': self.params,
            'environment': self.environment,
            'timestamp': self.timestamp,
            'overwritten_fields': self.overwritten_fields,
            'field_data': field_data
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of this run with key information.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with run summary
        """
        # Get basic information without field data
        summary = {
            'run_id': self.run_id,
            'adjusted_run_id': self.adjusted_run_id,
            'analysis_type': self.analysis_type,
            'timestamp': self.timestamp,
            'conditions': f"{self.params.get('condition1', 'unknown')} to {self.params.get('condition2', 'unknown')}",
            'obsm_key': self.params.get('obsm_key', 'unknown'),
            'layer': self.params.get('layer', None),
            'uses_sample_variance': self.params.get('use_sample_variance', False),
            'field_count': sum(len(fields) for fields in self.adata_fields.values()),
            'overwritten_field_count': len(self.overwritten_fields),
            'overwritten_fields': self.overwritten_fields
        }
        
        # Add group information if available
        raw_data = self.get_raw_data()
        has_groups = raw_data.get('has_groups', False)
        if has_groups:
            groups_summary = raw_data.get('groups_summary', {})
            summary['has_groups'] = True
            summary['groups_count'] = groups_summary.get('count', 0)
            summary['groups_names'] = groups_summary.get('names', [])
            # Provide a short preview of group names
            group_names_preview = ", ".join(summary['groups_names'][:3])
            if len(summary['groups_names']) > 3:
                group_names_preview += f" and {len(summary['groups_names']) - 3} more"
            summary['groups_preview'] = group_names_preview
        else:
            summary['has_groups'] = False
        
        # Don't add anndata_locations directly to summary
        # We'll use it to enhance field listings instead
        return summary
    
    def _repr_html_(self) -> str:
        """
        HTML representation for Jupyter notebooks.
        
        Returns
        -------
        str
            HTML representation
        """
        summary = self.get_summary()
        
        # Build HTML
        html = [
            "<div style='max-width:800px'>",
            f"<h3>Run {summary['adjusted_run_id']} ({summary['analysis_type'].upper()} Analysis)</h3>",
            "<table style='width:100%; border-collapse:collapse; margin-bottom:10px'>",
            "<tr style='background-color:#f0f0f0'><th style='text-align:left; padding:5px; width:30%'>Parameter</th><th style='text-align:left; padding:5px; width:70%'>Value</th></tr>"
        ]
        
        # Add summary rows
        for k, v in summary.items():
            if k not in ['run_id', 'analysis_type', 'overwritten_fields']:
                html.append(f"<tr><td style='padding:5px; border:1px solid #ddd'>{k}</td><td style='padding:5px; border:1px solid #ddd'>{v}</td></tr>")
        
        html.append("</table>")
        
        # Add groups section if available
        raw_data = self.get_raw_data()
        if raw_data.get('has_groups', False):
            groups_summary = raw_data.get('groups_summary', {})
            if groups_summary:
                html.append("<h4>Group Information</h4>")
                html.append("<p>This analysis was performed with group-based subsetting.</p>")
                
                # Basic group info table
                html.append("<table style='width:100%; border-collapse:collapse; margin-bottom:10px'>")
                html.append("<tr style='background-color:#f0f0f0'>")
                html.append("<th style='text-align:left; padding:5px; width:25%'>Parameter</th>")
                html.append("<th style='text-align:left; padding:5px; width:75%'>Value</th></tr>")
                
                html.append(f"<tr><td style='padding:5px; border:1px solid #ddd'>Number of Groups</td>")
                html.append(f"<td style='padding:5px; border:1px solid #ddd'>{groups_summary.get('count', 0)}</td></tr>")
                
                html.append(f"<tr><td style='padding:5px; border:1px solid #ddd'>Group Type</td>")
                html.append(f"<td style='padding:5px; border:1px solid #ddd'>{groups_summary.get('description', 'Unknown')}</td></tr>")
                
                html.append("</table>")
                
                # Group details table
                cells_per_group = groups_summary.get('cells_per_group', {})
                if cells_per_group:
                    html.append("<h5>Group Details</h5>")
                    html.append("<table style='width:100%; border-collapse:collapse'>")
                    html.append("<tr style='background-color:#f0f0f0'>")
                    html.append("<th style='text-align:left; padding:5px; width:30%'>Group Name</th>")
                    html.append("<th style='text-align:left; padding:5px; width:15%'>Cell Count</th>")
                    html.append("<th style='text-align:left; padding:5px; width:15%'>Percentage</th>")
                    html.append("<th style='text-align:left; padding:5px; width:40%'>Condition Distribution</th></tr>")
                    
                    for group_name, stats in cells_per_group.items():
                        cell_count = stats.get('count', 0)
                        percentage = stats.get('percentage', 0)
                        
                        # Create condition distribution text
                        condition_text = ""
                        if 'conditions' in stats:
                            conditions = stats['conditions']
                            for cond_name, cond_stats in conditions.items():
                                cond_count = cond_stats.get('count', 0)
                                cond_pct = cond_stats.get('percentage', 0)
                                condition_text += f"{cond_name}: {cond_count} ({cond_pct:.1f}%)<br>"
                        
                        html.append(f"<tr><td style='padding:5px; border:1px solid #ddd'>{group_name}</td>")
                        html.append(f"<td style='padding:5px; border:1px solid #ddd'>{cell_count}</td>")
                        html.append(f"<td style='padding:5px; border:1px solid #ddd'>{percentage:.1f}%</td>")
                        html.append(f"<td style='padding:5px; border:1px solid #ddd'>{condition_text}</td></tr>")
                    
                    html.append("</table>")
        
        # Add field section as a structured table
        if self.adata_fields:
            html.append("<h4>Fields Created by This Run</h4>")
            html.append("<table style='width:100%; border-collapse:collapse'>")
            html.append("<tr style='background-color:#f0f0f0'>")
            html.append("<th style='text-align:left; padding:5px; width:40%'>Field Name</th>") # Even wider field name
            html.append("<th style='text-align:left; padding:5px; width:10%'>Location</th>")
            html.append("<th style='text-align:left; padding:5px; width:35%'>Description</th>")
            html.append("<th style='text-align:left; padding:5px; width:15%'>Status</th>")
            html.append("</tr>")
            
            # Get all field info for formatting
            all_fields = []
            raw_data = self.get_raw_data()
            field_mapping = raw_data.get('field_mapping', {})
            
            # Collect all fields with their metadata
            for location, fields in self.adata_fields.items():
                for field in fields:
                    # Get field metadata
                    field_info = {
                        'name': field,
                        'location': location,
                        'type': None,
                        'description': None,
                        'overwritten': None
                    }
                    
                    # Check if field was overwritten
                    overwritten_info = next((info for info in self.overwritten_fields if 
                                        info['location'] == location and info['field'] == field), None)
                    if overwritten_info:
                        field_info['overwritten'] = overwritten_info['current_run_id']
                    
                    # Get additional info from field_mapping
                    if field in field_mapping:
                        mapping = field_mapping[field]
                        field_info['location'] = mapping.get('location', location)
                        field_info['type'] = mapping.get('type')
                        field_info['description'] = mapping.get('description')
                    
                    all_fields.append(field_info)
            
            # Sort fields by location and name for better organization
            all_fields.sort(key=lambda x: (x['location'], x['name']))
            
            # Add a row for each field
            for field_info in all_fields:
                name = field_info['name']
                location = field_info['location']
                field_type = field_info['type'] or ""
                description = field_info['description'] or ""
                overwritten = field_info['overwritten']
                
                # Style for overwritten fields
                row_style = " style='color:red;'" if overwritten else ""
                
                html.append(f"<tr{row_style}>")
                html.append(f"<td style='padding:5px; border:1px solid #ddd'>{name}</td>")
                html.append(f"<td style='padding:5px; border:1px solid #ddd'>{location}</td>")
                html.append(f"<td style='padding:5px; border:1px solid #ddd'>{description}</td>")
                if overwritten:
                    html.append(f"<td style='padding:5px; border:1px solid #ddd'>Overwritten by run {overwritten}</td>")
                else:
                    html.append(f"<td style='padding:5px; border:1px solid #ddd'>Active</td>")
                html.append("</tr>")
                
            html.append("</table>")
        
        # Add parameters section
        if self.params:
            html.append("<h4>Analysis Parameters</h4>")
            html.append("<table style='width:100%; border-collapse:collapse'>")
            html.append("<tr style='background-color:#f0f0f0'><th style='text-align:left; padding:5px; width:25%'>Parameter</th><th style='text-align:left; padding:5px; width:75%'>Value</th></tr>")
            
            for param, value in self.params.items():
                html.append(f"<tr><td style='padding:5px; border:1px solid #ddd'>{param}</td><td style='padding:5px; border:1px solid #ddd'>{value}</td></tr>")
                
            html.append("</table>")
        
        # Add environment information
        if self.environment:
            html.append("<h4>Execution Environment</h4>")
            html.append("<table style='width:100%; border-collapse:collapse'>")
            html.append("<tr style='background-color:#f0f0f0'><th style='text-align:left; padding:5px; width:25%'>Field</th><th style='text-align:left; padding:5px; width:75%'>Value</th></tr>")
            
            for key, value in self.environment.items():
                html.append(f"<tr><td style='padding:5px; border:1px solid #ddd'>{key}</td><td style='padding:5px; border:1px solid #ddd'>{value}</td></tr>")
                
            html.append("</table>")
        
        html.append("</div>")
        return "\n".join(html)
    
    def __str__(self) -> str:
        """
        String representation of the RunInfo object.
        
        Returns
        -------
        str
            String representation
        """
        summary = self.get_summary()
        
        # Build string representation
        lines = [
            f"RunInfo: {summary['analysis_type'].upper()} Analysis Run {summary['adjusted_run_id']}",
            f"Timestamp: {summary['timestamp']}",
            f"Conditions: {summary['conditions']}",
            f"OBSM Key: {summary['obsm_key']}",
            f"Layer: {summary['layer']}",
            f"Uses Sample Variance: {summary['uses_sample_variance']}",
            f"Total Fields: {summary['field_count']} (Overwritten: {summary['overwritten_field_count']})",
        ]
        
        # Add group information if available
        if summary.get('has_groups', False):
            lines.append(f"Groups: {summary.get('groups_count', 0)} ({summary.get('groups_preview', '')})")
            
            # Add more detailed group info if available
            raw_data = self.get_raw_data()
            groups_summary = raw_data.get('groups_summary', {})
            if groups_summary and 'cells_per_group' in groups_summary:
                lines.append("")
                lines.append("Group Details:")
                
                # Create a table for group details
                lines.append("  ┌─────────────────────────┬──────────┬──────────┬──────────────────────────────┐")
                lines.append("  │ Group Name              │ Cells    │ % Total  │ Condition Distribution       │")
                lines.append("  ├─────────────────────────┼──────────┼──────────┼──────────────────────────────┤")
                
                for group_name, stats in groups_summary['cells_per_group'].items():
                    name = str(group_name)[:23].ljust(23)
                    cell_count = str(stats.get('count', 0)).ljust(8)
                    percentage = f"{stats.get('percentage', 0):.1f}%".ljust(8)
                    
                    # Format condition distribution
                    if 'conditions' in stats:
                        condition_text = []
                        for cond_name, cond_stats in stats['conditions'].items():
                            cond_count = cond_stats.get('count', 0)
                            cond_pct = cond_stats.get('percentage', 0)
                            condition_text.append(f"{cond_name}: {cond_count} ({cond_pct:.1f}%)")
                        
                        # Join with commas and truncate if too long
                        cond_str = ", ".join(condition_text)
                        if len(cond_str) > 28:
                            cond_str = cond_str[:25] + "..."
                        
                        conditions = cond_str.ljust(28)
                    else:
                        conditions = "N/A".ljust(28)
                    
                    lines.append(f"  │ {name} │ {cell_count} │ {percentage} │ {conditions} │")
                
                lines.append("  └─────────────────────────┴──────────┴──────────┴──────────────────────────────┘")
        
        lines.append("")
        
        # Add fields in a tabular format
        if self.adata_fields:
            lines.append("Fields Created by This Run:")
            
            # Get all field info for tabular display
            all_fields = []
            raw_data = self.get_raw_data()
            field_mapping = raw_data.get('field_mapping', {})
            
            # Collect all fields with their metadata
            for location, fields in self.adata_fields.items():
                for field in fields:
                    # Get field metadata
                    field_info = {
                        'name': field,
                        'location': location,
                        'type': None,
                        'description': None,
                        'overwritten': None
                    }
                    
                    # Check if field was overwritten
                    overwritten_info = next((info for info in self.overwritten_fields if 
                                        info['location'] == location and info['field'] == field), None)
                    if overwritten_info:
                        field_info['overwritten'] = overwritten_info['current_run_id']
                    
                    # Get additional info from field_mapping
                    if field in field_mapping:
                        mapping = field_mapping[field]
                        field_info['location'] = mapping.get('location', location)
                        field_info['type'] = mapping.get('type')
                        field_info['description'] = mapping.get('description')
                    
                    all_fields.append(field_info)
            
            # Sort fields by location and name for better organization
            all_fields.sort(key=lambda x: (x['location'], x['name']))
            
            # Prepare tabular headers
            lines.append("  ┌─────────────────────────┬──────────┬──────────────────────────────┬─────────────────┐")
            lines.append("  │ Field Name              │ Location │ Description                  │ Status          │")
            lines.append("  ├─────────────────────────┼──────────┼──────────────────────────────┼─────────────────┤")
            
            # Add each field as a row
            for info in all_fields:
                name = info['name'][:23].ljust(23)
                location = info['location'][:8].ljust(8)
                desc = (info['description'] or "")[:28].ljust(28)
                
                if info['overwritten']:
                    status = f"Overwritten ({info['overwritten']})"[:15].ljust(15)
                else:
                    status = "Active".ljust(15)
                
                lines.append(f"  │ {name} │ {location} │ {desc} │ {status} │")
            
            lines.append("  └─────────────────────────┴──────────┴──────────────────────────────┴─────────────────┘")
            lines.append("")
            
        # Add parameter summary
        if self.params:
            lines.append("Parameters:")
            for param, value in self.params.items():
                lines.append(f"  {param}: {value}")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        """
        Return the string representation of the RunInfo object.
        
        Returns
        -------
        str
            String representation
        """
        return self.__str__()
    
    def as_dict(self) -> Dict[str, Any]:
        """
        Return the RunInfo object as a dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary representation
        """
        return self.get_data()
    
    def to_json(self, indent: int = 2) -> str:
        """
        Convert the RunInfo summary to a JSON string.
        
        Parameters
        ----------
        indent : int, optional
            Number of spaces for indentation, by default 2
            
        Returns
        -------
        str
            JSON string representation
        """
        summary = self.get_summary()
        
        # Convert any non-serializable objects to strings
        for k, v in summary.items():
            if not isinstance(v, (str, int, float, bool, list, dict, type(None))):
                summary[k] = str(v)
                
        return json.dumps(summary, indent=indent)
    
        
    def get_raw_data(self) -> Dict[str, Any]:
        """
        Get the raw run information dictionary from adata.uns.
        
        This provides direct access to the complete run information as stored in the AnnData object.
        
        Returns
        -------
        Dict[str, Any]
            The raw dictionary containing all run information
        """
        if self.run_info is None:
            return {}
        
        return self.run_info
    
    @staticmethod
    def get_runs(adata, analysis_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get a list of all available runs in the AnnData object.
        
        Parameters
        ----------
        adata : AnnData
            AnnData object containing run history
        analysis_type : str, optional
            Type of analysis: 'de', 'da', or None for both
            
        Returns
        -------
        List[Dict[str, Any]]
            List of run summaries
        """
        runs = []
        
        # Check DE runs
        if analysis_type in [None, 'de'] and 'kompot_de' in adata.uns and 'run_history' in adata.uns['kompot_de']:
            de_runs = []
            for i, run in enumerate(adata.uns['kompot_de']['run_history']):
                try:
                    run_info = RunInfo(adata, run_id=i, analysis_type='de')
                    de_runs.append(run_info.get_summary())
                except Exception as e:
                    logger.warning(f"Error loading DE run {i}: {e}")
            runs.extend(de_runs)
            
        # Check DA runs
        if analysis_type in [None, 'da'] and 'kompot_da' in adata.uns and 'run_history' in adata.uns['kompot_da']:
            da_runs = []
            for i, run in enumerate(adata.uns['kompot_da']['run_history']):
                try:
                    run_info = RunInfo(adata, run_id=i, analysis_type='da')
                    da_runs.append(run_info.get_summary())
                except Exception as e:
                    logger.warning(f"Error loading DA run {i}: {e}")
            runs.extend(da_runs)
            
        return runs
    
    @staticmethod
    def list_runs(adata, analysis_type: Optional[str] = None) -> str:
        """
        List all available runs in the AnnData object and print the result.
        
        Parameters
        ----------
        adata : AnnData
            AnnData object containing run history
        analysis_type : str, optional
            Type of analysis: 'de', 'da', or None for both
            
        Returns
        -------
        str
            Formatted list of runs
        """
        runs = RunInfo.get_runs(adata, analysis_type)
        
        if not runs:
            result = "No runs found."
        else:
            lines = ["Available Runs:"]
            for i, run in enumerate(runs):
                lines.append(f"{i}. {run['analysis_type'].upper()} Run {run['adjusted_run_id']}: {run['conditions']} ({run['timestamp']})")
            result = "\n".join(lines)
        
        # Print the result by default
        print(result)
        
        return result


def validate_field_run_id(
    adata: AnnData,
    field_name: str,
    location: str,
    requested_run_id: int,
    storage_key: str
) -> Tuple[bool, Optional[int], Optional[str]]:
    """
    Validate if a field was last written by the requested run_id.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object containing field tracking information
    field_name : str
        Name of the field to validate
    location : str
        Location of the field ('obs', 'var', 'uns', 'layers')
    requested_run_id : int
        The run ID that is being requested (must be positive/adjusted)
    storage_key : str
        The storage key where tracking information is stored (e.g., 'kompot_de', 'kompot_da')
        
    Returns
    -------
    Tuple[bool, Optional[int], Optional[str]]
        - Boolean indicating if the field was last written by the requested run
        - The actual run_id that last wrote to this field, or None if not found
        - Warning message if validation fails, or None if validation passes
    """
    # Check if we have tracking information
    if (storage_key in adata.uns and 
        "anndata_fields" in adata.uns[storage_key] and 
        location in adata.uns[storage_key]["anndata_fields"]):
        
        tracking_info = adata.uns[storage_key]["anndata_fields"][location]
        
        # Check if this specific field is being tracked
        if field_name in tracking_info:
            actual_run_id = tracking_info[field_name]
            
            if actual_run_id != requested_run_id:
                warning_msg = (f"Field '{field_name}' in {location} was last written by run_id={actual_run_id}, "
                              f"but you requested run_id={requested_run_id}. The data may be inconsistent.")
                return False, actual_run_id, warning_msg
            
            return True, actual_run_id, None
        
    # If no tracking information, we can't validate
    return True, None, None


def get_run_from_history(
    adata: AnnData, 
    run_id: Optional[int] = None, 
    history_key: str = 'kompot_run_history',
    analysis_type: Optional[str] = None,
    validate_field: Optional[str] = None,
    field_location: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Get run information from run history based on run_id.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object containing run history
    run_id : int, optional
        Run ID to retrieve. Negative indices count from the end.
        If None, returns None.
    history_key : str, optional
        Key in adata.uns where the run history is stored.
        Default is 'kompot_run_history' for the global history.
        For analysis-specific history, use either:
        - 'kompot_da.run_history' for differential abundance runs
        - 'kompot_de.run_history' for differential expression runs
        - Or set analysis_type instead for automatic lookup
        This is only used if analysis_type is None.
    analysis_type : str, optional
        Type of analysis to look up: "da", "de", or None.
        If provided, only looks in the specific analysis type's history
        and ignores history_key.
    validate_field : str, optional
        If provided, validate that this field was last written by the requested run_id.
        This helps ensure data consistency when retrieving data for a specific run.
    field_location : str, optional
        Location of the field to validate ('obs', 'var', 'uns', 'layers').
        Required if validate_field is provided.
        
    Returns
    -------
    dict or None
        The run information dict if found, or None if not found or run_id is None
        
    Notes
    -----
    The run history is always stored in fixed locations:
    - adata.uns['kompot_da'] for differential abundance runs
    - adata.uns['kompot_de'] for differential expression runs
    - adata.uns['kompot_run_history'] for combined runs
    """
    if run_id is None:
        return None
    
    # Determine storage_key
    storage_key = None
    
    # Use specific analysis history if provided
    if analysis_type is not None:
        if analysis_type == "da":
            history_key = "kompot_da.run_history"
            storage_key = "kompot_da"
        elif analysis_type == "de":
            history_key = "kompot_de.run_history"
            storage_key = "kompot_de"
        elif analysis_type == "combined":
            history_key = "kompot_run_history"
            storage_key = "kompot_run_history"
        else:
            logger.warning(f"Unknown analysis_type: {analysis_type}. Using provided history_key: {history_key}")
    
    # Handle case where history_key is specified as 'storage_key.run_history'
    if '.' in history_key:
        parts = history_key.split('.')
        storage_key = parts[0]
        subkey = parts[1]
        if storage_key in adata.uns and subkey in adata.uns[storage_key]:
            history = adata.uns[storage_key][subkey]
        else:
            # Only show a warning if this is not the run_history subkey - first-time runs shouldn't warn
            if subkey != 'run_history':
                logger.warning(f"Run history at {storage_key}.{subkey} not found.")
            return None
    
    # Direct access to specified history key
    elif history_key in adata.uns:
        history = adata.uns[history_key]
        # Try to infer storage_key if not already set
        if storage_key is None:
            if "kompot_da" in history_key:
                storage_key = "kompot_da"
            elif "kompot_de" in history_key:
                storage_key = "kompot_de"
            else:
                storage_key = history_key
    
    # Not found
    else:
        # Only show a warning if this is not a standard run_history key
        if not history_key.endswith('run_history'):
            logger.warning(f"No run history found at {history_key}.")
        return None
    
    # If history is empty
    if len(history) == 0:
        logger.warning(f"Run history at {history_key} is empty.")
        return None
        
    # Handle negative indices (e.g., -1 for latest run)
    if run_id < 0 and len(history) >= abs(run_id):
        adjusted_run_id = len(history) + run_id
    else:
        adjusted_run_id = run_id
    
    # Find the requested run
    if 0 <= adjusted_run_id < len(history):
        run_info = history[adjusted_run_id]
        run_info["adjusted_run_id"] = adjusted_run_id
        
        # Validate field if requested
        if validate_field is not None and field_location is not None and storage_key is not None:
            is_valid, actual_run_id, warning_msg = validate_field_run_id(
                adata=adata,
                field_name=validate_field,
                location=field_location,
                requested_run_id=adjusted_run_id,
                storage_key=storage_key
            )
            
            if not is_valid:
                logger.warning(warning_msg)
                
                # Add validation info to the run_info
                if "validation" not in run_info:
                    run_info["validation"] = {}
                
                run_info["validation"][validate_field] = {
                    "valid": False,
                    "field_location": field_location,
                    "requested_run_id": adjusted_run_id,
                    "actual_run_id": actual_run_id,
                    "warning": warning_msg
                }
        
        return run_info
    else:
        logger.warning(f"Run ID {run_id} not found in {history_key}.")
        return None


def apply_cell_filter(
    adata,
    cell_filter: Optional[Union[str, List[str], Dict[str, Any], List[Dict[str, Any]]]] = None,
    groups: Optional[Union[str, Dict[str, Any], List[Dict[str, Any]], pd.Series, np.ndarray, List[np.ndarray]]] = None
) -> Tuple[np.ndarray, int]:
    """
    Apply a cell filter to an AnnData object and return the filter mask.
    
    This function centralizes the filtering logic used in differential expression analysis.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object to filter
    cell_filter : str, List[str], Dict, List[Dict], optional
        Specification for cells or groups to exclude from the analysis:
        - str: a single group name from groups to exclude
        - List[str]: multiple group names from groups to exclude
        - Dict: keys are column names in adata.obs, values are categories to exclude
        - List[Dict]: multiple exclusion criteria dicts
        If None, no filtering is applied
    groups : str, Dict, List[Dict], pd.Series, np.ndarray, List[np.ndarray], optional
        Group specification used for subsetting when cell_filter is a string or list of strings.
        Only required when cell_filter is a string or list of strings, otherwise ignored.
        
    Returns
    -------
    Tuple[np.ndarray, int]
        - Boolean mask of cells to keep (True = keep, False = exclude)
        - Number of cells excluded by the filter
    
    Raises
    ------
    ValueError
        If cell_filter is invalid or uses columns not found in adata.obs
    """
    
    # Default: keep all cells
    filter_mask = np.ones(adata.n_obs, dtype=bool)
    
    # If no filter is provided, return all cells
    if cell_filter is None:
        return filter_mask, 0
    
    # Create exclude mask (cells to exclude)
    exclude_mask = np.zeros(adata.n_obs, dtype=bool)
    
    # String or list of strings only valid when groups is also provided
    if (isinstance(cell_filter, str) or 
        (isinstance(cell_filter, (list, tuple)) and all(isinstance(x, str) for x in cell_filter))):
        
        if groups is None:
            raise ValueError(
                "When cell_filter is a string or list of strings, the groups parameter must also be provided "
                "to specify which groups to exclude."
            )
        
        # Get subset masks from the groups parameter
        subset_masks, subset_names = parse_groups(adata, groups)
        
        # Case 1: String (single group to exclude from groups parameter)
        if isinstance(cell_filter, str):
            if cell_filter in subset_names:
                # Exclude the specified subset
                exclude_mask |= subset_masks[cell_filter]
                logger.info(f"Excluding group '{cell_filter}' from groups: {np.sum(exclude_mask):,} cells excluded")
            else:
                logger.warning(f"Group '{cell_filter}' not found in subset_names: {subset_names}. No cells excluded.")
        
        # Case 2: List of strings (multiple groups to exclude from groups parameter)
        else:  # Already checked it's a list of strings
            exclude_values = cell_filter
            for group_name in exclude_values:
                if group_name in subset_names:
                    exclude_mask |= subset_masks[group_name]
                else:
                    logger.warning(f"Group '{group_name}' not found in subset_names: {subset_names}. Skipping.")
            
            logger.info(f"Excluding groups {exclude_values} from groups parameter: {np.sum(exclude_mask):,} cells excluded")
    
    # Case 3: Dictionary (keys are column names, values are values to exclude)
    elif isinstance(cell_filter, dict):
        for col, values in cell_filter.items():
            if col == "__underrepresentation_data":
                continue  # Skip the special key
                
            if col not in adata.obs:
                logger.warning(f"Column '{col}' not found in adata.obs. Skipping this filter.")
                continue
            
            # Ensure values is a list
            if not isinstance(values, (list, tuple)):
                values = [values]
            
            # Exclude cells matching any of the values
            col_mask = (adata.obs[col].isin(values)).values
            exclude_mask |= col_mask
            logger.debug(f"Excluding {np.sum(col_mask):,} cells where {col} is in {values}")
        
        logger.info(f"Excluding cells based on filter dictionary: {np.sum(exclude_mask):,} cells excluded")
    
    # Case 4: List of dictionaries (multiple exclusion criteria)
    elif isinstance(cell_filter, (list, tuple)) and all(isinstance(x, dict) for x in cell_filter):
        for i, criteria in enumerate(cell_filter):
            criteria_mask = np.zeros(adata.n_obs, dtype=bool)
            
            for col, values in criteria.items():
                if col not in adata.obs:
                    raise ValueError(f"Column '{col}' not found in adata.obs")
                
                # Ensure values is a list
                if not isinstance(values, (list, tuple)):
                    values = [values]
                
                # Exclude cells matching any of the values
                col_mask = (adata.obs[col].isin(values)).values
                criteria_mask |= col_mask
            
            exclude_mask |= criteria_mask
            logger.debug(f"Exclusion criteria {i+1}: {np.sum(criteria_mask):,} cells excluded")
        
        logger.info(f"Excluding cells based on multiple criteria: {np.sum(exclude_mask):,} cells excluded")
    
    else:
        raise ValueError(
            "Invalid cell_filter parameter. It should be a string, a list of strings, "
            "a dictionary, or a list of dictionaries."
        )
    
    # Update the filter mask to exclude specified cells
    filter_mask = ~exclude_mask
    return filter_mask, np.sum(exclude_mask)