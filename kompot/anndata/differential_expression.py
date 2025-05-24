"""
Differential expression analysis for AnnData objects.
"""

import logging
import numpy as np
import pandas as pd
import datetime
from typing import Optional, Union, Dict, Any, List
from scipy import sparse
try:
    import anndata
except ImportError:
    raise ImportError(
        "Please install anndata: pip install anndata"
    )

from ..differential import DifferentialExpression, compute_weighted_mean_fold_change
from .utils import (
    _sanitize_name,
    parse_groups,
    generate_output_field_names,
    detect_output_field_overwrite,
    get_environment_info,
    check_underrepresentation,
    apply_cell_filter,
    refine_filter_for_underrepresentation,
    get_json_metadata,
    set_json_metadata,
    get_run_history,
    append_to_run_history
)

logger = logging.getLogger("kompot")


def compute_differential_expression(
    adata,
    groupby: str,
    condition1: str,
    condition2: str,
    obsm_key: str = "DM_EigenVectors",
    layer: Optional[str] = None,
    genes: Optional[List[str]] = None,
    n_landmarks: Optional[int] = 5000,
    landmarks: Optional[np.ndarray] = None,
    sample_col: Optional[str] = None,
    differential_abundance_key: Optional[str] = None,
    sigma: float = 1.0,
    ls: Optional[float] = None,
    ls_factor: float = 10.0,
    compute_mahalanobis: bool = True,
    jit_compile: bool = False,
    eps: float = 1e-8,  # Added epsilon parameter for numerical stability
    random_state: Optional[int] = None,
    batch_size: int = 100,
    store_arrays_on_disk: Optional[bool] = None,
    disk_storage_dir: Optional[str] = None,
    max_memory_ratio: float = 0.8,
    cell_filter: Optional[Union[str, List[str], Dict[str, Any], List[Dict[str, Any]]]] = None,
    groups: Optional[Union[str, Dict[str, Any], List[Dict[str, Any]], pd.Series, np.ndarray, List[np.ndarray]]] = None,
    min_cells: int = 2,
    min_percentage: Optional[float] = None,
    check_representation: Optional[bool] = None,
    copy: bool = False,
    inplace: bool = True,
    result_key: str = "kompot_de",
    overwrite: Optional[bool] = None,
    store_landmarks: bool = False,
    return_full_results: bool = False,
    store_posterior_covariance: bool = False,
    allow_single_condition_variance: bool = False,
    progress: bool = True,
    **function_kwargs
) -> Union[Dict[str, np.ndarray], Any]:
    """
    Compute differential expression between two conditions directly from an AnnData object.
    
    This function is a scverse-compatible wrapper around the DifferentialExpression class
    that operates directly on AnnData objects.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object containing cells from both conditions.
    groupby : str
        Column in adata.obs containing the condition labels.
    condition1 : str
        Label in the groupby column identifying the first condition.
    condition2 : str
        Label in the groupby column identifying the second condition.
    obsm_key : str, optional
        Key in adata.obsm containing the cell states (e.g., PCA, diffusion maps),
        by default "DM_EigenVectors".
    layer : str, optional
        Layer in adata.layers containing gene expression data. If None, use adata.X,
        by default None.
    genes : List[str], optional
        List of gene names to include in the analysis. If None, use all genes,
        by default None.
    n_landmarks : int, optional
        Number of landmarks to use for approximation. If None, use all points,
        by default 5000. Ignored if landmarks is provided.
    landmarks : np.ndarray, optional
        Pre-computed landmarks to use. If provided, n_landmarks will be ignored.
        Shape (n_landmarks, n_features).
    sample_col : str, optional
        Column name in adata.obs containing sample labels. If provided, these will be used
        to compute sample-specific variance and will automatically enable sample variance
        estimation.
    allow_single_condition_variance : bool, optional
        If True, allows variance estimation with only one condition having multiple samples.
        By default False, which requires both conditions to have multiple samples.
    differential_abundance_key : str, optional
        Key in adata.obs where abundance log-fold changes are stored, by default None.
        Will be used for weighted mean log-fold change computation.
    sigma : float, optional
        Noise level for function estimator, by default 1.0.
    ls : float, optional
        Length scale for the GP kernel. If None, it will be estimated, by default None.
    ls_factor : float, optional
        Multiplication factor to apply to length scale when it's automatically inferred,
        by default 10.0. Only used when ls is None.
    compute_mahalanobis : bool, optional
        Whether to compute Mahalanobis distances for gene ranking, by default True.
    jit_compile : bool, optional
        Whether to use JAX just-in-time compilation, by default False.
    eps : float, optional
        Small constant for numerical stability in covariance matrices, by default 1e-8.
        Increase this value if Cholesky decomposition fails during Mahalanobis distance computation.
    random_state : int, optional
        Random seed for reproducible landmark selection when n_landmarks is specified.
        Controls the random selection of points when using approximation, by default None.
    batch_size : int, optional
        Number of cells to process at once during prediction and Mahalanobis distance computation
        to manage memory usage. If None or 0, all samples will be processed at once. Default is 100.
    store_arrays_on_disk : bool, optional
        Whether to store large arrays on disk instead of in memory, by default None.
        If None, it will be determined based on disk_storage_dir (True if provided, False otherwise).
        This is useful for very large datasets with many genes, where covariance
        matrices would otherwise exceed available memory.
    disk_storage_dir : str, optional
        Directory to store arrays on disk. If provided and store_arrays_on_disk is None,
        store_arrays_on_disk will be set to True. If store_arrays_on_disk is False and
        this is provided, a warning will be logged and disk storage will not be used.
    max_memory_ratio : float, optional
        Maximum fraction of available memory that arrays should occupy before
        triggering warnings or enabling disk storage, by default 0.8 (80%).
    cell_filter : str, List[str], Dict, List[Dict], optional
        Specification for cells or groups to exclude from the analysis.
        Will be interpreted in the following ways:
        
        - If str and `groups` is provided: Interpreted as a group name to exclude from the 
          groups defined by the `groups` parameter.
        - If List[str] and `groups` is provided: Multiple group names to exclude from the
          groups defined by the `groups` parameter.
        - If Dict: Keys are column names in adata.obs, and values are specific values to exclude.
        - If List[Dict]: Multiple dictionaries specifying different exclusion criteria.
        
        Cells matching any of the specified exclusion criteria will be excluded from the analysis.
        The string and list of strings formats are only valid when the `groups` parameter is also provided,
        as they refer to excluding groups from the subset analysis. The dictionary formats work independently
        to exclude cells based on their metadata.

    groups : str, Dict, Dict[str, Dict], List[Dict], pd.Series, np.ndarray, List[np.ndarray], optional
        Specification for subsetting or grouping cells for additional analysis.
        Will be interpreted in the following ways:
        
        - If str: Used as column name in adata.obs.
          - If column is boolean: True values form a subset.
          - If column is categorical or string: Each unique value forms a subset.
          - If column doesn't allow grouping (e.g., floats): Raises an error.
        - If Dict: Interpreted as a filter with keys being column names of adata.obs 
          and values being allowed values in this column.
          Example: {'category': ['cat1', 'cat2'], 'is_selected': True} creates a subset of cells where 
          category is 'cat1' or 'cat2' AND is_selected is True.
        - If Dict[str, Dict]: Dict of filters for different subgroups, where outer dict keys are 
          used as subset names. Each inner dict defines a filter as above.
          Example: {'control_group': {'treatment': 'control'}, 'high_dose': {'treatment': 'drug', 'dose': 'high'}}
          creates two named subsets using the provided names as identifiers.
        - If List[Dict]: Each dictionary specifies a different subset using the same
          filtering mechanism as above, but subset names are auto-generated.
        - If pd.Series or np.ndarray: Interpreted like a column specified with a string.
        - If array of appropriate shape with boolean values: Each row specifies a subset.
        - If List of vectors/series: Each element is processed as above.
        
        When subsetting is defined, the global comparison is still run first, followed by
        analyses on each subset. Only the 'mean_log_fold_change', 'weighted_mean_log_fold_change',
        and 'mahalanobis_distances' metrics are saved for each subset with appropriate name suffixes.
    min_cells : int, optional
        Minimum number of cells required for a condition to be considered adequately represented 
        within each group, by default 10.
    min_percentage : float, optional
        Minimum percentage of cells required for a condition within each group, relative to
        total cells in the group. If None, uses 10% divided by the number of conditions, by default None.
    check_representation : None or bool, optional
        Controls checking for underrepresentation when groups are specified, by default None.
        - If None: Checks and warns about underrepresentation but does not filter automatically
        - If True: Checks for underrepresentation and automatically applies the filter
          (informs the user what was excluded)
        - If False: Skips the underrepresentation check entirely
    copy : bool, optional
        If True, return a copy of the AnnData object with results added,
        by default False.
    inplace : bool, optional
        If True, modify adata in place, by default True.
    result_key : str, optional
        Key in adata.uns where results will be stored, by default "kompot_de".
    overwrite : bool, optional
        Controls behavior when results with the same result_key already exist:
        
        - If None (default): Behaves contextually:
          
          * For partial reruns with sample variance added where other parameters match,
            logs an informative message at INFO level and proceeds with overwriting
          * For other cases, warns about existing results but proceeds with overwriting
        
        - If True: Silently overwrite existing results
        - If False: Raise an error if results would be overwritten
        
        Note: When running with sample_col for a subset of cells that were previously
        analyzed without sample variance, only fields affected by sample variance will 
        be modified. Fields unaffected by sample variance will be overwritten but not
        change if the parameters match.
    store_landmarks : bool, optional
        Whether to store landmarks in adata.uns for future reuse, by default False.
        Setting to True will allow reusing landmarks with future analyses but may 
        significantly increase the AnnData file size.
    return_full_results : bool, optional
        If True, return the full results dictionary including the differential model,
        by default False. If False, and if copy=True, only return the AnnData object.
    store_posterior_covariance : bool, optional
        Whether to store the posterior covariance matrix in adata.obsp. Only available when 
        not using sample variance (sample_col=None).
        The covariance matrix can be quite large, as it is of shape (n_cells, n_cells), so this
        should be used carefully with large datasets. Default is False.
    progress : bool, optional
        Whether to show progress bars during computation. When True, displays tqdm.auto progress 
        bars for all batch processing operations including prediction, uncertainty computation, 
        and Mahalanobis distance calculations. When False, all progress bars are disabled. 
        Default is True.
    **function_kwargs : dict
        Additional arguments to pass to the FunctionEstimator.
        
    Returns
    -------
    Union[Dict[str, np.ndarray], AnnData, Tuple[Dict[str, np.ndarray], AnnData]]
        If copy is True and return_full_results is False, returns the modified AnnData object.
        If copy is True and return_full_results is True, returns a tuple (results_dict, adata).
        If inplace is True and return_full_results is False, returns None (adata is modified in place).
        If inplace is True and return_full_results is True, returns the results dictionary.
        Otherwise, returns a dictionary of results.
    
    Notes
    -----
    Results are stored in various components of the AnnData object:
    
    - adata.var[f"{result_key}_mahalanobis"]: Mahalanobis distance for each gene
    - adata.var[f"{result_key}_weighted_lfc"]: Weighted mean log fold change for each gene
    - adata.layers[f"{result_key}_condition1_imputed"]: Imputed expression for condition 1
    - adata.layers[f"{result_key}_condition2_imputed"]: Imputed expression for condition 2
    - adata.layers[f"{result_key}_fold_change"]: Log fold change for each cell and gene
    - adata.obsp["posterior_covariance"]: If store_posterior_covariance=True and conditions are met, 
      the posterior covariance matrix. Shape (n_cells, n_cells).
    - adata.layers[f"{result_key}_fold_change_zscores"]: Z-scores of log fold changes accounting for uncertainty (and sample variance if sample_col is provided)
    - adata.uns[result_key]: Dictionary with additional information and parameters
    
    Posterior standard deviations of imputed expression values are stored in:
    - If sample_col is not None (with sample variance):
      - adata.layers[f"{result_key}_{condition1}_std"]: Cell-wise standard deviation for condition 1 (sparse matrix)
      - adata.layers[f"{result_key}_{condition2}_std"]: Cell-wise standard deviation for condition 2 (sparse matrix)
    - If sample_col is None (without sample variance):
      - adata.obs[f"{result_key}_{condition1}_std"]: Cell-wise standard deviation for condition 1 (same for all genes)
      - adata.obs[f"{result_key}_{condition2}_std"]: Cell-wise standard deviation for condition 2 (same for all genes)
    
    If landmarks are computed, they are stored in adata.uns[result_key]['landmarks']
    for potential reuse in other analyses.
    """
    
    # Generate standardized field names
    field_names = generate_output_field_names(
        result_key=result_key,
        condition1=condition1,
        condition2=condition2,
        analysis_type="de",
        with_sample_suffix=(sample_col is not None),
        sample_suffix="_sample_var" if sample_col is not None else ""
    )
    
    # Get all patterns from field_names
    all_patterns = field_names["all_patterns"]
    
    # Update all_patterns with differential abundance integration if needed
    if differential_abundance_key is not None:
        field_names["has_weighted_lfc"] = True
        all_patterns["var"].append(field_names["weighted_lfc_key"])
    
    # Update all_patterns with group information if needed
    if groups is not None:
        field_names["has_groups"] = True
        if "varm" not in all_patterns:
            all_patterns["varm"] = []
            
        # Always include mean LFC for groups
        all_patterns["varm"].append(field_names["mean_lfc_varm_key"])
        
        # Only add mahalanobis varm key if compute_mahalanobis is True
        if compute_mahalanobis:
            all_patterns["varm"].append(field_names["mahalanobis_varm_key"])
            
        # Only add weighted_lfc varm key if differential_abundance_key is provided
        if differential_abundance_key is not None:
            all_patterns["varm"].append(field_names["weighted_lfc_varm_key"])
            
        # Filter out None values
        all_patterns["varm"] = [k for k in all_patterns["varm"] if k is not None]
    
    # Track overall results
    has_overwrites = False
    existing_fields = []
    prev_run = None
    
    # Check each location only once to avoid duplicate warnings
    for location, patterns in all_patterns.items():
        # Detect if we'd overwrite any existing fields in this location
        has_loc_overwrites, loc_fields, loc_prev_run = detect_output_field_overwrite(
            adata=adata,
            result_key=result_key,
            output_patterns=patterns,
            location=location,
            with_sample_suffix=(sample_col is not None),
            sample_suffix="_sample_var" if sample_col is not None else "",
            result_type=f"differential expression ({location})",
            analysis_type="de"
        )
        
        # Update overall results
        has_overwrites = has_overwrites or has_loc_overwrites
        existing_fields.extend(loc_fields)
        if loc_prev_run is not None:
            prev_run = loc_prev_run
    
    # Handle overwrite detection results
    if has_overwrites:
        # Format the message about existing results
        message = f"Differential expression results with result_key='{result_key}' already exist in the dataset."
        
        if prev_run:
            prev_timestamp = prev_run.get('timestamp', 'unknown time')
            prev_params = prev_run.get('params', {})
            prev_conditions = f"{prev_params.get('condition1', 'unknown')} to {prev_params.get('condition2', 'unknown')}"
            message += f" Previous run was at {prev_timestamp} comparing {prev_conditions}."
            
            # List fields that will be overwritten
            if existing_fields:
                field_list = ", ".join(existing_fields[:5])
                if len(existing_fields) > 5:
                    field_list += f" and {len(existing_fields) - 5} more fields"
                
                # Add note about partial overwrites if switching sample variance mode
                prev_sample_var = prev_run.get('params', {}).get('use_sample_variance', False)
                current_sample_var = (sample_col is not None)
                
                # Check if parameters coincide for the partial rerun case
                params_match = True
                
                # These are the key parameters that should match for a valid partial rerun
                key_params = ['groupby', 'condition1', 'condition2', 'obsm_key', 'layer', 'ls_factor']
                
                for param in key_params:
                    curr_val = locals().get(param)
                    prev_val = prev_params.get(param)
                    if curr_val != prev_val:
                        params_match = False
                        logger.debug(f"Parameter mismatch: {param} (current: {curr_val}, previous: {prev_val})")
                
                if prev_sample_var != current_sample_var:
                    if current_sample_var and params_match:
                        message += (f" Fields that will be overwritten: {field_list}. "
                                   f"Note: Only fields NOT affected by sample variance (like mean_log_fold_change, "
                                   f"imputed data, fold_change) will be overwritten since they "
                                   f"don't use the sample variance suffix. These results will likely be identical "
                                   f"if other parameters haven't changed.")
                    elif current_sample_var:
                        message += (f" Fields that will be overwritten: {field_list}. "
                                   f"Note: Only fields NOT affected by sample variance (like mean_log_fold_change, "
                                   f"imputed data, fold_change) will be overwritten since they "
                                   f"don't use the sample variance suffix.")
                    else:
                        message += (f" Fields that will be overwritten: {field_list}. "
                                   f"Note: Only fields NOT affected by sample variance will be overwritten "
                                   f"since sample variance-specific fields use a different suffix.")
                else:
                    message += f" Fields that will be overwritten: {field_list}"
        
        # Handle overwrite settings
        if overwrite is False:
            message += " Set overwrite=True to overwrite or use a different result_key."
            raise ValueError(message)
        elif overwrite is None:
            # Determine if this is a partial rerun with sample variance where parameters match
            params_match = False
            if prev_run:
                prev_params = prev_run.get('params', {})
                prev_sample_var = prev_params.get('use_sample_variance', False)
                current_sample_var = (sample_col is not None)
                
                # Check if this is a partial rerun with sample variance added
                if current_sample_var and not prev_sample_var:
                    # Check if key parameters match
                    params_match = True
                    key_params = ['groupby', 'condition1', 'condition2', 'obsm_key', 'layer', 'ls_factor']
                    
                    for param in key_params:
                        curr_val = locals().get(param)
                        prev_val = prev_params.get(param)
                        if curr_val != prev_val:
                            params_match = False
                            logger.debug(f"Parameter mismatch: {param} (current: {curr_val}, previous: {prev_val})")
            
            # If this is a partial rerun with matching parameters, log as info instead of warning
            if prev_run and params_match and current_sample_var and not prev_sample_var:
                logger.info(message + " This is a partial rerun with sample variance added to a previous analysis with matching parameters. " +
                          "Set overwrite=False to prevent overwriting or overwrite=True to silence this message.")
            else:
                logger.warning(message + " Set overwrite=False to prevent overwriting or overwrite=True to silence this message.")

    # Extract cell states
    if obsm_key not in adata.obsm:
        error_msg = f"Key '{obsm_key}' not found in adata.obsm. Available keys: {list(adata.obsm.keys())}"
        
        # Add helpful guidance if the missing key is the default DM_EigenVectors
        if obsm_key == "DM_EigenVectors":
            error_msg += ("\n\nTo compute DM_EigenVectors (diffusion map eigenvectors), use the Palantir package:\n"
                        "```python\n"
                        "import palantir\n"
                        "# Compute diffusion maps - this automatically adds DM_EigenVectors to adata.obsm\n"
                        "palantir.utils.run_diffusion_maps(adata)\n"
                        "```\n"
                        "See https://github.com/dpeerlab/Palantir for installation and documentation.\n\n"
                        "Alternatively, specify a different obsm_key that exists in your dataset, such as 'X_pca'.")
                        
        raise ValueError(error_msg)
    
    if groupby not in adata.obs:
        raise ValueError(f"Column '{groupby}' not found in adata.obs. Available columns: {list(adata.obs.columns)}")

    # Check if differential_abundance_key-related columns exist instead of the key itself
    if differential_abundance_key is not None:
        # Sanitize condition names for use in column names
        cond1_safe = _sanitize_name(condition1)
        cond2_safe = _sanitize_name(condition2)
        
        # Check for condition-specific column names
        specific_cols = [f"{differential_abundance_key}_log_density_{cond1_safe}", 
                       f"{differential_abundance_key}_log_density_{cond2_safe}"]
        
        if not all(col in adata.obs for col in specific_cols):
            raise ValueError(f"Log density columns not found in adata.obs. "
                           f"Expected: {specific_cols}. "
                           f"Available columns: {list(adata.obs.columns)}")
    
    # Make a copy if requested
    if copy:
        adata = adata.copy()
    
    # Create masks for each condition
    mask1 = (adata.obs[groupby] == condition1).values
    mask2 = (adata.obs[groupby] == condition2).values
    
    if np.sum(mask1) == 0:
        raise ValueError(f"Condition '{condition1}' not found in '{groupby}'.")
    if np.sum(mask2) == 0:
        raise ValueError(f"Condition '{condition2}' not found in '{groupby}'.")
    
    logger.info(f"Condition 1 ({condition1}): {np.sum(mask1):,} cells")
    logger.info(f"Condition 2 ({condition2}): {np.sum(mask2):,} cells")

    # First, check for underrepresentation on the full dataset
    # This happens BEFORE any filtering
    underrep = {}
    auto_filter = False
    underrep_filter = None

    if check_representation is True and groups is None:
        raise ValueError(
            "Cannot check underrerpresentation if no groups are specified. Set `check_representation=False` or pass `groups`."
        )
    
    if groups is not None and check_representation is not False and cell_filter is None:
        logger.info("Checking for underrepresentation on the full dataset")
        underrep_result = check_underrepresentation(
            adata, 
            groupby=groupby, 
            groups=groups,
            conditions=[condition1, condition2],
            min_cells=min_cells,
            min_percentage=min_percentage,
            warn=(check_representation is None),  # Only warn if None, not if True
            print_summary=False  # Don't print summary when used internally
        )
        
        # Extract the underrepresentation data from the result for reporting
        if "__underrepresentation_data" in underrep_result:
            underrep = underrep_result.pop("__underrepresentation_data")
        
        # If check_representation is True and underrepresentation is found, create auto filter
        if check_representation is None and underrep:
                logger.warning("Please pass `check_representation=True` to enable filtering out these underrepresented groups.")
        elif check_representation is True and underrep:
            underrep_filter = underrep_result
            n_groups = len(underrep)
            logger.info(f"Found {n_groups:,} groups with underrepresented conditions")
            for group, conditions in underrep.items():
                logger.info(f"  - Group '{group}': Underrepresented conditions: {conditions}")
            
            # No user-provided filter, use underrepresentation filter
            logger.info("Automatically applying underrepresentation filter")
            cell_filter = underrep_filter
            auto_filter = True
    
    # Apply the cell_filter to get a filter mask
    filter_mask, filter_details = apply_cell_filter(
        adata=adata, 
        cell_filter=cell_filter,
        groups=groups,
        check_representation=check_representation,
        groupby=groupby,
        conditions=[condition1, condition2],
        min_cells=min_cells,
        min_percentage=min_percentage
    )
    
    # Extract filtered cell count from filter details
    excluded_cells = adata.n_obs - filter_details["filtered_cells"]

    # Apply filter mask to condition masks
    filtered_mask1 = mask1 & filter_mask
    filtered_mask2 = mask2 & filter_mask
    
    # Check if we have enough cells after filtering
    if np.sum(filtered_mask1) < 2:
        raise ValueError(f"After filtering, condition '{condition1}' has fewer than 2 cells ({np.sum(filtered_mask1)}). "
                      f"Consider adjusting your filter criteria.")
    if np.sum(filtered_mask2) < 2:
        raise ValueError(f"After filtering, condition '{condition2}' has fewer than 2 cells ({np.sum(filtered_mask2)}). "
                      f"Consider adjusting your filter criteria.")
    
    # Log filtered cell counts
    if cell_filter is not None or auto_filter:
        logger.info(f"After filtering - Condition 1 ({condition1}): {np.sum(filtered_mask1):,} cells")
        logger.info(f"After filtering - Condition 2 ({condition2}): {np.sum(filtered_mask2):,} cells")
    
    # Update masks with filtering
    mask1 = filtered_mask1
    mask2 = filtered_mask2
    
    # Extract cell states for each condition
    X_condition1 = adata.obsm[obsm_key][mask1]
    X_condition2 = adata.obsm[obsm_key][mask2]
    
    # Extract gene expression
    if layer is not None:
        if layer not in adata.layers:
            raise ValueError(f"Layer '{layer}' not found in adata.layers. Available layers: {list(adata.layers.keys())}")
        expr1 = adata.layers[layer][mask1]
        expr2 = adata.layers[layer][mask2]
    else:
        expr1 = adata.X[mask1]
        expr2 = adata.X[mask2]
    
    # Convert to dense if sparse
    if sparse.issparse(expr1):
        expr1 = expr1.toarray()
    if sparse.issparse(expr2):
        expr2 = expr2.toarray()
    
    # Filter genes if requested
    if genes is not None:
        # Create a set for efficient lookups
        genes_set = set(genes)
        
        # Check for missing genes
        missing_genes = [gene for gene in genes_set if gene not in adata.var_names]
        if missing_genes:
            raise ValueError(f"The following genes were not found in adata.var_names: {missing_genes[:10]}" +
                          (f"... and {len(missing_genes) - 10} more" if len(missing_genes) > 10 else ""))

        # Preserve the order of adata.var_names but filter to only the requested genes
        selected_genes = [gene for gene in adata.var_names if gene in genes_set]
        gene_indices = [list(adata.var_names).index(gene) for gene in selected_genes]
        expr1 = expr1[:, gene_indices]
        expr2 = expr2[:, gene_indices]
    else:
        selected_genes = adata.var_names.tolist()
    
    # Check if we have landmarks that can be reused
    stored_landmarks = None
    
    # First check if landmarks are directly provided
    if landmarks is not None:
        logger.info(f"Using provided landmarks with shape {landmarks.shape}")
    
    # Next, check if we have landmarks in uns for this specific result_key
    elif result_key in adata.uns and 'landmarks' in adata.uns[result_key]:
        stored_landmarks = adata.uns[result_key]['landmarks']
        landmarks_dim = stored_landmarks.shape[1]
        data_dim = adata.obsm[obsm_key].shape[1]
        
        # Only use the stored landmarks if dimensions match
        if landmarks_dim == data_dim:
            logger.info(f"Using stored landmarks from adata.uns['{result_key}']['landmarks'] with shape {stored_landmarks.shape}")
            landmarks = stored_landmarks
        else:
            logger.warning(f"Stored landmarks have dimension {landmarks_dim} but data has dimension {data_dim}. Will check for other landmarks.")
    
    # If we have differential_abundance_key, check if there are landmarks stored there
    if landmarks is None and differential_abundance_key is not None and differential_abundance_key in adata.uns and 'landmarks' in adata.uns[differential_abundance_key]:
        stored_abund_landmarks = adata.uns[differential_abundance_key]['landmarks']
        landmarks_dim = stored_abund_landmarks.shape[1]
        data_dim = adata.obsm[obsm_key].shape[1]
        
        # Only use the stored landmarks if dimensions match
        if landmarks_dim == data_dim:
            logger.info(f"Using landmarks from abundance analysis in adata.uns['{differential_abundance_key}']['landmarks'] with shape {stored_abund_landmarks.shape}")
            landmarks = stored_abund_landmarks
        else:
            logger.warning(f"Abundance landmarks have dimension {landmarks_dim} but data has dimension {data_dim}. Will check for other landmarks.")
    
    # If still no landmarks, check for any other landmarks in storage_key
    if landmarks is None:
        storage_key = "kompot_de"
        
        if storage_key in adata.uns and storage_key != result_key and 'landmarks' in adata.uns[storage_key]:
            other_landmarks = adata.uns[storage_key]['landmarks']
            landmarks_dim = other_landmarks.shape[1]
            data_dim = adata.obsm[obsm_key].shape[1]
            
            # Only use the stored landmarks if dimensions match
            if landmarks_dim == data_dim:
                logger.info(f"Reusing stored DE landmarks from adata.uns['{storage_key}']['landmarks'] with shape {other_landmarks.shape}")
                landmarks = other_landmarks
            else:
                logger.warning(f"Other stored DE landmarks have dimension {landmarks_dim} but data has dimension {data_dim}. Will check for other landmarks.")
    
    # As a last resort, check for DA landmarks if not already checked
    if landmarks is None and "kompot_da" in adata.uns and 'landmarks' in adata.uns["kompot_da"] and (differential_abundance_key != "kompot_da"):
        da_landmarks = adata.uns["kompot_da"]['landmarks']
        landmarks_dim = da_landmarks.shape[1]
        data_dim = adata.obsm[obsm_key].shape[1]
        
        # Only use the stored landmarks if dimensions match
        if landmarks_dim == data_dim:
            logger.info(f"Reusing differential abundance landmarks from adata.uns['kompot_da']['landmarks'] with shape {da_landmarks.shape}")
            landmarks = da_landmarks
        else:
            logger.warning(f"DA landmarks have dimension {landmarks_dim} but data has dimension {data_dim}. Computing new landmarks.")
    
    # Initialize and fit DifferentialExpression
    use_sample_variance = sample_col is not None
    
    diff_expression = DifferentialExpression(
        # Don't disable landmarks - we'll compute posterior covariance separately
        n_landmarks=n_landmarks,
        use_sample_variance=use_sample_variance,
        eps=eps,  # Pass the eps parameter 
        jit_compile=jit_compile,
        random_state=random_state,
        batch_size=batch_size,
        store_arrays_on_disk=store_arrays_on_disk,
        disk_storage_dir=disk_storage_dir,
        max_memory_ratio=max_memory_ratio
    )
    
    # Extract sample indices from sample_col if provided
    condition1_sample_indices = None
    condition2_sample_indices = None

    if sample_col is not None:
        if sample_col not in adata.obs:
            raise ValueError(f"Column '{sample_col}' not found in adata.obs. Available columns: {list(adata.obs.columns)}")
        
        # Extract sample indices for each condition
        condition1_sample_indices = adata.obs[sample_col][mask1].values
        condition2_sample_indices = adata.obs[sample_col][mask2].values
        
        logger.info(f"Using sample column '{sample_col}' for sample variance estimation")
        logger.info(f"Found {len(np.unique(condition1_sample_indices))} unique sample(s) in condition 1")
        logger.info(f"Found {len(np.unique(condition2_sample_indices))} unique sample(s) in condition 2")
    
    # Fit the estimators
    diff_expression.fit(
        X_condition1, expr1,
        X_condition2, expr2,
        sigma=sigma,
        ls=ls,
        ls_factor=ls_factor,
        landmarks=landmarks,
        condition1_sample_indices=condition1_sample_indices,
        condition2_sample_indices=condition2_sample_indices,
        allow_single_condition_variance=allow_single_condition_variance,
        **function_kwargs
    )
    
    # Handle landmarks for future reference
    if hasattr(diff_expression, 'computed_landmarks') and diff_expression.computed_landmarks is not None:
        # Initialize if needed
        if result_key not in adata.uns:
            adata.uns[result_key] = {}
        
        # Get landmarks info
        landmarks_shape = str(diff_expression.computed_landmarks.shape)
        landmarks_dtype = str(diff_expression.computed_landmarks.dtype)
        
        # Create landmarks info
        landmarks_info = {
            'shape': landmarks_shape,
            'dtype': landmarks_dtype,
            'source': 'computed',
            'n_landmarks': landmarks_shape[0],
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        # Store landmarks info in both locations
        adata.uns[result_key]['landmarks_info'] = landmarks_info
        
        storage_key = "kompot_de"
        if storage_key not in adata.uns:
            adata.uns[storage_key] = {}
        adata.uns[storage_key]['landmarks_info'] = landmarks_info.copy()
        
        # Store the actual landmarks if requested
        if store_landmarks:
            adata.uns[result_key]['landmarks'] = diff_expression.computed_landmarks
            
            # Store only in the result_key, not automatically in storage_key
            # We'll check across keys when searching for landmarks
            
            logger.info(f"Stored landmarks in adata.uns['{result_key}']['landmarks'] with shape {landmarks_shape} for future reuse")
        else:
            logger.info(f"Landmark storage skipped (store_landmarks=False). Compute with store_landmarks=True to enable landmark reuse.")
    else:
        logger.debug("No computed landmarks found to store. Check if landmarks were pre-computed or if n_landmarks is set correctly.")
    
    # Run prediction to compute fold changes, metrics, and Mahalanobis distances
    # Use the filter_mask to only get predictions for cells we're interested in
    X_for_prediction = adata.obsm[obsm_key][filter_mask]
    
    # Check if we can store posterior covariance (not using sample variance)
    # We now allow storing posterior covariance even when landmarks are used elsewhere
    can_store_covariance = store_posterior_covariance and not use_sample_variance
    
    # Log warning if requested but not possible
    if store_posterior_covariance and not can_store_covariance:
        if use_sample_variance:
            logger.warning("Cannot store posterior covariance when using sample variance. Posterior covariance will not be stored.")
    
    # Run prediction
    expression_results = diff_expression.predict(
        X_for_prediction, 
        compute_mahalanobis=compute_mahalanobis,
        progress=progress,
    )
    
    # Store posterior covariance matrix in obsp if requested and possible
    if can_store_covariance:
        logger.info("Computing posterior covariance matrix for storing in obsp...")
        
        # Get the posterior covariance matrix (need to compute it separately)
        try:
            # Get covariance matrices from the function predictors
            cov1 = diff_expression.function_predictor1.covariance(X_for_prediction, diag=False)
            cov2 = diff_expression.function_predictor2.covariance(X_for_prediction, diag=False)
            
            # The covariance of log fold change is the sum of covariances
            posterior_covariance = cov1 + cov2
            
            # Get the proper name for the obsp key from field_names
            posterior_cov_key = field_names["posterior_covariance_key"]
            
            # Store the covariance matrix in obsp - need to handle the full matrix
            # Create a full-size matrix initialized with zeros or NaN
            n_cells = adata.shape[0]
            full_covariance = np.zeros((n_cells, n_cells))
            
            # Fill in the values for the cells that were included in the analysis
            filtered_indices = np.where(filter_mask)[0]
            
            # Check shapes to ensure compatibility
            n_filtered = len(filtered_indices)
            if posterior_covariance.shape != (n_filtered, n_filtered):
                logger.warning(f"Shape mismatch: posterior_covariance {posterior_covariance.shape} vs expected ({n_filtered}, {n_filtered})")
                # Fall back to loop-based assignment which is safer
                for i, row_idx in enumerate(filtered_indices):
                    for j, col_idx in enumerate(filtered_indices):
                        full_covariance[row_idx, col_idx] = posterior_covariance[i, j]
            else:
                # Use vectorized assignment when shapes match
                for i, row_idx in enumerate(filtered_indices):
                    full_covariance[row_idx, filtered_indices] = posterior_covariance[i]
            
            # Store in obsp
            adata.obsp[posterior_cov_key] = full_covariance
            logger.info(f"Stored posterior covariance matrix in adata.obsp['{posterior_cov_key}']")
            
            # We'll add this to the field mapping later when it's created
        except Exception as e:
            logger.error(f"Failed to compute and store posterior covariance matrix: {str(e)}")
            logger.error("Posterior covariance matrix will not be stored in obsp.")
    
    
    # Separately compute weighted fold changes if needed
    if differential_abundance_key is not None:
        # Sanitize condition names for use in column names
        cond1_safe = _sanitize_name(condition1)
        cond2_safe = _sanitize_name(condition2)
        
        # Get log densities from adata with descriptive names
        density_col1 = f"{differential_abundance_key}_log_density_{cond1_safe}"
        density_col2 = f"{differential_abundance_key}_log_density_{cond2_safe}"
        
        if density_col1 in adata.obs and density_col2 in adata.obs:
            # Apply the filter mask to get only the cells we're predicting for
            log_density_condition1 = adata.obs[density_col1][filter_mask]
            log_density_condition2 = adata.obs[density_col2][filter_mask]
            
            # Calculate log density difference directly
            log_density_diff = log_density_condition2 - log_density_condition1
            
            # Use the standalone function to compute weighted mean fold change with pre-computed difference
            # The exp(abs()) is now handled inside the function
            expression_results['weighted_mean_log_fold_change'] = compute_weighted_mean_fold_change(
                expression_results['fold_change'],
                log_density_diff=log_density_diff
            )
        else:
            logger.warning(f"Log density columns not found in adata.obs. Expected: {density_col1}, {density_col2}. "
                           f"Will not compute weighted mean fold changes.")
    
    # Create result dictionary
    result_dict = {
        "mean_log_fold_change": expression_results['mean_log_fold_change'],
        "condition1_imputed": expression_results['condition1_imputed'],
        "condition2_imputed": expression_results['condition2_imputed'],
        "fold_change": expression_results['fold_change'],
        "fold_change_zscores": expression_results['fold_change_zscores'],
        "model": diff_expression,
    }
    
    # Add underrepresentation info if available
    if 'underrep' in locals():
        result_dict["underrepresentation"] = underrep
        if 'auto_filter' in locals() and auto_filter:
            result_dict["auto_filtered"] = True
    
    # Add optional result fields
    if compute_mahalanobis:
        result_dict["mahalanobis_distances"] = expression_results['mahalanobis_distances']
        
    if 'weighted_mean_log_fold_change' in expression_results:
        result_dict["weighted_mean_log_fold_change"] = expression_results['weighted_mean_log_fold_change']
        
    # Add landmarks to result dictionary if they were computed
    if hasattr(diff_expression, 'computed_landmarks') and diff_expression.computed_landmarks is not None:
        result_dict["landmarks"] = diff_expression.computed_landmarks
    
    if inplace:
        # Sanitize condition names for use in column names first
        cond1_safe = _sanitize_name(condition1)
        cond2_safe = _sanitize_name(condition2)
        
        # Add suffix when sample variance is used
        sample_suffix = "_sample_var" if sample_col is not None else ""
        
        # Create a dictionary to collect all new columns to add to adata.var
        # This will prevent dataframe fragmentation
        new_var_columns = {}
        
        # Add gene-level metrics to adata.var
        if compute_mahalanobis:
            # Make sure mahalanobis_distances is an array with the same length as selected_genes
            mahalanobis_distances = expression_results['mahalanobis_distances']
            # Convert list to numpy array if needed
            if isinstance(mahalanobis_distances, list):
                mahalanobis_distances = np.array(mahalanobis_distances)
                
            # Ensure mahalanobis_distances is 1D before reshaping
            if len(mahalanobis_distances.shape) > 1:
                logger.warning(f"mahalanobis_distances has shape {mahalanobis_distances.shape}, flattening to 1D.")
                # Take the first row if it's a 2D array
                if mahalanobis_distances.shape[0] < mahalanobis_distances.shape[1]:
                    mahalanobis_distances = mahalanobis_distances[0]  # Take first row if more columns than rows
                else:
                    mahalanobis_distances = mahalanobis_distances[:, 0]  # Take first column otherwise
            
            # Check if length matches the expected length
            if len(mahalanobis_distances) != len(selected_genes):
                logger.warning(f"Mahalanobis distances length {len(mahalanobis_distances)} doesn't match selected_genes length {len(selected_genes)}. Reshaping.")
                if len(mahalanobis_distances) < len(selected_genes):
                    # Pad with NaNs if the array is too short
                    padding = np.full(len(selected_genes) - len(mahalanobis_distances), np.nan)
                    mahalanobis_distances = np.concatenate([mahalanobis_distances, padding])
                else:
                    # Truncate if the array is too long
                    mahalanobis_distances = mahalanobis_distances[:len(selected_genes)]
                
            # Mahalanobis distance IS impacted by sample variance
            mahalanobis_key = field_names["mahalanobis_key"]
            
            # Add to collection for batch addition
            if mahalanobis_key in adata.var:
                # Only create a series for selected genes to avoid overwriting existing values
                new_var_columns[mahalanobis_key] = pd.Series(mahalanobis_distances, index=selected_genes)
            else:
                # Initialize with NaN for all genes if column doesn't exist yet
                new_var_columns[mahalanobis_key] = pd.Series(np.nan, index=adata.var_names)
                new_var_columns[mahalanobis_key].loc[selected_genes] = mahalanobis_distances
        
        if differential_abundance_key is not None:
            # Use the standardized field name from field_names
            # Weighted mean log fold change is NOT impacted by sample variance
            column_name = field_names["weighted_lfc_key"]
            
            # Extract and verify weighted_mean_log_fold_change
            weighted_lfc = expression_results['weighted_mean_log_fold_change']
            # Convert list to numpy array if needed
            if isinstance(weighted_lfc, list):
                weighted_lfc = np.array(weighted_lfc)
                
            # Ensure weighted_lfc is 1D before reshaping
            if len(weighted_lfc.shape) > 1:
                logger.warning(f"weighted_mean_log_fold_change has shape {weighted_lfc.shape}, flattening to 1D.")
                # Take the first row if it's a 2D array
                if weighted_lfc.shape[0] < weighted_lfc.shape[1]:
                    weighted_lfc = weighted_lfc[0]  # Take first row if more columns than rows
                else:
                    weighted_lfc = weighted_lfc[:, 0]  # Take first column otherwise
                
            if len(weighted_lfc) != len(selected_genes):
                logger.warning(f"weighted_mean_log_fold_change length {len(weighted_lfc)} doesn't match selected_genes length {len(selected_genes)}. Reshaping.")
                if len(weighted_lfc) < len(selected_genes):
                    # Pad with NaNs if the array is too short
                    padding = np.full(len(selected_genes) - len(weighted_lfc), np.nan)
                    weighted_lfc = np.concatenate([weighted_lfc, padding])
                else:
                    # Truncate if the array is too long
                    weighted_lfc = weighted_lfc[:len(selected_genes)]
            
            # Add to collection for batch addition
            if column_name in adata.var:
                # Only create a series for selected genes to avoid overwriting existing values
                new_var_columns[column_name] = pd.Series(weighted_lfc, index=selected_genes)
            else:
                # Initialize with NaN for all genes if column doesn't exist yet
                new_var_columns[column_name] = pd.Series(np.nan, index=adata.var_names)
                new_var_columns[column_name].loc[selected_genes] = weighted_lfc
        
        # Add mean log fold change with descriptive name
        # Use the standardized field name from field_names
        # Mean log fold change is NOT impacted by sample variance
        mean_lfc_column = field_names["mean_lfc_key"]
        
        # Extract and verify mean_log_fold_change
        mean_lfc = expression_results['mean_log_fold_change']
        # Convert list to numpy array if needed
        if isinstance(mean_lfc, list):
            mean_lfc = np.array(mean_lfc)
        
        # Ensure mean_lfc is 1D before reshaping
        if len(mean_lfc.shape) > 1:
            logger.warning(f"mean_log_fold_change has shape {mean_lfc.shape}, flattening to 1D.")
            # Take the first row if it's a 2D array
            if mean_lfc.shape[0] < mean_lfc.shape[1]:
                mean_lfc = mean_lfc[0]  # Take first row if more columns than rows
            else:
                mean_lfc = mean_lfc[:, 0]  # Take first column otherwise
            
        if len(mean_lfc) != len(selected_genes):
            logger.warning(f"mean_log_fold_change length {len(mean_lfc)} doesn't match selected_genes length {len(selected_genes)}. Reshaping.")
            if len(mean_lfc) < len(selected_genes):
                # Pad with NaNs if the array is too short
                padding = np.full(len(selected_genes) - len(mean_lfc), np.nan)
                mean_lfc = np.concatenate([mean_lfc, padding])
            else:
                # Truncate if the array is too long
                mean_lfc = mean_lfc[:len(selected_genes)]
        
        # Add to collection for batch addition
        if mean_lfc_column in adata.var:
            # Only create a series for selected genes to avoid overwriting existing values
            new_var_columns[mean_lfc_column] = pd.Series(mean_lfc, index=selected_genes)
        else:
            # Initialize with NaN for all genes if column doesn't exist yet
            new_var_columns[mean_lfc_column] = pd.Series(np.nan, index=adata.var_names)
            new_var_columns[mean_lfc_column].loc[selected_genes] = mean_lfc
        
        
        # Add all columns to adata.var at once to prevent dataframe fragmentation
        if new_var_columns:
            logger.debug(f"Adding {len(new_var_columns)} columns to adata.var at once")
            
            # Separate existing and new columns
            existing_columns = {}
            new_columns = {}
            
            for col, values in new_var_columns.items():
                if col in adata.var.columns:
                    existing_columns[col] = values
                else:
                    new_columns[col] = values
            
            # Update existing columns first, only for the selected genes
            for col, values in existing_columns.items():
                if len(values.index) == len(selected_genes):
                    # This is a Series with only selected_genes - update only those rows
                    # Vectorized update using loc with a list of genes
                    adata.var.loc[values.index, col] = values
                else:
                    # This is a full Series - should not happen with our changes, but just in case
                    adata.var[col] = values
            
            # Only concatenate for brand new columns
            if new_columns:
                new_df = pd.DataFrame(new_columns, index=adata.var.index)
                adata.var = pd.concat([adata.var, new_df], axis=1)
        
        # Add cell-gene level results
        n_selected_genes = len(selected_genes)
        
        # Process the data to match the shape of the full gene set
        if n_selected_genes < len(adata.var_names):
            # We need to expand the imputed data to the full gene set
            # Use the standardized field names
            # Create descriptive layer names - these are NOT affected by sample variance
            imputed1_key = field_names["imputed_key_1"]
            imputed2_key = field_names["imputed_key_2"]
            fold_change_key = field_names["fold_change_key"]

            # Initialize layers only if they don't already exist
            if imputed1_key not in adata.layers:
                # Only use sparse if working with a subset of genes
                if len(selected_genes) < len(adata.var_names):
                    adata.layers[imputed1_key] = sparse.csr_matrix(adata.shape)
                else:
                    adata.layers[imputed1_key] = np.zeros(adata.shape)
            if imputed2_key not in adata.layers:
                # Only use sparse if working with a subset of genes
                if len(selected_genes) < len(adata.var_names):
                    adata.layers[imputed2_key] = sparse.csr_matrix(adata.shape)
                else:
                    adata.layers[imputed2_key] = np.zeros(adata.shape)
            if fold_change_key not in adata.layers:
                # Only use sparse if working with a subset of genes
                if len(selected_genes) < len(adata.var_names):
                    adata.layers[fold_change_key] = sparse.csr_matrix(adata.shape)
                else:
                    adata.layers[fold_change_key] = np.zeros(adata.shape)
            if field_names["fold_change_zscores_key"] not in adata.layers:
                # Only use sparse if working with a subset of genes
                if len(selected_genes) < len(adata.var_names):
                    adata.layers[field_names["fold_change_zscores_key"]] = sparse.csr_matrix(adata.shape)
                else:
                    adata.layers[field_names["fold_change_zscores_key"]] = np.zeros(adata.shape)
            
            # Convert JAX arrays to NumPy arrays if needed
            condition1_imputed = np.array(expression_results['condition1_imputed'])
            condition2_imputed = np.array(expression_results['condition2_imputed'])
            fold_change = np.array(expression_results['fold_change'])
            fold_change_zscores = np.array(expression_results['fold_change_zscores'])
            condition1_std = np.array(expression_results['condition1_std'])
            condition2_std = np.array(expression_results['condition2_std'])
            
            # Create standard deviation keys
            if sample_col is not None:
                # With sample variance, store as cell-by-gene layers (sparse)
                # Initialize standard deviation layers if they don't exist
                if field_names["std_key_1"] not in adata.layers:
                    # Check if we're working with a subset of genes
                    if len(selected_genes) < len(adata.var_names):
                        # Create a sparse matrix with zeros for all genes
                        shape = (adata.shape[0], adata.shape[1])
                        adata.layers[field_names["std_key_1"]] = sparse.csr_matrix(shape)
                    else:
                        # If we're using all genes, use a dense array
                        shape = (adata.shape[0], adata.shape[1])
                        adata.layers[field_names["std_key_1"]] = np.zeros(shape)
                if field_names["std_key_2"] not in adata.layers:
                    # Check if we're working with a subset of genes
                    if len(selected_genes) < len(adata.var_names):
                        # Create a sparse matrix with zeros for all genes
                        shape = (adata.shape[0], adata.shape[1])
                        adata.layers[field_names["std_key_2"]] = sparse.csr_matrix(shape)
                    else:
                        # If we're using all genes, use a dense array
                        shape = (adata.shape[0], adata.shape[1])
                        adata.layers[field_names["std_key_2"]] = np.zeros(shape)
                
                # Create a mapping from selected gene indices to var_names indices for faster lookup
                gene_indices = np.array([adata.var_names.get_loc(gene) for gene in selected_genes])
                
                # Use vectorized operations for bulk assignment where possible
                # For sparse matrices we need to do this column by column
                for i, gene_idx in enumerate(gene_indices):
                    adata.layers[imputed1_key][:, gene_idx] = condition1_imputed[:, i]
                    adata.layers[imputed2_key][:, gene_idx] = condition2_imputed[:, i]
                    adata.layers[fold_change_key][:, gene_idx] = fold_change[:, i]
                    adata.layers[field_names["std_key_1"]][:, gene_idx] = condition1_std[:, i]
                    adata.layers[field_names["std_key_2"]][:, gene_idx] = condition2_std[:, i]
            else:
                # Without sample variance, store as .obs columns (same for all genes)
                # For this case, all genes have the same std, so we just take the first gene
                # No averaging over genes - direct extraction from the first column
                # Only assign to cells that passed the filter
                adata.obs[field_names["std_key_1"]] = np.nan  # Initialize with NaN
                adata.obs[field_names["std_key_2"]] = np.nan  # Initialize with NaN
                
                # Assign values only to cells that were included in the analysis
                adata.obs.loc[filter_mask, field_names["std_key_1"]] = condition1_std[:, 0]
                adata.obs.loc[filter_mask, field_names["std_key_2"]] = condition2_std[:, 0]
                
                # Map imputed values to the correct positions
                for i, gene in enumerate(selected_genes):
                    gene_idx = list(adata.var_names).index(gene)
                    adata.layers[imputed1_key][:, gene_idx] = condition1_imputed[:, i]
                    adata.layers[imputed2_key][:, gene_idx] = condition2_imputed[:, i]
                    adata.layers[fold_change_key][:, gene_idx] = fold_change[:, i]
                    adata.layers[field_names["fold_change_zscores_key"]][:, gene_idx] = fold_change_zscores[:, i]
        else:
            # Convert JAX arrays to NumPy arrays if needed
            condition1_imputed = np.array(expression_results['condition1_imputed'])
            condition2_imputed = np.array(expression_results['condition2_imputed'])
            fold_change = np.array(expression_results['fold_change'])
            fold_change_zscores = np.array(expression_results['fold_change_zscores'])
            condition1_std = np.array(expression_results['condition1_std'])
            condition2_std = np.array(expression_results['condition2_std'])
            
                
            # Use the standardized field names
            # Create descriptive layer names - these are NOT affected by sample variance
            imputed1_key = field_names["imputed_key_1"]
            imputed2_key = field_names["imputed_key_2"]
            fold_change_key = field_names["fold_change_key"]
            
            if sample_col is not None:
                # With sample variance, store as cell-by-gene layers
                adata.layers[imputed1_key] = condition1_imputed
                adata.layers[imputed2_key] = condition2_imputed
                adata.layers[fold_change_key] = fold_change
                adata.layers[field_names["fold_change_zscores_key"]] = fold_change_zscores
                adata.layers[field_names["std_key_1"]] = condition1_std
                adata.layers[field_names["std_key_2"]] = condition2_std
            else:
                # Without sample variance, store as .obs columns (same for all genes)
                # For this case, all genes have the same std, so we just take the first gene
                # Initialize with NaN
                adata.obs[field_names["std_key_1"]] = np.nan
                adata.obs[field_names["std_key_2"]] = np.nan
                
                # Assign values only to cells that were included in the analysis
                adata.obs.loc[filter_mask, field_names["std_key_1"]] = condition1_std[:, 0]
                adata.obs.loc[filter_mask, field_names["std_key_2"]] = condition2_std[:, 0]

                # Use dense arrays for layers, initialize with NaN
                imputed1_layer = np.full(adata.shape, np.nan)
                imputed2_layer = np.full(adata.shape, np.nan)
                fold_change_layer = np.full(adata.shape, np.nan)
                fold_change_zscores_layer = np.full(adata.shape, np.nan)
                
                # Assign values only to filtered cells - vectorized approach
                filtered_indices = np.where(filter_mask)[0]
                
                # Vectorized assignment using advanced indexing
                imputed1_layer[filtered_indices] = condition1_imputed
                imputed2_layer[filtered_indices] = condition2_imputed
                fold_change_layer[filtered_indices] = fold_change
                fold_change_zscores_layer[filtered_indices] = fold_change_zscores
                
                # Store in adata.layers
                adata.layers[imputed1_key] = imputed1_layer
                adata.layers[imputed2_key] = imputed2_layer
                adata.layers[fold_change_key] = fold_change_layer
                adata.layers[field_names["fold_change_zscores_key"]] = fold_change_zscores_layer
        
        # Prepare parameters, run timestamp, and field metadata
        current_timestamp = datetime.datetime.now().isoformat()
        
        # Define parameters dict - include ALL parameters (especially groups)
        params_dict = {
            "groupby": groupby,
            "condition1": condition1,
            "condition2": condition2,
            "obsm_key": obsm_key,
            "layer": layer,
            "genes": genes,
            "n_landmarks": n_landmarks,
            "landmarks": landmarks is not None,  # Just store if landmarks were provided, not the actual values
            "sample_col": sample_col,  # Keep this for documentation in the AnnData object
            "use_sample_variance": use_sample_variance,  # This is now inferred from sample_col
            "differential_abundance_key": differential_abundance_key,
            "sigma": sigma,
            "ls": ls,
            "ls_factor": ls_factor,
            "compute_mahalanobis": compute_mahalanobis,
            "jit_compile": jit_compile,
            "eps": eps,  # Include eps parameter for numerical stability
            "random_state": random_state,
            "used_landmarks": True if landmarks is not None else False,
            "store_arrays_on_disk": store_arrays_on_disk,
            "disk_storage_dir": disk_storage_dir,  # Store the directory path if provided
            "max_memory_ratio": max_memory_ratio,
            "batch_size": batch_size,
            "cell_filter": cell_filter,  # Store the cell filter parameter
            "groups": groups,  # Store the groups parameter - important for traceability
            "min_cells": min_cells,
            "min_percentage": min_percentage,
            "check_representation": check_representation,
            "auto_filtered": auto_filter if 'auto_filter' in locals() else False,
            "store_landmarks": store_landmarks,
            "store_posterior_covariance": store_posterior_covariance,
            "result_key": result_key,
            "copy": copy,
            "inplace": inplace,
            "overwrite": overwrite
        }
        
        # Get storage usage stats if disk storage was used
        storage_stats = None
        if store_arrays_on_disk and hasattr(diff_expression, '_disk_storage') and diff_expression._disk_storage is not None:
            try:
                storage_human, storage_bytes = diff_expression._disk_storage.total_storage_used
                storage_stats = {
                    "total_disk_usage": storage_human,
                    "disk_usage_bytes": storage_bytes,
                    "array_count": len(diff_expression._disk_storage.array_registry)
                }
            except Exception as e:
                logger.warning(f"Failed to get disk storage statistics: {e}")
                
        current_run_info = {
            "timestamp": current_timestamp,
            "function": "compute_differential_expression",
            "result_key": result_key,
            "analysis_type": "de",
            "lfc_key": field_names["mean_lfc_key"],
            "weighted_lfc_key": field_names["weighted_lfc_key"] if differential_abundance_key is not None else None,
            "mahalanobis_key": field_names["mahalanobis_key"] if compute_mahalanobis else None,
            "imputed_layer_keys": {
                "condition1": field_names["imputed_key_1"],
                "condition2": field_names["imputed_key_2"],
                "fold_change": field_names["fold_change_key"]
            },
            "field_names": field_names,
            "uses_sample_variance": sample_col is not None,
            "memory_analysis": memory_analysis if 'memory_analysis' in locals() else None,
            "storage_stats": storage_stats,
            "underrepresentation": underrep if 'underrep' in locals() else None,
            "auto_filtered": auto_filter if 'auto_filter' in locals() else False,
            "has_groups": groups is not None,
            "params": params_dict
        }
            
        # Add environment info to the run info
        env_info = get_environment_info()
        current_run_info["environment"] = env_info

        
        # Create a comprehensive field-to-location mapping for field tracking
        # This maps the full field names to their locations and descriptions
        field_mapping = {
            # Var fields
            field_names["mean_lfc_key"]: {"location": "var", "type": "mean_log_fold_change", "description": "Mean log fold change values"},
            
            # Layer fields
            field_names["imputed_key_1"]: {"location": "layers", "type": "imputed", "description": f"Imputed expression for {condition1}"},
            field_names["imputed_key_2"]: {"location": "layers", "type": "imputed", "description": f"Imputed expression for {condition2}"},
            field_names["fold_change_key"]: {"location": "layers", "type": "fold_change", "description": "Log fold change for each cell and gene"},
            field_names["fold_change_zscores_key"]: {"location": "layers", "type": "fold_change_zscores", "description": f"Z-scores of log fold changes accounting for uncertainty{' and sample variance' if sample_col is not None else ''}"},
        }
        
        # Add standard deviation fields to field_mapping
        if sample_col is not None:
            # With sample variance, posterior standard deviations are in layers (cell-by-gene)
            field_mapping[field_names["std_key_1"]] = {"location": "layers", "type": "std_with_sample_var", "description": f"Posterior standard deviation of imputed expression for {condition1} (with sample variance)"}
            field_mapping[field_names["std_key_2"]] = {"location": "layers", "type": "std_with_sample_var", "description": f"Posterior standard deviation of imputed expression for {condition2} (with sample variance)"}
        else:
            # Without sample variance, posterior standard deviations are in obs (same for all genes)
            field_mapping[field_names["std_key_1"]] = {"location": "obs", "type": "std", "description": f"Posterior standard deviation of imputed expression for {condition1} (same for all genes)"}
            field_mapping[field_names["std_key_2"]] = {"location": "obs", "type": "std", "description": f"Posterior standard deviation of imputed expression for {condition2} (same for all genes)"}
        
        # Add optional fields if present
        if compute_mahalanobis:
            field_mapping[field_names["mahalanobis_key"]] = {"location": "var", "type": "mahalanobis", "description": "Mahalanobis distances"}
            
        if differential_abundance_key is not None:
            field_mapping[field_names["weighted_lfc_key"]] = {"location": "var", "type": "weighted_mean_log_fold_change", "description": "Weighted mean log fold change"}
            
        # Add posterior covariance field if it was added to obsp
        if can_store_covariance and "posterior_covariance_key" in field_names:
            posterior_cov_key = field_names["posterior_covariance_key"]
            if posterior_cov_key in adata.obsp:
                # Create obsp section if it doesn't exist
                if "obsp" not in field_mapping:
                    field_mapping["obsp"] = {}
                    
                # Add the field mapping - the key already includes the condition pair
                field_mapping["obsp"][posterior_cov_key] = {
                    "location": "obsp", 
                    "type": "covariance", 
                    "description": f"Posterior covariance matrix for fold changes between {condition1} and {condition2}"
                }
                
                # Add to current_run_info
                current_run_info["posterior_covariance_key"] = posterior_cov_key
        
        # Process groups and perform subset-specific analyses if groups are provided
        group_results = {}
        if groups is not None:
            logger.info("Processing group-based subsetting for differential expression analysis")
            
            # Parse the groups parameter to get subset masks
            subset_masks, subset_names = parse_groups(adata, groups)
            
            # If filter is provided, apply it to the subset masks
            if filter is not None:
                filtered_subset_masks = {}
                for name, mask in subset_masks.items():
                    # Only include cells that are in both the subset and pass the filter
                    filtered_subset_masks[name] = mask & filter_mask
                subset_masks = filtered_subset_masks
            
            # Check if we have any valid subsets
            if not subset_masks:
                logger.warning("No valid subsets found based on the 'groups' parameter")
            else:
                # Log the identified subsets
                logger.info(f"Identified {len(subset_names):,} subset(s): {', '.join(subset_names)}")
                
                # Use the standardized field names for varm from field_names, but only include
                # those we need based on the computation parameters
                varm_keys = [field_names["mean_lfc_varm_key"]]  # Always include mean log fold change
                
                # Only include mahalanobis if compute_mahalanobis=True
                if compute_mahalanobis:
                    varm_keys.append(field_names["mahalanobis_varm_key"])
                
                # Only include weighted_lfc if differential_abundance_key is provided
                if differential_abundance_key is not None:
                    varm_keys.append(field_names["weighted_lfc_varm_key"])
                
                # Filter out None values
                varm_keys = [key for key in varm_keys if key is not None]
                
                # Create DataFrames with all subset columns initialized with 0.0 (to avoid NaN issues in tests)
                for varm_key in varm_keys:
                    if varm_key not in adata.varm:
                        # Create DataFrame with all subset names as columns
                        empty_df = pd.DataFrame(
                            0.0, 
                            index=adata.var_names,
                            columns=subset_names
                        )
                        adata.varm[varm_key] = empty_df
                        logger.debug(f"Initialized {varm_key} in adata.varm with columns for all {len(subset_names):,} subsets")
                
                # We're exclusively using adata.varm for storing group-specific metrics
                # This provides a cleaner design with metrics properly organized by group
                
                # For each subset, run prediction and store subset-specific metrics
                for subset_name, mask in subset_masks.items():
                    if np.sum(mask) < 2:  # Minimum cells for analysis
                        logger.warning(f"Subset '{subset_name}' has fewer than 2 cells ({np.sum(mask)}). Skipping analysis.")
                        continue
                    
                    # Only apply the mask to the current dataset's cells
                    subset_mask = mask
                    
                    logger.info(f"Running differential expression analysis on subset '{subset_name}' " 
                               f"with {np.sum(subset_mask):,} cells")
                    
                    # Get the subset of X_for_prediction for prediction
                    X_subset = adata.obsm[obsm_key][subset_mask]
                    
                    # For subsets, disable landmarks if they exist
                    use_subset_landmarks = False
                    subset_landmarks = None
                    
                    # If we have landmarks and the subset is larger than the number of landmarks
                    if hasattr(diff_expression, 'computed_landmarks') and diff_expression.computed_landmarks is not None:
                        n_landmarks = diff_expression.computed_landmarks.shape[0]
                        if np.sum(subset_mask) > n_landmarks:
                            # Subset is larger than the number of landmarks, compute new landmarks from subset
                            logger.info(f"Subset '{subset_name}' has {np.sum(subset_mask):,} cells, more than the {n_landmarks:,} landmarks. "
                                       f"Computing new landmarks from the subset.")
                            from mellon.parameters import compute_landmarks
                            subset_landmarks = compute_landmarks(
                                X_subset,
                                gp_type='fixed',
                                n_landmarks=n_landmarks,
                                random_state=diff_expression.random_state
                            )
                            use_subset_landmarks = True
                        else:
                            # Just disable landmarks for this subset
                            logger.debug(f"Disabling landmarks for subset '{subset_name}' with {np.sum(subset_mask):,} cells")
                    
                    # Run prediction for this subset using the fitted model
                    if use_subset_landmarks:
                        # Run with new landmarks computed from the subset
                        subset_results = diff_expression.predict(
                            X_subset,
                            compute_mahalanobis=compute_mahalanobis,
                            progress=progress,
                            landmarks_override=subset_landmarks
                        )
                    else:
                        # Run without using landmarks
                        subset_results = diff_expression.predict(
                            X_subset,
                            compute_mahalanobis=compute_mahalanobis,
                            progress=progress,
                            use_landmarks=False
                        )
                    
                    # Save the subset results for storage
                    group_results[subset_name] = subset_results
                    
                    # Generate subset-specific field names
                    subset_field_suffix = f"_{subset_name}"
                    
                    # Only store specific metrics for subsets
                    for metric_name in ["mean_log_fold_change", "mahalanobis_distances"]:
                        if metric_name in subset_results:
                            # Create field name with subset suffix
                            if metric_name == "mean_log_fold_change":
                                base_key = field_names["mean_lfc_key"]
                                subset_key = f"{base_key}_{subset_name}"
                                
                                # Extract values
                                subset_values = subset_results[metric_name]
                                
                                # Add to adata.varm - DataFrame already initialized with all columns
                                # Use standardized key from field_names
                                varm_key = field_names["mean_lfc_varm_key"]
                                
                                # Create a Series with proper index covering all genes
                                full_series = pd.Series(np.nan, index=adata.var_names)
                                # Assign values only to selected genes - convert to numpy array first to avoid pandas issues
                                values_array = np.array(subset_values)
                                full_series.loc[selected_genes] = values_array
                                # Assign the whole column at once
                                adata.varm[varm_key][subset_name] = full_series
                                
                            elif metric_name == "mahalanobis_distances" and compute_mahalanobis:
                                base_key = field_names["mahalanobis_key"]
                                subset_key = f"{base_key}_{subset_name}"
                                
                                # Extract values
                                subset_values = subset_results[metric_name]
                                
                                # Add to adata.varm - DataFrame already initialized with all columns
                                # Use standardized key from field_names (already includes sample suffix if needed)
                                varm_key = field_names["mahalanobis_varm_key"]
                                
                                # Create a Series with proper index covering all genes
                                full_series = pd.Series(np.nan, index=adata.var_names)
                                # Assign values only to selected genes - convert to numpy array first to avoid pandas issues
                                values_array = np.array(subset_values)
                                full_series.loc[selected_genes] = values_array
                                # Assign the whole column at once
                                adata.varm[varm_key][subset_name] = full_series
                    
                    # Handle weighted mean log fold change if needed
                    if differential_abundance_key is not None and "fold_change" in subset_results:
                        # Get density values for the subset
                        cond1_safe = _sanitize_name(condition1)
                        cond2_safe = _sanitize_name(condition2)
                        
                        density_col1 = f"{differential_abundance_key}_log_density_{cond1_safe}"
                        density_col2 = f"{differential_abundance_key}_log_density_{cond2_safe}"
                        
                        if density_col1 in adata.obs and density_col2 in adata.obs:
                            # Filter density values to the subset
                            log_density_condition1 = adata.obs[density_col1][subset_mask]
                            log_density_condition2 = adata.obs[density_col2][subset_mask]
                            
                            # Calculate log density difference
                            log_density_diff = log_density_condition2 - log_density_condition1
                            
                            # Compute weighted mean fold change for the subset
                            weighted_lfc = compute_weighted_mean_fold_change(
                                subset_results['fold_change'],
                                log_density_diff=log_density_diff
                            )
                            
                            # Add to adata.varm - DataFrame already initialized with all columns
                            # Use standardized key from field_names
                            varm_key = field_names["weighted_lfc_varm_key"]
                            
                            # Create a Series with proper index covering all genes, initialize with NaN
                            full_series = pd.Series(np.nan, index=adata.var_names)
                            # Assign values only to selected genes
                            full_series[selected_genes] = weighted_lfc
                            # Assign the whole column at once
                            adata.varm[varm_key][subset_name] = full_series
                
                # No need to add columns to adata.var anymore as we're using varm exclusively
                logger.info(f"Group-specific data stored in adata.varm matrices")
                
                # Create entries in field_mapping for each varm matrix, not individual columns
                if field_names["mean_lfc_varm_key"] in adata.varm:
                    field_mapping[field_names["mean_lfc_varm_key"]] = {
                        "location": "varm",
                        "type": "mean_log_fold_change",
                        "description": "Mean log fold change values for all subsets",
                        "contains_subsets": subset_names
                    }
                
                # Only include mahalanobis_varm_key if compute_mahalanobis=True
                if compute_mahalanobis and field_names["mahalanobis_varm_key"] in adata.varm:
                    field_mapping[field_names["mahalanobis_varm_key"]] = {
                        "location": "varm",
                        "type": "mahalanobis",
                        "description": "Mahalanobis distances for all subsets",
                        "contains_subsets": subset_names
                    }
                
                if differential_abundance_key is not None and field_names["weighted_lfc_varm_key"] in adata.varm:
                    field_mapping[field_names["weighted_lfc_varm_key"]] = {
                        "location": "varm",
                        "type": "weighted_mean_log_fold_change",
                        "description": "Weighted mean log fold change values for all subsets",
                        "contains_subsets": subset_names
                    }
        
        # Add this mapping to run info
        current_run_info["field_mapping"] = field_mapping
        
        # Also store this version right away to make sure field_mapping is saved
        # Import JSON serialization utilities for early storage
        from .utils import set_json_metadata
        
        # Constant storage key makes lookups easier and does not result in conflicts here
        storage_key = "kompot_de"
        
        # Make an early update to last_run_info to ensure field_mapping is saved
        set_json_metadata(adata, f"{storage_key}.last_run_info", current_run_info.copy())
        
        # Initialize subset_names as empty list if not defined (no groups case)
        if 'subset_names' not in locals():
            subset_names = []
            
        # Store subset and varm info if groups were provided
        if groups is not None and subset_names:
            # Store the full list of subset names
            current_run_info["subset_names"] = subset_names
            current_run_info["has_groups"] = True
            
            # Create a more detailed groups summary for reporting
            groups_summary = {
                "count": len(subset_names),
                "names": subset_names,
                "description": str(type(groups).__name__),  # Type of groups specification
                "cells_per_group": {}
            }
            
            # Add cell counts for each group
            for group_name, mask in subset_masks.items():
                cell_count = np.sum(mask)
                percentage = (cell_count / adata.n_obs) * 100
                # Record cell counts and percentages
                groups_summary["cells_per_group"][group_name] = {
                    "count": int(cell_count),
                    "percentage": float(percentage)
                }
                
                # Also count cells from each condition within this group
                if groupby in adata.obs:
                    condition_counts = {}
                    for condition in [condition1, condition2]:
                        condition_mask = (adata.obs[groupby] == condition).values
                        condition_count = np.sum(mask & condition_mask)
                        condition_percentage = (condition_count / cell_count) * 100 if cell_count > 0 else 0
                        condition_counts[condition] = {
                            "count": int(condition_count),
                            "percentage": float(condition_percentage)
                        }
                    groups_summary["cells_per_group"][group_name]["conditions"] = condition_counts
            
            # Add the groups summary to run info
            current_run_info["groups_summary"] = groups_summary
            
            # Also store the varm keys used for group-specific metrics
            current_run_info["varm_keys"] = {
                "mean_lfc": field_names["mean_lfc_varm_key"],
                "mahalanobis": field_names["mahalanobis_varm_key"],
                "weighted_lfc": field_names["weighted_lfc_varm_key"] if differential_abundance_key is not None else None
            }

        # Import JSON serialization utilities
        from .utils import append_to_run_history, set_json_metadata
        
        # Constant storage key makes lookups easier and does not result in conflicts here
        storage_key = "kompot_de"
        
        # Get the run ID by checking the length of run history
        if "run_history" not in adata.uns.get(storage_key, {}):
            # Initialize if needed
            if storage_key not in adata.uns:
                adata.uns[storage_key] = {}
            # Initialize with an empty JSON array
            adata.uns[storage_key]["run_history"] = "[]"
            new_run_id = 0
        else:
            # Get the length from the existing history
            from .utils import get_run_history
            current_history = get_run_history(adata, "de")
            new_run_id = len(current_history)
        
        logger.info(f"This run will have `run_id={new_run_id}`.")
        
        # Always append current run to the run history using the json encoding
        append_to_run_history(adata, current_run_info, "de")
        
        # Store current params and run info as JSON
        set_json_metadata(adata, f"{storage_key}.last_run_info", current_run_info)
        
        # Also track and update all AnnData keys that are being written to
        anndata_field_tracking = {}
        
        # Use all_patterns to track fields in each location
        for location, patterns in all_patterns.items():
            # Create a dictionary for each location
            if location not in anndata_field_tracking:
                anndata_field_tracking[location] = {}
            
            # For each pattern, store the current run_id
            for pattern in patterns:
                anndata_field_tracking[location][pattern] = new_run_id
        
        # Also track the result_key itself in uns
        if "uns" not in anndata_field_tracking:
            anndata_field_tracking["uns"] = {}
        anndata_field_tracking["uns"][result_key] = new_run_id
        
        # Track varm matrices used for group-specific results
        if groups is not None and "varm" in all_patterns:
            if "varm" not in anndata_field_tracking:
                anndata_field_tracking["varm"] = {}
            
            # Track the mean_lfc_varm_key if it exists
            if field_names["mean_lfc_varm_key"] in adata.varm:
                anndata_field_tracking["varm"][field_names["mean_lfc_varm_key"]] = new_run_id
            
            # Track mahalanobis_varm_key only if compute_mahalanobis=True
            if compute_mahalanobis and field_names["mahalanobis_varm_key"] in adata.varm:
                anndata_field_tracking["varm"][field_names["mahalanobis_varm_key"]] = new_run_id
            
            # Track weighted_lfc_varm_key only if differential_abundance_key is provided
            if differential_abundance_key is not None and field_names["weighted_lfc_varm_key"] in adata.varm:
                anndata_field_tracking["varm"][field_names["weighted_lfc_varm_key"]] = new_run_id
        
        # Add or update tracking information in adata.uns[storage_key]
        if "anndata_fields" not in adata.uns[storage_key]:
            # Store as JSON string
            set_json_metadata(adata, f"{storage_key}.anndata_fields", anndata_field_tracking)
        else:
            # Get existing tracking data, update it, and store back as JSON
            from .utils import get_json_metadata
            existing_tracking = get_json_metadata(adata, f"{storage_key}.anndata_fields")
            if existing_tracking is None:
                existing_tracking = {}
                
            # Update existing tracking dictionary
            for section, fields in anndata_field_tracking.items():
                if section not in existing_tracking:
                    existing_tracking[section] = {}
                
                for field, run_id in fields.items():
                    existing_tracking[section][field] = run_id
                    
            # Store back as JSON string
            set_json_metadata(adata, f"{storage_key}.anndata_fields", existing_tracking)
        
    if copy:
        if return_full_results:
            return result_dict, adata
        else:
            return adata
    if return_full_results:
        return result_dict
    return None