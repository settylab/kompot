"""
Differential abundance analysis for AnnData objects.
"""

import logging
import numpy as np
import pandas as pd
import datetime
from typing import Optional, Union, Dict, Any, List, Tuple

from ..differential import DifferentialAbundance
from ..utils import KOMPOT_COLORS
from .utils import (
    _sanitize_name,
    generate_output_field_names,
    parse_groups,
    detect_output_field_overwrite,
    get_environment_info,
    get_json_metadata,
    set_json_metadata,
    get_run_history,
    append_to_run_history,
)

logger = logging.getLogger("kompot")


def compute_differential_abundance(
    adata,
    groupby: str,
    condition1: str,
    condition2: str,
    obsm_key: str = "DM_EigenVectors",
    n_landmarks: Optional[int] = None,
    landmarks: Optional[np.ndarray] = None,
    sample_col: Optional[str] = None,
    log_fold_change_threshold: float = 1.0,
    ptp_threshold: float = 0.05,
    ls_factor: float = 10.0,
    jit_compile: bool = False,
    random_state: Optional[int] = None,
    copy: bool = False,
    inplace: bool = True,
    result_key: str = "kompot_da",
    batch_size: Optional[int] = None,
    overwrite: Optional[bool] = None,
    store_landmarks: bool = False,
    return_full_results: bool = False,
    allow_single_condition_variance: bool = False,
    **density_kwargs
) -> Union[Dict[str, np.ndarray], "AnnData", Tuple[Dict[str, np.ndarray], "AnnData"]]:
    """
    Compute differential abundance between two conditions directly from an AnnData object.
    
    This function is a scverse-compatible wrapper around the DifferentialAbundance class
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
    n_landmarks : int, optional
        Number of landmarks to use for approximation. If None, use all points,
        by default None. Ignored if landmarks is provided.
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
    log_fold_change_threshold : float, optional
        Threshold for considering a log fold change significant, by default 1.7.
    ptp_threshold : float, optional
        Threshold for considering a PTP (Posterior Tail Probability) significant, by default 1e-3. 
        The posterior tail probability is a significance measure similar to p-values.
    ls_factor : float, optional
        Multiplication factor to apply to length scale when it's automatically inferred,
        by default 10.0. Only used when length scale is not explicitly provided.
    jit_compile : bool, optional
        Whether to use JAX just-in-time compilation, by default False.
    random_state : int, optional
        Random seed for reproducible landmark selection when n_landmarks is specified.
        Controls the random selection of points when using approximation, by default None.
    copy : bool, optional
        If True, return a copy of the AnnData object with results added,
        by default False.
    inplace : bool, optional
        If True, modify adata in place, by default True.
    result_key : str, optional
        Key in adata.uns where results will be stored, by default "kompot_da".
    batch_size : int, optional
        Number of samples to process at once during density estimation to manage memory usage.
        If None or 0, all samples will be processed at once. If processing all at once
        causes a memory error, a default batch size of 500 will be used automatically.
        Default is None.
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
    **density_kwargs : dict
        Additional arguments to pass to the DensityEstimator.
        
    Returns
    -------
    Union[Dict[str, np.ndarray], AnnData, Tuple[Dict[str, np.ndarray], AnnData]]
        Return value depends on ``copy`` and ``return_full_results`` parameters:

        - If ``copy=True`` and ``return_full_results=False``: Returns modified AnnData object
        - If ``copy=True`` and ``return_full_results=True``: Returns tuple ``(results_dict, adata)``
        - If ``copy=False`` and ``return_full_results=False``: Returns ``None`` (modifies in place)
        - If ``copy=False`` and ``return_full_results=True``: Returns ``results_dict``

        The ``results_dict`` contains the following keys for programmatic access:

        - ``"log_fold_change"``: Cell-wise log fold change values (n_cells,)
        - ``"log_fold_change_zscore"``: Z-scores of log fold changes (n_cells,)
        - ``"neg_log10_fold_change_ptp"``: Negative log10 posterior tail probabilities (n_cells,)
        - ``"log_fold_change_direction"``: Direction labels ('up', 'neutral', 'down') (n_cells,)
        - ``"log_density_condition1"``: Log density estimates for condition 1 (n_cells,)
        - ``"log_density_condition2"``: Log density estimates for condition 2 (n_cells,)
        - ``"model"``: The fitted ``DifferentialAbundance`` object for additional analyses
        - ``"landmarks"``: Computed landmarks array if applicable (n_landmarks, n_features)
        - ``"field_names"``: Dictionary mapping result types to their AnnData field names

        The ``model`` object provides access to the complete density estimation model, enabling
        additional downstream analyses such as computing gradients, accessing density estimators,
        or performing custom predictions on new data.
    
    Notes
    -----
    Results are stored in various components of the AnnData object:

    - adata.obs[f"{result_key}_{condition1}_to_{condition2}_lfc"]: Log fold change values for each cell
    - adata.obs[f"{result_key}_{condition1}_to_{condition2}_lfc_zscore"]: Z-scores for each cell
    - adata.obs[f"{result_key}_{condition1}_to_{condition2}_neg_log10_lfc_ptp"]: Negative log10 PTPs (Posterior Tail Probabilities) for each cell
    - adata.obs[f"{result_key}_{condition1}_to_{condition2}_lfc_direction"]: Direction of change ('up', 'neutral', 'down')
    - adata.uns[f"{result_key}_lfc_direction_colors"]: Color mapping for direction categories
    - adata.uns[result_key]: Dictionary with additional information and parameters
    - If landmarks are computed, they are stored in adata.uns[result_key]['landmarks']
      for potential reuse in other analyses.
      
    The color scheme used for directions is:
    - "up": "#d73027" (red)
    - "down": "#4575b4" (blue)
    - "neutral": "#d3d3d3" (light gray)
    """
    try:
        import anndata
    except ImportError:
        raise ImportError(
            "Please install anndata: pip install anndata"
        )
    
    # Make a copy if requested
    if copy:
        adata = adata.copy()
        
    # Check for existing results using the utility functions
    
    # Generate standardized field names
    field_names = generate_output_field_names(
        result_key=result_key,
        condition1=condition1,
        condition2=condition2,
        analysis_type="da",
        with_sample_suffix=(sample_col is not None),
        sample_suffix="_sample_var" if sample_col is not None else ""
    )
    
    # Get all patterns from field_names
    all_patterns = field_names["all_patterns"]
    
    # Get the patterns for observation columns
    column_patterns = all_patterns["obs"]
    
    # Detect if we'd overwrite any existing output fields
    has_overwrites, existing_fields, prev_run = detect_output_field_overwrite(
        adata=adata,
        result_key=result_key,
        output_patterns=column_patterns,
        with_sample_suffix=(sample_col is not None),
        sample_suffix="_sample_var" if sample_col is not None else "",
        result_type="differential abundance",
        analysis_type="da"
    )
    
    # Handle overwrite detection results
    if has_overwrites:
        # Format the message about existing results
        message = f"Results with result_key='{result_key}' already exist in the dataset."
        
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
                
                if prev_sample_var != current_sample_var:
                    if current_sample_var:
                        message += (f" Fields that will be overwritten: {field_list}. "
                                   f"Note: Only fields NOT affected by sample variance (like lfc, log_density) "
                                   f"will be overwritten since they don't use the sample variance suffix. "
                                   f"These results will likely be identical if other parameters haven't changed.")
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
                    key_params = ['groupby', 'condition1', 'condition2', 'obsm_key', 'ls_factor']
                    
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
    
    # Create masks for each condition
    mask1 = adata.obs[groupby] == condition1
    mask2 = adata.obs[groupby] == condition2
    
    if np.sum(mask1) == 0:
        raise ValueError(f"Condition '{condition1}' not found in '{groupby}'.")
    if np.sum(mask2) == 0:
        raise ValueError(f"Condition '{condition2}' not found in '{groupby}'.")
    
    logger.info(f"Condition 1 ({condition1}): {np.sum(mask1):,} cells")
    logger.info(f"Condition 2 ({condition2}): {np.sum(mask2):,} cells")
    
    # Extract cell states for each condition
    X_condition1 = adata.obsm[obsm_key][mask1]
    X_condition2 = adata.obsm[obsm_key][mask2]
    
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
    
    # If still no landmarks, check for any other landmarks in standard storage_key
    if landmarks is None:
        storage_key = "kompot_da"
        
        if storage_key in adata.uns and storage_key != result_key and 'landmarks' in adata.uns[storage_key]:
            other_landmarks = adata.uns[storage_key]['landmarks']
            landmarks_dim = other_landmarks.shape[1]
            data_dim = adata.obsm[obsm_key].shape[1]
            
            # Only use the stored landmarks if dimensions match
            if landmarks_dim == data_dim:
                logger.info(f"Reusing stored landmarks from adata.uns['{storage_key}']['landmarks'] with shape {other_landmarks.shape}")
                landmarks = other_landmarks
            else:
                logger.warning(f"Other stored landmarks have dimension {landmarks_dim} but data has dimension {data_dim}. Will check for other landmarks.")
    
    # If still no landmarks, check in all other kompot_* keys for valid landmarks            
    if landmarks is None:
        for key in adata.uns.keys():
            if key.startswith('kompot_') and key != result_key and key != storage_key and 'landmarks' in adata.uns[key]:
                check_landmarks = adata.uns[key]['landmarks'] 
                landmarks_dim = check_landmarks.shape[1]
                data_dim = adata.obsm[obsm_key].shape[1]
                
                # Only use the stored landmarks if dimensions match
                if landmarks_dim == data_dim:
                    logger.info(f"Reusing landmarks from adata.uns['{key}']['landmarks'] with shape {check_landmarks.shape}")
                    landmarks = check_landmarks
                    break
                else:
                    logger.debug(f"Found landmarks in {key} but dimensions don't match: {landmarks_dim} vs {data_dim}")
                
    # If still no landmarks, specifically check for DE landmarks for backward compatibility
    if landmarks is None and "kompot_de" in adata.uns and 'landmarks' in adata.uns["kompot_de"]:
        de_landmarks = adata.uns["kompot_de"]['landmarks']
        landmarks_dim = de_landmarks.shape[1]
        data_dim = adata.obsm[obsm_key].shape[1]
        
        # Only use the stored landmarks if dimensions match
        if landmarks_dim == data_dim:
            logger.info(f"Reusing differential expression landmarks from adata.uns['kompot_de']['landmarks'] with shape {de_landmarks.shape}")
            landmarks = de_landmarks
        else:
            logger.warning(f"DE landmarks have dimension {landmarks_dim} but data has dimension {data_dim}. Computing new landmarks.")
    
    # Initialize and fit DifferentialAbundance
    diff_abundance = DifferentialAbundance(
        log_fold_change_threshold=log_fold_change_threshold,
        ptp_threshold=ptp_threshold,
        n_landmarks=n_landmarks,
        jit_compile=jit_compile,
        random_state=random_state,
        batch_size=batch_size
    )
    
    # Fit the estimators
    diff_abundance.fit(
        X_condition1, 
        X_condition2, 
        landmarks=landmarks, 
        ls_factor=ls_factor, 
        condition1_sample_indices=condition1_sample_indices,
        condition2_sample_indices=condition2_sample_indices,
        allow_single_condition_variance=allow_single_condition_variance,
        **density_kwargs
    )
    
    # Run prediction to compute fold changes and metrics
    X_for_prediction = adata.obsm[obsm_key]
    abundance_results = diff_abundance.predict(
        X_for_prediction,
        log_fold_change_threshold=log_fold_change_threshold,
        ptp_threshold=ptp_threshold
    )
    # Note: weighted_mean_lfc is no longer computed by default
    
    # Handle landmarks for future reference
    if hasattr(diff_abundance, 'computed_landmarks') and diff_abundance.computed_landmarks is not None:
        # Initialize if needed
        if result_key not in adata.uns:
            adata.uns[result_key] = {}
        
        # Get landmarks info
        landmarks_shape = diff_abundance.computed_landmarks.shape
        landmarks_dtype = str(diff_abundance.computed_landmarks.dtype)
        
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
        
        storage_key = "kompot_da"
        if storage_key not in adata.uns:
            adata.uns[storage_key] = {}
        adata.uns[storage_key]['landmarks_info'] = landmarks_info.copy()
        
        # Store the actual landmarks if requested
        if store_landmarks:
            adata.uns[result_key]['landmarks'] = diff_abundance.computed_landmarks
            
            # Store only in the result_key, not automatically in storage_key
            # We'll check across keys when searching for landmarks
            
            logger.info(f"Stored landmarks in adata.uns['{result_key}']['landmarks'] with shape {landmarks_shape} for future reuse")
        else:
            logger.info(f"Landmark storage skipped (store_landmarks=False). Compute with store_landmarks=True to enable landmark reuse.")
    else:
        logger.debug("No computed landmarks found to store. Check if landmarks were pre-computed or if n_landmarks is set correctly.")
    
    # Use the standardized field names already generated earlier
    # Assign values to masked cells with descriptive column names
    # For fields not impacted by sample variance, we don't add the suffix
    adata.obs[field_names["lfc_key"]] = abundance_results['log_fold_change']
    adata.obs[field_names["zscore_key"]] = abundance_results['log_fold_change_zscore']
    adata.obs[field_names["ptp_key"]] = abundance_results['neg_log10_fold_change_ptp']  # Now using negative log10 PTPs (Posterior Tail Probabilities) (higher = more significant)
    
    # Add the direction column
    direction_col = field_names["direction_key"]
    adata.obs[direction_col] = abundance_results['log_fold_change_direction']
    
    # Make sure the column is categorical with all three categories
    adata.obs[direction_col] = pd.Categorical(
        adata.obs[direction_col],
        categories=['up', 'neutral', 'down']
    )
    
    # Add standard color mapping for direction categories in adata.uns
    # Use colors from the central color definition
    direction_colors = KOMPOT_COLORS["direction"]
    
    # Get the categories in the order they appear in the categorical column
    categories = adata.obs[direction_col].cat.categories
    
    # Create the list of colors in the same order as the categories
    color_list = [direction_colors.get(cat, "#d3d3d3") for cat in categories]
    
    # Save colors to adata.uns with the _colors postfix
    adata.uns[f"{direction_col}_colors"] = color_list
    
    # Store log densities for each condition with descriptive names - these are not impacted by sample variance
    adata.obs[field_names["density_key_1"]] = abundance_results['log_density_condition1']
    adata.obs[field_names["density_key_2"]] = abundance_results['log_density_condition2']
    
    # Prepare parameters, run timestamp, and field metadata
    current_timestamp = datetime.datetime.now().isoformat()
    
    # Define parameters dict - include ALL parameters for reproducibility
    params_dict = {
        "groupby": groupby,
        "condition1": condition1,
        "condition2": condition2,
        "obsm_key": obsm_key,
        "n_landmarks": n_landmarks,
        "landmarks": landmarks is not None,  # Just store if landmarks were provided, not the actual values
        "sample_col": sample_col,
        "use_sample_variance": sample_col is not None,
        "log_fold_change_threshold": log_fold_change_threshold,
        "ptp_threshold": ptp_threshold, 
        "ls_factor": ls_factor,
        "jit_compile": jit_compile,
        "random_state": random_state,
        "used_landmarks": True if landmarks is not None else False,
        "batch_size": batch_size,
        "store_landmarks": store_landmarks,
        "result_key": result_key,
        "copy": copy,
        "inplace": inplace,
        "overwrite": overwrite
    }
    
    current_run_info = {
        "timestamp": current_timestamp,
        "function": "compute_differential_abundance",
        "result_key": result_key,
        "analysis_type": "da",
        "lfc_key": field_names["lfc_key"],
        "zscore_key": field_names["zscore_key"],
        "ptp_key": field_names["ptp_key"],
        "direction_key": field_names["direction_key"],
        "density_keys": {
            "condition1": field_names["density_key_1"],
            "condition2": field_names["density_key_2"]
        },
        "field_names": field_names,
        "uses_sample_variance": sample_col is not None,
        # Store explicit groups flag for future group support
        "has_groups": False,  # Currently DA doesn't support groups, but prepared for future compatibility
        "params": params_dict
    }
    
    storage_key = "kompot_da"
    
    # Initialize or update adata.uns[storage_key]
    if storage_key not in adata.uns:
        adata.uns[storage_key] = {}
    
    # Add environment info to the run info
    env_info = get_environment_info()
    current_run_info["environment"] = env_info
    
    # Import JSON serialization utilities
    from .utils import append_to_run_history, set_json_metadata
    
    # Get the run ID by checking the length of run history
    storage_key = "kompot_da"
    
    if "run_history" not in adata.uns.get(storage_key, {}):
        # Initialize if needed
        if storage_key not in adata.uns:
            adata.uns[storage_key] = {}
        # Initialize with an empty JSON array
        adata.uns[storage_key]["run_history"] = "[]"
        new_run_id = 0
    else:
        # Get the length from the existing history
        current_history = get_run_history(adata, "da")
        new_run_id = len(current_history)
        
    logger.info(f"This run will have `run_id={new_run_id}`.")
    
    # Always append current run to the run history using the json encoding
    append_to_run_history(adata, current_run_info, "da")
    
    # Create a comprehensive field-to-location mapping for field tracking
    # This maps the full field names to their locations
    field_mapping = {
        # Obs fields
        field_names["lfc_key"]: {"location": "obs", "type": "log_fold_change", "description": "Log fold change values"},
        field_names["zscore_key"]: {"location": "obs", "type": "log_fold_change_zscore", "description": "Z-scores for fold changes"},
        field_names["ptp_key"]: {"location": "obs", "type": "ptp", "description": "Negative log10 PTPs (Posterior Tail Probabilities)"},
        field_names["direction_key"]: {"location": "obs", "type": "direction", "description": "Direction classification (up/down/neutral)"},
        field_names["density_key_1"]: {"location": "obs", "type": "density", "description": f"Log density for {condition1}"},
        field_names["density_key_2"]: {"location": "obs", "type": "density", "description": f"Log density for {condition2}"},
        
        # Uns fields
        f"{field_names['direction_key']}_colors": {"location": "uns", "type": "colors", "description": "Color mapping for direction categories"}
    }

    # Add this mapping to run info
    current_run_info["field_mapping"] = field_mapping
    
    # Store current params and run info as JSON
    set_json_metadata(adata, f"{storage_key}.last_run_info", current_run_info)

    # Also track and update all AnnData keys that are being written to
    anndata_field_tracking = {
        "obs": {},
        "uns": {}
    }
    
    # Track obs fields
    obs_fields = [
        field_names["lfc_key"],
        field_names["zscore_key"],
        field_names["ptp_key"],
        field_names["direction_key"],
        field_names["density_key_1"],
        field_names["density_key_2"]
    ]
    
    # Add each field to tracking with current run_id
    for field in obs_fields:
        anndata_field_tracking["obs"][field] = new_run_id
    
    # Also track the result_key itself in uns
    anndata_field_tracking["uns"][result_key] = new_run_id
    # Also track the direction colors key
    direction_colors_key = f"{field_names['direction_key']}_colors"
    anndata_field_tracking["uns"][direction_colors_key] = new_run_id
    
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
    
    # Return results as a dictionary with fixed keys for programmatic access
    result_dict = {
        "log_fold_change": abundance_results['log_fold_change'],
        "log_fold_change_zscore": abundance_results['log_fold_change_zscore'],
        "neg_log10_fold_change_ptp": abundance_results['neg_log10_fold_change_ptp'],  # Now using negative log10 PTPs (Posterior Tail Probabilities)
        "log_fold_change_direction": abundance_results['log_fold_change_direction'],
        "log_density_condition1": abundance_results['log_density_condition1'],
        "log_density_condition2": abundance_results['log_density_condition2'],
        "model": diff_abundance,
    }

    # Add landmarks to result dictionary if they were computed
    if hasattr(diff_abundance, 'computed_landmarks') and diff_abundance.computed_landmarks is not None:
        result_dict["landmarks"] = diff_abundance.computed_landmarks

    # Add field_names dictionary for programmatic access to AnnData field names
    result_dict["field_names"] = field_names
    
    if copy:
        if return_full_results:
            return result_dict, adata
        else:
            return adata
    if return_full_results:
        return result_dict
    return None