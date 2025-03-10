"""
AnnData integration functions for Kompot.
"""

import logging
import numpy as np
import pandas as pd
import datetime
from typing import Optional, Union, Dict, Any, List, Tuple

from ..differential import DifferentialAbundance, DifferentialExpression, compute_weighted_mean_fold_change
from ..reporter import HTMLReporter

logger = logging.getLogger("kompot")


def _sanitize_name(name):
    """Convert a string to a valid column/key name by replacing invalid characters."""
    # Replace spaces, slashes, and other common problematic characters
    return str(name).replace(' ', '_').replace('/', '_').replace('\\', '_').replace('-', '_').replace('.', '_')


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
    pvalue_threshold: float = 1e-3,
    ls_factor: float = 10.0,
    jit_compile: bool = False,
    random_state: Optional[int] = None,
    copy: bool = False,
    inplace: bool = True,
    result_key: str = "kompot_da",
    batch_size: Optional[int] = None,
    overwrite: Optional[bool] = None,
    **density_kwargs
) -> Union[Dict[str, np.ndarray], "AnnData"]:
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
    log_fold_change_threshold : float, optional
        Threshold for considering a log fold change significant, by default 1.7.
    pvalue_threshold : float, optional
        Threshold for considering a p-value significant, by default 1e-3.
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
        - If None (default): Warn about existing results but proceed with overwriting
        - If True: Silently overwrite existing results
        - If False: Raise an error if results would be overwritten
    **density_kwargs : dict
        Additional arguments to pass to the DensityEstimator.
        
    Returns
    -------
    Union[Dict[str, np.ndarray], AnnData]
        If copy is True, returns a copy of the AnnData object with results added.
        If inplace is True, returns None (adata is modified in place).
        Otherwise, returns a dictionary of results.
    
    Notes
    -----
    Results are stored in various components of the AnnData object:
    
    - adata.obs[f"{result_key}_log_fold_change"]: Log fold change values for each cell
    - adata.obs[f"{result_key}_log_fold_change_zscore"]: Z-scores for each cell
    - adata.obs[f"{result_key}_neg_log10_fold_change_pvalue"]: Negative log10 p-values for each cell
    - adata.obs[f"{result_key}_log_fold_change_direction"]: Direction of change ('up', 'down', 'neutral')
    - adata.uns[f"{result_key}_log_fold_change_direction_colors"]: Color mapping for direction categories
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
    from ..utils import detect_output_field_overwrite, generate_output_field_names
    
    # Generate standardized field names
    field_names = generate_output_field_names(
        result_key=result_key,
        condition1=condition1,
        condition2=condition2,
        analysis_type="da",
        with_sample_suffix=(sample_col is not None),
        sample_suffix="_sample_var" if sample_col is not None else ""
    )
    
    # Define patterns to check for overwrites using standardized field names
    # Include both sample-variance-impacted and non-impacted fields
    column_patterns = [
        field_names["lfc_key"],             # Not impacted by sample variance
        field_names["density_key_1"],       # Not impacted by sample variance
        field_names["zscore_key"],          # Impacted by sample variance
        field_names["direction_key"]        # Impacted by sample variance
    ]
    
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
            prev_conditions = f"{prev_params.get('condition1', 'unknown')} vs {prev_params.get('condition2', 'unknown')}"
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
                                   f"Note: Only fields NOT affected by sample variance (like log_fold_change, log_density) "
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
            logger.warning(message + " Results will be overwritten.")
    
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
    
    # Check if we have landmarks in uns for this key and can reuse them
    stored_landmarks = None
    if landmarks is None and result_key in adata.uns and 'landmarks' in adata.uns[result_key]:
        stored_landmarks = adata.uns[result_key]['landmarks']
        landmarks_dim = stored_landmarks.shape[1]
        data_dim = adata.obsm[obsm_key].shape[1]
        
        # Only use the stored landmarks if dimensions match
        if landmarks_dim == data_dim:
            logger.info(f"Using stored landmarks from adata.uns['{result_key}']['landmarks'] with shape {stored_landmarks.shape}")
            landmarks = stored_landmarks
        else:
            logger.warning(f"Stored landmarks have dimension {landmarks_dim} but data has dimension {data_dim}. Computing new landmarks.")
            landmarks = None
    
    # Initialize and fit DifferentialAbundance
    diff_abundance = DifferentialAbundance(
        log_fold_change_threshold=log_fold_change_threshold,
        pvalue_threshold=pvalue_threshold,
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
        **density_kwargs
    )
    
    # Run prediction to compute fold changes and metrics
    X_for_prediction = adata.obsm[obsm_key]
    abundance_results = diff_abundance.predict(
        X_for_prediction,
        log_fold_change_threshold=log_fold_change_threshold,
        pvalue_threshold=pvalue_threshold
    )
    # Note: mean_log_fold_change is no longer computed by default
    
    # Store landmark metadata for future reference (not the actual array)
    if hasattr(diff_abundance, 'computed_landmarks') and diff_abundance.computed_landmarks is not None:
        # Initialize if needed
        if result_key not in adata.uns:
            adata.uns[result_key] = {}
        
        # Store metadata about landmarks, not the actual array (which is large)
        landmarks_shape = diff_abundance.computed_landmarks.shape
        landmarks_dtype = str(diff_abundance.computed_landmarks.dtype)
        
        adata.uns[result_key]['landmarks_info'] = {
            'shape': landmarks_shape,
            'dtype': landmarks_dtype,
            'source': 'computed',
            'n_landmarks': landmarks_shape[0]
        }
        # The actual landmarks are accessible through the model in the result
        logger.info(f"Stored landmarks metadata in adata.uns['{result_key}']['landmarks_info']")
    
    # Use the standardized field names already generated earlier
    # Assign values to masked cells with descriptive column names
    # For fields not impacted by sample variance, we don't add the suffix
    adata.obs[field_names["lfc_key"]] = abundance_results['log_fold_change']
    adata.obs[field_names["zscore_key"]] = abundance_results['log_fold_change_zscore']
    adata.obs[field_names["pval_key"]] = abundance_results['neg_log10_fold_change_pvalue']  # Now using negative log10 p-values (higher = more significant)
    
    # Add the direction column
    direction_col = field_names["direction_key"]
    adata.obs[direction_col] = abundance_results['log_fold_change_direction']
    
    # Add standard color mapping for direction categories in adata.uns
    # Use colors from the central color definition
    from ..utils import KOMPOT_COLORS
    direction_colors = KOMPOT_COLORS["direction"]
    
    # Get the unique categories in the order they appear in the categorical column
    if hasattr(adata.obs[direction_col], 'cat'):
        # If it's already a categorical type
        categories = adata.obs[direction_col].cat.categories
    else:
        # Convert to categorical to get ordered categories
        adata.obs[direction_col] = adata.obs[direction_col].astype('category')
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
    
    # Define parameters dict
    params_dict = {
        "groupby": groupby,
        "condition1": condition1,
        "condition2": condition2,
        "obsm_key": obsm_key,
        "log_fold_change_threshold": log_fold_change_threshold,
        "pvalue_threshold": pvalue_threshold,
        "n_landmarks": n_landmarks,
        "ls_factor": ls_factor,
        "used_landmarks": True if landmarks is not None else False,
        "sample_col": sample_col,
        "use_sample_variance": sample_col is not None,
    }
    
    current_run_info = {
        "timestamp": current_timestamp,
        "function": "compute_differential_abundance",
        "result_key": result_key,
        "analysis_type": "da",
        "lfc_key": field_names["lfc_key"],
        "zscore_key": field_names["zscore_key"],
        "pval_key": field_names["pval_key"],
        "direction_key": field_names["direction_key"],
        "density_keys": {
            "condition1": field_names["density_key_1"],
            "condition2": field_names["density_key_2"]
        },
        "field_names": field_names,
        "uses_sample_variance": sample_col is not None,
        "params": params_dict
    }
    
    # Also maintain separate params for backward compatibility
    current_params = params_dict.copy()
    
    storage_key = "kompot_da"
    
    # Initialize or update adata.uns[storage_key]
    if storage_key not in adata.uns:
        adata.uns[storage_key] = {}
    
    # Add environment info to the run info
    from ..utils import get_environment_info
    env_info = get_environment_info()
    current_run_info["environment"] = env_info
    
    # Initialize run history if it doesn't exist
    if "run_history" not in adata.uns[storage_key]:
        adata.uns[storage_key]["run_history"] = []
    
    # Always append current run to the run history
    adata.uns[storage_key]["run_history"].append(current_run_info)
    
    # Store current params and run info
    adata.uns[storage_key]["params"] = current_params
    adata.uns[storage_key]["run_info"] = current_run_info
    
    # Return results as a dictionary
    return {
        "log_fold_change": abundance_results['log_fold_change'],
        "log_fold_change_zscore": abundance_results['log_fold_change_zscore'],
        "neg_log10_fold_change_pvalue": abundance_results['neg_log10_fold_change_pvalue'],  # Now using negative log10 p-values
        "log_fold_change_direction": abundance_results['log_fold_change_direction'],
        "log_density_condition1": abundance_results['log_density_condition1'],
        "log_density_condition2": abundance_results['log_density_condition2'],
        "model": diff_abundance,
    }


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
    random_state: Optional[int] = None,
    batch_size: int = 100,
    copy: bool = False,
    inplace: bool = True,
    result_key: str = "kompot_de",
    overwrite: Optional[bool] = None,
    **function_kwargs
) -> Union[Dict[str, np.ndarray], "AnnData"]:
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
    random_state : int, optional
        Random seed for reproducible landmark selection when n_landmarks is specified.
        Controls the random selection of points when using approximation, by default None.
    batch_size : int, optional
        Number of genes to process in each batch during Mahalanobis distance computation.
        Smaller values use less memory but are slower, by default 100. For large datasets
        with memory constraints, try a smaller value like 20-50. 
        This parameter is also passed to the DifferentialAbundance class when compute_abundance=True.
    copy : bool, optional
        If True, return a copy of the AnnData object with results added,
        by default False.
    inplace : bool, optional
        If True, modify adata in place, by default True.
    result_key : str, optional
        Key in adata.uns where results will be stored, by default "kompot_de".
    overwrite : bool, optional
        Controls behavior when results with the same result_key already exist:
        - If None (default): Warn about existing results but proceed with overwriting
        - If True: Silently overwrite existing results
        - If False: Raise an error if results would be overwritten
    **function_kwargs : dict
        Additional arguments to pass to the FunctionEstimator.
        
    Returns
    -------
    Union[Dict[str, np.ndarray], AnnData]
        If copy is True, returns a copy of the AnnData object with results added.
        If inplace is True, returns None (adata is modified in place).
        Otherwise, returns a dictionary of results.
    
    Notes
    -----
    Results are stored in various components of the AnnData object:
    
    - adata.var[f"{result_key}_mahalanobis"]: Mahalanobis distance for each gene
    - adata.var[f"{result_key}_weighted_lfc"]: Weighted mean log fold change for each gene
    - adata.var[f"{result_key}_lfc_std"]: Standard deviation of log fold change for each gene
    - adata.var[f"{result_key}_bidirectionality"]: Bidirectionality score for each gene
    - adata.layers[f"{result_key}_condition1_imputed"]: Imputed expression for condition 1
    - adata.layers[f"{result_key}_condition2_imputed"]: Imputed expression for condition 2
    - adata.layers[f"{result_key}_fold_change"]: Log fold change for each cell and gene
    - adata.uns[result_key]: Dictionary with additional information and parameters
    - If landmarks are computed, they are stored in adata.uns[result_key]['landmarks']
      for potential reuse in other analyses.
    """
    try:
        import anndata
        from scipy import sparse
    except ImportError:
        raise ImportError(
            "Please install anndata and scipy: pip install anndata scipy"
        )
    
    # Check for existing results using the utility functions
    from ..utils import detect_output_field_overwrite, generate_output_field_names
    
    # Generate standardized field names
    field_names = generate_output_field_names(
        result_key=result_key,
        condition1=condition1,
        condition2=condition2,
        analysis_type="de",
        with_sample_suffix=(sample_col is not None),
        sample_suffix="_sample_var" if sample_col is not None else ""
    )
    
    # Define patterns to check for overwrites in var columns using standardized field names
    var_column_patterns = [
        field_names["mahalanobis_key"],      # Impacted by sample variance
        field_names["mean_lfc_key"],         # Not impacted by sample variance
        field_names["bidirectionality_key"], # Not impacted by sample variance
        field_names["lfc_std_key"]           # Impacted by sample variance
    ]
    
    # Detect if we'd overwrite any existing var columns
    has_var_overwrites, var_existing_fields, var_prev_run = detect_output_field_overwrite(
        adata=adata,
        result_key=result_key,
        output_patterns=var_column_patterns,
        location="var",
        with_sample_suffix=(sample_col is not None),
        sample_suffix="_sample_var" if sample_col is not None else "",
        result_type="differential expression (var columns)",
        analysis_type="de"
    )
    
    # Define patterns to check for overwrites in layers using standardized field names
    layer_patterns = [
        field_names["imputed_key_1"],        # Not impacted by sample variance
        field_names["fold_change_key"]       # Not impacted by sample variance
    ]
    
    # Detect if we'd overwrite any existing layers
    has_layer_overwrites, layer_existing_fields, layer_prev_run = detect_output_field_overwrite(
        adata=adata,
        result_key=result_key,
        output_patterns=layer_patterns,
        location="layers",
        with_sample_suffix=(sample_col is not None),
        sample_suffix="_sample_var" if sample_col is not None else "",
        result_type="differential expression (layers)",
        analysis_type="de"
    )
    
    # Combine results
    has_overwrites = has_var_overwrites or has_layer_overwrites
    existing_fields = var_existing_fields + layer_existing_fields
    prev_run = var_prev_run or layer_prev_run
    
    # Handle overwrite detection results
    if has_overwrites:
        # Format the message about existing results
        message = f"Differential expression results with result_key='{result_key}' already exist in the dataset."
        
        if prev_run:
            prev_timestamp = prev_run.get('timestamp', 'unknown time')
            prev_params = prev_run.get('params', {})
            prev_conditions = f"{prev_params.get('condition1', 'unknown')} vs {prev_params.get('condition2', 'unknown')}"
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
                                   f"Note: Only fields NOT affected by sample variance (like mean_log_fold_change, "
                                   f"bidirectionality, imputed data, fold_change) will be overwritten since they "
                                   f"don't use the sample variance suffix. These results will likely be identical "
                                   f"if other parameters haven't changed.")
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
            logger.warning(message + " Results will be overwritten.")

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
        if not all(gene in adata.var_names for gene in genes):
            missing_genes = [gene for gene in genes if gene not in adata.var_names]
            raise ValueError(f"The following genes were not found in adata.var_names: {missing_genes[:10]}" +
                          (f"... and {len(missing_genes) - 10} more" if len(missing_genes) > 10 else ""))
        
        gene_indices = [list(adata.var_names).index(gene) for gene in genes]
        expr1 = expr1[:, gene_indices]
        expr2 = expr2[:, gene_indices]
        selected_genes = genes
    else:
        selected_genes = adata.var_names.tolist()
    
    # Check if we have landmarks in uns for this key and can reuse them
    stored_landmarks = None
    if landmarks is None and result_key in adata.uns and 'landmarks' in adata.uns[result_key]:
        stored_landmarks = adata.uns[result_key]['landmarks']
        landmarks_dim = stored_landmarks.shape[1]
        data_dim = adata.obsm[obsm_key].shape[1]
        
        # Only use the stored landmarks if dimensions match
        if landmarks_dim == data_dim:
            logger.info(f"Using stored landmarks from adata.uns['{result_key}']['landmarks'] with shape {stored_landmarks.shape}")
            landmarks = stored_landmarks
        else:
            logger.warning(f"Stored landmarks have dimension {landmarks_dim} but data has dimension {data_dim}. Computing new landmarks.")
            landmarks = None
    
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
            logger.warning(f"Abundance landmarks have dimension {landmarks_dim} but data has dimension {data_dim}. Computing new landmarks.")
            landmarks = None
    
    # Initialize and fit DifferentialExpression
    use_sample_variance = sample_col is not None
    
    diff_expression = DifferentialExpression(
        n_landmarks=n_landmarks,
        use_sample_variance=use_sample_variance,
        jit_compile=jit_compile,
        random_state=random_state,
        batch_size=batch_size
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
        **function_kwargs
    )
    
    # Store landmark metadata for future reference (not the actual array)
    if hasattr(diff_expression, 'computed_landmarks') and diff_expression.computed_landmarks is not None:
        # Initialize if needed
        if result_key not in adata.uns:
            adata.uns[result_key] = {}
        
        # Store metadata about landmarks, not the actual array (which is large)
        landmarks_shape = diff_expression.computed_landmarks.shape
        landmarks_dtype = str(diff_expression.computed_landmarks.dtype)
        
        adata.uns[result_key]['landmarks_info'] = {
            'shape': landmarks_shape,
            'dtype': landmarks_dtype,
            'source': 'computed',
            'n_landmarks': landmarks_shape[0]
        }
        # The actual landmarks are accessible through the model in the result
        logger.info(f"Stored landmarks metadata in adata.uns['{result_key}']['landmarks_info']")
    
    # Run prediction to compute fold changes, metrics, and Mahalanobis distances
    X_for_prediction = adata.obsm[obsm_key]
    expression_results = diff_expression.predict(
        X_for_prediction, 
        compute_mahalanobis=compute_mahalanobis,
    )
    
    # Separately compute weighted fold changes if needed
    if differential_abundance_key is not None:
        # Sanitize condition names for use in column names
        cond1_safe = _sanitize_name(condition1)
        cond2_safe = _sanitize_name(condition2)
        
        # Get log densities from adata with descriptive names
        density_col1 = f"{differential_abundance_key}_log_density_{cond1_safe}"
        density_col2 = f"{differential_abundance_key}_log_density_{cond2_safe}"
        
        if density_col1 in adata.obs and density_col2 in adata.obs:
            log_density_condition1 = adata.obs[density_col1]
            log_density_condition2 = adata.obs[density_col2]
            
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
        "lfc_stds": expression_results['lfc_stds'],
        "bidirectionality": expression_results['bidirectionality'],
        "mean_log_fold_change": expression_results['mean_log_fold_change'],
        "condition1_imputed": expression_results['condition1_imputed'],
        "condition2_imputed": expression_results['condition2_imputed'],
        "fold_change": expression_results['fold_change'],
        "fold_change_zscores": expression_results['fold_change_zscores'],
        "model": diff_expression,
    }
    
    # Add optional result fields
    if compute_mahalanobis:
        result_dict["mahalanobis_distances"] = expression_results['mahalanobis_distances']
        
    if 'weighted_mean_log_fold_change' in expression_results:
        result_dict["weighted_mean_log_fold_change"] = expression_results['weighted_mean_log_fold_change']
    
    if inplace:
        # Sanitize condition names for use in column names first
        cond1_safe = _sanitize_name(condition1)
        cond2_safe = _sanitize_name(condition2)
        
        # Add suffix when sample variance is used
        sample_suffix = "_sample_var" if sample_col is not None else ""
        
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
            adata.var[mahalanobis_key] = pd.Series(np.nan, index=adata.var_names)
            adata.var.loc[selected_genes, mahalanobis_key] = mahalanobis_distances
        
        if differential_abundance_key is not None:
            # Initialize with np.nan of appropriate shape - use more descriptive column name
            # Use the standardized field name from field_names
            # Weighted mean log fold change is NOT impacted by sample variance
            column_name = field_names["weighted_lfc_key"]
            adata.var[column_name] = pd.Series(np.nan, index=adata.var_names)
            
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
            adata.var.loc[selected_genes, column_name] = weighted_lfc
        
        # Initialize with np.nan of appropriate shape - use more descriptive column names
        # Add mean log fold change with descriptive name
        # Use the standardized field name from field_names
        # Mean log fold change is NOT impacted by sample variance
        mean_lfc_column = field_names["mean_lfc_key"]
        adata.var[mean_lfc_column] = pd.Series(np.nan, index=adata.var_names)
        
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
        adata.var.loc[selected_genes, mean_lfc_column] = mean_lfc
        
        # Standard deviation of log fold change - this IS impacted by sample variance
        lfc_std_key = field_names["lfc_std_key"]
        adata.var[lfc_std_key] = pd.Series(np.nan, index=adata.var_names)
        
        # Extract and verify lfc_stds
        lfc_stds = expression_results['lfc_stds']
        # Convert list to numpy array if needed
        if isinstance(lfc_stds, list):
            lfc_stds = np.array(lfc_stds)
        
        # Ensure lfc_stds is 1D before reshaping
        if len(lfc_stds.shape) > 1:
            logger.warning(f"lfc_stds has shape {lfc_stds.shape}, flattening to 1D.")
            # Take the first row if it's a 2D array
            if lfc_stds.shape[0] < lfc_stds.shape[1]:
                lfc_stds = lfc_stds[0]  # Take first row if more columns than rows
            else:
                lfc_stds = lfc_stds[:, 0]  # Take first column otherwise
            
        if len(lfc_stds) != len(selected_genes):
            logger.warning(f"lfc_stds length {len(lfc_stds)} doesn't match selected_genes length {len(selected_genes)}. Reshaping.")
            if len(lfc_stds) < len(selected_genes):
                # Pad with NaNs if the array is too short
                padding = np.full(len(selected_genes) - len(lfc_stds), np.nan)
                lfc_stds = np.concatenate([lfc_stds, padding])
            else:
                # Truncate if the array is too long
                lfc_stds = lfc_stds[:len(selected_genes)]
        adata.var.loc[selected_genes, lfc_std_key] = lfc_stds
        
        # Bidirectionality score - NOT impacted by sample variance
        bidir_key = field_names["bidirectionality_key"]
        adata.var[bidir_key] = pd.Series(np.nan, index=adata.var_names)
        
        # Extract and verify bidirectionality
        bidirectionality = expression_results['bidirectionality']
        # Convert list to numpy array if needed
        if isinstance(bidirectionality, list):
            bidirectionality = np.array(bidirectionality)
        
        # Ensure bidirectionality is 1D before reshaping
        if len(bidirectionality.shape) > 1:
            logger.warning(f"bidirectionality has shape {bidirectionality.shape}, flattening to 1D.")
            # Take the first row if it's a 2D array
            if bidirectionality.shape[0] < bidirectionality.shape[1]:
                bidirectionality = bidirectionality[0]  # Take first row if more columns than rows
            else:
                bidirectionality = bidirectionality[:, 0]  # Take first column otherwise
            
        if len(bidirectionality) != len(selected_genes):
            logger.warning(f"bidirectionality length {len(bidirectionality)} doesn't match selected_genes length {len(selected_genes)}. Reshaping.")
            if len(bidirectionality) < len(selected_genes):
                # Pad with NaNs if the array is too short
                padding = np.full(len(selected_genes) - len(bidirectionality), np.nan)
                bidirectionality = np.concatenate([bidirectionality, padding])
            else:
                # Truncate if the array is too long
                bidirectionality = bidirectionality[:len(selected_genes)]
        adata.var.loc[selected_genes, bidir_key] = bidirectionality
        
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

            if imputed1_key not in adata.layers:
                adata.layers[imputed1_key] = np.zeros_like(adata.X)
            if imputed2_key not in adata.layers:
                adata.layers[imputed2_key] = np.zeros_like(adata.X)
            if fold_change_key not in adata.layers:
                adata.layers[fold_change_key] = np.zeros_like(adata.X)
            
            # Convert JAX arrays to NumPy arrays if needed
            condition1_imputed = np.array(expression_results['condition1_imputed'])
            condition2_imputed = np.array(expression_results['condition2_imputed'])
            fold_change = np.array(expression_results['fold_change'])
            
            # Map the imputed values to the correct positions
            for i, gene in enumerate(selected_genes):
                gene_idx = list(adata.var_names).index(gene)
                adata.layers[imputed1_key][:, gene_idx] = condition1_imputed[:, i]
                adata.layers[imputed2_key][:, gene_idx] = condition2_imputed[:, i]
                adata.layers[fold_change_key][:, gene_idx] = fold_change[:, i]
        else:
            # Convert JAX arrays to NumPy arrays if needed
            condition1_imputed = np.array(expression_results['condition1_imputed'])
            condition2_imputed = np.array(expression_results['condition2_imputed'])
            fold_change = np.array(expression_results['fold_change'])
            
            # Check shapes and reshape if necessary
            if condition1_imputed.shape != adata.shape:
                logger.warning(f"condition1_imputed shape {condition1_imputed.shape} doesn't match adata shape {adata.shape}. Skipping layer creation.")
                return result_dict
                
            # Use the standardized field names
            # Create descriptive layer names - these are NOT affected by sample variance
            imputed1_key = field_names["imputed_key_1"]
            imputed2_key = field_names["imputed_key_2"]
            fold_change_key = field_names["fold_change_key"]

            adata.layers[imputed1_key] = condition1_imputed
            adata.layers[imputed2_key] = condition2_imputed
            adata.layers[fold_change_key] = fold_change
        
        # Prepare parameters, run timestamp, and field metadata
        current_timestamp = datetime.datetime.now().isoformat()
        
        # Define parameters dict
        params_dict = {
            "groupby": groupby,
            "condition1": condition1,
            "condition2": condition2,
            "obsm_key": obsm_key,
            "layer": layer,
            "genes": genes,
            "n_landmarks": n_landmarks,
            "sample_col": sample_col,  # Keep this for documentation in the AnnData object
            "use_sample_variance": use_sample_variance,  # This is now inferred from sample_col
            "differential_abundance_key": differential_abundance_key,
            "ls_factor": ls_factor,
            "used_landmarks": True if landmarks is not None else False,
        }
        
        current_run_info = {
            "timestamp": current_timestamp,
            "function": "compute_differential_expression",
            "result_key": result_key,
            "analysis_type": "de",
            "lfc_key": field_names["mean_lfc_key"],
            "weighted_lfc_key": field_names["weighted_lfc_key"] if differential_abundance_key is not None else None,
            "mahalanobis_key": field_names["mahalanobis_key"] if compute_mahalanobis else None,
            "lfc_std_key": field_names["lfc_std_key"],
            "bidirectionality_key": field_names["bidirectionality_key"],
            "imputed_layer_keys": {
                "condition1": field_names["imputed_key_1"],
                "condition2": field_names["imputed_key_2"],
                "fold_change": field_names["fold_change_key"]
            },
            "field_names": field_names,
            "uses_sample_variance": sample_col is not None,
            "params": params_dict
        }
        
        # Also maintain separate params for backward compatibility
        current_params = params_dict.copy()
        
        # Always use fixed key "kompot_de" regardless of result_key
        storage_key = "kompot_de"
        
        # Initialize or update adata.uns[storage_key]
        if storage_key not in adata.uns:
            adata.uns[storage_key] = {}
            
        # Add environment info to the run info
        from ..utils import get_environment_info
        env_info = get_environment_info()
        current_run_info["environment"] = env_info
        
        # Initialize run history if it doesn't exist
        if "run_history" not in adata.uns[storage_key]:
            adata.uns[storage_key]["run_history"] = []
        
        # Always append current run to the run history
        adata.uns[storage_key]["run_history"].append(current_run_info)
        
        # Store current params and run info
        adata.uns[storage_key]["params"] = current_params
        adata.uns[storage_key]["run_info"] = current_run_info
        
    # Return the results dictionary
    return result_dict


def run_differential_analysis(
    adata,
    groupby: str,
    condition1: str,
    condition2: str,
    obsm_key: str = "DM_EigenVectors",
    layer: Optional[str] = None,
    genes: Optional[List[str]] = None,
    n_landmarks: Optional[int] = None,
    landmarks: Optional[np.ndarray] = None,
    compute_abundance: bool = True,
    compute_expression: bool = True,
    abundance_key: str = "kompot_da",
    expression_key: str = "kompot_de",
    share_landmarks: bool = True,
    ls_factor: float = 10.0,
    jit_compile: bool = False,
    random_state: Optional[int] = None,
    copy: bool = False,
    generate_html_report: bool = True,
    report_dir: str = "kompot_report",
    open_browser: bool = True,
    overwrite: Optional[bool] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Run a complete differential analysis workflow on an AnnData object.
    
    This function computes both differential abundance and differential expression
    between two conditions and stores the results in the AnnData object.
    
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
        by default None. Ignored if landmarks is provided.
    landmarks : np.ndarray, optional
        Pre-computed landmarks to use. If provided, n_landmarks will be ignored.
        Shape (n_landmarks, n_features).
    compute_abundance : bool, optional
        Whether to compute differential abundance, by default True.
    compute_expression : bool, optional
        Whether to compute differential expression, by default True.
    abundance_key : str, optional
        Key in adata.uns where differential abundance results will be stored,
        by default "kompot_da".
    expression_key : str, optional
        Key in adata.uns where differential expression results will be stored,
        by default "kompot_de".
    share_landmarks : bool, optional
        Whether to share landmarks between abundance and expression analyses,
        by default True.
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
    generate_html_report : bool, optional
        Whether to generate an HTML report with the results, by default True.
    report_dir : str, optional
        Directory where the HTML report will be saved, by default "kompot_report".
    open_browser : bool, optional
        Whether to open the HTML report in a browser, by default True.
    overwrite : bool, optional
        Controls behavior when results with the same result_key already exist:
        - If None (default): Warn about existing results but proceed with overwriting
        - If True: Silently overwrite existing results
        - If False: Raise an error if results would be overwritten
    **kwargs : dict
        Additional arguments to pass to compute_differential_abundance and 
        compute_differential_expression.
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
            - "adata": The AnnData object with analysis results added (a new object if copy=True)
            - "differential_abundance": The DifferentialAbundance model if compute_abundance=True
            - "differential_expression": The DifferentialExpression model if compute_expression=True
    
    Notes
    -----
    This function runs the full Kompot differential analysis workflow and provides
    a simplified interface for both differential abundance and expression analysis.
    Results are stored in the AnnData object's obs, var, layers, and uns attributes.

    The `log_fold_change_direction` column in adata.obs is assigned categorical values
    ('up', 'down', or 'neutral'), and matching colors are stored in adata.uns with
    the '_colors' postfix for easy visualization in scanpy and other tools:
    
    ```python
    # Color scheme used
    direction_colors = {"up": "#d73027", "down": "#4575b4", "neutral": "#d3d3d3"}
    
    # This allows direct use with scanpy's plotting functions
    import scanpy as sc
    sc.pl.umap(adata, color=f"{abundance_key}_log_fold_change_direction")
    ```
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
    
    # Separate kwargs for each analysis type
    abundance_kwargs = {k: v for k, v in kwargs.items() if k in [
        'log_fold_change_threshold', 'pvalue_threshold', 'batch_size', 'sample_col'
    ]}
    
    expression_kwargs = {k: v for k, v in kwargs.items() if k in [
        'sample_col', 'compute_weighted_fold_change', 'sigma', 'ls',
        'compute_mahalanobis'
    ]}
    
    report_kwargs = {k: v for k, v in kwargs.items() if k in [
        'title', 'subtitle', 'template_dir', 'use_cdn', 'top_n', 'groupby', 'embedding_key'
    ]}
    
    # Run differential abundance if requested
    abundance_result = None
    abundance_landmarks = None
    if compute_abundance:
        logger.info("Computing differential abundance...")
        abundance_result = compute_differential_abundance(
            adata,
            groupby=groupby,
            condition1=condition1,
            condition2=condition2,
            obsm_key=obsm_key,
            n_landmarks=n_landmarks,
            landmarks=landmarks,
            ls_factor=ls_factor,
            jit_compile=jit_compile,
            random_state=random_state,
            inplace=True,
            result_key=abundance_key,
            overwrite=overwrite,
            **abundance_kwargs
        )
        
        # Check if landmarks are stored in abundance_key
        if abundance_key in adata.uns and 'landmarks' in adata.uns[abundance_key]:
            abundance_landmarks = adata.uns[abundance_key]['landmarks']
            if share_landmarks:
                logger.info(f"Will reuse landmarks from differential abundance analysis for expression analysis")
                # Update abundance_key uns to indicate landmarks were shared
                if 'params' in adata.uns[abundance_key]:
                    adata.uns[abundance_key]['params']['landmarks_shared_with_expression'] = True
    
    # Run differential expression if requested
    expression_result = None
    if compute_expression:
        logger.info("Computing differential expression...")
        # Check if the abundance_key log density fields exist
        diff_abund_key = abundance_key if compute_abundance else None
        
        # Make sure log density columns exist if we're trying to use differential abundance key
        if diff_abund_key is not None:
            # Sanitize condition names for use in column names
            cond1_safe = _sanitize_name(condition1)
            cond2_safe = _sanitize_name(condition2)
            
            # Check for condition-specific column names
            specific_cols = [f"{diff_abund_key}_log_density_{cond1_safe}", f"{diff_abund_key}_log_density_{cond2_safe}"]
            
            if not all(col in adata.obs for col in specific_cols):
                logger.warning(f"Log density columns not found in adata.obs. "
                              f"Expected: {specific_cols}. "
                              f"Will not compute weighted mean fold changes.")
                diff_abund_key = None
        
        # Use abundance landmarks for expression analysis if available and sharing is enabled
        expr_landmarks = landmarks
        if share_landmarks and abundance_landmarks is not None:
            expr_landmarks = abundance_landmarks
        
        expression_result = compute_differential_expression(
            adata,
            groupby=groupby,
            condition1=condition1,
            condition2=condition2,
            obsm_key=obsm_key,
            layer=layer,
            genes=genes,
            n_landmarks=n_landmarks,
            landmarks=expr_landmarks,
            ls_factor=ls_factor,
            jit_compile=jit_compile,
            random_state=random_state,
            differential_abundance_key=diff_abund_key,
            inplace=True,
            result_key=expression_key,
            overwrite=overwrite,
            **expression_kwargs
        )
    
    # Generate HTML report if requested
    if generate_html_report and compute_expression and expression_result is not None:
        logger.info("Generating HTML report...")
        # Get the model from the expression_result
        diff_expr = expression_result["model"]
        report_path = generate_report(
            diff_expr,
            output_dir=report_dir,
            adata=adata,
            condition1_name=condition1,
            condition2_name=condition2,
            open_browser=open_browser,
            **report_kwargs
        )
        logger.info(f"HTML report generated at: {report_path}")
        
    # Store information about landmark sharing in adata.uns
    if compute_abundance and compute_expression and share_landmarks and abundance_landmarks is not None:
        # Create a landmarks_info entry if needed
        if 'landmarks_info' not in adata.uns:
            adata.uns['landmarks_info'] = {}
        
        # Record that landmarks were shared between abundance and expression
        adata.uns['landmarks_info']['shared_between_analyses'] = True
        adata.uns['landmarks_info']['source'] = abundance_key
        adata.uns['landmarks_info']['targets'] = [expression_key]
        
        # Store timestamp of sharing
        adata.uns['landmarks_info']['timestamp'] = datetime.datetime.now().isoformat()
    
    # Store combined run information with enhanced environment data
    # Get environment info
    from ..utils import get_environment_info
    env_info = get_environment_info()
    
    # Create a kompot_run_history entry if it doesn't exist
    if 'kompot_run_history' not in adata.uns:
        adata.uns['kompot_run_history'] = []
    
    # Add current run info to the history with a run_id for reference
    run_id = len(adata.uns['kompot_run_history'])
    
    # Create parameters dictionary
    parameters_dict = {
        "obsm_key": obsm_key,
        "share_landmarks": share_landmarks,
        "ls_factor": ls_factor,
        "generate_html_report": generate_html_report,
        "groupby": groupby,
        "condition1": condition1,
        "condition2": condition2,
        "layer": layer,
        "genes": genes,
        "n_landmarks": n_landmarks,
        "compute_abundance": compute_abundance,
        "compute_expression": compute_expression
    }
    
    run_info = {
        "run_id": run_id,
        "timestamp": env_info["timestamp"],
        "function": "run_differential_analysis",
        "analysis_type": "combined",
        "result_key": abundance_key if compute_abundance else (expression_key if compute_expression else None),
        "abundance_key": abundance_key if compute_abundance else None,  # Keep for reference
        "expression_key": expression_key if compute_expression else None,  # Keep for reference
        "conditions": {
            "groupby": groupby,
            "condition1": condition1,
            "condition2": condition2
        },
        "params": parameters_dict,
        "environment": env_info  # Add all environment info
    }
    
    # Add to global history
    adata.uns['kompot_run_history'].append(run_info)
    
    # Also add to the appropriate specific history locations
    if compute_abundance:
        # Make sure kompot_da exists and has a run_history
        if 'kompot_da' not in adata.uns:
            adata.uns['kompot_da'] = {}
        if 'run_history' not in adata.uns['kompot_da']:
            adata.uns['kompot_da']['run_history'] = []
        
        # Add to DA specific history
        adata.uns['kompot_da']['run_history'].append(run_info)
    
    if compute_expression:
        # Make sure kompot_de exists and has a run_history
        if 'kompot_de' not in adata.uns:
            adata.uns['kompot_de'] = {}
        if 'run_history' not in adata.uns['kompot_de']:
            adata.uns['kompot_de']['run_history'] = []
        
        # Add to DE specific history
        adata.uns['kompot_de']['run_history'].append(run_info)
    
    # Store the latest run as a separate key for easy access
    adata.uns['kompot_latest_run'] = run_info
    
    # Note: The individual analyses (DA and DE) already store their runs 
    # in the fixed storage locations during their execution.
    # The global history is maintained separately for combined runs.
    
    # Return the results along with the AnnData
    return {
        "adata": adata,
        "differential_abundance": abundance_result["model"] if abundance_result else None,
        "differential_expression": expression_result["model"] if expression_result else None,
    }


def generate_report(
    diff_expr,
    output_dir="kompot_report",
    adata=None,
    condition1_name="Condition 1",
    condition2_name="Condition 2",
    **kwargs
):
    """Generate an interactive HTML report for differential expression results.
    
    Parameters
    ----------
    diff_expr : DifferentialExpression
        DifferentialExpression object with results
    output_dir : str, optional
        Directory where the report will be saved, by default "kompot_report"
    adata : AnnData, optional
        AnnData object with cell annotations, by default None
    condition1_name : str, optional
        Name of the first condition, by default "Condition 1"
    condition2_name : str, optional
        Name of the second condition, by default "Condition 2"
    **kwargs : dict
        Additional arguments to pass to HTMLReporter or to reporter methods
        
    Returns
    -------
    str
        Path to the generated report
    """
    # Extract parameters for different methods
    reporter_params = {k: v for k, v in kwargs.items() if k in [
        'title', 'subtitle', 'template_dir', 'use_cdn'
    ]}
    
    diff_expr_params = {k: v for k, v in kwargs.items() if k in [
        'gene_names', 'top_n'
    ]}
    
    anndata_params = {k: v for k, v in kwargs.items() if k in [
        'groupby', 'embedding_key', 'cell_annotations'
    ]}
    
    # Whether to open browser at the end
    open_browser = kwargs.get("open_browser", True)
    
    # Create reporter
    reporter = HTMLReporter(output_dir=output_dir, **reporter_params)
    
    # Add differential expression results
    reporter.add_differential_expression(
        diff_expr,
        condition1_name=condition1_name,
        condition2_name=condition2_name,
        **diff_expr_params
    )
    
    if adata is not None:
        # Try to use default parameters, but allow override
        groupby = anndata_params.get("groupby", "leiden" if hasattr(adata.obs, "get") and adata.obs.get("leiden") is not None else "louvain")
        embedding_key = anndata_params.get("embedding_key", "X_umap" if hasattr(adata.obsm, "get") and adata.obsm.get("X_umap") is not None else "X_tsne")
        
        # Add AnnData only if the required columns exist
        if hasattr(adata.obs, "get") and adata.obs.get(groupby) is not None and hasattr(adata.obsm, "get") and adata.obsm.get(embedding_key) is not None:
            cell_annotations = anndata_params.get("cell_annotations", None)
            reporter.add_anndata(
                adata,
                groupby=groupby, 
                embedding_key=embedding_key,
                cell_annotations=cell_annotations
            )
    
    return reporter.generate(open_browser=open_browser)