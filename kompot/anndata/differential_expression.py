"""
Differential expression analysis for AnnData objects.
"""

import logging
import numpy as np
import pandas as pd
import datetime
from typing import Optional, Union, Dict, Any, List, Tuple
from scipy import sparse

from ..differential import DifferentialExpression, compute_weighted_mean_fold_change
from ..utils import (
    detect_output_field_overwrite, 
    generate_output_field_names,
    get_environment_info,
    KOMPOT_COLORS
)
from ..memory_utils import analyze_covariance_memory_requirements
from .core import _sanitize_name

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
    random_state: Optional[int] = None,
    batch_size: int = 100,
    store_arrays_on_disk: bool = False,
    disk_storage_dir: Optional[str] = None,
    max_memory_ratio: float = 0.8,
    mahalanobis_batch_size: Optional[int] = None,
    copy: bool = False,
    inplace: bool = True,
    result_key: str = "kompot_de",
    overwrite: Optional[bool] = None,
    store_landmarks: bool = False,
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
        Number of cells to process at once during prediction to manage memory usage.
        If None or 0, all samples will be processed at once. Default is 100.
    store_arrays_on_disk : bool, optional
        Whether to store large arrays on disk instead of in memory, by default False.
        This is useful for very large datasets with many genes, where covariance
        matrices would otherwise exceed available memory.
    disk_storage_dir : str, optional
        Directory to store arrays on disk. If None and store_arrays_on_disk is True,
        a temporary directory will be created and cleaned up afterwards.
    max_memory_ratio : float, optional
        Maximum fraction of available memory that arrays should occupy before
        triggering warnings or enabling disk storage, by default 0.8 (80%).
    mahalanobis_batch_size : int, optional
        Number of genes to process in each batch during Mahalanobis distance computation.
        Smaller values use less memory but are slower. If None, uses batch_size.
        Increase for faster computation if you have sufficient memory.
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
    
    # Collect all patterns for both var columns and layers
    all_patterns = {
        "var": [
            field_names["mahalanobis_key"],      # Impacted by sample variance
            field_names["mean_lfc_key"],         # Not impacted by sample variance
            field_names["bidirectionality_key"], # Not impacted by sample variance
            field_names["lfc_std_key"]           # Impacted by sample variance
        ],
        "layers": [
            field_names["imputed_key_1"],        # Not impacted by sample variance
            field_names["fold_change_key"]       # Not impacted by sample variance
        ]
    }
    
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
                                   f"bidirectionality, imputed data, fold_change) will be overwritten since they "
                                   f"don't use the sample variance suffix. These results will likely be identical "
                                   f"if other parameters haven't changed.")
                    elif current_sample_var:
                        message += (f" Fields that will be overwritten: {field_list}. "
                                   f"Note: Only fields NOT affected by sample variance (like mean_log_fold_change, "
                                   f"bidirectionality, imputed data, fold_change) will be overwritten since they "
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
    
    # Analyze memory requirements to provide guidance
    
    # Determine points count (actual points or landmarks if used)
    points_count = n_landmarks if n_landmarks is not None else max(len(X_condition1), len(X_condition2))
    genes_count = len(selected_genes)
    
    # Run memory analysis
    memory_analysis = analyze_covariance_memory_requirements(
        n_points=points_count,
        n_genes=genes_count,
        max_memory_ratio=max_memory_ratio,
        analysis_name="Differential Expression Memory Analysis"
    )
    
    # If memory usage is high but store_arrays_on_disk wasn't specified, provide guidance
    if memory_analysis['should_use_disk'] and not store_arrays_on_disk:
        logger.warning(
            f"High memory requirements detected: would need {memory_analysis['total_size']} "
            f"for covariance matrices with {points_count} points and {genes_count} genes. "
            f"Consider using store_arrays_on_disk=True to enable disk-backed storage."
        )
    
    # If user specifically enabled disk storage, log that
    if store_arrays_on_disk:
        storage_location = disk_storage_dir if disk_storage_dir else "a temporary directory"
        logger.info(f"Disk-backed storage enabled. Covariance matrices will be stored in {storage_location}")
    
    # Initialize and fit DifferentialExpression
    use_sample_variance = sample_col is not None
    
    diff_expression = DifferentialExpression(
        n_landmarks=n_landmarks,
        use_sample_variance=use_sample_variance,
        jit_compile=jit_compile,
        random_state=random_state,
        batch_size=batch_size,
        mahalanobis_batch_size=mahalanobis_batch_size,
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
        **function_kwargs
    )
    
    # Handle landmarks for future reference
    if hasattr(diff_expression, 'computed_landmarks') and diff_expression.computed_landmarks is not None:
        # Initialize if needed
        if result_key not in adata.uns:
            adata.uns[result_key] = {}
        
        # Get landmarks info
        landmarks_shape = diff_expression.computed_landmarks.shape
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
        
    # Add landmarks to result dictionary if they were computed
    if hasattr(diff_expression, 'computed_landmarks') and diff_expression.computed_landmarks is not None:
        result_dict["landmarks"] = diff_expression.computed_landmarks
    
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
            # Check if column already exists and initialize only if it doesn't
            if mahalanobis_key not in adata.var:
                adata.var[mahalanobis_key] = pd.Series(np.nan, index=adata.var_names)
            adata.var.loc[selected_genes, mahalanobis_key] = mahalanobis_distances
        
        if differential_abundance_key is not None:
            # Use the standardized field name from field_names
            # Weighted mean log fold change is NOT impacted by sample variance
            column_name = field_names["weighted_lfc_key"]
            # Check if column already exists and initialize only if it doesn't
            if column_name not in adata.var:
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
        
        # Add mean log fold change with descriptive name
        # Use the standardized field name from field_names
        # Mean log fold change is NOT impacted by sample variance
        mean_lfc_column = field_names["mean_lfc_key"]
        # Check if column already exists and initialize only if it doesn't
        if mean_lfc_column not in adata.var:
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
        # Check if column already exists and initialize only if it doesn't
        if lfc_std_key not in adata.var:
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
        # Check if column already exists and initialize only if it doesn't
        if bidir_key not in adata.var:
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

            # Initialize layers only if they don't already exist
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

            # Only create or overwrite these layers if the data shape matches
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
            "store_arrays_on_disk": store_arrays_on_disk,
            "max_memory_ratio": max_memory_ratio,
            "mahalanobis_batch_size": mahalanobis_batch_size
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
            "lfc_std_key": field_names["lfc_std_key"],
            "bidirectionality_key": field_names["bidirectionality_key"],
            "imputed_layer_keys": {
                "condition1": field_names["imputed_key_1"],
                "condition2": field_names["imputed_key_2"],
                "fold_change": field_names["fold_change_key"]
            },
            "field_names": field_names,
            "uses_sample_variance": sample_col is not None,
            "memory_analysis": memory_analysis if 'memory_analysis' in locals() else None,
            "storage_stats": storage_stats,
            "params": params_dict
        }
        
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

        new_run_id = len(adata.uns[storage_key]["run_history"])
        logger.info(f"This run will have `run_id={new_run_id}`.")
        
        # Always append current run to the run history
        adata.uns[storage_key]["run_history"].append(current_run_info)
        
        # Store current params and run info
        adata.uns[storage_key]["last_run_info"] = current_run_info
        
    # Return the results dictionary
    return result_dict