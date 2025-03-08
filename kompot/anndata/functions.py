"""
AnnData integration functions for Kompot.
"""

import logging
import numpy as np
import pandas as pd
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
    log_fold_change_threshold: float = 1.7,
    pvalue_threshold: float = 1e-3,
    jit_compile: bool = False,
    random_state: Optional[int] = None,
    copy: bool = False,
    inplace: bool = True,
    result_key: str = "kompot_da",
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
    log_fold_change_threshold : float, optional
        Threshold for considering a log fold change significant, by default 1.7.
    pvalue_threshold : float, optional
        Threshold for considering a p-value significant, by default 1e-3.
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
    - adata.obs[f"{result_key}_log_fold_change_pvalue"]: P-values for each cell
    - adata.obs[f"{result_key}_log_fold_change_direction"]: Direction of change
    - adata.uns[result_key]: Dictionary with additional information and parameters
    - If landmarks are computed, they are stored in adata.uns[result_key]['landmarks']
      for potential reuse in other analyses.
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
        random_state=random_state
    )
    
    # Fit the estimators
    diff_abundance.fit(X_condition1, X_condition2, landmarks=landmarks, **density_kwargs)
    
    # Run prediction to compute fold changes and metrics
    X_for_prediction = adata.obsm[obsm_key]
    abundance_results = diff_abundance.predict(
        X_for_prediction,
        log_fold_change_threshold=log_fold_change_threshold,
        pvalue_threshold=pvalue_threshold
    )
    # Note: mean_log_fold_change is no longer computed by default
    
    # Store landmarks if they were computed for future reuse
    if hasattr(diff_abundance, 'computed_landmarks') and diff_abundance.computed_landmarks is not None:
        # Store the landmarks in adata.uns for future use
        if result_key not in adata.uns:
            adata.uns[result_key] = {}
        adata.uns[result_key]['landmarks'] = diff_abundance.computed_landmarks
        logger.info(f"Stored computed landmarks in adata.uns['{result_key}']['landmarks']")
    
    # Sanitize condition names for use in column names
    cond1_safe = _sanitize_name(condition1)
    cond2_safe = _sanitize_name(condition2)
    
    # Assign values to masked cells with more descriptive column names
    adata.obs[f"{result_key}_log_fold_change_{cond2_safe}_vs_{cond1_safe}"] = abundance_results['log_fold_change']
    adata.obs[f"{result_key}_log_fold_change_zscore"] = abundance_results['log_fold_change_zscore']
    adata.obs[f"{result_key}_log_fold_change_pvalue"] = abundance_results['log_fold_change_pvalue']
    adata.obs[f"{result_key}_log_fold_change_direction"] = abundance_results['log_fold_change_direction']
    
    # Store log densities for each condition with descriptive names
    adata.obs[f"{result_key}_log_density_{cond1_safe}"] = abundance_results['log_density_condition1']
    adata.obs[f"{result_key}_log_density_{cond2_safe}"] = abundance_results['log_density_condition2']
    
    # Store parameters in adata.uns, but NOT the model (too large for serialization)
    adata.uns[result_key] = {
        "params": {
            "groupby": groupby,
            "condition1": condition1,
            "condition2": condition2,
            "obsm_key": obsm_key,
            "log_fold_change_threshold": log_fold_change_threshold,
            "pvalue_threshold": pvalue_threshold,
            "n_landmarks": n_landmarks,
            "used_landmarks": True if landmarks is not None else False,
        }
    }
    
    # Return results as a dictionary
    return {
        "log_fold_change": abundance_results['log_fold_change'],
        "log_fold_change_zscore": abundance_results['log_fold_change_zscore'],
        "log_fold_change_pvalue": abundance_results['log_fold_change_pvalue'],
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
    compute_mahalanobis: bool = True,
    jit_compile: bool = False,
    random_state: Optional[int] = None,
    batch_size: int = 100,
    copy: bool = False,
    inplace: bool = True,
    result_key: str = "kompot_de",
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
    copy : bool, optional
        If True, return a copy of the AnnData object with results added,
        by default False.
    inplace : bool, optional
        If True, modify adata in place, by default True.
    result_key : str, optional
        Key in adata.uns where results will be stored, by default "kompot_de".
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
        landmarks=landmarks,
        condition1_sample_indices=condition1_sample_indices,
        condition2_sample_indices=condition2_sample_indices,
        **function_kwargs
    )
    
    # Store landmarks if they were computed for future reuse
    if hasattr(diff_expression, 'computed_landmarks') and diff_expression.computed_landmarks is not None:
        # Store the landmarks in adata.uns for future use
        if result_key not in adata.uns:
            adata.uns[result_key] = {}
        adata.uns[result_key]['landmarks'] = diff_expression.computed_landmarks
        logger.info(f"Stored computed landmarks in adata.uns['{result_key}']['landmarks']")
    
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
        else:
            raise ValueError(f"Log density columns not found in adata.obs. Expected: {density_col1}, {density_col2}")
        
        # Calculate log density difference directly
        log_density_diff = log_density_condition2 - log_density_condition1
        
        # Use the standalone function to compute weighted mean fold change with pre-computed difference
        # The exp(abs()) is now handled inside the function
        expression_results['weighted_mean_log_fold_change'] = compute_weighted_mean_fold_change(
            expression_results['fold_change'],
            log_density_diff=log_density_diff
        )
    
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
                
            adata.var[f"{result_key}_mahalanobis"] = pd.Series(np.nan, index=adata.var_names)
            adata.var.loc[selected_genes, f"{result_key}_mahalanobis"] = mahalanobis_distances
        
        # Sanitize condition names for use in column names
        cond1_safe = _sanitize_name(condition1)
        cond2_safe = _sanitize_name(condition2)
        
        if differential_abundance_key is not None:
            # Initialize with np.nan of appropriate shape - use more descriptive column name
            column_name = f"{result_key}_weighted_lfc_{cond2_safe}_vs_{cond1_safe}"
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
        mean_lfc_column = f"{result_key}_mean_lfc_{cond2_safe}_vs_{cond1_safe}"
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
        
        # Standard deviation of log fold change
        adata.var[f"{result_key}_lfc_std"] = pd.Series(np.nan, index=adata.var_names)
        
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
        adata.var.loc[selected_genes, f"{result_key}_lfc_std"] = lfc_stds
        
        # Bidirectionality score
        adata.var[f"{result_key}_bidirectionality"] = pd.Series(np.nan, index=adata.var_names)
        
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
        adata.var.loc[selected_genes, f"{result_key}_bidirectionality"] = bidirectionality
        
        # Add cell-gene level results
        n_selected_genes = len(selected_genes)
        
        # Process the data to match the shape of the full gene set
        if n_selected_genes < len(adata.var_names):
            # We need to expand the imputed data to the full gene set
            # Sanitize condition names for use in layer names
            cond1_safe = _sanitize_name(condition1)
            cond2_safe = _sanitize_name(condition2)
            
            # Create descriptive layer names
            imputed1_key = f"{result_key}_imputed_{cond1_safe}"
            imputed2_key = f"{result_key}_imputed_{cond2_safe}"
            fold_change_key = f"{result_key}_fold_change_{cond2_safe}_vs_{cond1_safe}"
            
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
                
            # Sanitize condition names for use in layer names
            cond1_safe = _sanitize_name(condition1)
            cond2_safe = _sanitize_name(condition2)
            
            # Create descriptive layer names
            imputed1_key = f"{result_key}_imputed_{cond1_safe}"
            imputed2_key = f"{result_key}_imputed_{cond2_safe}"
            fold_change_key = f"{result_key}_fold_change_{cond2_safe}_vs_{cond1_safe}"
            
            # Create zero matrices and fill gene by gene to avoid shape issues
            adata.layers[imputed1_key] = np.zeros_like(adata.X)
            adata.layers[imputed2_key] = np.zeros_like(adata.X)
            adata.layers[fold_change_key] = np.zeros_like(adata.X)
            
            # Use the same approach as in the selected_genes < len(adata.var_names) case
            for i, gene in enumerate(selected_genes):
                if i < condition1_imputed.shape[1]:  # Make sure we don't go out of bounds
                    gene_idx = list(adata.var_names).index(gene)
                    adata.layers[imputed1_key][:, gene_idx] = condition1_imputed[:, i]
                    adata.layers[imputed2_key][:, gene_idx] = condition2_imputed[:, i]
                    adata.layers[fold_change_key][:, gene_idx] = fold_change[:, i]
        
        # Store model and parameters in adata.uns
        adata.uns[result_key] = {
            "params": {
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
                "used_landmarks": True if landmarks is not None else False,
            },
        }
        
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
    jit_compile: bool = False,
    random_state: Optional[int] = None,
    copy: bool = False,
    generate_html_report: bool = True,
    report_dir: str = "kompot_report",
    open_browser: bool = True,
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
        'log_fold_change_threshold', 'pvalue_threshold'
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
            jit_compile=jit_compile,
            random_state=random_state,
            inplace=True,
            result_key=abundance_key,
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
            jit_compile=jit_compile,
            random_state=random_state,
            differential_abundance_key=diff_abund_key,
            inplace=True,
            result_key=expression_key,
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
        from datetime import datetime
        adata.uns['landmarks_info']['timestamp'] = datetime.now().isoformat()
    
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