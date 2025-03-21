"""Utilities for fold change computations in Kompot."""

import numpy as np
import logging
import pandas as pd
from typing import Optional, Tuple
from anndata import AnnData

# Get the pre-configured logger
logger = logging.getLogger("kompot")

def compute_weighted_mean_fold_change(
    fold_change: np.ndarray,
    log_density_condition1: np.ndarray = None,
    log_density_condition2: np.ndarray = None,
    log_density_diff: np.ndarray = None
) -> np.ndarray:
    """
    Compute weighted mean fold change using density differences as weights.
    
    This utility function computes weighted mean fold changes from expression and density log fold changes.
    
    Parameters
    ----------
    fold_change : np.ndarray
        Expression fold change for each cell and gene. Shape (n_cells, n_genes).
    log_density_condition1 : np.ndarray or pandas.Series, optional
        Log density for condition 1. Shape (n_cells,). Can be omitted if log_density_diff is provided.
    log_density_condition2 : np.ndarray or pandas.Series, optional
        Log density for condition 2. Shape (n_cells,). Can be omitted if log_density_diff is provided.
    log_density_diff : np.ndarray, optional
        Pre-computed log density difference. If provided, log_density_condition1 and 
        log_density_condition2 are ignored. Shape (n_cells,).
        
    Returns
    -------
    np.ndarray
        Weighted mean log fold change for each gene. Shape (n_genes,).
    """
    if log_density_diff is None:
        if log_density_condition1 is None or log_density_condition2 is None:
            raise ValueError("Either log_density_diff or both log_density_condition1 and log_density_condition2 must be provided")
        
        # Convert pandas Series to numpy arrays if needed
        if hasattr(log_density_condition1, 'to_numpy'):
            log_density_condition1 = log_density_condition1.to_numpy()
        if hasattr(log_density_condition2, 'to_numpy'):
            log_density_condition2 = log_density_condition2.to_numpy()
            
        # Calculate the density difference for each cell
        log_density_diff = log_density_condition2 - log_density_condition1
    elif hasattr(log_density_diff, 'to_numpy'):
        # Convert pandas Series to numpy arrays if needed
        log_density_diff = log_density_diff.to_numpy()
    
    # Convert to numpy array if it's a list
    if isinstance(fold_change, list):
        fold_change = np.array(fold_change)
    
    # Create a weights array with shape (n_cells, 1) for broadcasting
    # Apply np.exp(np.abs(...)) to the log_density_diff as part of the function's logic
    weights = np.exp(np.abs(log_density_diff.reshape(-1, 1)))
    
    # Weight the fold changes by density difference
    weighted_fold_change = fold_change * weights
    
    return np.sum(weighted_fold_change, axis=0) / np.sum(weights)


def update_direction_column(
    adata: AnnData,
    lfc_threshold: Optional[float] = None,
    pval_threshold: Optional[float] = None,
    direction_column: Optional[str] = None,
    lfc_key: Optional[str] = None,
    pval_key: Optional[str] = None,
    run_id: int = -1,
    inplace: bool = True,
) -> Optional[AnnData]:
    """
    Update the direction column in an AnnData object based on new thresholds.
    
    This function recalculates the direction categories (up, down, neutral) for cells
    based on new log fold change and p-value thresholds.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object containing differential abundance results
    lfc_threshold : float, optional
        New log fold change threshold for determining direction.
        If None, uses the threshold from the specified run_id.
    pval_threshold : float, optional
        New p-value threshold for determining direction (raw p-value, not -log10).
        If None, uses the threshold from the specified run_id.
    direction_column : str, optional
        Direction column to update. If None, infers from run_id.
    lfc_key : str, optional
        Log fold change column in adata.obs. If None, infers from run_id.
    pval_key : str, optional
        P-value column in adata.obs. If None, infers from run_id.
    run_id : int, optional
        Run ID to use for inferring column names. Default is -1 (latest run).
    inplace : bool, optional
        Whether to modify adata in place or return a copy. Default is True.
        
    Returns
    -------
    AnnData or None
        If inplace is False, returns a modified copy of the AnnData object.
        Otherwise, returns None (adata is modified in place).
    """
    from ..utils import get_run_from_history
    from ..plot.volcano import _infer_da_keys
    from ..plot.heatmap.direction_plot import _infer_direction_key
    
    # Make a copy if requested
    if not inplace:
        adata = adata.copy()
    
    # Get column names if not provided
    if lfc_key is None or pval_key is None:
        inferred_lfc_key, inferred_pval_key, thresholds = _infer_da_keys(adata, run_id)
        auto_lfc_threshold, auto_pval_threshold = thresholds
        
        # Use inferred keys if not explicitly provided
        if lfc_key is None:
            lfc_key = inferred_lfc_key
        if pval_key is None:
            pval_key = inferred_pval_key
    else:
        # If both keys are provided, still try to get thresholds from run info
        run_info = get_run_from_history(adata, run_id, analysis_type="da")
        if run_info is not None and 'params' in run_info:
            params = run_info['params']
            auto_lfc_threshold = params.get('log_fold_change_threshold')
            auto_pval_threshold = params.get('pvalue_threshold')
        else:
            auto_lfc_threshold, auto_pval_threshold = None, None
    
    # Use run thresholds if new thresholds not provided
    if lfc_threshold is None and auto_lfc_threshold is not None:
        lfc_threshold = auto_lfc_threshold
        logger.debug(f"Using run_id={run_id} lfc_threshold: {lfc_threshold}")
    elif lfc_threshold is None:
        raise ValueError("No log fold change threshold found. Please provide lfc_threshold.")
        
    if pval_threshold is None and auto_pval_threshold is not None:
        pval_threshold = auto_pval_threshold
        logger.debug(f"Using run_id={run_id} pval_threshold: {pval_threshold}")
    elif pval_threshold is None:
        raise ValueError("No p-value threshold found. Please provide pval_threshold.")
    
    # Find direction column if not provided
    if direction_column is None:
        direction_column, _, _ = _infer_direction_key(adata, run_id)
        
        if direction_column is None:
            # If still not found, try to construct from field names
            run_info = get_run_from_history(adata, run_id, analysis_type="da")
            if run_info is not None and "field_names" in run_info:
                field_names = run_info["field_names"]
                if "direction_key" in field_names:
                    direction_column = field_names["direction_key"]
                    
            # If still not found, raise an error
            if direction_column is None:
                raise ValueError("Could not find direction column. Please provide direction_column.")
    
    logger.info(f"Updating direction column '{direction_column}' with thresholds: "
                f"lfc_threshold={lfc_threshold}, pval_threshold={pval_threshold}")
    
    # Get log fold change and p-value data
    lfc = adata.obs[lfc_key].values
    
    # Check if p-values are already -log10 transformed
    if 'neg_log10' in pval_key.lower() or '-log10' in pval_key.lower():
        # Already in -log10 form, so higher values are more significant
        pvals = adata.obs[pval_key].values
        log10_pval_threshold = -np.log10(pval_threshold)
        is_significant = pvals > log10_pval_threshold
    else:
        # Raw p-values, so lower is more significant
        pvals = adata.obs[pval_key].values
        is_significant = pvals < pval_threshold
        
    # Create direction array
    direction = np.full(len(lfc), 'neutral', dtype=object)
    direction[np.logical_and(lfc > lfc_threshold, is_significant)] = 'up'
    direction[np.logical_and(lfc < -lfc_threshold, is_significant)] = 'down'
    
    # Update AnnData
    adata.obs[direction_column] = direction
    
    # Make sure the column is categorical
    if not pd.api.types.is_categorical_dtype(adata.obs[direction_column]):
        adata.obs[direction_column] = adata.obs[direction_column].astype('category')
    
    # Log counts of each direction
    up_count = np.sum(direction == 'up')
    down_count = np.sum(direction == 'down')
    neutral_count = np.sum(direction == 'neutral')
    logger.info(f"Updated direction counts: up={up_count}, down={down_count}, neutral={neutral_count}")
    
    # Return the modified object if not inplace
    if not inplace:
        return adata
    return None