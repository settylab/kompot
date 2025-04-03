"""Utility functions for volcano plots."""

import numpy as np
from typing import Optional, Tuple
from anndata import AnnData
import logging

# Get the pre-configured logger
logger = logging.getLogger("kompot")


def _extract_conditions_from_key(key: str) -> Optional[Tuple[str, str]]:
    """
    Extract condition names from a key name containing 'to'.
    
    Parameters
    ----------
    key : str
        Key name, containing 'to' between condition names
        
    Returns
    -------
    tuple or None
        (condition1, condition2) if found, None otherwise
    """
    if key is None:
        return None
        
    # Try to extract from key name, assuming format like "kompot_de_mean_lfc_Old_to_Young"
    key_parts = key.split('_')
    
    # Extract using the 'to' format
    if len(key_parts) >= 2 and 'to' in key_parts:
        to_index = key_parts.index('to')
        if to_index > 0 and to_index < len(key_parts) - 1:
            condition1 = key_parts[to_index-1]
            condition2 = key_parts[to_index+1]
            return condition1, condition2
    
    return None


def _infer_de_keys(adata: AnnData, run_id: int = -1, lfc_key: Optional[str] = None, 
                   score_key: Optional[str] = None):
    """
    Infer differential expression keys from AnnData object.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object with differential expression results
    run_id : int, optional
        Run ID to use. Default is -1 (the latest run).
    lfc_key : str, optional
        Log fold change key. If provided, will be returned as is.
    score_key : str, optional
        Score key. If provided, will be returned as is unless the default
        value needs to be replaced with a run-specific key.
        
    Returns
    -------
    tuple
        (lfc_key, score_key) with the inferred keys
    """
    inferred_lfc_key = lfc_key
    inferred_score_key = score_key
    
    # If keys already provided, just return them
    if inferred_lfc_key is not None and inferred_score_key is not None:
        return inferred_lfc_key, inferred_score_key
    
    # Get run info from specified run_id - specifically from kompot_de
    from ...anndata.utils import get_run_history
    
    # Get run history, which will handle JSON deserialization
    run_history = get_run_history(adata, "de")
    
    # Get specific run from the history
    adjusted_run_id = run_id
    if run_id < 0 and len(run_history) >= abs(run_id):
        adjusted_run_id = len(run_history) + run_id
    
    if 0 <= adjusted_run_id < len(run_history):
        run_info = run_history[adjusted_run_id]
        run_info["adjusted_run_id"] = adjusted_run_id  # Add this for compatibility
    else:
        run_info = None
    
    # If the run_info is None but a run_id was specified, log this
    if run_info is None and run_id is not None:
        logger.warning(f"Could not find run information for run_id={run_id}, analysis_type=de")
    
    if run_info is not None and 'field_names' in run_info:
        field_names = run_info['field_names']
        adjusted_run_id = run_info.get("adjusted_run_id")
        
        # Get lfc_key from field_names
        if inferred_lfc_key is None and 'mean_lfc_key' in field_names:
            inferred_lfc_key = field_names['mean_lfc_key']
            # Check that column exists
            if inferred_lfc_key not in adata.var.columns:
                inferred_lfc_key = None
                logger.warning(f"Found mean_lfc_key '{inferred_lfc_key}' in run info, but column not in adata.var")
            # Skip validation for now, can be implemented separately if needed
            # In most cases, the run_id/field association is unambiguous
            pass
        
        # Get score_key from field_names
        if inferred_score_key is None and 'mahalanobis_key' in field_names:
            inferred_score_key = field_names['mahalanobis_key']
            # Check that column exists
            if inferred_score_key not in adata.var.columns:
                logger.warning(f"Found mahalanobis_key '{inferred_score_key}' in run info, but column not in adata.var")
                inferred_score_key = None
            # Skip validation for now, can be implemented separately if needed
            # In most cases, the run_id/field association is unambiguous
            pass
    
    # If lfc_key still not found, raise error
    if inferred_lfc_key is None:
        raise ValueError("Could not infer lfc_key from the specified run. Please specify manually.")
    
    return inferred_lfc_key, inferred_score_key


def _infer_da_keys(adata: AnnData, run_id: int = -1, lfc_key: Optional[str] = None, 
                  pval_key: Optional[str] = None):
    """
    Infer differential abundance keys from AnnData object.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object with differential abundance results
    run_id : int, optional
        Run ID to use. Default is -1 (the latest run).
    lfc_key : str, optional
        Log fold change key. If provided, will be returned as is.
    pval_key : str, optional
        P-value key. If provided, will be returned as is.
        
    Returns
    -------
    tuple
        (lfc_key, pval_key) with the inferred keys, and a tuple of (lfc_threshold, pval_threshold)
    """
    inferred_lfc_key = lfc_key
    inferred_pval_key = pval_key
    lfc_threshold = None
    pval_threshold = None
    
    # If both keys already provided, just check for thresholds and return
    if inferred_lfc_key is not None and inferred_pval_key is not None:
        # Get run info to check for thresholds
        run_info = get_run_from_history(adata, run_id, analysis_type="da")
        if run_info is not None and 'params' in run_info:
            params = run_info['params']
            lfc_threshold = params.get('log_fold_change_threshold')
            pval_threshold = params.get('pvalue_threshold')
            
        return inferred_lfc_key, inferred_pval_key, (lfc_threshold, pval_threshold)
    
    # Get run info from specified run_id - specifically from kompot_da
    from ...anndata.utils import get_run_history
    
    # Get run history, which will handle JSON deserialization
    run_history = get_run_history(adata, "da")
    
    # Get specific run from the history
    adjusted_run_id = run_id
    if run_id < 0 and len(run_history) >= abs(run_id):
        adjusted_run_id = len(run_history) + run_id
    
    if 0 <= adjusted_run_id < len(run_history):
        run_info = run_history[adjusted_run_id]
        run_info["adjusted_run_id"] = adjusted_run_id  # Add this for compatibility
    else:
        run_info = None
    
    if run_info is not None:
        # Check for thresholds in params
        if 'params' in run_info:
            params = run_info['params']
            lfc_threshold = params.get('log_fold_change_threshold')
            pval_threshold = params.get('pvalue_threshold')
        
        # Get field names directly from the run_info
        if 'field_names' in run_info:
            field_names = run_info['field_names']
            adjusted_run_id = run_info.get("adjusted_run_id")
            
            # Get lfc_key from field_names
            if inferred_lfc_key is None and 'lfc_key' in field_names:
                inferred_lfc_key = field_names['lfc_key']
                # Check that column exists
                if inferred_lfc_key not in adata.obs.columns:
                    logger.warning(f"Found lfc_key '{inferred_lfc_key}' in run info, but column not in adata.obs")
                    inferred_lfc_key = None
                # Skip validation for now, can be implemented separately if needed
                # In most cases, the run_id/field association is unambiguous
                pass
            
            # Get pval_key from field_names
            if inferred_pval_key is None and 'pval_key' in field_names:
                inferred_pval_key = field_names['pval_key']
                # Check that column exists
                if inferred_pval_key not in adata.obs.columns:
                    logger.warning(f"Found pval_key '{inferred_pval_key}' in run info, but column not in adata.obs")
                    inferred_pval_key = None
                # Skip validation for now, can be implemented separately if needed
                # In most cases, the run_id/field association is unambiguous
                pass
    
    # If keys still not found, raise error
    if inferred_lfc_key is None:
        raise ValueError("Could not infer lfc_key from the specified run. Please specify manually.")
    
    if inferred_pval_key is None:
        raise ValueError("Could not infer pval_key from the specified run. Please specify manually.")
    
    return inferred_lfc_key, inferred_pval_key, (lfc_threshold, pval_threshold)
