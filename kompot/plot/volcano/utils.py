"""Utility functions for volcano plots."""

import numpy as np
from typing import Optional, Tuple
from anndata import AnnData
import logging
from kompot.anndata.utils.field_tracking import get_run_from_history

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
    Infer differential expression keys from AnnData object using robust field inference.

    This function uses the robust field inference system with overwrite detection
    and comprehensive warnings for safer field inference.

    Parameters
    ----------
    adata : AnnData
        AnnData object with differential expression results
    run_id : int, optional
        Run ID to use. Default is -1 (the latest run).
    lfc_key : str, optional
        Log fold change key. If provided, will be returned as is.
    score_key : str, optional
        Score key. If provided, will be returned as is.

    Returns
    -------
    tuple
        (lfc_key, score_key) with the inferred keys
    """
    # If both keys already provided, just return them
    if lfc_key is not None and score_key is not None:
        return lfc_key, score_key

    # Use the robust field inference system
    from ..field_inference import infer_fields_from_run_info

    # Define required fields for DE analysis
    required_fields = []
    if lfc_key is None:
        required_fields.append("mean_lfc_key")
    if score_key is None:
        required_fields.append("mahalanobis_key")

    # If no inference needed, return provided values
    if not required_fields:
        return lfc_key, score_key

    try:
        # Use robust field inference with overwrite detection and warnings
        inferred_fields = infer_fields_from_run_info(
            adata=adata,
            analysis_type="de",
            run_id=run_id,
            required_fields=required_fields,
            strict=False  # Allow fallback inference with warnings
        )

        # Extract the inferred fields
        inferred_lfc_key = lfc_key if lfc_key is not None else inferred_fields.get("mean_lfc_key")
        inferred_score_key = score_key if score_key is not None else inferred_fields.get("mahalanobis_key")

        # Validate that required fields were found
        if inferred_lfc_key is None:
            raise ValueError(
                f"Could not infer mean_lfc_key from run_id={run_id}. "
                f"Please specify lfc_key manually or check run history."
            )

        if inferred_score_key is None:
            logger.warning(
                f"Could not infer mahalanobis_key from run_id={run_id}. "
                f"Score-based functionality may be limited."
            )

        return inferred_lfc_key, inferred_score_key

    except Exception as e:
        # Fallback to manual error with helpful message
        error_msg = (f"Failed to infer DE keys from run_id={run_id}: {e}. "
                    f"Please check run history or specify keys manually.")
        raise ValueError(error_msg) from e


def _infer_da_keys(adata: AnnData, run_id: int = -1, lfc_key: Optional[str] = None,
                  ptp_key: Optional[str] = None):
    """
    Infer differential abundance keys from AnnData object using robust field inference.

    This function uses the robust field inference system with overwrite detection
    and comprehensive warnings for safer field inference.

    Parameters
    ----------
    adata : AnnData
        AnnData object with differential abundance results
    run_id : int, optional
        Run ID to use. Default is -1 (the latest run).
    lfc_key : str, optional
        Log fold change key. If provided, will be returned as is.
    ptp_key : str, optional
        PTP (Posterior Tail Probability) key. If provided, will be returned as is.
        Posterior Tail Probability is a significance measure score similar to p-value.

    Returns
    -------
    tuple
        (lfc_key, ptp_key) with the inferred keys, and a tuple of (lfc_threshold, ptp_threshold)
    """
    # Initialize thresholds
    lfc_threshold = None
    ptp_threshold = None

    # Get run info for thresholds (always needed)
    try:
        run_info = get_run_from_history(adata, run_id, analysis_type="da")
        if run_info is not None and 'params' in run_info:
            params = run_info['params']
            lfc_threshold = params.get('log_fold_change_threshold')
            ptp_threshold = params.get('ptp_threshold')
    except Exception:
        # Continue without thresholds if run info access fails
        pass

    # If both keys already provided, just return with thresholds
    if lfc_key is not None and ptp_key is not None:
        return lfc_key, ptp_key, (lfc_threshold, ptp_threshold)

    # Use the robust field inference system
    from ..field_inference import infer_fields_from_run_info

    # Define required fields for DA analysis
    required_fields = []
    if lfc_key is None:
        required_fields.append("lfc_key")
    if ptp_key is None:
        required_fields.append("ptp_key")

    # If no inference needed, return provided values
    if not required_fields:
        return lfc_key, ptp_key, (lfc_threshold, ptp_threshold)

    try:
        # Use robust field inference with overwrite detection and warnings
        inferred_fields = infer_fields_from_run_info(
            adata=adata,
            analysis_type="da",
            run_id=run_id,
            required_fields=required_fields,
            strict=False  # Allow fallback inference with warnings
        )

        # Extract the inferred fields
        inferred_lfc_key = lfc_key if lfc_key is not None else inferred_fields.get("lfc_key")
        inferred_ptp_key = ptp_key if ptp_key is not None else inferred_fields.get("ptp_key")

        # Validate that required fields were found
        if inferred_lfc_key is None:
            raise ValueError(
                f"Could not infer lfc_key from run_id={run_id}. "
                f"Please specify lfc_key manually or check run history."
            )

        if inferred_ptp_key is None:
            logger.warning(
                f"Could not infer ptp_key from run_id={run_id}. "
                f"Significance-based functionality may be limited."
            )

        return inferred_lfc_key, inferred_ptp_key, (lfc_threshold, ptp_threshold)

    except Exception as e:
        # Fallback to manual error with helpful message
        error_msg = (f"Failed to infer DA keys from run_id={run_id}: {e}. "
                    f"Please check run history or specify keys manually.")
        raise ValueError(error_msg) from e
