"""Field tracking utilities for AnnData objects."""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Union, Tuple, Callable
from anndata import AnnData
import logging
import copy

from .json_utils import from_json_string, get_json_metadata, set_json_metadata

logger = logging.getLogger("kompot")


def get_run_history(adata: AnnData, analysis_type: str = "da") -> List[Dict[str, Any]]:
    """
    Get the run history for a specific analysis type, deserializing from JSON if needed.
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object containing the run history.
    analysis_type : str, optional
        The analysis type ("da" or "de"), by default "da"
        
    Returns
    -------
    List[Dict[str, Any]]
        The deserialized run history, or an empty list if not found.
    """
    storage_key = f"kompot_{analysis_type}"
    
    # Check if the key exists in adata.uns
    if storage_key not in adata.uns or "run_history" not in adata.uns[storage_key]:
        return []
    
    # Get the run history - this will be deserialized by get_json_metadata
    run_history = get_json_metadata(adata, f"{storage_key}.run_history")
    
    if run_history is None:
        return []
    
    # Ensure run_history is a list
    if not isinstance(run_history, list):
        if isinstance(run_history, str):
            try:
                run_history = from_json_string(run_history)
                if not isinstance(run_history, list):
                    logger.warning(f"Parsed run_history is not a list for {analysis_type}: {type(run_history)}")
                    return []
            except Exception:
                logger.warning(f"Failed to parse run_history as JSON for {analysis_type}")
                return []
        else:
            logger.warning(f"Unexpected type for run_history: {type(run_history)}")
            return []
    
    # Ensure each item in the list is properly deserialized
    result = []
    for item in run_history:
        if isinstance(item, str):
            try:
                item = from_json_string(item)
            except Exception:
                logger.warning(f"Failed to parse run history item as JSON: {item[:50]}...")
                # Skip invalid items
                continue
        
        if not isinstance(item, dict):
            logger.warning(f"Run history item is not a dictionary: {type(item)}")
            continue
            
        result.append(item)
        
    return result


def append_to_run_history(adata: AnnData, run_info: Dict[str, Any], analysis_type: str = "da") -> bool:
    """
    Append a new run info entry to the run history, serializing to JSON.
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object to update.
    run_info : Dict[str, Any]
        The run info to append.
    analysis_type : str, optional
        The analysis type ("da" or "de"), by default "da"
        
    Returns
    -------
    bool
        True if successful, False otherwise.
    """
    storage_key = f"kompot_{analysis_type}"
    
    # Initialize uns key if needed
    if storage_key not in adata.uns:
        adata.uns[storage_key] = {}
    
    # Get current run history (will be deserialized if needed)
    current_history = get_run_history(adata, analysis_type)
    
    # Create a new list with the current history and the new run info
    updated_history = current_history + [run_info]
    
    # Store back as JSON string
    return set_json_metadata(adata, f"{storage_key}.run_history", updated_history)


def get_last_run_info(adata: AnnData, analysis_type: str = "da") -> Optional[Dict[str, Any]]:
    """
    Get the last run info, deserializing from JSON if needed.
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object containing the run info.
    analysis_type : str, optional
        The analysis type ("da" or "de"), by default "da"
        
    Returns
    -------
    Optional[Dict[str, Any]]
        The deserialized last run info, or None if not found.
    """
    storage_key = f"kompot_{analysis_type}"
    return get_json_metadata(adata, f"{storage_key}.last_run_info")


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
        # Fields like lfc, log_density are not affected by sample variance
        sample_variance_impacted = ["zscore_key", "ptp_key", "direction_key"]
        
        # Differential abundance field names
        field_names.update({
            "lfc_key": f"{result_key}_{cond1_safe}_to_{cond2_safe}_lfc",
            "zscore_key": f"{result_key}_{cond1_safe}_to_{cond2_safe}_lfc_zscore{suffix}",
            "ptp_key": f"{result_key}_{cond1_safe}_to_{cond2_safe}_neg_log10_lfc_ptp{suffix}",
            "direction_key": f"{result_key}_{cond1_safe}_to_{cond2_safe}_lfc_direction{suffix}",
            "density_key_1": f"{result_key}_{cond1_safe}_log_density",
            "density_key_2": f"{result_key}_{cond2_safe}_log_density"
        })
        field_names["sample_variance_impacted_fields"] = sample_variance_impacted

        # Generate all_patterns for DA - all metrics are in obs
        field_names["all_patterns"] = {
            "obs": [
                field_names["lfc_key"],        # Not impacted by sample variance
                field_names["zscore_key"],     # Impacted by sample variance
                field_names["ptp_key"],       # Impacted by sample variance
                field_names["direction_key"],  # Impacted by sample variance
                field_names["density_key_1"],  # Not impacted by sample variance
                field_names["density_key_2"]   # Not impacted by sample variance
            ]
        }
        
    elif analysis_type == "de":
        # Define which fields are actually impacted by sample variance
        # Fields like mean_lfc, imputed data, fold_change are not affected by sample variance
        sample_variance_impacted = ["mahalanobis_key", "ptp_key", "lfc_std_key", "mahalanobis_varm_key", "std_key_1", "std_key_2", "fold_change_zscores_key"]
        
        # Differential expression field names
        field_names.update({
            "mahalanobis_key": f"{result_key}_{cond1_safe}_to_{cond2_safe}_mahalanobis{suffix}",
            "ptp_key": f"{result_key}_{cond1_safe}_to_{cond2_safe}_ptp{suffix}",
            "mean_lfc_key": f"{result_key}_{cond1_safe}_to_{cond2_safe}_mean_lfc",
            "imputed_key_1": f"{result_key}_{cond1_safe}_imputed",
            "imputed_key_2": f"{result_key}_{cond2_safe}_imputed",
            "fold_change_key": f"{result_key}_{cond1_safe}_to_{cond2_safe}_fold_change",
            "fold_change_zscores_key": f"{result_key}_{cond1_safe}_to_{cond2_safe}_fold_change_zscores{suffix}",
            "std_key_1": f"{result_key}_{cond1_safe}_std",
            "std_key_2": f"{result_key}_{cond2_safe}_std",

            # FDR-related field names with condition names
            "mahalanobis_pvalue_key": f"{result_key}_{cond1_safe}_to_{cond2_safe}_mahalanobis_pvalue{suffix}",
            "mahalanobis_local_fdr_key": f"{result_key}_{cond1_safe}_to_{cond2_safe}_mahalanobis_local_fdr{suffix}",
            "mahalanobis_tail_fdr_key": f"{result_key}_{cond1_safe}_to_{cond2_safe}_mahalanobis_tail_fdr{suffix}",
            "is_de_key": f"{result_key}_{cond1_safe}_to_{cond2_safe}_is_de{suffix}",

            # Add varm field names for group-specific metrics
            "mean_lfc_varm_key": f"{result_key}_{cond1_safe}_to_{cond2_safe}_mean_lfc_groups",
            "mahalanobis_varm_key": f"{result_key}_{cond1_safe}_to_{cond2_safe}_mahalanobis{suffix}_groups",
        })
        field_names["sample_variance_impacted_fields"] = sample_variance_impacted

        # Add posterior covariance key - specific to the condition pair
        field_names["posterior_covariance_key"] = f"{result_key}_{cond1_safe}_to_{cond2_safe}_posterior_covariance"
        
        # Generate all_patterns for DE
        field_names["all_patterns"] = {
            "var": [
                field_names["mahalanobis_key"],      # Impacted by sample variance
                field_names["ptp_key"],        # Impacted by sample variance
                field_names["mean_lfc_key"],         # Not impacted by sample variance
                # FDR fields will be added conditionally below when null genes are used
            ],
            "layers": [
                field_names["imputed_key_1"],        # Not impacted by sample variance
                field_names["imputed_key_2"],        # Not impacted by sample variance
                field_names["fold_change_key"],      # Not impacted by sample variance
                field_names["fold_change_zscores_key"] # Impacted by sample variance
            ],
            "obsp": [
                field_names["posterior_covariance_key"] # Not impacted by sample variance
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
    
    return env_info


def detect_output_field_overwrite(
    adata: AnnData,
    analysis_type: str = None,
    field_names: Dict[str, str] = None,
    overwrite: bool = False,
    result_key: str = None,
    output_patterns: List[str] = None,
    with_sample_suffix: bool = False,
    sample_suffix: str = "",
    result_type: str = None,
    **kwargs  # Accept additional arguments for backward compatibility
) -> Tuple[bool, List[str], Optional[int]]:
    """
    Detect if any output fields would be overwritten by the current operation.
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object to check
    analysis_type : str, optional
        The analysis type ("da" or "de"), by default None
    field_names : Dict[str, str], optional
        Dictionary of field names to check, by default None
    overwrite : bool, optional
        Whether overwriting is permitted, by default False
    result_key : str, optional
        The result key to check, by default None
    output_patterns : List[str], optional
        List of output patterns to check, by default None
    with_sample_suffix : bool, optional
        Whether to use sample suffix, by default False
    sample_suffix : str, optional
        The sample suffix to use, by default ""
    result_type : str, optional
        The type of result, by default None
    
    Returns
    -------
    Tuple[bool, List[str], Optional[int]]
        - Boolean indicating if any fields would be overwritten
        - List of field names that would be overwritten
        - Run ID of previous run if it exists, otherwise None
    """
    # Determine storage key
    if analysis_type is not None:
        storage_key = f"kompot_{analysis_type}"
    elif result_type is not None:
        if "abundance" in result_type.lower():
            storage_key = "kompot_da"
        elif "expression" in result_type.lower():
            storage_key = "kompot_de"
        else:
            raise ValueError(f"Unknown result_type: {result_type}")
    else:
        raise ValueError("Either analysis_type or result_type must be provided")
    
    # Early return if overwrite is allowed
    if overwrite:
        return False, [], None
    
    # Identify fields to check and location
    location = kwargs.get("location", "obs")  # Default to obs if not specified
    
    if output_patterns is not None:
        # If output_patterns is provided, use it
        fields_to_check = output_patterns
    elif field_names is not None:
        # Extract all field patterns from field_names
        all_patterns = field_names.get("all_patterns", {})
        
        # Collect all fields that would be written
        fields_to_check = []
        for loc, fields in all_patterns.items():
            if loc == location:
                fields_to_check.extend(fields)
    else:
        raise ValueError("Either field_names or output_patterns must be provided")
    
    # Initialize result
    overwritten_fields = []
    prev_run = None
    
    # Check for existing fields based on location
    for field in fields_to_check:
        if location == "obs" and field in adata.obs:
            overwritten_fields.append(f"obs.{field}")
        elif location == "var" and field in adata.var:
            overwritten_fields.append(f"var.{field}")
        elif location == "uns" and field in adata.uns:
            overwritten_fields.append(f"uns.{field}")
        elif location == "layers" and field in adata.layers:
            overwritten_fields.append(f"layers.{field}")
        elif location == "obsm" and field in adata.obsm:
            overwritten_fields.append(f"obsm.{field}")
        elif location == "varm" and field in adata.varm:
            overwritten_fields.append(f"varm.{field}")
        elif location == "obsp" and field in adata.obsp:
            overwritten_fields.append(f"obsp.{field}")
    
    # Also check if the result_key is already in use
    if result_key is not None:
        # Check if we have tracking information for this key
        if (storage_key in adata.uns and
            "anndata_fields" in adata.uns[storage_key] and
            "uns" in adata.uns[storage_key]["anndata_fields"]):
            
            # Get tracking information - be sure to deserialize JSON
            from .json_utils import get_json_metadata
            tracking = get_json_metadata(adata, f"{storage_key}.anndata_fields")
            
            # Check if this result_key is being tracked
            if "uns" in tracking and result_key in tracking["uns"]:
                # Get the previous run info
                run_history = get_json_metadata(adata, f"{storage_key}.run_history")
                if run_history and isinstance(run_history, list):
                    # Use the run ID to find the run info
                    run_id = tracking["uns"][result_key]
                    
                    # Adjust for negative indices if needed
                    if run_id < 0:
                        run_id = len(run_history) + run_id
                    
                    # Make sure the run ID is valid
                    if 0 <= run_id < len(run_history):
                        prev_run = run_history[run_id]
    
    return len(overwritten_fields) > 0, overwritten_fields, prev_run


def _sanitize_name(name):
    """
    Sanitize condition names for field names.
    
    Parameters
    ----------
    name : str
        The name to sanitize
        
    Returns
    -------
    str
        Sanitized name
    """
    if name is None:
        return "None"
    return str(name).replace(" ", "_").replace("-", "_").replace(".", "_").replace("/", "_")


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
        "anndata_fields" in adata.uns[storage_key]):
        
        anndata_fields = adata.uns[storage_key]["anndata_fields"]
        
        # Check if anndata_fields is a string (JSON serialized)
        if isinstance(anndata_fields, str):
            anndata_fields = from_json_string(anndata_fields)
        
        # Now check if location exists in the deserialized data
        if location in anndata_fields:
            tracking_info = anndata_fields[location]
            
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
    run_id: int = -1, 
    analysis_type: str = "da",
    history_key: str = None  # Parameter to allow direct access to a specific history key
) -> Optional[Dict[str, Any]]:
    """
    Get a specific run's info from the run history, with support for negative indexing.
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object to get run history from
    run_id : int, optional
        The run ID to retrieve. Negative indices count from the end.
        By default -1 (most recent run)
    analysis_type : str, optional
        The analysis type ("da" or "de"), by default "da"
    history_key : str, optional
        Direct access to a specific history key using dotted notation (e.g., "kompot_de.run_history")
        If provided, this overrides the analysis_type parameter
        
    Returns
    -------
    Optional[Dict[str, Any]]
        The run info dictionary, or None if not found
    """
    # If None is provided for run_id, return None immediately
    if run_id is None:
        return None

    # Define the storage key based on analysis_type or history_key
    if history_key is not None:
        # If a specific history key is provided, use that
        if "." in history_key:
            # Split into main key and subkey for dotted notation
            parts = history_key.split(".", 1)
            storage_key = parts[0]
            history_path = parts[1]
        else:
            # If no dot, assume the key is a direct path to run history
            storage_key = history_key
            history_path = "run_history"
    else:
        # Use standard analysis type approach
        storage_key = f"kompot_{analysis_type}"
        history_path = "run_history"
    
    # Check if storage key exists
    if storage_key not in adata.uns:
        return None
    
    # Get the container that should hold run_history
    container = adata.uns[storage_key]
    
    # Handle case where history_path is nested
    if "." in history_path:
        # Split the path and traverse the nested structure
        path_parts = history_path.split(".")
        current = container
        for part in path_parts[:-1]:  # All but the last part
            if part not in current:
                return None
            current = current[part]
        
        # The last part should point to run_history
        if path_parts[-1] not in current:
            return None
        run_history = current[path_parts[-1]]
    else:
        # Direct access to run_history
        if history_path not in container:
            return None
        run_history = container[history_path]
    
    # Ensure run_history is a list by deserializing if needed
    if isinstance(run_history, str):
        try:
            run_history = from_json_string(run_history)
        except Exception as e:
            logger.warning(f"Failed to parse run_history as JSON: {e}")
            return None
    
    if not isinstance(run_history, list):
        logger.warning(f"Run history is not a list, got {type(run_history)}")
        return None
    
    if not run_history:
        return None
    
    # Handle negative indices (e.g., -1 for most recent)
    if run_id < 0:
        run_id = len(run_history) + run_id
    
    # Validate run_id is within range
    if run_id < 0 or run_id >= len(run_history):
        return None
    
    # Get the specified run
    run_info = run_history[run_id]
    
    # Ensure it's a dictionary
    if not isinstance(run_info, dict):
        if isinstance(run_info, str):
            # Try to parse as JSON
            try:
                run_info = from_json_string(run_info)
                if not isinstance(run_info, dict):
                    logger.warning(f"Expected run_info to be a dict, but got {type(run_info)}")
                    # Use an empty dict instead of None to keep consistent structure
                    run_info = {}
            except Exception as e:
                logger.warning(f"Failed to parse run_info as JSON: {e}")
                # Use an empty dict instead of None
                run_info = {}
        else:
            logger.warning(f"Expected run_info to be a dict, but got {type(run_info)}")
            # Use an empty dict instead of None
            run_info = {}
    
    # Make a copy to avoid modifying the original
    run_info = copy.deepcopy(run_info)
    
    # Add adjusted_run_id (the requested index) but preserve original run_id
    run_info["adjusted_run_id"] = run_id
    # Don't overwrite the original run_id - preserve data integrity
    
    return run_info