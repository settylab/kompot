"""Robust field inference utilities for plotting functions.

This module provides safe field name inference that relies on run info
and provides proper warnings for overwrites and fallbacks.
"""

import logging
from typing import Optional, Tuple, Dict, Any, List
from anndata import AnnData

logger = logging.getLogger("kompot")


def infer_fields_from_run_info(
    adata: AnnData,
    analysis_type: str,
    run_id: int = -1,
    condition1: Optional[str] = None,
    condition2: Optional[str] = None,
    result_key: Optional[str] = None,
    required_fields: Optional[List[str]] = None,
    strict: bool = True
) -> Dict[str, Optional[str]]:
    """
    Safely infer field names from run info with proper warnings.

    This function prioritizes run info over pattern matching and provides
    comprehensive warnings about overwrites, fallbacks, and ambiguous cases.

    Parameters
    ----------
    adata : AnnData
        AnnData object with analysis results and run history
    analysis_type : str
        Type of analysis ('da' or 'de')
    run_id : int, optional
        Run ID to use. Default is -1 (latest run)
    condition1 : str, optional
        First condition name to match against run info
    condition2 : str, optional
        Second condition name to match against run info
    result_key : str, optional
        Specific result key to look for
    required_fields : list of str, optional
        List of field types that must be found (e.g., ['lfc_key', 'score_key'])
    strict : bool, optional
        If True, raise errors on ambiguous cases. If False, use warnings.

    Returns
    -------
    dict
        Dictionary mapping field types to their inferred column names.
        Missing fields will have None values.

    Raises
    ------
    ValueError
        If strict=True and required fields cannot be uniquely identified.
    """
    from ..anndata.utils import get_run_from_history, get_run_history
    from ..plot.volcano import _extract_conditions_from_key

    if required_fields is None:
        if analysis_type == "da":
            required_fields = ["lfc_key", "direction_key"]
        elif analysis_type == "de":
            required_fields = ["mean_lfc_key", "mahalanobis_key"]
        else:
            required_fields = []

    inferred_fields = {field: None for field in required_fields}
    warnings_issued = []

    # Step 1: Try to get field names from run info
    try:
        run_info = get_run_from_history(adata, run_id, analysis_type=analysis_type)
    except Exception as e:
        logger.warning(f"Error accessing run history: {e}")
        run_info = None

    if run_info is not None:
        logger.info(f"Found {analysis_type.upper()} run info for run_id={run_id}")

        # Check if user-specified conditions match run info
        run_conditions = None
        if 'params' in run_info:
            params = run_info['params']
            if 'condition1' in params and 'condition2' in params:
                run_conditions = (params['condition1'], params['condition2'])

                # Warn if user conditions don't match run info
                if condition1 is not None and condition2 is not None:
                    if (condition1, condition2) != run_conditions:
                        warning = f"User-specified conditions ({condition1}, {condition2}) don't match run info conditions {run_conditions}"
                        logger.warning(warning)
                        warnings_issued.append(warning)

                        if strict:
                            raise ValueError(f"Condition mismatch: {warning}")

                # Use run info conditions if not specified by user
                if condition1 is None:
                    condition1 = run_conditions[0]
                if condition2 is None:
                    condition2 = run_conditions[1]

        # Extract field names from run info
        if "field_names" in run_info:
            field_names = run_info["field_names"]

            # Map the field names to our required fields
            field_mapping = {}
            if analysis_type == "da":
                field_mapping = {
                    "lfc_key": field_names.get("lfc_key"),
                    "direction_key": field_names.get("direction_key"),
                    "zscore_key": field_names.get("zscore_key"),
                    "ptp_key": field_names.get("ptp_key"),
                    "density_key_1": field_names.get("density_key_1"),
                    "density_key_2": field_names.get("density_key_2")
                }
            elif analysis_type == "de":
                field_mapping = {
                    "mean_lfc_key": field_names.get("mean_lfc_key"),
                    "mahalanobis_key": field_names.get("mahalanobis_key"),
                    "ptp_key": field_names.get("ptp_key"),
                    "is_de_key": field_names.get("is_de_key"),
                    "mahalanobis_pvalue_key": field_names.get("mahalanobis_pvalue_key"),
                    "mahalanobis_local_fdr_key": field_names.get("mahalanobis_local_fdr_key"),
                    "mahalanobis_tail_fdr_key": field_names.get("mahalanobis_tail_fdr_key")
                }

            # Check if the fields exist in the data
            for field_type, field_name in field_mapping.items():
                if field_name is not None and field_type in required_fields:
                    # For DA fields, check adata.obs; for DE fields, check adata.var
                    data_section = adata.obs if analysis_type == "da" else adata.var

                    if field_name in data_section.columns:
                        inferred_fields[field_type] = field_name
                        logger.info(f"Found {field_type}='{field_name}' from run info")
                    else:
                        warning = f"Run info specifies {field_type}='{field_name}' but column not found in data"
                        logger.warning(warning)
                        warnings_issued.append(warning)
    else:
        warning = f"No {analysis_type.upper()} run info found for run_id={run_id}"
        logger.warning(warning)
        warnings_issued.append(warning)

    # Step 2: For missing fields, try intelligent fallback with proper warnings
    missing_fields = [field for field in required_fields if inferred_fields[field] is None]

    if missing_fields:
        logger.warning(f"Attempting fallback inference for missing fields: {missing_fields}")

        # Get the appropriate data section
        data_section = adata.obs if analysis_type == "da" else adata.var

        for field_type in missing_fields:
            inferred_field = _fallback_field_inference(
                data_section,
                field_type,
                analysis_type,
                condition1,
                condition2,
                result_key,
                strict
            )

            if inferred_field is not None:
                inferred_fields[field_type] = inferred_field
                warning = f"Fallback inference: using {field_type}='{inferred_field}'"
                logger.warning(warning)
                warnings_issued.append(warning)

    # Step 3: Check for overwrite warnings
    _check_for_overwrites(adata, analysis_type, inferred_fields, warnings_issued)

    # Step 4: Validate that required fields were found
    still_missing = [field for field in required_fields if inferred_fields[field] is None]

    if still_missing:
        error_msg = f"Could not infer required fields: {still_missing}"
        if warnings_issued:
            error_msg += f". Warnings: {'; '.join(warnings_issued)}"

        if strict:
            raise ValueError(error_msg)
        else:
            logger.error(error_msg)

    # Log summary
    found_fields = {k: v for k, v in inferred_fields.items() if v is not None}
    if found_fields:
        logger.info(f"Successfully inferred fields: {found_fields}")

    if warnings_issued:
        logger.warning(f"Field inference completed with {len(warnings_issued)} warnings")

    return inferred_fields


def _fallback_field_inference(
    data_section,
    field_type: str,
    analysis_type: str,
    condition1: Optional[str],
    condition2: Optional[str],
    result_key: Optional[str],
    strict: bool
) -> Optional[str]:
    """Fallback field inference using pattern matching."""
    from ..plot.volcano import _extract_conditions_from_key

    # Define search patterns for different field types
    patterns = {
        "lfc_key": ["lfc", "log_fold_change", "fold_change"],
        "direction_key": ["direction"],
        "mean_lfc_key": ["mean_lfc", "lfc", "log_fold_change", "fold_change"],
        "mahalanobis_key": ["mahalanobis", "score"],
        "ptp_key": ["ptp"],
        "is_de_key": ["is_de", "significant"],
        "zscore_key": ["zscore", "z_score"],
        "density_key_1": ["log_density"],
        "density_key_2": ["log_density"]
    }

    if field_type not in patterns:
        return None

    # Find candidate columns
    candidates = []
    for col in data_section.columns:
        col_lower = col.lower()
        # Check if any pattern matches
        if any(pattern in col_lower for pattern in patterns[field_type]):
            # Prefer kompot columns
            if "kompot" in col_lower:
                candidates.append(col)

    if not candidates:
        return None

    if len(candidates) == 1:
        return candidates[0]

    # Multiple candidates - try to filter by conditions
    if condition1 and condition2:
        condition_filtered = []
        for col in candidates:
            if f"{condition1}_to_{condition2}" in col:
                condition_filtered.append(col)

        if len(condition_filtered) == 1:
            return condition_filtered[0]
        elif len(condition_filtered) > 1:
            candidates = condition_filtered

    # Multiple candidates remain - try to filter by result_key
    if result_key:
        key_filtered = [col for col in candidates if col.startswith(result_key)]
        if len(key_filtered) == 1:
            return key_filtered[0]
        elif len(key_filtered) > 1:
            candidates = key_filtered

    # Still multiple candidates - log the ambiguity
    conditions_info = []
    for col in candidates:
        conditions = _extract_conditions_from_key(col)
        if conditions:
            conditions_info.append(f"{col} ({conditions[0]} → {conditions[1]})")
        else:
            conditions_info.append(col)

    logger.warning(f"Multiple candidates for {field_type}: {conditions_info}")

    if strict:
        return None  # Let the caller handle the error
    else:
        # Return the first candidate with a warning
        return candidates[0]


def _check_for_overwrites(
    adata: AnnData,
    analysis_type: str,
    inferred_fields: Dict[str, Optional[str]],
    warnings_issued: List[str]
) -> None:
    """
    Check for potential data overwrites using the robust field tracking system.

    This function is consistent with the RunInfo._check_overwritten_fields() approach
    and uses the same field tracking mechanism for detecting overwrites.
    """
    from ..anndata.utils.field_tracking import validate_field_run_id, get_run_from_history
    from ..anndata.utils.json_utils import from_json_string

    storage_key = f"kompot_{analysis_type}"

    # Check if we have field tracking information
    if (storage_key not in adata.uns or
        'anndata_fields' not in adata.uns[storage_key]):
        return  # No tracking information available

    # Get field tracking data
    try:
        tracking = adata.uns[storage_key]['anndata_fields']
        if isinstance(tracking, str):
            tracking = from_json_string(tracking)
    except Exception as e:
        logger.warning(f"Error accessing field tracking data: {e}")
        return

    # Determine the data location for this analysis type
    location = "obs" if analysis_type == "da" else "var"

    if location not in tracking:
        return  # No fields tracked for this location

    location_tracking = tracking[location]
    if isinstance(location_tracking, str):
        try:
            location_tracking = from_json_string(location_tracking)
        except Exception:
            return

    # Check each inferred field for overwrites
    for field_type, field_name in inferred_fields.items():
        if field_name is None or field_name not in location_tracking:
            continue

        # Get the run that currently owns this field
        current_owner_run = location_tracking[field_name]

        # Try to get the latest run to compare
        try:
            latest_run_info = get_run_from_history(adata, run_id=-1, analysis_type=analysis_type)
            if latest_run_info and 'adjusted_run_id' in latest_run_info:
                latest_run_id = latest_run_info['adjusted_run_id']
            else:
                continue  # Can't determine latest run
        except Exception:
            continue

        # Check if the field was written by a different run than the latest
        if current_owner_run != latest_run_id:
            warning = (f"Field '{field_name}' was last written by run {current_owner_run}, "
                      f"but current context expects run {latest_run_id}. "
                      f"The field may have been overwritten.")
            logger.warning(warning)
            warnings_issued.append(warning)

        # Additional check: Count potential writers from run history to detect multiple runs
        # that could have written to the same field (indicating overwrite potential)
        potential_writers = _count_potential_field_writers(
            adata, analysis_type, field_type, field_name
        )

        if potential_writers > 1:
            warning = (f"Field '{field_name}' has been written by {potential_writers} different runs, "
                      f"indicating potential overwrites")
            logger.warning(warning)
            warnings_issued.append(warning)


def _count_potential_field_writers(
    adata: AnnData,
    analysis_type: str,
    field_type: str,
    field_name: str
) -> int:
    """
    Count how many runs could have written to a specific field.

    This provides compatibility with the original detection logic while
    using the robust field tracking system.
    """
    from ..anndata.utils import get_run_history

    run_history = get_run_history(adata, analysis_type)
    if not run_history:
        return 0

    potential_writers = 0
    for run_info in run_history:
        if "field_names" in run_info:
            field_names = run_info["field_names"]
            if field_names.get(field_type) == field_name:
                potential_writers += 1

    return potential_writers


def get_comparison_specific_fields(
    adata: AnnData,
    analysis_type: str,
    condition1: str,
    condition2: str,
    run_id: int = -1,
    result_key: Optional[str] = None
) -> Dict[str, str]:
    """
    Get field names for a specific comparison with validation.

    This function ensures that we get the exact fields for the requested
    comparison and warns if data might be from a different comparison.

    Parameters
    ----------
    adata : AnnData
        AnnData object with analysis results
    analysis_type : str
        Type of analysis ('da' or 'de')
    condition1 : str
        First condition name
    condition2 : str
        Second condition name
    run_id : int, optional
        Run ID to use. Default is -1 (latest run)
    result_key : str, optional
        Specific result key to look for

    Returns
    -------
    dict
        Dictionary mapping field types to column names for the specific comparison

    Raises
    ------
    ValueError
        If the requested comparison cannot be found or validated
    """
    # Use strict inference to ensure we get the right comparison
    fields = infer_fields_from_run_info(
        adata=adata,
        analysis_type=analysis_type,
        run_id=run_id,
        condition1=condition1,
        condition2=condition2,
        result_key=result_key,
        strict=True
    )

    # Validate that the fields actually correspond to the requested comparison
    validated_fields = {}
    for field_type, field_name in fields.items():
        if field_name is not None:
            # Check that the field name contains the expected comparison
            expected_pattern = f"{condition1}_to_{condition2}"
            if expected_pattern in field_name:
                validated_fields[field_type] = field_name
            else:
                raise ValueError(
                    f"Field '{field_name}' does not match expected comparison "
                    f"{condition1} → {condition2}"
                )

    logger.info(f"Validated fields for {condition1} → {condition2}: {validated_fields}")
    return validated_fields