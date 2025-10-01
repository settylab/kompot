"""Cleanup utilities for removing large data from AnnData after differential analysis."""

import logging
from typing import Optional, Union, List, Dict, Set
from anndata import AnnData

from .utils.runinfo import RunInfo
from .utils.field_tracking import get_run_from_history
from .utils.json_utils import from_json_string

logger = logging.getLogger("kompot")


def cleanup(
    adata: AnnData,
    run_ids: Optional[Union[int, List[int]]] = None,
    analysis_type: str = 'de',
    keep_layers: Optional[Union[bool, List[str]]] = None,
    keep_var_fields: Optional[Union[bool, List[str]]] = True,
    keep_obs_fields: Optional[Union[bool, List[str]]] = True,
    keep_obsp_fields: Optional[Union[bool, List[str]]] = None,
    keep_varm_fields: Optional[Union[bool, List[str]]] = None,
    inplace: bool = True,
) -> Optional[AnnData]:
    """
    Remove large data (layers, obsp, varm) from differential analysis results.

    This function helps reduce AnnData object size by removing large arrays like
    imputed expression layers, fold change layers, and posterior covariance matrices
    while retaining the statistical results in var/obs columns.

    Parameters
    ----------
    adata : AnnData
        AnnData object with differential analysis results
    run_ids : int, list of int, or None, optional
        Run ID(s) to clean up. Negative indices count from the end.
        - If None (default): Cleans up ALL runs
        - If int: Cleans up single run
        - If list: Cleans up specified runs
    analysis_type : str, default 'de'
        Type of analysis: 'de' for differential expression or 'da' for differential abundance
    keep_layers : bool or list of str, optional
        - If None (default): Remove all layers from specified run(s)
        - If False: Remove all layers from specified run(s)
        - If True: Keep all layers from specified run(s)
        - If list: Keep only the specified layer types

    keep_var_fields : bool or list of str, optional
        - If True (default): Keep all var fields from specified run(s)
        - If False: Remove all var fields from specified run(s)
        - If list: Keep only the specified var field types

    keep_obs_fields : bool or list of str, optional
        - If True (default): Keep all obs fields from specified run(s)
        - If False: Remove all obs fields from specified run(s)
        - If list: Keep only the specified obs field types

    keep_obsp_fields : bool or list of str, optional
        - If None (default): Remove all obsp fields from specified run(s)
        - If False: Remove all obsp fields from specified run(s)
        - If True: Keep all obsp fields from specified run(s)
        - If list: Keep only the specified obsp field types

    keep_varm_fields : bool or list of str, optional
        - If None (default): Remove all varm fields from specified run(s)
        - If False: Remove all varm fields from specified run(s)
        - If True: Keep all varm fields from specified run(s)
        - If list: Keep only the specified varm field types

    inplace : bool, default True
        If True, modify adata in place. If False, return a copy.

    Returns
    -------
    AnnData or None
        If inplace=False, returns modified copy. If inplace=True, returns None.

    Field Types
    -----------
    **Layer field types:**
        - 'imputed': Imputed expression for each condition
        - 'fold_change': Log fold change for each cell and gene
        - 'fold_change_zscores': Z-scores of log fold changes (requires store_additional_stats=True)
        - 'std_with_sample_var': Posterior standard deviations with sample variance

    **Var field types:**
        - 'mean_log_fold_change': Mean log fold change values
        - 'mahalanobis': Mahalanobis distances
        - 'ptp': Posterior tail probability (requires store_additional_stats=True)
        - 'mahalanobis_pvalue': P-values from empirical null (requires store_additional_stats=True)
        - 'mahalanobis_local_fdr': Local FDR values (primary significance measure)
        - 'mahalanobis_tail_fdr': Tail-based FDR values (requires store_additional_stats=True)
        - 'is_de': Boolean indicator of differential expression
        - 'weighted_mean_log_fold_change': Weighted mean log fold change (with differential abundance)

    **Obs field types:**
        - 'std': Posterior standard deviations (without sample variance, same for all genes)

    **Obsp field types:**
        - 'covariance': Posterior covariance matrices for fold changes

    **Varm field types:**
        - 'mean_log_fold_change': Mean log fold change per group (when using groups parameter)
        - 'mahalanobis': Mahalanobis distances per group
        - 'weighted_mean_log_fold_change': Weighted mean log fold change per group

    Examples
    --------
    # Remove all layers from all runs (default behavior)
    cleanup(adata)

    # Remove layers from specific run
    cleanup(adata, run_ids=0)

    # Remove layers from multiple specific runs
    cleanup(adata, run_ids=[0, 2, 5])

    # Keep only fold change layer, remove everything else large
    cleanup(adata, keep_layers=['fold_change'])

    # Remove all layers and obsp covariance matrices
    cleanup(adata, keep_layers=False, keep_obsp_fields=False)

    # Keep only essential statistical fields from run 0
    cleanup(
        adata,
        run_ids=0,
        keep_layers=False,
        keep_var_fields=['mahalanobis', 'mahalanobis_local_fdr', 'is_de', 'mean_log_fold_change'],
        keep_obs_fields=False
    )

    Notes
    -----
    - By default, cleans up ALL runs to maximize space savings
    - By default, keeps all statistical results (var/obs fields) but removes layers
    - Large data typically in: layers (imputed, fold_change), obsp (covariance)
    - This does NOT modify the run history - deleted fields are marked as missing
    - Use RunInfo to check which fields are present vs deleted
    """
    if not inplace:
        adata = adata.copy()

    # Determine which runs to clean up
    storage_key = f"kompot_{analysis_type}"

    # Check if run history exists
    if (storage_key not in adata.uns or
        'run_history' not in adata.uns[storage_key] or
        len(adata.uns[storage_key]['run_history']) == 0):
        logger.warning(f"No run history found for {analysis_type} analysis.")
        return None if not inplace else adata

    # Get total number of runs
    run_history = adata.uns[storage_key]['run_history']
    if isinstance(run_history, str):
        run_history = from_json_string(run_history)
    total_runs = len(run_history)

    # Determine run IDs to process
    if run_ids is None:
        # Default: clean up ALL runs
        run_ids_to_process = list(range(total_runs))
        logger.info(f"Cleaning up all {total_runs} run(s)")
    elif isinstance(run_ids, int):
        # Single run
        run_ids_to_process = [run_ids]
    else:
        # List of runs
        run_ids_to_process = run_ids

    # Process each run
    total_deleted = 0
    for run_id in run_ids_to_process:
        try:
            run_info_obj = RunInfo(adata, run_id=run_id, analysis_type=analysis_type)
        except ValueError as e:
            logger.warning(f"Cannot get run info for run_id={run_id}: {e}")
            continue

        adjusted_run_id = run_info_obj.adjusted_run_id
        run_info = run_info_obj.get_raw_data()
        field_mapping = run_info.get('field_mapping', {})

        # Deserialize field_mapping if needed
        if isinstance(field_mapping, str):
            field_mapping = from_json_string(field_mapping)

        if not field_mapping:
            logger.warning(f"No field_mapping found for run {adjusted_run_id}. Skipping.")
            continue

        # Organize fields by location and type
        fields_by_location: Dict[str, Dict[str, List[str]]] = {
            'layers': {},
            'var': {},
            'obs': {},
            'obsp': {},
            'varm': {},
        }

        for field_name, field_info in field_mapping.items():
            if isinstance(field_info, str):
                field_info = from_json_string(field_info)

            if not isinstance(field_info, dict):
                continue

            location = field_info.get('location')
            field_type = field_info.get('type', 'unknown')

            if location in fields_by_location:
                if field_type not in fields_by_location[location]:
                    fields_by_location[location][field_type] = []
                fields_by_location[location][field_type].append(field_name)

        # Process each location based on user preferences
        deletion_params = {
            'layers': keep_layers,
            'var': keep_var_fields,
            'obs': keep_obs_fields,
            'obsp': keep_obsp_fields,
            'varm': keep_varm_fields,
        }

        deleted_fields: Dict[str, List[str]] = {}

        for location, keep_param in deletion_params.items():
            fields_to_delete = _determine_fields_to_delete(
                fields_by_location[location],
                keep_param
            )

            if fields_to_delete:
                deleted_fields[location] = []
                for field_name in fields_to_delete:
                    if _delete_field(adata, location, field_name):
                        deleted_fields[location].append(field_name)

        # Log summary for this run
        run_deleted = sum(len(fields) for fields in deleted_fields.values())
        if run_deleted > 0:
            logger.info(f"Cleaned up {run_deleted} field(s) from run {adjusted_run_id}:")
            for location, fields in deleted_fields.items():
                if fields:
                    logger.info(f"  {location} ({len(fields)} field(s)):")
                    for field_name in fields:
                        logger.info(f"    - {field_name}")
            total_deleted += run_deleted

    # Final summary
    if total_deleted > 0:
        logger.info(f"Total: Cleaned up {total_deleted} field(s) across {len(run_ids_to_process)} run(s)")
    else:
        logger.info(f"No fields deleted.")

    return None if inplace else adata


def _determine_fields_to_delete(
    fields_by_type: Dict[str, List[str]],
    keep_param: Optional[Union[bool, List[str]]]
) -> List[str]:
    """
    Determine which fields to delete based on user preference.

    Parameters
    ----------
    fields_by_type : dict
        Dictionary mapping field types to lists of field names
    keep_param : bool, list of str, or None
        User's preference for what to keep

    Returns
    -------
    list of str
        List of field names to delete
    """
    all_fields = []
    for field_list in fields_by_type.values():
        all_fields.extend(field_list)

    # If keep_param is None or False, delete everything
    if keep_param is None or keep_param is False:
        return all_fields

    # If keep_param is True, keep everything (delete nothing)
    if keep_param is True:
        return []

    # If keep_param is a list, only keep those types
    if isinstance(keep_param, list):
        fields_to_keep = []
        for field_type in keep_param:
            if field_type in fields_by_type:
                fields_to_keep.extend(fields_by_type[field_type])

        # Delete fields not in the keep list
        return [f for f in all_fields if f not in fields_to_keep]

    # Default: delete nothing
    return []


def _delete_field(adata: AnnData, location: str, field_name: str) -> bool:
    """
    Delete a field from the specified AnnData location.

    Parameters
    ----------
    adata : AnnData
        AnnData object
    location : str
        Location in AnnData (var, obs, layers, obsp, varm)
    field_name : str
        Name of the field to delete

    Returns
    -------
    bool
        True if field was deleted, False if field didn't exist
    """
    try:
        if location == 'var':
            if field_name in adata.var.columns:
                adata.var.drop(columns=[field_name], inplace=True)
                return True
        elif location == 'obs':
            if field_name in adata.obs.columns:
                adata.obs.drop(columns=[field_name], inplace=True)
                return True
        elif location == 'layers':
            if field_name in adata.layers:
                del adata.layers[field_name]
                return True
        elif location == 'obsp':
            if field_name in adata.obsp:
                del adata.obsp[field_name]
                return True
        elif location == 'varm':
            if field_name in adata.varm:
                del adata.varm[field_name]
                return True
    except Exception as e:
        logger.warning(f"Error deleting {location}.{field_name}: {e}")
        return False

    return False


def get_field_status(
    adata: AnnData,
    run_id: Optional[int] = None,
    analysis_type: str = 'de'
) -> Dict[str, Dict[str, Dict[str, bool]]]:
    """
    Get the status of all fields from a differential analysis run.

    Shows which fields are present vs missing/deleted.

    Parameters
    ----------
    adata : AnnData
        AnnData object with differential analysis results
    run_id : int, optional
        Run ID to check. If None, uses most recent run.
    analysis_type : str, default 'de'
        Type of analysis: 'de' or 'da'

    Returns
    -------
    dict
        Nested dictionary with structure:
        {location: {field_type: {field_name: is_present}}}

    Examples
    --------
    >>> status = get_field_status(adata)
    >>> print(status['layers']['imputed'])
    {'result_A_imputed': True, 'result_B_imputed': False}
    """
    try:
        run_info_obj = RunInfo(adata, run_id=run_id, analysis_type=analysis_type)
    except ValueError as e:
        logger.error(f"Cannot get run info: {e}")
        return {}

    run_info = run_info_obj.get_raw_data()
    field_mapping = run_info.get('field_mapping', {})

    # Deserialize field_mapping if needed
    if isinstance(field_mapping, str):
        field_mapping = from_json_string(field_mapping)

    if not field_mapping:
        return {}

    # Build status dictionary
    status: Dict[str, Dict[str, Dict[str, bool]]] = {}

    for field_name, field_info in field_mapping.items():
        if isinstance(field_info, str):
            field_info = from_json_string(field_info)

        if not isinstance(field_info, dict):
            continue

        location = field_info.get('location')
        field_type = field_info.get('type', 'unknown')

        if location not in status:
            status[location] = {}
        if field_type not in status[location]:
            status[location][field_type] = {}

        # Check if field exists
        is_present = _check_field_exists(adata, location, field_name)
        status[location][field_type][field_name] = is_present

    return status


def _check_field_exists(adata: AnnData, location: str, field_name: str) -> bool:
    """Check if a field exists in the specified AnnData location."""
    if location == 'var':
        return field_name in adata.var.columns
    elif location == 'obs':
        return field_name in adata.obs.columns
    elif location == 'layers':
        return field_name in adata.layers
    elif location == 'obsp':
        return field_name in adata.obsp
    elif location == 'varm':
        return field_name in adata.varm
    return False
