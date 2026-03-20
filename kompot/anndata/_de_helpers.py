"""
Private helpers for compute_differential_expression.

These functions decompose the monolithic compute_differential_expression into
focused, testable pieces.  They are anndata-specific (not part of the model
classes) and should not be imported by end users.
"""

import logging
import datetime
import numpy as np
import pandas as pd
from typing import Optional, Union, Dict, Any, List, Tuple
from scipy import sparse

from .utils import check_underrepresentation, apply_cell_filter

logger = logging.getLogger("kompot")


# ---------------------------------------------------------------------------
# 1. Overwrite detection
# ---------------------------------------------------------------------------

def _check_overwrites(
    adata,
    result_key: str,
    field_names: dict,
    all_patterns: dict,
    sample_col: Optional[str],
    overwrite: Optional[bool],
    compute_mahalanobis: bool,
    use_fdr: bool,
    store_additional_stats: bool,
    groups,
    # For parameter matching against previous runs
    groupby: str,
    condition1: str,
    condition2: str,
    obsm_key: str,
    layer: Optional[str],
    ls_factor: float,
):
    """Check for existing results and handle overwrite logic.

    Modifies ``all_patterns`` in-place (adds FDR and group varm keys).
    Returns ``(has_overwrites, existing_fields, prev_run)``.
    """
    # Update all_patterns with FDR fields if needed
    if use_fdr:
        all_patterns["var"].extend([
            field_names["mahalanobis_local_fdr_key"],
            field_names["is_de_key"],
        ])
        if store_additional_stats:
            all_patterns["var"].extend([
                field_names["mahalanobis_pvalue_key"],
                field_names["mahalanobis_tail_fdr_key"],
            ])

    # Update all_patterns with group information if needed
    if groups is not None:
        field_names["has_groups"] = True
        if "varm" not in all_patterns:
            all_patterns["varm"] = []
        all_patterns["varm"].append(field_names["mean_lfc_varm_key"])
        if compute_mahalanobis:
            all_patterns["varm"].append(field_names["mahalanobis_varm_key"])
        all_patterns["varm"] = [k for k in all_patterns["varm"] if k is not None]

    from .utils import detect_output_field_overwrite

    has_overwrites = False
    existing_fields = []
    prev_run = None

    for location, patterns in all_patterns.items():
        has_loc, loc_fields, loc_prev = detect_output_field_overwrite(
            adata=adata,
            result_key=result_key,
            output_patterns=patterns,
            location=location,
            with_sample_suffix=(sample_col is not None),
            sample_suffix="_sample_var" if sample_col is not None else "",
            result_type=f"differential expression ({location})",
            analysis_type="de",
        )
        has_overwrites = has_overwrites or has_loc
        existing_fields.extend(loc_fields)
        if loc_prev is not None:
            prev_run = loc_prev

    if not has_overwrites:
        return

    # Format message
    message = (
        f"Differential expression results with result_key='{result_key}' "
        f"already exist in the dataset."
    )

    current_sample_var = sample_col is not None
    params_match = False

    if prev_run:
        prev_timestamp = prev_run.get("timestamp", "unknown time")
        prev_params = prev_run.get("params", {})
        prev_conditions = (
            f"{prev_params.get('condition1', 'unknown')} to "
            f"{prev_params.get('condition2', 'unknown')}"
        )
        message += (
            f" Previous run was at {prev_timestamp} comparing {prev_conditions}."
        )

        if existing_fields:
            field_list = ", ".join(existing_fields[:5])
            if len(existing_fields) > 5:
                field_list += f" and {len(existing_fields) - 5} more fields"

            prev_sample_var = prev_params.get("use_sample_variance", False)

            # Check parameter match
            params_match = True
            for param_name in [
                "groupby", "condition1", "condition2", "obsm_key", "layer", "ls_factor",
            ]:
                curr_val = locals().get(param_name)
                prev_val = prev_params.get(param_name)
                if curr_val != prev_val:
                    params_match = False
                    logger.debug(
                        f"Parameter mismatch: {param_name} "
                        f"(current: {curr_val}, previous: {prev_val})"
                    )

            if prev_sample_var != current_sample_var:
                if current_sample_var and params_match:
                    message += (
                        f" Fields that will be overwritten: {field_list}. "
                        f"Note: Only fields NOT affected by sample variance (like "
                        f"mean_log_fold_change, imputed data, fold_change) will be "
                        f"overwritten since they don't use the sample variance suffix. "
                        f"These results will likely be identical if other parameters "
                        f"haven't changed."
                    )
                elif current_sample_var:
                    message += (
                        f" Fields that will be overwritten: {field_list}. "
                        f"Note: Only fields NOT affected by sample variance (like "
                        f"mean_log_fold_change, imputed data, fold_change) will be "
                        f"overwritten since they don't use the sample variance suffix."
                    )
                else:
                    message += (
                        f" Fields that will be overwritten: {field_list}. "
                        f"Note: Only fields NOT affected by sample variance will be "
                        f"overwritten since sample variance-specific fields use a "
                        f"different suffix."
                    )
            else:
                message += f" Fields that will be overwritten: {field_list}"

    if overwrite is False:
        message += " Set overwrite=True to overwrite or use a different result_key."
        raise ValueError(message)
    elif overwrite is None:
        # Check for partial rerun
        if prev_run:
            prev_params = prev_run.get("params", {})
            prev_sample_var = prev_params.get("use_sample_variance", False)

            # Re-check params_match for the overwrite-is-None branch
            params_match = True
            for param_name in [
                "groupby", "condition1", "condition2", "obsm_key", "layer", "ls_factor",
            ]:
                curr_val = locals().get(param_name)
                prev_val = prev_params.get(param_name)
                if curr_val != prev_val:
                    params_match = False

            if (
                params_match
                and current_sample_var
                and not prev_sample_var
            ):
                logger.info(
                    message
                    + " This is a partial rerun with sample variance added to a "
                    "previous analysis with matching parameters. "
                    "Set overwrite=False to prevent overwriting or overwrite=True "
                    "to silence this message."
                )
            else:
                logger.warning(
                    message
                    + " Set overwrite=False to prevent overwriting or overwrite=True "
                    "to silence this message."
                )


# ---------------------------------------------------------------------------
# 2. Data extraction
# ---------------------------------------------------------------------------

def _extract_de_data(
    adata,
    groupby: str,
    condition1: str,
    condition2: str,
    obsm_key: str,
    layer: Optional[str],
    genes: Optional[List[str]],
    sample_col: Optional[str],
    cell_filter,
    groups,
    min_cells: int,
    min_percentage: Optional[float],
    check_representation: Optional[bool],
) -> dict:
    """Validate inputs, apply filters, extract arrays from adata.

    Returns a dict with keys:
        X1, X2, expr1, expr2, selected_genes, filter_mask, mask1, mask2,
        condition1_sample_indices, condition2_sample_indices,
        auto_filter, underrep, subset_masks, subset_names
    """
    # Validate obsm_key
    if obsm_key not in adata.obsm:
        error_msg = (
            f"Key '{obsm_key}' not found in adata.obsm. "
            f"Available keys: {list(adata.obsm.keys())}"
        )
        if obsm_key == "DM_EigenVectors":
            error_msg += (
                "\n\nTo compute DM_EigenVectors (diffusion map eigenvectors), "
                "use the Palantir package:\n"
                "```python\n"
                "import palantir\n"
                "palantir.utils.run_diffusion_maps(adata)\n"
                "```\n"
                "See https://github.com/dpeerlab/Palantir for installation.\n\n"
                "Alternatively, specify a different obsm_key that exists in your "
                "dataset, such as 'X_pca'."
            )
        raise ValueError(error_msg)

    if groupby not in adata.obs:
        raise ValueError(
            f"Column '{groupby}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )

    # Condition masks
    mask1 = (adata.obs[groupby] == condition1).values
    mask2 = (adata.obs[groupby] == condition2).values

    if np.sum(mask1) == 0:
        raise ValueError(f"Condition '{condition1}' not found in '{groupby}'.")
    if np.sum(mask2) == 0:
        raise ValueError(f"Condition '{condition2}' not found in '{groupby}'.")

    logger.info(f"Condition 1 ({condition1}): {np.sum(mask1):,} cells")
    logger.info(f"Condition 2 ({condition2}): {np.sum(mask2):,} cells")

    # Underrepresentation check
    underrep = {}
    auto_filter = False
    underrep_filter = None

    if check_representation is True and groups is None:
        raise ValueError(
            "Cannot check underrepresentation if no groups are specified. "
            "Set `check_representation=False` or pass `groups`."
        )

    if groups is not None and check_representation is not False and cell_filter is None:
        logger.info("Checking for underrepresentation on the full dataset")
        underrep_result = check_underrepresentation(
            adata,
            groupby=groupby,
            groups=groups,
            conditions=[condition1, condition2],
            min_cells=min_cells,
            min_percentage=min_percentage,
            warn=(check_representation is None),
            print_summary=False,
        )
        if "__underrepresentation_data" in underrep_result:
            underrep = underrep_result.pop("__underrepresentation_data")

        if check_representation is None and underrep:
            logger.warning(
                "Please pass `check_representation=True` to enable filtering "
                "out these underrepresented groups."
            )
        elif check_representation is True and underrep:
            underrep_filter = underrep_result
            n_groups = len(underrep)
            logger.info(
                f"Found {n_groups:,} groups with underrepresented conditions"
            )
            for group, conditions in underrep.items():
                logger.info(
                    f"  - Group '{group}': Underrepresented conditions: {conditions}"
                )
            logger.info("Automatically applying underrepresentation filter")
            cell_filter = underrep_filter
            auto_filter = True

    # Apply cell filter
    filter_mask, filter_details = apply_cell_filter(
        adata=adata,
        cell_filter=cell_filter,
        groups=groups,
        check_representation=check_representation,
        groupby=groupby,
        conditions=[condition1, condition2],
        min_cells=min_cells,
        min_percentage=min_percentage,
    )

    mask1 = mask1 & filter_mask
    mask2 = mask2 & filter_mask

    if np.sum(mask1) < 2:
        raise ValueError(
            f"After filtering, condition '{condition1}' has fewer than 2 cells "
            f"({np.sum(mask1)}). Consider adjusting your filter criteria."
        )
    if np.sum(mask2) < 2:
        raise ValueError(
            f"After filtering, condition '{condition2}' has fewer than 2 cells "
            f"({np.sum(mask2)}). Consider adjusting your filter criteria."
        )

    if cell_filter is not None or auto_filter:
        logger.info(
            f"After filtering - Condition 1 ({condition1}): {np.sum(mask1):,} cells"
        )
        logger.info(
            f"After filtering - Condition 2 ({condition2}): {np.sum(mask2):,} cells"
        )

    # Extract cell states
    X1 = adata.obsm[obsm_key][mask1]
    X2 = adata.obsm[obsm_key][mask2]

    # Extract expression
    if layer is not None:
        if layer not in adata.layers:
            raise ValueError(
                f"Layer '{layer}' not found in adata.layers. "
                f"Available layers: {list(adata.layers.keys())}"
            )
        expr1 = adata.layers[layer][mask1]
        expr2 = adata.layers[layer][mask2]
    else:
        expr1 = adata.X[mask1]
        expr2 = adata.X[mask2]

    if sparse.issparse(expr1):
        expr1 = expr1.toarray()
    if sparse.issparse(expr2):
        expr2 = expr2.toarray()

    # Filter genes
    if genes is not None:
        if len(genes) == 0:
            raise ValueError("genes list is empty. Provide at least one gene.")
        genes_set = set(genes)
        missing_genes = [g for g in genes_set if g not in adata.var_names]
        if missing_genes:
            raise ValueError(
                f"The following genes were not found in adata.var_names: "
                f"{missing_genes[:10]}"
                + (f"... and {len(missing_genes) - 10} more"
                   if len(missing_genes) > 10 else "")
            )
        selected_genes = [g for g in adata.var_names if g in genes_set]
        gene_indices = [list(adata.var_names).index(g) for g in selected_genes]
        expr1 = expr1[:, gene_indices]
        expr2 = expr2[:, gene_indices]
    else:
        selected_genes = adata.var_names.tolist()

    # Sample indices
    condition1_sample_indices = None
    condition2_sample_indices = None
    if sample_col is not None:
        if sample_col not in adata.obs:
            raise ValueError(
                f"Column '{sample_col}' not found in adata.obs. "
                f"Available columns: {list(adata.obs.columns)}"
            )
        condition1_sample_indices = adata.obs[sample_col][mask1].values
        condition2_sample_indices = adata.obs[sample_col][mask2].values
        logger.info(
            f"Using sample column '{sample_col}' for sample variance estimation"
        )
        logger.info(
            f"Found {len(np.unique(condition1_sample_indices))} unique sample(s) "
            f"in condition 1"
        )
        logger.info(
            f"Found {len(np.unique(condition2_sample_indices))} unique sample(s) "
            f"in condition 2"
        )

    return {
        "X1": X1,
        "X2": X2,
        "expr1": expr1,
        "expr2": expr2,
        "selected_genes": selected_genes,
        "filter_mask": filter_mask,
        "mask1": mask1,
        "mask2": mask2,
        "condition1_sample_indices": condition1_sample_indices,
        "condition2_sample_indices": condition2_sample_indices,
        "auto_filter": auto_filter,
        "underrep": underrep,
    }


# ---------------------------------------------------------------------------
# 3. Landmark resolution
# ---------------------------------------------------------------------------

def _resolve_landmarks(
    adata,
    landmarks: Optional[np.ndarray],
    n_landmarks: Optional[int],
    obsm_key: str,
    result_key: str,
) -> Optional[np.ndarray]:
    """Resolve landmarks from provided array, adata.uns, or None."""
    if landmarks is not None:
        logger.info(f"Using provided landmarks with shape {landmarks.shape}")
        return landmarks

    data_dim = adata.obsm[obsm_key].shape[1]

    # Check result_key first
    if result_key in adata.uns and "landmarks" in adata.uns[result_key]:
        stored = adata.uns[result_key]["landmarks"]
        if stored.shape[1] == data_dim:
            logger.info(
                f"Using stored landmarks from adata.uns['{result_key}']['landmarks'] "
                f"with shape {stored.shape}"
            )
            return stored
        else:
            logger.warning(
                f"Stored landmarks have dimension {stored.shape[1]} but data has "
                f"dimension {data_dim}. Will check for other landmarks."
            )

    # Check default DE key
    storage_key = "kompot_de"
    if (
        storage_key in adata.uns
        and storage_key != result_key
        and "landmarks" in adata.uns[storage_key]
    ):
        other = adata.uns[storage_key]["landmarks"]
        if other.shape[1] == data_dim:
            logger.info(
                f"Reusing stored DE landmarks from adata.uns['{storage_key}']"
                f"['landmarks'] with shape {other.shape}"
            )
            return other
        else:
            logger.warning(
                f"Other stored DE landmarks have dimension {other.shape[1]} but "
                f"data has dimension {data_dim}. Will check for other landmarks."
            )

    # Check DA landmarks
    if (
        "kompot_da" in adata.uns
        and "landmarks" in adata.uns["kompot_da"]
    ):
        da = adata.uns["kompot_da"]["landmarks"]
        if da.shape[1] == data_dim:
            logger.info(
                f"Reusing differential abundance landmarks from "
                f"adata.uns['kompot_da']['landmarks'] with shape {da.shape}"
            )
            return da
        else:
            logger.warning(
                f"DA landmarks have dimension {da.shape[1]} but data has "
                f"dimension {data_dim}. Computing new landmarks."
            )

    return None


# ---------------------------------------------------------------------------
# 4. Null gene augmentation
# ---------------------------------------------------------------------------

def _augment_with_null_genes(
    adata,
    expr1: np.ndarray,
    expr2: np.ndarray,
    mask1: np.ndarray,
    mask2: np.ndarray,
    selected_genes: list,
    null_genes,
    null_seed: Optional[int],
    compute_mahalanobis: bool,
    layer: Optional[str],
) -> Tuple[np.ndarray, np.ndarray, list, list, bool]:
    """Augment expression matrices with shuffled null genes for FDR.

    Returns (expr1, expr2, expanded_genes, null_gene_indices, use_fdr).
    """
    from .fdr_utils import prepare_null_genes, generate_shuffled_expression

    use_fdr = (
        null_genes is not None
        and null_genes != 0
        and compute_mahalanobis
    )

    null_gene_indices = []

    if null_genes is not None and null_genes != 0 and not compute_mahalanobis:
        logger.warning(
            "FDR calculation requires compute_mahalanobis=True. "
            "Skipping FDR calculation."
        )

    if not use_fdr:
        return expr1, expr2, selected_genes, null_gene_indices, False

    logger.debug(
        f"Preparing null distribution with null_genes={null_genes}, "
        f"null_seed={null_seed}"
    )

    available_genes = adata.var_names.tolist()
    null_gene_indices, used_replacement = prepare_null_genes(
        null_genes=null_genes,
        available_genes=available_genes,
        null_seed=null_seed,
    )

    if not null_gene_indices:
        return expr1, expr2, selected_genes, null_gene_indices, True

    # Get original expression for null gene columns
    orig_expr1 = adata.X[mask1] if layer is None else adata.layers[layer][mask1]
    orig_expr2 = adata.X[mask2] if layer is None else adata.layers[layer][mask2]
    if sparse.issparse(orig_expr1):
        orig_expr1 = orig_expr1.toarray()
    if sparse.issparse(orig_expr2):
        orig_expr2 = orig_expr2.toarray()

    null_expr1, null_expr2 = generate_shuffled_expression(
        expr1=orig_expr1,
        expr2=orig_expr2,
        null_gene_indices=null_gene_indices,
        null_seed=null_seed,
    )

    logger.debug(
        f"Generated shuffled expression for {len(null_gene_indices)} null genes"
        + (" (with replacement)" if used_replacement else "")
    )

    expr1 = np.hstack([expr1, null_expr1])
    expr2 = np.hstack([expr2, null_expr2])

    null_gene_names = [
        f"NULL_{i}_{adata.var_names[idx]}"
        for i, idx in enumerate(null_gene_indices)
    ]
    expanded_genes = selected_genes + null_gene_names

    return expr1, expr2, expanded_genes, null_gene_indices, True


# ---------------------------------------------------------------------------
# 5. FDR computation (split null results)
# ---------------------------------------------------------------------------

def _compute_fdr(
    expression_results: dict,
    selected_genes: list,
    expanded_genes: list,
    null_gene_indices: list,
    fdr_threshold: float,
) -> dict:
    """Compute FDR statistics and strip null genes from expression_results.

    Modifies ``expression_results`` in-place (removes null gene columns).
    Returns the fdr_results dict (empty if nothing computed).
    """
    from .fdr_utils import compute_fdr_statistics, annotate_differential_genes

    n_real = len(selected_genes)

    if "mahalanobis_distances" not in expression_results:
        logger.warning(
            "Mahalanobis distances not computed - cannot perform FDR calculation"
        )
        return {}

    all_mahalanobis = expression_results["mahalanobis_distances"]
    real_mahalanobis = all_mahalanobis[:n_real]
    null_mahalanobis = all_mahalanobis[n_real:]

    pvalues, local_fdr, tail_fdr, is_significant = compute_fdr_statistics(
        real_mahalanobis=real_mahalanobis,
        null_mahalanobis=null_mahalanobis,
        fdr_threshold=fdr_threshold,
    )

    de_annotation, summary_stats = annotate_differential_genes(
        fdr_values=local_fdr,
        mahalanobis_distances=real_mahalanobis,
        gene_names=selected_genes,
        fdr_threshold=fdr_threshold,
    )

    fdr_results = {
        "pvalues": pvalues,
        "local_fdr_values": local_fdr,
        "tail_fdr_values": tail_fdr,
        "is_significant": is_significant,
        "de_annotation": de_annotation,
        "summary_stats": summary_stats,
    }

    logger.info(
        f"FDR analysis complete: {summary_stats['n_significant']}/"
        f"{summary_stats['n_total']} genes significantly DE at "
        f"FDR < {fdr_threshold}"
    )
    if summary_stats["n_significant"] > 0:
        logger.info(
            f"Mahalanobis distance threshold for FDR < {fdr_threshold}: "
            f"{summary_stats['min_significant_mahalanobis']:.4f}"
        )

    # Strip null genes from expression_results
    for key in [
        "mean_log_fold_change",
        "condition1_imputed", "condition2_imputed",
        "fold_change", "fold_change_zscores",
        "condition1_std", "condition2_std",
    ]:
        if key in expression_results:
            if key in (
                "condition1_imputed", "condition2_imputed",
                "fold_change", "fold_change_zscores",
                "condition1_std", "condition2_std",
            ):
                expression_results[key] = expression_results[key][:, :n_real]
            else:
                expression_results[key] = expression_results[key][:n_real]

    if "mahalanobis_distances" in expression_results:
        expression_results["mahalanobis_distances"] = real_mahalanobis
    if "ptp" in expression_results:
        expression_results["ptp"] = expression_results["ptp"][:n_real]

    return fdr_results


# ---------------------------------------------------------------------------
# 6. Store results in adata
# ---------------------------------------------------------------------------

def _ensure_1d(arr, name, expected_len, logger):
    """Coerce an array to 1-D and validate length."""
    if isinstance(arr, list):
        arr = np.array(arr)
    if arr.ndim > 1:
        logger.warning(f"{name} has shape {arr.shape}, flattening to 1D.")
        if arr.shape[0] < arr.shape[1]:
            arr = arr[0]
        else:
            arr = arr[:, 0]
    if len(arr) != expected_len:
        logger.warning(
            f"{name} length {len(arr)} doesn't match expected {expected_len}. "
            f"Reshaping."
        )
        if len(arr) < expected_len:
            arr = np.concatenate([arr, np.full(expected_len - len(arr), np.nan)])
        else:
            arr = arr[:expected_len]
    return arr


def _add_var_column(new_var_columns, key, values, selected_genes, adata):
    """Add a var column, creating or updating as appropriate."""
    if key in adata.var:
        new_var_columns[key] = pd.Series(values, index=selected_genes)
    else:
        new_var_columns[key] = pd.Series(np.nan, index=adata.var_names)
        new_var_columns[key].loc[selected_genes] = values


def _add_var_column_bool(new_var_columns, key, values, selected_genes, adata):
    """Add a boolean var column."""
    if key in adata.var:
        new_var_columns[key] = pd.Series(values, index=selected_genes)
    else:
        new_var_columns[key] = pd.Series(False, index=adata.var_names)
        new_var_columns[key].loc[selected_genes] = values


def _store_de_results(
    adata,
    expression_results: dict,
    fdr_results: dict,
    field_names: dict,
    selected_genes: list,
    filter_mask: np.ndarray,
    sample_col: Optional[str],
    compute_mahalanobis: bool,
    use_fdr: bool,
    store_additional_stats: bool,
):
    """Store gene-level and cell-level DE results into adata.

    Writes to ``adata.var``, ``adata.layers``, and ``adata.obs``.
    """
    n_selected = len(selected_genes)
    new_var_columns: Dict[str, pd.Series] = {}

    # --- var columns ---
    if compute_mahalanobis:
        mahal = _ensure_1d(
            expression_results["mahalanobis_distances"],
            "mahalanobis_distances", n_selected, logger,
        )
        _add_var_column(
            new_var_columns, field_names["mahalanobis_key"],
            mahal, selected_genes, adata,
        )

    if compute_mahalanobis and "ptp" in expression_results and store_additional_stats:
        ptp = _ensure_1d(
            expression_results["ptp"], "ptp", n_selected, logger,
        )
        _add_var_column(
            new_var_columns, field_names["ptp_key"],
            ptp, selected_genes, adata,
        )

    mean_lfc = _ensure_1d(
        expression_results["mean_log_fold_change"],
        "mean_log_fold_change", n_selected, logger,
    )
    _add_var_column(
        new_var_columns, field_names["mean_lfc_key"],
        mean_lfc, selected_genes, adata,
    )

    # FDR var columns
    if fdr_results and use_fdr:
        if store_additional_stats:
            _add_var_column(
                new_var_columns, field_names["mahalanobis_pvalue_key"],
                fdr_results["pvalues"], selected_genes, adata,
            )
        _add_var_column(
            new_var_columns, field_names["mahalanobis_local_fdr_key"],
            fdr_results["local_fdr_values"], selected_genes, adata,
        )
        if store_additional_stats:
            _add_var_column(
                new_var_columns, field_names["mahalanobis_tail_fdr_key"],
                fdr_results["tail_fdr_values"], selected_genes, adata,
            )
        _add_var_column_bool(
            new_var_columns, field_names["is_de_key"],
            fdr_results["is_significant"], selected_genes, adata,
        )

    # Flush var columns
    if new_var_columns:
        logger.debug(f"Adding {len(new_var_columns)} columns to adata.var at once")
        existing_cols = {}
        new_cols = {}
        for col, values in new_var_columns.items():
            if col in adata.var.columns:
                existing_cols[col] = values
            else:
                new_cols[col] = values

        for col, values in existing_cols.items():
            if len(values.index) == n_selected:
                adata.var.loc[values.index, col] = values
            else:
                adata.var[col] = values

        if new_cols:
            new_df = pd.DataFrame(new_cols, index=adata.var.index)
            adata.var = pd.concat([adata.var, new_df], axis=1)

    # --- layers ---
    condition1_imputed = np.array(expression_results["condition1_imputed"])
    condition2_imputed = np.array(expression_results["condition2_imputed"])
    fold_change = np.array(expression_results["fold_change"])
    fold_change_zscores = np.array(expression_results["fold_change_zscores"])
    condition1_std = np.array(expression_results["condition1_std"])
    condition2_std = np.array(expression_results["condition2_std"])

    imputed1_key = field_names["imputed_key_1"]
    imputed2_key = field_names["imputed_key_2"]
    fold_change_key = field_names["fold_change_key"]

    subset_of_genes = n_selected < len(adata.var_names)

    if subset_of_genes:
        _store_layers_subset(
            adata, selected_genes, filter_mask,
            condition1_imputed, condition2_imputed,
            fold_change, fold_change_zscores,
            condition1_std, condition2_std,
            field_names, sample_col, store_additional_stats,
        )
    else:
        _store_layers_full(
            adata, filter_mask,
            condition1_imputed, condition2_imputed,
            fold_change, fold_change_zscores,
            condition1_std, condition2_std,
            field_names, sample_col, store_additional_stats,
        )


def _init_layer(adata, key, use_sparse):
    """Initialize a layer if it doesn't exist."""
    if key not in adata.layers:
        if use_sparse:
            adata.layers[key] = sparse.csr_matrix(adata.shape)
        else:
            adata.layers[key] = np.zeros(adata.shape)


def _store_layers_subset(
    adata, selected_genes, filter_mask,
    cond1_imputed, cond2_imputed,
    fold_change, fold_change_zscores,
    cond1_std, cond2_std,
    field_names, sample_col, store_additional_stats,
):
    """Store layers when only a subset of genes was analysed."""
    imputed1_key = field_names["imputed_key_1"]
    imputed2_key = field_names["imputed_key_2"]
    fold_change_key = field_names["fold_change_key"]
    use_sparse = len(selected_genes) < len(adata.var_names)

    _init_layer(adata, imputed1_key, use_sparse)
    _init_layer(adata, imputed2_key, use_sparse)
    _init_layer(adata, fold_change_key, use_sparse)
    if store_additional_stats:
        _init_layer(adata, field_names["fold_change_zscores_key"], use_sparse)

    gene_indices = np.array(
        [adata.var_names.get_loc(g) for g in selected_genes]
    )

    if sample_col is not None:
        # With sample variance — std stored as layers
        _init_layer(adata, field_names["std_key_1"], use_sparse)
        _init_layer(adata, field_names["std_key_2"], use_sparse)

        layers_to_convert = [
            imputed1_key, imputed2_key, fold_change_key,
            field_names["std_key_1"], field_names["std_key_2"],
        ]
        lil_layers, needs_conversion = _to_lil(adata, layers_to_convert)

        if needs_conversion:
            pct = 100 * len(gene_indices) / adata.shape[1]
            sparse_names = [
                k for k in layers_to_convert if sparse.issparse(adata.layers[k])
            ]
            logger.info(
                f"Updating {len(gene_indices)} genes ({pct:.1f}% of total) in "
                f"existing sparse layers: {', '.join(sparse_names)}. "
                f"Temporarily converting CSR->LIL->CSR to avoid slow per-gene "
                f"modifications."
                + (f" Consider using dense layers if you typically analyze "
                   f">{pct:.0f}% of genes." if pct > 50 else "")
            )

        for i, gi in enumerate(gene_indices):
            lil_layers[imputed1_key][:, gi] = cond1_imputed[:, i]
            lil_layers[imputed2_key][:, gi] = cond2_imputed[:, i]
            lil_layers[fold_change_key][:, gi] = fold_change[:, i]
            lil_layers[field_names["std_key_1"]][:, gi] = cond1_std[:, i]
            lil_layers[field_names["std_key_2"]][:, gi] = cond2_std[:, i]

        _from_lil(adata, layers_to_convert, lil_layers, needs_conversion)
    else:
        # Without sample variance — std stored as obs columns
        adata.obs[field_names["std_key_1"]] = np.nan
        adata.obs[field_names["std_key_2"]] = np.nan
        adata.obs.loc[filter_mask, field_names["std_key_1"]] = cond1_std[:, 0]
        adata.obs.loc[filter_mask, field_names["std_key_2"]] = cond2_std[:, 0]

        layers_to_convert = [imputed1_key, imputed2_key, fold_change_key]
        if store_additional_stats:
            layers_to_convert.append(field_names["fold_change_zscores_key"])

        lil_layers, needs_conversion = _to_lil(adata, layers_to_convert)

        if needs_conversion:
            pct = 100 * len(selected_genes) / adata.shape[1]
            sparse_names = [
                k for k in layers_to_convert if sparse.issparse(adata.layers[k])
            ]
            logger.info(
                f"Updating {len(selected_genes)} genes ({pct:.1f}% of total) in "
                f"existing sparse layers: {', '.join(sparse_names)}. "
                f"Temporarily converting CSR->LIL->CSR."
                + (f" Consider using dense layers if you typically analyze "
                   f">{pct:.0f}% of genes." if pct > 50 else "")
            )

        for i, gene in enumerate(selected_genes):
            gi = list(adata.var_names).index(gene)
            lil_layers[imputed1_key][:, gi] = cond1_imputed[:, i]
            lil_layers[imputed2_key][:, gi] = cond2_imputed[:, i]
            lil_layers[fold_change_key][:, gi] = fold_change[:, i]
            if store_additional_stats:
                lil_layers[field_names["fold_change_zscores_key"]][:, gi] = (
                    fold_change_zscores[:, i]
                )

        _from_lil(adata, layers_to_convert, lil_layers, needs_conversion)


def _store_layers_full(
    adata, filter_mask,
    cond1_imputed, cond2_imputed,
    fold_change, fold_change_zscores,
    cond1_std, cond2_std,
    field_names, sample_col, store_additional_stats,
):
    """Store layers when all genes were analysed (no subsetting)."""
    imputed1_key = field_names["imputed_key_1"]
    imputed2_key = field_names["imputed_key_2"]
    fold_change_key = field_names["fold_change_key"]

    if sample_col is not None:
        adata.layers[imputed1_key] = cond1_imputed
        adata.layers[imputed2_key] = cond2_imputed
        adata.layers[fold_change_key] = fold_change
        if store_additional_stats:
            adata.layers[field_names["fold_change_zscores_key"]] = fold_change_zscores
        adata.layers[field_names["std_key_1"]] = cond1_std
        adata.layers[field_names["std_key_2"]] = cond2_std
    else:
        adata.obs[field_names["std_key_1"]] = np.nan
        adata.obs[field_names["std_key_2"]] = np.nan
        adata.obs.loc[filter_mask, field_names["std_key_1"]] = cond1_std[:, 0]
        adata.obs.loc[filter_mask, field_names["std_key_2"]] = cond2_std[:, 0]

        filtered_indices = np.where(filter_mask)[0]

        imputed1_layer = np.full(adata.shape, np.nan)
        imputed2_layer = np.full(adata.shape, np.nan)
        fc_layer = np.full(adata.shape, np.nan)
        fc_z_layer = np.full(adata.shape, np.nan)

        imputed1_layer[filtered_indices] = cond1_imputed
        imputed2_layer[filtered_indices] = cond2_imputed
        fc_layer[filtered_indices] = fold_change
        fc_z_layer[filtered_indices] = fold_change_zscores

        adata.layers[imputed1_key] = imputed1_layer
        adata.layers[imputed2_key] = imputed2_layer
        adata.layers[fold_change_key] = fc_layer
        if store_additional_stats:
            adata.layers[field_names["fold_change_zscores_key"]] = fc_z_layer


def _to_lil(adata, layer_keys):
    """Convert sparse layers to LIL format for efficient column assignment."""
    lil = {}
    needs_conversion = False
    for key in layer_keys:
        if sparse.issparse(adata.layers[key]):
            needs_conversion = True
            lil[key] = adata.layers[key].tolil()
        else:
            lil[key] = adata.layers[key]
    return lil, needs_conversion


def _from_lil(adata, layer_keys, lil_layers, needs_conversion):
    """Convert LIL layers back to CSR."""
    if not needs_conversion:
        return
    for key in layer_keys:
        if sparse.issparse(lil_layers[key]):
            adata.layers[key] = lil_layers[key].tocsr()
        else:
            adata.layers[key] = lil_layers[key]


# ---------------------------------------------------------------------------
# 7. Group (subset) analysis
# ---------------------------------------------------------------------------

def _compute_group_results(
    adata,
    diff_expression,
    groups,
    filter_mask: np.ndarray,
    field_names: dict,
    selected_genes: list,
    expanded_genes: list,
    null_gene_indices: list,
    obsm_key: str,
    compute_mahalanobis: bool,
    use_fdr: bool,
    store_additional_stats: bool,
    fdr_threshold: float,
    progress: bool,
) -> dict:
    """Run per-group subset predictions and store results in adata.varm.

    Returns group_results dict.
    """
    from .utils import parse_groups
    from .fdr_utils import compute_fdr_statistics

    logger.info(
        "Processing group-based subsetting for differential expression analysis"
    )

    subset_masks, subset_names = parse_groups(adata, groups)

    # Apply filter to subset masks
    filtered_subset_masks = {}
    for name, mask in subset_masks.items():
        filtered_subset_masks[name] = mask & filter_mask
    subset_masks = filtered_subset_masks

    if not subset_masks:
        logger.warning("No valid subsets found based on the 'groups' parameter")
        return {}, subset_masks, subset_names

    logger.info(
        f"Identified {len(subset_names):,} subset(s): {', '.join(subset_names)}"
    )

    # Determine varm keys
    varm_keys = [field_names["mean_lfc_varm_key"]]
    if compute_mahalanobis:
        varm_keys.append(field_names["mahalanobis_varm_key"])
    if use_fdr and null_gene_indices and compute_mahalanobis:
        varm_keys.append(f"{field_names['mahalanobis_local_fdr_key']}_groups")
        varm_keys.append(f"{field_names['is_de_key']}_groups")
    if compute_mahalanobis and store_additional_stats:
        varm_keys.append(f"{field_names['ptp_key']}_groups")
    varm_keys = [k for k in varm_keys if k is not None]

    # Initialize varm DataFrames
    for varm_key in varm_keys:
        if varm_key not in adata.varm:
            adata.varm[varm_key] = pd.DataFrame(
                0.0, index=adata.var_names, columns=subset_names
            )
            logger.debug(
                f"Initialized {varm_key} in adata.varm with columns for all "
                f"{len(subset_names):,} subsets"
            )

    group_results = {}
    n_real = len(selected_genes)

    for subset_name, mask in subset_masks.items():
        if np.sum(mask) < 2:
            logger.warning(
                f"Subset '{subset_name}' has fewer than 2 cells "
                f"({np.sum(mask)}). Skipping analysis."
            )
            continue

        logger.info(
            f"Running differential expression analysis on subset "
            f"'{subset_name}' with {np.sum(mask):,} cells"
        )

        X_subset = adata.obsm[obsm_key][mask]

        # Handle landmarks for subsets
        use_subset_landmarks = False
        subset_landmarks = None
        if (
            hasattr(diff_expression, "computed_landmarks")
            and diff_expression.computed_landmarks is not None
        ):
            n_lm = diff_expression.computed_landmarks.shape[0]
            if np.sum(mask) > n_lm:
                logger.info(
                    f"Subset '{subset_name}' has {np.sum(mask):,} cells, "
                    f"more than the {n_lm:,} landmarks. Computing new landmarks."
                )
                from mellon.parameters import compute_landmarks
                subset_landmarks = compute_landmarks(
                    X_subset,
                    gp_type="fixed",
                    n_landmarks=n_lm,
                    random_state=diff_expression.random_state,
                )
                use_subset_landmarks = True
            else:
                logger.debug(
                    f"Disabling landmarks for subset '{subset_name}' with "
                    f"{np.sum(mask):,} cells"
                )

        if use_subset_landmarks:
            subset_results = diff_expression.predict(
                X_subset,
                compute_mahalanobis=compute_mahalanobis,
                progress=progress,
                landmarks_override=subset_landmarks,
            )
        else:
            subset_results = diff_expression.predict(
                X_subset,
                compute_mahalanobis=compute_mahalanobis,
                progress=progress,
                use_landmarks=False,
            )

        group_results[subset_name] = subset_results

        # Store mean_lfc and mahalanobis in varm
        for metric, varm_key in [
            ("mean_log_fold_change", field_names["mean_lfc_varm_key"]),
            ("mahalanobis_distances", field_names["mahalanobis_varm_key"]),
        ]:
            if metric not in subset_results:
                continue
            if metric == "mahalanobis_distances" and not compute_mahalanobis:
                continue

            vals = subset_results[metric]
            if isinstance(vals, np.ndarray) and vals.ndim == 2:
                vals = vals[:, 0]
            # Strip null genes
            if len(vals) == len(expanded_genes):
                vals = vals[:n_real]
            elif len(vals) != n_real:
                logger.warning(
                    f"Subset {subset_name} {metric} length {len(vals)} "
                    f"doesn't match expected. Reshaping."
                )
                if len(vals) < n_real:
                    vals = np.concatenate(
                        [vals, np.full(n_real - len(vals), np.nan)]
                    )
                else:
                    vals = vals[:n_real]

            full_series = pd.Series(np.nan, index=adata.var_names)
            full_series.loc[selected_genes] = np.array(vals)
            adata.varm[varm_key][subset_name] = full_series

        # Group-wise FDR
        if use_fdr and null_gene_indices and compute_mahalanobis:
            if "mahalanobis_distances" in subset_results:
                all_sub_mahal = subset_results["mahalanobis_distances"]
                if len(all_sub_mahal) == len(expanded_genes):
                    sub_real = all_sub_mahal[:n_real]
                    sub_null = all_sub_mahal[n_real:]

                    sub_pv, sub_lfdr, sub_tfdr, sub_sig = compute_fdr_statistics(
                        real_mahalanobis=sub_real,
                        null_mahalanobis=sub_null,
                        fdr_threshold=fdr_threshold,
                    )

                    lfdr_varm = f"{field_names['mahalanobis_local_fdr_key']}_groups"
                    isde_varm = f"{field_names['is_de_key']}_groups"

                    s = pd.Series(np.nan, index=adata.var_names)
                    s.loc[selected_genes] = sub_lfdr
                    adata.varm[lfdr_varm][subset_name] = s

                    s = pd.Series(False, index=adata.var_names)
                    s.loc[selected_genes] = sub_sig
                    adata.varm[isde_varm][subset_name] = s

                    n_de = np.sum(sub_sig)
                    logger.info(
                        f"Group '{subset_name}': {n_de}/{n_real} genes "
                        f"significantly DE at FDR < {fdr_threshold}"
                    )

        # Group-wise ptp
        if compute_mahalanobis and store_additional_stats and "ptp" in subset_results:
            sub_ptp = subset_results["ptp"]
            if len(sub_ptp) == len(expanded_genes):
                sub_ptp = sub_ptp[:n_real]
            elif len(sub_ptp) != n_real:
                sub_ptp = sub_ptp[:n_real]

            ptp_varm = f"{field_names['ptp_key']}_groups"
            s = pd.Series(np.nan, index=adata.var_names)
            s.loc[selected_genes] = sub_ptp
            adata.varm[ptp_varm][subset_name] = s

    logger.info("Group-specific data stored in adata.varm matrices")
    return group_results, subset_masks, subset_names


# ---------------------------------------------------------------------------
# 8. Posterior covariance storage
# ---------------------------------------------------------------------------

def _store_posterior_covariance(
    adata,
    diff_expression,
    X_for_prediction: np.ndarray,
    filter_mask: np.ndarray,
    field_names: dict,
    condition1: str,
    condition2: str,
):
    """Compute and store the posterior covariance matrix in adata.obsp."""
    try:
        cov1 = diff_expression.function_predictor1.covariance(
            X_for_prediction, diag=False
        )
        cov2 = diff_expression.function_predictor2.covariance(
            X_for_prediction, diag=False
        )
        posterior_covariance = cov1 + cov2

        posterior_cov_key = field_names["posterior_covariance_key"]
        n_cells = adata.shape[0]
        full_covariance = np.zeros((n_cells, n_cells))

        filtered_indices = np.where(filter_mask)[0]
        n_filtered = len(filtered_indices)

        if posterior_covariance.shape != (n_filtered, n_filtered):
            logger.warning(
                f"Shape mismatch: posterior_covariance "
                f"{posterior_covariance.shape} vs expected "
                f"({n_filtered}, {n_filtered})"
            )
            for i, row_idx in enumerate(filtered_indices):
                for j, col_idx in enumerate(filtered_indices):
                    full_covariance[row_idx, col_idx] = posterior_covariance[i, j]
        else:
            for i, row_idx in enumerate(filtered_indices):
                full_covariance[row_idx, filtered_indices] = posterior_covariance[i]

        adata.obsp[posterior_cov_key] = full_covariance
        logger.info(
            f"Stored posterior covariance matrix in "
            f"adata.obsp['{posterior_cov_key}']"
        )
        return posterior_cov_key
    except Exception as e:
        logger.error(
            f"Failed to compute and store posterior covariance matrix: {e}"
        )
        logger.error("Posterior covariance matrix will not be stored in obsp.")
        return None


# ---------------------------------------------------------------------------
# 9. Landmark storage
# ---------------------------------------------------------------------------

def _store_landmarks(
    adata,
    diff_expression,
    result_key: str,
    store_landmarks: bool,
):
    """Store computed landmarks in adata.uns if requested."""
    if not (
        hasattr(diff_expression, "computed_landmarks")
        and diff_expression.computed_landmarks is not None
    ):
        logger.debug(
            "No computed landmarks found to store. Check if landmarks were "
            "pre-computed or if n_landmarks is set correctly."
        )
        return

    if result_key not in adata.uns:
        adata.uns[result_key] = {}

    lm = diff_expression.computed_landmarks
    landmarks_info = {
        "shape": str(lm.shape),
        "dtype": str(lm.dtype),
        "source": "computed",
        "n_landmarks": lm.shape[0],
        "timestamp": datetime.datetime.now().isoformat(),
    }

    adata.uns[result_key]["landmarks_info"] = landmarks_info

    storage_key = "kompot_de"
    if storage_key not in adata.uns:
        adata.uns[storage_key] = {}
    adata.uns[storage_key]["landmarks_info"] = landmarks_info.copy()

    if store_landmarks:
        adata.uns[result_key]["landmarks"] = lm
        logger.info(
            f"Stored landmarks in adata.uns['{result_key}']['landmarks'] "
            f"with shape {lm.shape} for future reuse"
        )
    else:
        logger.debug(
            "Landmark storage skipped (store_landmarks=False). Compute with "
            "store_landmarks=True to enable landmark reuse."
        )


# ---------------------------------------------------------------------------
# 10. Run info / metadata recording
# ---------------------------------------------------------------------------

def _build_field_mapping(
    field_names: dict,
    condition1: str,
    condition2: str,
    sample_col: Optional[str],
    compute_mahalanobis: bool,
    use_fdr: bool,
    fdr_results: dict,
    store_additional_stats: bool,
    fdr_threshold: float,
):
    """Build the field-to-location mapping dictionary."""
    fm = {
        field_names["mean_lfc_key"]: {
            "location": "var",
            "type": "mean_log_fold_change",
            "description": "Mean log fold change values",
        },
        field_names["imputed_key_1"]: {
            "location": "layers",
            "type": "imputed",
            "description": f"Imputed expression for {condition1}",
        },
        field_names["imputed_key_2"]: {
            "location": "layers",
            "type": "imputed",
            "description": f"Imputed expression for {condition2}",
        },
        field_names["fold_change_key"]: {
            "location": "layers",
            "type": "fold_change",
            "description": "Log fold change for each cell and gene",
        },
    }

    if store_additional_stats:
        fm[field_names["fold_change_zscores_key"]] = {
            "location": "layers",
            "type": "fold_change_zscores",
            "description": (
                f"Z-scores of log fold changes accounting for uncertainty"
                f"{' and sample variance' if sample_col is not None else ''}"
            ),
        }

    # Std fields
    if sample_col is not None:
        for key, cond in [
            (field_names["std_key_1"], condition1),
            (field_names["std_key_2"], condition2),
        ]:
            fm[key] = {
                "location": "layers",
                "type": "std_with_sample_var",
                "description": (
                    f"Posterior standard deviation of imputed expression for "
                    f"{cond} (with sample variance)"
                ),
            }
    else:
        for key, cond in [
            (field_names["std_key_1"], condition1),
            (field_names["std_key_2"], condition2),
        ]:
            fm[key] = {
                "location": "obs",
                "type": "std",
                "description": (
                    f"Posterior standard deviation of imputed expression for "
                    f"{cond} (same for all genes)"
                ),
            }

    if compute_mahalanobis:
        fm[field_names["mahalanobis_key"]] = {
            "location": "var",
            "type": "mahalanobis",
            "description": "Mahalanobis distances",
        }
        if store_additional_stats:
            fm[field_names["ptp_key"]] = {
                "location": "var",
                "type": "ptp",
                "description": (
                    "Posterior tail probability from chi-squared distribution"
                ),
            }

    if fdr_results and use_fdr:
        if store_additional_stats:
            fm[field_names["mahalanobis_pvalue_key"]] = {
                "location": "var",
                "type": "mahalanobis_pvalue",
                "description": "P-values from empirical null distribution",
            }
        fm[field_names["mahalanobis_local_fdr_key"]] = {
            "location": "var",
            "type": "mahalanobis_local_fdr",
            "description": (
                "Local FDR values using empirical null estimation "
                "similar to R's fdrtool"
            ),
        }
        if store_additional_stats:
            fm[field_names["mahalanobis_tail_fdr_key"]] = {
                "location": "var",
                "type": "mahalanobis_tail_fdr",
                "description": (
                    "Tail-based FDR values using Benjamini-Hochberg correction"
                ),
            }
        fm[field_names["is_de_key"]] = {
            "location": "var",
            "type": "is_de",
            "description": (
                f"Boolean indicator of differential expression at "
                f"local FDR < {fdr_threshold}"
            ),
        }

    return fm


def _add_group_field_mapping(
    field_mapping: dict,
    adata,
    field_names: dict,
    subset_names: list,
    compute_mahalanobis: bool,
    use_fdr: bool,
    null_gene_indices: list,
    store_additional_stats: bool,
):
    """Add group-related entries to the field mapping."""
    if field_names["mean_lfc_varm_key"] in adata.varm:
        field_mapping[field_names["mean_lfc_varm_key"]] = {
            "location": "varm",
            "type": "mean_log_fold_change",
            "description": "Mean log fold change values for all subsets",
            "contains_subsets": subset_names,
        }
    if compute_mahalanobis and field_names["mahalanobis_varm_key"] in adata.varm:
        field_mapping[field_names["mahalanobis_varm_key"]] = {
            "location": "varm",
            "type": "mahalanobis",
            "description": "Mahalanobis distances for all subsets",
            "contains_subsets": subset_names,
        }
    if use_fdr and null_gene_indices and compute_mahalanobis:
        lfdr_k = f"{field_names['mahalanobis_local_fdr_key']}_groups"
        isde_k = f"{field_names['is_de_key']}_groups"
        if lfdr_k in adata.varm:
            field_mapping[lfdr_k] = {
                "location": "varm",
                "type": "local_fdr",
                "description": "Local FDR values for all subsets",
                "contains_subsets": subset_names,
            }
        if isde_k in adata.varm:
            field_mapping[isde_k] = {
                "location": "varm",
                "type": "is_de",
                "description": (
                    "Differential expression significance for all subsets"
                ),
                "contains_subsets": subset_names,
            }
    if compute_mahalanobis and store_additional_stats:
        ptp_k = f"{field_names['ptp_key']}_groups"
        if ptp_k in adata.varm:
            field_mapping[ptp_k] = {
                "location": "varm",
                "type": "ptp",
                "description": "Peak-to-peak values for all subsets",
                "contains_subsets": subset_names,
            }


def _record_de_run_info(
    adata,
    diff_expression,
    field_names: dict,
    field_mapping: dict,
    all_patterns: dict,
    selected_genes: list,
    condition1: str,
    condition2: str,
    sample_col: Optional[str],
    compute_mahalanobis: bool,
    use_fdr: bool,
    fdr_results: dict,
    fdr_threshold: float,
    store_additional_stats: bool,
    store_landmarks: bool,
    store_posterior_covariance: bool,
    store_arrays_on_disk: Optional[bool],
    result_key: str,
    groups,
    subset_names: list,
    subset_masks: dict,
    auto_filter: bool,
    underrep: dict,
    posterior_cov_key: Optional[str],
    params_dict: Optional[dict] = None,
):
    """Record run info, field tracking, and run history in adata.uns."""
    from .utils import (
        get_environment_info,
        set_json_metadata,
        append_to_run_history,
        get_run_history,
        get_json_metadata,
    )

    current_timestamp = datetime.datetime.now().isoformat()
    storage_key = "kompot_de"

    # Storage stats
    storage_stats = None
    if (
        store_arrays_on_disk
        and hasattr(diff_expression, "_disk_storage")
        and diff_expression._disk_storage is not None
    ):
        try:
            storage_human, storage_bytes = (
                diff_expression._disk_storage.total_storage_used
            )
            storage_stats = {
                "total_disk_usage": storage_human,
                "disk_usage_bytes": storage_bytes,
                "array_count": len(diff_expression._disk_storage.array_registry),
            }
        except Exception as e:
            logger.warning(f"Failed to get disk storage statistics: {e}")

    current_run_info = {
        "timestamp": current_timestamp,
        "function": "compute_differential_expression",
        "result_key": result_key,
        "analysis_type": "de",
        "lfc_key": field_names["mean_lfc_key"],
        "mahalanobis_key": (
            field_names["mahalanobis_key"] if compute_mahalanobis else None
        ),
        "ptp_key": (
            field_names["ptp_key"] if compute_mahalanobis else None
        ),
        "fdr_keys": (
            {
                "pvalue_key": field_names.get("mahalanobis_pvalue_key"),
                "local_fdr_key": field_names.get("mahalanobis_local_fdr_key"),
                "tail_fdr_key": field_names.get("mahalanobis_tail_fdr_key"),
                "is_de_key": field_names.get("is_de_key"),
            }
            if use_fdr
            else None
        ),
        "fdr_results": fdr_results.get("summary_stats") if fdr_results else None,
        "imputed_layer_keys": {
            "condition1": field_names["imputed_key_1"],
            "condition2": field_names["imputed_key_2"],
            "fold_change": field_names["fold_change_key"],
        },
        "field_names": field_names,
        "field_mapping": field_mapping,
        "uses_sample_variance": sample_col is not None,
        "memory_analysis": None,
        "storage_stats": storage_stats,
        "underrepresentation": underrep,
        "auto_filtered": auto_filter,
        "has_groups": groups is not None,
        "params": params_dict or {},
        "environment": get_environment_info(),
    }

    # Posterior covariance key
    if posterior_cov_key is not None:
        current_run_info["posterior_covariance_key"] = posterior_cov_key

    # Groups summary
    if groups is not None and subset_names:
        current_run_info["subset_names"] = subset_names
        current_run_info["has_groups"] = True

        groups_summary = {
            "count": len(subset_names),
            "names": subset_names,
            "description": str(type(groups).__name__),
            "cells_per_group": {},
        }

        _params = params_dict or {}
        groupby = _params.get("groupby", "")
        for group_name, mask in subset_masks.items():
            cell_count = int(np.sum(mask))
            pct = (cell_count / adata.n_obs) * 100
            group_info = {"count": cell_count, "percentage": float(pct)}
            if groupby in adata.obs:
                cond_counts = {}
                for cond in [condition1, condition2]:
                    cmask = (adata.obs[groupby] == cond).values
                    cc = int(np.sum(mask & cmask))
                    cp = (cc / cell_count * 100) if cell_count > 0 else 0
                    cond_counts[cond] = {"count": cc, "percentage": float(cp)}
                group_info["conditions"] = cond_counts
            groups_summary["cells_per_group"][group_name] = group_info

        current_run_info["groups_summary"] = groups_summary
        current_run_info["varm_keys"] = {
            "mean_lfc": field_names["mean_lfc_varm_key"],
            "mahalanobis": field_names["mahalanobis_varm_key"],
        }

    # Early store
    set_json_metadata(adata, f"{storage_key}.last_run_info", current_run_info.copy())

    # Run history
    if "run_history" not in adata.uns.get(storage_key, {}):
        if storage_key not in adata.uns:
            adata.uns[storage_key] = {}
        adata.uns[storage_key]["run_history"] = "[]"
        new_run_id = 0
    else:
        current_history = get_run_history(adata, "de")
        new_run_id = len(current_history)

    logger.info(f"This run will have `run_id={new_run_id}`.")
    append_to_run_history(adata, current_run_info, "de")
    set_json_metadata(adata, f"{storage_key}.last_run_info", current_run_info)

    # Field tracking
    anndata_field_tracking = {}
    for location, patterns in all_patterns.items():
        if location not in anndata_field_tracking:
            anndata_field_tracking[location] = {}
        for pattern in patterns:
            anndata_field_tracking[location][pattern] = new_run_id

    if "uns" not in anndata_field_tracking:
        anndata_field_tracking["uns"] = {}
    anndata_field_tracking["uns"][result_key] = new_run_id

    if groups is not None and "varm" in all_patterns:
        if "varm" not in anndata_field_tracking:
            anndata_field_tracking["varm"] = {}
        if field_names["mean_lfc_varm_key"] in adata.varm:
            anndata_field_tracking["varm"][
                field_names["mean_lfc_varm_key"]
            ] = new_run_id
        if (
            compute_mahalanobis
            and field_names["mahalanobis_varm_key"] in adata.varm
        ):
            anndata_field_tracking["varm"][
                field_names["mahalanobis_varm_key"]
            ] = new_run_id

    if "anndata_fields" not in adata.uns.get(storage_key, {}):
        set_json_metadata(
            adata, f"{storage_key}.anndata_fields", anndata_field_tracking
        )
    else:
        existing = get_json_metadata(adata, f"{storage_key}.anndata_fields")
        if existing is None:
            existing = {}
        for section, fields in anndata_field_tracking.items():
            if section not in existing:
                existing[section] = {}
            for field, rid in fields.items():
                existing[section][field] = rid
        set_json_metadata(adata, f"{storage_key}.anndata_fields", existing)

    return current_run_info
