"""
Differential expression analysis for AnnData objects.
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional, Union, Dict, Any, List

from ..differential import DifferentialExpression
from .utils import (
    _sanitize_name,
    generate_output_field_names,
    apply_cell_filter,  # re-exported for backward compat (tests may patch this)
)
from ._de_helpers import (
    _check_overwrites,
    _extract_de_data,
    _resolve_landmarks,
    _augment_with_null_genes,
    _compute_fdr,
    _store_de_results,
    _store_landmarks,
    _store_posterior_covariance,
    _compute_group_results,
    _build_field_mapping,
    _add_group_field_mapping,
    _record_de_run_info,
)

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
    sigma: float = 1.0,
    ls: Optional[float] = None,
    ls_factor: float = 10.0,
    compute_mahalanobis: bool = True,
    jit_compile: bool = False,
    eps: float = 1e-8,
    random_state: Optional[int] = None,
    batch_size: int = 100,
    store_arrays_on_disk: Optional[bool] = None,
    disk_storage_dir: Optional[str] = None,
    max_memory_ratio: float = 0.8,
    cell_filter: Optional[Union[str, List[str], Dict[str, Any], List[Dict[str, Any]]]] = None,
    groups: Optional[
        Union[str, Dict[str, Any], List[Dict[str, Any]], pd.Series, np.ndarray, List[np.ndarray]]
    ] = None,
    min_cells: int = 2,
    min_percentage: Optional[float] = None,
    check_representation: Optional[bool] = None,
    copy: bool = False,
    inplace: bool = True,
    result_key: str = "kompot_de",
    overwrite: Optional[bool] = None,
    store_landmarks: bool = False,
    return_full_results: bool = False,
    store_posterior_covariance: bool = False,
    allow_single_condition_variance: bool = False,
    use_empirical_variance: bool = False,
    progress: bool = True,
    null_genes: Union[int, List[int], str, None] = "auto",
    null_seed: Optional[int] = 42,
    fdr_threshold: float = 0.05,
    store_additional_stats: bool = False,
    **function_kwargs,
) -> Union[Dict[str, np.ndarray], Any]:
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
    allow_single_condition_variance : bool, optional
        If True, allows variance estimation with only one condition having multiple samples.
        By default False, which requires both conditions to have multiple samples.
    use_empirical_variance : bool, optional
        Whether to estimate per-gene empirical variance from GP residuals.
        When True, the expression GP is fitted with ``obs_variance=True``,
        which computes leverage-corrected squared residuals and smooths them
        with a second GP to produce an input-dependent noise surface.
        This captures gene-specific heteroscedastic noise without requiring
        biological replicates. By default False.
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
    eps : float, optional
        Small constant for numerical stability in covariance matrices, by default 1e-8.
    random_state : int, optional
        Random seed for reproducible landmark selection when n_landmarks is specified.
    batch_size : int, optional
        Number of cells to process at once during prediction, by default 100.
    store_arrays_on_disk : bool, optional
        Whether to store large arrays on disk instead of in memory, by default None.
    disk_storage_dir : str, optional
        Directory to store arrays on disk.
    max_memory_ratio : float, optional
        Maximum fraction of available memory before triggering disk storage, by default 0.8.
    cell_filter : str, List[str], Dict, List[Dict], optional
        Specification for cells to include in the analysis.
    groups : str, Dict, Dict[str, Dict], List[Dict], pd.Series, np.ndarray, List[np.ndarray], optional
        Specification for subsetting or grouping cells for additional analysis.
    min_cells : int, optional
        Minimum number of cells required for a condition, by default 2.
    min_percentage : float, optional
        Minimum percentage of cells required for a condition within each group.
    check_representation : None or bool, optional
        Controls checking for underrepresentation when groups are specified.
    copy : bool, optional
        If True, return a copy of the AnnData object with results added, by default False.
    inplace : bool, optional
        If True, modify adata in place, by default True.
    result_key : str, optional
        Key in adata.uns where results will be stored, by default "kompot_de".
    overwrite : bool, optional
        Controls behavior when results with the same result_key already exist.
    store_landmarks : bool, optional
        Whether to store landmarks in adata.uns for future reuse, by default False.
    return_full_results : bool, optional
        If True, return the full results dictionary including the differential model.
    store_posterior_covariance : bool, optional
        Whether to store the posterior covariance matrix in adata.obsp.
    progress : bool, optional
        Whether to show progress bars during computation, by default True.
    null_genes : int, List[int], None, or "auto", optional
        Specification for generating null distribution to compute FDR-corrected p-values.
    null_seed : int, optional
        Random seed for reproducible null gene selection, by default 42.
    fdr_threshold : float, optional
        FDR threshold for identifying significantly DE genes, by default 0.05.
    store_additional_stats : bool, optional
        Whether to store additional statistical measures, by default False.
    **function_kwargs : dict
        Additional arguments to pass to the FunctionEstimator.

    Returns
    -------
    Union[Dict[str, np.ndarray], AnnData, Tuple[Dict[str, np.ndarray], AnnData]]
        Return value depends on ``copy`` and ``return_full_results`` parameters.
    """
    # ---- 0. Resolve defaults ----
    if null_genes == "auto":
        if sample_col is not None:
            null_genes = 0
            logger.info(
                "Defaulting null_genes=0 (FDR disabled) because sample_col is "
                "provided."
            )
        else:
            null_genes = 2000

    use_fdr = (
        null_genes is not None
        and null_genes != 0
        and compute_mahalanobis
    )
    use_sample_variance = sample_col is not None

    # ---- 1. Field names & overwrite check ----
    field_names = generate_output_field_names(
        result_key=result_key,
        condition1=condition1,
        condition2=condition2,
        analysis_type="de",
        with_sample_suffix=use_sample_variance,
        sample_suffix="_sample_var" if use_sample_variance else "",
    )
    all_patterns = field_names["all_patterns"]

    _check_overwrites(
        adata, result_key, field_names, all_patterns, sample_col, overwrite,
        compute_mahalanobis, use_fdr, store_additional_stats, groups,
        groupby=groupby, condition1=condition1, condition2=condition2,
        obsm_key=obsm_key, layer=layer, ls_factor=ls_factor,
    )

    # ---- 2. Copy if requested ----
    if copy:
        adata = adata.copy()

    # ---- 3. Extract data from AnnData ----
    data = _extract_de_data(
        adata, groupby, condition1, condition2, obsm_key, layer, genes,
        sample_col, cell_filter, groups, min_cells, min_percentage,
        check_representation,
    )

    X1 = data["X1"]
    X2 = data["X2"]
    expr1 = data["expr1"]
    expr2 = data["expr2"]
    selected_genes = data["selected_genes"]
    filter_mask = data["filter_mask"]
    mask1 = data["mask1"]
    mask2 = data["mask2"]
    condition1_sample_indices = data["condition1_sample_indices"]
    condition2_sample_indices = data["condition2_sample_indices"]
    auto_filter = data["auto_filter"]
    underrep = data["underrep"]

    # ---- 4. Null gene augmentation ----
    expr1, expr2, expanded_genes, null_gene_indices, use_fdr = (
        _augment_with_null_genes(
            adata, expr1, expr2, mask1, mask2, selected_genes,
            null_genes, null_seed, compute_mahalanobis, layer,
        )
    )

    # ---- 5. Resolve landmarks ----
    landmarks = _resolve_landmarks(
        adata, landmarks, n_landmarks, obsm_key, result_key,
    )

    # ---- 6. Fit model ----
    diff_expression = DifferentialExpression(
        n_landmarks=n_landmarks,
        use_sample_variance=use_sample_variance,
        use_empirical_variance=use_empirical_variance,
        eps=eps,
        jit_compile=jit_compile,
        random_state=random_state,
        batch_size=batch_size,
        store_arrays_on_disk=store_arrays_on_disk,
        disk_storage_dir=disk_storage_dir,
        max_memory_ratio=max_memory_ratio,
    )

    diff_expression.fit(
        X1, expr1, X2, expr2,
        sigma=sigma, ls=ls, ls_factor=ls_factor,
        landmarks=landmarks,
        condition1_sample_indices=condition1_sample_indices,
        condition2_sample_indices=condition2_sample_indices,
        allow_single_condition_variance=allow_single_condition_variance,
        **function_kwargs,
    )

    # ---- 7. Store landmarks ----
    _store_landmarks(adata, diff_expression, result_key, store_landmarks)

    # ---- 8. Predict ----
    X_for_prediction = adata.obsm[obsm_key][filter_mask]

    can_store_covariance = store_posterior_covariance and not use_sample_variance
    if store_posterior_covariance and not can_store_covariance:
        if use_sample_variance:
            logger.warning(
                "Cannot store posterior covariance when using sample variance. "
                "Posterior covariance will not be stored."
            )

    expression_results = diff_expression.predict(
        X_for_prediction,
        compute_mahalanobis=compute_mahalanobis,
        progress=progress,
    )

    # ---- 9. FDR ----
    fdr_results = {}
    if use_fdr and null_gene_indices and compute_mahalanobis:
        logger.debug("Computing FDR statistics from null distribution")
        fdr_results = _compute_fdr(
            expression_results, selected_genes, expanded_genes,
            null_gene_indices, fdr_threshold,
        )

    # ---- 10. Posterior covariance ----
    posterior_cov_key = None
    if can_store_covariance:
        logger.info("Computing posterior covariance matrix for storing in obsp...")
        posterior_cov_key = _store_posterior_covariance(
            adata, diff_expression, X_for_prediction, filter_mask,
            field_names, condition1, condition2,
        )

    # ---- 11. Build result dict ----
    results_data = {"mean_lfc": expression_results["mean_log_fold_change"]}
    result_dict = {"model": diff_expression}

    if fdr_results:
        results_data["pvalue"] = fdr_results["pvalues"]
        results_data["local_fdr"] = fdr_results["local_fdr_values"]
        results_data["tail_fdr"] = fdr_results["tail_fdr_values"]
        results_data["is_de"] = fdr_results["is_significant"]
        result_dict["fdr_summary"] = fdr_results["summary_stats"]

    if compute_mahalanobis and "mahalanobis_distances" in expression_results:
        results_data["mahalanobis"] = expression_results["mahalanobis_distances"]
    if "ptp" in expression_results:
        results_data["ptp"] = expression_results["ptp"]

    result_dict["table"] = pd.DataFrame(results_data, index=selected_genes)
    result_dict["underrepresentation"] = underrep

    if (
        hasattr(diff_expression, "computed_landmarks")
        and diff_expression.computed_landmarks is not None
    ):
        result_dict["landmarks"] = diff_expression.computed_landmarks

    result_dict["field_names"] = field_names

    # ---- 12. Store results in adata ----
    if inplace:
        _store_de_results(
            adata, expression_results, fdr_results, field_names,
            selected_genes, filter_mask, sample_col,
            compute_mahalanobis, use_fdr, store_additional_stats,
        )

        # ---- 13. Groups ----
        subset_names = []
        subset_masks = {}
        group_results = {}

        if groups is not None:
            group_results, subset_masks, subset_names = _compute_group_results(
                adata, diff_expression, groups, filter_mask,
                field_names, selected_genes, expanded_genes,
                null_gene_indices, obsm_key, compute_mahalanobis,
                use_fdr, store_additional_stats, fdr_threshold, progress,
            )

        # ---- 14. Field mapping & run info ----
        field_mapping = _build_field_mapping(
            field_names, condition1, condition2, sample_col,
            compute_mahalanobis, use_fdr, fdr_results,
            store_additional_stats, fdr_threshold,
        )

        if groups is not None and subset_names:
            _add_group_field_mapping(
                field_mapping, adata, field_names, subset_names,
                compute_mahalanobis, use_fdr, null_gene_indices,
                store_additional_stats,
            )

        if posterior_cov_key is not None and "posterior_covariance_key" in field_names:
            if posterior_cov_key in adata.obsp:
                field_mapping[posterior_cov_key] = {
                    "location": "obsp",
                    "type": "covariance",
                    "description": (
                        f"Posterior covariance matrix for fold changes "
                        f"between {condition1} and {condition2}"
                    ),
                }

        # Build params dict for recording
        params_dict = {
            "groupby": groupby,
            "condition1": condition1,
            "condition2": condition2,
            "obsm_key": obsm_key,
            "layer": layer,
            "genes": genes,
            "n_landmarks": n_landmarks,
            "landmarks": landmarks is not None,
            "sample_col": sample_col,
            "use_sample_variance": use_sample_variance,
            "use_empirical_variance": use_empirical_variance,
            "sigma": sigma,
            "ls": ls,
            "ls_factor": ls_factor,
            "compute_mahalanobis": compute_mahalanobis,
            "jit_compile": jit_compile,
            "eps": eps,
            "random_state": random_state,
            "used_landmarks": landmarks is not None,
            "store_arrays_on_disk": store_arrays_on_disk,
            "disk_storage_dir": disk_storage_dir,
            "max_memory_ratio": max_memory_ratio,
            "batch_size": batch_size,
            "cell_filter": cell_filter,
            "groups": groups,
            "null_genes": null_genes,
            "null_seed": null_seed,
            "fdr_threshold": fdr_threshold,
            "min_cells": min_cells,
            "min_percentage": min_percentage,
            "check_representation": check_representation,
            "auto_filtered": auto_filter,
            "store_landmarks": store_landmarks,
            "store_posterior_covariance": store_posterior_covariance,
            "result_key": result_key,
            "copy": copy,
            "inplace": inplace,
            "overwrite": overwrite,
            **function_kwargs,
        }

        _record_de_run_info(
            adata, diff_expression, field_names, field_mapping, all_patterns,
            selected_genes, condition1, condition2, sample_col,
            compute_mahalanobis, use_fdr, fdr_results, fdr_threshold,
            store_additional_stats, store_landmarks, store_posterior_covariance,
            store_arrays_on_disk, result_key, groups, subset_names,
            subset_masks, auto_filter, underrep, posterior_cov_key,
            params_dict=params_dict,
        )

    # ---- 15. Return ----
    if copy:
        if return_full_results:
            return result_dict, adata
        return adata
    if return_full_results:
        return result_dict
    return None
