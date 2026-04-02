"""
Differential expression analysis for AnnData objects.
"""

import logging
import warnings
import numpy as np
import pandas as pd
from typing import Optional, Union, Dict, Any, List

from ..differential import DifferentialExpression
from ..settings import GPSettings, FDRSettings, FilterSettings, StorageSettings, OutputSettings, ModelSettings
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
    _augment_with_external_null_expression,
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


def de(
    adata,
    groupby: str,
    condition1: str,
    condition2: str,
    obsm_key: str = "DM_EigenVectors",
    layer=None,
    genes=None,
    sample_col=None,
    gp: "GPSettings | None" = None,
    fdr: "FDRSettings | None" = None,
    filter: "FilterSettings | None" = None,
    storage: "StorageSettings | None" = None,
    output: "OutputSettings | None" = None,
    model: "ModelSettings | None" = None,
    dry_run: bool = False,
    **function_kwargs,
) -> "Union[Dict[str, np.ndarray], Any]":
    """Run differential expression analysis on an AnnData object.

    The most common call is just::

        kompot.de(adata, "condition", "Young", "Old")

    Advanced options are available through the settings dataclasses
    (:class:`~kompot.GPSettings`, :class:`~kompot.FDRSettings`,
    :class:`~kompot.FilterSettings`, :class:`~kompot.StorageSettings`,
    :class:`~kompot.OutputSettings`).  Any field left at its default is
    equivalent to omitting it entirely.  Extra ``**function_kwargs`` are
    forwarded to mellon's :class:`~mellon.FunctionEstimator`.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing cells from both conditions.
    groupby : str
        Column in ``adata.obs`` with condition labels.
    condition1, condition2 : str
        Labels identifying the two conditions.
    obsm_key : str
        Key in ``adata.obsm`` for cell-state coordinates.
    layer : str, optional
        Layer with expression data (``None`` → ``adata.X``).
    genes : list of str, optional
        Subset of genes to analyse.
    sample_col : str, optional
        Column with biological-replicate labels.
    gp : GPSettings, optional
        GP model parameters (sigma, ls, n_landmarks, etc.).
    fdr : FDRSettings, optional
        FDR / null-distribution parameters.
    filter : FilterSettings, optional
        Cell filtering and group-subsetting.
    storage : StorageSettings, optional
        Where and how results are stored.
    output : OutputSettings, optional
        Return-value and progress-bar control.
    model : ModelSettings, optional
        Pre-fitted models or predictors to inject.  When provided,
        fitting is skipped for the corresponding components.
        See :class:`~kompot.ModelSettings`.
    dry_run : bool, optional
        If True, estimate resource requirements and print a report
        instead of running the analysis.  Returns a
        :class:`~kompot.resource_estimation.ResourcePlan`.
    **function_kwargs
        Forwarded to :class:`~mellon.FunctionEstimator`.

    Returns
    -------
    Union[Dict[str, np.ndarray], AnnData, Tuple[Dict[str, np.ndarray], AnnData]]
        Return value depends on ``copy`` and ``return_full_results`` in
        :class:`~kompot.OutputSettings`.
    """
    # ---- Unpack settings ----
    _gp = gp if gp is not None else GPSettings()
    _fdr = fdr if fdr is not None else FDRSettings()
    _filter = filter if filter is not None else FilterSettings()
    _storage = storage if storage is not None else StorageSettings()
    _output = output if output is not None else OutputSettings()

    sigma = _gp.sigma
    ls = _gp.ls
    ls_factor = _gp.ls_factor
    n_landmarks = _gp.n_landmarks
    landmarks = _gp.landmarks
    use_empirical_variance = _gp.use_empirical_variance
    batch_size = _gp.batch_size
    eps = _gp.eps
    jit_compile = _gp.jit_compile
    random_state = _gp.random_state

    null_genes = _fdr.null_genes
    null_seed = _fdr.null_seed
    fdr_threshold = _fdr.threshold
    ext_null_mahalanobis = _fdr.null_mahalanobis
    ext_null_expression = _fdr.null_expression
    combine_with_internal = _fdr.combine_with_internal

    cell_filter = _filter.cell_filter
    groups = _filter.groups
    min_cells = _filter.min_cells
    min_percentage = _filter.min_percentage
    check_representation = _filter.check_representation

    result_key = _storage.result_key or "kompot_de"
    overwrite = _storage.overwrite
    store_landmarks = _storage.store_landmarks
    store_posterior_covariance = _storage.store_posterior_covariance
    store_additional_stats = _storage.store_additional_stats
    store_arrays_on_disk = _storage.store_arrays_on_disk
    disk_storage_dir = _storage.disk_storage_dir
    max_memory_ratio = _storage.max_memory_ratio

    copy = _output.copy
    inplace = _output.inplace
    return_full_results = _output.return_full_results
    return_null_data = _output.return_null_data
    compute_mahalanobis = _output.compute_mahalanobis
    allow_single_condition_variance = _output.allow_single_condition_variance
    progress = _output.progress

    # ---- Validate external null settings ----
    if ext_null_mahalanobis is not None and ext_null_expression is not None:
        raise ValueError(
            "null_mahalanobis and null_expression are mutually exclusive. "
            "Provide one or the other, not both."
        )

    if ext_null_expression is not None:
        _model_check = model if model is not None else ModelSettings()
        _has_prefitted_for_null = (
            _model_check.function_predictor1 is not None
            or _model_check.function_predictor2 is not None
            or _model_check.model1 is not None
            or _model_check.model2 is not None
        )
        if _has_prefitted_for_null:
            raise ValueError(
                "null_expression cannot be used with pre-fitted models or "
                "predictors. The expression columns must go through the GP "
                "fitting step, but pre-fitted predictors skip fitting."
            )

    if combine_with_internal and (null_genes is None or null_genes == 0):
        if ext_null_mahalanobis is not None or ext_null_expression is not None:
            warnings.warn(
                "combine_with_internal=True but null_genes=0/None — there are "
                "no internal null genes to combine with. Only the external null "
                "distribution will be used.",
                UserWarning,
                stacklevel=2,
            )

    # ---- dry run ----
    if dry_run:
        from ..resource_estimation import estimate_differential_expression_resources
        plan = estimate_differential_expression_resources(
            adata, condition1, condition2, groupby,
            obsm_key=obsm_key, layer=layer, genes=genes,
            sample_col=sample_col, n_landmarks=n_landmarks,
            landmarks=landmarks, sigma=sigma, ls=ls,
            ls_factor=ls_factor, batch_size=batch_size, eps=eps,
            use_empirical_variance=use_empirical_variance,
            null_genes=null_genes, null_seed=null_seed,
            fdr_threshold=fdr_threshold,
            compute_mahalanobis=compute_mahalanobis,
            result_key=result_key, overwrite=overwrite,
            store_landmarks=store_landmarks,
            store_posterior_covariance=store_posterior_covariance,
            store_additional_stats=store_additional_stats,
            store_arrays_on_disk=store_arrays_on_disk,
            disk_storage_dir=disk_storage_dir,
            max_memory_ratio=max_memory_ratio,
            cell_filter=cell_filter, groups=groups,
            min_cells=min_cells, min_percentage=min_percentage,
            allow_single_condition_variance=allow_single_condition_variance,
            **function_kwargs,
        )
        print(plan.format_report())
        return plan

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

    has_external_null = (
        ext_null_mahalanobis is not None or ext_null_expression is not None
    )
    use_fdr = (
        (
            (null_genes is not None and null_genes != 0)
            or has_external_null
        )
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
    _model = model if model is not None else ModelSettings()

    # When predictors are injected, null features are already baked into
    # the data (the predictors were trained on real + null features).
    # Skip shuffling/appending and just split selected_genes.
    _has_prefitted = (
        _model.function_predictor1 is not None
        or _model.function_predictor2 is not None
        or _model.model1 is not None
        or _model.model2 is not None
    )
    if _has_prefitted and isinstance(null_genes, list) and len(null_genes) > 0:
        # Null features are pre-existing in the data — validate indices
        n_features = len(selected_genes)
        bad = [i for i in null_genes if i < 0 or i >= n_features]
        if bad:
            raise ValueError(
                f"null_genes indices {bad[:5]}{'...' if len(bad) > 5 else ''} "
                f"are out of range for {n_features} features. When using "
                f"pre-fitted predictors, null_genes must be a list of "
                f"feature indices that are already in the data."
            )
        null_gene_indices = null_genes
        expanded_genes = list(selected_genes)
        null_set = set(null_gene_indices)
        selected_genes = [
            g for i, g in enumerate(expanded_genes) if i not in null_set
        ]
        use_fdr = compute_mahalanobis
    elif _has_prefitted and isinstance(null_genes, int) and null_genes > 0:
        raise ValueError(
            "Cannot auto-generate null genes (null_genes=int) when using "
            "pre-fitted predictors. The predictors were trained on a fixed "
            "set of features and cannot cover newly generated null columns. "
            "Either include null features in your data before fitting and "
            "pass their indices as null_genes=[...], or set null_genes=0."
        )
    else:
        expr1, expr2, expanded_genes, null_gene_indices, _internal_use_fdr = (
            _augment_with_null_genes(
                adata, expr1, expr2, mask1, mask2, selected_genes,
                null_genes, null_seed, compute_mahalanobis, layer,
            )
        )
        # Preserve use_fdr=True when external null is provided even if
        # internal null genes are disabled (null_genes=0).
        if not has_external_null:
            use_fdr = _internal_use_fdr

    # ---- 4b. External null expression augmentation ----
    if ext_null_expression is not None:
        null_expr1, null_expr2 = ext_null_expression
        expr1, expr2, expanded_genes, null_gene_indices = (
            _augment_with_external_null_expression(
                expr1, expr2, expanded_genes, null_gene_indices,
                null_expr1, null_expr2,
            )
        )
        use_fdr = compute_mahalanobis
        logger.info(
            f"Appended {null_expr1.shape[1] if null_expr1.ndim > 1 else 1} "
            f"external null expression columns."
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
        model1=_model.model1,
        model2=_model.model2,
        function_predictor1=_model.function_predictor1,
        function_predictor2=_model.function_predictor2,
        obs_variance_predictor1=_model.obs_variance_predictor1,
        obs_variance_predictor2=_model.obs_variance_predictor2,
        variance_predictor1=_model.variance_predictor1,
        variance_predictor2=_model.variance_predictor2,
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
    _has_null_genes = null_gene_indices and len(null_gene_indices) > 0
    _has_ext_null = ext_null_mahalanobis is not None
    if use_fdr and (_has_null_genes or _has_ext_null) and compute_mahalanobis:
        logger.debug("Computing FDR statistics from null distribution")
        fdr_results = _compute_fdr(
            expression_results, selected_genes, expanded_genes,
            null_gene_indices, fdr_threshold,
            external_null_mahalanobis=ext_null_mahalanobis,
            combine_nulls=combine_with_internal,
            return_null_data=return_null_data,
            return_full_results=return_full_results,
            null_seed=null_seed,
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
        if "null" in fdr_results:
            result_dict["null"] = fdr_results["null"]

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
        from .utils.params import build_params_dict
        params_dict = build_params_dict(
            top_level=dict(
                groupby=groupby, condition1=condition1, condition2=condition2,
                obsm_key=obsm_key, layer=layer, genes=genes,
                sample_col=sample_col,
            ),
            gp=GPSettings(
                sigma=sigma, ls=ls, ls_factor=ls_factor,
                n_landmarks=n_landmarks,
                use_empirical_variance=use_empirical_variance,
                batch_size=batch_size, eps=eps, jit_compile=jit_compile,
                random_state=random_state,
            ),
            fdr=FDRSettings(
                null_genes=null_genes, null_seed=null_seed,
                threshold=fdr_threshold,
            ),
            filter=FilterSettings(
                cell_filter=cell_filter, groups=groups,
                min_cells=min_cells, min_percentage=min_percentage,
                check_representation=check_representation,
            ),
            storage=StorageSettings(
                result_key=result_key, overwrite=overwrite,
                store_landmarks=store_landmarks,
                store_posterior_covariance=store_posterior_covariance,
                store_additional_stats=store_additional_stats,
                store_arrays_on_disk=store_arrays_on_disk,
                disk_storage_dir=disk_storage_dir,
                max_memory_ratio=max_memory_ratio,
            ),
            output=OutputSettings(
                copy=copy, inplace=inplace,
                return_full_results=return_full_results,
                return_null_data=return_null_data,
                compute_mahalanobis=compute_mahalanobis,
                allow_single_condition_variance=allow_single_condition_variance,
                progress=progress,
            ),
            extra_kwargs=function_kwargs,
            landmarks_provided=landmarks is not None,
        )

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
    _return_dict = return_full_results or return_null_data
    if copy:
        if _return_dict:
            return result_dict, adata
        return adata
    if _return_dict:
        return result_dict
    return None


def compute_differential_expression(
    adata,
    groupby: str,
    condition1: str,
    condition2: str,
    obsm_key: str = "DM_EigenVectors",
    layer=None,
    genes=None,
    n_landmarks=5000,
    landmarks=None,
    sample_col=None,
    sigma: float = 1.0,
    ls=None,
    ls_factor: float = 10.0,
    compute_mahalanobis: bool = True,
    jit_compile: bool = False,
    eps: float = 1e-8,
    random_state=None,
    batch_size: int = 100,
    store_arrays_on_disk=None,
    disk_storage_dir=None,
    max_memory_ratio: float = 0.8,
    cell_filter=None,
    groups=None,
    min_cells: int = 2,
    min_percentage=None,
    check_representation=None,
    copy: bool = False,
    inplace: bool = True,
    result_key: str = "kompot_de",
    overwrite=None,
    store_landmarks: bool = False,
    return_full_results: bool = False,
    store_posterior_covariance: bool = False,
    allow_single_condition_variance: bool = False,
    use_empirical_variance: bool = True,
    progress: bool = True,
    null_genes="auto",
    null_seed=42,
    fdr_threshold: float = 0.05,
    store_additional_stats: bool = False,
    **function_kwargs,
):
    """Deprecated. Use :func:`kompot.de` instead.

    .. deprecated::
        Use :func:`kompot.de` with Settings dataclasses instead.
    """
    warnings.warn(
        "compute_differential_expression() is deprecated and will be removed "
        "in a future version. Use kompot.de() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return de(
        adata,
        groupby=groupby,
        condition1=condition1,
        condition2=condition2,
        obsm_key=obsm_key,
        layer=layer,
        genes=genes,
        sample_col=sample_col,
        gp=GPSettings(
            sigma=sigma, ls=ls, ls_factor=ls_factor,
            n_landmarks=n_landmarks, landmarks=landmarks,
            use_empirical_variance=use_empirical_variance,
            batch_size=batch_size, eps=eps,
            jit_compile=jit_compile, random_state=random_state,
        ),
        fdr=FDRSettings(
            null_genes=null_genes, null_seed=null_seed,
            threshold=fdr_threshold,
        ),
        filter=FilterSettings(
            cell_filter=cell_filter, groups=groups,
            min_cells=min_cells, min_percentage=min_percentage,
            check_representation=check_representation,
        ),
        storage=StorageSettings(
            result_key=result_key, overwrite=overwrite,
            store_landmarks=store_landmarks,
            store_posterior_covariance=store_posterior_covariance,
            store_additional_stats=store_additional_stats,
            store_arrays_on_disk=store_arrays_on_disk,
            disk_storage_dir=disk_storage_dir,
            max_memory_ratio=max_memory_ratio,
        ),
        output=OutputSettings(
            copy=copy, inplace=inplace,
            return_full_results=return_full_results,
            compute_mahalanobis=compute_mahalanobis,
            allow_single_condition_variance=allow_single_condition_variance,
            progress=progress,
        ),
        **function_kwargs,
    )
