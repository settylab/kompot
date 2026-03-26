"""
Differential abundance analysis for AnnData objects.
"""

import logging
import warnings
import numpy as np
import pandas as pd
from typing import Optional, Union, Dict, Any, List, Tuple

from ..differential import DifferentialAbundance
from ..settings import GPSettings, DAThresholdSettings, StorageSettings, OutputSettings, ModelSettings
from .utils import generate_output_field_names
from ._da_helpers import (
    _check_da_overwrites,
    _extract_da_data,
    _resolve_da_landmarks,
    _store_da_landmarks,
    _store_da_results,
    _record_da_run_info,
)

logger = logging.getLogger("kompot")


def da(
    adata,
    groupby: str,
    condition1: str,
    condition2: str,
    obsm_key: str = "DM_EigenVectors",
    sample_col=None,
    gp: "GPSettings | None" = None,
    threshold: "DAThresholdSettings | None" = None,
    storage: "StorageSettings | None" = None,
    output: "OutputSettings | None" = None,
    model: "ModelSettings | None" = None,
    **density_kwargs,
) -> "Union[Dict[str, np.ndarray], Any]":
    """Run differential abundance analysis on an AnnData object.

    The most common call is just::

        kompot.da(adata, "condition", "Young", "Old")

    Advanced options are available through the settings dataclasses
    (:class:`~kompot.GPSettings`, :class:`~kompot.DAThresholdSettings`,
    :class:`~kompot.StorageSettings`, :class:`~kompot.OutputSettings`).
    Any field left at its default is equivalent to omitting it entirely.
    Extra ``**density_kwargs`` are forwarded to mellon's
    :class:`~mellon.DensityEstimator`.

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
    sample_col : str, optional
        Column with biological-replicate labels.
    gp : GPSettings, optional
        GP model parameters (``ls_factor``, ``n_landmarks``,
        ``landmarks``, ``batch_size``, ``jit_compile``,
        ``random_state``).
    threshold : DAThresholdSettings, optional
        Significance thresholds for abundance changes.
    storage : StorageSettings, optional
        Where and how results are stored.
    output : OutputSettings, optional
        Return-value control.
    model : ModelSettings, optional
        Pre-fitted models or predictors to inject.  For DA, only
        ``density_predictor1/2`` and ``variance_predictor1/2`` are used.
        See :class:`~kompot.ModelSettings`.
    **density_kwargs
        Forwarded to :class:`~mellon.DensityEstimator`.

    Returns
    -------
    Union[Dict[str, np.ndarray], AnnData, Tuple[Dict[str, np.ndarray], AnnData]]
        Return value depends on ``copy`` and ``return_full_results`` in
        :class:`~kompot.OutputSettings`.
    """
    # ---- Unpack settings ----
    # DA has different defaults than GPSettings for n_landmarks (None vs 5000)
    # and batch_size (None vs 100), so unpack with DA-specific defaults.
    n_landmarks = gp.n_landmarks if gp is not None else None
    landmarks = gp.landmarks if gp is not None else None
    ls_factor = gp.ls_factor if gp is not None else 10.0
    batch_size = gp.batch_size if gp is not None else None
    jit_compile = gp.jit_compile if gp is not None else False
    random_state = gp.random_state if gp is not None else None

    _threshold = threshold if threshold is not None else DAThresholdSettings()
    log_fold_change_threshold = _threshold.lfc_threshold
    ptp_threshold = _threshold.ptp_threshold

    _storage = storage if storage is not None else StorageSettings()
    result_key = _storage.result_key or "kompot_da"
    overwrite = _storage.overwrite
    store_landmarks = _storage.store_landmarks

    _output = output if output is not None else OutputSettings()
    copy = _output.copy
    inplace = _output.inplace
    return_full_results = _output.return_full_results
    allow_single_condition_variance = _output.allow_single_condition_variance
    progress = _output.progress

    # ---- 0. Field names & overwrite check ----
    field_names = generate_output_field_names(
        result_key=result_key,
        condition1=condition1,
        condition2=condition2,
        analysis_type="da",
        with_sample_suffix=(sample_col is not None),
        sample_suffix="_sample_var" if sample_col is not None else "",
    )

    _check_da_overwrites(
        adata, result_key, field_names, sample_col, overwrite,
        groupby=groupby, condition1=condition1, condition2=condition2,
        obsm_key=obsm_key, ls_factor=ls_factor,
    )

    # ---- 1. Copy if requested ----
    if copy:
        adata = adata.copy()

    # ---- 2. Extract data ----
    data = _extract_da_data(
        adata, groupby, condition1, condition2, obsm_key, sample_col,
    )

    # ---- 3. Resolve landmarks ----
    landmarks = _resolve_da_landmarks(adata, landmarks, obsm_key, result_key)

    # ---- 4. Fit model ----
    _model = model if model is not None else ModelSettings()
    diff_abundance = DifferentialAbundance(
        log_fold_change_threshold=log_fold_change_threshold,
        ptp_threshold=ptp_threshold,
        n_landmarks=n_landmarks,
        jit_compile=jit_compile,
        random_state=random_state,
        batch_size=batch_size,
        density_predictor1=_model.density_predictor1,
        density_predictor2=_model.density_predictor2,
        variance_predictor1=_model.variance_predictor1,
        variance_predictor2=_model.variance_predictor2,
    )

    diff_abundance.fit(
        data["X1"],
        data["X2"],
        landmarks=landmarks,
        ls_factor=ls_factor,
        condition1_sample_indices=data["condition1_sample_indices"],
        condition2_sample_indices=data["condition2_sample_indices"],
        allow_single_condition_variance=allow_single_condition_variance,
        **density_kwargs,
    )

    # ---- 5. Store landmarks ----
    _store_da_landmarks(adata, diff_abundance, result_key, store_landmarks)

    # ---- 6. Predict ----
    X_for_prediction = adata.obsm[obsm_key]
    abundance_results = diff_abundance.predict(
        X_for_prediction,
        log_fold_change_threshold=log_fold_change_threshold,
        ptp_threshold=ptp_threshold,
        progress=progress,
    )

    # ---- 7. Store results ----
    _store_da_results(
        adata, abundance_results, field_names, condition1, condition2,
    )

    # ---- 8. Record run info ----
    from .utils.params import build_params_dict
    params_dict = build_params_dict(
        top_level=dict(
            groupby=groupby, condition1=condition1, condition2=condition2,
            obsm_key=obsm_key, sample_col=sample_col,
        ),
        gp=GPSettings(
            ls_factor=ls_factor, n_landmarks=n_landmarks,
            batch_size=batch_size, jit_compile=jit_compile,
            random_state=random_state,
        ),
        threshold=DAThresholdSettings(
            lfc_threshold=log_fold_change_threshold,
            ptp_threshold=ptp_threshold,
        ),
        storage=StorageSettings(
            result_key=result_key, overwrite=overwrite,
            store_landmarks=store_landmarks,
        ),
        output=OutputSettings(
            copy=copy, inplace=inplace,
            allow_single_condition_variance=allow_single_condition_variance,
            progress=progress,
        ),
        extra_kwargs=density_kwargs,
        landmarks_provided=landmarks is not None,
    )

    _record_da_run_info(
        adata, field_names, condition1, condition2,
        sample_col, result_key, params_dict,
    )

    # ---- 9. Build result dict ----
    results_table = pd.DataFrame(
        {
            "lfc": abundance_results["log_fold_change"],
            "lfc_zscore": abundance_results["log_fold_change_zscore"],
            "neg_log10_ptp": abundance_results["neg_log10_fold_change_ptp"],
            "direction": abundance_results["log_fold_change_direction"],
        },
        index=adata.obs_names,
    )

    result_dict = {
        "table": results_table,
        "model": diff_abundance,
        "field_names": field_names,
    }

    if (
        hasattr(diff_abundance, "computed_landmarks")
        and diff_abundance.computed_landmarks is not None
    ):
        result_dict["landmarks"] = diff_abundance.computed_landmarks

    # ---- 10. Return ----
    if copy:
        if return_full_results:
            return result_dict, adata
        return adata
    if return_full_results:
        return result_dict
    return None


def compute_differential_abundance(
    adata,
    groupby: str,
    condition1: str,
    condition2: str,
    obsm_key: str = "DM_EigenVectors",
    n_landmarks=None,
    landmarks=None,
    sample_col=None,
    log_fold_change_threshold: float = 1.0,
    ptp_threshold: float = 0.05,
    ls_factor: float = 10.0,
    jit_compile: bool = False,
    random_state=None,
    copy: bool = False,
    inplace: bool = True,
    result_key: str = "kompot_da",
    batch_size=None,
    overwrite=None,
    store_landmarks: bool = False,
    return_full_results: bool = False,
    allow_single_condition_variance: bool = False,
    **density_kwargs,
):
    """Deprecated. Use :func:`kompot.da` instead.

    .. deprecated::
        Use :func:`kompot.da` with Settings dataclasses instead.
    """
    warnings.warn(
        "compute_differential_abundance() is deprecated and will be removed "
        "in a future version. Use kompot.da() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return da(
        adata,
        groupby=groupby,
        condition1=condition1,
        condition2=condition2,
        obsm_key=obsm_key,
        sample_col=sample_col,
        gp=GPSettings(
            n_landmarks=n_landmarks,
            landmarks=landmarks,
            ls_factor=ls_factor,
            batch_size=batch_size,
            jit_compile=jit_compile,
            random_state=random_state,
        ),
        threshold=DAThresholdSettings(
            lfc_threshold=log_fold_change_threshold,
            ptp_threshold=ptp_threshold,
        ),
        storage=StorageSettings(
            result_key=result_key,
            overwrite=overwrite,
            store_landmarks=store_landmarks,
        ),
        output=OutputSettings(
            copy=copy,
            inplace=inplace,
            return_full_results=return_full_results,
            allow_single_condition_variance=allow_single_condition_variance,
        ),
        **density_kwargs,
    )
