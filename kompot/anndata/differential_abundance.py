"""
Differential abundance analysis for AnnData objects.
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional, Union, Dict, Any, List, Tuple

from ..differential import DifferentialAbundance
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


def compute_differential_abundance(
    adata,
    groupby: str,
    condition1: str,
    condition2: str,
    obsm_key: str = "DM_EigenVectors",
    n_landmarks: Optional[int] = None,
    landmarks: Optional[np.ndarray] = None,
    sample_col: Optional[str] = None,
    log_fold_change_threshold: float = 1.0,
    ptp_threshold: float = 0.05,
    ls_factor: float = 10.0,
    jit_compile: bool = False,
    random_state: Optional[int] = None,
    copy: bool = False,
    inplace: bool = True,
    result_key: str = "kompot_da",
    batch_size: Optional[int] = None,
    overwrite: Optional[bool] = None,
    store_landmarks: bool = False,
    return_full_results: bool = False,
    allow_single_condition_variance: bool = False,
    **density_kwargs,
) -> Union[Dict[str, np.ndarray], Any]:
    """
    Compute differential abundance between two conditions directly from an AnnData object.

    This function is a scverse-compatible wrapper around the DifferentialAbundance class
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
        Key in adata.obsm containing the cell states, by default "DM_EigenVectors".
    n_landmarks : int, optional
        Number of landmarks to use for approximation, by default None.
    landmarks : np.ndarray, optional
        Pre-computed landmarks to use.
    sample_col : str, optional
        Column name in adata.obs containing sample labels.
    allow_single_condition_variance : bool, optional
        If True, allows variance estimation with only one condition having
        multiple samples. By default False.
    log_fold_change_threshold : float, optional
        Threshold for considering a log fold change significant, by default 1.0.
    ptp_threshold : float, optional
        Threshold for considering a PTP significant, by default 0.05.
    ls_factor : float, optional
        Multiplication factor for auto-inferred length scale, by default 10.0.
    jit_compile : bool, optional
        Whether to use JAX JIT compilation, by default False.
    random_state : int, optional
        Random seed for reproducible landmark selection.
    copy : bool, optional
        If True, return a copy of the AnnData, by default False.
    inplace : bool, optional
        If True, modify adata in place, by default True.
    result_key : str, optional
        Key in adata.uns where results will be stored, by default "kompot_da".
    batch_size : int, optional
        Number of samples to process at once during density estimation.
    overwrite : bool, optional
        Controls behavior when results already exist.
    store_landmarks : bool, optional
        Whether to store landmarks in adata.uns, by default False.
    return_full_results : bool, optional
        If True, return the full results dictionary, by default False.
    **density_kwargs : dict
        Additional arguments to pass to the DensityEstimator.

    Returns
    -------
    Union[Dict[str, np.ndarray], AnnData, Tuple[Dict[str, np.ndarray], AnnData]]
        Return value depends on ``copy`` and ``return_full_results`` parameters.
    """
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
    diff_abundance = DifferentialAbundance(
        log_fold_change_threshold=log_fold_change_threshold,
        ptp_threshold=ptp_threshold,
        n_landmarks=n_landmarks,
        jit_compile=jit_compile,
        random_state=random_state,
        batch_size=batch_size,
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
    )

    # ---- 7. Store results ----
    _store_da_results(
        adata, abundance_results, field_names, condition1, condition2,
    )

    # ---- 8. Record run info ----
    params_dict = {
        "groupby": groupby,
        "condition1": condition1,
        "condition2": condition2,
        "obsm_key": obsm_key,
        "n_landmarks": n_landmarks,
        "landmarks": landmarks is not None,
        "sample_col": sample_col,
        "use_sample_variance": sample_col is not None,
        "log_fold_change_threshold": log_fold_change_threshold,
        "ptp_threshold": ptp_threshold,
        "ls_factor": ls_factor,
        "jit_compile": jit_compile,
        "random_state": random_state,
        "used_landmarks": landmarks is not None,
        "batch_size": batch_size,
        "store_landmarks": store_landmarks,
        "result_key": result_key,
        "copy": copy,
        "inplace": inplace,
        "overwrite": overwrite,
        **density_kwargs,
    }

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
