"""Settings dataclasses for the ``kompot.de()`` and ``kompot.da()`` interfaces.

These group related parameters into discoverable objects with sensible
defaults.  Override only what you need::

    kompot.de(adata, "condition", "Young", "Old",
              gp=GPSettings(sigma=0.5))

All settings are optional — ``None`` means "use kompot's default".
"""

from dataclasses import dataclass, field, fields, asdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


@dataclass
class GPSettings:
    """Gaussian-process parameters for the expression model.

    Parameters
    ----------
    sigma : float
        Noise level for the GP.
    ls : float, optional
        Length scale.  If *None*, estimated automatically using
        ``ls_factor``.
    ls_factor : float
        Multiplier applied to the automatically inferred length scale.
    n_landmarks : int, optional
        Number of landmarks for the Nystrom approximation.
    landmarks : np.ndarray, optional
        Pre-computed landmark coordinates.
    use_empirical_variance : bool
        Estimate per-gene heteroscedastic noise from GP residuals.
    batch_size : int, optional
        Number of cells processed at once during prediction.
    eps : float
        Small constant for numerical stability.
    jit_compile : bool
        Use JAX JIT compilation.
    random_state : int, optional
        Random seed for landmark selection.
    """

    sigma: float = 1.0
    ls: Optional[float] = None
    ls_factor: float = 10.0
    n_landmarks: Optional[int] = 5000
    landmarks: Optional[np.ndarray] = None
    use_empirical_variance: bool = False
    batch_size: Optional[int] = 100
    eps: float = 1e-8
    jit_compile: bool = False
    random_state: Optional[int] = None


@dataclass
class FDRSettings:
    """False-discovery-rate and null-distribution parameters.

    Parameters
    ----------
    null_genes : int, List[int], None, or "auto"
        Controls null-distribution calibration for FDR.

        * ``"auto"`` (default) — generates 2 000 null genes by column
          shuffling when ``sample_col`` is not set, 0 otherwise.
        * ``int`` — number of null genes to auto-generate.  Not
          compatible with pre-fitted predictors in
          :class:`ModelSettings` (raises ``ValueError``).
        * ``List[int]`` — explicit column indices of pre-baked null
          features already present in the data.  Use this when
          injecting pre-fitted predictors via :class:`ModelSettings`,
          since the predictors were trained on a fixed set of features
          and cannot cover newly generated columns.  The null features
          are used to calibrate the FDR null distribution and are
          then stripped from all output (table and layers contain
          only the real genes).
        * ``0`` or ``None`` — disable FDR estimation.
    null_seed : int, optional
        Random seed for null-gene sampling.
    threshold : float
        FDR threshold for the ``is_de`` boolean column.
    null_mahalanobis : np.ndarray, optional
        Pre-computed null Mahalanobis distances.  When provided, these
        are used directly as the null distribution for FDR estimation,
        bypassing internal null gene generation.  Mutually exclusive
        with ``null_expression``.
    null_expression : tuple of (np.ndarray, np.ndarray), optional
        External null expression data as ``(expr1, expr2)``.  These
        columns are appended to the expression matrices and fitted
        through the same GP model, then their Mahalanobis distances
        form the null distribution.  Mutually exclusive with
        ``null_mahalanobis``.
    combine_with_internal : bool
        If ``True``, concatenate external null distances with
        internally generated null distances.  If ``False`` (default),
        the external null replaces the internal one entirely.
    """

    null_genes: Union[int, List[int], str, None] = "auto"
    null_seed: Optional[int] = 42
    threshold: float = 0.05

    # External null distribution
    null_mahalanobis: Optional[np.ndarray] = None
    null_expression: Optional[Tuple[np.ndarray, np.ndarray]] = None
    combine_with_internal: bool = False


@dataclass
class DAThresholdSettings:
    """Significance thresholds for differential abundance.

    Parameters
    ----------
    lfc_threshold : float
        Log fold change threshold for significance classification.
    ptp_threshold : float
        Posterior tail probability threshold for significance.
    """

    lfc_threshold: float = 1.0
    ptp_threshold: float = 0.05


@dataclass
class FilterSettings:
    """Cell-filtering and group-subsetting parameters.

    Parameters
    ----------
    cell_filter : optional
        Specification for cells to include (boolean column name, dict
        of column->values, etc.).
    groups : optional
        Column or specification for per-group analyses.
    min_cells : int
        Minimum cells per condition within a group.
    min_percentage : float, optional
        Minimum percentage of cells per condition within a group.
    check_representation : bool, optional
        ``None`` warns, ``True`` auto-filters, ``False`` skips.
    """

    cell_filter: Optional[
        Union[str, List[str], Dict[str, Any], List[Dict[str, Any]]]
    ] = None
    groups: Optional[
        Union[
            str,
            Dict[str, Any],
            List[Dict[str, Any]],
            pd.Series,
            np.ndarray,
            List[np.ndarray],
        ]
    ] = None
    min_cells: int = 2
    min_percentage: Optional[float] = None
    check_representation: Optional[bool] = None


@dataclass
class StorageSettings:
    """Output-storage and memory-management parameters.

    Parameters
    ----------
    result_key : str
        Key prefix used in ``adata.var``, ``adata.layers``, ``adata.uns``.
        Defaults to ``"kompot_de"`` for DE and ``"kompot_da"`` for DA.
    overwrite : bool, optional
        ``None`` (default) warns, ``True`` silently overwrites,
        ``False`` raises on conflict.
    store_landmarks : bool
        Persist landmarks in ``adata.uns`` for future reuse.
    store_posterior_covariance : bool
        Store the (n_cells x n_cells) posterior covariance in ``adata.obsp``.
    store_additional_stats : bool
        Store extra columns (p-values, tail FDR, PTP, z-scores).
    store_arrays_on_disk : bool, optional
        Use disk-backed arrays for large intermediate matrices.
    disk_storage_dir : str, optional
        Directory for disk-backed arrays.
    max_memory_ratio : float
        Fraction of RAM before triggering disk storage.
    """

    result_key: Optional[str] = None
    overwrite: Optional[bool] = None
    store_landmarks: bool = False
    store_posterior_covariance: bool = False
    store_additional_stats: bool = False
    store_arrays_on_disk: Optional[bool] = None
    disk_storage_dir: Optional[str] = None
    max_memory_ratio: float = 0.8


@dataclass
class ModelSettings:
    """Pre-fitted models or predictors to inject into ``de()`` or ``da()``.

    When provided, these skip internal fitting for the corresponding
    component.  ``model1``/``model2`` take precedence over individual
    predictors.

    When using pre-fitted predictors with FDR, null features must be
    included in the data *before* fitting (the predictors cannot cover
    columns that are added later).  Pass their column indices via
    ``FDRSettings(null_genes=[...])``.  Passing ``null_genes=int``
    with pre-fitted predictors raises ``ValueError``.

    Parameters
    ----------
    model1, model2 : ExpressionModel, optional
        Full pre-fitted :class:`~kompot.ExpressionModel` for each
        condition (DE only).  Takes precedence over individual predictors.
    function_predictor1, function_predictor2 : callable, optional
        Pre-fitted mellon Predictor for each condition (DE only).
    obs_variance_predictor1, obs_variance_predictor2 : callable, optional
        Pre-fitted empirical variance predictor for each condition (DE only).
    variance_predictor1, variance_predictor2 : callable, optional
        Pre-fitted sample variance predictor for each condition.
        Signature: ``(X, diag=True/False) -> array``.
    density_predictor1, density_predictor2 : callable, optional
        Pre-fitted density predictor for each condition (DA only).
    """

    model1: Optional[Any] = None
    model2: Optional[Any] = None
    function_predictor1: Optional[Any] = None
    function_predictor2: Optional[Any] = None
    obs_variance_predictor1: Optional[Any] = None
    obs_variance_predictor2: Optional[Any] = None
    variance_predictor1: Optional[Any] = None
    variance_predictor2: Optional[Any] = None
    density_predictor1: Optional[Any] = None
    density_predictor2: Optional[Any] = None


@dataclass
class OutputSettings:
    """Control what ``de()`` / ``da()`` returns and how it behaves.

    Parameters
    ----------
    copy : bool
        Return a copy of the AnnData instead of modifying in place.
    inplace : bool
        Write results into the AnnData object.
    return_full_results : bool
        Return the full results dict (model, table, landmarks, ...).
        When True, ``result_dict["null"]`` includes the full null gene
        expression matrices, fold changes, and imputations alongside
        the lightweight metadata.
    return_null_data : bool
        Include lightweight null-distribution metadata in the returned
        results dict (gene indices, names, seed, Mahalanobis distances)
        without the full expression matrices.  Ignored when
        ``return_full_results`` is True (which already includes
        everything).
    compute_mahalanobis : bool
        Compute per-gene Mahalanobis distances (DE only).
    allow_single_condition_variance : bool
        Allow sample-variance estimation when only one condition has
        multiple samples.
    progress : bool
        Show progress bars.
    """

    copy: bool = False
    inplace: bool = True
    return_full_results: bool = False
    return_null_data: bool = False
    compute_mahalanobis: bool = True
    allow_single_condition_variance: bool = False
    progress: bool = True

