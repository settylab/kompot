"""Settings dataclasses for the ``kompot.de()``, ``kompot.da()``, and
``kompot.smooth_expression()`` interfaces.

These group related parameters into discoverable objects with sensible
defaults.  Override only what you need::

    kompot.de(adata, "condition", "Young", "Old",
              gp=GPSettings(sigma=0.5))

All settings are optional — ``None`` means "use kompot's default".

Every dataclass validates its fields at construction time via
``__post_init__``, so typos and out-of-range values are caught
immediately instead of deep inside mellon or JAX.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .validation import (
    validate_positive_float,
    validate_positive_int,
    validate_non_negative_float,
    validate_probability,
    validate_bool,
)

#: Recognised values for ``GPSettings.param_scheme``.  ``None`` is also
#: accepted and means "this entry point's default".
PARAM_SCHEMES = ("condition1", "condition2", "symmetric", "pooled", "separate")


def validate_param_scheme(value, optional: bool = False):
    """Return ``value`` if it is a recognised ``param_scheme``, else raise."""
    if value is None:
        if optional:
            return None
        raise ValueError(f"'param_scheme' must be one of {PARAM_SCHEMES} (got None).")
    if value not in PARAM_SCHEMES:
        raise ValueError(
            f"'param_scheme' must be one of {PARAM_SCHEMES} (got {value!r})."
        )
    return value


def resolve_param_scheme(value, default: str) -> str:
    """Map ``None`` onto an entry point's default and validate the result."""
    return validate_param_scheme(default if value is None else value)


@dataclass
class GPSettings:
    """Gaussian-process parameters for the expression model.

    Parameters
    ----------
    sigma : float
        Noise level for the GP.
    ls : float, optional
        Length scale of the GP kernel, shared by both conditions.  If *None*
        it is estimated from the cells, from where ``param_scheme`` says.
    ls_factor : float
        Multiplier applied to the automatically inferred length scale.
    n_landmarks : int, optional
        Number of landmarks for the Nystrom approximation.  Under sample
        variance (``sample_col`` set on :func:`kompot.de`) the dominant
        allocation is ``2 * n_landmarks**2 * n_genes * 8`` bytes, so memory is
        **quadratic** in this value: halving it quarters the covariance
        footprint.  It also cuts the per-gene Cholesky factorisation, though by
        an amount worth measuring rather than extrapolating.  See
        https://kompot.readthedocs.io/en/latest/resource_planning.html
    landmarks : np.ndarray, optional
        Pre-computed landmark coordinates.
    use_empirical_variance : bool
        Estimate per-gene heteroscedastic noise from GP residuals.
    batch_size : int, optional
        Number of cells processed at once during prediction.  Also bounds the
        gene batches of the *shared*-covariance Mahalanobis computation.  It
        does **not** bound the per-gene covariance loop used under sample
        variance, which processes one gene at a time regardless.
    eps : float
        Small constant for numerical stability.
    jit_compile : bool
        Use JAX JIT compilation.
    random_state : int, optional
        Random seed for landmark selection.
    param_scheme : str, optional
        Which cells the length scale is estimated from; in :func:`kompot.da`
        it covers ``d`` and ``mu`` as well.  A value given explicitly (``ls``
        here, or ``d`` / ``mu`` passed to ``da()``) pins that parameter and
        takes precedence.

        ``None`` (default) keeps each entry point's own default:
        ``"condition1"`` for ``de()`` and ``"separate"`` for ``da()``.

        * ``"condition1"`` — estimate from condition 1's cells and reuse for
          condition 2.  Makes the result depend on which condition is passed
          first.
        * ``"condition2"`` — the mirror: estimate from condition 2's cells and
          reuse for condition 1.  Equally asymmetric, by design; it exists so
          that ``de(X, Y, "condition1")`` and ``de(Y, X, "condition2")`` are
          the same computation with the labels exchanged
          **provided both runs use the same landmarks**.  The defaults here
          (``n_landmarks=5000``, ``landmarks=None``) do *not* satisfy that and
          the equivalence then holds only approximately — see
          :meth:`kompot.differential.DifferentialExpression.fit`.
        * ``"symmetric"`` — estimate from each condition separately and share
          the size-weighted combination (geometric mean for ``ls``).
          Invariant under swapping the conditions.
        * ``"pooled"`` — estimate from both conditions' cells taken together.
          Swap-invariant up to the row order of the stacked union, and the
          union is denser than either condition, so the length scale shrinks
          with cell count alone.
        * ``"separate"`` — each condition estimates its own; nothing shared.

        Which to pick, and the measurements behind each:
        :meth:`kompot.differential.DifferentialExpression.fit` and
        :meth:`kompot.differential.DifferentialAbundance.fit`.
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
    param_scheme: Optional[str] = None

    def __post_init__(self):
        validate_positive_float(self.sigma, "sigma")
        validate_positive_float(self.ls, "ls", optional=True)
        validate_positive_float(self.ls_factor, "ls_factor")
        validate_param_scheme(self.param_scheme, optional=True)
        validate_positive_int(self.n_landmarks, "n_landmarks", optional=True)
        validate_bool(self.use_empirical_variance, "use_empirical_variance")
        validate_positive_int(self.batch_size, "batch_size", optional=True)
        validate_positive_float(self.eps, "eps")
        validate_bool(self.jit_compile, "jit_compile")
        if self.landmarks is not None:
            lm = np.asarray(self.landmarks)
            if lm.ndim != 2:
                raise ValueError(
                    f"'landmarks' must be a 2-D array "
                    f"(got {lm.ndim}-D with shape {lm.shape})."
                )
        if self.random_state is not None:
            if not isinstance(self.random_state, int) or isinstance(
                self.random_state, bool
            ):
                raise TypeError(
                    f"'random_state' must be an integer "
                    f"(got {type(self.random_state).__name__})."
                )


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

    def __post_init__(self):
        # null_genes: "auto", int >= 0, list of ints, or None
        ng = self.null_genes
        if ng is not None:
            if isinstance(ng, str):
                if ng != "auto":
                    raise ValueError(
                        f"'null_genes' string value must be 'auto' (got '{ng}')."
                    )
            elif isinstance(ng, (list, tuple)):
                for i, idx in enumerate(ng):
                    if not isinstance(idx, int) or isinstance(idx, bool):
                        raise TypeError(
                            f"'null_genes[{i}]' must be an integer "
                            f"(got {type(idx).__name__})."
                        )
                    if idx < 0:
                        raise ValueError(f"'null_genes[{i}]' must be >= 0 (got {idx}).")
            elif isinstance(ng, (int, np.integer)) and not isinstance(ng, bool):
                if ng < 0:
                    raise ValueError(f"'null_genes' must be >= 0 (got {ng}).")
            else:
                raise TypeError(
                    f"'null_genes' must be 'auto', an integer, a list of "
                    f"integers, or None (got {type(ng).__name__})."
                )

        validate_probability(self.threshold, "threshold")
        validate_bool(self.combine_with_internal, "combine_with_internal")

        if self.null_mahalanobis is not None and self.null_expression is not None:
            raise ValueError(
                "'null_mahalanobis' and 'null_expression' are mutually "
                "exclusive. Provide one or the other, not both."
            )


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

    def __post_init__(self):
        validate_non_negative_float(self.lfc_threshold, "lfc_threshold")
        validate_probability(self.ptp_threshold, "ptp_threshold")


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

    def __post_init__(self):
        validate_positive_int(self.min_cells, "min_cells")
        if self.min_percentage is not None:
            validate_positive_float(self.min_percentage, "min_percentage")
            if self.min_percentage > 100:
                raise ValueError(
                    f"'min_percentage' must be <= 100 (got {self.min_percentage})."
                )
        if self.check_representation is not None:
            validate_bool(self.check_representation, "check_representation")


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
        Keep the per-gene sample-variance covariance tensors out of memory by
        consuming them one gene at a time.  With ``dask`` installed each
        tensor is a lazy Dask graph and nothing is written to disk; without
        ``dask`` each condition's tensor is written as a memory-mapped
        ``.npy`` under ``disk_storage_dir``, which is slower because a gene
        slice of that layout is strided.  Defaults to ``None``, meaning *on if
        and only if* ``disk_storage_dir`` is set; nothing enables it
        automatically in response to memory pressure.  See
        https://kompot.readthedocs.io/en/latest/resource_planning.html
    disk_storage_dir : str, optional
        Directory for disk-backed arrays.  A real run creates it if missing,
        but ``dry_run=True`` raises ``FileNotFoundError`` on a path that does
        not exist, so create it before you plan against it.  When unset, the
        system temporary directory is used (honouring ``TMPDIR``), which on a
        shared cluster is often small or RAM-backed.
    max_memory_ratio : float
        Fraction of available RAM above which the resource estimator escalates
        its warnings.  It does **not** switch storage modes and does not cap
        allocation.
    """

    result_key: Optional[str] = None
    overwrite: Optional[bool] = None
    store_landmarks: bool = False
    store_posterior_covariance: bool = False
    store_additional_stats: bool = False
    store_arrays_on_disk: Optional[bool] = None
    disk_storage_dir: Optional[str] = None
    max_memory_ratio: float = 0.8

    def __post_init__(self):
        if self.result_key is not None and not isinstance(self.result_key, str):
            raise TypeError(
                f"'result_key' must be a string (got {type(self.result_key).__name__})."
            )
        if self.overwrite is not None:
            validate_bool(self.overwrite, "overwrite")
        validate_bool(self.store_landmarks, "store_landmarks")
        validate_bool(self.store_posterior_covariance, "store_posterior_covariance")
        validate_bool(self.store_additional_stats, "store_additional_stats")
        if self.store_arrays_on_disk is not None:
            validate_bool(self.store_arrays_on_disk, "store_arrays_on_disk")
        if self.disk_storage_dir is not None:
            if not isinstance(self.disk_storage_dir, str):
                raise TypeError(
                    f"'disk_storage_dir' must be a string "
                    f"(got {type(self.disk_storage_dir).__name__})."
                )
        validate_positive_float(self.max_memory_ratio, "max_memory_ratio")
        if self.max_memory_ratio > 1.0:
            raise ValueError(
                f"'max_memory_ratio' must be <= 1.0 (got {self.max_memory_ratio})."
            )


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

    def __post_init__(self):
        # Validate that predictors are callable when provided
        for name in (
            "function_predictor1",
            "function_predictor2",
            "obs_variance_predictor1",
            "obs_variance_predictor2",
            "variance_predictor1",
            "variance_predictor2",
            "density_predictor1",
            "density_predictor2",
        ):
            val = getattr(self, name)
            if val is not None and not callable(val):
                raise TypeError(
                    f"'{name}' must be callable (got {type(val).__name__})."
                )


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
        Return the results dict with lightweight null-distribution
        metadata (gene indices, names, seed, Mahalanobis distances)
        without the full expression matrices.  When
        ``return_full_results`` is also True, the null data
        additionally includes the full expression matrices.
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

    def __post_init__(self):
        validate_bool(self.copy, "copy")
        validate_bool(self.inplace, "inplace")
        validate_bool(self.return_full_results, "return_full_results")
        validate_bool(self.return_null_data, "return_null_data")
        validate_bool(self.compute_mahalanobis, "compute_mahalanobis")
        validate_bool(
            self.allow_single_condition_variance,
            "allow_single_condition_variance",
        )
        validate_bool(self.progress, "progress")
