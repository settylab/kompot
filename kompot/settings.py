"""Settings dataclasses for kompot's public API.

These group related parameters into discoverable objects with sensible
defaults.  Users can override individual fields while the rest stay at
their defaults::

    kompot.de(adata, "condition", "Young", "Old",
              gp=GPSettings(sigma=0.5))

All settings are optional — ``None`` means "use kompot's default".
"""

from dataclasses import dataclass, field, fields, asdict
from typing import Any, Dict, List, Optional, Union

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
        Number of landmarks for the Nyström approximation.
    landmarks : np.ndarray, optional
        Pre-computed landmark coordinates.
    use_empirical_variance : bool
        Estimate per-gene heteroscedastic noise from GP residuals.
    batch_size : int
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
    batch_size: int = 100
    eps: float = 1e-8
    jit_compile: bool = False
    random_state: Optional[int] = None


@dataclass
class FDRSettings:
    """False-discovery-rate and null-distribution parameters.

    Parameters
    ----------
    null_genes : int, List[int], None, or "auto"
        Number of null genes (or explicit indices).  ``"auto"`` uses
        2 000 when no ``sample_col`` is set and 0 otherwise.
    null_seed : int, optional
        Random seed for null-gene sampling.
    threshold : float
        FDR threshold for the ``is_de`` boolean column.
    """

    null_genes: Union[int, List[int], str, None] = "auto"
    null_seed: Optional[int] = 42
    threshold: float = 0.05


@dataclass
class FilterSettings:
    """Cell-filtering and group-subsetting parameters.

    Parameters
    ----------
    cell_filter : optional
        Specification for cells to include (boolean column name, dict
        of column→values, etc.).
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
    overwrite : bool, optional
        ``None`` (default) warns, ``True`` silently overwrites,
        ``False`` raises on conflict.
    store_landmarks : bool
        Persist landmarks in ``adata.uns`` for future reuse.
    store_posterior_covariance : bool
        Store the (n_cells × n_cells) posterior covariance in ``adata.obsp``.
    store_additional_stats : bool
        Store extra columns (p-values, tail FDR, PTP, z-scores).
    store_arrays_on_disk : bool, optional
        Use disk-backed arrays for large intermediate matrices.
    disk_storage_dir : str, optional
        Directory for disk-backed arrays.
    max_memory_ratio : float
        Fraction of RAM before triggering disk storage.
    """

    result_key: str = "kompot_de"
    overwrite: Optional[bool] = None
    store_landmarks: bool = False
    store_posterior_covariance: bool = False
    store_additional_stats: bool = False
    store_arrays_on_disk: Optional[bool] = None
    disk_storage_dir: Optional[str] = None
    max_memory_ratio: float = 0.8


@dataclass
class OutputSettings:
    """Control what ``de()`` returns and how it behaves.

    Parameters
    ----------
    copy : bool
        Return a copy of the AnnData instead of modifying in place.
    inplace : bool
        Write results into the AnnData object.
    return_full_results : bool
        Return the full results dict (model, table, landmarks, …).
    compute_mahalanobis : bool
        Compute per-gene Mahalanobis distances.
    allow_single_condition_variance : bool
        Allow sample-variance estimation when only one condition has
        multiple samples.
    progress : bool
        Show progress bars.
    """

    copy: bool = False
    inplace: bool = True
    return_full_results: bool = False
    compute_mahalanobis: bool = True
    allow_single_condition_variance: bool = False
    progress: bool = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _settings_to_kwargs(
    gp: Optional[GPSettings] = None,
    fdr: Optional[FDRSettings] = None,
    filter: Optional[FilterSettings] = None,
    storage: Optional[StorageSettings] = None,
    output: Optional[OutputSettings] = None,
) -> dict:
    """Flatten settings objects into the kwargs expected by
    ``compute_differential_expression``."""
    kw: dict = {}

    if gp is not None:
        kw["sigma"] = gp.sigma
        kw["ls"] = gp.ls
        kw["ls_factor"] = gp.ls_factor
        kw["n_landmarks"] = gp.n_landmarks
        kw["landmarks"] = gp.landmarks
        kw["use_empirical_variance"] = gp.use_empirical_variance
        kw["batch_size"] = gp.batch_size
        kw["eps"] = gp.eps
        kw["jit_compile"] = gp.jit_compile
        kw["random_state"] = gp.random_state

    if fdr is not None:
        kw["null_genes"] = fdr.null_genes
        kw["null_seed"] = fdr.null_seed
        kw["fdr_threshold"] = fdr.threshold

    if filter is not None:
        kw["cell_filter"] = filter.cell_filter
        kw["groups"] = filter.groups
        kw["min_cells"] = filter.min_cells
        kw["min_percentage"] = filter.min_percentage
        kw["check_representation"] = filter.check_representation

    if storage is not None:
        kw["result_key"] = storage.result_key
        kw["overwrite"] = storage.overwrite
        kw["store_landmarks"] = storage.store_landmarks
        kw["store_posterior_covariance"] = storage.store_posterior_covariance
        kw["store_additional_stats"] = storage.store_additional_stats
        kw["store_arrays_on_disk"] = storage.store_arrays_on_disk
        kw["disk_storage_dir"] = storage.disk_storage_dir
        kw["max_memory_ratio"] = storage.max_memory_ratio

    if output is not None:
        kw["copy"] = output.copy
        kw["inplace"] = output.inplace
        kw["return_full_results"] = output.return_full_results
        kw["compute_mahalanobis"] = output.compute_mahalanobis
        kw["allow_single_condition_variance"] = output.allow_single_condition_variance
        kw["progress"] = output.progress

    return kw
