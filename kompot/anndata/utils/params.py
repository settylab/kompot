"""Serialize and reconstruct Settings dataclasses for run history.

This module is the single source of truth for converting between
Settings objects and the JSON-serializable dicts stored in
``adata.uns[...]['run_history']``.
"""

from dataclasses import asdict, fields
from typing import Any, Dict, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Serialization (Settings → dict for storage)
# ---------------------------------------------------------------------------


def build_params_dict(
    top_level: Dict[str, Any],
    extra_kwargs: Optional[Dict[str, Any]] = None,
    landmarks_provided: bool = False,
    **settings,
) -> Dict[str, Any]:
    """Build a JSON-safe params dict from top-level args and Settings objects.

    Parameters
    ----------
    top_level : dict
        Direct arguments (groupby, condition1, condition2, obsm_key, …).
    extra_kwargs : dict, optional
        Mellon passthrough kwargs (``density_kwargs`` / ``function_kwargs``).
    landmarks_provided : bool
        Whether landmarks were provided (the array itself is stored
        separately; we only record a boolean here).
    **settings
        Keyword arguments whose values are Settings dataclass instances
        (e.g. ``gp=GPSettings(…)``, ``fdr=FDRSettings(…)``).

    Returns
    -------
    dict
        ``{"groupby": …, "gp": {…}, "fdr": {…}, …}`` — ready for JSON
        serialization via :func:`kompot.anndata.utils.json_utils.to_json_string`.
    """
    params: Dict[str, Any] = dict(top_level)
    for name, obj in settings.items():
        d = _settings_to_dict(obj)
        params[name] = d
    if landmarks_provided and "gp" in params:
        params["gp"]["landmarks"] = True
    if extra_kwargs:
        params["extra_kwargs"] = extra_kwargs
    return params


def _settings_to_dict(obj) -> Dict[str, Any]:
    """Convert a single Settings dataclass to a JSON-safe dict.

    numpy arrays and other non-JSON types are replaced with a
    human-readable placeholder.
    """
    d = asdict(obj)
    # Replace non-serializable values
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            d[k] = f"<array shape={v.shape}>"
        elif isinstance(v, (tuple, list)) and any(isinstance(x, np.ndarray) for x in v):
            d[k] = f"<tuple of {len(v)} arrays>"
    return d


# ---------------------------------------------------------------------------
# Reconstruction (stored dict → Settings objects / call_args)
# ---------------------------------------------------------------------------

# Maps settings key → (Settings class, {stored_name: field_name})
# Only list renames; identical names are mapped automatically.
_SETTINGS_REGISTRY = {}


def _ensure_registry():
    """Populate ``_SETTINGS_REGISTRY`` on first use (avoids circular imports)."""
    if _SETTINGS_REGISTRY:
        return
    from ...settings import (
        GPSettings,
        FDRSettings,
        DAThresholdSettings,
        FilterSettings,
        StorageSettings,
        OutputSettings,
    )

    _SETTINGS_REGISTRY.update(
        {
            "gp": GPSettings,
            "fdr": FDRSettings,
            "threshold": DAThresholdSettings,
            "filter": FilterSettings,
            "storage": StorageSettings,
            "output": OutputSettings,
        }
    )


def reconstruct_settings(params: Dict[str, Any]) -> Dict[str, Any]:
    """Reconstruct Settings objects from a stored params dict.

    Works with both the new nested format and the legacy flat format.

    Parameters
    ----------
    params : dict
        The ``params`` dict from ``run_info``.

    Returns
    -------
    dict
        ``{"gp": GPSettings(…), "fdr": FDRSettings(…), …}``
        Only keys present in *params* are returned.
    """
    _ensure_registry()
    result = {}
    for key, cls in _SETTINGS_REGISTRY.items():
        if key in params and isinstance(params[key], dict):
            valid_fields = {f.name for f in fields(cls)}
            filtered = {k: v for k, v in params[key].items() if k in valid_fields}
            # Replace placeholder strings for non-reconstructible values
            for k, v in filtered.items():
                if isinstance(v, str) and v.startswith("<array"):
                    filtered[k] = None
            result[key] = cls(**filtered)
    return result


def params_to_call_args(
    params: Dict[str, Any],
    analysis_type: str,
) -> Dict[str, Any]:
    """Convert stored params into kwargs suitable for ``da()`` / ``de()``.

    The returned dict can be unpacked directly::

        kwargs = run_info.call_args()
        kwargs["fdr"].threshold = 0.01  # edit in place
        kompot.de(adata, **kwargs)

    Parameters
    ----------
    params : dict
        The ``params`` dict from ``run_info``.
    analysis_type : str
        ``"da"`` or ``"de"``.

    Returns
    -------
    dict
        Ready for ``kompot.da(adata, **result)`` or
        ``kompot.de(adata, **result)``.
    """
    settings = reconstruct_settings(params)
    kwargs: Dict[str, Any] = {}

    # Top-level args
    for key in (
        "groupby",
        "condition1",
        "condition2",
        "obsm_key",
        "sample_col",
        "layer",
        "genes",
    ):
        if key in params:
            kwargs[key] = params[key]

    # Settings objects
    for key, obj in settings.items():
        kwargs[key] = obj

    # Extra mellon kwargs
    extra = params.get("extra_kwargs")
    if extra:
        kwargs.update(extra)

    return kwargs


# ---------------------------------------------------------------------------
# Backward-compat helper for reading previous-run params
# ---------------------------------------------------------------------------

# Maps flat legacy key → (settings_group, field_name)
_LEGACY_MAP = {
    # GP
    "sigma": ("gp", "sigma"),
    "ls": ("gp", "ls"),
    "ls_factor": ("gp", "ls_factor"),
    "n_landmarks": ("gp", "n_landmarks"),
    "use_empirical_variance": ("gp", "use_empirical_variance"),
    "batch_size": ("gp", "batch_size"),
    "eps": ("gp", "eps"),
    "jit_compile": ("gp", "jit_compile"),
    "random_state": ("gp", "random_state"),
    # DA threshold
    "log_fold_change_threshold": ("threshold", "lfc_threshold"),
    "ptp_threshold": ("threshold", "ptp_threshold"),
    # FDR
    "null_genes": ("fdr", "null_genes"),
    "null_seed": ("fdr", "null_seed"),
    "fdr_threshold": ("fdr", "threshold"),
    # Filter
    "cell_filter": ("filter", "cell_filter"),
    "groups": ("filter", "groups"),
    "min_cells": ("filter", "min_cells"),
    "min_percentage": ("filter", "min_percentage"),
    "check_representation": ("filter", "check_representation"),
    # Storage
    "result_key": ("storage", "result_key"),
    "overwrite": ("storage", "overwrite"),
    "store_landmarks": ("storage", "store_landmarks"),
    "store_posterior_covariance": ("storage", "store_posterior_covariance"),
    "store_additional_stats": ("storage", "store_additional_stats"),
    "store_arrays_on_disk": ("storage", "store_arrays_on_disk"),
    "disk_storage_dir": ("storage", "disk_storage_dir"),
    "max_memory_ratio": ("storage", "max_memory_ratio"),
    # Output
    "copy": ("output", "copy"),
    "inplace": ("output", "inplace"),
    "return_full_results": ("output", "return_full_results"),
    "compute_mahalanobis": ("output", "compute_mahalanobis"),
    "allow_single_condition_variance": ("output", "allow_single_condition_variance"),
    "progress": ("output", "progress"),
}


def params_get(params: Dict[str, Any], key: str, default=None):
    """Read a parameter from either nested or legacy flat format.

    Use this in helpers that compare current vs previous run params.
    """
    # Top-level keys (present in both formats)
    if key in params and not isinstance(params[key], dict):
        return params[key]

    # Nested format: look up via registry
    if key in _LEGACY_MAP:
        group, field_name = _LEGACY_MAP[key]
        if group in params and isinstance(params[group], dict):
            return params[group].get(field_name, default)

    # Legacy flat format: direct lookup
    if key in params:
        return params[key]

    # Derived keys
    if key == "use_sample_variance":
        return params.get("sample_col") is not None

    return default
