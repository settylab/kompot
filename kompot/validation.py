"""Input validation for kompot's public API.

Provides two layers:
1. Scalar/type validators (re-exported from mellon + kompot-specific).
2. AnnData cross-cutting checks used at the top of ``de()``, ``da()``,
   ``smooth_expression()``.
"""

import logging
from math import isnan

import numpy as np
from mellon.validation import (  # noqa: F401
    validate_positive_float,
    validate_positive_int,
    validate_float,
    validate_bool,
    validate_string,
    validate_array,
)

logger = logging.getLogger("kompot")


def validate_probability(value, param_name, optional=False):
    """Validate that *value* is a float in (0, 1]."""
    if value is None and optional:
        return None
    value = validate_positive_float(value, param_name)
    if value > 1.0:
        raise ValueError(f"'{param_name}' must be between 0 and 1 (got {value}).")
    return value


def validate_non_negative_int(value, param_name, optional=False):
    """Validate that *value* is a non-negative integer."""
    if value is None and optional:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(
            f"'{param_name}' must be an integer (got {type(value).__name__})."
        )
    if value < 0:
        raise ValueError(f"'{param_name}' must be >= 0 (got {value}).")
    return value


def validate_non_negative_float(value, param_name, optional=False):
    """Validate that *value* is a non-negative float."""
    if value is None and optional:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(
            f"'{param_name}' must be a number (got {type(value).__name__})."
        )
    if isnan(value):
        raise ValueError(f"'{param_name}' must not be NaN.")
    if value < 0:
        raise ValueError(f"'{param_name}' must be >= 0 (got {value}).")
    return float(value)


# ---------------------------------------------------------------------------
# AnnData-level validation
# ---------------------------------------------------------------------------


def validate_obsm_key(adata, obsm_key):
    """Check that *obsm_key* exists and has the right shape."""
    if obsm_key not in adata.obsm:
        avail = list(adata.obsm.keys())
        msg = f"Key '{obsm_key}' not found in adata.obsm. Available keys: {avail}"
        if obsm_key == "DM_EigenVectors":
            msg += (
                "\n\nTo compute DM_EigenVectors (diffusion map eigenvectors), "
                "use the Palantir package:\n"
                "```python\n"
                "import palantir\n"
                "palantir.utils.run_diffusion_maps(adata)\n"
                "```\n"
                "See https://github.com/dpeerlab/Palantir for installation.\n\n"
                "Alternatively, specify a different obsm_key that exists in "
                "your dataset, such as 'X_pca'."
            )
        raise ValueError(msg)

    X = adata.obsm[obsm_key]
    if X.shape[0] != adata.n_obs:
        raise ValueError(
            f"adata.obsm['{obsm_key}'] has {X.shape[0]} rows but adata "
            f"has {adata.n_obs} observations. Shape mismatch."
        )
    if X.ndim != 2:
        raise ValueError(
            f"adata.obsm['{obsm_key}'] must be 2-D "
            f"(got {X.ndim}-D array with shape {X.shape})."
        )


def validate_groupby(adata, groupby):
    """Check that *groupby* column exists in adata.obs."""
    if groupby not in adata.obs:
        raise ValueError(
            f"Column '{groupby}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )


def validate_conditions(adata, groupby, condition1, condition2):
    """Check that both conditions exist and are distinct."""
    if condition1 == condition2:
        raise ValueError(
            f"condition1 and condition2 must be different (both are '{condition1}')."
        )
    obs_col = adata.obs[groupby]
    if (obs_col == condition1).sum() == 0:
        unique = sorted(obs_col.unique().tolist())
        raise ValueError(
            f"Condition '{condition1}' not found in column '{groupby}'. "
            f"Available values: {unique}"
        )
    if (obs_col == condition2).sum() == 0:
        unique = sorted(obs_col.unique().tolist())
        raise ValueError(
            f"Condition '{condition2}' not found in column '{groupby}'. "
            f"Available values: {unique}"
        )


def validate_layer(adata, layer):
    """Check that *layer* exists in adata.layers."""
    if layer is not None and layer not in adata.layers:
        raise ValueError(
            f"Layer '{layer}' not found in adata.layers. "
            f"Available layers: {list(adata.layers.keys())}"
        )


def validate_genes(adata, genes):
    """Check that *genes* exist in adata.var_names."""
    if genes is None:
        return
    if len(genes) == 0:
        raise ValueError("genes list is empty. Provide at least one gene.")
    missing = [g for g in genes if g not in adata.var_names]
    if missing:
        raise ValueError(
            f"The following genes were not found in adata.var_names: "
            f"{missing[:10]}"
            + (f"... and {len(missing) - 10} more" if len(missing) > 10 else "")
        )


def validate_sample_col(adata, sample_col):
    """Check that *sample_col* exists in adata.obs."""
    if sample_col is not None and sample_col not in adata.obs:
        raise ValueError(
            f"Column '{sample_col}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )


def validate_expression_data(expr, context="expression data"):
    """Warn on NaN/Inf in expression matrices (before GP fitting)."""
    if expr is None:
        return
    arr = np.asarray(expr)
    n_nan = np.isnan(arr).sum()
    n_inf = np.isinf(arr).sum()
    if n_nan > 0:
        raise ValueError(
            f"{context} contains {n_nan:,} NaN values. "
            "GP fitting will fail. Check your input data or "
            "pre-filter genes/cells with missing values."
        )
    if n_inf > 0:
        raise ValueError(
            f"{context} contains {n_inf:,} Inf values. "
            "GP fitting will fail. Check your input data for "
            "overflow or log-transform issues."
        )


def validate_landmarks_shape(landmarks, obsm_key, n_features):
    """Check that pre-computed landmarks match obsm feature count."""
    if landmarks is None:
        return
    landmarks = np.asarray(landmarks)
    if landmarks.ndim != 2:
        raise ValueError(
            f"landmarks must be a 2-D array "
            f"(got {landmarks.ndim}-D with shape {landmarks.shape})."
        )
    if landmarks.shape[1] != n_features:
        raise ValueError(
            f"landmarks have {landmarks.shape[1]} features but "
            f"adata.obsm['{obsm_key}'] has {n_features}. "
            f"Landmarks must match the cell-state coordinate dimensions."
        )
