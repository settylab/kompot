"""Deprecated aliases for expression smoothing (formerly imputation).

.. deprecated::
    Use :mod:`kompot.anndata.smooth` instead. This module will be removed
    in a future version.
"""

import warnings
from typing import Optional, Union, Dict, Any, List

from .smooth import smooth_expression, compute_smoothed_expression  # noqa: F401


def impute_expression(
    adata,
    groupby=None,
    condition=None,
    obsm_key="DM_EigenVectors",
    layer=None,
    genes=None,
    sample_col=None,
    gp=None,
    storage=None,
    output=None,
    model=None,
    **function_kwargs,
):
    """Deprecated. Use :func:`kompot.smooth_expression` instead."""
    warnings.warn(
        "impute_expression() is deprecated and will be removed in a future "
        "version. Use kompot.smooth_expression() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return smooth_expression(
        adata,
        groupby=groupby,
        condition=condition,
        obsm_key=obsm_key,
        layer=layer,
        genes=genes,
        sample_col=sample_col,
        gp=gp,
        storage=storage,
        output=output,
        model=model,
        **function_kwargs,
    )


def compute_imputed_expression(*args, **kwargs):
    """Deprecated. Use :func:`kompot.smooth_expression` instead."""
    warnings.warn(
        "compute_imputed_expression() is deprecated and will be removed "
        "in a future version. Use kompot.smooth_expression() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return smooth_expression(*args, **kwargs)
