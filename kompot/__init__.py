"""
Kompot: A package for differential abundance and gene expression analysis
using Mahalanobis distance with JAX backend.
"""

import logging.config
import sys
from typing import Optional, Union, Dict, Any, List, Tuple

import numpy as np
import pandas as pd

from .version import __version__

# Re-export Mellon tools directly
from mellon import DensityEstimator, FunctionEstimator, Predictor

# Set mellon logger level to warning to reduce verbosity
import mellon
import logging
mellon.logger.setLevel(logging.WARNING)

# Import core functionality directly - using relative imports
from .differential.differential_abundance import DifferentialAbundance
from .differential.differential_expression import DifferentialExpression
from .differential.expression_model import ExpressionModel, Matern52Linear
from .differential.sample_variance_estimator import SampleVarianceEstimator

# Import utility functions
from .utils import compute_mahalanobis_distance, find_landmarks
from .batch_utils import batch_process, apply_batched

# Import resource estimation utilities
from .resource_estimation import dry_run_differential_expression

# Import settings dataclasses
from .settings import (
    GPSettings,
    FDRSettings,
    FilterSettings,
    StorageSettings,
    OutputSettings,
    _settings_to_kwargs,
)

# Now import submodules - after the classes are imported
from . import plot
from . import anndata

# Export anndata functions
from .anndata import (
    compute_differential_abundance,
    compute_differential_expression,
    impute_expression,
    check_underrepresentation,
    RunInfo,
    RunComparison,
    cleanup,
    get_field_status
)


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
    **function_kwargs,
):
    """Run differential expression analysis on an AnnData object.

    The most common call is just::

        kompot.de(adata, "condition", "Young", "Old")

    Advanced options are available through the settings dataclasses
    (:class:`GPSettings`, :class:`FDRSettings`, :class:`FilterSettings`,
    :class:`StorageSettings`, :class:`OutputSettings`).  Any field left
    at its default is equivalent to omitting it entirely.  Extra
    ``**function_kwargs`` are forwarded to mellon's
    :class:`~mellon.FunctionEstimator`.

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
        GP model parameters.
    fdr : FDRSettings, optional
        FDR / null-distribution parameters.
    filter : FilterSettings, optional
        Cell filtering and group-subsetting.
    storage : StorageSettings, optional
        Where and how results are stored.
    output : OutputSettings, optional
        Return-value and progress-bar control.
    **function_kwargs
        Forwarded to :class:`~mellon.FunctionEstimator`.

    Returns
    -------
    See :func:`compute_differential_expression`.
    """
    kw = _settings_to_kwargs(gp=gp, fdr=fdr, filter=filter,
                             storage=storage, output=output)
    return compute_differential_expression(
        adata,
        groupby=groupby,
        condition1=condition1,
        condition2=condition2,
        obsm_key=obsm_key,
        layer=layer,
        genes=genes,
        sample_col=sample_col,
        **kw,
        **function_kwargs,
    )

# Configure logging
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "[%(asctime)s] [%(levelname)-8s] %(message)s",
        },
    },
    "handlers": {
        "console": {
            "level": "DEBUG",
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "stream": sys.stdout,
        },
    },
    "loggers": {
        "kompot": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
    },
}

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("kompot")

__all__ = [
    # Version
    "__version__",

    # Mellon re-exports
    "DensityEstimator", "FunctionEstimator", "Predictor",

    # Core differential analysis classes
    "DifferentialAbundance", "DifferentialExpression", "ExpressionModel", "Matern52Linear", "SampleVarianceEstimator",

    # AnnData interface
    "de",

    # Settings dataclasses
    "GPSettings", "FDRSettings", "FilterSettings", "StorageSettings", "OutputSettings",

    # Utility functions
    "compute_mahalanobis_distance", "find_landmarks",
    "batch_process", "apply_batched",

    # Resource estimation
    "dry_run_differential_expression",

    # AnnData functionality
    "compute_differential_abundance", "compute_differential_expression",
    "impute_expression",
    "check_underrepresentation", "RunInfo", "RunComparison",
    "cleanup", "get_field_status",

    # Submodules
    "plot", "anndata"
]

