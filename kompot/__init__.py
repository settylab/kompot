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
from .differential.expression_model import ExpressionModel
from .differential.sample_variance_estimator import SampleVarianceEstimator

# Import utility functions
from .utils import compute_mahalanobis_distance, find_landmarks
from .batch_utils import batch_process, apply_batched

# Import resource estimation utilities
from .resource_estimation import dry_run_differential_expression

# Import standalone FDR function
from .fdr import compute_fdr

# Import settings dataclasses
from .settings import (
    GPSettings,
    FDRSettings,
    DAThresholdSettings,
    FilterSettings,
    StorageSettings,
    OutputSettings,
    ModelSettings,
)

# Now import submodules - after the classes are imported
from . import plot
from . import anndata

# Export anndata functions
from .anndata import (
    de,
    da,
    compute_differential_abundance,
    compute_differential_expression,
    smooth_expression,
    compute_smoothed_expression,
    check_underrepresentation,
    RunInfo,
    RunComparison,
    cleanup,
    get_field_status,
    recompute_fdr,
    extract_null_distribution,
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

# The single console handler kompot installs above. configure_logging()
# retargets only this handler and never mutates foreign handlers that other
# code may attach to the non-propagating "kompot" logger (e.g. pytest's
# log-capture handlers, whose streams must stay StringIO buffers).
_console_handler = next(
    (h for h in logger.handlers if isinstance(h, logging.StreamHandler)), None
)


def configure_logging(stream=None):
    """Reconfigure the kompot logger to write to a different stream.

    Only kompot's own console handler is retargeted; any additional handlers
    attached to the ``kompot`` logger by other code are left untouched.

    Parameters
    ----------
    stream : file-like, optional
        Output stream for log messages. Defaults to ``sys.stdout``.
        CLI tools typically pass ``sys.stderr`` so stdout stays clean
        for machine-parseable output.
    """
    if stream is None:
        stream = sys.stdout
    for handler in logger.handlers:
        if handler is _console_handler:
            handler.setStream(stream)

__all__ = [
    # Version
    "__version__",
    # Mellon re-exports
    "DensityEstimator",
    "FunctionEstimator",
    "Predictor",
    # Core differential analysis classes
    "DifferentialAbundance",
    "DifferentialExpression",
    "ExpressionModel",
    "SampleVarianceEstimator",
    # AnnData interface
    "de",
    "da",
    # Settings dataclasses
    "GPSettings",
    "FDRSettings",
    "DAThresholdSettings",
    "FilterSettings",
    "StorageSettings",
    "OutputSettings",
    "ModelSettings",
    # Configuration
    "configure_logging",
    # Utility functions
    "compute_mahalanobis_distance",
    "find_landmarks",
    "batch_process",
    "apply_batched",
    # Resource estimation
    "dry_run_differential_expression",
    # Standalone FDR
    "compute_fdr",
    "recompute_fdr",
    "extract_null_distribution",
    # Expression smoothing
    "smooth_expression",
    "compute_smoothed_expression",
    # AnnData functionality (deprecated, use de()/da())
    "compute_differential_abundance",
    "compute_differential_expression",
    "check_underrepresentation",
    "RunInfo",
    "RunComparison",
    "cleanup",
    "get_field_status",
    # Submodules
    "plot",
    "anndata",
]
