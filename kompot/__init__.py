"""
Kompot: A package for differential abundance and gene expression analysis
using Mahalanobis distance with JAX backend.
"""

import logging.config
import sys
from typing import Optional, Union, Dict, Any, List, Tuple

import numpy as np
import pandas as pd

from kompot.version import __version__

# Re-export Mellon tools directly
from mellon import DensityEstimator, FunctionEstimator, Predictor

# Export Kompot's additional functionality
from kompot.differential import DifferentialAbundance, DifferentialExpression
from kompot.anndata import (
    compute_differential_abundance,
    compute_differential_expression,
    run_differential_analysis,
    generate_report
)

# Import plot module
from kompot import plot

# Add docstring for clarity in import statements
DifferentialAbundance.__doc__ = """
Compute differential abundance between two conditions.

This class analyzes differences in cell density between two conditions
using density estimation and fold change analysis. The class now tracks
which cells belong to which condition to facilitate proper interpretation
of results.

Key features:
- Density estimation using Gaussian processes (via Mellon)
- Log fold change computation with uncertainty
- Cell ordering tracking for accurate result interpretation
- Methods to access condition-specific results (get_condition1_results(), get_condition2_results())
"""

DifferentialExpression.__doc__ = """
Compute differential expression between two conditions.

This class analyzes differences in gene expression between two conditions
using imputation, Mahalanobis distance, and log fold change analysis.

Key features:
- Expression imputation using Gaussian processes (via Mellon)
- Log fold change computation with uncertainty
- Mahalanobis distance for gene ranking
- Support for weighted fold change using density information
- Improved cell ordering handling to prevent mixing conditions
- Support for condition-specific result extraction using cell_condition_labels
"""
from kompot.utils import compute_mahalanobis_distance, find_landmarks
from kompot.reporter import HTMLReporter
from kompot.batch_utils import batch_process, apply_batched

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
            "level": "INFO",
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
    "DensityEstimator", "FunctionEstimator", "Predictor", 
    "DifferentialAbundance", "DifferentialExpression",
    "compute_mahalanobis_distance", "find_landmarks",
    "HTMLReporter", "generate_report", "__version__",
    "compute_differential_abundance", "compute_differential_expression",
    "run_differential_analysis", 
    "batch_process", "apply_batched",
    "plot"
]

