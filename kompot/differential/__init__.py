"""
Differential analysis module for Kompot.

This module provides classes for differential abundance and expression analysis.
"""

from .differential_abundance import DifferentialAbundance
from .differential_expression import DifferentialExpression
from .sample_variance_estimator import SampleVarianceEstimator
#from .utils import compute_weighted_mean_fold_change, update_direction_column
from .utils import  update_direction_column

__all__ = [
    "DifferentialAbundance",
    "DifferentialExpression",
    "SampleVarianceEstimator",
    #"compute_weighted_mean_fold_change",
    "update_direction_column"
]