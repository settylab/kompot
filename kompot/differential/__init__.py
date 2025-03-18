"""
Differential analysis module for Kompot.

This module provides classes for differential abundance and expression analysis.
"""

from kompot.differential.differential_abundance import DifferentialAbundance
from kompot.differential.differential_expression import DifferentialExpression
from kompot.differential.sample_variance_estimator import SampleVarianceEstimator
from kompot.differential.utils import compute_weighted_mean_fold_change

__all__ = [
    "DifferentialAbundance",
    "DifferentialExpression",
    "SampleVarianceEstimator",
    "compute_weighted_mean_fold_change"
]