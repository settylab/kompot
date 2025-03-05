"""
Kompot: A package for differential abundance and gene expression analysis
using Mahalanobis distance with JAX backend.
"""

from kompot.version import __version__

# Re-export Mellon tools directly
from mellon import DensityEstimator, FunctionEstimator, Predictor

# Export Kompot's additional functionality
from kompot.differential import DifferentialAbundance, DifferentialExpression
from kompot.utils import compute_mahalanobis_distance, find_landmarks