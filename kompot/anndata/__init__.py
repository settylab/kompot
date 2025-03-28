"""
AnnData integration for Kompot.
"""

from .differential_abundance import compute_differential_abundance
from .differential_expression import compute_differential_expression
from .utils import RunInfo, RunComparison

__all__ = [
    "compute_differential_abundance",
    "compute_differential_expression",
    "RunInfo",
    "RunComparison"
]
