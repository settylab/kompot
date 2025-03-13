"""
AnnData integration for Kompot.
"""

from .differential_abundance import compute_differential_abundance
from .differential_expression import compute_differential_expression
from .workflows import run_differential_analysis

__all__ = [
    "compute_differential_abundance",
    "compute_differential_expression",
    "run_differential_analysis"
]