"""
AnnData integration for Kompot.
"""

from .functions import (
    compute_differential_abundance,
    compute_differential_expression,
    run_differential_analysis,
    generate_report
)

__all__ = [
    "compute_differential_abundance",
    "compute_differential_expression",
    "run_differential_analysis",
    "generate_report"
]