"""
AnnData integration for Kompot.
"""

# Import submodules so they are accessible as attributes
# (e.g. kompot.anndata.differential_abundance, kompot.anndata.utils).
# Required on Python 3.10 for unittest.mock.patch to resolve module paths.
from . import differential_abundance  # noqa: F401
from . import differential_expression  # noqa: F401
from . import utils  # noqa: F401

from .differential_abundance import da, compute_differential_abundance
from .differential_expression import de, compute_differential_expression
from .impute import impute_expression, compute_imputed_expression
from .utils import RunInfo, RunComparison, check_underrepresentation
from .cleanup import cleanup, get_field_status
from .fdr_utils import recompute_fdr, extract_null_distribution

__all__ = [
    "de",
    "da",
    "compute_differential_abundance",
    "compute_differential_expression",
    "impute_expression",
    "compute_imputed_expression",
    "RunInfo",
    "RunComparison",
    "check_underrepresentation",
    "cleanup",
    "get_field_status",
    "recompute_fdr",
    "extract_null_distribution",
]
