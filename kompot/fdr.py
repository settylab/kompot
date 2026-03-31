"""Standalone FDR computation from Mahalanobis distances.

This module provides a public function for computing FDR statistics
without requiring an AnnData object or running the full DE pipeline.
"""

import numpy as np
import pandas as pd
from typing import List, Optional

from .anndata.fdr_utils import compute_fdr_statistics


def compute_fdr(
    real_mahalanobis: np.ndarray,
    null_mahalanobis: np.ndarray,
    threshold: float = 0.05,
    gene_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compute FDR from real and null Mahalanobis distances.

    Parameters
    ----------
    real_mahalanobis : np.ndarray
        Mahalanobis distances for real (test) genes.
    null_mahalanobis : np.ndarray
        Mahalanobis distances from a null distribution.
    threshold : float
        FDR threshold for the ``is_de`` boolean column.
    gene_names : list of str, optional
        Gene names for the DataFrame index.  If ``None``, uses
        integer indices.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ``mahalanobis``, ``pvalue``,
        ``local_fdr``, ``tail_fdr``, ``is_de``.
    """
    real_mahalanobis = np.asarray(real_mahalanobis, dtype=float)
    null_mahalanobis = np.asarray(null_mahalanobis, dtype=float)

    pvalues, local_fdr, tail_fdr, is_significant = compute_fdr_statistics(
        real_mahalanobis=real_mahalanobis,
        null_mahalanobis=null_mahalanobis,
        fdr_threshold=threshold,
    )

    index = gene_names if gene_names is not None else range(len(real_mahalanobis))

    return pd.DataFrame(
        {
            "mahalanobis": real_mahalanobis,
            "pvalue": pvalues,
            "local_fdr": local_fdr,
            "tail_fdr": tail_fdr,
            "is_de": is_significant,
        },
        index=index,
    )
