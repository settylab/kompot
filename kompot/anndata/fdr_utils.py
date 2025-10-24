"""
FDR utilities for differential expression analysis.

Provides local and tail-based FDR estimation similar to R's fdrtool package,
specifically designed for Mahalanobis distance-based differential expression.
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional, Union, Dict, Any, List, Tuple
from scipy import stats

logger = logging.getLogger("kompot")


def prepare_null_genes(
    null_genes: Union[int, List[int], None], available_genes: List[str], null_seed: Optional[int]
) -> Tuple[List[int], bool]:
    """
    Select null genes for FDR calculation.

    Args:
        null_genes: Specification of null genes (int for random sampling, list for specific genes, None to disable)
        available_genes: List of available gene names
        null_seed: Random seed for reproducible selection

    Returns:
        (null_gene_indices, used_replacement): List of gene indices and whether sampling with replacement was used
    """
    if null_genes is None or null_genes == 0:
        return [], False

    n_available = len(available_genes)

    if isinstance(null_genes, int):
        if null_genes <= 0:
            raise ValueError(f"null_genes must be positive, got {null_genes}")

        rng = np.random.RandomState(null_seed)

        if null_genes > n_available:
            logger.warning(
                f"Requested {null_genes} null genes but only {n_available} genes available. "
                f"Using sampling with replacement."
            )
            null_gene_indices = rng.choice(n_available, size=null_genes, replace=True).tolist()
            used_replacement = True
        else:
            null_gene_indices = rng.choice(n_available, size=null_genes, replace=False).tolist()
            used_replacement = False

    elif isinstance(null_genes, list):
        null_gene_indices = null_genes.copy()
        used_replacement = False

        if not all(isinstance(idx, int) for idx in null_gene_indices):
            raise ValueError("All elements in null_genes list must be integers")

        invalid_indices = [idx for idx in null_gene_indices if idx < 0 or idx >= n_available]
        if invalid_indices:
            raise ValueError(
                f"Invalid gene indices: {invalid_indices}. Must be between 0 and {n_available-1}"
            )

    else:
        raise ValueError(f"null_genes must be int, list of ints, or None, got {type(null_genes)}")

    return null_gene_indices, used_replacement


def generate_shuffled_expression(
    expr1: np.ndarray, expr2: np.ndarray, null_gene_indices: List[int], null_seed: Optional[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate shuffled expression matrices for null genes.

    Each gene gets differently shuffled expression values between conditions to break
    the association between cell state and gene expression.

    Args:
        expr1: Expression matrix for condition 1 (cells x genes)
        expr2: Expression matrix for condition 2 (cells x genes)
        null_gene_indices: Indices of genes to use for null distribution
        null_seed: Random seed for reproducible shuffling

    Returns:
        (shuffled_expr1, shuffled_expr2): Expression matrices with shuffled values between conditions
    """
    if not null_gene_indices:
        return np.empty((expr1.shape[0], 0)), np.empty((expr2.shape[0], 0))

    # Combine expression matrices
    combined_expr = np.vstack([expr1, expr2])
    n_cells_1 = expr1.shape[0]
    n_cells_2 = expr2.shape[0]

    # Initialize output arrays for shuffled null genes
    shuffled_expr_combined = np.zeros((n_cells_1 + n_cells_2, len(null_gene_indices)))

    # Set up base random state
    base_rng = np.random.RandomState(null_seed)

    # For each null gene, create a differently shuffled version
    for i, gene_idx in enumerate(null_gene_indices):
        # Create a unique random state for this gene instance
        gene_seed = base_rng.randint(0, 2**31 - 1)
        gene_rng = np.random.RandomState(gene_seed)

        # Get expression values for this gene from both conditions
        gene_expr = combined_expr[:, gene_idx].copy()

        # Shuffle the expression values to break condition-expression association
        # This creates a null distribution where expression is random w.r.t. conditions
        shuffled_expr_combined[:, i] = gene_rng.permutation(gene_expr)

    # Split back into condition-specific matrices
    shuffled_expr1 = shuffled_expr_combined[:n_cells_1, :]
    shuffled_expr2 = shuffled_expr_combined[n_cells_1:, :]

    return shuffled_expr1, shuffled_expr2


def compute_fdr_statistics(
    real_mahalanobis: np.ndarray, null_mahalanobis: np.ndarray, fdr_threshold: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute p-values, local FDR, and tail-based FDR from null distribution of Mahalanobis distances.

    Note: Larger Mahalanobis distances indicate MORE significance (deviation from null).
    Mahalanobis distances are always non-negative.

    Args:
        real_mahalanobis: Mahalanobis distances for real genes (non-negative)
        null_mahalanobis: Mahalanobis distances for null genes (background)
        fdr_threshold: FDR threshold for significance

    Returns:
        (pvalues, local_fdr_values, tail_fdr_values, is_significant): P-values, local FDR, tail-based FDR, and boolean significance
    """
    from statsmodels.stats.multitest import local_fdr, multipletests

    # Step 1: Compute empirical p-values from null distribution
    # For Mahalanobis distances: larger distance = more extreme = smaller p-value
    pvalues = np.zeros(len(real_mahalanobis))

    for i, real_dist in enumerate(real_mahalanobis):
        # Right-tailed p-value: fraction of null distances >= real distance
        # This is correct because larger Mahalanobis = more significant
        pvalues[i] = np.sum(null_mahalanobis >= real_dist) / len(null_mahalanobis)

    # Ensure p-values are not exactly 0 for downstream calculations
    # Only add pseudocount for truly zero p-values
    zero_pvalues = pvalues == 0.0
    if np.any(zero_pvalues):
        # Use a much smaller minimum to allow very small FDR values
        # Handle case where ALL p-values are zero
        non_zero_pvalues = pvalues[~zero_pvalues]
        if len(non_zero_pvalues) > 0:
            min_pvalue = np.min(non_zero_pvalues)
        else:
            # All p-values are zero - use 1/number_of_nulls as minimum
            min_pvalue = 1.0 / len(null_mahalanobis)
            logger.debug(f"All {len(pvalues)} p-values are zero, using default min_pvalue={min_pvalue}")
        pvalues[zero_pvalues] = min_pvalue
        logger.debug(f"Set minimum p-value to {min_pvalue} for {np.sum(zero_pvalues)} zero p-values")

    # Step 2: Compute tail-based FDR using Benjamini-Hochberg
    _, tail_fdr_values, _, _ = multipletests(pvalues, method="fdr_bh")

    # Step 3: Convert to appropriate statistics for local FDR calculation
    # For local FDR, we need z-scores. Convert using inverse normal CDF
    # Since larger Mahalanobis = smaller p-value = larger |z-score|
    zscores = -stats.norm.ppf(pvalues)  # One-tailed conversion

    # Handle potential infinite z-scores by clipping, but use a larger range
    # to allow for very small p-values and correspondingly small FDR values
    zscores = np.clip(zscores, -20, 20)

    # Step 4: Apply local FDR estimation (similar to fdrtool approach)
    try:
        # Local FDR estimation with empirical null modeling
        local_fdr_values = local_fdr(
            zscores,
            null_proportion=1.0,  # Assume theoretical null proportion
            null_pdf=None,  # Let it estimate empirical null distribution
            deg=7,  # Polynomial degree for spline fitting (fdrtool default)
            nbins=500,  # Number of bins for density estimation
            alpha=0,  # No higher criticism threshold
        )

        # Ensure local FDR values are valid probabilities
        local_fdr_values = np.clip(local_fdr_values, 0, 1)

        # Fix edge case: when all p-values are very high (no signal), local FDR can incorrectly return 0
        # In such cases, use tail-based FDR as more reliable
        high_pvalue_mask = pvalues > 0.9
        if np.all(high_pvalue_mask):
            logger.debug("All p-values > 0.9, using tail-based FDR instead of local FDR")
            local_fdr_values = tail_fdr_values.copy()
        elif np.any(high_pvalue_mask):
            # For individual high p-values, ensure local FDR is at least as high as tail FDR
            problematic = high_pvalue_mask & (local_fdr_values < tail_fdr_values)
            if np.any(problematic):
                local_fdr_values[problematic] = tail_fdr_values[problematic]
                logger.debug(
                    f"Fixed {np.sum(problematic)} genes where local FDR was lower than tail FDR for high p-values"
                )

    except Exception as e:
        logger.warning(f"Local FDR estimation failed: {e}. Using tail-based FDR as fallback.")
        # Fallback to tail-based FDR if local FDR computation fails
        local_fdr_values = tail_fdr_values.copy()

    # Step 5: Determine significance based on local FDR threshold (more conservative)
    # Use local FDR as primary significance criterion
    is_significant = local_fdr_values < fdr_threshold

    return pvalues, local_fdr_values, tail_fdr_values, is_significant


def annotate_differential_genes(
    fdr_values: np.ndarray,
    mahalanobis_distances: np.ndarray,
    gene_names: List[str],
    fdr_threshold: float,
) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Create boolean DE annotation and summary statistics.

    Args:
        fdr_values: FDR-corrected p-values
        mahalanobis_distances: Mahalanobis distances for genes
        gene_names: Names of genes
        fdr_threshold: FDR threshold for significance

    Returns:
        (de_boolean_series, summary_stats): Boolean DE annotation and summary statistics
    """
    # Create boolean series
    is_significant = fdr_values < fdr_threshold
    de_boolean_series = pd.Series(is_significant, index=gene_names)

    # Calculate summary statistics
    n_significant = np.sum(is_significant)
    n_total = len(gene_names)

    # Find threshold Mahalanobis distance corresponding to FDR threshold
    if n_significant > 0:
        significant_distances = mahalanobis_distances[is_significant]
        min_significant_distance = np.min(significant_distances)
    else:
        min_significant_distance = np.inf

    summary_stats = {
        "n_significant": n_significant,
        "n_total": n_total,
        "fraction_significant": n_significant / n_total,
        "fdr_threshold": fdr_threshold,
        "min_significant_mahalanobis": min_significant_distance,
    }

    return de_boolean_series, summary_stats
