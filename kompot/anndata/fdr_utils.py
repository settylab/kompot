"""
FDR utilities for differential expression analysis.

Provides local and tail-based FDR estimation similar to R's fdrtool package,
specifically designed for Mahalanobis distance-based differential expression.
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional, Union, Dict, Any, List, Tuple

# np.trapz was renamed to np.trapezoid in NumPy 2.0
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

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

    Uses monotone-constrained density estimation (Grenander estimator) to compute
    local FDR directly from Mahalanobis distances. Both null and mixture densities are
    constrained to be monotonically declining with Mahalanobis distance, and the
    resulting local FDR is guaranteed to be monotonically declining with distance.

    Note: Larger Mahalanobis distances indicate MORE significance (deviation from null).
    Mahalanobis distances are always non-negative.

    Args:
        real_mahalanobis: Mahalanobis distances for real genes (non-negative)
        null_mahalanobis: Mahalanobis distances for null genes (background)
        fdr_threshold: FDR threshold for significance

    Returns:
        (pvalues, local_fdr_values, tail_fdr_values, is_significant): P-values, local FDR, tail-based FDR, and boolean significance
    """
    # Step 1: Compute empirical p-values (vectorized with searchsorted)
    sorted_null = np.sort(null_mahalanobis)
    n_null = len(sorted_null)
    pvalues = (
        n_null - np.searchsorted(sorted_null, real_mahalanobis, side="left")
    ).astype(float) / n_null

    # Floor zero p-values
    zero_mask = pvalues == 0.0
    if np.any(zero_mask):
        non_zero = pvalues[~zero_mask]
        min_pval = np.min(non_zero) if len(non_zero) > 0 else 1.0 / n_null
        pvalues[zero_mask] = min_pval
        logger.debug(f"Set minimum p-value to {min_pval} for {np.sum(zero_mask)} zero p-values")

    # Step 2: Compute local FDR and tail FDR from Grenander-estimated densities.
    # Both are derived from the same monotone density/survival function estimates
    # (fdrtool approach), avoiding the resolution limit of BH on discrete
    # empirical p-values (which fails when n_null << n_genes).
    local_fdr_values, tail_fdr_values = _compute_fdr_from_densities(
        real_mahalanobis, null_mahalanobis
    )

    # Step 3: Determine significance using local FDR
    is_significant = local_fdr_values < fdr_threshold

    return pvalues, local_fdr_values, tail_fdr_values, is_significant


def _benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """
    Benjamini-Hochberg FDR correction for multiple testing.

    Args:
        pvalues: Raw p-values (1-d array)

    Returns:
        BH-adjusted p-values (q-values), same order as input
    """
    pvalues = np.asarray(pvalues, dtype=float)
    n = len(pvalues)
    sort_idx = np.argsort(pvalues)
    pvals_sorted = pvalues[sort_idx]

    # BH adjusted p-value: p_i * n / rank_i, then enforce monotonicity
    ranks = np.arange(1, n + 1, dtype=float)
    adjusted = pvals_sorted * n / ranks
    # Cumulative minimum from right to enforce monotonicity
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    # Restore original order
    result = np.empty_like(adjusted)
    result[sort_idx] = adjusted
    return result


def _compute_local_fdr_monotone(
    real_distances: np.ndarray, null_distances: np.ndarray, n_grid: int = 500
) -> np.ndarray:
    """
    Compute local FDR using monotone-constrained density estimation.

    Both null and mixture densities are estimated using the Grenander estimator
    (boundary-corrected KDE + isotonic regression via PAVA), which enforces
    monotonically declining density with respect to Mahalanobis distance.
    The resulting local FDR ratio is also constrained to be monotonically declining.

    Args:
        real_distances: Mahalanobis distances for real genes
        null_distances: Mahalanobis distances for null genes
        n_grid: Number of grid points for density evaluation

    Returns:
        Local FDR values for each real gene
    """
    from scipy.interpolate import interp1d

    if len(null_distances) < 2 or len(real_distances) < 2:
        logger.warning("Too few distances for local FDR estimation, returning lfdr=1")
        return np.ones(len(real_distances))

    max_dist = max(np.max(real_distances), np.max(null_distances))
    if max_dist <= 0:
        return np.ones(len(real_distances))
    grid = np.linspace(0, max_dist * 1.05, n_grid)

    # Estimate null density f0 with monotone decreasing constraint
    f0 = _estimate_monotone_density(null_distances, grid)

    # Estimate mixture density f with monotone decreasing constraint
    f_mix = _estimate_monotone_density(real_distances, grid)

    # Local FDR = f0(d) / f_mix(d) with pi0 = 1 (conservative)
    eps = np.max(f_mix) * 1e-10 if np.max(f_mix) > 0 else 1e-20
    raw_lfdr = np.where(f_mix > eps, f0 / f_mix, 1.0)
    raw_lfdr = np.clip(raw_lfdr, 0.0, 1.0)

    # Enforce monotone decreasing local FDR with distance
    monotone_lfdr = _pava_decreasing(raw_lfdr)

    # Interpolate to evaluate at real distances
    lfdr_func = interp1d(
        grid, monotone_lfdr, kind="linear", bounds_error=False, fill_value=(1.0, 0.0)
    )

    return np.clip(lfdr_func(real_distances), 0.0, 1.0)


def _compute_fdr_from_densities(
    real_distances: np.ndarray, null_distances: np.ndarray, n_grid: int = 500
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute local FDR and tail FDR from Grenander-estimated densities.

    Uses the fdrtool approach (Strimmer, 2008):
      local_fdr(d) = f_null(d) / f_mix(d)        [density ratio]
      tail_Fdr(d)  = S_null(d) / S_mix(d)         [survival function ratio]

    where S(d) = 1 - CDF(d) is the survival function. Both are guaranteed
    monotonically decreasing with distance (by PAVA constraint on densities).
    pi0 = 1 (conservative assumption that all genes are null).

    Args:
        real_distances: Mahalanobis distances for real genes
        null_distances: Mahalanobis distances for null genes
        n_grid: Number of grid points for density evaluation

    Returns:
        (local_fdr, tail_fdr): Tuple of arrays, one value per real gene
    """
    from scipy.interpolate import interp1d

    if len(null_distances) < 2 or len(real_distances) < 2:
        logger.warning("Too few distances for FDR estimation, returning fdr=1")
        ones = np.ones(len(real_distances))
        return ones, ones

    max_dist = max(np.max(real_distances), np.max(null_distances))
    if max_dist <= 0:
        ones = np.ones(len(real_distances))
        return ones, ones
    grid = np.linspace(0, max_dist * 1.05, n_grid)
    dg = grid[1] - grid[0]

    # Estimate monotone-decreasing densities (Grenander estimator)
    f0 = _estimate_monotone_density(null_distances, grid)
    f_mix = _estimate_monotone_density(real_distances, grid)

    eps = np.max(f_mix) * 1e-10 if np.max(f_mix) > 0 else 1e-20

    # --- Local FDR: f0(d) / f_mix(d) with pi0 = 1 ---
    raw_lfdr = np.where(f_mix > eps, f0 / f_mix, 1.0)
    raw_lfdr = np.clip(raw_lfdr, 0.0, 1.0)
    monotone_lfdr = _pava_decreasing(raw_lfdr)

    # --- Tail FDR: S0(d) / S_mix(d) where S = survival function ---
    # Survival function = integral of density from d to infinity
    S0 = np.cumsum(f0[::-1])[::-1] * dg
    S_mix = np.cumsum(f_mix[::-1])[::-1] * dg

    raw_tail_fdr = np.where(S_mix > eps, S0 / S_mix, 1.0)
    raw_tail_fdr = np.clip(raw_tail_fdr, 0.0, 1.0)
    monotone_tail_fdr = _pava_decreasing(raw_tail_fdr)

    # Interpolate to evaluate at real distances
    lfdr_func = interp1d(grid, monotone_lfdr, kind="linear",
                         bounds_error=False, fill_value=(1.0, 0.0))
    tfdr_func = interp1d(grid, monotone_tail_fdr, kind="linear",
                         bounds_error=False, fill_value=(1.0, 0.0))

    local_fdr = np.clip(lfdr_func(real_distances), 0.0, 1.0)
    tail_fdr = np.clip(tfdr_func(real_distances), 0.0, 1.0)

    return local_fdr, tail_fdr


def _estimate_monotone_density(distances: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """
    Estimate a monotonically declining density using the Grenander estimator.

    Uses boundary-corrected KDE (reflection method at 0) for the initial density
    estimate, then applies the Pool Adjacent Violators Algorithm (PAVA) to enforce
    the monotone decreasing constraint. The result is normalized to a proper density.

    Args:
        distances: Non-negative distance values
        grid: Evaluation grid points (must be evenly spaced starting near 0)

    Returns:
        Monotonically declining density evaluated at grid points
    """
    from scipy.stats import gaussian_kde

    try:
        # Boundary-corrected KDE using reflection at 0.
        # This handles the non-negative support of Mahalanobis distances by
        # mirroring the data around 0, which prevents density leakage below 0.
        reflected = np.concatenate([-distances, distances])
        kde = gaussian_kde(reflected)
        density = 2.0 * kde(grid)  # Factor 2 to account for the reflected half
    except (np.linalg.LinAlgError, ValueError):
        # Fallback: histogram-based density if KDE fails (e.g., zero variance)
        logger.debug("KDE failed, falling back to histogram density")
        bin_edges = np.linspace(grid[0], grid[-1], len(grid) + 1)
        counts, _ = np.histogram(distances, bins=bin_edges)
        bin_width = bin_edges[1] - bin_edges[0]
        density = counts.astype(float) / (len(distances) * bin_width)

    # Apply PAVA for monotone decreasing constraint (Grenander estimator)
    density = _pava_decreasing(density)

    # Ensure non-negative and normalize to proper density
    density = np.maximum(density, 0.0)
    integral = _trapz(density, grid[: len(density)])
    if integral > 0:
        density /= integral

    return density


def _pava_decreasing(y: np.ndarray, w: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Pool Adjacent Violators Algorithm for monotone non-increasing constraint.

    Returns the monotone non-increasing sequence closest to y in weighted L2 norm.
    Uses the numerically stable incremental update from fdrtool's isomean.c
    to avoid catastrophic cancellation when merging blocks with similar values.

    Args:
        y: Input sequence
        w: Optional weights (default: uniform weights of 1.0)

    Returns:
        Monotone non-increasing sequence
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n <= 1:
        return y.copy()

    if w is None:
        w = np.ones(n, dtype=float)
    else:
        w = np.asarray(w, dtype=float)

    # Antitonic regression = negate, isotonic (PAVA), negate
    # Following fdrtool's C_isomean implementation for numerical stability
    neg_y = -y.copy()
    ghat = np.empty(n, dtype=float)
    gew = np.empty(n, dtype=float)
    k = np.empty(n, dtype=int)  # block start indices

    c = 0
    k[c] = 0
    gew[c] = w[0]
    ghat[c] = neg_y[0]

    for j in range(1, n):
        c += 1
        k[c] = j
        gew[c] = w[j]
        ghat[c] = neg_y[j]

        while c > 0 and ghat[c - 1] >= ghat[c]:
            # Merge blocks using fdrtool's stable incremental update:
            # ghat[c-1] += (gew[c]/total) * (ghat[c] - ghat[c-1])
            # This avoids (w1*g1 + w2*g2)/total which can lose precision.
            neu = gew[c] + gew[c - 1]
            ghat[c - 1] = ghat[c - 1] + (gew[c] / neu) * (ghat[c] - ghat[c - 1])
            gew[c - 1] = neu
            c -= 1

    # Write back: fill each block with its pooled value
    output = np.empty(n, dtype=float)
    nn = n
    while nn >= 1:
        start = k[c]
        output[start:nn] = -ghat[c]  # negate back for antitonic
        nn = start
        c -= 1

    return output


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
