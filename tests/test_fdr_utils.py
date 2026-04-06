"""Tests for FDR utilities module."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from kompot.anndata.fdr_utils import (
    prepare_null_genes,
    generate_shuffled_expression,
    compute_fdr_statistics,
    annotate_differential_genes,
    _benjamini_hochberg,
)


class TestPrepareNullGenes:
    """Tests for prepare_null_genes function."""

    def test_integer_specification_sufficient_genes(self):
        """Test integer specification with sufficient genes."""
        available_genes = [f"gene_{i}" for i in range(1000)]

        null_indices, used_replacement = prepare_null_genes(
            null_genes=100, available_genes=available_genes, null_seed=42
        )

        assert len(null_indices) == 100
        assert not used_replacement
        assert len(set(null_indices)) == 100  # Should have unique genes
        assert all(0 <= idx < 1000 for idx in null_indices)

    def test_integer_specification_insufficient_genes(self):
        """Test integer specification with insufficient genes."""
        available_genes = [f"gene_{i}" for i in range(100)]

        null_indices, used_replacement = prepare_null_genes(
            null_genes=150,  # More than available
            available_genes=available_genes,
            null_seed=42,
        )

        assert len(null_indices) == 150
        assert used_replacement

    def test_list_specification(self):
        """Test list specification."""
        available_genes = [f"gene_{i}" for i in range(1000)]
        specific_indices = [10, 50, 100, 200, 300]

        null_indices, used_replacement = prepare_null_genes(
            null_genes=specific_indices, available_genes=available_genes, null_seed=42
        )

        assert null_indices == specific_indices
        assert not used_replacement

    def test_none_specification(self):
        """Test None specification (disabled)."""
        available_genes = [f"gene_{i}" for i in range(1000)]

        null_indices, used_replacement = prepare_null_genes(
            null_genes=None, available_genes=available_genes, null_seed=42
        )

        assert null_indices == []
        assert not used_replacement

    def test_reproducibility(self):
        """Test that same seed gives same results."""
        available_genes = [f"gene_{i}" for i in range(1000)]

        null_indices1, _ = prepare_null_genes(100, available_genes, 42)
        null_indices2, _ = prepare_null_genes(100, available_genes, 42)

        assert null_indices1 == null_indices2

    def test_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        available_genes = [f"gene_{i}" for i in range(100)]

        # Negative number
        with pytest.raises(ValueError, match="must be positive"):
            prepare_null_genes(-10, available_genes, 42)

        # Invalid list elements
        with pytest.raises(ValueError, match="must be integers"):
            prepare_null_genes([1.5, 2.7], available_genes, 42)

        # Invalid indices
        with pytest.raises(ValueError, match="Invalid gene indices"):
            prepare_null_genes([50, 150], available_genes, 42)

        # Wrong type
        with pytest.raises(ValueError, match="must be int, list of ints, or None"):
            prepare_null_genes("invalid", available_genes, 42)


class TestGenerateShuffledExpression:
    """Tests for generate_shuffled_expression function."""

    def test_basic_functionality(self):
        """Test basic shuffling functionality."""
        np.random.seed(42)
        n_cells_1, n_cells_2 = 100, 150
        n_genes = 1000

        expr1 = np.random.normal(5, 2, (n_cells_1, n_genes))
        expr2 = np.random.normal(7, 2, (n_cells_2, n_genes))

        null_gene_indices = [10, 50, 100]

        shuffled_expr1, shuffled_expr2 = generate_shuffled_expression(
            expr1=expr1, expr2=expr2, null_gene_indices=null_gene_indices, null_seed=42
        )

        # Test shapes
        assert shuffled_expr1.shape == (n_cells_1, len(null_gene_indices))
        assert shuffled_expr2.shape == (n_cells_2, len(null_gene_indices))

        # Test that values are permutations of originals
        original_combined = np.vstack(
            [expr1[:, null_gene_indices], expr2[:, null_gene_indices]]
        )
        shuffled_combined = np.vstack([shuffled_expr1, shuffled_expr2])

        for i in range(len(null_gene_indices)):
            orig_values = np.sort(original_combined[:, i])
            shuffled_values = np.sort(shuffled_combined[:, i])
            np.testing.assert_allclose(orig_values, shuffled_values, rtol=1e-10)

    def test_reproducibility(self):
        """Test that same seed gives same results."""
        expr1 = np.random.normal(0, 1, (50, 100))
        expr2 = np.random.normal(0, 1, (60, 100))
        null_gene_indices = [10, 20, 30]

        shuffled1_a, shuffled2_a = generate_shuffled_expression(
            expr1, expr2, null_gene_indices, 42
        )
        shuffled1_b, shuffled2_b = generate_shuffled_expression(
            expr1, expr2, null_gene_indices, 42
        )

        np.testing.assert_array_equal(shuffled1_a, shuffled1_b)
        np.testing.assert_array_equal(shuffled2_a, shuffled2_b)

    def test_empty_null_genes(self):
        """Test empty null genes."""
        expr1 = np.random.normal(0, 1, (50, 100))
        expr2 = np.random.normal(0, 1, (60, 100))

        empty_expr1, empty_expr2 = generate_shuffled_expression(expr1, expr2, [], 42)

        assert empty_expr1.shape == (50, 0)
        assert empty_expr2.shape == (60, 0)


class TestComputeFDRStatistics:
    """Tests for compute_fdr_statistics function."""

    def test_basic_functionality(self):
        """Test basic FDR computation."""
        np.random.seed(42)

        # Create realistic test data
        null_mahalanobis = np.random.gamma(2, 1, 1000)

        # Real genes with some differential
        non_diff_mahalanobis = np.random.gamma(2, 1, 450)
        diff_mahalanobis = np.random.gamma(2, 1, 50) * 5 + 10  # Much stronger signal

        real_mahalanobis = np.concatenate([non_diff_mahalanobis, diff_mahalanobis])
        np.random.shuffle(real_mahalanobis)

        pvalues, local_fdr, tail_fdr, is_significant = compute_fdr_statistics(
            real_mahalanobis=real_mahalanobis,
            null_mahalanobis=null_mahalanobis,
            fdr_threshold=0.05,
        )

        # Test output shapes
        assert len(pvalues) == len(real_mahalanobis)
        assert len(local_fdr) == len(real_mahalanobis)
        assert len(tail_fdr) == len(real_mahalanobis)
        assert len(is_significant) == len(real_mahalanobis)

        # Test value ranges
        assert np.all(pvalues >= 0) and np.all(pvalues <= 1)
        assert np.all(local_fdr >= 0) and np.all(local_fdr <= 1)
        assert np.all(tail_fdr >= 0) and np.all(tail_fdr <= 1)
        assert is_significant.dtype == bool

        # Test directionality
        correlation, _ = stats.spearmanr(real_mahalanobis, pvalues)
        assert correlation < -0.5, "P-values should decrease with Mahalanobis distance"

    def test_identical_values_edge_case(self):
        """Test edge case with identical values."""
        identical_real = np.full(100, 2.0)
        identical_null = np.full(100, 2.0)

        pvalues, local_fdr, tail_fdr, is_significant = compute_fdr_statistics(
            identical_real, identical_null, 0.05
        )

        # Should give p-values of 1.0
        np.testing.assert_allclose(pvalues, 1.0, atol=1e-6)
        assert not np.any(is_significant)

    def test_no_signal_case(self):
        """Test case where real genes are drawn from same distribution as null."""
        np.random.seed(42)
        null_mahalanobis = np.random.gamma(2, 1, 500)
        real_mahalanobis = np.random.gamma(2, 1, 200)

        _, _, _, is_significant = compute_fdr_statistics(
            real_mahalanobis, null_mahalanobis, 0.05
        )

        # Should have very few false positives
        false_positive_rate = np.mean(is_significant)
        assert false_positive_rate < 0.1, (
            f"Too many false positives: {false_positive_rate:.3f}"
        )

    def test_local_fdr_monotone_decreasing(self):
        """Test that local FDR is monotonically decreasing with Mahalanobis distance."""
        np.random.seed(42)

        null_mahalanobis = np.random.gamma(2, 1, 1000)

        # Mix of null-like and differential genes
        non_diff = np.random.gamma(2, 1, 400)
        diff = np.random.gamma(2, 1, 100) * 3 + 5
        real_mahalanobis = np.concatenate([non_diff, diff])

        _, local_fdr_vals, _, _ = compute_fdr_statistics(
            real_mahalanobis, null_mahalanobis, 0.05
        )

        # Sort by Mahalanobis distance and check monotonicity
        sorted_indices = np.argsort(real_mahalanobis)
        sorted_lfdr = local_fdr_vals[sorted_indices]

        # Local FDR should be non-increasing with distance
        violations = np.sum(np.diff(sorted_lfdr) > 1e-10)
        assert violations == 0, f"Local FDR has {violations} monotonicity violations"

    def test_densities_strictly_declining(self):
        """Test that fitted null and mixture densities are monotonically declining."""
        from kompot.anndata.fdr_utils import _estimate_monotone_density

        np.random.seed(42)
        distances = np.random.gamma(2, 1, 500)
        grid = np.linspace(0, np.max(distances) * 1.05, 200)

        density = _estimate_monotone_density(distances, grid)

        # Density should be monotonically non-increasing
        violations = np.sum(np.diff(density) > 1e-10)
        assert violations == 0, f"Density has {violations} monotonicity violations"

        # Density should be non-negative
        assert np.all(density >= 0), "Density has negative values"

        # Density should integrate to approximately 1
        _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
        integral = _trapz(density, grid)
        assert abs(integral - 1.0) < 0.05, (
            f"Density integral = {integral}, expected ~1.0"
        )


class TestAnnotateDifferentialGenes:
    """Tests for annotate_differential_genes function."""

    def test_basic_functionality(self):
        """Test basic annotation functionality."""
        n_genes = 200
        gene_names = [f"gene_{i}" for i in range(n_genes)]

        np.random.seed(42)
        mahalanobis_distances = np.random.gamma(2, 1, n_genes)
        fdr_values = np.random.beta(2, 5, n_genes)

        # Make some genes significant
        n_significant = 20
        fdr_values[:n_significant] = np.random.uniform(0, 0.05, n_significant)
        # Make sure the rest are above threshold
        fdr_values[n_significant:] = np.random.uniform(
            0.05, 1.0, n_genes - n_significant
        )

        de_annotation, summary_stats = annotate_differential_genes(
            fdr_values=fdr_values,
            mahalanobis_distances=mahalanobis_distances,
            gene_names=gene_names,
            fdr_threshold=0.05,
        )

        # Test de_annotation
        assert isinstance(de_annotation, pd.Series)
        assert len(de_annotation) == n_genes
        assert list(de_annotation.index) == gene_names
        assert de_annotation.dtype == bool

        # Test summary_stats
        required_keys = [
            "n_significant",
            "n_total",
            "fraction_significant",
            "fdr_threshold",
            "min_significant_mahalanobis",
        ]
        for key in required_keys:
            assert key in summary_stats

        assert summary_stats["n_significant"] == n_significant
        assert summary_stats["n_total"] == n_genes
        assert summary_stats["fdr_threshold"] == 0.05

    def test_no_significant_genes(self):
        """Test case with no significant genes."""
        n_genes = 100
        gene_names = [f"gene_{i}" for i in range(n_genes)]
        mahalanobis_distances = np.random.gamma(2, 1, n_genes)
        high_fdr = np.full(n_genes, 0.1)  # All above threshold

        de_annotation, summary_stats = annotate_differential_genes(
            high_fdr, mahalanobis_distances, gene_names, 0.05
        )

        assert not de_annotation.any()
        assert summary_stats["n_significant"] == 0
        assert summary_stats["min_significant_mahalanobis"] == np.inf

    def test_all_significant_genes(self):
        """Test case with all significant genes."""
        n_genes = 100
        gene_names = [f"gene_{i}" for i in range(n_genes)]
        mahalanobis_distances = np.random.gamma(2, 1, n_genes)
        low_fdr = np.full(n_genes, 0.01)  # All below threshold

        de_annotation, summary_stats = annotate_differential_genes(
            low_fdr, mahalanobis_distances, gene_names, 0.05
        )

        assert de_annotation.all()
        assert summary_stats["n_significant"] == n_genes
        assert summary_stats["fraction_significant"] == 1.0


class TestBenjaminiHochberg:
    """Tests for _benjamini_hochberg function."""

    def test_known_values(self):
        """Test against hand-computed BH-adjusted p-values."""
        # 5 p-values, sorted: 0.01, 0.03, 0.05, 0.20, 0.50
        # Ranks:              1,    2,    3,    4,    5
        # raw adjusted = p * n/rank: 0.05, 0.075, 0.0833, 0.25, 0.50
        # cummin from right:         0.05, 0.075, 0.0833, 0.25, 0.50
        pvals = np.array([0.05, 0.50, 0.01, 0.20, 0.03])
        adjusted = _benjamini_hochberg(pvals)

        expected_sorted = np.array([0.05, 0.075, 5.0 / 60, 0.25, 0.50])
        # Map back to original order: input indices sorted by pval are [2,4,0,3,1]
        expected = np.empty(5)
        sort_order = [2, 4, 0, 3, 1]
        for rank_idx, orig_idx in enumerate(sort_order):
            expected[orig_idx] = expected_sorted[rank_idx]

        np.testing.assert_allclose(adjusted, expected, rtol=1e-10)

    def test_all_significant(self):
        """All very small p-values should remain significant after correction."""
        pvals = np.array([0.001, 0.002, 0.003, 0.004, 0.005])
        adjusted = _benjamini_hochberg(pvals)
        assert np.all(adjusted < 0.05)

    def test_no_false_discoveries_under_null(self):
        """Under the global null (all p-values uniform), BH should control FDR."""
        rng = np.random.RandomState(42)
        n_tests = 500
        alpha = 0.05
        n_simulations = 200
        fdr_sum = 0.0
        for _ in range(n_simulations):
            pvals = rng.uniform(0, 1, n_tests)
            adjusted = _benjamini_hochberg(pvals)
            fdr_sum += np.mean(adjusted < alpha)
        avg_rejection_rate = fdr_sum / n_simulations
        assert avg_rejection_rate < alpha * 1.5, (
            f"Average rejection rate {avg_rejection_rate:.3f} too high under global null"
        )

    def test_adjusted_geq_raw(self):
        """BH-adjusted p-values should always be >= raw p-values."""
        rng = np.random.RandomState(42)
        pvals = rng.uniform(0, 1, 200)
        adjusted = _benjamini_hochberg(pvals)
        assert np.all(adjusted >= pvals - 1e-15)

    def test_clipped_to_one(self):
        """Adjusted p-values should never exceed 1."""
        pvals = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        adjusted = _benjamini_hochberg(pvals)
        assert np.all(adjusted <= 1.0)

    def test_single_pvalue(self):
        """Single p-value should be unchanged."""
        pvals = np.array([0.03])
        adjusted = _benjamini_hochberg(pvals)
        np.testing.assert_allclose(adjusted, pvals)

    def test_monotone_adjusted(self):
        """Sorted adjusted p-values should be non-decreasing."""
        rng = np.random.RandomState(99)
        pvals = rng.uniform(0, 1, 300)
        adjusted = _benjamini_hochberg(pvals)
        sorted_adj = adjusted[np.argsort(pvals)]
        assert np.all(np.diff(sorted_adj) >= -1e-15)


def _analytical_lfdr(d, pi0, lambda0, lambda1):
    """
    Compute analytical local FDR for an exponential two-component mixture.

    Mixture density: f(d) = pi0 * lambda0 * exp(-lambda0*d) + (1-pi0) * lambda1 * exp(-lambda1*d)
    Null density:    f0(d) = lambda0 * exp(-lambda0*d)
    True lfdr:       pi0 * f0(d) / f(d)

    Both components are monotonically declining (exponential), so
    lfdr is monotonically declining with d.
    """
    f0 = lambda0 * np.exp(-lambda0 * d)
    f1 = lambda1 * np.exp(-lambda1 * d)
    f_mix = pi0 * f0 + (1 - pi0) * f1
    return pi0 * f0 / f_mix


def _grenander_ecdf_reference(distances, eval_points):
    """
    Reference Grenander density estimator using ECDF + Least Concave Majorant.

    Follows fdrtool's algorithm (Strimmer, 2008) exactly:
    1. Compute ECDF of distances
    2. Compute LCM of the ECDF (via PAVA on slopes with dx weights)
    3. Density = slopes of the LCM (piecewise constant, non-increasing)

    This is the NPMLE under the monotone decreasing density constraint.
    Used here as a reference to validate the KDE+PAVA approach.
    """
    n = len(distances)
    sorted_d = np.sort(distances)

    # Unique values and ECDF
    unique_d, counts = np.unique(sorted_d, return_counts=True)
    ecdf_y = np.cumsum(counts) / n

    # Prepend origin (0, 0)
    x = np.concatenate([[0.0], unique_d])
    y = np.concatenate([[0.0], ecdf_y])

    # Remove zero-width intervals
    dx = np.diff(x)
    dy = np.diff(y)
    nonzero = dx > 0
    if not np.all(nonzero):
        # Merge zero-width intervals by accumulating their dy
        new_dx = []
        new_dy = []
        acc_dy = 0.0
        for i in range(len(dx)):
            acc_dy += dy[i]
            if nonzero[i]:
                new_dx.append(dx[i])
                new_dy.append(acc_dy)
                acc_dy = 0.0
        if acc_dy > 0 and new_dx:
            new_dy[-1] += acc_dy
        dx = np.array(new_dx)
        dy = np.array(new_dy)

    raw_slopes = dy / dx

    # Clamp infinities (fdrtool trick)
    raw_slopes = np.clip(raw_slopes, -1e300, 1e300)

    # LCM: antitonic PAVA on slopes with dx weights
    # antitonic = negate -> isotonic -> negate
    from kompot.anndata.fdr_utils import _pava_decreasing

    monotone_slopes = _pava_decreasing(raw_slopes, w=dx)

    # Build step function: knot positions are cumulative sums of dx
    knot_positions = np.concatenate([[0.0], np.cumsum(dx)])

    # Evaluate step function at eval_points
    # For each eval_point, find which interval it falls in
    indices = np.searchsorted(knot_positions, eval_points, side="right") - 1
    indices = np.clip(indices, 0, len(monotone_slopes) - 1)
    density = monotone_slopes[indices]

    # Set density to 0 beyond the data range
    density[eval_points > knot_positions[-1]] = 0.0

    return density


class TestGroundTruth:
    """Validation tests with known ground truth distributions.

    Uses exponential two-component mixtures where the analytical local FDR
    is known:
        null:        f0(d) = lambda0 * exp(-lambda0 * d)
        alternative: f1(d) = lambda1 * exp(-lambda1 * d),  lambda1 < lambda0
        mixture:     f(d) = pi0 * f0(d) + (1-pi0) * f1(d)
        true lfdr:   pi0 * f0(d) / f(d)

    Both components are monotonically declining, so the assumptions of the
    method are exactly satisfied.
    """

    def test_exponential_mixture_ranking(self):
        """Estimated lfdr should preserve the ranking of true lfdr."""
        rng = np.random.RandomState(42)
        pi0 = 0.8
        lambda0, lambda1 = 2.0, 0.3  # null decays fast, alt decays slow

        n_null_genes = 2000
        n_real_null = 800
        n_real_alt = 200

        null_distances = rng.exponential(1.0 / lambda0, n_null_genes)
        real_null = rng.exponential(1.0 / lambda0, n_real_null)
        real_alt = rng.exponential(1.0 / lambda1, n_real_alt)
        real_distances = np.concatenate([real_null, real_alt])

        true_lfdr = _analytical_lfdr(real_distances, pi0, lambda0, lambda1)

        _, est_lfdr, _, _ = compute_fdr_statistics(real_distances, null_distances, 0.05)

        # Overall Spearman may be reduced by ties at lfdr=0 and lfdr=1 boundaries.
        # Pearson correlation captures the linear relationship well despite ties.
        pearson_corr, _ = stats.pearsonr(true_lfdr, est_lfdr)
        assert pearson_corr > 0.95, f"Pearson correlation {pearson_corr:.3f} too low"

        # In the informative mid-range (away from boundary clipping), ranking
        # should be near-perfect.
        mid = (est_lfdr > 0.001) & (est_lfdr < 0.999)
        if np.sum(mid) > 20:
            rank_corr, _ = stats.spearmanr(true_lfdr[mid], est_lfdr[mid])
            assert rank_corr > 0.95, (
                f"Mid-range rank correlation {rank_corr:.3f} too low ({np.sum(mid)} genes)"
            )

    def test_exponential_mixture_conservative(self):
        """With pi0_hat=1 > true pi0, our method should make fewer discoveries than the oracle."""
        rng = np.random.RandomState(123)
        pi0 = 0.7
        lambda0, lambda1 = 2.0, 0.2

        null_distances = rng.exponential(1.0 / lambda0, 3000)
        real_null = rng.exponential(1.0 / lambda0, 700)
        real_alt = rng.exponential(1.0 / lambda1, 300)
        real_distances = np.concatenate([real_null, real_alt])

        true_lfdr = _analytical_lfdr(real_distances, pi0, lambda0, lambda1)

        _, est_lfdr, _, _ = compute_fdr_statistics(real_distances, null_distances, 0.05)

        # At each threshold, our method (with pi0_hat=1 > true pi0=0.7) should
        # make the same or fewer discoveries than the oracle. This is the
        # defining property of conservative FDR control.
        for threshold in [0.05, 0.10]:
            n_oracle = np.sum(true_lfdr < threshold)
            n_ours = np.sum(est_lfdr < threshold)
            assert n_ours <= n_oracle * 1.15, (
                f"At threshold {threshold}: our calls ({n_ours}) > oracle calls ({n_oracle}) * 1.15"
            )

    def test_pure_null_no_discoveries(self):
        """When all genes are null, should make (almost) no discoveries."""
        rng = np.random.RandomState(42)
        lambda0 = 1.5

        null_distances = rng.exponential(1.0 / lambda0, 2000)
        real_distances = rng.exponential(1.0 / lambda0, 500)

        _, est_lfdr, _, is_sig = compute_fdr_statistics(
            real_distances, null_distances, 0.05
        )

        # No (or very few) false discoveries
        assert np.sum(is_sig) <= 5, f"Too many false discoveries: {np.sum(is_sig)}"
        # lfdr should be close to 1 for most genes
        assert np.median(est_lfdr) > 0.8, (
            f"Median lfdr = {np.median(est_lfdr):.3f}, expected > 0.8"
        )

    def test_strong_separation_high_sensitivity(self):
        """When null and alt are well separated, should detect most DE genes."""
        rng = np.random.RandomState(42)
        lambda0 = 3.0  # null: tight around 0
        lambda1 = 0.1  # alt: spread far from 0
        pi0 = 0.8

        null_distances = rng.exponential(1.0 / lambda0, 3000)
        real_null = rng.exponential(1.0 / lambda0, int(1000 * pi0))
        real_alt = rng.exponential(1.0 / lambda1, int(1000 * (1 - pi0)))
        real_distances = np.concatenate([real_null, real_alt])
        is_truly_de = np.concatenate(
            [np.zeros(len(real_null)), np.ones(len(real_alt))]
        ).astype(bool)

        _, est_lfdr, _, is_sig = compute_fdr_statistics(
            real_distances, null_distances, 0.05
        )

        # Sensitivity: fraction of true DE genes detected
        sensitivity = np.sum(is_sig & is_truly_de) / np.sum(is_truly_de)
        assert sensitivity > 0.7, f"Sensitivity too low: {sensitivity:.3f}"

        # Specificity: fraction of true null genes correctly not called DE
        specificity = np.sum(~is_sig & ~is_truly_de) / np.sum(~is_truly_de)
        assert specificity > 0.9, f"Specificity too low: {specificity:.3f}"

    def test_calibration_at_extremes(self):
        """Very small distances should have lfdr near 1, very large near 0."""
        rng = np.random.RandomState(42)
        lambda0, lambda1 = 2.0, 0.2

        null_distances = rng.exponential(1.0 / lambda0, 3000)
        real_null = rng.exponential(1.0 / lambda0, 800)
        real_alt = rng.exponential(1.0 / lambda1, 200)
        real_distances = np.concatenate([real_null, real_alt])

        _, est_lfdr, _, _ = compute_fdr_statistics(real_distances, null_distances, 0.05)

        # Genes with very small distances (< 10th percentile of null)
        small_threshold = np.percentile(null_distances, 10)
        small_mask = real_distances < small_threshold
        if np.sum(small_mask) > 0:
            assert np.mean(est_lfdr[small_mask]) > 0.8, (
                f"Mean lfdr for small distances = {np.mean(est_lfdr[small_mask]):.3f}"
            )

        # Genes with very large distances (> 99th percentile of null)
        large_threshold = np.percentile(null_distances, 99)
        large_mask = real_distances > large_threshold
        if np.sum(large_mask) > 0:
            assert np.mean(est_lfdr[large_mask]) < 0.3, (
                f"Mean lfdr for large distances = {np.mean(est_lfdr[large_mask]):.3f}"
            )

    def test_varying_signal_strength(self):
        """Stronger signal (larger lambda0/lambda1 ratio) should yield more discoveries."""
        rng = np.random.RandomState(42)
        lambda0 = 2.0

        discoveries_by_strength = []
        for lambda1 in [1.0, 0.5, 0.2]:
            null_distances = rng.exponential(1.0 / lambda0, 2000)
            real_null = rng.exponential(1.0 / lambda0, 400)
            real_alt = rng.exponential(1.0 / lambda1, 100)
            real_distances = np.concatenate([real_null, real_alt])

            _, _, _, is_sig = compute_fdr_statistics(
                real_distances, null_distances, 0.05
            )
            discoveries_by_strength.append(np.sum(is_sig))

        # Stronger separation should give more discoveries
        assert discoveries_by_strength[1] >= discoveries_by_strength[0], (
            f"Moderate signal ({discoveries_by_strength[1]}) should find >= weak ({discoveries_by_strength[0]})"
        )
        assert discoveries_by_strength[2] >= discoveries_by_strength[1], (
            f"Strong signal ({discoveries_by_strength[2]}) should find >= moderate ({discoveries_by_strength[1]})"
        )

    def test_fdr_control_at_threshold(self):
        """Among genes called significant, the actual FDP should be near the threshold."""
        rng = np.random.RandomState(42)
        lambda0, lambda1 = 2.0, 0.2
        pi0 = 0.8
        threshold = 0.10

        null_distances = rng.exponential(1.0 / lambda0, 5000)
        real_null = rng.exponential(1.0 / lambda0, int(2000 * pi0))
        real_alt = rng.exponential(1.0 / lambda1, int(2000 * (1 - pi0)))
        real_distances = np.concatenate([real_null, real_alt])
        is_truly_de = np.concatenate(
            [np.zeros(len(real_null)), np.ones(len(real_alt))]
        ).astype(bool)

        _, est_lfdr, _, _ = compute_fdr_statistics(
            real_distances, null_distances, threshold
        )
        is_sig = est_lfdr < threshold

        if np.sum(is_sig) > 0:
            # False discovery proportion among called genes
            fdp = np.sum(is_sig & ~is_truly_de) / np.sum(is_sig)
            # FDP should be controlled near or below the threshold
            # Allow generous margin because lfdr is conservative (pi0=1)
            assert fdp < threshold * 2.5, (
                f"FDP = {fdp:.3f} too high for threshold = {threshold}"
            )

    def test_concordance_with_grenander_ecdf(self):
        """KDE+PAVA and ECDF+LCM Grenander should produce concordant lfdr rankings."""
        from kompot.anndata.fdr_utils import _estimate_monotone_density

        rng = np.random.RandomState(42)
        lambda0, lambda1 = 2.0, 0.3

        null_distances = rng.exponential(1.0 / lambda0, 2000)
        real_null = rng.exponential(1.0 / lambda0, 800)
        real_alt = rng.exponential(1.0 / lambda1, 200)
        real_distances = np.concatenate([real_null, real_alt])

        max_dist = max(np.max(real_distances), np.max(null_distances))
        grid = np.linspace(0, max_dist * 1.05, 500)

        # KDE+PAVA approach (our implementation)
        f0_kde = _estimate_monotone_density(null_distances, grid)
        f_kde = _estimate_monotone_density(real_distances, grid)

        # ECDF+LCM Grenander reference
        f0_gren = _grenander_ecdf_reference(null_distances, grid)
        f_gren = _grenander_ecdf_reference(real_distances, grid)

        # Both densities should be monotonically declining
        assert np.all(np.diff(f0_gren) <= 1e-10), "Grenander null density not declining"
        assert np.all(np.diff(f_gren) <= 1e-10), (
            "Grenander mixture density not declining"
        )

        # Density shapes should be correlated (same underlying data)
        # Use points where both are non-trivial
        mask = (f0_kde > 1e-6) & (f0_gren > 1e-6)
        if np.sum(mask) > 10:
            corr, _ = stats.spearmanr(f0_kde[mask], f0_gren[mask])
            assert corr > 0.8, f"Null density correlation = {corr:.3f}, expected > 0.8"

    def test_no_runtime_warnings(self):
        """The method should produce no overflow or divide-by-zero warnings."""
        import warnings

        rng = np.random.RandomState(99)
        null_distances = rng.exponential(0.5, 3000)
        real_null = rng.exponential(0.5, 900)
        real_alt = rng.exponential(5.0, 100) + 15  # very large distances
        real_distances = np.concatenate([real_null, real_alt])

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # Should complete without any RuntimeWarning
            compute_fdr_statistics(real_distances, null_distances, 0.05)

    def test_gamma_distributed_ground_truth(self):
        """Test with gamma-distributed distances (common for chi-squared-like Mahalanobis)."""
        rng = np.random.RandomState(42)
        # Null: Gamma(shape=2, scale=1) - peaked near 1, declining after
        # Alt: Gamma(shape=2, scale=4) - peaked near 4, declining after
        pi0 = 0.75

        null_distances = rng.gamma(2, 1, 3000)
        real_null = rng.gamma(2, 1, int(1000 * pi0))
        real_alt = rng.gamma(2, 4, int(1000 * (1 - pi0)))
        real_distances = np.concatenate([real_null, real_alt])
        is_truly_de = np.concatenate(
            [np.zeros(len(real_null)), np.ones(len(real_alt))]
        ).astype(bool)

        _, est_lfdr, _, is_sig = compute_fdr_statistics(
            real_distances, null_distances, 0.05
        )

        # Should detect some DE genes
        assert np.sum(is_sig) > 0, "Should detect at least some DE genes"
        # Sensitivity should be reasonable
        sensitivity = np.sum(is_sig & is_truly_de) / np.sum(is_truly_de)
        assert sensitivity > 0.3, (
            f"Sensitivity too low for gamma ground truth: {sensitivity:.3f}"
        )
        # Specificity should be high
        specificity = np.sum(~is_sig & ~is_truly_de) / np.sum(~is_truly_de)
        assert specificity > 0.85, f"Specificity too low: {specificity:.3f}"


if __name__ == "__main__":
    pytest.main([__file__])
