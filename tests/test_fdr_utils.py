"""Tests for FDR utilities module."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from kompot.anndata.fdr_utils import (
    prepare_null_genes,
    generate_shuffled_expression,
    compute_fdr_statistics,
    annotate_differential_genes
)


class TestPrepareNullGenes:
    """Tests for prepare_null_genes function."""
    
    def test_integer_specification_sufficient_genes(self):
        """Test integer specification with sufficient genes."""
        available_genes = [f"gene_{i}" for i in range(1000)]
        
        null_indices, used_replacement = prepare_null_genes(
            null_genes=100,
            available_genes=available_genes,
            null_seed=42
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
            null_seed=42
        )
        
        assert len(null_indices) == 150
        assert used_replacement
    
    def test_list_specification(self):
        """Test list specification."""
        available_genes = [f"gene_{i}" for i in range(1000)]
        specific_indices = [10, 50, 100, 200, 300]
        
        null_indices, used_replacement = prepare_null_genes(
            null_genes=specific_indices,
            available_genes=available_genes,
            null_seed=42
        )
        
        assert null_indices == specific_indices
        assert not used_replacement
    
    def test_none_specification(self):
        """Test None specification (disabled)."""
        available_genes = [f"gene_{i}" for i in range(1000)]
        
        null_indices, used_replacement = prepare_null_genes(
            null_genes=None,
            available_genes=available_genes,
            null_seed=42
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
            expr1=expr1,
            expr2=expr2,
            null_gene_indices=null_gene_indices,
            null_seed=42
        )
        
        # Test shapes
        assert shuffled_expr1.shape == (n_cells_1, len(null_gene_indices))
        assert shuffled_expr2.shape == (n_cells_2, len(null_gene_indices))
        
        # Test that values are permutations of originals
        original_combined = np.vstack([expr1[:, null_gene_indices], expr2[:, null_gene_indices]])
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
        
        shuffled1_a, shuffled2_a = generate_shuffled_expression(expr1, expr2, null_gene_indices, 42)
        shuffled1_b, shuffled2_b = generate_shuffled_expression(expr1, expr2, null_gene_indices, 42)
        
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
            fdr_threshold=0.05
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
        
        _, _, _, is_significant = compute_fdr_statistics(real_mahalanobis, null_mahalanobis, 0.05)
        
        # Should have very few false positives
        false_positive_rate = np.mean(is_significant)
        assert false_positive_rate < 0.1, f"Too many false positives: {false_positive_rate:.3f}"


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
        fdr_values[n_significant:] = np.random.uniform(0.05, 1.0, n_genes - n_significant)
        
        de_annotation, summary_stats = annotate_differential_genes(
            fdr_values=fdr_values,
            mahalanobis_distances=mahalanobis_distances,
            gene_names=gene_names,
            fdr_threshold=0.05
        )
        
        # Test de_annotation
        assert isinstance(de_annotation, pd.Series)
        assert len(de_annotation) == n_genes
        assert list(de_annotation.index) == gene_names
        assert de_annotation.dtype == bool
        
        # Test summary_stats
        required_keys = ['n_significant', 'n_total', 'fraction_significant', 
                        'fdr_threshold', 'min_significant_mahalanobis']
        for key in required_keys:
            assert key in summary_stats
        
        assert summary_stats['n_significant'] == n_significant
        assert summary_stats['n_total'] == n_genes
        assert summary_stats['fdr_threshold'] == 0.05
    
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
        assert summary_stats['n_significant'] == 0
        assert summary_stats['min_significant_mahalanobis'] == np.inf
    
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
        assert summary_stats['n_significant'] == n_genes
        assert summary_stats['fraction_significant'] == 1.0


if __name__ == "__main__":
    pytest.main([__file__])