"""
Test cases for single condition variance functionality.
"""

import pytest
import numpy as np
import anndata as ad
import pandas as pd
import kompot


class TestSingleConditionVariance:
    """Test single condition variance estimation in DE and DA."""
    
    def setup_method(self):
        """Set up test data with single condition having multiple samples."""
        np.random.seed(42)

        # Create test data - optimized to 60 cells for faster tests
        n_cells = 60
        n_features = 10
        n_genes = 5
        
        # Cell states and expression
        X = np.random.randn(n_cells, n_features)
        expression = np.random.randn(n_cells, n_genes)

        # Create conditions where only one has multiple samples
        conditions = ['cond1'] * 30 + ['cond2'] * 30
        samples = ['sample1'] * 15 + ['sample2'] * 15 + ['sample3'] * 30  # cond2 has only one sample
        
        # Create AnnData object
        self.adata = ad.AnnData(expression)
        self.adata.obsm['DM_EigenVectors'] = X
        self.adata.obs['condition'] = pd.Categorical(conditions)
        self.adata.obs['sample'] = pd.Categorical(samples)
        self.adata.var_names = [f'gene_{i}' for i in range(n_genes)]
        
    def test_de_single_condition_variance_enabled(self):
        """Test DE with allow_single_condition_variance=True."""
        results = kompot.compute_differential_expression(
            self.adata,
            groupby='condition',
            condition1='cond1',
            condition2='cond2',
            sample_col='sample',
            allow_single_condition_variance=True,
            n_landmarks=10,
            inplace=False,
            return_full_results=True
        )

        # Should complete successfully
        assert isinstance(results, dict)
        assert 'mahalanobis' in results['table'].columns
        assert 'mean_lfc' in results['table'].columns

    def test_de_single_condition_variance_disabled(self):
        """Test DE with allow_single_condition_variance=False (default)."""
        with pytest.raises(ValueError, match="At least 2 groups with sufficient cells"):
            kompot.compute_differential_expression(
                self.adata,
                groupby='condition',
                condition1='cond1',
                condition2='cond2',
                sample_col='sample',
                allow_single_condition_variance=False,
                n_landmarks=10,
                inplace=False
            )
            
    def test_da_single_condition_variance_enabled(self):
        """Test DA with allow_single_condition_variance=True."""
        results = kompot.compute_differential_abundance(
            self.adata,
            groupby='condition',
            condition1='cond1',
            condition2='cond2',
            sample_col='sample',
            allow_single_condition_variance=True,
            n_landmarks=10,
            inplace=False,
            return_full_results=True
        )
        
        # Should complete successfully
        assert isinstance(results, dict)
        assert 'lfc' in results['table'].columns
        assert 'neg_log10_ptp' in results['table'].columns
        
    def test_da_single_condition_variance_disabled(self):
        """Test DA with allow_single_condition_variance=False (default)."""
        with pytest.raises(ValueError, match="At least 2 groups with sufficient cells"):
            kompot.compute_differential_abundance(
                self.adata,
                groupby='condition',
                condition1='cond1',
                condition2='cond2',
                sample_col='sample',
                allow_single_condition_variance=False,
                n_landmarks=10,
                inplace=False
            )
            
    def test_both_conditions_multiple_samples_works(self):
        """Test that normal case (both conditions with multiple samples) still works."""
        # Modify data so both conditions have multiple samples
        samples_both = ['sample1'] * 15 + ['sample2'] * 15 + ['sample3'] * 15 + ['sample4'] * 15
        self.adata.obs['sample_both'] = pd.Categorical(samples_both)
        
        # Should work with default setting
        results = kompot.compute_differential_expression(
            self.adata,
            groupby='condition',
            condition1='cond1',
            condition2='cond2',
            sample_col='sample_both',
            allow_single_condition_variance=False,
            n_landmarks=10,
            inplace=False,
            return_full_results=True
        )
        
        assert isinstance(results, dict)
        assert 'mahalanobis' in results['table'].columns

    def test_no_sample_col_works_normally(self):
        """Test that cases without sample_col work normally regardless of flag."""
        # Without sample_col, the flag should have no effect
        results1 = kompot.compute_differential_expression(
            self.adata,
            groupby='condition',
            condition1='cond1',
            condition2='cond2',
            sample_col=None,
            allow_single_condition_variance=False,
            use_empirical_variance=False,
            n_landmarks=10,
            inplace=False,
            return_full_results=True
        )

        results2 = kompot.compute_differential_expression(
            self.adata,
            groupby='condition',
            condition1='cond1',
            condition2='cond2',
            sample_col=None,
            allow_single_condition_variance=True,
            use_empirical_variance=False,
            n_landmarks=10,
            inplace=False,
            return_full_results=True
        )
        
        # Both should work and produce similar results
        assert isinstance(results1, dict)
        assert isinstance(results2, dict)
        np.testing.assert_allclose(
            results1['table']['mahalanobis'].values,
            results2['table']['mahalanobis'].values,
            rtol=0.10  # Relaxed for ADVI stochasticity on small dataset
        )
        
    def test_single_variance_fallback_mechanism(self):
        """Test that single variance estimator is used for both conditions when one fails."""
        # Create data where one condition has only 1 sample (should fail)
        # and the other has multiple samples (should succeed)
        samples_fallback = ['sample1'] * 15 + ['sample2'] * 15 + ['sample3'] * 30  # cond2 has only one sample
        self.adata.obs['sample_fallback'] = pd.Categorical(samples_fallback)
        
        # This should work by using condition 1's variance for both conditions
        results = kompot.compute_differential_expression(
            self.adata,
            groupby='condition',
            condition1='cond1',
            condition2='cond2',
            sample_col='sample_fallback',
            allow_single_condition_variance=True,
            n_landmarks=10,
            inplace=False,
            return_full_results=True
        )
        
        # Should complete successfully
        assert isinstance(results, dict)
        assert 'mahalanobis' in results['table'].columns
        assert 'mean_lfc' in results['table'].columns

        # Test the reverse case (condition 1 fails, condition 2 succeeds)
        samples_reverse = ['sample1'] * 30 + ['sample2'] * 15 + ['sample3'] * 15  # cond1 has only one sample
        self.adata.obs['sample_reverse'] = pd.Categorical(samples_reverse)
        
        results_reverse = kompot.compute_differential_expression(
            self.adata,
            groupby='condition',
            condition1='cond1',
            condition2='cond2',
            sample_col='sample_reverse',
            allow_single_condition_variance=True,
            n_landmarks=10,
            inplace=False,
            return_full_results=True
        )
        
        # Should also complete successfully
        assert isinstance(results_reverse, dict)
        assert 'mahalanobis' in results_reverse['table'].columns
        assert 'mean_lfc' in results_reverse['table'].columns

    def test_both_conditions_fail_raises_error(self):
        """Test that if both conditions fail to generate variance, an error is raised."""
        # Create data where both conditions have only 1 sample each
        samples_both_fail = ['sample1'] * 30 + ['sample2'] * 30  # Each condition has only one sample
        self.adata.obs['sample_both_fail'] = pd.Categorical(samples_both_fail)
        
        with pytest.raises(ValueError, match="Both variance estimators failed to fit"):
            kompot.compute_differential_expression(
                self.adata,
                groupby='condition',
                condition1='cond1',
                condition2='cond2',
                sample_col='sample_both_fail',
                allow_single_condition_variance=True,
                n_landmarks=10,
                inplace=False
            )