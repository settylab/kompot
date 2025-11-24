"""Tests for the ptp (posterior tail probability) functionality."""

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import jax.scipy.stats as jax_stats
import jax.numpy as jnp

import anndata


def create_test_adata_with_ptp(n_cells=60, n_genes=50):
    """Create test AnnData with realistic ptp values computed from Mahalanobis distances."""
    np.random.seed(42)
    
    # Create test data
    X = np.random.normal(0, 1, (n_cells, n_genes))
    var_names = [f'gene_{i}' for i in range(n_genes)]
    
    # Create AnnData
    adata = anndata.AnnData(X, var=pd.DataFrame(index=var_names))
    adata.obs['condition'] = ['A'] * (n_cells // 2) + ['B'] * (n_cells // 2)
    
    # Create realistic Mahalanobis distances
    mahalanobis_distances = np.abs(np.random.gamma(2, 1, n_genes))  # Positive values
    
    # Compute ptp from Mahalanobis distances using chi2 distribution
    degrees_of_freedom = 10  # Typical number of dimensions
    mahalanobis_squared = mahalanobis_distances ** 2
    ptp_values = np.array(jax_stats.chi2.sf(mahalanobis_squared, df=degrees_of_freedom))
    
    # Add differential expression metrics
    adata.var['kompot_de_mahalanobis_A_to_B'] = mahalanobis_distances
    adata.var['kompot_de_ptp_A_to_B'] = ptp_values
    adata.var['kompot_de_mean_lfc_A_to_B'] = np.random.normal(0, 2, n_genes)
    
    # Add FDR values for comparison
    adata.var['kompot_de_mahalanobis_local_fdr_A_to_B'] = np.random.uniform(0, 0.5, n_genes)
    adata.var['kompot_de_mahalanobis_tail_fdr_A_to_B'] = np.random.uniform(0, 0.5, n_genes)
    adata.var['kompot_de_is_de_A_to_B'] = ptp_values < 0.05  # Significant at p < 0.05
    
    # Add run history for proper testing
    adata.uns['kompot_de_run_history'] = [{
        'params': {'condition1': 'A', 'condition2': 'B'},
        'field_names': {
            'mahalanobis_key': 'kompot_de_mahalanobis_A_to_B',
            'ptp_key': 'kompot_de_ptp_A_to_B',
            'mean_lfc_key': 'kompot_de_mean_lfc_A_to_B'
        },
        'fdr_keys': {
            'local_fdr_key': 'kompot_de_mahalanobis_local_fdr_A_to_B',
            'tail_fdr_key': 'kompot_de_mahalanobis_tail_fdr_A_to_B',
            'is_de_key': 'kompot_de_is_de_A_to_B'
        },
        'ptp_key': 'kompot_de_ptp_A_to_B'  # This should be in field_names
    }]
    
    return adata


class TestPTPFunctionality:
    """Test class for ptp functionality."""
    
    def test_ptp_computation_basic(self):
        """Test basic ptp computation logic."""
        # Test with known values
        mahalanobis_distances = np.array([1.0, 2.0, 3.0])
        degrees_of_freedom = 5
        
        # Square distances and compute ptp
        mahalanobis_squared = mahalanobis_distances ** 2
        ptp = np.array(jax_stats.chi2.sf(mahalanobis_squared, df=degrees_of_freedom))
        
        # Verify properties
        assert np.all(ptp >= 0), "PTP values should be non-negative"
        assert np.all(ptp <= 1), "PTP values should be <= 1"
        assert np.all(np.diff(ptp) <= 0), "PTP should decrease with increasing Mahalanobis distance"
        
    def test_ptp_transform_for_volcano(self):
        """Test -log10 transformation of ptp for volcano plots."""
        ptp_values = np.array([0.1, 0.01, 0.001])
        
        # Apply transformation
        transformed = -np.log10(np.maximum(ptp_values, 1e-300))
        
        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(transformed, expected, rtol=1e-10)
        
    def test_volcano_de_with_ptp(self):
        """Test volcano_de plot with ptp y-axis type."""
        from kompot.plot.volcano import volcano_de
        
        adata = create_test_adata_with_ptp()
        
        # Test ptp y-axis
        fig, ax = volcano_de(
            adata,
            lfc_key="kompot_de_mean_lfc_A_to_B",
            score_key="kompot_de_mahalanobis_A_to_B",
            y_axis_type="ptp",
            significance_threshold=0.05,
            return_fig=True
        )
        
        assert ax.get_ylabel() == "-log10(Posterior Tail Probability)"
        plt.close(fig)
        
    def test_volcano_de_ptp_threshold_line(self):
        """Test that significance threshold line is shown for ptp."""
        from kompot.plot.volcano import volcano_de
        
        adata = create_test_adata_with_ptp()
        
        fig, ax = volcano_de(
            adata,
            lfc_key="kompot_de_mean_lfc_A_to_B",
            score_key="kompot_de_mahalanobis_A_to_B",
            y_axis_type="ptp",
            significance_threshold=0.01,
            show_thresholds=True,
            return_fig=True
        )
        
        # Check that a horizontal line was added
        hlines = [line for line in ax.lines if hasattr(line, '_y') and len(np.unique(line._y)) == 1]
        assert len(hlines) > 0, "Expected horizontal threshold line to be present"
        
        plt.close(fig)
        
    def test_volcano_de_ptp_gene_selection(self):
        """Test gene selection with ptp threshold."""
        from kompot.plot.volcano import volcano_de
        
        adata = create_test_adata_with_ptp()
        
        # Set some genes to be clearly significant
        adata.var.loc['gene_0', 'kompot_de_ptp_A_to_B'] = 0.001  # Very significant
        adata.var.loc['gene_1', 'kompot_de_ptp_A_to_B'] = 0.005  # Significant
        adata.var.loc['gene_2', 'kompot_de_ptp_A_to_B'] = 0.1    # Not significant
        
        fig, ax = volcano_de(
            adata,
            lfc_key="kompot_de_mean_lfc_A_to_B",
            score_key="kompot_de_mahalanobis_A_to_B",
            y_axis_type="ptp",
            significance_threshold=0.01,
            return_fig=True
        )
        
        # Should highlight genes with ptp < 0.01
        plt.close(fig)
        
    def test_ptp_column_inference(self):
        """Test that ptp column is correctly inferred from mahalanobis key."""
        from kompot.plot.volcano import volcano_de
        
        adata = create_test_adata_with_ptp()
        
        # Remove run history to test fallback inference
        del adata.uns['kompot_de_run_history']
        
        fig, ax = volcano_de(
            adata,
            lfc_key='kompot_de_mean_lfc_A_to_B',
            score_key='kompot_de_mahalanobis_A_to_B',  # Should infer ptp from this
            y_axis_type="ptp",
            return_fig=True
        )
        
        assert ax.get_ylabel() == "-log10(Posterior Tail Probability)"
        plt.close(fig)
        
    def test_ptp_vs_fdr_consistency(self):
        """Test that ptp and FDR can be used interchangeably in volcano plots."""
        from kompot.plot.volcano import volcano_de
        
        adata = create_test_adata_with_ptp()
        
        # Test both ptp and local_fdr
        fig1, ax1 = volcano_de(
            adata,
            lfc_key="kompot_de_mean_lfc_A_to_B",
            score_key="kompot_de_mahalanobis_A_to_B",
            y_axis_type="ptp",
            significance_threshold=0.05,
            return_fig=True
        )
        
        fig2, ax2 = volcano_de(
            adata,
            lfc_key="kompot_de_mean_lfc_A_to_B",
            score_key="kompot_de_mahalanobis_A_to_B",
            y_axis_type="local_fdr",
            significance_threshold=0.05,
            return_fig=True
        )
        
        # Both should work without errors
        assert ax1.get_ylabel() == "-log10(Posterior Tail Probability)"
        assert ax2.get_ylabel() == "-log10(Local FDR)"
        
        plt.close(fig1)
        plt.close(fig2)
        
    def test_custom_column_name(self):
        """Test using custom column name for y-axis."""
        from kompot.plot.volcano import volcano_de
        
        adata = create_test_adata_with_ptp()
        
        # Test using ptp column directly
        fig, ax = volcano_de(
            adata,
            lfc_key="kompot_de_mean_lfc_A_to_B",
            score_key="kompot_de_mahalanobis_A_to_B",
            y_axis_type="kompot_de_ptp_A_to_B",  # Custom column name
            return_fig=True
        )
        
        # Should work without applying transformation (since it's not a known type)
        plt.close(fig)
        
    def test_ptp_error_handling(self):
        """Test error handling for missing ptp columns."""
        from kompot.plot.volcano import volcano_de
        
        adata = create_test_adata_with_ptp()
        
        # Remove ptp column
        del adata.var['kompot_de_ptp_A_to_B']
        
        # Should fall back to mahalanobis when ptp not found
        fig, ax = volcano_de(
            adata,
            lfc_key="kompot_de_mean_lfc_A_to_B",
            score_key="kompot_de_mahalanobis_A_to_B",
            y_axis_type="ptp",
            return_fig=True
        )
        
        # Should use mahalanobis as fallback
        plt.close(fig)
        
    def test_significance_threshold_parameter(self):
        """Test the new significance_threshold parameter vs old fdr_threshold."""
        from kompot.plot.volcano import volcano_de
        
        adata = create_test_adata_with_ptp()
        
        # Test with significance_threshold
        fig, ax = volcano_de(
            adata,
            lfc_key="kompot_de_mean_lfc_A_to_B",
            score_key="kompot_de_mahalanobis_A_to_B",
            y_axis_type="ptp",
            significance_threshold=0.01,
            return_fig=True
        )
        
        plt.close(fig)
        
        # Test with mahalanobis threshold
        fig, ax = volcano_de(
            adata,
            lfc_key="kompot_de_mean_lfc_A_to_B",
            score_key="kompot_de_mahalanobis_A_to_B",
            y_axis_type="mahalanobis",
            significance_threshold=2.0,
            return_fig=True
        )
        
        plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])