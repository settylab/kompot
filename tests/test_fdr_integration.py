"""Tests for FDR integration in compute_differential_expression."""

import numpy as np
import pandas as pd
import pytest

from tests.test_anndata_functions import create_test_anndata


def create_test_anndata_with_differential_genes(n_cells=350, n_genes=500, n_differential=50):
    """Create test AnnData with known differential genes."""
    import anndata as ad
    
    np.random.seed(42)
    
    # Create conditions
    n_cells_cond1 = n_cells // 2
    n_cells_cond2 = n_cells - n_cells_cond1
    
    # Create cell states
    X_cond1 = np.random.normal(0, 1, (n_cells_cond1, 10))
    X_cond2 = np.random.normal([0.5, 0.2] + [0]*8, 1, (n_cells_cond2, 10))
    X_combined = np.vstack([X_cond1, X_cond2])
    
    # Create expression data
    expr_base = np.random.negative_binomial(10, 0.3, (n_cells, n_genes)).astype(float)
    
    # Add differential expression
    differential_genes = np.random.choice(n_genes, n_differential, replace=False)
    for gene_idx in differential_genes:
        fold_change = np.random.uniform(2, 4)
        expr_base[n_cells_cond1:, gene_idx] *= fold_change
    
    # Create AnnData
    gene_names = [f"Gene_{i:04d}" for i in range(n_genes)]
    cell_names = [f"Cell_{i:04d}" for i in range(n_cells)]
    
    adata = ad.AnnData(
        X=expr_base,
        obs=pd.DataFrame({
            'condition': ['Ctrl'] * n_cells_cond1 + ['Treat'] * n_cells_cond2,
        }, index=cell_names),
        var=pd.DataFrame({
            'is_differential': [i in differential_genes for i in range(n_genes)]
        }, index=gene_names)
    )
    
    adata.obsm['DM_EigenVectors'] = X_combined
    
    return adata, differential_genes


class TestFDRIntegration:
    """Tests for FDR functionality integration."""
    
    def test_fdr_enabled_basic(self):
        """Test basic FDR functionality."""
        try:
            import anndata
            from kompot.anndata import compute_differential_expression
        except ImportError:
            pytest.skip("anndata not installed")
        
        adata, true_differential_genes = create_test_anndata_with_differential_genes()
        
        # Run with FDR
        results = compute_differential_expression(
            adata,
            groupby='condition',
            condition1='Ctrl',
            condition2='Treat',
            null_genes=100,
            null_seed=42,
            fdr_threshold=0.05,
            return_full_results=True,
            result_key="test_fdr",
            overwrite=True
        )
        
        # Check FDR results are present
        fdr_keys = ['mahalanobis_pvalues', 'mahalanobis_local_fdr', 
                   'mahalanobis_tail_fdr', 'is_differentially_expressed']
        for key in fdr_keys:
            assert key in results, f"Missing key: {key}"
            assert len(results[key]) == adata.n_vars
        
        # Check AnnData columns
        fdr_columns = ['test_fdr_mahalanobis_pvalue', 'test_fdr_mahalanobis_local_fdr',
                      'test_fdr_mahalanobis_tail_fdr', 'test_fdr_is_de']
        for col in fdr_columns:
            assert col in adata.var.columns, f"Missing column: {col}"
        
        # Check value ranges
        assert np.all(adata.var['test_fdr_mahalanobis_pvalue'] >= 0)
        assert np.all(adata.var['test_fdr_mahalanobis_pvalue'] <= 1)
        assert np.all(adata.var['test_fdr_mahalanobis_local_fdr'] >= 0)
        assert np.all(adata.var['test_fdr_mahalanobis_local_fdr'] <= 1)
        assert adata.var['test_fdr_is_de'].dtype == bool
        
        # Should detect some genes as significant
        n_significant = np.sum(adata.var['test_fdr_is_de'])
        assert n_significant > 0, "Should detect at least some significant genes"
        assert n_significant < adata.n_vars * 0.5, "Shouldn't detect too many genes"
    
    def test_fdr_disabled(self):
        """Test that function works when FDR is disabled."""
        try:
            import anndata
            from kompot.anndata import compute_differential_expression
        except ImportError:
            pytest.skip("anndata not installed")
        
        adata, _ = create_test_anndata_with_differential_genes()
        
        # Run without FDR
        results = compute_differential_expression(
            adata,
            groupby='condition',
            condition1='Ctrl',
            condition2='Treat',
            null_genes=None,  # Disable FDR
            return_full_results=True,
            result_key="no_fdr_test",
            overwrite=True
        )
        
        # Should not have FDR results
        fdr_keys = ['mahalanobis_pvalues', 'mahalanobis_local_fdr', 
                   'mahalanobis_tail_fdr', 'is_differentially_expressed']
        for key in fdr_keys:
            assert key not in results, f"Should not have key when disabled: {key}"
        
        # Should still have regular results
        assert 'mahalanobis_distances' in results
        
        # Should not have FDR columns
        fdr_columns = ['no_fdr_test_mahalanobis_pvalue', 'no_fdr_test_mahalanobis_local_fdr',
                      'no_fdr_test_mahalanobis_tail_fdr', 'no_fdr_test_is_de']
        for col in fdr_columns:
            assert col not in adata.var.columns, f"Should not have column when disabled: {col}"
    
    def test_fdr_reproducibility(self):
        """Test that FDR results are reproducible."""
        try:
            import anndata
            from kompot.anndata import compute_differential_expression
        except ImportError:
            pytest.skip("anndata not installed")
        
        adata1, _ = create_test_anndata_with_differential_genes()
        adata2 = adata1.copy()
        
        params = dict(
            groupby='condition',
            condition1='Ctrl',
            condition2='Treat',
            null_genes=50,
            null_seed=123,
            random_state=456,
            return_full_results=True,
            overwrite=True
        )
        
        results1 = compute_differential_expression(adata1, result_key="repro1", **params)
        results2 = compute_differential_expression(adata2, result_key="repro2", **params)
        
        # P-values should be identical
        np.testing.assert_array_equal(
            results1['mahalanobis_pvalues'],
            results2['mahalanobis_pvalues'],
            err_msg="P-values should be reproducible"
        )
        
        # DE calls should be identical
        np.testing.assert_array_equal(
            results1['is_differentially_expressed'],
            results2['is_differentially_expressed'],
            err_msg="DE calls should be reproducible"
        )
    
    def test_fdr_with_specific_genes(self):
        """Test FDR with user-specified null genes."""
        try:
            import anndata
            from kompot.anndata import compute_differential_expression
        except ImportError:
            pytest.skip("anndata not installed")
        
        adata, _ = create_test_anndata_with_differential_genes()
        
        # Use specific null gene indices
        null_gene_indices = [10, 50, 100, 200, 300]
        
        results = compute_differential_expression(
            adata,
            groupby='condition',
            condition1='Ctrl',
            condition2='Treat',
            null_genes=null_gene_indices,
            null_seed=42,
            return_full_results=True,
            result_key="specific_null",
            overwrite=True
        )
        
        # Should work with specific indices
        assert 'mahalanobis_pvalues' in results
        assert len(results['mahalanobis_pvalues']) == adata.n_vars
    
    def test_fdr_thresholds(self):
        """Test different FDR thresholds."""
        try:
            import anndata
            from kompot.anndata import compute_differential_expression
        except ImportError:
            pytest.skip("anndata not installed")
        
        adata1, _ = create_test_anndata_with_differential_genes()
        adata2 = adata1.copy()
        
        # Run with different thresholds
        results_strict = compute_differential_expression(
            adata1, groupby='condition', condition1='Ctrl', condition2='Treat',
            null_genes=100, fdr_threshold=0.01, return_full_results=True,
            result_key="strict", overwrite=True
        )
        
        results_lenient = compute_differential_expression(
            adata2, groupby='condition', condition1='Ctrl', condition2='Treat',
            null_genes=100, fdr_threshold=0.1, return_full_results=True,
            result_key="lenient", overwrite=True
        )
        
        n_strict = np.sum(results_strict['is_differentially_expressed'])
        n_lenient = np.sum(results_lenient['is_differentially_expressed'])
        
        # Lenient threshold should detect more genes
        assert n_lenient >= n_strict, "Lenient threshold should detect >= genes than strict"
    
    def test_fdr_backwards_compatibility(self):
        """Test that adding FDR doesn't break existing functionality."""
        try:
            import anndata
            from kompot.anndata import compute_differential_expression
        except ImportError:
            pytest.skip("anndata not installed")
        
        adata = create_test_anndata()
        
        # Run with old parameters (no FDR parameters)
        results = compute_differential_expression(
            adata,
            groupby='group',
            condition1='A',
            condition2='B',
            return_full_results=True
        )
        
        # Should work without errors
        assert 'mean_log_fold_change' in results
        assert 'mahalanobis_distances' in results


if __name__ == "__main__":
    pytest.main([__file__])