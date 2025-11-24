"""Comprehensive tests for kompot.anndata.differential_expression module to improve coverage."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
import tempfile
import os


def create_basic_test_anndata(n_cells=30, n_genes=15):
    """Create basic test AnnData object for this test file."""
    try:
        import anndata
    except ImportError:
        pytest.skip("anndata not installed, skipping test")
        
    np.random.seed(42)
    
    # Create test data
    X = np.random.normal(0, 1, (n_cells, n_genes))
    
    # Create cell groups for testing
    groups = np.array(['A'] * (n_cells // 2) + ['B'] * (n_cells // 2))
    
    # Create embedding - smaller embedding dimension for faster computation
    obsm = {
        'DM_EigenVectors': np.random.normal(0, 1, (n_cells, 5))
    }
    
    # Create observation dataframe
    obs = pd.DataFrame({'group': groups})
    
    # Create var_names
    var_names = [f'gene_{i}' for i in range(n_genes)]
    
    # Create var DataFrame with var_names as index
    var = pd.DataFrame(index=var_names)
    
    return anndata.AnnData(X=X, obs=obs, var=var, obsm=obsm)


class TestDifferentialExpressionCoverage:
    """Comprehensive tests to improve differential_expression.py coverage."""

    @pytest.fixture
    def basic_adata(self):
        """Create basic AnnData for testing."""
        return create_basic_test_anndata(n_cells=30, n_genes=15)

    def test_prepare_null_genes_integer(self, basic_adata):
        """Test _prepare_null_genes with integer input."""
        from kompot.anndata.differential_expression import _prepare_null_genes
        
        available_genes = basic_adata.var_names.tolist()
        
        # Test normal case
        indices, used_replacement = _prepare_null_genes(5, available_genes, 42)
        assert len(indices) == 5
        assert not used_replacement
        assert all(0 <= idx < len(available_genes) for idx in indices)
        
        # Test with more genes than available
        indices, used_replacement = _prepare_null_genes(20, available_genes, 42)
        assert len(indices) == 20
        assert used_replacement
        
        # Test with zero
        indices, used_replacement = _prepare_null_genes(0, available_genes, 42)
        assert len(indices) == 0
        assert not used_replacement
        
        # Test with None
        indices, used_replacement = _prepare_null_genes(None, available_genes, 42)
        assert len(indices) == 0
        assert not used_replacement

    def test_prepare_null_genes_list(self, basic_adata):
        """Test _prepare_null_genes with list input."""
        from kompot.anndata.differential_expression import _prepare_null_genes
        
        available_genes = basic_adata.var_names.tolist()
        
        # Test valid list
        indices, used_replacement = _prepare_null_genes([0, 1, 2], available_genes, 42)
        assert indices == [0, 1, 2]
        assert not used_replacement
        
        # Test invalid indices
        with pytest.raises(ValueError, match="Invalid gene indices"):
            _prepare_null_genes([0, 999], available_genes, 42)
        
        # Test non-integer list
        with pytest.raises(ValueError, match="All elements in null_genes list must be integers"):
            _prepare_null_genes([0, "gene1"], available_genes, 42)

    def test_prepare_null_genes_errors(self, basic_adata):
        """Test _prepare_null_genes error cases."""
        from kompot.anndata.differential_expression import _prepare_null_genes
        
        available_genes = basic_adata.var_names.tolist()
        
        # Test negative integer
        with pytest.raises(ValueError, match="null_genes must be positive"):
            _prepare_null_genes(-5, available_genes, 42)
        
        # Test invalid type
        with pytest.raises(ValueError, match="null_genes must be int, list of ints, or None"):
            _prepare_null_genes("invalid", available_genes, 42)

    def test_generate_shuffled_expression(self, basic_adata):
        """Test _generate_shuffled_expression function."""
        from kompot.anndata.differential_expression import _generate_shuffled_expression
        
        # Create test expression matrices
        expr1 = np.random.rand(15, 10)  # 15 cells, 10 genes
        expr2 = np.random.rand(15, 10)  # 15 cells, 10 genes
        null_gene_indices = [0, 1, 2]
        
        shuffled1, shuffled2 = _generate_shuffled_expression(expr1, expr2, null_gene_indices, 42)
        
        # Check shapes - function returns only null genes
        assert shuffled1.shape == (15, 3)  # 15 cells, 3 null genes
        assert shuffled2.shape == (15, 3)  # 15 cells, 3 null genes
        
        # Check that values are not all the same (shuffling happened)
        combined_orig = np.concatenate([expr1[:, 0], expr2[:, 0]])  # Original first gene
        combined_shuffled = np.concatenate([shuffled1[:, 0], shuffled2[:, 0]])  # Shuffled first gene
        
        # Values should be the same set but in different order
        np.testing.assert_array_equal(np.sort(combined_orig), np.sort(combined_shuffled))
        
        # Test with empty indices
        empty_shuffled1, empty_shuffled2 = _generate_shuffled_expression(expr1, expr2, [], 42)
        assert empty_shuffled1.shape == (15, 0)
        assert empty_shuffled2.shape == (15, 0)

    def test_compute_differential_expression_basic_parameters(self, basic_adata):
        """Test basic parameter handling in compute_differential_expression."""
        from kompot.anndata import compute_differential_expression
        
        # Test with minimal parameters
        result = compute_differential_expression(
            basic_adata,
            groupby='group',
            condition1='A',
            condition2='B',
            compute_mahalanobis=False,
            n_landmarks=3,
            return_full_results=True
        )
        
        assert isinstance(result, dict)
        assert 'table' in result
        assert 'mean_lfc' in result['table'].columns
        assert 'model' in result

    def test_compute_differential_expression_with_layer(self, basic_adata):
        """Test compute_differential_expression with layer parameter."""
        from kompot.anndata import compute_differential_expression
        
        # Add a layer to the adata
        basic_adata.layers['test_layer'] = basic_adata.X.copy() * 2
        
        result = compute_differential_expression(
            basic_adata,
            groupby='group',
            condition1='A',
            condition2='B',
            layer='test_layer',
            compute_mahalanobis=False,
            n_landmarks=3,
            return_full_results=True
        )
        
        assert isinstance(result, dict)
        assert 'table' in result
        assert 'mean_lfc' in result['table'].columns

    def test_compute_differential_expression_with_genes_subset(self, basic_adata):
        """Test compute_differential_expression with specific genes."""
        from kompot.anndata import compute_differential_expression
        
        # Test with subset of genes
        subset_genes = basic_adata.var_names[:5].tolist()
        
        result = compute_differential_expression(
            basic_adata,
            groupby='group',
            condition1='A',
            condition2='B',
            genes=subset_genes,
            compute_mahalanobis=False,
            n_landmarks=3,
            return_full_results=True
        )
        
        assert isinstance(result, dict)
        assert 'table' in result
        assert 'mean_lfc' in result['table'].columns

    def test_compute_differential_expression_with_landmarks(self, basic_adata):
        """Test compute_differential_expression with provided landmarks."""
        from kompot.anndata import compute_differential_expression
        
        # Create custom landmarks
        landmarks = np.random.rand(5, 5)  # 5 landmarks with 5D embedding
        
        result = compute_differential_expression(
            basic_adata,
            groupby='group',
            condition1='A',
            condition2='B',
            landmarks=landmarks,
            compute_mahalanobis=False,
            n_landmarks=None,  # Should be ignored when landmarks provided
            return_full_results=True
        )
        
        assert isinstance(result, dict)
        assert 'table' in result
        assert 'mean_lfc' in result['table'].columns

    def test_compute_differential_expression_with_cell_filter(self, basic_adata):
        """Test compute_differential_expression with cell filtering."""
        from kompot.anndata import compute_differential_expression
        
        # Create cell filter
        cell_filter = {'group': 'A'}
        
        with pytest.raises(Exception):  # Should raise because not enough cells in one condition
            compute_differential_expression(
                basic_adata,
                groupby='group',
                condition1='A',
                condition2='B',
                cell_filter=cell_filter,
                compute_mahalanobis=False,
                n_landmarks=3,
                return_full_results=True
            )

    def test_compute_differential_expression_memory_options(self, basic_adata):
        """Test compute_differential_expression with memory/disk options."""
        from kompot.anndata import compute_differential_expression
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = compute_differential_expression(
                basic_adata,
                groupby='group',
                condition1='A',
                condition2='B',
                compute_mahalanobis=False,
                n_landmarks=3,
                store_arrays_on_disk=True,
                disk_storage_dir=tmpdir,
                max_memory_ratio=0.5,
                return_full_results=True
            )
            
            assert isinstance(result, dict)
            assert 'table' in result
            assert 'mean_lfc' in result['table'].columns

    def test_compute_differential_expression_with_copy(self, basic_adata):
        """Test compute_differential_expression with copy=True."""
        from kompot.anndata import compute_differential_expression
        
        original_obs_cols = set(basic_adata.obs.columns)
        
        result = compute_differential_expression(
            basic_adata,
            groupby='group',
            condition1='A',
            condition2='B',
            compute_mahalanobis=False,
            n_landmarks=3,
            copy=True,
            return_full_results=True
        )
        
        # When copy=True and return_full_results=True, result is a tuple (results_dict, adata_copy)
        if isinstance(result, tuple):
            results_dict, adata_copy = result
            assert isinstance(results_dict, dict)
        else:
            assert isinstance(result, dict)
            
        # Original should be unchanged
        assert set(basic_adata.obs.columns) == original_obs_cols

    def test_compute_differential_expression_fdr_disabled(self, basic_adata):
        """Test compute_differential_expression with FDR disabled."""
        from kompot.anndata import compute_differential_expression
        
        result = compute_differential_expression(
            basic_adata,
            groupby='group',
            condition1='A',
            condition2='B',
            compute_mahalanobis=False,
            n_landmarks=3,
            null_genes=None,  # Disable FDR
            return_full_results=True
        )
        
        assert isinstance(result, dict)
        assert 'table' in result
        assert 'mean_lfc' in result['table'].columns
        # Should not have FDR-related columns
        assert 'is_de' not in result['table'].columns

    def test_compute_differential_expression_with_single_condition_variance(self, basic_adata):
        """Test compute_differential_expression with allow_single_condition_variance."""
        from kompot.anndata import compute_differential_expression
        
        result = compute_differential_expression(
            basic_adata,
            groupby='group',
            condition1='A',
            condition2='B',
            compute_mahalanobis=False,
            n_landmarks=3,
            allow_single_condition_variance=True,
            return_full_results=True
        )
        
        assert isinstance(result, dict)
        assert 'table' in result
        assert 'mean_lfc' in result['table'].columns

    def test_compute_differential_expression_store_landmarks(self, basic_adata):
        """Test compute_differential_expression with store_landmarks=True."""
        from kompot.anndata import compute_differential_expression
        
        result = compute_differential_expression(
            basic_adata,
            groupby='group',
            condition1='A',
            condition2='B',
            compute_mahalanobis=False,
            n_landmarks=3,
            store_landmarks=True,
            return_full_results=True
        )
        
        assert isinstance(result, dict)
        assert 'table' in result
        assert 'mean_lfc' in result['table'].columns

        # Check that landmarks were stored
        assert 'kompot_de' in basic_adata.uns


    def test_compute_differential_expression_jit_compile(self, basic_adata):
        """Test compute_differential_expression with JIT compilation."""
        from kompot.anndata import compute_differential_expression
        
        result = compute_differential_expression(
            basic_adata,
            groupby='group',
            condition1='A',
            condition2='B',
            compute_mahalanobis=False,
            n_landmarks=3,
            jit_compile=True,
            return_full_results=True
        )
        
        assert isinstance(result, dict)
        assert 'table' in result
        assert 'mean_lfc' in result['table'].columns

    def test_compute_differential_expression_custom_parameters(self, basic_adata):
        """Test compute_differential_expression with custom sigma, ls, eps parameters."""
        from kompot.anndata import compute_differential_expression
        
        result = compute_differential_expression(
            basic_adata,
            groupby='group',
            condition1='A',
            condition2='B',
            compute_mahalanobis=False,
            n_landmarks=3,
            sigma=2.0,
            ls=0.5,
            eps=1e-6,
            return_full_results=True
        )
        
        assert isinstance(result, dict)
        assert 'table' in result
        assert 'mean_lfc' in result['table'].columns

    def test_compute_differential_expression_progress_disabled(self, basic_adata):
        """Test compute_differential_expression with progress=False."""
        from kompot.anndata import compute_differential_expression
        
        result = compute_differential_expression(
            basic_adata,
            groupby='group',
            condition1='A',
            condition2='B',
            compute_mahalanobis=False,
            n_landmarks=3,
            progress=False,
            return_full_results=True
        )
        
        assert isinstance(result, dict)
        assert 'table' in result
        assert 'mean_lfc' in result['table'].columns

    def test_compute_differential_expression_batch_size(self, basic_adata):
        """Test compute_differential_expression with different batch sizes."""
        from kompot.anndata import compute_differential_expression
        
        result = compute_differential_expression(
            basic_adata,
            groupby='group',
            condition1='A',
            condition2='B',
            compute_mahalanobis=False,
            n_landmarks=3,
            batch_size=50,
            return_full_results=True
        )
        
        assert isinstance(result, dict)
        assert 'table' in result
        assert 'mean_lfc' in result['table'].columns

    def test_compute_differential_expression_store_posterior_covariance(self, basic_adata):
        """Test compute_differential_expression with store_posterior_covariance."""
        from kompot.anndata import compute_differential_expression
        
        result = compute_differential_expression(
            basic_adata,
            groupby='group',
            condition1='A',
            condition2='B',
            compute_mahalanobis=False,
            n_landmarks=3,
            store_posterior_covariance=True,
            return_full_results=True
        )
        
        assert isinstance(result, dict)
        assert 'table' in result
        assert 'mean_lfc' in result['table'].columns