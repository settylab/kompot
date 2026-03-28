"""Tests for kompot/plot/imputation.py targeting uncovered lines.

Covers plot_imputation rendering with and without scanpy, std/obs_variance
layers, auto gene selection, missing basis/layers, save paths, and
internal helpers (_detect_condition_label, _get_raw, _scatter_fallback).
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from anndata import AnnData
import os


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _close_figures():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


class TestPlotImputation:
    """Tests for plot_imputation targeting uncovered lines."""

    def _make_imputation_adata(self, with_std=True, with_obs_var=True):
        """Build AnnData with imputation layers."""
        rng = np.random.RandomState(42)
        n_obs, n_vars = 50, 8
        X = rng.randn(n_obs, n_vars).astype(np.float32)
        adata = AnnData(X=X)
        adata.var_names = [f"gene_{i}" for i in range(n_vars)]
        adata.obs_names = [f"cell_{i}" for i in range(n_obs)]
        adata.obsm["X_umap"] = rng.randn(n_obs, 2).astype(np.float32)

        cond = "treated"
        prefix = "kompot_impute"
        adata.layers[f"{prefix}_{cond}_imputed"] = rng.randn(n_obs, n_vars).astype(np.float32)
        if with_std:
            adata.layers[f"{prefix}_{cond}_std"] = np.abs(rng.randn(n_obs, n_vars)).astype(np.float32)
        if with_obs_var:
            adata.layers[f"{prefix}_{cond}_obs_variance"] = np.abs(rng.randn(n_obs, n_vars)).astype(np.float32)
        return adata

    def test_plot_imputation_basic(self):
        """Lines 218-226, 242-249, 268-275, 293-300: basic imputation plot with scanpy."""
        from kompot.plot.imputation import plot_imputation
        adata = self._make_imputation_adata()
        fig = plot_imputation(
            adata, genes=["gene_0", "gene_1"],
            return_fig=True,
        )
        assert fig is not None

    def test_plot_imputation_no_scanpy(self):
        """Lines 218-226, 242-249, 268-275, 293-300: fallback without scanpy."""
        from kompot.plot import imputation as imp_mod
        old = imp_mod._has_scanpy
        try:
            imp_mod._has_scanpy = False
            adata = self._make_imputation_adata()
            fig = imp_mod.plot_imputation(
                adata, genes=["gene_0", "gene_1"],
                return_fig=True,
            )
            assert fig is not None
        finally:
            imp_mod._has_scanpy = old

    def test_plot_imputation_no_std(self):
        """Lines 329, 332: no std layer."""
        from kompot.plot.imputation import plot_imputation
        adata = self._make_imputation_adata(with_std=False, with_obs_var=False)
        fig = plot_imputation(adata, genes=["gene_0"], return_fig=True)
        assert fig is not None

    def test_plot_imputation_auto_genes(self):
        """Lines 337-343: auto gene selection."""
        from kompot.plot.imputation import plot_imputation
        adata = self._make_imputation_adata()
        fig = plot_imputation(adata, genes=None, n_top_genes=3, return_fig=True)
        assert fig is not None

    def test_plot_imputation_missing_basis(self):
        """Lines 97: missing basis raises ValueError."""
        from kompot.plot.imputation import plot_imputation
        adata = self._make_imputation_adata()
        with pytest.raises(ValueError, match="not found"):
            plot_imputation(adata, basis="X_nonexistent")

    def test_plot_imputation_missing_layers(self):
        """Line 110: missing imputation layers raises ValueError."""
        from kompot.plot.imputation import plot_imputation
        adata = self._make_imputation_adata()
        with pytest.raises(ValueError, match="No imputation layers"):
            plot_imputation(adata, result_key="nonexistent_key")

    def test_plot_imputation_custom_title(self):
        """Line 329: custom title."""
        from kompot.plot.imputation import plot_imputation
        adata = self._make_imputation_adata()
        fig = plot_imputation(adata, genes=["gene_0"], title="Custom Title", return_fig=True)
        assert fig is not None

    def test_plot_imputation_save(self, tmp_path):
        """Lines 349-358: save and return."""
        from kompot.plot.imputation import plot_imputation
        adata = self._make_imputation_adata()
        path = str(tmp_path / "imputation.png")
        fig = plot_imputation(adata, genes=["gene_0"], save=path, return_fig=True)
        assert fig is not None
        assert os.path.exists(path)

    def test_scanpy_import_false(self):
        """Lines 20-21: scanpy import failure flag."""
        from kompot.plot import imputation as imp_mod
        # Just verify the flag exists
        assert hasattr(imp_mod, "_has_scanpy")

    def test_plot_imputation_no_obs_variance(self):
        """Lines 122-124: has_std but no obs_variance."""
        from kompot.plot.imputation import plot_imputation
        adata = self._make_imputation_adata(with_obs_var=False)
        fig = plot_imputation(adata, genes=["gene_0"], return_fig=True)
        assert fig is not None


class TestImputationDeep:
    """Deeper coverage for imputation.py."""

    def _make_imputation_adata(self, with_std=True, with_obs_var=True):
        rng = np.random.RandomState(42)
        n_obs, n_vars = 50, 8
        X = rng.randn(n_obs, n_vars).astype(np.float32)
        adata = AnnData(X=X)
        adata.var_names = [f"gene_{i}" for i in range(n_vars)]
        adata.obs_names = [f"cell_{i}" for i in range(n_obs)]
        adata.obsm["X_umap"] = rng.randn(n_obs, 2).astype(np.float32)
        cond = "treated"
        prefix = "kompot_impute"
        adata.layers[f"{prefix}_{cond}_imputed"] = rng.randn(n_obs, n_vars).astype(np.float32)
        if with_std:
            adata.layers[f"{prefix}_{cond}_std"] = np.abs(rng.randn(n_obs, n_vars)).astype(np.float32)
        if with_obs_var:
            adata.layers[f"{prefix}_{cond}_obs_variance"] = np.abs(rng.randn(n_obs, n_vars)).astype(np.float32)
        return adata

    def test_detect_condition_label_error(self):
        """Lines 361-371: _detect_condition_label raises when no matching layers."""
        from kompot.plot.imputation import _detect_condition_label
        adata = AnnData(np.random.randn(5, 3))
        with pytest.raises(ValueError, match="No imputation layers"):
            _detect_condition_label(adata, "kompot_impute")

    def test_get_raw_sparse(self):
        """Lines 335-343: _get_raw with sparse matrix."""
        from kompot.plot.imputation import _get_raw
        from scipy.sparse import csr_matrix
        adata = AnnData(csr_matrix(np.random.randn(10, 5)))
        adata.var_names = [f"g{i}" for i in range(5)]
        result = _get_raw(adata, "g0", layer=None)
        assert result.shape == (10,)

    def test_scatter_fallback(self):
        """Lines 346-358: _scatter_fallback."""
        from kompot.plot.imputation import _scatter_fallback
        fig, ax = plt.subplots()
        xy = np.random.randn(20, 2)
        values = np.random.randn(20)
        _scatter_fallback(ax, xy, values, cmap="viridis", n_obs=20)
        # Should not raise

    def test_embedding_dim_too_small(self):
        """Line 97: embedding has < 2 dimensions."""
        from kompot.plot.imputation import plot_imputation
        adata = self._make_imputation_adata()
        adata.obsm["X_1d"] = np.random.randn(adata.n_obs, 1)
        with pytest.raises(ValueError, match="2 dimensions"):
            plot_imputation(adata, basis="X_1d")
