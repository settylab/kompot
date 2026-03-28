"""Tests for kompot/plot/expression.py targeting uncovered lines.

Covers plot_gene_expression rendering, key inference, layer fallbacks,
basis handling, and edge cases for condition swapping.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from anndata import AnnData
import json


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _close_figures():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


def _make_de_adata(n_obs=30, n_vars=10, condition1="Young", condition2="Old",
                   result_key="kompot_de", with_layers=True, with_run_history=True):
    """Build a minimal AnnData with DE results and run history."""
    rng = np.random.RandomState(0)
    X = rng.randn(n_obs, n_vars).astype(np.float32)
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)])

    lfc_col = f"{result_key}_{condition1}_to_{condition2}_mean_lfc"
    mahal_col = f"{result_key}_{condition1}_to_{condition2}_mahalanobis"

    var[lfc_col] = rng.randn(n_vars).astype(np.float32)
    var[mahal_col] = rng.uniform(0, 5, n_vars).astype(np.float32)

    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_obs)])
    adata = AnnData(X=X, obs=obs, var=var)
    adata.obsm["X_umap"] = rng.randn(n_obs, 2).astype(np.float32)

    imputed1 = f"{result_key}_{condition1}_imputed"
    imputed2 = f"{result_key}_{condition2}_imputed"
    fc_layer = f"{result_key}_{condition1}_to_{condition2}_fold_change"

    if with_layers:
        adata.layers[imputed1] = rng.randn(n_obs, n_vars).astype(np.float32)
        adata.layers[imputed2] = rng.randn(n_obs, n_vars).astype(np.float32)
        adata.layers[fc_layer] = rng.randn(n_obs, n_vars).astype(np.float32)

    if with_run_history:
        adata.uns[result_key] = {
            "run_history": json.dumps([{
                "timestamp": "2025-01-01T00:00:00",
                "params": {
                    "condition1": condition1,
                    "condition2": condition2,
                },
                "field_names": {
                    "mean_lfc_key": lfc_col,
                    "mahalanobis_key": mahal_col,
                    "imputed_key_1": imputed1,
                    "imputed_key_2": imputed2,
                    "fold_change_key": fc_layer,
                },
                "imputed_layer_keys": {
                    "condition1": imputed1,
                    "condition2": imputed2,
                    "fold_change": fc_layer,
                },
                "adjusted_run_id": 0,
            }]),
        }

    return adata


class TestPlotGeneExpression:
    """Tests for plot_gene_expression targeting uncovered lines."""

    def test_scanpy_not_available(self):
        """Lines 79-80: scanpy not available returns None."""
        from kompot.plot import expression as expr_mod
        old = expr_mod._has_scanpy
        try:
            expr_mod._has_scanpy = False
            adata = _make_de_adata()
            result = expr_mod.plot_gene_expression(adata, "gene_0")
            assert result is None
        finally:
            expr_mod._has_scanpy = old

    def test_no_keys_inferred(self):
        """Lines 169, 172: keys cannot be inferred, warnings logged."""
        from kompot.plot.expression import plot_gene_expression
        adata = _make_de_adata(with_run_history=False, with_layers=False)
        # No run history and no recognizable keys
        adata.var.drop(columns=[c for c in adata.var.columns if "lfc" in c or "mahalanobis" in c],
                       inplace=True, errors="ignore")
        result = plot_gene_expression(adata, "gene_0", return_fig=True)
        # Should still produce a figure (with limited panels)
        assert result is not None

    def test_no_run_history_default_conditions(self):
        """Lines 182, 209-214: no run_history -> default condition names."""
        from kompot.plot.expression import plot_gene_expression
        adata = _make_de_adata(with_run_history=False)
        lfc_col = [c for c in adata.var.columns if "mean_lfc" in c][0]
        mahal_col = [c for c in adata.var.columns if "mahalanobis" in c][0]
        fig = plot_gene_expression(
            adata, "gene_0",
            lfc_key=lfc_col, score_key=mahal_col,
            return_fig=True,
        )
        assert fig is not None

    def test_positive_run_id(self):
        """Line 184: positive run_id path."""
        from kompot.plot.expression import plot_gene_expression
        adata = _make_de_adata()
        lfc_col = [c for c in adata.var.columns if "mean_lfc" in c][0]
        mahal_col = [c for c in adata.var.columns if "mahalanobis" in c][0]
        fig = plot_gene_expression(
            adata, "gene_0",
            lfc_key=lfc_col, score_key=mahal_col,
            run_id=0,
            return_fig=True,
        )
        assert fig is not None

    def test_infer_layer_from_run_info(self):
        """Lines 227, 234: layer inferred from run_info params."""
        from kompot.plot.expression import plot_gene_expression
        adata = _make_de_adata()
        lfc_col = [c for c in adata.var.columns if "mean_lfc" in c][0]
        mahal_col = [c for c in adata.var.columns if "mahalanobis" in c][0]
        # Add a layer param to run info
        run_history = json.loads(adata.uns["kompot_de"]["run_history"])
        run_history[0]["params"]["layer"] = "test_layer"
        adata.uns["kompot_de"]["run_history"] = json.dumps(run_history)
        adata.layers["test_layer"] = np.random.randn(adata.n_obs, adata.n_vars).astype(np.float32)
        fig = plot_gene_expression(
            adata, "gene_0",
            lfc_key=lfc_col, score_key=mahal_col,
            return_fig=True,
        )
        assert fig is not None

    def test_ignore_fold_change_layer(self):
        """Line 234: fold_change layer in params is ignored."""
        from kompot.plot.expression import plot_gene_expression
        adata = _make_de_adata()
        lfc_col = [c for c in adata.var.columns if "mean_lfc" in c][0]
        mahal_col = [c for c in adata.var.columns if "mahalanobis" in c][0]
        run_history = json.loads(adata.uns["kompot_de"]["run_history"])
        run_history[0]["params"]["layer"] = "fold_change_something"
        adata.uns["kompot_de"]["run_history"] = json.dumps(run_history)
        fig = plot_gene_expression(
            adata, "gene_0",
            lfc_key=lfc_col, score_key=mahal_col,
            return_fig=True,
        )
        assert fig is not None

    def test_basis_none_fallback_scatter(self):
        """Lines 342-343, 347, 353: basis=None -> fallback scatter (scanpy available)."""
        from kompot.plot.expression import plot_gene_expression
        adata = _make_de_adata(with_layers=True)
        lfc_col = [c for c in adata.var.columns if "mean_lfc" in c][0]
        mahal_col = [c for c in adata.var.columns if "mahalanobis" in c][0]
        fig = plot_gene_expression(
            adata, "gene_0",
            lfc_key=lfc_col, score_key=mahal_col,
            basis=None,
            return_fig=True,
        )
        assert fig is not None

    def test_basis_none_with_layer(self):
        """Lines 342-343: basis=None with explicit layer."""
        from kompot.plot.expression import plot_gene_expression
        adata = _make_de_adata(with_layers=True)
        lfc_col = [c for c in adata.var.columns if "mean_lfc" in c][0]
        mahal_col = [c for c in adata.var.columns if "mahalanobis" in c][0]
        adata.layers["raw"] = adata.X.copy()
        fig = plot_gene_expression(
            adata, "gene_0",
            lfc_key=lfc_col, score_key=mahal_col,
            basis=None,
            layer="raw",
            return_fig=True,
        )
        assert fig is not None

    def test_basis_none_missing_layer_fallback(self):
        """Lines 346-347: basis=None, layer doesn't exist -> fallback to X."""
        from kompot.plot.expression import plot_gene_expression
        adata = _make_de_adata(with_layers=True)
        lfc_col = [c for c in adata.var.columns if "mean_lfc" in c][0]
        mahal_col = [c for c in adata.var.columns if "mahalanobis" in c][0]
        fig = plot_gene_expression(
            adata, "gene_0",
            lfc_key=lfc_col, score_key=mahal_col,
            basis=None,
            layer="nonexistent_layer",
            return_fig=True,
        )
        assert fig is not None

    def test_missing_basis(self):
        """Lines 243, 266-267: basis not in obsm."""
        from kompot.plot.expression import plot_gene_expression
        adata = _make_de_adata()
        lfc_col = [c for c in adata.var.columns if "mean_lfc" in c][0]
        mahal_col = [c for c in adata.var.columns if "mahalanobis" in c][0]
        fig = plot_gene_expression(
            adata, "gene_0",
            lfc_key=lfc_col, score_key=mahal_col,
            basis="X_nonexistent",
            return_fig=True,
        )
        assert fig is not None

    def test_missing_layers_warning(self):
        """Lines 278-279, 293: missing imputed layers -> warning."""
        from kompot.plot.expression import plot_gene_expression
        adata = _make_de_adata(with_layers=False)
        lfc_col = [c for c in adata.var.columns if "mean_lfc" in c][0]
        mahal_col = [c for c in adata.var.columns if "mahalanobis" in c][0]
        fig = plot_gene_expression(
            adata, "gene_0",
            lfc_key=lfc_col, score_key=mahal_col,
            return_fig=True,
        )
        assert fig is not None

    def test_field_names_fallback(self):
        """Lines 280-282: field_names fallback when no imputed_layer_keys."""
        from kompot.plot.expression import plot_gene_expression
        adata = _make_de_adata(with_layers=True)
        lfc_col = [c for c in adata.var.columns if "mean_lfc" in c][0]
        mahal_col = [c for c in adata.var.columns if "mahalanobis" in c][0]
        # Remove imputed_layer_keys from run info to hit field_names fallback
        run_history = json.loads(adata.uns["kompot_de"]["run_history"])
        del run_history[0]["imputed_layer_keys"]
        adata.uns["kompot_de"]["run_history"] = json.dumps(run_history)
        fig = plot_gene_expression(
            adata, "gene_0",
            lfc_key=lfc_col, score_key=mahal_col,
            return_fig=True,
        )
        assert fig is not None

    def test_both_keys_provided(self):
        """Line 49-50: both lfc_key and score_key provided skips inference."""
        from kompot.plot.expression import _infer_expression_keys
        adata = _make_de_adata()
        result = _infer_expression_keys(adata, lfc_key="my_lfc", score_key="my_score")
        assert result == ("my_lfc", "my_score")

    def test_gene_not_found(self):
        """Line 158: gene not in var_names."""
        from kompot.plot.expression import plot_gene_expression
        adata = _make_de_adata()
        with pytest.raises(ValueError, match="not found"):
            plot_gene_expression(adata, "nonexistent_gene")


class TestExpressionEdgeCases:
    """Additional expression plot edge cases."""

    def test_swapped_conditions(self):
        """Lines 265-267: swapped condition order -> imputed_layer_keys swap."""
        from kompot.plot.expression import plot_gene_expression
        adata = _make_de_adata(condition1="Young", condition2="Old")
        lfc_col = [c for c in adata.var.columns if "mean_lfc" in c][0]
        mahal_col = [c for c in adata.var.columns if "mahalanobis" in c][0]
        # Pass conditions in reversed order
        fig = plot_gene_expression(
            adata, "gene_0",
            lfc_key=lfc_col, score_key=mahal_col,
            condition1="Old", condition2="Young",
            return_fig=True,
        )
        assert fig is not None

    def test_layer_missing_fallback(self):
        """Lines 346-347: layer not found -> fallback to adata.X (scanpy available, basis=None)."""
        from kompot.plot.expression import plot_gene_expression
        adata = _make_de_adata(with_layers=False)
        lfc_col = [c for c in adata.var.columns if "mean_lfc" in c][0]
        mahal_col = [c for c in adata.var.columns if "mahalanobis" in c][0]
        fig = plot_gene_expression(
            adata, "gene_0",
            lfc_key=lfc_col, score_key=mahal_col,
            basis=None,
            layer="nonexistent_layer",
            return_fig=True,
        )
        assert fig is not None
