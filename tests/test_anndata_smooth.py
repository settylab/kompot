"""Tests for kompot.smooth_expression AnnData interface."""

import numpy as np
import pandas as pd
import pytest
import anndata
import kompot
from kompot import GPSettings, StorageSettings, OutputSettings


@pytest.fixture
def adata():
    rng = np.random.RandomState(42)
    n_cells, n_features, n_genes = 80, 5, 8
    X_obsm = rng.randn(n_cells, n_features)
    X_expr = rng.randn(n_cells, n_genes)
    obs = pd.DataFrame(
        {
            "condition": ["A"] * 40 + ["B"] * 40,
            "sample": ["s1"] * 20 + ["s2"] * 20 + ["s3"] * 20 + ["s4"] * 20,
        }
    )
    ad = anndata.AnnData(X=X_expr, obs=obs)
    ad.obsm["DM_EigenVectors"] = X_obsm
    ad.var_names = [f"gene_{i}" for i in range(n_genes)]
    return ad


GP_FAST = GPSettings(n_landmarks=30)


class TestSmoothExpression:
    def test_basic_smooth(self, adata):
        kompot.smooth_expression(
            adata,
            gp=GP_FAST,
            output=OutputSettings(progress=False),
        )
        assert "kompot_smooth_all_smoothed" in adata.layers
        assert "kompot_smooth_all_std" in adata.layers
        assert not np.all(np.isnan(adata.layers["kompot_smooth_all_smoothed"]))
        assert not np.all(np.isnan(adata.layers["kompot_smooth_all_std"]))

    def test_smooth_with_condition(self, adata):
        kompot.smooth_expression(
            adata,
            groupby="condition",
            condition="A",
            gp=GP_FAST,
            output=OutputSettings(progress=False),
        )
        imputed = adata.layers["kompot_smooth_A_smoothed"]
        mask_a = (adata.obs["condition"] == "A").values
        # Condition A cells should have values
        assert not np.all(np.isnan(imputed[mask_a]))
        # All cells get smoothed values (GP extrapolates to non-condition cells)
        assert not np.all(np.isnan(imputed[~mask_a]))

    def test_smooth_with_empirical_variance(self, adata):
        kompot.smooth_expression(
            adata,
            gp=GPSettings(n_landmarks=30, use_empirical_variance=True),
            output=OutputSettings(progress=False),
        )
        assert "kompot_smooth_all_obs_variance" in adata.layers
        assert not np.all(np.isnan(adata.layers["kompot_smooth_all_obs_variance"]))

    def test_smooth_with_sample_col(self, adata):
        kompot.smooth_expression(
            adata,
            groupby="condition",
            condition="A",
            sample_col="sample",
            gp=GP_FAST,
            output=OutputSettings(progress=False),
        )
        assert "kompot_smooth_A_sample_variance" in adata.layers

    def test_return_full_results(self, adata):
        result = kompot.smooth_expression(
            adata,
            gp=GP_FAST,
            output=OutputSettings(progress=False, return_full_results=True),
        )
        assert isinstance(result, dict)
        assert "model" in result
        assert "table" in result
        assert "field_names" in result

    def test_overwrite_false_raises(self, adata):
        kompot.smooth_expression(
            adata,
            gp=GP_FAST,
            output=OutputSettings(progress=False),
        )
        with pytest.raises(ValueError, match="already exist"):
            kompot.smooth_expression(
                adata,
                gp=GP_FAST,
                output=OutputSettings(progress=False),
                storage=StorageSettings(overwrite=False),
            )

    def test_overwrite_true_succeeds(self, adata):
        kompot.smooth_expression(
            adata,
            gp=GP_FAST,
            output=OutputSettings(progress=False),
        )
        kompot.smooth_expression(
            adata,
            gp=GP_FAST,
            output=OutputSettings(progress=False),
            storage=StorageSettings(overwrite=True),
        )
        assert "kompot_smooth_all_smoothed" in adata.layers

    def test_genes_subset(self, adata):
        genes = ["gene_0", "gene_1"]
        kompot.smooth_expression(
            adata,
            genes=genes,
            gp=GP_FAST,
            output=OutputSettings(progress=False),
        )
        imputed = adata.layers["kompot_smooth_all_smoothed"]
        # Selected genes should have values
        gene_idx = [list(adata.var_names).index(g) for g in genes]
        other_idx = [i for i in range(adata.n_vars) if i not in gene_idx]
        assert not np.all(np.isnan(imputed[:, gene_idx]))
        # Other genes should be NaN
        assert np.all(np.isnan(imputed[:, other_idx]))

    def test_missing_obsm_key_raises(self, adata):
        with pytest.raises(ValueError, match="not found in adata.obsm"):
            kompot.smooth_expression(adata, obsm_key="nonexistent")

    def test_missing_groupby_raises(self, adata):
        with pytest.raises(ValueError, match="groupby.*required"):
            kompot.smooth_expression(adata, condition="A")

    def test_uns_metadata_stored(self, adata):
        kompot.smooth_expression(
            adata,
            gp=GP_FAST,
            output=OutputSettings(progress=False),
        )
        assert "kompot_smooth" in adata.uns
        uns = adata.uns["kompot_smooth"]
        assert "last_run_info" in uns

    def test_copy_returns_adata(self, adata):
        result = kompot.smooth_expression(
            adata,
            gp=GP_FAST,
            output=OutputSettings(progress=False, copy=True),
        )
        assert isinstance(result, anndata.AnnData)
        assert "kompot_smooth_all_smoothed" in result.layers
        # Original should not be modified
        assert "kompot_smooth_all_smoothed" not in adata.layers

    def test_deprecated_compute_smoothed_expression(self, adata):
        with pytest.warns(DeprecationWarning, match="deprecated"):
            kompot.compute_smoothed_expression(
                adata,
                n_landmarks=30,
                progress=False,
            )
        assert "kompot_smooth_all_smoothed" in adata.layers
