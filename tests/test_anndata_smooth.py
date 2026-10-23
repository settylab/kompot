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


GP_SEEDED = GPSettings(n_landmarks=30, random_state=0)


class TestCellMask:
    """cell_mask: masked-out cells are excluded from fitting, never evaluated, and NaN in every output."""

    @staticmethod
    def _keep(adata):
        keep = np.ones(adata.n_obs, dtype=bool)
        keep[[0, 7, 41, 79]] = False
        return keep

    def test_masked_rows_nan_kept_rows_finite(self, adata):
        keep = self._keep(adata)
        kompot.smooth_expression(
            adata, gp=GP_SEEDED, cell_mask=keep, output=OutputSettings(progress=False)
        )
        for key in ("kompot_smooth_all_smoothed", "kompot_smooth_all_std"):
            layer = adata.layers[key]
            assert np.all(np.isnan(layer[~keep]))
            assert np.all(np.isfinite(layer[keep]))

    def test_masked_coordinates_are_never_read(self, adata):
        keep = self._keep(adata)
        adata.obsm["DM_EigenVectors"][~keep] = np.nan  # e.g. cells left out of the diffusion map
        kompot.smooth_expression(
            adata, gp=GP_SEEDED, cell_mask=keep, output=OutputSettings(progress=False)
        )
        layer = adata.layers["kompot_smooth_all_smoothed"]
        assert np.all(np.isnan(layer[~keep])) and np.all(np.isfinite(layer[keep]))

    def test_equals_smoothing_the_kept_subset(self, adata):
        keep = self._keep(adata)
        sub = adata[keep].copy()
        kompot.smooth_expression(sub, gp=GP_SEEDED, output=OutputSettings(progress=False))
        kompot.smooth_expression(
            adata, gp=GP_SEEDED, cell_mask=keep, output=OutputSettings(progress=False)
        )
        for key in ("kompot_smooth_all_smoothed", "kompot_smooth_all_std"):
            np.testing.assert_allclose(adata.layers[key][keep], sub.layers[key], rtol=1e-6, atol=1e-8)

    def test_obs_column_and_condition(self, adata):
        keep = self._keep(adata)
        adata.obs["in_state_space"] = keep
        kompot.smooth_expression(
            adata,
            groupby="condition",
            condition="A",
            cell_mask="in_state_space",
            gp=GP_SEEDED,
            output=OutputSettings(progress=False),
        )
        layer = adata.layers["kompot_smooth_A_smoothed"]
        in_a = (adata.obs["condition"] == "A").values
        assert np.all(np.isnan(layer[~keep]))
        # kept cells outside the condition are still evaluated, as without a mask
        assert np.all(np.isfinite(layer[keep & ~in_a]))
        info = adata.uns["kompot_smooth"]["last_run_info"]
        assert "in_state_space" in str(info)

    def test_variance_layers_masked(self, adata):
        keep = self._keep(adata)
        kompot.smooth_expression(
            adata,
            groupby="condition",
            condition="A",
            sample_col="sample",
            cell_mask=keep,
            gp=GPSettings(n_landmarks=30, random_state=0, use_empirical_variance=True),
            output=OutputSettings(progress=False),
        )
        for key in ("kompot_smooth_A_obs_variance", "kompot_smooth_A_sample_variance"):
            if key in adata.layers:
                assert np.all(np.isnan(adata.layers[key][~keep]))
        assert "kompot_smooth_A_obs_variance" in adata.layers

    def test_genes_subset_with_mask(self, adata):
        keep = self._keep(adata)
        kompot.smooth_expression(
            adata, genes=["gene_0", "gene_1"], gp=GP_SEEDED, cell_mask=keep,
            output=OutputSettings(progress=False),
        )
        layer = adata.layers["kompot_smooth_all_smoothed"]
        assert np.all(np.isnan(layer[~keep])) and np.all(np.isfinite(layer[keep][:, :2]))
        assert np.all(np.isnan(layer[:, 2:]))

    def test_no_mask_unchanged(self, adata):
        """cell_mask=None (default) must give the same result as before the option existed"""
        a2 = adata.copy()
        kompot.smooth_expression(adata, gp=GP_SEEDED, output=OutputSettings(progress=False))
        kompot.smooth_expression(a2, gp=GP_SEEDED, cell_mask=None, output=OutputSettings(progress=False))
        np.testing.assert_array_equal(adata.layers["kompot_smooth_all_smoothed"], a2.layers["kompot_smooth_all_smoothed"])
        assert np.all(np.isfinite(adata.layers["kompot_smooth_all_smoothed"]))

    @pytest.mark.parametrize(
        "bad, match",
        [
            ("no_such_column", "not found"),
            (np.ones(5, dtype=bool), "entries"),
            (np.zeros(80, dtype=bool), "keeps no cell"),
            (np.ones(80, dtype=int), "boolean"),
        ],
    )
    def test_invalid_masks_raise(self, adata, bad, match):
        with pytest.raises(ValueError, match=match):
            kompot.smooth_expression(adata, gp=GP_SEEDED, cell_mask=bad, output=OutputSettings(progress=False))

    def test_non_bool_obs_column_raises(self, adata):
        adata.obs["flag"] = ["yes"] * 80
        with pytest.raises(ValueError, match="boolean"):
            kompot.smooth_expression(adata, gp=GP_SEEDED, cell_mask="flag", output=OutputSettings(progress=False))

    def test_nan_coordinates_inside_mask_raise(self, adata):
        keep = self._keep(adata)
        adata.obsm["DM_EigenVectors"][3] = np.nan  # cell 3 is kept
        with pytest.raises(ValueError, match="non-finite"):
            kompot.smooth_expression(adata, gp=GP_SEEDED, cell_mask=keep, output=OutputSettings(progress=False))
