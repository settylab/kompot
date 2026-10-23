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
BAD = [0, 7, 41, 79]


def _invalidate(adata, rows=BAD):
    """set the coordinates of `rows` to NaN; return the boolean array of cells with finite coordinates"""
    adata.obsm["DM_EigenVectors"][rows] = np.nan
    ok = np.ones(adata.n_obs, dtype=bool)
    ok[rows] = False
    return ok


class TestNonFiniteCoordinates:
    """Cells whose obsm coordinates are not all finite are excluded from fitting and NaN in every output; prediction
    is per cell, so every other cell's values are unchanged. No mask argument is needed."""

    def test_nan_rows_nan_other_rows_finite(self, adata):
        ok = _invalidate(adata)
        kompot.smooth_expression(adata, gp=GP_SEEDED, output=OutputSettings(progress=False))
        for key in ("kompot_smooth_all_smoothed", "kompot_smooth_all_std"):
            layer = adata.layers[key]
            assert np.all(np.isnan(layer[~ok]))
            assert np.all(np.isfinite(layer[ok]))

    def test_other_rows_bitwise_equal_same_fit(self, adata):
        """reference: the SAME training cells (groupby on a finiteness column) with the invalid cells given arbitrary
        finite coordinates. Valid cells are bitwise equal; only the invalid rows differ (NaN instead of a value)."""
        ref = adata.copy()
        ok = _invalidate(adata)
        ref.obs["finite"] = ok
        ref.obsm["DM_EigenVectors"][~ok] = 5.0
        kompot.smooth_expression(ref, groupby="finite", condition=True, gp=GP_SEEDED,
                                 output=OutputSettings(progress=False))
        kompot.smooth_expression(adata, gp=GP_SEEDED, output=OutputSettings(progress=False))
        for a, b in (("all_smoothed", "True_smoothed"), ("all_std", "True_std")):
            m = adata.layers[f"kompot_smooth_{a}"]
            r = ref.layers[f"kompot_smooth_{b}"]
            np.testing.assert_array_equal(m[ok], r[ok])
            assert np.all(np.isnan(m[~ok])) and np.all(np.isfinite(r[~ok]))

    def test_equals_smoothing_the_valid_subset(self, adata):
        ok = _invalidate(adata)
        sub = adata[ok].copy()
        kompot.smooth_expression(sub, gp=GP_SEEDED, output=OutputSettings(progress=False))
        kompot.smooth_expression(adata, gp=GP_SEEDED, output=OutputSettings(progress=False))
        for key in ("kompot_smooth_all_smoothed", "kompot_smooth_all_std"):
            np.testing.assert_allclose(adata.layers[key][ok], sub.layers[key], rtol=1e-6, atol=1e-8)

    def test_partial_nan_and_inf_rows(self, adata):
        X = adata.obsm["DM_EigenVectors"]
        X[3, 2] = np.nan        # one coordinate
        X[11, 0] = np.inf
        X[12, :] = -np.inf
        kompot.smooth_expression(adata, gp=GP_SEEDED, output=OutputSettings(progress=False))
        layer = adata.layers["kompot_smooth_all_smoothed"]
        bad = [3, 11, 12]
        assert np.all(np.isnan(layer[bad]))
        assert np.all(np.isfinite(np.delete(layer, bad, axis=0)))

    def test_all_finite_control(self, adata):
        from kompot.anndata.utils.json_utils import get_json_metadata

        kompot.smooth_expression(adata, gp=GP_SEEDED, output=OutputSettings(progress=False))
        assert np.all(np.isfinite(adata.layers["kompot_smooth_all_smoothed"]))
        p = get_json_metadata(adata, "kompot_smooth.last_run_info")["params"]
        assert p["n_cells_nonfinite"] == 0 and p["nonfinite_cells"] is None

    def test_run_info_records_the_cells(self, adata):
        from kompot.anndata.utils.json_utils import get_json_metadata

        _invalidate(adata)
        kompot.smooth_expression(adata, gp=GP_SEEDED, output=OutputSettings(progress=False))
        info = get_json_metadata(adata, "kompot_smooth.last_run_info")
        p = info["params"]
        assert p["n_cells_nonfinite"] == len(BAD) and p["n_cells"] == adata.n_obs - len(BAD)
        assert "NaN" in p["nonfinite_cells"] and "DM_EigenVectors" in p["nonfinite_cells"]
        desc = info["field_mapping"]["kompot_smooth_all_smoothed"]["description"]
        assert f"NaN for the {len(BAD)} cells with non-finite coordinates in obsm['DM_EigenVectors']" in desc

    def test_with_condition(self, adata):
        ok = _invalidate(adata)
        kompot.smooth_expression(
            adata, groupby="condition", condition="A", gp=GP_SEEDED, output=OutputSettings(progress=False)
        )
        layer = adata.layers["kompot_smooth_A_smoothed"]
        in_a = (adata.obs["condition"] == "A").values
        assert np.all(np.isnan(layer[~ok]))
        assert np.all(np.isfinite(layer[ok & ~in_a]))   # valid cells outside the condition are still evaluated

    def test_variance_layers(self, adata):
        ok = _invalidate(adata)
        kompot.smooth_expression(
            adata, groupby="condition", condition="A", sample_col="sample",
            gp=GPSettings(n_landmarks=30, random_state=0, use_empirical_variance=True),
            output=OutputSettings(progress=False),
        )
        assert "kompot_smooth_A_obs_variance" in adata.layers
        for key in ("kompot_smooth_A_obs_variance", "kompot_smooth_A_sample_variance"):
            if key in adata.layers:
                assert np.all(np.isnan(adata.layers[key][~ok]))
                assert np.all(np.isfinite(adata.layers[key][ok]))

    def test_genes_subset(self, adata):
        ok = _invalidate(adata)
        kompot.smooth_expression(adata, genes=["gene_0", "gene_1"], gp=GP_SEEDED, output=OutputSettings(progress=False))
        layer = adata.layers["kompot_smooth_all_smoothed"]
        assert np.all(np.isnan(layer[~ok])) and np.all(np.isfinite(layer[ok][:, :2]))
        assert np.all(np.isnan(layer[:, 2:]))

    def test_no_valid_cell_raises(self, adata):
        adata.obsm["DM_EigenVectors"][:] = np.nan
        with pytest.raises(ValueError, match="finite coordinates"):
            kompot.smooth_expression(adata, gp=GP_SEEDED, output=OutputSettings(progress=False))
