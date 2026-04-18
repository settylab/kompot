"""Tests for variance-stratified residual Mahalanobis FDR."""

import numpy as np
import pandas as pd
import pytest

import anndata as ad

import kompot
from kompot import FDRSettings, GPSettings, OutputSettings
from kompot.residualization import (
    NullTrend,
    compute_gene_features,
    fit_null_trend,
    residualize_mahalanobis,
    residual_local_fdr,
)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


class TestComputeGeneFeatures:
    def test_shapes_and_nonneg(self):
        rng = np.random.default_rng(0)
        X1 = rng.exponential(1.0, (30, 10))
        X2 = rng.exponential(1.0, (20, 10))
        m, v = compute_gene_features([X1, X2])
        assert m.shape == (10,)
        assert v.shape == (10,)
        assert np.all(m >= 0)
        assert np.all(v >= 0)

    def test_zero_variance_gene_is_finite(self):
        # A gene with zero variance (all cells equal) must not blow up
        X1 = np.zeros((5, 3))
        X2 = np.zeros((5, 3))
        X1[:, 0] = 2.0
        X2[:, 0] = 2.0
        m, v = compute_gene_features([X1, X2])
        assert np.isfinite(m).all()
        assert np.isfinite(v).all()
        assert v[0] == 0.0  # zero variance -> log1p(0) = 0
        assert v[1] == 0.0
        assert v[2] == 0.0

    def test_stacking_matches_pooled(self):
        rng = np.random.default_rng(1)
        X1 = rng.normal(0, 1, (40, 5))
        X2 = rng.normal(0, 1, (60, 5))
        m, v = compute_gene_features([X1, X2])
        stacked = np.vstack([X1, X2])
        np.testing.assert_allclose(m, np.log1p(stacked.mean(axis=0)))
        np.testing.assert_allclose(v, np.log1p(stacked.var(axis=0)))

    def test_mismatched_gene_count_raises(self):
        with pytest.raises(ValueError, match="matching gene count"):
            compute_gene_features([np.zeros((5, 3)), np.zeros((5, 4))])


# ---------------------------------------------------------------------------
# Null trend fitting
# ---------------------------------------------------------------------------


class TestFitNullTrend:
    def test_known_trend_is_recovered(self):
        """Reconstruct a synthetic polynomial trend with high R^2."""
        rng = np.random.default_rng(42)
        n = 2000
        m = rng.uniform(0, 3, n)
        v = rng.uniform(0, 3, n)
        # Ground-truth surface: 2*m + 1.5*v + 0.3*m*v - 0.1*m**2
        phi_true = 2.0 * m + 1.5 * v + 0.3 * m * v - 0.1 * m * m
        # log1p of the Mahalanobis-like signal is phi + noise
        y = phi_true + rng.normal(0, 0.1, n)

        trend = fit_null_trend(y, m, v, model="poly3")

        # High R^2 because our model covers the ground-truth terms
        assert trend.fit_r2 > 0.95
        # The residual scale should be near the noise sd we injected
        assert abs(trend.sigma - 0.1) < 0.02
        # Predicting at the fit points must be near the ground truth
        phi_hat = trend.predict(m, v)
        np.testing.assert_allclose(phi_hat, phi_true, atol=0.05)

    def test_short_input_raises(self):
        with pytest.raises(ValueError, match="more null draws"):
            fit_null_trend(
                np.arange(3.0),
                np.arange(3.0),
                np.arange(3.0),
                model="poly3",
            )

    def test_mean_only_model_ignores_variance(self):
        """Residuals should not change when variance is shuffled."""
        rng = np.random.default_rng(0)
        n = 1000
        m = rng.uniform(0, 3, n)
        v = rng.uniform(0, 3, n)
        y = 1.5 * m + 0.2 * m * m + rng.normal(0, 0.15, n)

        trend_full = fit_null_trend(y, m, v, model="poly3_mean_only")
        trend_shuffled = fit_null_trend(
            y, m, rng.permutation(v), model="poly3_mean_only"
        )

        # Mean-only model must give identical coefficients regardless of v
        np.testing.assert_allclose(trend_full.coef, trend_shuffled.coef)
        assert trend_full.features == ("log_mean",)

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown null_trend_model"):
            fit_null_trend(np.zeros(10), np.zeros(10), np.zeros(10), model="xyz")

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="matching length"):
            fit_null_trend(np.zeros(10), np.zeros(9), np.zeros(10))


# ---------------------------------------------------------------------------
# Residualization end-to-end
# ---------------------------------------------------------------------------


class TestResidualizeMahalanobis:
    def test_removes_known_trend(self):
        """
        If we build a Mahalanobis-like statistic that is exactly a smooth
        function of (m, v) plus noise, the residual should be dominated by
        that noise, not by m or v.
        """
        rng = np.random.default_rng(7)
        n_genes = 500
        m = rng.uniform(0.1, 3.0, n_genes)
        v = rng.uniform(0.1, 3.0, n_genes)

        # True null surface
        phi_true = 2.0 * m + 1.5 * v + 0.3 * m * v
        # Null draws: sample null_idx uniformly over genes
        n_null = 3000
        null_idx = rng.integers(0, n_genes, n_null)
        noise = rng.normal(0, 0.2, n_null)
        null_log = phi_true[null_idx] + noise
        null_mahal = np.expm1(null_log)

        # Real: 490 variance-matched null genes, 10 true DE genes
        real_log = phi_true + rng.normal(0, 0.2, n_genes)
        de_idx = np.array([13, 71, 142, 205, 298, 337, 400, 423, 470, 491])
        real_log[de_idx] += 3.0
        real_mahal = np.expm1(real_log)

        res = residualize_mahalanobis(
            real_mahalanobis=real_mahal,
            null_mahalanobis=null_mahal,
            real_log_mean=m,
            real_log_var=v,
            null_gene_indices=null_idx,
            model="poly3",
        )

        # Residual should correlate strongly with the injected spike, not with m/v
        spike = np.zeros(n_genes)
        spike[de_idx] = 1.0
        corr_spike = np.corrcoef(res.residual, spike)[0, 1]
        corr_mean = np.corrcoef(res.residual, m)[0, 1]
        corr_var = np.corrcoef(res.residual, v)[0, 1]
        assert corr_spike > 0.5
        assert abs(corr_mean) < 0.2
        assert abs(corr_var) < 0.2

        # Top-10 by residual must include all 10 DE genes
        top10 = set(np.argsort(res.residual)[::-1][:10].tolist())
        assert set(de_idx.tolist()).issubset(top10)

        # Top-10 by RAW should be dominated by high-(m,v) genes, not DE
        top10_raw = set(np.argsort(real_mahal)[::-1][:10].tolist())
        assert len(top10_raw.intersection(de_idx)) < 10

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="per-gene feature length"):
            residualize_mahalanobis(
                real_mahalanobis=np.ones(5),
                null_mahalanobis=np.ones(10),
                real_log_mean=np.ones(6),
                real_log_var=np.ones(6),
                null_gene_indices=np.zeros(10, int),
            )

    def test_bad_null_index_raises(self):
        with pytest.raises(ValueError, match="out-of-range"):
            residualize_mahalanobis(
                real_mahalanobis=np.ones(3),
                null_mahalanobis=np.ones(100),
                real_log_mean=np.ones(3),
                real_log_var=np.ones(3),
                null_gene_indices=np.array([0, 1, 2, 99] + [0] * 96),
            )

    def test_residual_local_fdr_negative_z_maps_to_high_lfdr(self):
        """A below-null-mean gene gets local_fdr close to 1."""
        rng = np.random.default_rng(99)
        n_genes = 200
        m = rng.uniform(0, 2, n_genes)
        v = rng.uniform(0, 2, n_genes)
        null_idx = rng.integers(0, n_genes, 1500)
        null_log = 2.0 * m[null_idx] + v[null_idx] + rng.normal(0, 0.2, 1500)
        null_mahal = np.expm1(null_log)
        real_log = 2.0 * m + v + rng.normal(0, 0.2, n_genes)
        # Force one gene well below the null mean
        real_log[5] = -1.0
        real_mahal = np.expm1(np.clip(real_log, 0.0, None))

        res = residualize_mahalanobis(
            real_mahal, null_mahal, m, v, null_idx, model="poly3"
        )
        _pvalues, lfdr, _tail_fdr, is_de = residual_local_fdr(res, 0.05)
        assert lfdr[5] >= 0.5
        assert not is_de[5]


# ---------------------------------------------------------------------------
# End-to-end integration through kompot.de
# ---------------------------------------------------------------------------


def _make_variance_stratified_adata(seed: int = 0) -> "ad.AnnData":
    """Two conditions, 80 genes: 60 low-variance, 20 high-variance
    "background" genes.  True DE is injected on 10 low-variance genes so
    the raw Mahalanobis is dominated by the high-variance background and
    residualisation is needed to see the signal.
    """
    rng = np.random.default_rng(seed)
    n_cells = 200
    n_genes = 80
    de_idx = np.arange(20, 30)  # low-variance genes
    var_scale = np.concatenate([np.full(20, 3.0), np.full(60, 0.5)])

    X_A = rng.normal(0.0, var_scale, (n_cells // 2, n_genes))
    X_B = rng.normal(0.0, var_scale, (n_cells // 2, n_genes))
    X_B[:, de_idx] += 2.5  # injected DE on low-variance genes
    X = np.vstack([X_A, X_B])

    states = rng.normal(0.0, 1.0, (n_cells, 5))
    adata = ad.AnnData(X)
    adata.obsm["DM_EigenVectors"] = states
    adata.obs["condition"] = pd.Categorical(
        ["A"] * (n_cells // 2) + ["B"] * (n_cells // 2)
    )
    adata.var_names = [f"g{i}" for i in range(n_genes)]
    return adata, de_idx


class TestDeIntegration:
    def test_residual_mode_writes_residual_columns(self):
        adata, de_idx = _make_variance_stratified_adata()
        kompot.de(
            adata,
            "condition",
            "A",
            "B",
            gp=GPSettings(n_landmarks=30, random_state=0),
            fdr=FDRSettings(
                null_genes=500, null_seed=0, mode="variance_stratified"
            ),
            output=OutputSettings(progress=False, allow_single_condition_variance=True),
        )
        cols = adata.var.columns
        # raw columns still there
        assert "kompot_de_A_to_B_mahalanobis" in cols
        assert "kompot_de_A_to_B_mahalanobis_local_fdr" in cols
        assert "kompot_de_A_to_B_is_de" in cols
        # residual columns added
        assert "kompot_de_A_to_B_residual_mahalanobis" in cols
        assert "kompot_de_A_to_B_residual_z" in cols
        assert "kompot_de_A_to_B_residual_local_fdr" in cols
        assert "kompot_de_A_to_B_residual_is_de" in cols

        # Residual mode should recover more of the injected DE genes than raw.
        is_de_raw = adata.var["kompot_de_A_to_B_is_de"].values.astype(bool)
        is_de_res = adata.var["kompot_de_A_to_B_residual_is_de"].values.astype(bool)
        de_recall_raw = is_de_raw[de_idx].sum()
        de_recall_res = is_de_res[de_idx].sum()
        assert de_recall_res >= de_recall_raw
        # Require at least 7/10 DE genes called by the residual mode
        assert de_recall_res >= 7

    def test_raw_mode_is_default_and_writes_no_residual(self):
        adata, _ = _make_variance_stratified_adata()
        kompot.de(
            adata,
            "condition",
            "A",
            "B",
            gp=GPSettings(n_landmarks=30, random_state=0),
            fdr=FDRSettings(null_genes=500, null_seed=0),  # default mode="raw"
            output=OutputSettings(progress=False, allow_single_condition_variance=True),
        )
        # residual columns must NOT be present when mode='raw'
        for col in adata.var.columns:
            assert "residual" not in col, f"unexpected residual column: {col}"


# ---------------------------------------------------------------------------
# Settings validation
# ---------------------------------------------------------------------------


class TestFDRSettingsModes:
    def test_default_mode_is_raw(self):
        s = FDRSettings()
        assert s.mode == "raw"

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="'mode' must be one of"):
            FDRSettings(mode="bogus")

    def test_invalid_trend_features_raises(self):
        with pytest.raises(ValueError, match="null_trend_features"):
            FDRSettings(null_trend_features=("foo",))

    def test_invalid_trend_model_raises(self):
        with pytest.raises(ValueError, match="null_trend_model"):
            FDRSettings(null_trend_model="spline")
