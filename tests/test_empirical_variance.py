"""Tests for the empirical variance feature (use_empirical_variance).

Covers:
1. Factor trick in compute_mahalanobis_distances (diagonal_variance parameter)
2. DifferentialExpression with use_empirical_variance=True
3. Combination of sample variance + empirical variance
4. AnnData wrapper parameter threading
"""

import numpy as np
import pytest
import logging

from kompot.utils import compute_mahalanobis_distances
from kompot.differential import DifferentialExpression

logger = logging.getLogger(__name__)


# ===== Factor trick (utils.py) =====


class TestDiagonalVarianceFactorTrick:
    """Tests for the diagonal_variance parameter in compute_mahalanobis_distances."""

    def _make_cov(self, n, seed=42):
        rng = np.random.RandomState(seed)
        A = rng.randn(n, n)
        return A @ A.T + np.eye(n) * 0.1

    def test_zero_diagonal_matches_standard(self):
        """Zero diagonal variance should produce identical distances."""
        rng = np.random.RandomState(0)
        n_genes, m = 15, 8
        cov = self._make_cov(m, seed=0)
        diffs = rng.randn(n_genes, m)

        d_std = compute_mahalanobis_distances(diffs, cov, progress=False)
        d_zero = compute_mahalanobis_distances(
            diffs, cov, progress=False,
            diagonal_variance=np.zeros((n_genes, m)),
        )
        np.testing.assert_allclose(d_std, d_zero, atol=1e-5)

    def test_positive_variance_deflates_distances(self):
        """Adding positive diagonal variance should reduce Mahalanobis distances."""
        rng = np.random.RandomState(1)
        n_genes, m = 20, 10
        cov = self._make_cov(m, seed=1)
        diffs = rng.randn(n_genes, m)

        d_std = compute_mahalanobis_distances(diffs, cov, progress=False)
        var = rng.exponential(1.0, (n_genes, m))
        d_var = compute_mahalanobis_distances(
            diffs, cov, progress=False, diagonal_variance=var,
        )
        assert np.mean(d_var) < np.mean(d_std), (
            "Mean distance with positive diagonal variance should be smaller"
        )

    def test_large_variance_strongly_deflates(self):
        """Very large diagonal variance should make distances approach zero."""
        rng = np.random.RandomState(2)
        n_genes, m = 10, 6
        cov = self._make_cov(m, seed=2)
        diffs = rng.randn(n_genes, m)

        huge_var = np.full((n_genes, m), 1e6)
        d_huge = compute_mahalanobis_distances(
            diffs, cov, progress=False, diagonal_variance=huge_var,
        )
        # With extremely large variance the effective covariance overwhelms
        # the signal, distances should be very small
        assert np.all(d_huge < 0.1), (
            f"Distances with huge variance should be near zero, got max {np.max(d_huge):.4f}"
        )

    def test_output_shape(self):
        """Output should be (n_genes,) regardless of batch boundaries."""
        rng = np.random.RandomState(3)
        for n_genes in [1, 7, 50]:
            m = 5
            cov = self._make_cov(m, seed=3)
            diffs = rng.randn(n_genes, m)
            var = rng.exponential(0.5, (n_genes, m))
            d = compute_mahalanobis_distances(
                diffs, cov, batch_size=13, progress=False,
                diagonal_variance=var,
            )
            assert d.shape == (n_genes,), f"Expected ({n_genes},), got {d.shape}"

    def test_no_nans_for_valid_input(self):
        """Valid inputs should not produce NaN distances."""
        rng = np.random.RandomState(4)
        n_genes, m = 30, 8
        cov = self._make_cov(m, seed=4)
        diffs = rng.randn(n_genes, m)
        var = rng.exponential(0.5, (n_genes, m))

        d = compute_mahalanobis_distances(
            diffs, cov, progress=False, diagonal_variance=var,
        )
        assert not np.any(np.isnan(d)), "No NaN values expected for valid input"

    def test_per_gene_variance_ordering(self):
        """Genes with higher variance should have smaller distances (same diff)."""
        rng = np.random.RandomState(5)
        m = 8
        cov = self._make_cov(m, seed=5)
        # Same difference vector for all genes
        shared_diff = rng.randn(m)
        n_genes = 10
        diffs = np.tile(shared_diff, (n_genes, 1))

        # Increasing variance per gene
        var = np.zeros((n_genes, m))
        for g in range(n_genes):
            var[g] = (g + 1) * 0.5

        d = compute_mahalanobis_distances(
            diffs, cov, progress=False, diagonal_variance=var,
        )
        # Distances should be monotonically decreasing
        for i in range(n_genes - 1):
            assert d[i] >= d[i + 1] - 1e-6, (
                f"Gene {i} (var={var[i,0]:.1f}) should have >= distance than "
                f"gene {i+1} (var={var[i+1,0]:.1f}): {d[i]:.4f} vs {d[i+1]:.4f}"
            )


# ===== DifferentialExpression class =====


class TestDifferentialExpressionEmpiricalVariance:
    """Tests for DifferentialExpression with use_empirical_variance=True."""

    @pytest.fixture
    def synth_data(self):
        """Synthetic dataset with known heteroscedastic genes."""
        rng = np.random.RandomState(42)
        n_cells = 60
        n_genes = 8
        n_features = 3

        X1 = rng.randn(n_cells, n_features)
        X2 = rng.randn(n_cells, n_features) + 0.3

        # First few genes: low noise.  Last few: high noise.
        y1 = rng.randn(n_cells, n_genes) * 0.1
        y2 = rng.randn(n_cells, n_genes) * 0.1
        # Add a mean shift to make fold changes non-zero
        y2[:, :4] += 1.0
        # High-noise genes: add large noise to both conditions
        y1[:, 4:] += rng.randn(n_cells, 4) * 5.0
        y2[:, 4:] += rng.randn(n_cells, 4) * 5.0

        return X1, y1, X2, y2

    def test_fit_stores_predictors(self, synth_data):
        """Fitting with use_empirical_variance should store variance predictors."""
        X1, y1, X2, y2 = synth_data
        de = DifferentialExpression(
            use_empirical_variance=True,
            n_landmarks=20,
            batch_size=0,
        )
        de.fit(X1, y1, X2, y2, ls_factor=10.0)

        assert de.empirical_variance_predictor1 is not None
        assert de.empirical_variance_predictor2 is not None

    def test_fit_without_flag_no_predictors(self, synth_data):
        """Without the flag, empirical variance predictors should stay None."""
        X1, y1, X2, y2 = synth_data
        de = DifferentialExpression(
            use_empirical_variance=False,
            n_landmarks=20,
            batch_size=0,
        )
        de.fit(X1, y1, X2, y2, ls_factor=10.0)

        assert de.empirical_variance_predictor1 is None
        assert de.empirical_variance_predictor2 is None

    def test_mahalanobis_distances_differ(self, synth_data):
        """Mahalanobis distances should differ with and without empirical variance."""
        X1, y1, X2, y2 = synth_data

        de_off = DifferentialExpression(
            use_empirical_variance=False, n_landmarks=20, batch_size=0,
        )
        de_off.fit(X1, y1, X2, y2, ls_factor=10.0)
        res_off = de_off.predict(X1, compute_mahalanobis=True, progress=False)

        de_on = DifferentialExpression(
            use_empirical_variance=True, n_landmarks=20, batch_size=0,
        )
        de_on.fit(X1, y1, X2, y2, ls_factor=10.0)
        res_on = de_on.predict(X1, compute_mahalanobis=True, progress=False)

        d_off = res_off["mahalanobis_distances"]
        d_on = res_on["mahalanobis_distances"]

        assert d_off.shape == d_on.shape
        assert not np.allclose(d_off, d_on, atol=1e-3), (
            "Distances should differ when empirical variance is enabled"
        )

    def test_high_noise_genes_deflated(self, synth_data):
        """High-noise genes (indices 4-7) should have lower Mahalanobis distance
        relative to without empirical variance."""
        X1, y1, X2, y2 = synth_data

        de_off = DifferentialExpression(
            use_empirical_variance=False, n_landmarks=20, batch_size=0,
        )
        de_off.fit(X1, y1, X2, y2, ls_factor=10.0)
        res_off = de_off.predict(X1, compute_mahalanobis=True, progress=False)

        de_on = DifferentialExpression(
            use_empirical_variance=True, n_landmarks=20, batch_size=0,
        )
        de_on.fit(X1, y1, X2, y2, ls_factor=10.0)
        res_on = de_on.predict(X1, compute_mahalanobis=True, progress=False)

        d_off = res_off["mahalanobis_distances"]
        d_on = res_on["mahalanobis_distances"]

        # High-noise genes should see the largest deflation
        ratio_low_noise = np.mean(d_on[:4]) / (np.mean(d_off[:4]) + 1e-10)
        ratio_high_noise = np.mean(d_on[4:]) / (np.mean(d_off[4:]) + 1e-10)

        assert ratio_high_noise < ratio_low_noise, (
            f"High-noise genes should be deflated more: "
            f"high-noise ratio={ratio_high_noise:.3f}, low-noise ratio={ratio_low_noise:.3f}"
        )

    def test_predict_std_incorporates_empirical_variance(self, synth_data):
        """Posterior std should be larger when empirical variance is enabled."""
        X1, y1, X2, y2 = synth_data

        de_off = DifferentialExpression(
            use_empirical_variance=False, n_landmarks=20, batch_size=0,
        )
        de_off.fit(X1, y1, X2, y2, ls_factor=10.0)
        res_off = de_off.predict(X1, progress=False)

        de_on = DifferentialExpression(
            use_empirical_variance=True, n_landmarks=20, batch_size=0,
        )
        de_on.fit(X1, y1, X2, y2, ls_factor=10.0)
        res_on = de_on.predict(X1, progress=False)

        # Std should be at least as large with empirical variance
        mean_std_off = np.mean(res_off["condition1_std"])
        mean_std_on = np.mean(res_on["condition1_std"])
        assert mean_std_on >= mean_std_off, (
            f"Std with empirical variance ({mean_std_on:.4f}) should be >= "
            f"without ({mean_std_off:.4f})"
        )

    def test_predict_zscores_reflect_variance(self, synth_data):
        """Z-scores should be smaller (less significant) with empirical variance for noisy genes."""
        X1, y1, X2, y2 = synth_data

        de_off = DifferentialExpression(
            use_empirical_variance=False, n_landmarks=20, batch_size=0,
        )
        de_off.fit(X1, y1, X2, y2, ls_factor=10.0)
        res_off = de_off.predict(X1, progress=False)

        de_on = DifferentialExpression(
            use_empirical_variance=True, n_landmarks=20, batch_size=0,
        )
        de_on.fit(X1, y1, X2, y2, ls_factor=10.0)
        res_on = de_on.predict(X1, progress=False)

        # Mean absolute z-score for high-noise genes should decrease
        z_off_high = np.mean(np.abs(res_off["fold_change_zscores"][:, 4:]))
        z_on_high = np.mean(np.abs(res_on["fold_change_zscores"][:, 4:]))
        assert z_on_high < z_off_high, (
            f"Z-scores for high-noise genes should decrease: "
            f"off={z_off_high:.3f}, on={z_on_high:.3f}"
        )

    def test_output_shapes(self, synth_data):
        """All output arrays should have correct shapes."""
        X1, y1, X2, y2 = synth_data
        n_cells, n_genes = X1.shape[0], y1.shape[1]

        de = DifferentialExpression(
            use_empirical_variance=True, n_landmarks=20, batch_size=0,
        )
        de.fit(X1, y1, X2, y2, ls_factor=10.0)
        res = de.predict(X1, compute_mahalanobis=True, progress=False)

        assert res["condition1_imputed"].shape == (n_cells, n_genes)
        assert res["condition2_imputed"].shape == (n_cells, n_genes)
        assert res["condition1_std"].shape == (n_cells, n_genes)
        assert res["condition2_std"].shape == (n_cells, n_genes)
        assert res["fold_change"].shape == (n_cells, n_genes)
        assert res["fold_change_zscores"].shape == (n_cells, n_genes)
        assert res["mean_log_fold_change"].shape == (n_genes,)
        assert res["mahalanobis_distances"].shape == (n_genes,)


# ===== Combination with sample variance =====


class TestEmpiricalPlusSampleVariance:
    """Tests for the combination of sample variance and empirical variance."""

    @pytest.fixture
    def synth_data_with_samples(self):
        """Synthetic dataset with sample structure for sample variance."""
        rng = np.random.RandomState(99)
        n_per_sample = 25
        n_samples = 3
        n_cells = n_per_sample * n_samples
        n_genes = 6
        n_features = 3

        X1 = rng.randn(n_cells, n_features)
        X2 = rng.randn(n_cells, n_features) + 0.2

        y1 = rng.randn(n_cells, n_genes) * 0.5
        y2 = rng.randn(n_cells, n_genes) * 0.5 + 0.5

        # Add high noise to last 2 genes
        y1[:, -2:] += rng.randn(n_cells, 2) * 3.0
        y2[:, -2:] += rng.randn(n_cells, 2) * 3.0

        # Sample indices
        idx1 = np.repeat(np.arange(n_samples), n_per_sample)
        idx2 = np.repeat(np.arange(n_samples), n_per_sample)

        return X1, y1, X2, y2, idx1, idx2

    def test_both_enabled_runs(self, synth_data_with_samples):
        """Using both sample variance and empirical variance should not error."""
        X1, y1, X2, y2, idx1, idx2 = synth_data_with_samples

        de = DifferentialExpression(
            use_sample_variance=True,
            use_empirical_variance=True,
            n_landmarks=15,
            batch_size=0,
        )
        de.fit(
            X1, y1, X2, y2,
            ls_factor=10.0,
            condition1_sample_indices=idx1,
            condition2_sample_indices=idx2,
        )
        res = de.predict(X1, compute_mahalanobis=True, progress=False)

        assert "mahalanobis_distances" in res
        assert res["mahalanobis_distances"].shape == (y1.shape[1],)
        assert not np.any(np.isnan(res["mahalanobis_distances"]))

    def test_combined_deflates_more_than_sample_alone(self, synth_data_with_samples):
        """Combining both variance sources should deflate distances more than
        sample variance alone for high-noise genes."""
        X1, y1, X2, y2, idx1, idx2 = synth_data_with_samples

        # Sample variance only
        de_sv = DifferentialExpression(
            use_sample_variance=True,
            use_empirical_variance=False,
            n_landmarks=15,
            batch_size=0,
        )
        de_sv.fit(
            X1, y1, X2, y2, ls_factor=10.0,
            condition1_sample_indices=idx1,
            condition2_sample_indices=idx2,
        )
        res_sv = de_sv.predict(X1, compute_mahalanobis=True, progress=False)

        # Both
        de_both = DifferentialExpression(
            use_sample_variance=True,
            use_empirical_variance=True,
            n_landmarks=15,
            batch_size=0,
        )
        de_both.fit(
            X1, y1, X2, y2, ls_factor=10.0,
            condition1_sample_indices=idx1,
            condition2_sample_indices=idx2,
        )
        res_both = de_both.predict(X1, compute_mahalanobis=True, progress=False)

        d_sv = res_sv["mahalanobis_distances"]
        d_both = res_both["mahalanobis_distances"]

        # High-noise genes (last 2) should see extra deflation
        assert np.mean(d_both[-2:]) <= np.mean(d_sv[-2:]), (
            f"Combined should deflate high-noise genes more: "
            f"combined={np.mean(d_both[-2:]):.3f}, sv_only={np.mean(d_sv[-2:]):.3f}"
        )

    def test_per_sample_empirical_variance_uses_per_sample_gps(self):
        """With sample_indices, empirical variance should be computed per
        sample (averaged) rather than across all pooled cells, to avoid
        double-counting between-sample variance."""
        rng = np.random.RandomState(77)
        n_per_sample = 30
        n_samples = 3
        n_cells = n_per_sample * n_samples
        n_genes = 4
        n_features = 3

        X = rng.randn(n_cells, n_features)
        # Within-sample noise is small
        y_base = np.sin(X[:, :1]) * np.ones((1, n_genes))
        y = y_base + rng.randn(n_cells, n_genes) * 0.3

        # Add large between-sample shifts (this is sample variance, not
        # aleatoric noise)
        sample_indices = np.repeat(np.arange(n_samples), n_per_sample)
        for s in range(n_samples):
            mask = sample_indices == s
            y[mask] += rng.randn(n_genes) * 5.0  # large sample shift

        from kompot.differential import ExpressionModel

        # Per-sample empirical variance (correct: captures within-sample noise)
        model_per = ExpressionModel(
            use_empirical_variance=True, n_landmarks=15, batch_size=0,
        )
        model_per.fit(X, y, ls_factor=10.0, sample_indices=sample_indices)
        assert model_per._within_sample_obs_var_predictor is not None
        var_per = model_per.obs_variance(X[:5], progress=False)

        # Pooled empirical variance (wrong: captures between-sample variance too)
        model_pooled = ExpressionModel(
            use_empirical_variance=True, n_landmarks=15, batch_size=0,
        )
        model_pooled.fit(X, y, ls_factor=10.0, sample_indices=None)
        assert model_pooled._within_sample_obs_var_predictor is None
        var_pooled = model_pooled.obs_variance(X[:5], progress=False)

        # Pooled variance should be substantially larger because it includes
        # between-sample variance that the per-sample approach correctly excludes
        assert np.mean(var_pooled) > np.mean(var_per) * 1.5, (
            f"Pooled variance ({np.mean(var_pooled):.3f}) should be much larger "
            f"than per-sample variance ({np.mean(var_per):.3f}) when between-sample "
            f"shifts are large"
        )

    def test_per_sample_predictors_not_created_without_samples(self):
        """Without sample_indices, should use the standard pooled obs_variance."""
        rng = np.random.RandomState(88)
        n, g, d = 50, 4, 3
        X = rng.randn(n, d)
        y = rng.randn(n, g)

        from kompot.differential import ExpressionModel

        model = ExpressionModel(
            use_empirical_variance=True, n_landmarks=15, batch_size=0,
        )
        model.fit(X, y, ls_factor=10.0)
        assert model._within_sample_obs_var_predictor is None
        assert model.has_empirical_variance


# ===== AnnData wrapper =====


class TestAnnDataEmpiricalVariance:
    """Tests for use_empirical_variance parameter threading through the anndata wrapper."""

    def test_parameter_accepted(self, tiny_adata, fast_de_params):
        """compute_differential_expression should accept use_empirical_variance."""
        from kompot.anndata.differential_expression import compute_differential_expression

        # Should not raise
        compute_differential_expression(
            tiny_adata,
            groupby="condition",
            condition1="A",
            condition2="B",
            obsm_key="X_pca",
            use_empirical_variance=True,
            **fast_de_params,
        )

    def test_parameter_stored_in_run_info(self, tiny_adata, fast_de_params):
        """use_empirical_variance should be recorded in run_info params."""
        import json
        from kompot.anndata.differential_expression import compute_differential_expression

        compute_differential_expression(
            tiny_adata,
            groupby="condition",
            condition1="A",
            condition2="B",
            obsm_key="X_pca",
            use_empirical_variance=True,
            **fast_de_params,
        )

        # Check run_info in uns (stored as JSON string via set_json_metadata)
        raw = tiny_adata.uns.get("kompot_de", {}).get("last_run_info", "{}")
        run_info = json.loads(raw) if isinstance(raw, str) else raw
        params = run_info.get("params", {})
        assert params.get("use_empirical_variance") is True

    def test_results_with_empirical_variance(self, small_adata, fast_de_params):
        """Full run with empirical variance should produce valid results."""
        from kompot.anndata.differential_expression import compute_differential_expression

        result = compute_differential_expression(
            small_adata,
            groupby="condition",
            condition1="A",
            condition2="B",
            obsm_key="X_pca",
            use_empirical_variance=True,
            return_full_results=True,
            **fast_de_params,
        )

        assert result is not None
        assert "table" in result
        assert "model" in result
        model = result["model"]
        assert model.use_empirical_variance is True
        assert model.empirical_variance_predictor1 is not None
        assert model.empirical_variance_predictor2 is not None

    def test_default_is_on(self, tiny_adata, fast_de_params):
        """Default should be use_empirical_variance=True."""
        from kompot.anndata.differential_expression import compute_differential_expression

        result = compute_differential_expression(
            tiny_adata,
            groupby="condition",
            condition1="A",
            condition2="B",
            obsm_key="X_pca",
            return_full_results=True,
            **fast_de_params,
        )

        model = result["model"]
        assert model.use_empirical_variance is True
        assert model.empirical_variance_predictor1 is not None


# ===== Leverage correction =====


class TestLeverageCorrection:
    """Tests for the leverage correction via mellon's predictor.leverage()."""

    @pytest.fixture
    def synth_data(self):
        rng = np.random.RandomState(42)
        n, g, d = 80, 6, 3
        X = rng.randn(n, d)
        y = rng.randn(n, g)
        return X, y

    def test_leverage_values_in_range(self, synth_data):
        """Leverage h_i should be in [0, 1) for all points."""
        X, y = synth_data
        de = DifferentialExpression(
            use_empirical_variance=True, n_landmarks=20, batch_size=0,
        )
        de.fit(X, y, X, y, sigma=1.0, ls_factor=10.0)

        h = np.asarray(de.function_predictor1.leverage(X))
        assert np.all(h >= 0), f"Negative leverage: {h.min()}"
        assert np.all(h < 1), f"Leverage >= 1: {h.max()}"

    def test_leverage_trace_bounded_by_landmarks(self, synth_data):
        """tr(H) = sum of leverages should be <= number of landmarks."""
        X, y = synth_data
        m = 20
        de = DifferentialExpression(
            use_empirical_variance=True, n_landmarks=m, batch_size=0,
        )
        de.fit(X, y, X, y, sigma=1.0, ls_factor=10.0)

        h = np.asarray(de.function_predictor1.leverage(X))
        assert h.sum() <= m + 0.1, (
            f"tr(H)={h.sum():.2f} should be <= m={m}"
        )

    def test_corrected_residuals_larger(self, synth_data):
        """Leverage-corrected squared residuals should be >= uncorrected."""
        X, y = synth_data
        import mellon
        est = mellon.FunctionEstimator(n_landmarks=20, sigma=1.0, optimizer='advi')
        est.fit(X, y)
        imputed = np.asarray(est.predict(X))
        raw_sq = (y - imputed) ** 2

        # loo_residuals_squared returns leverage-corrected squared residuals
        corrected_sq = np.asarray(est.predict.loo_residuals_squared(X, y))

        assert np.all(corrected_sq >= raw_sq - 1e-10), (
            "Corrected residuals should be >= uncorrected (h >= 0)"
        )

    def test_correction_reduces_bias(self):
        """Leverage correction should reduce variance estimation bias on average."""
        import mellon

        n, d, m = 200, 3, 40
        sigma_true = 3.0
        n_trials = 5
        raw_biases, corr_biases = [], []

        for seed in range(n_trials):
            rng = np.random.RandomState(seed)
            X = rng.randn(n, d)
            true_func = np.sin(X[:, 0]) + 0.5 * np.cos(X[:, 1])
            y = true_func + rng.randn(n) * sigma_true

            est = mellon.FunctionEstimator(n_landmarks=m, sigma=1.0, optimizer='advi')
            est.fit(X, y)
            imputed = np.asarray(est.predict(X))
            raw_sq = (y - imputed) ** 2

            corrected_sq = np.asarray(est.predict.loo_residuals_squared(X, y))

            true_var = sigma_true ** 2
            raw_biases.append((raw_sq.mean() - true_var) / true_var)
            corr_biases.append((corrected_sq.mean() - true_var) / true_var)

        # Raw residuals should be biased low (negative); corrected should be closer to 0
        mean_raw = np.mean(raw_biases)
        mean_corr = np.mean(corr_biases)
        assert abs(mean_corr) < abs(mean_raw), (
            f"Corrected mean bias ({mean_corr:.4f}) should be closer to 0 "
            f"than raw ({mean_raw:.4f})"
        )


# ===== Variance GP parameter reuse and overrides =====


class TestObsVarianceIntegration:
    """Tests for obs_variance integration with mellon."""

    @pytest.fixture
    def synth_data(self):
        rng = np.random.RandomState(42)
        n, g, d = 60, 6, 3
        X = rng.randn(n, d)
        y = rng.randn(n, g)
        return X, y

    def test_obs_variance_uses_expression_cov_func(self, synth_data):
        """obs_variance surface should use the expression GP's cov_func."""
        X, y = synth_data
        de = DifferentialExpression(
            use_empirical_variance=True, n_landmarks=20, batch_size=0,
        )
        de.fit(X, y, X, y, ls_factor=10.0)

        # obs_variance uses the same cov_func as the expression GP
        expr_ls = de.function_predictor1.cov_func.ls
        assert expr_ls > 0, "Expression GP should have a positive length scale"
        # Calling the variance predictor should produce the same result
        # as calling obs_variance on the expression predictor directly
        var_a = np.asarray(de.empirical_variance_predictor1(X[:5]))
        var_b = np.asarray(de.function_predictor1.obs_variance(X[:5]))
        np.testing.assert_allclose(var_a, var_b)

    def test_obs_variance_returns_correct_shape(self, synth_data):
        """obs_variance should return (n_points, n_genes)."""
        X, y = synth_data
        de = DifferentialExpression(
            use_empirical_variance=True, n_landmarks=20, batch_size=0,
        )
        de.fit(X, y, X, y, ls_factor=10.0)

        var = np.asarray(de.empirical_variance_predictor1(X))
        assert var.shape == y.shape, f"Expected {y.shape}, got {var.shape}"

    def test_obs_variance_is_smooth(self, synth_data):
        """obs_variance should be smoother than raw squared residuals."""
        X, y = synth_data
        de = DifferentialExpression(
            use_empirical_variance=True, n_landmarks=20, batch_size=0,
        )
        de.fit(X, y, X, y, sigma=1.0, ls_factor=10.0)

        smoothed = np.asarray(de.empirical_variance_predictor1(X))
        raw_hc3 = np.asarray(
            de.function_predictor1.loo_residuals_squared(X, y)
        )

        # Smoothed should have lower coefficient of variation per gene
        cv_raw = np.std(raw_hc3, axis=0) / (np.mean(raw_hc3, axis=0) + 1e-10)
        cv_smooth = np.std(smoothed, axis=0) / (np.mean(smoothed, axis=0) + 1e-10)
        assert np.mean(cv_smooth) < np.mean(cv_raw), (
            f"Smoothed CV ({np.mean(cv_smooth):.2f}) should be less than "
            f"raw CV ({np.mean(cv_raw):.2f})"
        )
