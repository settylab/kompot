"""Tests for ModelSettings and pre-fitted predictor injection.

Covers:
1. ModelSettings dataclass construction and defaults
2. obs_variance_predictor injection into ExpressionModel
3. ExpressionModel.fit() skip logic when obs_variance is injected
4. DifferentialExpression: injected obs_variance changes std output
5. de() with model=ModelSettings(...): injected predictors change results
6. da() with model=ModelSettings(...): injected predictors change results
"""

import numpy as np
import pytest
from unittest.mock import MagicMock

from kompot.settings import ModelSettings
from kompot.differential.expression_model import ExpressionModel
from kompot.differential.differential_expression import DifferentialExpression


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_data():
    rng = np.random.RandomState(42)
    n_cells, n_features, n_genes = 80, 5, 8
    X = rng.randn(n_cells, n_features)
    y = rng.randn(n_cells, n_genes)
    return X, y


@pytest.fixture
def two_condition_data():
    rng = np.random.RandomState(42)
    n_cells, n_features, n_genes = 60, 5, 8
    X1 = rng.randn(n_cells, n_features)
    X2 = rng.randn(n_cells, n_features) + 0.5
    y1 = rng.randn(n_cells, n_genes)
    y2 = rng.randn(n_cells, n_genes) + 0.3
    return X1, X2, y1, y2


@pytest.fixture
def fitted_de(two_condition_data):
    """Fit a DifferentialExpression with empirical variance."""
    X1, X2, y1, y2 = two_condition_data
    de = DifferentialExpression(
        n_landmarks=30,
        use_empirical_variance=True,
    )
    de.fit(X1, y1, X2, y2, sigma=1.0, ls_factor=10.0)
    return de, X1, X2, y1, y2


# ---------------------------------------------------------------------------
# 1. ModelSettings dataclass
# ---------------------------------------------------------------------------

class TestModelSettingsDataclass:

    def test_defaults_all_none(self):
        ms = ModelSettings()
        for field in (
            "model1", "model2",
            "function_predictor1", "function_predictor2",
            "obs_variance_predictor1", "obs_variance_predictor2",
            "variance_predictor1", "variance_predictor2",
            "density_predictor1", "density_predictor2",
        ):
            assert getattr(ms, field) is None

    def test_set_individual_fields(self):
        mock = MagicMock()
        ms = ModelSettings(function_predictor1=mock, obs_variance_predictor2=mock)
        assert ms.function_predictor1 is mock
        assert ms.obs_variance_predictor2 is mock
        assert ms.function_predictor2 is None

    def test_importable_from_kompot(self):
        import kompot
        assert kompot.ModelSettings is ModelSettings


# ---------------------------------------------------------------------------
# 2. obs_variance_predictor injection into ExpressionModel
# ---------------------------------------------------------------------------

class TestExpressionModelObsVarianceInjection:

    def test_has_empirical_variance_true_when_injected(self):
        model = ExpressionModel(
            obs_variance_predictor=MagicMock(),
            use_empirical_variance=True,
        )
        assert model.has_empirical_variance is True

    def test_injected_predictor_value_surfaces_in_obs_variance(self, synthetic_data):
        """Injecting a predictor that returns a known constant should produce
        that constant (clipped by eps) from obs_variance()."""
        X, y = synthetic_data
        n_cells, n_genes = X.shape[0], y.shape[1]
        INJECTED_VALUE = 42.0

        model = ExpressionModel(
            obs_variance_predictor=lambda Xb: np.full(
                (Xb.shape[0], n_genes), INJECTED_VALUE
            ),
            use_empirical_variance=True,
        )
        result = model.obs_variance(X, progress=False)
        np.testing.assert_allclose(result, INJECTED_VALUE, atol=1e-6)

    def test_injected_obs_var_increases_total_variance(self, synthetic_data):
        """Injecting a large obs_variance should make total_variance larger
        than GP covariance alone."""
        X, y = synthetic_data
        n_cells, n_genes = X.shape[0], y.shape[1]

        # Fit a model without empirical variance to get a baseline
        base = ExpressionModel(n_landmarks=30, use_empirical_variance=False)
        base.fit(X, y, sigma=1.0, ls_factor=10.0)
        base_total = base.total_variance(X[:10], diag=True, progress=False)

        # Now create a model with the same function predictor but a huge
        # injected obs_variance
        big_obs = ExpressionModel(
            function_predictor=base.predictor,
            obs_variance_predictor=lambda Xb: np.full(
                (Xb.shape[0], n_genes), 1000.0
            ),
            use_empirical_variance=True,
        )
        big_total = big_obs.total_variance(X[:10], diag=True, progress=False)

        # Every element should be larger
        assert np.all(big_total > base_total)
        # And the difference should be approximately 1000
        diff = big_total - base_total
        np.testing.assert_allclose(diff, 1000.0, atol=1.0)


# ---------------------------------------------------------------------------
# 3. ExpressionModel.fit() skip logic when obs_variance is injected
# ---------------------------------------------------------------------------

class TestExpressionModelFitSkipLogic:

    def test_fit_preserves_injected_obs_var(self, synthetic_data):
        """fit() should not overwrite an injected obs_variance_predictor."""
        X, y = synthetic_data
        sentinel = lambda Xb: np.ones((Xb.shape[0], y.shape[1]))

        model = ExpressionModel(
            n_landmarks=30,
            use_empirical_variance=True,
            obs_variance_predictor=sentinel,
        )
        model.fit(X, y, sigma=1.0, ls_factor=10.0)

        assert model._within_sample_obs_var_predictor is sentinel
        # Function predictor should still be fitted
        assert model.predictor is not None

    def test_fit_uses_injected_not_internal_obs_var(self, synthetic_data):
        """When obs_var is injected, obs_variance() should return the
        injected value, not anything mellon computed internally."""
        X, y = synthetic_data
        n_genes = y.shape[1]
        SENTINEL_VALUE = 123.0
        sentinel = lambda Xb: np.full((Xb.shape[0], n_genes), SENTINEL_VALUE)

        model = ExpressionModel(
            n_landmarks=30,
            use_empirical_variance=True,
            obs_variance_predictor=sentinel,
        )
        model.fit(X, y, sigma=1.0, ls_factor=10.0)

        result = model.obs_variance(X[:5], progress=False)
        np.testing.assert_allclose(result, SENTINEL_VALUE, atol=1e-6)

    def test_fit_preserves_injected_obs_var_with_samples(self, synthetic_data):
        """Per-sample empirical variance path should be skipped when
        obs_var predictor is already injected."""
        X, y = synthetic_data
        n_cells = X.shape[0]
        sample_indices = np.repeat(np.arange(3), n_cells // 3 + 1)[:n_cells]
        sentinel = lambda Xb: np.ones((Xb.shape[0], y.shape[1]))

        model = ExpressionModel(
            n_landmarks=30,
            use_empirical_variance=True,
            obs_variance_predictor=sentinel,
        )
        model.fit(
            X, y, sigma=1.0, ls_factor=10.0,
            sample_indices=sample_indices,
        )
        assert model._within_sample_obs_var_predictor is sentinel


# ---------------------------------------------------------------------------
# 4. DifferentialExpression: injected obs_variance changes std output
# ---------------------------------------------------------------------------

class TestDifferentialExpressionObsVarInjection:

    def test_model_takes_precedence_over_individual(self):
        """model1/2 should take precedence over individual predictors."""
        mock_model = ExpressionModel(use_empirical_variance=True)
        de = DifferentialExpression(
            model1=mock_model,
            function_predictor1=MagicMock(),
            obs_variance_predictor1=MagicMock(),
        )
        assert de.model1 is mock_model

    def test_large_obs_var_inflates_std(self, fitted_de):
        """Replacing the obs_variance predictors with huge values should
        inflate condition_std relative to the original."""
        de_orig, X1, X2, y1, y2 = fitted_de
        X_eval = X1[:10]
        n_genes = y1.shape[1]

        orig = de_orig.predict(X_eval, progress=False)

        # Build a new DE with same function predictors but huge obs_var
        de_big = DifferentialExpression(
            function_predictor1=de_orig.function_predictor1,
            function_predictor2=de_orig.function_predictor2,
            obs_variance_predictor1=lambda Xb: np.full(
                (Xb.shape[0], n_genes), 1e4
            ),
            obs_variance_predictor2=lambda Xb: np.full(
                (Xb.shape[0], n_genes), 1e4
            ),
            use_empirical_variance=True,
        )
        big = de_big.predict(X_eval, progress=False)

        # fold_change should be identical (same function predictors)
        np.testing.assert_allclose(
            big["fold_change"], orig["fold_change"], atol=1e-5,
        )
        # std should be much larger
        assert np.all(big["condition1_std"] > orig["condition1_std"])
        assert np.all(big["condition2_std"] > orig["condition2_std"])

    def test_zero_obs_var_matches_no_empirical(self, two_condition_data):
        """Injecting obs_var~=0 should give approximately the same std as
        use_empirical_variance=False."""
        X1, X2, y1, y2 = two_condition_data
        n_genes = y1.shape[1]

        # Fit without empirical variance
        de_no_ev = DifferentialExpression(
            n_landmarks=30,
            use_empirical_variance=False,
        )
        de_no_ev.fit(X1, y1, X2, y2, sigma=1.0, ls_factor=10.0)

        # Build a new DE with same function predictors and zero obs_var
        de_zero = DifferentialExpression(
            function_predictor1=de_no_ev.function_predictor1,
            function_predictor2=de_no_ev.function_predictor2,
            obs_variance_predictor1=lambda Xb: np.full(
                (Xb.shape[0], n_genes), 0.0
            ),
            obs_variance_predictor2=lambda Xb: np.full(
                (Xb.shape[0], n_genes), 0.0
            ),
            use_empirical_variance=True,
            eps=de_no_ev.eps,
        )

        X_eval = X1[:10]
        res_no = de_no_ev.predict(X_eval, progress=False)
        res_zero = de_zero.predict(X_eval, progress=False)

        # obs_variance returns max(pred, eps) so with 0.0 input it returns eps.
        # Without empirical var, std shape is (n, 1) due to broadcast;
        # with zero obs_var it's (n, n_genes).  Compare element-wise after
        # broadcasting.
        std_no = np.broadcast_to(
            res_no["condition1_std"],
            res_zero["condition1_std"].shape,
        )
        np.testing.assert_allclose(
            res_zero["condition1_std"], std_no, atol=1e-3,
        )

    def test_fit_validation_passes_with_injected_obs_var(self, two_condition_data):
        """fit() validation should pass when obs_variance is injected
        (no need for mellon predictor to have .obs_variance)."""
        X1, X2, y1, y2 = two_condition_data
        n_genes = y1.shape[1]

        de = DifferentialExpression(
            n_landmarks=30,
            use_empirical_variance=True,
            obs_variance_predictor1=lambda Xb: np.ones((Xb.shape[0], n_genes)),
            obs_variance_predictor2=lambda Xb: np.ones((Xb.shape[0], n_genes)),
        )
        de.fit(X1, y1, X2, y2, sigma=1.0, ls_factor=10.0)
        assert de.model1.has_empirical_variance is True
        assert de.model2.has_empirical_variance is True


# ---------------------------------------------------------------------------
# 5. de() with model=ModelSettings(...)
# ---------------------------------------------------------------------------

class TestDeWithModelSettings:

    @pytest.fixture
    def test_adata(self):
        import anndata
        rng = np.random.RandomState(42)
        n_cells, n_genes, n_features = 60, 8, 5
        X_expr = rng.randn(n_cells, n_genes)
        obsm = {"DM_EigenVectors": rng.randn(n_cells, n_features)}
        obs = {"condition": ["Young"] * 30 + ["Old"] * 30}
        var_names = [f"gene_{i}" for i in range(n_genes)]
        adata = anndata.AnnData(
            X=X_expr,
            obs=obs,
            var={"_": np.zeros(n_genes)},
            obsm=obsm,
        )
        adata.var_names = var_names
        return adata

    def test_injected_models_are_used(self, test_adata):
        """Inject pre-fitted ExpressionModels through model= and verify
        the returned model object is the same instance (not re-fitted)."""
        import kompot

        # First run: fit models
        result1 = kompot.de(
            test_adata.copy(),
            "condition", "Young", "Old",
            gp=kompot.GPSettings(n_landmarks=30, use_empirical_variance=True),
            fdr=kompot.FDRSettings(null_genes=0),
            output=kompot.OutputSettings(
                compute_mahalanobis=False, return_full_results=True,
            ),
        )
        model1 = result1["model"].model1
        model2 = result1["model"].model2

        # Second run: inject pre-fitted models
        result2 = kompot.de(
            test_adata.copy(),
            "condition", "Young", "Old",
            gp=kompot.GPSettings(n_landmarks=30, use_empirical_variance=True),
            fdr=kompot.FDRSettings(null_genes=0),
            output=kompot.OutputSettings(
                compute_mahalanobis=False, return_full_results=True,
            ),
            model=kompot.ModelSettings(model1=model1, model2=model2),
        )

        # The returned DE should contain the exact same model instances
        assert result2["model"].model1 is model1
        assert result2["model"].model2 is model2
        # And produce the same LFC
        np.testing.assert_allclose(
            result2["table"]["mean_lfc"].values,
            result1["table"]["mean_lfc"].values,
            atol=1e-5,
        )

    def test_injected_obs_var_changes_de_output(self, test_adata):
        """Injecting huge obs_variance predictors through de() should
        inflate z-scores toward zero (larger variance → smaller z)."""
        import kompot

        # Baseline run
        result_base = kompot.de(
            test_adata.copy(),
            "condition", "Young", "Old",
            gp=kompot.GPSettings(n_landmarks=30, use_empirical_variance=True),
            fdr=kompot.FDRSettings(null_genes=0),
            output=kompot.OutputSettings(
                compute_mahalanobis=False, return_full_results=True,
            ),
        )
        de_base = result_base["model"]
        n_genes = test_adata.n_vars

        # Re-run with same function predictors but enormous obs_variance
        result_big = kompot.de(
            test_adata.copy(),
            "condition", "Young", "Old",
            gp=kompot.GPSettings(n_landmarks=30, use_empirical_variance=True),
            fdr=kompot.FDRSettings(null_genes=0),
            output=kompot.OutputSettings(
                compute_mahalanobis=False, return_full_results=True,
            ),
            model=kompot.ModelSettings(
                function_predictor1=de_base.function_predictor1,
                function_predictor2=de_base.function_predictor2,
                obs_variance_predictor1=lambda Xb: np.full(
                    (Xb.shape[0], n_genes), 1e6
                ),
                obs_variance_predictor2=lambda Xb: np.full(
                    (Xb.shape[0], n_genes), 1e6
                ),
            ),
        )

        # LFC should be identical (same function predictors)
        np.testing.assert_allclose(
            result_big["table"]["mean_lfc"].values,
            result_base["table"]["mean_lfc"].values,
            atol=1e-5,
        )

        # With huge variance the z-scores stored in layers should be
        # closer to zero.  Access them from the DE model's predict output.
        X_eval = test_adata.obsm["DM_EigenVectors"]
        pred_base = de_base.predict(X_eval, progress=False)
        pred_big = result_big["model"].predict(X_eval, progress=False)

        mean_abs_z_base = np.mean(np.abs(pred_base["fold_change_zscores"]))
        mean_abs_z_big = np.mean(np.abs(pred_big["fold_change_zscores"]))
        assert mean_abs_z_big < mean_abs_z_base


# ---------------------------------------------------------------------------
# 6. da() with model=ModelSettings(...)
# ---------------------------------------------------------------------------

class TestDaWithModelSettings:

    @pytest.fixture
    def test_adata(self):
        import anndata
        rng = np.random.RandomState(42)
        n_cells, n_genes, n_features = 60, 8, 5
        X_expr = rng.randn(n_cells, n_genes)
        obsm = {"DM_EigenVectors": rng.randn(n_cells, n_features)}
        obs = {"condition": ["Young"] * 30 + ["Old"] * 30}
        adata = anndata.AnnData(
            X=X_expr,
            obs=obs,
            obsm=obsm,
        )
        return adata

    def test_injected_density_predictors_are_used(self, test_adata):
        """Inject pre-fitted density predictors and verify same instance."""
        import kompot

        result1 = kompot.da(
            test_adata.copy(),
            "condition", "Young", "Old",
            output=kompot.OutputSettings(return_full_results=True),
        )
        dp1 = result1["model"].density_predictor1
        dp2 = result1["model"].density_predictor2

        result2 = kompot.da(
            test_adata.copy(),
            "condition", "Young", "Old",
            output=kompot.OutputSettings(return_full_results=True),
            model=kompot.ModelSettings(
                density_predictor1=dp1,
                density_predictor2=dp2,
            ),
        )

        # Same predictor instances → same results
        assert result2["model"].density_predictor1 is dp1
        assert result2["model"].density_predictor2 is dp2
        np.testing.assert_allclose(
            result2["table"]["lfc"].values,
            result1["table"]["lfc"].values,
            atol=1e-5,
        )

    def test_swapped_density_predictors_flip_lfc(self, test_adata):
        """Swapping density_predictor1 and density_predictor2 should
        negate the log fold change (lfc = log_density2 - log_density1)."""
        import kompot

        result_normal = kompot.da(
            test_adata.copy(),
            "condition", "Young", "Old",
            output=kompot.OutputSettings(return_full_results=True),
        )
        dp1 = result_normal["model"].density_predictor1
        dp2 = result_normal["model"].density_predictor2

        result_swapped = kompot.da(
            test_adata.copy(),
            "condition", "Young", "Old",
            output=kompot.OutputSettings(return_full_results=True),
            model=kompot.ModelSettings(
                density_predictor1=dp2,  # swapped
                density_predictor2=dp1,  # swapped
            ),
        )

        np.testing.assert_allclose(
            result_swapped["table"]["lfc"].values,
            -result_normal["table"]["lfc"].values,
            atol=1e-4,
        )
