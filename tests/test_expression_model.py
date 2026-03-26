"""Tests for ExpressionModel class.

Covers:
1. Basic fit and predict (shape, no NaNs)
2. predict() matches direct predictor call
3. covariance() diagonal and full modes
4. obs_variance() enabled/disabled
5. sample_variance() enabled/disabled
6. total_variance composition
7. std computation
8. Properties: landmarks, ls, cov_func, sigma
9. Pre-fitted constructor
10. has_sample_variance and has_empirical_variance flags
"""

import numpy as np
import pytest

from kompot.differential.expression_model import ExpressionModel


@pytest.fixture
def synthetic_data():
    rng = np.random.RandomState(42)
    n_cells, n_features, n_genes = 100, 5, 10
    X = rng.randn(n_cells, n_features)
    y = rng.randn(n_cells, n_genes)
    return X, y


@pytest.fixture
def fitted_model(synthetic_data):
    X, y = synthetic_data
    model = ExpressionModel(n_landmarks=50, use_empirical_variance=False)
    model.fit(X, y, sigma=1.0, ls_factor=10.0)
    return model, X, y


@pytest.fixture
def fitted_model_empirical(synthetic_data):
    X, y = synthetic_data
    model = ExpressionModel(n_landmarks=50, use_empirical_variance=True)
    model.fit(X, y, sigma=1.0, ls_factor=10.0)
    return model, X, y


class TestBasicFitPredict:

    def test_predict_shape(self, fitted_model):
        model, X, y = fitted_model
        result = np.asarray(model.predict(X, progress=False))
        assert result.shape == (X.shape[0], y.shape[1])

    def test_predict_no_nans(self, fitted_model):
        model, X, y = fitted_model
        result = np.asarray(model.predict(X, progress=False))
        assert not np.any(np.isnan(result))

    def test_predict_before_fit_raises(self):
        model = ExpressionModel()
        X = np.random.randn(10, 3)
        with pytest.raises(ValueError, match="not fitted"):
            model.predict(X)


class TestPredictMatchesPredictor:

    def test_predict_matches_direct_call(self, fitted_model):
        model, X, y = fitted_model
        predicted = np.asarray(model.predict(X, progress=False))
        direct = np.asarray(model.predictor(X))
        np.testing.assert_allclose(predicted, direct, atol=1e-5)


class TestCovariance:

    def test_covariance_diag_shape(self, fitted_model):
        model, X, y = fitted_model
        cov = model.covariance(X, diag=True, progress=False)
        # GP covariance is shared across genes: shape (n_points,)
        assert cov.shape == (X.shape[0],)

    def test_covariance_diag_positive(self, fitted_model):
        model, X, y = fitted_model
        cov = model.covariance(X, diag=True, progress=False)
        assert np.all(cov >= 0)

    def test_covariance_full_shape(self, fitted_model):
        model, X, y = fitted_model
        # Use a small subset to keep memory manageable
        X_small = X[:10]
        cov = model.covariance(X_small, diag=False, progress=False)
        assert cov.shape[0] == 10
        assert cov.shape[1] == 10

    def test_covariance_before_fit_raises(self):
        model = ExpressionModel()
        with pytest.raises(ValueError, match="not fitted"):
            model.covariance(np.random.randn(5, 3))


class TestObsVariance:

    def test_obs_variance_disabled_returns_zero(self, fitted_model):
        model, X, y = fitted_model
        assert model.obs_variance(X, progress=False) == 0

    def test_obs_variance_enabled_returns_array(self, fitted_model_empirical):
        model, X, y = fitted_model_empirical
        obs_var = model.obs_variance(X, progress=False)
        assert isinstance(obs_var, np.ndarray)
        assert obs_var.shape == (X.shape[0], y.shape[1])

    def test_obs_variance_enabled_positive(self, fitted_model_empirical):
        model, X, y = fitted_model_empirical
        obs_var = model.obs_variance(X, progress=False)
        assert np.all(obs_var > 0)


class TestSampleVariance:

    def test_sample_variance_disabled_returns_zero(self, fitted_model):
        model, X, y = fitted_model
        assert model.sample_variance(X, progress=False) == 0

    def test_sample_variance_enabled(self, synthetic_data):
        X, y = synthetic_data
        n_cells = X.shape[0]
        sample_indices = np.repeat(np.arange(3), n_cells // 3 + 1)[:n_cells]

        model = ExpressionModel(n_landmarks=50)
        model.fit(
            X, y, sigma=1.0, ls_factor=10.0,
            sample_indices=sample_indices,
        )
        sv = model.sample_variance(X, diag=True, progress=False)
        assert isinstance(sv, np.ndarray)
        assert sv.shape[0] == n_cells


class TestTotalVariance:

    def test_total_variance_without_empirical(self, fitted_model):
        """total_variance = covariance when obs_variance and sample_variance are 0."""
        model, X, y = fitted_model
        X_eval = X[:20]

        total = model.total_variance(X_eval, diag=True, progress=False)
        cov = model.covariance(X_eval, diag=True, progress=False)
        # obs=0, sam=0; total_variance reshapes cov to (n, 1)
        np.testing.assert_allclose(total.ravel(), cov.ravel(), atol=1e-6)

    def test_total_variance_with_empirical(self, fitted_model_empirical):
        """total_variance should combine covariance + obs_variance + sample_variance."""
        model, X, y = fitted_model_empirical
        X_eval = X[:20]
        total = model.total_variance(X_eval, diag=True, progress=False)
        assert total.shape == (20, y.shape[1])


class TestStd:

    def test_std_without_empirical(self, fitted_model):
        """std = sqrt(total_variance + eps) when empirical variance is off."""
        model, X, y = fitted_model
        X_eval = X[:20]

        std = model.std(X_eval, progress=False)
        total = model.total_variance(X_eval, diag=True, progress=False)
        expected = np.sqrt(total + model.eps)

        np.testing.assert_allclose(std, expected, atol=1e-6)

    def test_std_with_empirical(self, fitted_model_empirical):
        """std should work with empirical variance enabled."""
        model, X, y = fitted_model_empirical
        X_eval = X[:20]
        std = model.std(X_eval, progress=False)
        assert std.shape == (20, y.shape[1])


class TestProperties:

    def test_landmarks_accessible(self, fitted_model):
        model, X, y = fitted_model
        landmarks = model.landmarks
        assert landmarks is not None
        assert landmarks.shape[0] == 50
        assert landmarks.shape[1] == X.shape[1]

    def test_ls_accessible(self, fitted_model):
        model, X, y = fitted_model
        ls = model.ls
        assert ls is not None
        assert ls > 0

    def test_cov_func_accessible(self, fitted_model):
        model, X, y = fitted_model
        cov_func = model.cov_func
        assert cov_func is not None

    def test_sigma_accessible(self, fitted_model):
        model, X, y = fitted_model
        assert model.sigma == 1.0


class TestPreFittedConstructor:

    def test_with_function_predictor(self, fitted_model):
        model, X, y = fitted_model
        predictor = model.predictor

        new_model = ExpressionModel(function_predictor=predictor)
        result = np.asarray(new_model.predict(X, progress=False))
        original = np.asarray(model.predict(X, progress=False))

        np.testing.assert_allclose(result, original, atol=1e-5)

    def test_pre_fitted_no_fit_needed(self, fitted_model):
        model, X, y = fitted_model
        new_model = ExpressionModel(function_predictor=model.predictor)
        # Should work without calling fit()
        result = np.asarray(new_model.predict(X[:5], progress=False))
        assert result.shape == (5, y.shape[1])


class TestHasFlags:

    def test_has_empirical_variance_false(self, fitted_model):
        model, X, y = fitted_model
        assert model.has_empirical_variance is False

    def test_has_empirical_variance_true(self, fitted_model_empirical):
        model, X, y = fitted_model_empirical
        assert model.has_empirical_variance is True

    def test_has_sample_variance_false(self, fitted_model):
        model, X, y = fitted_model
        assert model.has_sample_variance is False

    def test_has_sample_variance_true(self, synthetic_data):
        X, y = synthetic_data
        n_cells = X.shape[0]
        sample_indices = np.repeat(np.arange(3), n_cells // 3 + 1)[:n_cells]

        model = ExpressionModel(n_landmarks=50)
        model.fit(X, y, sigma=1.0, ls_factor=10.0, sample_indices=sample_indices)
        assert model.has_sample_variance is True

    def test_unfitted_flags(self):
        model = ExpressionModel(use_empirical_variance=True)
        assert model.has_empirical_variance is False
        assert model.has_sample_variance is False
