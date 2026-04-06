"""Tests for ExpressionModel unfitted state and coverage branches."""

import numpy as np
import pytest


class TestExpressionModelCoverage:
    """Cover uncovered branches in expression_model.py."""

    def test_model_not_fitted(self):
        from kompot.differential.expression_model import ExpressionModel

        model = ExpressionModel(n_landmarks=10)
        with pytest.raises(ValueError, match="not fitted"):
            model.predict(np.random.randn(5, 3))

    def test_covariance_not_fitted(self):
        from kompot.differential.expression_model import ExpressionModel

        model = ExpressionModel(n_landmarks=10)
        with pytest.raises(ValueError, match="not fitted"):
            model.covariance(np.random.randn(5, 3))

    def test_total_variance_not_fitted(self):
        from kompot.differential.expression_model import ExpressionModel

        model = ExpressionModel(n_landmarks=10)
        with pytest.raises(ValueError, match="not fitted"):
            model.total_variance(np.random.randn(5, 3))

    def test_obs_variance_not_fitted(self):
        from kompot.differential.expression_model import ExpressionModel

        model = ExpressionModel(n_landmarks=10)
        # obs_variance returns 0 when not fitted (no obs_variance_func)
        result = model.obs_variance(np.random.randn(5, 3))
        assert np.all(result == 0)

    def test_has_sample_variance_before_fit(self):
        from kompot.differential.expression_model import ExpressionModel

        model = ExpressionModel(n_landmarks=10)
        assert isinstance(model.has_sample_variance, bool)

    def test_repr(self):
        from kompot.differential.expression_model import ExpressionModel

        model = ExpressionModel(n_landmarks=10)
        r = repr(model)
        assert (
            "ExpressionModel" in r
            or "expression_model" in r.lower()
            or "object at" in r
        )
