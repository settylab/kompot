"""Regression tests pinning manuscript-aligned statistical behavior.

These tests guard the v0.8.0 corrections so the implementation cannot
silently drift away from the manuscript's definitions:

* Mahalanobis denominator must SUM covariances (not average) so that
  ``D(a,b) = sqrt((mu_a - mu_b)^T (Sigma_a + Sigma_b)^(-1) (mu_a - mu_b))``.
* DA posterior tail probability must be ONE-sided: ``Phi(-|z|)``.
* ``use_empirical_variance`` must default to ``False`` at every
  publicly-exposed entry point (the manuscript states that empirical
  variance is disabled by default).
"""

import inspect

import numpy as np
import pytest


# -----------------------------------------------------------------------------
# Mahalanobis denominator is Σ_a + Σ_b (sum), not the average
# -----------------------------------------------------------------------------


class TestMahalanobisDenominatorIsSum:
    """The covariance denominator in the gene-wise Mahalanobis distance
    is the *sum* of the two posterior covariance matrices.
    """

    def test_combined_cov_equals_sum_via_compute_mahalanobis_distances(
        self, monkeypatch
    ):
        """Capture the ``combined_cov`` argument that
        ``DifferentialExpression.compute_mahalanobis_distances`` passes
        into the underlying ``compute_mahalanobis_distances`` utility
        and assert it equals ``cov1 + cov2`` (not ``(cov1+cov2)/2``).
        """
        from kompot.differential import DifferentialExpression
        from kompot.differential import differential_expression as de_module

        captured = {}

        def fake_compute(
            diff_values,
            covariance=None,
            batch_size=500,
            jit_compile=False,
            progress=False,
            eps=1e-10,
            diagonal_variance=None,
            **_kwargs,
        ):
            captured["combined_cov"] = np.asarray(covariance)
            n_genes = np.asarray(diff_values).shape[0]
            return np.zeros(n_genes, dtype=float)

        monkeypatch.setattr(
            de_module, "compute_mahalanobis_distances", fake_compute
        )

        # Synthetic predictors with controllable covariance kernels:
        # cov1 returns 2*I, cov2 returns 3*I, so cov1+cov2 = 5*I and the
        # (buggy) average would be 2.5*I.
        class _Pred:
            def __init__(self, scale):
                self.scale = scale

            def covariance(self, X, diag=False):
                k = X.shape[0]
                return self.scale * np.eye(k)

            def __call__(self, X):
                # Return an (n_cells, n_genes) zero-mean expression so
                # downstream `fold_change_subset` is well-defined.
                return np.zeros((X.shape[0], 3), dtype=float)

        de = DifferentialExpression(
            n_landmarks=None,
            use_sample_variance=False,
            use_empirical_variance=False,
            function_predictor1=_Pred(2.0),
            function_predictor2=_Pred(3.0),
        )

        X_new = np.random.RandomState(0).randn(8, 4)
        de.compute_mahalanobis_distances(X_new, use_landmarks=False, progress=False)

        combined_cov = captured["combined_cov"]
        expected_sum = 5.0 * np.eye(X_new.shape[0])
        np.testing.assert_allclose(
            combined_cov,
            expected_sum,
            rtol=1e-12,
            atol=0,
            err_msg=(
                "Regression: combined posterior covariance should be "
                "cov1 + cov2 (= 5*I here), got something else. The pre-"
                "0.8.0 (buggy) value would have been 2.5*I (= "
                "(cov1 + cov2) / 2)."
            ),
        )


# -----------------------------------------------------------------------------
# DA PTP is one-sided: Phi(-|z|), not 2*Phi(-|z|)
# -----------------------------------------------------------------------------


class TestDifferentialAbundancePTPOneSided:
    """The differential-abundance posterior tail probability matches
    the one-sided manuscript definition ``PTP = Phi(-|z|)``.
    """

    def test_ptp_one_sided_synthetic_z(self):
        from scipy.stats import norm

        # Replicate the exact ln_ptp computation from
        # kompot.differential.differential_abundance, fed with controlled
        # z-scores so we can compare against the closed-form one-sided
        # tail probability.
        import jax.scipy.stats.norm as normal

        z = np.array([-3.0, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0])

        ln_ptp = np.minimum(
            np.asarray(normal.logcdf(z)),
            np.asarray(normal.logcdf(-z)),
        )
        ptp = np.exp(ln_ptp)

        expected_one_sided = norm.cdf(-np.abs(z))
        np.testing.assert_allclose(
            ptp,
            expected_one_sided,
            rtol=1e-10,
            atol=1e-12,
            err_msg=(
                "Regression: PTP should be the one-sided tail Phi(-|z|). "
                "Pre-0.8.0 code emitted 2*Phi(-|z|) (two-sided)."
            ),
        )

        # And explicitly that it is NOT the two-sided variant
        two_sided = 2.0 * norm.cdf(-np.abs(z))
        # Allow the symmetric `z == 0` boundary case (where both sides
        # collapse to 0.5 and 1.0 respectively) by checking the strict
        # off-axis values.
        nonzero = z != 0
        assert np.all(
            np.abs(ptp[nonzero] - two_sided[nonzero]) > 1e-3
        ), "PTP unexpectedly equals 2*Phi(-|z|) (two-sided)."

    def test_da_predict_emits_one_sided_ptp(self):
        """End-to-end: fit DA on a clearly-separated synthetic pair and
        verify the recovered PTP at each evaluation point equals
        ``Phi(-|z|)`` computed from the same fit's z-score, not twice
        that value.
        """
        from scipy.stats import norm
        from kompot.differential import DifferentialAbundance

        rng = np.random.RandomState(42)
        X1 = rng.randn(80, 3)
        X2 = rng.randn(80, 3) + 0.4

        da = DifferentialAbundance()
        da.fit(X1, X2)

        X_eval = np.vstack([X1[:20], X2[:20]])
        out = da.predict(X_eval, progress=False)

        z = np.asarray(out["log_fold_change_zscore"])
        neg_log10_ptp = np.asarray(out["neg_log10_fold_change_ptp"])
        ptp = 10.0 ** (-neg_log10_ptp)

        expected = norm.cdf(-np.abs(z))
        np.testing.assert_allclose(
            ptp,
            expected,
            rtol=1e-4,
            atol=1e-6,
            err_msg=(
                "Regression: PTP returned by DifferentialAbundance."
                "predict() does not match the one-sided Phi(-|z|)."
            ),
        )


# -----------------------------------------------------------------------------
# use_empirical_variance defaults to False at every public entry point
# -----------------------------------------------------------------------------


class TestUseEmpiricalVarianceDefaultIsFalse:
    """Every publicly-exposed entry point that accepts
    ``use_empirical_variance`` must default to ``False`` (matching the
    manuscript's "empirical variance is disabled by default" statement).
    """

    def _default_for(self, callable_obj, param_name="use_empirical_variance"):
        sig = inspect.signature(callable_obj)
        assert param_name in sig.parameters, (
            f"{callable_obj.__qualname__} does not expose {param_name}"
        )
        param = sig.parameters[param_name]
        assert param.default is not inspect.Parameter.empty, (
            f"{callable_obj.__qualname__} parameter {param_name} has "
            f"no default value"
        )
        return param.default

    def test_gpsettings_default_is_false(self):
        from kompot.settings import GPSettings

        assert GPSettings().use_empirical_variance is False

    def test_differential_expression_init_default_is_false(self):
        from kompot.differential import DifferentialExpression

        assert (
            self._default_for(DifferentialExpression.__init__) is False
        )

    def test_expression_model_init_default_is_false(self):
        from kompot.differential.expression_model import ExpressionModel

        assert self._default_for(ExpressionModel.__init__) is False

    def test_deprecated_compute_differential_expression_default_is_false(self):
        from kompot.anndata.differential_expression import (
            compute_differential_expression,
        )

        assert self._default_for(compute_differential_expression) is False

    def test_deprecated_compute_smoothed_expression_default_is_false(self):
        from kompot.anndata.smooth import compute_smoothed_expression

        assert self._default_for(compute_smoothed_expression) is False

    def test_smooth_config_template_default_is_false(self):
        import pathlib

        import yaml

        import kompot

        template = (
            pathlib.Path(kompot.__file__).parent
            / "cli"
            / "templates"
            / "smooth_config_template.yaml"
        )
        cfg = yaml.safe_load(template.read_text())
        assert cfg["use_empirical_variance"] is False

    def test_de_config_template_default_is_false(self):
        import pathlib

        import yaml

        import kompot

        template = (
            pathlib.Path(kompot.__file__).parent
            / "cli"
            / "templates"
            / "de_config_template.yaml"
        )
        cfg = yaml.safe_load(template.read_text())
        assert cfg["use_empirical_variance"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
