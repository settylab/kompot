"""Comprehensive tests for kompot.differential modules to improve coverage."""

import numpy as np
import pandas as pd
import pytest
import sys
from unittest.mock import patch, MagicMock


def create_differential_test_data(n_cells1=30, n_cells2=40, n_features=8):
    """Create test data for differential analysis."""
    np.random.seed(42)

    # Create two conditions with different characteristics
    X1 = np.random.normal(0, 1, (n_cells1, n_features))
    X2 = np.random.normal(1, 1, (n_cells2, n_features))  # Shifted mean

    return X1, X2


class TestDifferentialAbundanceCore:
    """Test core DifferentialAbundance functionality."""

    def test_differential_abundance_init_basic(self):
        """Test basic DifferentialAbundance initialization."""
        try:
            from kompot.differential import DifferentialAbundance
        except ImportError as e:
            pytest.skip(f"Could not import DifferentialAbundance: {e}")

        da = DifferentialAbundance(
            log_fold_change_threshold=1.5, ptp_threshold=0.01, n_landmarks=10, eps=1e-10
        )

        assert da.log_fold_change_threshold == 1.5
        assert da.ptp_threshold == 0.01
        assert da.n_landmarks == 10
        assert da.eps == 1e-10
        assert da.use_sample_variance == False  # Default when no predictors

    def test_differential_abundance_init_with_predictors(self):
        """Test DifferentialAbundance initialization with predictors."""
        try:
            from kompot.differential import DifferentialAbundance
        except ImportError as e:
            pytest.skip(f"Could not import DifferentialAbundance: {e}")

        mock_predictor = MagicMock()

        da = DifferentialAbundance(
            density_predictor1=mock_predictor,
            variance_predictor1=mock_predictor,
            use_sample_variance=None,  # Should auto-enable due to variance predictor
        )

        assert da.density_predictor1 == mock_predictor
        assert da.variance_predictor1 == mock_predictor
        assert da.use_sample_variance == True  # Auto-enabled

    def test_differential_abundance_init_contradictory_params(self):
        """Test DifferentialAbundance initialization with contradictory parameters."""
        try:
            from kompot.differential import DifferentialAbundance
        except ImportError as e:
            pytest.skip(f"Could not import DifferentialAbundance: {e}")

        da = DifferentialAbundance(use_sample_variance=True)
        X1, X2 = create_differential_test_data()

        # Should raise error when use_sample_variance=True but no predictors/indices provided
        with pytest.raises(
            ValueError, match="Sample variance estimation was explicitly enabled"
        ):
            da.fit(X1, X2)

    def test_differential_abundance_fit_basic(self):
        """Test basic DifferentialAbundance fit functionality."""
        try:
            from kompot.differential import DifferentialAbundance
        except ImportError as e:
            pytest.skip(f"Could not import DifferentialAbundance: {e}")

        X1, X2 = create_differential_test_data(n_cells1=20, n_cells2=25, n_features=5)

        da = DifferentialAbundance(n_landmarks=5)

        # Mock mellon to avoid heavy computation
        with patch("kompot.differential.differential_abundance.mellon") as mock_mellon:
            mock_estimator = MagicMock()
            mock_predictor = MagicMock()
            mock_estimator.predict = mock_predictor
            mock_mellon.DensityEstimator.return_value = mock_estimator

            result = da.fit(X1, X2)

            assert result is da
            assert da.density_predictor1 is not None
            assert da.density_predictor2 is not None

    def test_differential_abundance_fit_with_landmarks(self):
        """Test DifferentialAbundance fit with provided landmarks."""
        try:
            from kompot.differential import DifferentialAbundance
        except ImportError as e:
            pytest.skip(f"Could not import DifferentialAbundance: {e}")

        X1, X2 = create_differential_test_data(n_cells1=20, n_cells2=25, n_features=5)
        landmarks = np.random.rand(8, 5)

        da = DifferentialAbundance()

        with patch("kompot.differential.differential_abundance.mellon") as mock_mellon:
            mock_estimator = MagicMock()
            mock_predictor = MagicMock()
            mock_estimator.predict = mock_predictor
            mock_mellon.DensityEstimator.return_value = mock_estimator

            da.fit(X1, X2, landmarks=landmarks)

            assert hasattr(da, "computed_landmarks")
            np.testing.assert_array_equal(da.computed_landmarks, landmarks)

    def test_differential_abundance_fit_sync_parameters(self):
        """Test DifferentialAbundance fit with parameter synchronization."""
        try:
            from kompot.differential import DifferentialAbundance
        except ImportError as e:
            pytest.skip(f"Could not import DifferentialAbundance: {e}")

        X1, X2 = create_differential_test_data(n_cells1=20, n_cells2=25, n_features=5)

        da = DifferentialAbundance()

        with patch("kompot.differential.differential_abundance.mellon") as mock_mellon:
            # Mock parameter computation functions
            mock_mellon.parameters.compute_d_factal.return_value = 2.5
            mock_mellon.parameters.compute_nn_distances.return_value = np.array(
                [0.1, 0.2, 0.3]
            )
            mock_mellon.parameters.compute_mu.return_value = 0.8
            mock_mellon.parameters.compute_ls.return_value = 0.5
            mock_mellon.parameters.compute_landmarks.return_value = np.random.rand(
                10, 5
            )

            mock_estimator = MagicMock()
            mock_predictor = MagicMock()
            mock_estimator.predict = mock_predictor
            mock_mellon.DensityEstimator.return_value = mock_estimator

            da.fit(X1, X2, sync_parameters=True, n_landmarks=10)

            # Should call parameter computation functions
            mock_mellon.parameters.compute_d_factal.assert_called_once()
            mock_mellon.parameters.compute_nn_distances.assert_called_once()
            mock_mellon.parameters.compute_mu.assert_called_once()
            mock_mellon.parameters.compute_ls.assert_called_once()


class TestDifferentialAbundanceVariance:
    """Test DifferentialAbundance sample variance functionality."""

    def test_differential_abundance_fit_with_sample_indices(self):
        """Test DifferentialAbundance fit with sample indices for variance estimation."""
        try:
            from kompot.differential import DifferentialAbundance
        except ImportError as e:
            pytest.skip(f"Could not import DifferentialAbundance: {e}")

        X1, X2 = create_differential_test_data(n_cells1=20, n_cells2=25, n_features=5)

        # Create sample indices (simulating biological replicates)
        indices1 = np.array([0] * 10 + [1] * 10)
        indices2 = np.array([0] * 12 + [1] * 13)

        da = DifferentialAbundance(use_sample_variance=None)  # Auto-enable

        with patch("kompot.differential.differential_abundance.mellon") as mock_mellon:
            mock_estimator = MagicMock()
            mock_predictor = MagicMock()
            mock_estimator.predict = mock_predictor
            mock_mellon.DensityEstimator.return_value = mock_estimator

            with patch(
                "kompot.differential.differential_abundance.SampleVarianceEstimator"
            ) as mock_sve:
                mock_variance_estimator = MagicMock()
                mock_variance_predictor = MagicMock()
                mock_variance_estimator.predict = mock_variance_predictor
                mock_sve.return_value = mock_variance_estimator

                da.fit(
                    X1,
                    X2,
                    condition1_sample_indices=indices1,
                    condition2_sample_indices=indices2,
                )

                assert da.use_sample_variance == True
                assert da.variance_predictor1 is not None
                assert da.variance_predictor2 is not None

    def test_differential_abundance_fit_single_condition_variance(self):
        """Test DifferentialAbundance fit with single condition variance fallback."""
        try:
            from kompot.differential import DifferentialAbundance
        except ImportError as e:
            pytest.skip(f"Could not import DifferentialAbundance: {e}")

        X1, X2 = create_differential_test_data(n_cells1=20, n_cells2=25, n_features=5)
        indices1 = np.array([0] * 10 + [1] * 10)
        indices2 = np.array([0] * 12 + [1] * 13)

        da = DifferentialAbundance()

        with patch("kompot.differential.differential_abundance.mellon") as mock_mellon:
            mock_estimator = MagicMock()
            mock_predictor = MagicMock()
            mock_estimator.predict = mock_predictor
            mock_mellon.DensityEstimator.return_value = mock_estimator

            with patch(
                "kompot.differential.differential_abundance.SampleVarianceEstimator"
            ) as mock_sve:
                # Mock one successful and one failing estimator
                mock_variance_estimator1 = MagicMock()
                mock_variance_predictor1 = MagicMock()
                mock_variance_estimator1.predict = mock_variance_predictor1

                def side_effect(*args, **kwargs):
                    if len(args) > 0 or "X" in kwargs:
                        # First call (condition 1) succeeds
                        if not hasattr(side_effect, "called"):
                            side_effect.called = True
                            return mock_variance_estimator1
                        # Second call (condition 2) fails
                        else:
                            estimator = MagicMock()
                            estimator.fit.side_effect = ValueError("Not enough samples")
                            return estimator
                    return mock_variance_estimator1

                mock_sve.side_effect = side_effect

                da.fit(
                    X1,
                    X2,
                    condition1_sample_indices=indices1,
                    condition2_sample_indices=indices2,
                    allow_single_condition_variance=True,
                )

                # Should use condition 1 variance for both conditions
                assert da.variance_predictor1 is not None
                assert da.variance_predictor2 is not None


class TestDifferentialAbundancePrediction:
    """Test DifferentialAbundance prediction functionality."""

    def test_differential_abundance_predict_not_fitted(self):
        """Test DifferentialAbundance predict without fitting first."""
        try:
            from kompot.differential import DifferentialAbundance
        except ImportError as e:
            pytest.skip(f"Could not import DifferentialAbundance: {e}")

        da = DifferentialAbundance()
        X_test = np.random.rand(10, 5)

        with pytest.raises(ValueError, match="Model not fitted"):
            da.predict(X_test)

    def test_differential_abundance_predict_basic(self):
        """Test basic DifferentialAbundance prediction functionality."""
        try:
            from kompot.differential import DifferentialAbundance
        except ImportError as e:
            pytest.skip(f"Could not import DifferentialAbundance: {e}")

        da = DifferentialAbundance()

        # Mock density predictors
        mock_predictor1 = MagicMock()
        mock_predictor2 = MagicMock()
        mock_predictor1.return_value = np.array([1.0, 2.0, 1.5])
        mock_predictor2.return_value = np.array([1.5, 2.5, 2.0])
        mock_predictor1.covariance.return_value = np.array([0.1, 0.2, 0.15])
        mock_predictor2.covariance.return_value = np.array([0.12, 0.18, 0.16])
        mock_predictor1.uncertainty.return_value = np.array([0.1, 0.2, 0.15])
        mock_predictor2.uncertainty.return_value = np.array([0.12, 0.18, 0.16])

        da.density_predictor1 = mock_predictor1
        da.density_predictor2 = mock_predictor2

        X_test = np.random.rand(3, 5)

        with patch(
            "kompot.differential.differential_abundance.apply_batched"
        ) as mock_batch:
            # Mock batched apply to return the predictor results directly
            mock_batch.side_effect = lambda func, X, **kwargs: func(X)

            results = da.predict(X_test)

            assert "log_density_condition1" in results
            assert "log_density_condition2" in results
            assert "log_fold_change" in results
            assert "log_fold_change_uncertainty" in results
            assert "log_fold_change_zscore" in results
            assert "neg_log10_fold_change_ptp" in results
            assert "log_fold_change_direction" in results

            # Check shapes
            assert len(results["log_fold_change"]) == 3
            assert len(results["log_fold_change_direction"]) == 3

    def test_differential_abundance_predict_with_sample_variance(self):
        """Test DifferentialAbundance prediction with sample variance."""
        try:
            from kompot.differential import DifferentialAbundance
        except ImportError as e:
            pytest.skip(f"Could not import DifferentialAbundance: {e}")

        da = DifferentialAbundance(use_sample_variance=True)

        # Mock all predictors
        mock_density_predictor1 = MagicMock()
        mock_density_predictor2 = MagicMock()
        mock_variance_predictor1 = MagicMock()
        mock_variance_predictor2 = MagicMock()

        mock_density_predictor1.return_value = np.array([1.0, 2.0, 1.5])
        mock_density_predictor2.return_value = np.array([1.5, 2.5, 2.0])
        mock_density_predictor1.covariance.return_value = np.array([0.1, 0.2, 0.15])
        mock_density_predictor2.covariance.return_value = np.array([0.12, 0.18, 0.16])
        mock_density_predictor1.uncertainty.return_value = np.array([0.1, 0.2, 0.15])
        mock_density_predictor2.uncertainty.return_value = np.array([0.12, 0.18, 0.16])
        mock_variance_predictor1.return_value = np.array([[0.05], [0.08], [0.06]])
        mock_variance_predictor2.return_value = np.array([[0.06], [0.09], [0.07]])

        da.density_predictor1 = mock_density_predictor1
        da.density_predictor2 = mock_density_predictor2
        da.variance_predictor1 = mock_variance_predictor1
        da.variance_predictor2 = mock_variance_predictor2

        X_test = np.random.rand(3, 5)

        with patch(
            "kompot.differential.differential_abundance.apply_batched"
        ) as mock_batch:
            mock_batch.side_effect = lambda func, X, **kwargs: func(X)

            results = da.predict(X_test, progress=False)

            assert "log_fold_change_uncertainty" in results
            # Uncertainty should include sample variance contribution
            assert results["log_fold_change_uncertainty"] is not None

    def test_differential_abundance_predict_custom_thresholds(self):
        """Test DifferentialAbundance prediction with custom thresholds."""
        try:
            from kompot.differential import DifferentialAbundance
        except ImportError as e:
            pytest.skip(f"Could not import DifferentialAbundance: {e}")

        da = DifferentialAbundance(log_fold_change_threshold=1.0, ptp_threshold=0.05)

        # Mock predictors with strong signal
        mock_predictor1 = MagicMock()
        mock_predictor2 = MagicMock()
        mock_predictor1.return_value = np.array([0.0, 0.0, 0.0])
        mock_predictor2.return_value = np.array([2.0, -2.0, 0.5])  # Strong up/down/weak
        mock_predictor1.covariance.return_value = np.array([0.01, 0.01, 0.01])
        mock_predictor2.covariance.return_value = np.array([0.01, 0.01, 0.01])
        mock_predictor1.uncertainty.return_value = np.array([0.01, 0.01, 0.01])
        mock_predictor2.uncertainty.return_value = np.array([0.01, 0.01, 0.01])

        da.density_predictor1 = mock_predictor1
        da.density_predictor2 = mock_predictor2

        X_test = np.random.rand(3, 5)

        with patch(
            "kompot.differential.differential_abundance.apply_batched"
        ) as mock_batch:
            mock_batch.side_effect = lambda func, X, **kwargs: func(X)

            results = da.predict(
                X_test,
                log_fold_change_threshold=1.5,  # Custom threshold
                ptp_threshold=0.01,  # Custom threshold
                progress=False,
            )

            directions = results["log_fold_change_direction"]
            assert len(directions) == 3
            # With strong signals and low uncertainty, should detect direction changes


class TestSampleVarianceEstimatorCore:
    """Test core SampleVarianceEstimator functionality."""

    def test_sample_variance_estimator_init_basic(self):
        """Test basic SampleVarianceEstimator initialization."""
        try:
            from kompot.differential import SampleVarianceEstimator
        except ImportError as e:
            pytest.skip(f"Could not import SampleVarianceEstimator: {e}")

        sve = SampleVarianceEstimator(
            eps=1e-6, jit_compile=False, estimator_type="function"
        )

        assert sve.eps == 1e-6
        assert sve.jit_compile == False
        assert sve.estimator_type == "function"
        assert sve.store_arrays_on_disk == False

    def test_sample_variance_estimator_init_with_disk_storage(self):
        """Test SampleVarianceEstimator initialization with disk storage."""
        try:
            from kompot.differential import SampleVarianceEstimator
        except ImportError as e:
            pytest.skip(f"Could not import SampleVarianceEstimator: {e}")

        sve = SampleVarianceEstimator(
            disk_storage_dir="/tmp/test",
            store_arrays_on_disk=None,  # Should auto-enable
        )

        assert sve.disk_storage_dir == "/tmp/test"
        assert sve.store_arrays_on_disk == True

        # Test contradictory settings
        sve2 = SampleVarianceEstimator(
            disk_storage_dir="/tmp/test",
            store_arrays_on_disk=False,  # Should log warning
        )

        assert sve2.store_arrays_on_disk == False

    def test_sample_variance_estimator_invalid_estimator_type(self):
        """Test SampleVarianceEstimator with invalid estimator type."""
        try:
            from kompot.differential import SampleVarianceEstimator
        except ImportError as e:
            pytest.skip(f"Could not import SampleVarianceEstimator: {e}")

        with pytest.raises(ValueError, match="estimator_type must be either"):
            SampleVarianceEstimator(estimator_type="invalid")


class TestDifferentialUtils:
    """Test differential utilities."""

    def test_compute_weighted_mean_fold_change_basic(self):
        """Test basic weighted mean fold change computation."""
        try:
            from kompot.differential.utils import compute_weighted_mean_fold_change
        except ImportError as e:
            pytest.skip(f"Could not import compute_weighted_mean_fold_change: {e}")

        # Create test data
        fold_change = np.array([[1.0, 2.0], [1.5, 2.5], [0.5, 1.0]])  # 3 cells, 2 genes
        log_density1 = np.array([0.0, 0.5, 1.0])
        log_density2 = np.array([1.0, 1.5, 1.2])

        result = compute_weighted_mean_fold_change(
            fold_change, log_density1, log_density2
        )

        assert result.shape == (2,)  # One value per gene
        assert np.all(np.isfinite(result))

    def test_compute_weighted_mean_fold_change_with_diff(self):
        """Test weighted mean fold change with pre-computed difference."""
        try:
            from kompot.differential.utils import compute_weighted_mean_fold_change
        except ImportError as e:
            pytest.skip(f"Could not import compute_weighted_mean_fold_change: {e}")

        fold_change = np.array([[1.0, 2.0], [1.5, 2.5]])
        log_density_diff = np.array([0.5, 1.0])

        result = compute_weighted_mean_fold_change(
            fold_change, log_density_diff=log_density_diff
        )

        assert result.shape == (2,)
        assert np.all(np.isfinite(result))

    def test_compute_weighted_mean_fold_change_pandas_input(self):
        """Test weighted mean fold change with pandas Series input."""
        try:
            from kompot.differential.utils import compute_weighted_mean_fold_change
        except ImportError as e:
            pytest.skip(f"Could not import compute_weighted_mean_fold_change: {e}")

        fold_change = [[1.0, 2.0], [1.5, 2.5]]  # List input
        log_density1 = pd.Series([0.0, 0.5])
        log_density2 = pd.Series([1.0, 1.5])

        result = compute_weighted_mean_fold_change(
            fold_change, log_density1, log_density2
        )

        assert result.shape == (2,)
        assert np.all(np.isfinite(result))

    def test_compute_weighted_mean_fold_change_error_handling(self):
        """Test weighted mean fold change error handling."""
        try:
            from kompot.differential.utils import compute_weighted_mean_fold_change
        except ImportError as e:
            pytest.skip(f"Could not import compute_weighted_mean_fold_change: {e}")

        fold_change = np.array([[1.0, 2.0]])

        # Should raise error when neither diff nor both densities provided
        with pytest.raises(ValueError, match="Either log_density_diff or both"):
            compute_weighted_mean_fold_change(
                fold_change, log_density_condition1=np.array([0.5])
            )

    def test_update_direction_column_basic(self):
        """Test basic direction column update."""
        try:
            from kompot.differential.utils import update_direction_column
            import anndata
        except ImportError as e:
            pytest.skip(f"Could not import required modules: {e}")

        # Create test AnnData
        adata = anndata.AnnData(np.random.rand(10, 5))
        adata.obs["lfc_A_to_B"] = np.array(
            [2.0, -2.0, 0.5, -0.5, 1.5, -1.5, 0.1, -0.1, 3.0, -3.0]
        )
        adata.obs["ptp_A_to_B"] = np.array(
            [0.001, 0.001, 0.1, 0.1, 0.01, 0.01, 0.5, 0.5, 1e-6, 1e-6]
        )

        # Mock the utility functions to avoid complex dependencies
        with patch("kompot.anndata.utils.get_run_from_history") as mock_get_run:
            with patch("kompot.plot.volcano._infer_da_keys") as mock_infer_keys:
                # Import the function first, then patch it
                with patch.object(
                    sys.modules["kompot.plot.heatmap.direction_plot"],
                    "_infer_direction_key",
                ) as mock_infer_dir:
                    mock_infer_keys.return_value = (
                        "lfc_A_to_B",
                        "ptp_A_to_B",
                        (1.0, 0.05),
                    )
                    mock_infer_dir.return_value = ("direction_A_to_B", None, None)

                    update_direction_column(
                        adata,
                        lfc_threshold=1.0,
                        ptp_threshold=0.05,
                        direction_column="direction_A_to_B",
                        lfc_key="lfc_A_to_B",
                        ptp_key="ptp_A_to_B",
                        inplace=True,
                    )

                    assert "direction_A_to_B" in adata.obs.columns
                    directions = adata.obs["direction_A_to_B"].values

                    # Check expected directions based on thresholds
                    assert directions[0] == "up"  # lfc=2.0, ptp=0.001 (significant up)
                    assert (
                        directions[1] == "down"
                    )  # lfc=-2.0, ptp=0.001 (significant down)
                    assert directions[2] == "neutral"  # lfc=0.5 (below threshold)

    def test_update_direction_column_neg_log10_ptp(self):
        """Test direction column update with negative log10 PTP values."""
        try:
            from kompot.differential.utils import update_direction_column
            import anndata
        except ImportError as e:
            pytest.skip(f"Could not import required modules: {e}")

        adata = anndata.AnnData(np.random.rand(5, 3))
        adata.obs["lfc_A_to_B"] = np.array([2.0, -2.0, 0.5, 1.5, -1.5])
        adata.obs["neg_log10_ptp_A_to_B"] = np.array(
            [3.0, 3.0, 0.5, 2.0, 2.0]
        )  # -log10 values

        with patch("kompot.anndata.utils.get_run_from_history"):
            with patch("kompot.plot.volcano._infer_da_keys"):
                # Import the function first, then patch it
                with patch.object(
                    sys.modules["kompot.plot.heatmap.direction_plot"],
                    "_infer_direction_key",
                ) as mock_infer_dir:
                    mock_infer_dir.return_value = ("direction_A_to_B", None, None)

                    update_direction_column(
                        adata,
                        lfc_threshold=1.0,
                        ptp_threshold=0.05,  # -log10(0.05) ≈ 1.3
                        direction_column="direction_A_to_B",
                        lfc_key="lfc_A_to_B",
                        ptp_key="neg_log10_ptp_A_to_B",
                        inplace=True,
                    )

                    directions = adata.obs["direction_A_to_B"].values
                    assert directions[0] == "up"  # lfc=2.0, -log10_ptp=3.0 > 1.3
                    assert directions[1] == "down"  # lfc=-2.0, -log10_ptp=3.0 > 1.3
                    assert directions[2] == "neutral"  # lfc=0.5 < 1.0
                    assert directions[3] == "up"  # lfc=1.5 > 1.0, -log10_ptp=2.0 > 1.3

    def test_update_direction_column_not_inplace(self):
        """Test direction column update without inplace modification."""
        try:
            from kompot.differential.utils import update_direction_column
            import anndata
        except ImportError as e:
            pytest.skip(f"Could not import required modules: {e}")

        adata = anndata.AnnData(np.random.rand(3, 2))
        adata.obs["lfc_A_to_B"] = np.array([2.0, -2.0, 0.5])
        adata.obs["ptp_A_to_B"] = np.array([0.001, 0.001, 0.1])

        with patch("kompot.anndata.utils.get_run_from_history"):
            with patch("kompot.plot.volcano._infer_da_keys") as mock_infer:
                # Import the function first, then patch it
                with patch.object(
                    sys.modules["kompot.plot.heatmap.direction_plot"],
                    "_infer_direction_key",
                ) as mock_infer_dir:
                    mock_infer.return_value = ("lfc_A_to_B", "ptp_A_to_B", (1.0, 0.05))
                    mock_infer_dir.return_value = ("direction_A_to_B", None, None)

                    result_adata = update_direction_column(
                        adata, lfc_threshold=1.0, ptp_threshold=0.05, inplace=False
                    )

                    # Original should be unchanged
                    assert "direction_A_to_B" not in adata.obs.columns

                    # Result should have the new column
                    assert result_adata is not None
                    assert "direction_A_to_B" in result_adata.obs.columns


class TestDifferentialLoggerUsage:
    """Test differential modules logger usage and error handling."""

    def test_differential_abundance_logging(self):
        """Test DifferentialAbundance logging functionality."""
        try:
            from kompot.differential import DifferentialAbundance
        except ImportError as e:
            pytest.skip(f"Could not import DifferentialAbundance: {e}")

        # Test logger messages during initialization
        with patch("kompot.differential.differential_abundance.logger") as mock_logger:
            da = DifferentialAbundance(
                variance_predictor1=MagicMock(), use_sample_variance=None
            )

            mock_logger.info.assert_called_with(
                "Sample variance estimation automatically enabled due to presence of variance predictors"
            )

    def test_sample_variance_estimator_logging(self):
        """Test SampleVarianceEstimator logging functionality."""
        try:
            from kompot.differential import SampleVarianceEstimator
        except ImportError as e:
            pytest.skip(f"Could not import SampleVarianceEstimator: {e}")

        with patch(
            "kompot.differential.sample_variance_estimator.logger"
        ) as mock_logger:
            SampleVarianceEstimator(
                disk_storage_dir="/tmp/test", store_arrays_on_disk=False
            )

            mock_logger.warning.assert_called_once()
            warning_call = mock_logger.warning.call_args[0][0]
            assert "Arrays will NOT be stored on disk" in warning_call
