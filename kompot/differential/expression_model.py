"""Single-condition GP expression model."""

import numpy as np
import logging
from typing import Optional, Any

import mellon
from mellon.base_cov import Covariance
from mellon.cov import Matern52

from ..batch_utils import apply_batched
from .sample_variance_estimator import SampleVarianceEstimator

logger = logging.getLogger("kompot")


class CenteredLinear(Covariance):
    """Linear kernel with built-in centering.

    Computes ``k(x, y) = (x - mu) . (y - mu) / ls`` where ``mu`` is the
    training-data centroid.  This makes the kernel shift-invariant: adding
    a constant offset to all coordinates does not change the kernel matrix.

    Without centering the linear kernel models ``f(x) = beta . x`` (through
    the origin), so with noise the GP regularises towards zero everywhere,
    which is wrong when the data centroid is far from the origin.  Centering
    gives ``f(x) = beta . (x - mu)``, equivalent to linear regression with
    a free intercept where only the slope is regularised.

    Parameters
    ----------
    ls : float
        Amplitude scaling (divides the dot product).
    mu : array-like
        Centroid to subtract, shape ``(n_features,)``.
    active_dims : array-like, optional
        Indices of active input dimensions.
    """

    def __init__(self, ls=1.0, mu=None, active_dims=None):
        super().__init__(active_dims=active_dims)
        self.ls = ls
        self.mu = np.asarray(mu) if mu is not None else None

    def k(self, x, y):
        from jax.numpy import einsum
        from mellon.base_cov import select_active_dims
        x = select_active_dims(x, self.active_dims)
        y = select_active_dims(y, self.active_dims)
        if self.mu is not None:
            x = x - self.mu
            y = y - self.mu
        return einsum("ij,kj->ik", x, y) / self.ls

    def diag(self, x):
        from jax.numpy import sum as jsum
        from mellon.base_cov import select_active_dims
        x = select_active_dims(x, self.active_dims)
        if self.mu is not None:
            x = x - self.mu
        return jsum(x * x, axis=-1) / self.ls


def _build_matern52_linear(X, ls):
    """Build a Matern-5/2 + CenteredLinear covariance function.

    The Linear component centers coordinates by the training-data centroid
    and normalises its amplitude by ``mean(||x - mu||^2)`` so that its
    diagonal contribution is ~1.0 on average, matching the Matern-5/2.
    This decouples the Matern length-scale from the Linear amplitude
    and makes the kernel shift-invariant.

    Parameters
    ----------
    X : np.ndarray
        Training coordinates, used to compute the centroid and normalisation.
    ls : float
        Length scale for the Matern-5/2 component.

    Returns
    -------
    cov_func : mellon.Covariance
        ``Matern52(ls) + CenteredLinear(ls=mean(||x-mu||^2), mu=mu)``.
    """
    X = np.asarray(X)
    mu = X.mean(axis=0)
    ls_linear = float(np.mean(np.sum((X - mu) ** 2, axis=1)))
    if ls_linear == 0:
        ls_linear = 1.0  # degenerate case: constant coordinates
    cov_func = Matern52(ls=ls) + CenteredLinear(ls=ls_linear, mu=mu)
    logger.debug(
        "Matern52+CenteredLinear: ls_matern=%.4f, ls_linear=%.4f "
        "(mean||x-mu||^2)",
        ls, ls_linear,
    )
    return cov_func



class ExpressionModel:
    """Single-condition GP expression model.

    Encapsulates one fitted Gaussian Process for gene expression, along with
    optional empirical (aleatoric) variance and sample variance components.
    Can be used standalone for imputation or paired inside
    :class:`DifferentialExpression` for two-condition comparisons.

    Parameters
    ----------
    n_landmarks : int, optional
        Number of landmarks for Nystrom approximation.
    use_empirical_variance : bool
        Whether to estimate per-gene empirical variance from GP residuals.
    eps : float
        Small constant for numerical stability.
    random_state : int, optional
        Random seed for landmark selection.
    batch_size : int
        Batch size for prediction.
    store_arrays_on_disk : bool, optional
        Whether to store large arrays on disk.
    disk_storage_dir : str, optional
        Directory for disk-backed arrays.
    function_predictor : callable, optional
        Pre-fitted mellon Predictor (skips ``fit()``).
    obs_variance_predictor : callable, optional
        Pre-fitted empirical (aleatoric) variance predictor.
        When provided, the internal empirical-variance computation is
        skipped during ``fit()``.
    variance_predictor : callable, optional
        Pre-fitted sample variance predictor.
    """

    def __init__(
        self,
        n_landmarks: Optional[int] = None,
        use_empirical_variance: bool = True,
        eps: float = 1e-8,
        random_state: Optional[int] = None,
        batch_size: int = 500,
        store_arrays_on_disk: Optional[bool] = None,
        disk_storage_dir: Optional[str] = None,
        function_predictor: Optional[Any] = None,
        obs_variance_predictor: Optional[Any] = None,
        variance_predictor: Optional[Any] = None,
    ):
        self.n_landmarks = n_landmarks
        self.use_empirical_variance = use_empirical_variance
        self.eps = eps
        self.random_state = random_state
        self.batch_size = batch_size

        if store_arrays_on_disk is None:
            self.store_arrays_on_disk = disk_storage_dir is not None
        else:
            self.store_arrays_on_disk = store_arrays_on_disk
        self.disk_storage_dir = disk_storage_dir

        # Internal state
        self._estimator = None
        self._predictor = function_predictor
        self._sample_variance_predictor = variance_predictor
        self._within_sample_obs_var_predictor = obs_variance_predictor
        self._sigma = None

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sigma: float = 1.0,
        ls: Optional[float] = None,
        ls_factor: float = 10.0,
        landmarks: Optional[np.ndarray] = None,
        sample_indices: Optional[np.ndarray] = None,
        sample_estimator_ls: Optional[float] = None,
        allow_single_condition_variance: bool = False,
        **function_kwargs,
    ) -> "ExpressionModel":
        """Fit the expression GP on one condition.

        Parameters
        ----------
        X : np.ndarray
            Cell-state coordinates, shape ``(n_cells, n_features)``.
        y : np.ndarray
            Expression matrix, shape ``(n_cells, n_genes)``.
        sigma : float
            Noise level for the GP.
        ls : float, optional
            Length scale. If *None*, estimated with ``ls_factor``.
        ls_factor : float
            Multiplier applied to the automatically inferred length scale.
        landmarks : np.ndarray, optional
            Pre-computed landmarks (e.g. shared across conditions).
        sample_indices : np.ndarray, optional
            Per-cell sample labels for biological-replicate variance.
        sample_estimator_ls : float, optional
            Length scale override for the sample-variance estimator.
        allow_single_condition_variance : bool
            Passed through to :class:`SampleVarianceEstimator`.
        **function_kwargs
            Forwarded to :class:`mellon.FunctionEstimator`.

        Returns
        -------
        ExpressionModel
            ``self``, for chaining.
        """
        self._sigma = sigma

        # ---- build estimator kwargs ----
        estimator_defaults = {
            "sigma": sigma,
            "predictor_with_uncertainty": True,
            "n_landmarks": self.n_landmarks,
        }

        mellon_kwargs = {
            k: v for k, v in function_kwargs.items() if k not in ("progress",)
        }
        estimator_defaults.update(mellon_kwargs)

        if landmarks is not None:
            estimator_defaults["landmarks"] = landmarks
            estimator_defaults["gp_type"] = "fixed"
        elif self.n_landmarks is not None and self.n_landmarks > 0:
            from mellon.parameters import compute_landmarks

            computed = compute_landmarks(
                X,
                gp_type="fixed",
                n_landmarks=self.n_landmarks,
                random_state=self.random_state,
            )
            estimator_defaults["landmarks"] = computed
            estimator_defaults["gp_type"] = "fixed"

        # ---- covariance function ----
        # Unless the caller provides an explicit cov_func or cov_func_curry,
        # build a Matern52 kernel.
        if "cov_func" not in estimator_defaults and \
                "cov_func_curry" not in estimator_defaults:
            from mellon.parameters import compute_nn_distances, compute_ls

            if ls is not None:
                _ls = ls
            else:
                _nn = compute_nn_distances(X)
                _ls = compute_ls(_nn) * ls_factor
            estimator_defaults["cov_func"] = Matern52(ls=_ls)
            estimator_defaults["ls"] = _ls
        else:
            if ls is not None:
                estimator_defaults["ls"] = ls
            else:
                estimator_defaults["ls_factor"] = ls_factor

        # Skip empirical variance computation if an obs_variance_predictor
        # was injected at construction time.
        obs_var_injected = self._within_sample_obs_var_predictor is not None

        # When sample_indices are provided, obs_variance is computed per
        # sample to avoid double-counting the between-sample variance that
        # the SampleVarianceEstimator already captures.
        per_sample_empirical = (
            self.use_empirical_variance
            and sample_indices is not None
            and not obs_var_injected
        )

        if self.use_empirical_variance and not per_sample_empirical and not obs_var_injected:
            estimator_defaults["obs_variance"] = True

        # ---- fit mellon GP ----
        logger.debug("Fitting expression estimator...")
        self._estimator = mellon.FunctionEstimator(**estimator_defaults)
        self._estimator.fit(X, y)
        self._predictor = self._estimator.predict

        # ---- empirical variance validation ----
        if self.use_empirical_variance and not per_sample_empirical and not obs_var_injected:
            if not hasattr(self._predictor, "obs_variance"):
                raise ValueError(
                    "use_empirical_variance=True requires predictors fitted with "
                    "obs_variance=True."
                )

        # ---- sample variance ----
        if sample_indices is not None:
            sample_kw = estimator_defaults.copy()
            if sample_estimator_ls is not None:
                sample_kw["ls"] = sample_estimator_ls
                # Rebuild kernel with the overridden ls when using built-in
                if "cov_func" in sample_kw and \
                        "cov_func_curry" not in estimator_defaults:
                    sample_kw["cov_func"] = Matern52(ls=sample_estimator_ls)
            else:
                try:
                    sample_kw["ls"] = self.ls
                except (AttributeError, TypeError) as e:
                    logger.warning(
                        f"Could not extract ls for sample variance: {e}"
                    )

            try:
                sve = SampleVarianceEstimator(
                    eps=self.eps,
                    store_arrays_on_disk=self.store_arrays_on_disk,
                    disk_storage_dir=self.disk_storage_dir,
                )
                sve._called_from_differential = True
                sve.fit(
                    X=X,
                    Y=y,
                    grouping_vector=sample_indices,
                    ls_factor=ls_factor,
                    estimator_kwargs=sample_kw,
                )
                self._sample_variance_predictor = sve.predict
            except ValueError:
                if not allow_single_condition_variance:
                    raise
                logger.info(
                    "Sample variance estimation failed for this condition."
                )
                self._sample_variance_predictor = None

        # ---- per-sample empirical variance ----
        # Skipped when an obs_variance_predictor was injected.
        # Fit per-sample GPs and use their loo_residuals_squared to get
        # within-sample LOO-corrected squared residuals.  These are residuals
        # relative to each sample's own GP mean, so they capture only
        # within-sample noise without between-sample variance.  The pooled
        # squared residuals are then smoothed with a single variance GP.
        if per_sample_empirical:
            unique_samples = np.unique(sample_indices)
            logger.info(
                f"Computing per-sample empirical variance "
                f"({len(unique_samples)} samples)..."
            )
            per_sample_kw = estimator_defaults.copy()
            per_sample_kw.pop("obs_variance", None)

            # Collect LOO-corrected squared residuals from per-sample GPs
            all_loo_sq = np.empty_like(y)
            for sid in unique_samples:
                mask = sample_indices == sid
                sample_est = mellon.FunctionEstimator(**per_sample_kw)
                sample_est.fit(X[mask], y[mask])
                all_loo_sq[mask] = sample_est.predict.loo_residuals_squared(
                    X[mask], y[mask],
                )

            # Smooth pooled within-sample squared residuals with a single GP
            var_kw = estimator_defaults.copy()
            var_kw.pop("obs_variance", None)
            var_est = mellon.FunctionEstimator(**var_kw)
            var_est.fit(X, all_loo_sq)
            self._within_sample_obs_var_predictor = var_est.predict

        return self

    # ------------------------------------------------------------------
    # Core accessors
    # ------------------------------------------------------------------

    @property
    def predictor(self):
        """The underlying mellon Predictor."""
        return self._predictor

    def predict(self, X, batch_size=None, progress=True):
        """Imputed expression, shape ``(n_points, n_genes)``.

        Parameters
        ----------
        X : np.ndarray
            Evaluation points.
        batch_size : int, optional
            Override instance batch size.
        progress : bool
            Show progress bar.
        """
        if self._predictor is None:
            raise ValueError("Model not fitted. Call fit() first.")

        bs = batch_size if batch_size is not None else self.batch_size
        return apply_batched(
            self._predictor,
            X,
            batch_size=bs,
            show_progress=progress,
            desc="Predicting expression" if progress else None,
        )

    def covariance(self, X, diag=True, batch_size=None, progress=True):
        """GP posterior covariance (epistemic uncertainty).

        Parameters
        ----------
        X : np.ndarray
            Evaluation points.
        diag : bool
            If True return diagonal only.
        batch_size : int, optional
            Override instance batch size.
        progress : bool
            Show progress bar.
        """
        if self._predictor is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if diag:
            bs = batch_size if batch_size is not None else self.batch_size
            result = apply_batched(
                lambda Xb: self._predictor.covariance(Xb, diag=True),
                X,
                batch_size=bs,
                show_progress=progress,
                desc="Computing GP covariance (diag)" if progress else None,
            )
        else:
            result = self._predictor.covariance(X, diag=False)
        return np.asarray(result)

    def obs_variance(self, X, batch_size=None, progress=True):
        """Smoothed aleatoric noise, shape ``(n_points, n_genes)``.

        Returns scalar ``0`` when empirical variance is disabled.
        When per-sample predictors are available (fitted with
        ``sample_indices``), returns the mean of per-sample estimates
        to avoid double-counting between-sample variance.
        """
        if not self.has_empirical_variance:
            return 0

        bs = batch_size if batch_size is not None else self.batch_size
        func = self._obs_variance_func
        return np.maximum(
            apply_batched(
                func,
                X,
                batch_size=bs,
                show_progress=progress,
                desc="Computing obs_variance" if progress else None,
            ),
            self.eps,
        )

    @property
    def _obs_variance_func(self):
        """Return the callable for empirical (obs) variance."""
        if self._within_sample_obs_var_predictor is not None:
            return self._within_sample_obs_var_predictor
        if self._predictor is not None and hasattr(self._predictor, "obs_variance"):
            return self._predictor.obs_variance
        return None

    def sample_variance(self, X, diag=True, batch_size=None, progress=True):
        """Biological-replicate variance.

        Returns scalar ``0`` when no sample indices were provided.
        """
        if not self.has_sample_variance:
            return 0
        if diag:
            bs = batch_size if batch_size is not None else self.batch_size
            return apply_batched(
                lambda Xb: self._sample_variance_predictor(Xb, diag=True),
                X,
                batch_size=bs,
                show_progress=progress,
                desc="Computing sample variance" if progress else None,
            )
        else:
            return self._sample_variance_predictor(X, diag=False, progress=progress)

    def total_variance(self, X, diag=True, batch_size=None, progress=True):
        """Sum of GP posterior, obs_variance, and sample variance (diagonal)."""
        cov = self.covariance(X, diag=True, batch_size=batch_size, progress=progress)
        # Ensure cov broadcasts with per-gene arrays: (n,) -> (n, 1)
        if isinstance(cov, np.ndarray) and cov.ndim == 1:
            cov = cov[:, np.newaxis]
        obs = self.obs_variance(X, batch_size=batch_size, progress=progress)
        sam = self.sample_variance(X, diag=True, batch_size=batch_size, progress=progress)
        return cov + obs + sam

    def std(self, X, batch_size=None, progress=True):
        """Square root of total variance plus eps."""
        return np.sqrt(
            self.total_variance(X, diag=True, batch_size=batch_size, progress=progress)
            + self.eps
        )

    # ------------------------------------------------------------------
    # Metadata properties
    # ------------------------------------------------------------------

    @property
    def landmarks(self):
        """Landmarks used by the GP, or *None*."""
        if self._predictor is not None and hasattr(self._predictor, "landmarks"):
            return self._predictor.landmarks
        if self._estimator is not None and hasattr(self._estimator, "landmarks"):
            return self._estimator.landmarks
        return None

    @property
    def ls(self):
        """Length scale of the GP kernel (Matern component)."""
        if self._predictor is not None and hasattr(self._predictor, "cov_func"):
            cf = self._predictor.cov_func
            if hasattr(cf, "ls"):
                return cf.ls
            # Combined kernel (Add): extract from the left (Matern) component
            if hasattr(cf, "left") and hasattr(cf.left, "ls"):
                return cf.left.ls
        return None

    @property
    def cov_func(self):
        """Covariance function of the GP."""
        if self._predictor is not None and hasattr(self._predictor, "cov_func"):
            return self._predictor.cov_func
        return None

    @property
    def sigma(self):
        """Noise level used during fitting."""
        return self._sigma

    @property
    def has_sample_variance(self) -> bool:
        """Whether sample variance is available."""
        return self._sample_variance_predictor is not None

    @property
    def has_empirical_variance(self) -> bool:
        """Whether empirical (obs) variance is available."""
        if not self.use_empirical_variance:
            return False
        if self._within_sample_obs_var_predictor is not None:
            return True
        return (
            self._predictor is not None
            and hasattr(self._predictor, "obs_variance")
        )
