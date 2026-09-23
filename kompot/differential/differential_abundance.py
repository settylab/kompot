"""Differential abundance analysis for cell density comparison."""

import numpy as np
from typing import Optional, Dict, Any
import logging
from scipy.stats import norm as normal

import mellon
from mellon.parameters import compute_landmarks

from ..batch_utils import apply_batched
from ..settings import PARAM_SCHEMES, resolve_param_scheme
from .sample_variance_estimator import SampleVarianceEstimator

logger = logging.getLogger("kompot")

#: The ``param_scheme`` an abundance fit uses when none is given.
DEFAULT_PARAM_SCHEME = "separate"

#: The density hyperparameters a ``param_scheme`` covers.
_DENSITY_PARAMS = ("d", "mu", "ls")

# ``compute_d_factal`` averages local dimensionality over at most this many
# query cells (mellon ``parameters.py``); the ``"symmetric"`` scheme weights
# each condition's ``d`` by the number of cells it actually averaged over.
_D_FACTAL_QUERY_CELLS = 500


def _condition_density_params(X, ls_factor, seed, d=None):
    """``d``, ``mu`` and ``ls`` exactly as a ``mellon.DensityEstimator`` derives them.

    Mirrors ``DensityEstimator._compute_d`` (``d_method="fractal"``),
    ``_compute_nn_distances``, ``_compute_mu`` and ``_compute_ls``, so that a
    value estimated here *before* either estimator is fitted is the value the
    estimator would have reached on its own.  Returns the validated
    nearest-neighbour distances as well, for ``"symmetric"``.  ``d``, when
    given, is a pinned value and replaces the fractal estimate everywhere.
    """
    from mellon.validation import validate_nn_distances

    X = np.asarray(X)
    nn = validate_nn_distances(mellon.parameters.compute_nn_distances(X, seed=seed))
    # A pinned ``d`` is what the estimator will use, so ``mu`` is derived at it
    # -- never at the fractal estimate the pin replaces.
    if d is None:
        d = mellon.parameters.compute_d_factal(X)
    mu = mellon.parameters.compute_mu(nn, d)
    ls = mellon.parameters.compute_ls(nn) * ls_factor
    return {"d": d, "mu": mu, "ls": ls}, nn


def _resolve_scheme_density_params(
    param_scheme, X_condition1, X_condition2, ls_factor, seed, d=None
):
    """Return the shared ``{d, mu, ls}`` a one-sided or symmetric scheme prescribes.

    ``"separate"`` and ``"pooled"`` are handled by the caller; this covers the
    three schemes that estimate from each condition on its own.
    """
    if param_scheme == "condition1":
        return _condition_density_params(X_condition1, ls_factor, seed, d)[0]
    if param_scheme == "condition2":
        return _condition_density_params(X_condition2, ls_factor, seed, d)[0]
    if param_scheme == "symmetric":
        # Each shared value is the model's own estimator applied to the two
        # conditions' WITHIN-condition statistics pooled together -- never to
        # the stacked union, whose row order depends on the orientation.  Every
        # combination below is written so that exchanging the conditions
        # exchanges two commutative terms, which makes the result bit-identical
        # under a swap by construction, not merely close.
        p1, nn1 = _condition_density_params(X_condition1, ls_factor, seed, d)
        p2, nn2 = _condition_density_params(X_condition2, ls_factor, seed, d)
        n1, n2 = np.asarray(X_condition1).shape[0], np.asarray(X_condition2).shape[0]
        # ls: geometric mean of nn distances pooled == size-weighted geometric
        # mean of the per-condition values (the differential-expression rule).
        ls = float(
            np.exp((n1 * np.log(p1["ls"]) + n2 * np.log(p2["ls"])) / (n1 + n2))
        )
        # d: mean local dimensionality pooled over the query cells each
        # condition's estimate averaged.
        if d is None:
            q1 = min(n1, _D_FACTAL_QUERY_CELLS)
            q2 = min(n2, _D_FACTAL_QUERY_CELLS)
            d = float((q1 * p1["d"] + q2 * p2["d"]) / (q1 + q2))
        # mu: the 1st-percentile rule over the pooled within-condition nn
        # distances, at the shared d.  A quantile sorts, so the order the two
        # conditions are concatenated in does not reach the result.
        mu = mellon.parameters.compute_mu(np.concatenate([nn1, nn2]), d)
        return {"d": d, "mu": mu, "ls": ls}
    raise ValueError(
        f"Unknown param_scheme {param_scheme!r}. Expected one of {PARAM_SCHEMES}."
    )


class DifferentialAbundance:
    """
    Compute differential abundance between two conditions.

    This class analyzes the differences in cell density between two conditions
    (e.g., control to treatment) using density estimation and fold change analysis.

    Whether the two conditions' density estimators share their ``d``, ``mu``
    and ``ls``, and which cells the shared values come from, is set by
    ``param_scheme`` in :meth:`fit`.

    Attributes
    ----------
    log_density_condition1 : np.ndarray
        Log density values for the first condition.
    log_density_condition2 : np.ndarray
        Log density values for the second condition.
    log_fold_change : np.ndarray
        Log fold change between conditions (condition2 - condition1).
    log_fold_change_uncertainty : np.ndarray
        Uncertainty in the log fold change estimates.
    log_fold_change_zscore : np.ndarray
        Z-scores for the log fold changes.
    log_fold_change_ptp : np.ndarray
        PTP (Posterior Tail Probability) for the log fold changes. The PTP is the significance measure similar to p-value.
    log_fold_change_direction : np.ndarray
        Direction of change ('up', 'down', or 'neutral') based on thresholds.

    Methods
    -------
    fit(X_condition1, X_condition2, param_scheme=None, **density_kwargs)
        Fit density estimators for both conditions, optionally sharing parameters.
    predict(X_new)
        Predict log density and log fold change for new points.
    """

    def __init__(
        self,
        log_fold_change_threshold: float = 1.0,
        ptp_threshold: float = 0.05,
        n_landmarks: Optional[int] = None,
        use_sample_variance: Optional[bool] = None,
        eps: float = 1e-12,
        jit_compile: bool = False,
        density_predictor1: Optional[Any] = None,
        density_predictor2: Optional[Any] = None,
        variance_predictor1: Optional[Any] = None,
        variance_predictor2: Optional[Any] = None,
        random_state: Optional[int] = None,
        batch_size: Optional[int] = None,
    ):
        """
        Initialize DifferentialAbundance.

        Parameters
        ----------
        log_fold_change_threshold : float, optional
            Threshold for considering a log fold change significant, by default 1.0.
        ptp_threshold : float, optional
            Threshold for considering a PTP significant, by default 1e-2.
        n_landmarks : int, optional
            Number of landmarks to use for approximation. If None, use all points, by default None.
            At or above the number of cells in both conditions together, no landmarks
            are built and each condition gets its full GP. That equals None up to 5000 cells
            per condition; above that, None leaves mellon at its default of 5000 landmarks.
        use_sample_variance : bool, optional
            Whether to use sample variance for uncertainty estimation. By default None.
            - If None (recommended): Automatically determined based on variance_predictor1/2
              or whether sample indices are provided in fit().
            - If True: Force use of sample variance (even if no predictors/indices available).
            - If False: Disable sample variance (even if predictors/indices are available).
        eps : float, optional
            Small constant for numerical stability, by default 1e-12.
        jit_compile : bool, optional
            Whether to use JAX just-in-time compilation, by default False.
        density_predictor1 : Any, optional
            Precomputed density predictor for condition 1, typically from DensityEstimator.predict
        density_predictor2 : Any, optional
            Precomputed density predictor for condition 2, typically from DensityEstimator.predict
        variance_predictor1 : Any, optional
            Precomputed variance predictor for condition 1. If provided, will be used for uncertainty calculation
            and will automatically enable sample variance calculation (unless explicitly disabled).
        variance_predictor2 : Any, optional
            Precomputed variance predictor for condition 2. If provided, will be used for uncertainty calculation
            and will automatically enable sample variance calculation (unless explicitly disabled).
        random_state : int, optional
            Random seed for reproducible landmark selection when n_landmarks is specified.
            Controls the random selection of points when using approximation, by default None.
        batch_size : int, optional
            Number of samples to process at once during prediction to manage memory usage.
            If None or 0, all samples will be processed at once. If processing all at once
            causes a memory error, a default batch size of 500 will be used automatically.
            Default is None.
        """
        self.log_fold_change_threshold = log_fold_change_threshold
        self.ptp_threshold = ptp_threshold
        self.n_landmarks = n_landmarks
        self.eps = eps
        self.jit_compile = jit_compile
        self.random_state = random_state
        self.batch_size = batch_size

        # Store whether user explicitly set use_sample_variance
        self.use_sample_variance_explicit = use_sample_variance is not None

        # Set use_sample_variance based on variance predictors
        # If variance predictors are provided, automatically use sample variance unless explicitly disabled
        if use_sample_variance is None:
            self.use_sample_variance = (
                variance_predictor1 is not None or variance_predictor2 is not None
            )
            if self.use_sample_variance:
                logger.info(
                    "Sample variance estimation automatically enabled due to presence of variance predictors"
                )
        else:
            self.use_sample_variance = use_sample_variance

        # These will be populated after fitting
        self.log_density_condition1 = None
        self.log_density_condition2 = None
        self.log_density_uncertainty_condition1 = None
        self.log_density_uncertainty_condition2 = None
        self.log_fold_change = None
        self.log_fold_change_uncertainty = None
        self.log_fold_change_zscore = None
        self.log_fold_change_ptp = None
        self.log_fold_change_direction = None

        # Density estimators or predictors
        self.density_predictor1 = density_predictor1
        self.density_predictor2 = density_predictor2

        # Variance predictors
        self.variance_predictor1 = variance_predictor1
        self.variance_predictor2 = variance_predictor2

    def fit(
        self,
        X_condition1: np.ndarray,
        X_condition2: np.ndarray,
        landmarks: Optional[np.ndarray] = None,
        ls_factor: float = 10.0,
        condition1_sample_indices: Optional[np.ndarray] = None,
        condition2_sample_indices: Optional[np.ndarray] = None,
        sample_estimator_ls: Optional[float] = None,
        param_scheme: Optional[str] = None,
        allow_single_condition_variance: bool = False,
        **density_kwargs,
    ):
        """
        Fit density estimators for both conditions.

        This method only creates the estimators and does not compute fold changes.
        Call predict() to compute fold changes on any set of points.

        Parameters
        ----------
        X_condition1 : np.ndarray
            Cell states for the first condition. Shape (n_cells, n_features).
        X_condition2 : np.ndarray
            Cell states for the second condition. Shape (n_cells, n_features).
        landmarks : np.ndarray, optional
            Pre-computed landmarks to use. If provided, n_landmarks will be ignored.
            Shape (n_landmarks, n_features).

            Automatic landmarks are computed from ``np.vstack([X_condition1,
            X_condition2])``, whose **row order differs between the two
            orientations** of a contrast, so ``da(X, Y)`` and ``da(Y, X)`` select
            different landmark sets and their results stop being exact mirrors of
            one another. The uncertainty is hit hardest, because mellon's
            Laplace uncertainty depends on the order of the landmark rows, not
            only on the set. Passing one array here to both runs removes that; so
            does ``n_landmarks=None``, or any ``n_landmarks`` at least the number
            of cells, which build no landmarks at all. With
            ``random_state=None`` (the default) the selection is not even
            reproducible between two runs of the same orientation. This is
            independent of ``param_scheme``: see there for the other half.
        ls_factor : float, optional
            Multiplication factor to apply to length scale when it's automatically inferred,
            by default 10.0. Only used when ls is not explicitly provided in density_kwargs.
        condition1_sample_indices : np.ndarray, optional
            Sample indices for first condition. Used for sample variance estimation.
            Unique values in this array define different sample groups.
        condition2_sample_indices : np.ndarray, optional
            Sample indices for second condition. Used for sample variance estimation.
            Unique values in this array define different sample groups.
        sample_estimator_ls : float, optional
            Length scale for the sample-specific variance estimators. If None, will use
            the same value as ls or it will be estimated, by default None.
        param_scheme : str, optional
            Where the density hyperparameters ``d``, ``mu`` and ``ls`` come from.
            A value passed explicitly through ``density_kwargs`` (``d=``,
            ``mu=``, ``ls=``) pins that parameter and takes precedence. ``None``
            (default) means ``"separate"``.

            * ``"separate"`` (default) — each density estimator derives its own
              ``d``, ``mu`` and ``ls`` from its own condition; nothing is shared.
            * ``"pooled"`` — estimate all three once from both conditions' cells
              taken together and share them.
            * ``"condition1"`` / ``"condition2"`` — estimate all three from one
              condition's cells, exactly as that condition's own estimator
              would, and share them. As asymmetric as the names say.
            * ``"symmetric"`` — estimate from each condition separately and share
              the pooled estimate: ``ls`` is the size-weighted geometric mean of
              the two, ``d`` the mean local dimensionality over both
              conditions' query cells, and ``mu`` the
              1st-percentile rule over both conditions' within-condition
              nearest-neighbour distances at that ``d``. Bit-identical under
              swapping the conditions.

            A shared ``d`` is always the fractal estimate; pass ``d`` to use
            another.

            **Order dependence.** ``"pooled"`` works on
            ``np.vstack([X_condition1, X_condition2])``, so the parameters it
            derives depend on the order the conditions were passed in, and
            ``da(X, Y)`` and ``da(Y, X)`` stop agreeing. ``mu`` and ``ls`` come
            from the union's nearest-neighbour distances; ``d`` additionally
            subsamples 500 indices once the **combined** cell count is **above**
            500, and so becomes order-dependent there too.

            To make the two orientations agree under ``"pooled"``, pass ``d``,
            ``mu`` and ``ls`` explicitly through ``density_kwargs``. **Two
            remedies are needed and neither substitutes for the other**: passing
            all three fixes the parameter half, and it leaves an uncertainty
            discrepancy behind whenever landmarks are still chosen
            automatically — so pass one ``landmarks`` array to both runs as well
            (or use ``n_landmarks=None``). Measured on 400 + 400 cells: all
            three parameters with automatic landmarks still differs between
            orientations, while all three with a shared landmark array agrees
            exactly. Below 500 combined cells ``mu`` and ``ls`` alone suffice
            for the parameter half, but passing ``d`` as well is always safe.

            ``"separate"``, ``"symmetric"`` and the ``"condition1"`` /
            ``"condition2"`` mirror never touch the union, so only the landmark
            half applies to them: ``da(X, Y, "condition1")`` and
            ``da(Y, X, "condition2")`` are the same computation with the labels
            exchanged when both runs use the same landmarks.
        allow_single_condition_variance : bool, optional
            Allow sample-variance estimation when only one condition has
            multiple samples, by default False.
        **density_kwargs : dict
            Additional arguments to pass to the DensityEstimator.

        Returns
        -------
        self
            The fitted instance.
        """

        param_scheme = resolve_param_scheme(param_scheme, DEFAULT_PARAM_SCHEME)

        # Create or use density predictors
        if self.density_predictor1 is None or self.density_predictor2 is None:
            # Configure density estimator defaults
            estimator_defaults = {
                "d_method": "fractal",
                "predictor_with_uncertainty": True,
            }

            # Add ls_factor to estimator_defaults if ls is not already specified
            if "ls" not in density_kwargs:
                estimator_defaults["ls_factor"] = ls_factor

            # Update defaults with user-provided values (user-provided settings will override ls_factor if ls is specified)
            estimator_defaults.update(density_kwargs)

            # Use provided landmarks if available, otherwise compute them if requested.
            # gp_type="fixed" is set on the estimator so the landmarks act as fixed
            # inducing points for both per-condition density estimators — i.e. shared
            # across conditions, mirroring DifferentialExpression. Without this, mellon's
            # auto-select picks FULL whenever a condition has fewer cells than the
            # shared landmark grid, silently discarding the cross-condition sharing.
            if landmarks is not None:
                logger.info(f"Using provided landmarks with shape {landmarks.shape}")
                estimator_defaults["landmarks"] = landmarks
                estimator_defaults["gp_type"] = "fixed"
                # Store provided landmarks for future use
                self.computed_landmarks = landmarks
            elif self.n_landmarks is not None and self.n_landmarks >= (
                len(X_condition1) + len(X_condition2)
            ):
                # As many landmarks as cells would make the stacked union itself
                # the landmark array, ordered by which condition was passed first.
                # mellon's Laplace uncertainty depends on the landmarks' row order,
                # so da(X, Y) and da(Y, X) then disagree on the uncertainty
                # (settylab/kompot#32). No approximation was asked for: fit each
                # condition's exact GP, on its own cells (n_landmarks=0 to mellon; the
                # same as n_landmarks=None only up to mellon's 5000-landmark default).
                logger.info(
                    f"n_landmarks={self.n_landmarks:,} >= {len(X_condition1) + len(X_condition2):,} "
                    "cells: fitting each condition's full GP instead of landmarks."
                )
                estimator_defaults.setdefault("n_landmarks", 0)
            elif self.n_landmarks is not None:
                # Use mellon's compute_landmarks function to get properly distributed landmarks.
                X_combined = np.vstack([X_condition1, X_condition2])
                computed_landmarks = compute_landmarks(
                    X_combined,
                    gp_type="fixed",
                    n_landmarks=self.n_landmarks,
                    random_state=self.random_state,
                )
                estimator_defaults["landmarks"] = computed_landmarks
                estimator_defaults["gp_type"] = "fixed"
                # Store computed landmarks for future use
                self.computed_landmarks = computed_landmarks

            # "pooled": estimate d / mu / ls from the stacked union and share.
            if param_scheme == "pooled":
                # Combine data from both conditions for parameter estimation
                X_combined = np.vstack([X_condition1, X_condition2])
                logger.info(
                    f"Synchronizing parameters using combined data with shape {X_combined.shape}"
                )

                # Compute the fractal dimension if not provided
                if "d" not in density_kwargs:
                    d = mellon.parameters.compute_d_factal(X_combined)
                    estimator_defaults["d"] = d
                    logger.info(f"Synchronizing parameter d to {d:.4f}")

                # Precompute nearest neighbor distances if needed for mu or ls
                if "mu" not in density_kwargs or "ls" not in density_kwargs:
                    nn_distances = mellon.parameters.compute_nn_distances(X_combined)

                # Compute mu if not provided
                if "mu" not in density_kwargs:
                    d = estimator_defaults["d"]
                    mu = mellon.parameters.compute_mu(nn_distances, d)
                    estimator_defaults["mu"] = mu
                    logger.info(f"Synchronizing parameter mu to {mu:.4f}")

                # Compute length scale if not provided
                if "ls" not in density_kwargs:
                    base_ls = mellon.parameters.compute_ls(nn_distances)
                    ls = base_ls * ls_factor
                    estimator_defaults["ls"] = ls
                    logger.info(f"Synchronizing parameter ls to {ls:.4f}")

            # "condition1" / "condition2" / "symmetric": estimate from each
            # condition on its own, before either estimator is fitted, and share.
            elif param_scheme != "separate":
                unpinned = [p for p in _DENSITY_PARAMS if p not in density_kwargs]
                if unpinned:
                    # the seed a DensityEstimator uses for its own nn search
                    from mellon.parameters import DEFAULT_RANDOM_SEED

                    seed = density_kwargs.get("random_state")
                    if seed is None:
                        seed = DEFAULT_RANDOM_SEED
                    shared = _resolve_scheme_density_params(
                        param_scheme,
                        X_condition1,
                        X_condition2,
                        ls_factor,
                        seed,
                        d=density_kwargs.get("d"),
                    )
                    for name in unpinned:
                        estimator_defaults[name] = shared[name]
                    logger.info(
                        f"param_scheme={param_scheme!r}: sharing "
                        + ", ".join(
                            f"{name}={float(shared[name]):.4f}" for name in unpinned
                        )
                    )

            # Fit density estimators for both conditions
            logger.info("Fitting density estimator for condition 1...")
            density_estimator_condition1 = mellon.DensityEstimator(**estimator_defaults)
            density_estimator_condition1.fit(X_condition1)
            self.density_predictor1 = density_estimator_condition1.predict

            logger.info("Fitting density estimator for condition 2...")
            density_estimator_condition2 = mellon.DensityEstimator(**estimator_defaults)
            density_estimator_condition2.fit(X_condition2)
            self.density_predictor2 = density_estimator_condition2.predict
            logger.debug(
                "Density estimators fitted. Call predict() to compute fold changes."
            )
        else:
            logger.info(
                "Density estimators have already been fitted. Call predict() to compute fold changes."
            )

        # Check if sample indices are provided
        have_sample_indices = (
            condition1_sample_indices is not None
            or condition2_sample_indices is not None
        )

        # Auto-enable sample variance if sample indices are provided
        if have_sample_indices:
            if not self.use_sample_variance_explicit:
                self.use_sample_variance = True
                logger.info(
                    "Sample variance estimation automatically enabled due to provided sample indices"
                )

        # Check for contradictory inputs - user explicitly requested sample variance but didn't provide indices
        if (
            self.use_sample_variance_explicit
            and self.use_sample_variance is True
            and not have_sample_indices
            and self.variance_predictor1 is None
            and self.variance_predictor2 is None
        ):
            raise ValueError(
                "Sample variance estimation was explicitly enabled (use_sample_variance=True), "
                "but no sample indices or variance predictors were provided. "
                "Please provide at least one of: condition1_sample_indices, condition2_sample_indices, "
                "variance_predictor1, or variance_predictor2."
            )

        # Handle sample-specific variance if enabled and sample indices are provided
        if self.use_sample_variance and have_sample_indices:
            logger.info("Setting up sample variance estimation...")

            # Set up density estimator parameters for sample-specific models
            sample_estimator_kwargs = (
                estimator_defaults.copy()
                if "estimator_defaults" in locals()
                else density_kwargs.copy()
            )

            # Use specific length scale if provided
            if sample_estimator_ls is not None:
                sample_estimator_kwargs["ls"] = sample_estimator_ls

            # Fit variance estimators for both conditions, with fallback logic when allow_single_condition_variance=True
            condition1_variance_estimator = None
            condition2_variance_estimator = None

            # Try to fit variance estimator for condition 1
            if condition1_sample_indices is not None:
                logger.debug(
                    "Fitting sample-specific variance estimator for condition 1 using provided indices..."
                )

                try:
                    condition1_variance_estimator = SampleVarianceEstimator(
                        eps=self.eps, estimator_type="density"
                    )
                    # Set a flag to indicate this estimator is called from DifferentialAbundance
                    condition1_variance_estimator._called_from_differential = True

                    condition1_variance_estimator.fit(
                        X=X_condition1,
                        grouping_vector=condition1_sample_indices,
                        ls_factor=ls_factor,
                        estimator_kwargs=sample_estimator_kwargs,
                    )
                    self.variance_predictor1 = condition1_variance_estimator.predict
                    logger.debug(
                        "Successfully fitted variance estimator for condition 1"
                    )

                except ValueError as e:
                    if allow_single_condition_variance:
                        logger.info(
                            f"Variance estimation failed for condition 1: {e}. Will use condition 2 variance if available."
                        )
                        condition1_variance_estimator = None
                        self.variance_predictor1 = None
                    else:
                        raise e

            # Try to fit variance estimator for condition 2
            if condition2_sample_indices is not None:
                logger.debug(
                    "Fitting sample-specific variance estimator for condition 2 using provided indices..."
                )

                try:
                    condition2_variance_estimator = SampleVarianceEstimator(
                        eps=self.eps, estimator_type="density"
                    )
                    # Set a flag to indicate this estimator is called from DifferentialAbundance
                    condition2_variance_estimator._called_from_differential = True

                    condition2_variance_estimator.fit(
                        X=X_condition2,
                        grouping_vector=condition2_sample_indices,
                        ls_factor=ls_factor,
                        estimator_kwargs=sample_estimator_kwargs,
                    )
                    self.variance_predictor2 = condition2_variance_estimator.predict
                    logger.debug(
                        "Successfully fitted variance estimator for condition 2"
                    )

                except ValueError as e:
                    if allow_single_condition_variance:
                        logger.info(
                            f"Variance estimation failed for condition 2: {e}. Will use condition 1 variance if available."
                        )
                        condition2_variance_estimator = None
                        self.variance_predictor2 = None
                    else:
                        raise e

            # Handle single variance fallback when allow_single_condition_variance=True
            if allow_single_condition_variance and (
                condition1_variance_estimator is None
                or condition2_variance_estimator is None
            ):
                if (
                    condition1_variance_estimator is not None
                    and condition2_variance_estimator is None
                ):
                    logger.info(
                        "Using condition 1 variance estimator for both conditions"
                    )
                    self.variance_predictor2 = condition1_variance_estimator.predict
                elif (
                    condition2_variance_estimator is not None
                    and condition1_variance_estimator is None
                ):
                    logger.info(
                        "Using condition 2 variance estimator for both conditions"
                    )
                    self.variance_predictor1 = condition2_variance_estimator.predict
                elif (
                    condition1_variance_estimator is None
                    and condition2_variance_estimator is None
                ):
                    if (
                        condition1_sample_indices is not None
                        or condition2_sample_indices is not None
                    ):
                        raise ValueError(
                            "Both variance estimators failed to fit. Cannot proceed with sample variance estimation."
                        )

        return self

    def predict(
        self,
        X_new: np.ndarray,
        log_fold_change_threshold: Optional[float] = None,
        ptp_threshold: Optional[float] = None,
        progress: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Predict log density and log fold change for new points.

        This method computes all fold changes and related metrics.
        It uses internal batching for efficient computation with large datasets.

        Parameters
        ----------
        X_new : np.ndarray
            New cell states to predict. Shape (n_cells, n_features).
        log_fold_change_threshold : float, optional
            Threshold for considering a log fold change significant. If None, uses
            the threshold specified during initialization.
        ptp_threshold : float, optional
            Threshold for considering a PTP (Posterior Tail Probability) significant. If None, uses the
            threshold specified during initialization.
        progress : bool, optional
            Whether to show progress bars for operations, by default True.

        Returns
        -------
        dict
            Dictionary containing the predictions:
            - 'log_density_condition1': Log density for condition 1
            - 'log_density_condition2': Log density for condition 2
            - 'log_fold_change': Log fold change between conditions
            - 'log_fold_change_uncertainty': Uncertainty in the log fold change
            - 'log_fold_change_zscore': Z-scores for the log fold change
            - 'neg_log10_fold_change_ptp': Negative log10 PTP (Posterior Tail Probability) for the log fold change
            - 'log_fold_change_direction': Direction of change ('up', 'down', or 'neutral')
        """
        if self.density_predictor1 is None or self.density_predictor2 is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Use provided thresholds if specified, otherwise use class defaults
        if log_fold_change_threshold is None:
            log_fold_change_threshold = self.log_fold_change_threshold
        if ptp_threshold is None:
            ptp_threshold = self.ptp_threshold

        # Get batch size (from DifferentialAbundance class attribute)
        batch_size = getattr(self, "batch_size", None)

        # Define functions for batched processing
        def compute_density1(X_batch):
            return self.density_predictor1(X_batch, normalize=True)

        def compute_density2(X_batch):
            return self.density_predictor2(X_batch, normalize=True)

        def compute_uncertainty1(X_batch):
            return self.density_predictor1.uncertainty(X_batch)

        def compute_uncertainty2(X_batch):
            return self.density_predictor2.uncertainty(X_batch)

        # Functions for computing empirical variances using batches when sample variance is enabled
        if self.use_sample_variance and self.variance_predictor1 is not None:

            def compute_sample_variance1(X_batch):
                return self.variance_predictor1(X_batch, diag=True)
        else:

            def compute_sample_variance1(X_batch):
                return np.zeros(len(X_batch))

        if self.use_sample_variance and self.variance_predictor2 is not None:

            def compute_sample_variance2(X_batch):
                return self.variance_predictor2(X_batch, diag=True)
        else:

            def compute_sample_variance2(X_batch):
                return np.zeros(len(X_batch))

        # Apply batched processing to the expensive operations
        log_density_condition1 = apply_batched(
            compute_density1,
            X_new,
            batch_size=batch_size,
            show_progress=progress,
            desc="Computing density (condition 1)",
        )

        log_density_condition2 = apply_batched(
            compute_density2,
            X_new,
            batch_size=batch_size,
            show_progress=progress,
            desc="Computing density (condition 2)",
        )

        log_density_uncertainty_condition1 = apply_batched(
            compute_uncertainty1,
            X_new,
            batch_size=batch_size,
            show_progress=progress,
            desc="Computing uncertainty (condition 1)",
        )

        log_density_uncertainty_condition2 = apply_batched(
            compute_uncertainty2,
            X_new,
            batch_size=batch_size,
            show_progress=progress,
            desc="Computing uncertainty (condition 2)",
        )

        # Compute sample variance if enabled
        if self.use_sample_variance:
            if self.variance_predictor1 is not None:
                logger.info("Computing sample-specific variance for condition 1...")
                sample_variance1 = apply_batched(
                    compute_sample_variance1,
                    X_new,
                    batch_size=batch_size,
                    show_progress=progress,
                    desc="Computing sample variance (condition 1)",
                )
                # Add sample variance to uncertainty
                # For density, sample_variance1 will be of shape (n_cells, 1)
                # We need to flatten it to match log_density_uncertainty_condition1
                sample_variance1 = sample_variance1.flatten()
                log_density_uncertainty_condition1 += sample_variance1

            if self.variance_predictor2 is not None:
                logger.info("Computing sample-specific variance for condition 2...")
                sample_variance2 = apply_batched(
                    compute_sample_variance2,
                    X_new,
                    batch_size=batch_size,
                    show_progress=progress,
                    desc="Computing sample variance (condition 2)",
                )
                # Add sample variance to uncertainty
                sample_variance2 = sample_variance2.flatten()
                log_density_uncertainty_condition2 += sample_variance2

        # The rest of the computation is lightweight and can be done all at once
        # Compute log fold change and uncertainty
        log_fold_change = log_density_condition2 - log_density_condition1
        log_fold_change_uncertainty = (
            log_density_uncertainty_condition1 + log_density_uncertainty_condition2
        )

        # Compute z-scores
        sd = np.sqrt(log_fold_change_uncertainty + self.eps)
        log_fold_change_zscore = log_fold_change / sd

        # Compute PTP (Posterior Tail Probability) in natural log (base e).
        # One-sided per manuscript: PTP = Φ(−|z|) = min(Φ(z), Φ(−z)) for real z.
        ln_ptp = np.minimum(
            normal.logcdf(log_fold_change_zscore),
            normal.logcdf(-log_fold_change_zscore),
        )

        # Convert from natural log to negative log10 (for better volcano plot visualization)
        # ln_ptp is a log of a small value (typically < 1), so it's negative
        # We want -log10(ptp), which is positive for small PTP (Posterior Tail Probability)
        neg_log10_fold_change_ptp = -(ln_ptp / np.log(10))

        # Determine direction of change based on thresholds
        log_fold_change_direction = np.full(
            len(log_fold_change), "neutral", dtype=object
        )
        # For negative log10 PTPs (Posterior Tail Probabilities), we need to check if they are greater than -log10(threshold)
        # e.g., -log10(0.05) ≈ 1.3, so we check if neg_log10_fold_change_ptp > 1.3
        significant = (np.abs(log_fold_change) > log_fold_change_threshold) & (
            neg_log10_fold_change_ptp > -np.log10(ptp_threshold)
        )

        log_fold_change_direction[significant & (log_fold_change > 0)] = "up"
        log_fold_change_direction[significant & (log_fold_change < 0)] = "down"

        return {
            "log_density_condition1": log_density_condition1,
            "log_density_condition2": log_density_condition2,
            "log_fold_change": log_fold_change,
            "log_fold_change_uncertainty": log_fold_change_uncertainty,
            "log_fold_change_zscore": log_fold_change_zscore,
            "neg_log10_fold_change_ptp": neg_log10_fold_change_ptp,  # Using negative log10 PTP (Posterior Tail Probability) (higher = more significant)
            "log_fold_change_direction": log_fold_change_direction,
        }
