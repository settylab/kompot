"""Differential abundance analysis for cell density comparison."""

import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple, List, Optional, Union, Dict, Any, Callable
import logging
from scipy.stats import norm as normal
from tqdm.auto import tqdm

import mellon
from mellon.parameters import compute_landmarks

from ..utils import find_landmarks
from ..batch_utils import apply_batched
from .sample_variance_estimator import SampleVarianceEstimator

logger = logging.getLogger("kompot")


class DifferentialAbundance:
    """
    Compute differential abundance between two conditions.
    
    This class analyzes the differences in cell density between two conditions
    (e.g., control vs. treatment) using density estimation and fold change analysis.
    
    The analysis can be performed with synchronized parameters between conditions
    by setting sync_parameters=True in the fit method, which ensures consistent
    density estimation across both conditions.
    
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
    log_fold_change_pvalue : np.ndarray
        P-values for the log fold changes.
    log_fold_change_direction : np.ndarray
        Direction of change ('up', 'down', or 'neutral') based on thresholds.
    
    Methods
    -------
    fit(X_condition1, X_condition2, sync_parameters=False, **density_kwargs)
        Fit density estimators for both conditions, optionally with synchronized parameters.
    predict(X_new)
        Predict log density and log fold change for new points.
    """
    
    def __init__(
        self,
        log_fold_change_threshold: float = 1.0,
        pvalue_threshold: float = 1e-2,
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
        pvalue_threshold : float, optional
            Threshold for considering a p-value significant, by default 1e-2.
        n_landmarks : int, optional
            Number of landmarks to use for approximation. If None, use all points, by default None.
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
        self.pvalue_threshold = pvalue_threshold
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
            self.use_sample_variance = (variance_predictor1 is not None or variance_predictor2 is not None)
            if self.use_sample_variance:
                logger.info("Sample variance estimation automatically enabled due to presence of variance predictors")
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
        self.log_fold_change_pvalue = None
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
        sync_parameters: bool = False,
        **density_kwargs
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
        sync_parameters : bool, optional
            Whether to synchronize model parameters (d, mu, ls) between both conditions using 
            the combined dataset. When True, parameters are computed once from the combined data 
            to ensure models for both conditions use identical parameter values. This is especially 
            important for consistent density estimation across conditions. Default is False.
        **density_kwargs : dict
            Additional arguments to pass to the DensityEstimator.
            
        Returns
        -------
        self
            The fitted instance.
        """

        # Create or use density predictors
        if self.density_predictor1 is None or self.density_predictor2 is None:
            # Configure density estimator defaults
            estimator_defaults = {
                'd_method': 'fractal',
                'predictor_with_uncertainty': True,
                'optimizer': 'advi',
            }
            
            # Add ls_factor to estimator_defaults if ls is not already specified
            if 'ls' not in density_kwargs:
                estimator_defaults['ls_factor'] = ls_factor
            
            # Update defaults with user-provided values (user-provided settings will override ls_factor if ls is specified)
            estimator_defaults.update(density_kwargs)
            
            # Use provided landmarks if available, otherwise compute them if requested
            if landmarks is not None:
                logger.info(f"Using provided landmarks with shape {landmarks.shape}")
                estimator_defaults['landmarks'] = landmarks
                # Store provided landmarks for future use
                self.computed_landmarks = landmarks
            elif self.n_landmarks is not None:
                # Use mellon's compute_landmarks function to get properly distributed landmarks
                # Pass the random_state parameter directly to ensure reproducible results
                X_combined = np.vstack([X_condition1, X_condition2])
                computed_landmarks = compute_landmarks(
                    X_combined, 
                    n_landmarks=self.n_landmarks,
                    random_state=self.random_state
                )
                estimator_defaults['landmarks'] = computed_landmarks
                # Store computed landmarks for future use
                self.computed_landmarks = computed_landmarks

            # Handle synchronization of parameters when sync_parameters is True
            if sync_parameters:
                # Combine data from both conditions for parameter estimation
                X_combined = np.vstack([X_condition1, X_condition2])
                logger.info(f"Synchronizing parameters using combined data with shape {X_combined.shape}")
                
                # Compute the fractal dimension if not provided
                if "d" not in density_kwargs:
                    d = mellon.parameters.compute_d_factal(X_combined)
                    estimator_defaults["d"] = d
                    logger.info(f"Synchronizing parameter d to {d:.4f}")
                
                # Precompute nearest neighbor distances if needed for mu or ls
                if ("mu" not in density_kwargs or "ls" not in density_kwargs):
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
                
                
            # Fit density estimators for both conditions
            logger.info("Fitting density estimator for condition 1...")
            density_estimator_condition1 = mellon.DensityEstimator(**estimator_defaults)
            density_estimator_condition1.fit(X_condition1)
            self.density_predictor1 = density_estimator_condition1.predict
            
            logger.info("Fitting density estimator for condition 2...")
            density_estimator_condition2 = mellon.DensityEstimator(**estimator_defaults)
            density_estimator_condition2.fit(X_condition2)
            self.density_predictor2 = density_estimator_condition2.predict
            logger.debug("Density estimators fitted. Call predict() to compute fold changes.")
        else:
            logger.info("Density estimators have already been fitted. Call predict() to compute fold changes.")
            
        # Check if sample indices are provided
        have_sample_indices = (condition1_sample_indices is not None or condition2_sample_indices is not None)
        
        # Auto-enable sample variance if sample indices are provided
        if have_sample_indices:
            if not self.use_sample_variance_explicit:
                self.use_sample_variance = True
                logger.info("Sample variance estimation automatically enabled due to provided sample indices")
        
        # Check for contradictory inputs - user explicitly requested sample variance but didn't provide indices
        if self.use_sample_variance_explicit and self.use_sample_variance is True and not have_sample_indices and self.variance_predictor1 is None and self.variance_predictor2 is None:
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
            sample_estimator_kwargs = estimator_defaults.copy() if 'estimator_defaults' in locals() else density_kwargs.copy()
            
            # Use specific length scale if provided
            if sample_estimator_ls is not None:
                sample_estimator_kwargs['ls'] = sample_estimator_ls
            
            # Fit variance estimator for condition 1
            if condition1_sample_indices is not None:
                logger.debug("Fitting sample-specific variance estimator for condition 1 using provided indices...")
                condition1_variance_estimator = SampleVarianceEstimator(
                    eps=self.eps,
                    estimator_type='density'
                )
                # Set a flag to indicate this estimator is called from DifferentialAbundance
                condition1_variance_estimator._called_from_differential = True
                
                condition1_variance_estimator.fit(
                    X=X_condition1, 
                    grouping_vector=condition1_sample_indices,
                    ls_factor=ls_factor,
                    estimator_kwargs=sample_estimator_kwargs
                )
                self.variance_predictor1 = condition1_variance_estimator.predict
            
            # Fit variance estimator for condition 2
            if condition2_sample_indices is not None:
                logger.debug("Fitting sample-specific variance estimator for condition 2 using provided indices...")
                condition2_variance_estimator = SampleVarianceEstimator(
                    eps=self.eps,
                    estimator_type='density'
                )
                # Set a flag to indicate this estimator is called from DifferentialAbundance
                condition2_variance_estimator._called_from_differential = True
                
                condition2_variance_estimator.fit(
                    X=X_condition2, 
                    grouping_vector=condition2_sample_indices,
                    ls_factor=ls_factor,
                    estimator_kwargs=sample_estimator_kwargs
                )
                self.variance_predictor2 = condition2_variance_estimator.predict
        
        return self
    
    def predict(
        self, 
        X_new: np.ndarray,
        log_fold_change_threshold: Optional[float] = None,
        pvalue_threshold: Optional[float] = None,
        progress: bool = True
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
        pvalue_threshold : float, optional
            Threshold for considering a p-value significant. If None, uses the
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
            - 'neg_log10_fold_change_pvalue': Negative log10 p-values for the log fold change
            - 'log_fold_change_direction': Direction of change ('up', 'down', or 'neutral')
        """
        if self.density_predictor1 is None or self.density_predictor2 is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Use provided thresholds if specified, otherwise use class defaults
        if log_fold_change_threshold is None:
            log_fold_change_threshold = self.log_fold_change_threshold
        if pvalue_threshold is None:
            pvalue_threshold = self.pvalue_threshold
        
        # Get batch size (from DifferentialAbundance class attribute)
        batch_size = getattr(self, 'batch_size', None)
        
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
            desc="Computing density (condition 1)" if progress else None
        )
        
        log_density_condition2 = apply_batched(
            compute_density2, 
            X_new, 
            batch_size=batch_size,
            desc="Computing density (condition 2)" if progress else None
        )
        
        log_density_uncertainty_condition1 = apply_batched(
            compute_uncertainty1, 
            X_new, 
            batch_size=batch_size,
            desc="Computing uncertainty (condition 1)" if progress else None
        )
        
        log_density_uncertainty_condition2 = apply_batched(
            compute_uncertainty2, 
            X_new, 
            batch_size=batch_size,
            desc="Computing uncertainty (condition 2)" if progress else None
        )
        
        # Compute sample variance if enabled
        if self.use_sample_variance:
            if self.variance_predictor1 is not None:
                logger.info("Computing sample-specific variance for condition 1...")
                sample_variance1 = apply_batched(
                    compute_sample_variance1,
                    X_new,
                    batch_size=batch_size,
                    desc="Computing sample variance (condition 1)" if progress else None
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
                    desc="Computing sample variance (condition 2)" if progress else None
                )
                # Add sample variance to uncertainty
                sample_variance2 = sample_variance2.flatten()
                log_density_uncertainty_condition2 += sample_variance2
        
        # The rest of the computation is lightweight and can be done all at once
        # Compute log fold change and uncertainty
        log_fold_change = log_density_condition2 - log_density_condition1
        log_fold_change_uncertainty = log_density_uncertainty_condition1 + log_density_uncertainty_condition2
        
        # Compute z-scores
        sd = np.sqrt(log_fold_change_uncertainty + self.eps)
        log_fold_change_zscore = log_fold_change / sd
        
        # Compute p-values in natural log (base e)
        ln_pvalue = np.minimum(
            normal.logcdf(log_fold_change_zscore), 
            normal.logcdf(-log_fold_change_zscore)
        ) + np.log(2)
        
        # Convert from natural log to negative log10 (for better volcano plot visualization)
        # ln_pvalue is a log of a small value (typically < 1), so it's negative
        # We want -log10(p), which is positive for small p-values
        neg_log10_fold_change_pvalue = -(ln_pvalue / np.log(10))
        
        # Determine direction of change based on thresholds
        log_fold_change_direction = np.full(len(log_fold_change), 'neutral', dtype=object)
        # For negative log10 p-values, we need to check if they are greater than -log10(threshold)
        # e.g., -log10(0.05) ≈ 1.3, so we check if neg_log10_fold_change_pvalue > 1.3
        significant = (np.abs(log_fold_change) > log_fold_change_threshold) & \
                     (neg_log10_fold_change_pvalue > -np.log10(pvalue_threshold))
        
        log_fold_change_direction[significant & (log_fold_change > 0)] = 'up'
        log_fold_change_direction[significant & (log_fold_change < 0)] = 'down'
        
        return {
            'log_density_condition1': log_density_condition1,
            'log_density_condition2': log_density_condition2,
            'log_fold_change': log_fold_change,
            'log_fold_change_uncertainty': log_fold_change_uncertainty,
            'log_fold_change_zscore': log_fold_change_zscore,
            'neg_log10_fold_change_pvalue': neg_log10_fold_change_pvalue,  # Using negative log10 p-values (higher = more significant)
            'log_fold_change_direction': log_fold_change_direction,
        }