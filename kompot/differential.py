"""Differential analysis for gene expression and abundance."""

import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple, List, Optional, Union, Dict, Any, Callable
import logging
from scipy.stats import norm as normal
from scipy.sparse import csr_matrix
from tqdm.auto import tqdm
from sklearn.neighbors import NearestNeighbors

import mellon
from mellon.parameters import compute_landmarks
from .utils import (
    compute_mahalanobis_distance, 
    compute_mahalanobis_distances, 
    prepare_mahalanobis_matrix, 
    find_landmarks,
    KOMPOT_COLORS
)
from .batch_utils import batch_process, apply_batched, is_jax_memory_error

logger = logging.getLogger("kompot")


class DifferentialAbundance:
    """
    Compute differential abundance between two conditions.
    
    This class analyzes the differences in cell density between two conditions
    (e.g., control vs. treatment) using density estimation and fold change analysis.
    
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
    fit(X_condition1, X_condition2, **density_kwargs)
        Fit density estimators for both conditions and compute differential metrics.
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
            
            # If user explicitly enabled sample variance but no variance predictors are provided, log a warning
            if self.use_sample_variance and variance_predictor1 is None and variance_predictor2 is None:
                logger.warning(
                    "Sample variance estimation was explicitly enabled (use_sample_variance=True) "
                    "but no variance predictors were provided. "
                    "You will need to provide sample indices in the fit() method."
                )
        
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
                
            # Fit density estimators for both conditions
            logger.info("Fitting density estimator for condition 1...")
            density_estimator_condition1 = mellon.DensityEstimator(**estimator_defaults)
            density_estimator_condition1.fit(X_condition1)
            self.density_predictor1 = density_estimator_condition1.predict
            
            logger.info("Fitting density estimator for condition 2...")
            density_estimator_condition2 = mellon.DensityEstimator(**estimator_defaults)
            density_estimator_condition2.fit(X_condition2)
            self.density_predictor2 = density_estimator_condition2.predict
            logger.info("Density estimators fitted. Call predict() to compute fold changes.")
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
                logger.info("Fitting sample-specific variance estimator for condition 1 using provided indices...")
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
                logger.info("Fitting sample-specific variance estimator for condition 2 using provided indices...")
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
        

class SampleVarianceEstimator:
    """
    Compute local sample variances of gene expressions or density.
    
    This class manages the computation of empirical variance by fitting function estimators
    or density estimators for each group in the data and computing the variance between their
    predictions.
    
    Attributes
    ----------
    group_predictors : Dict
        Dictionary of prediction functions for each group.
    estimator_type : str
        Type of estimator used ('function' for gene expression, 'density' for cell density).
    """
    
    def __init__(
        self,
        eps: float = 1e-12,
        jit_compile: bool = True,
        estimator_type: str = 'function'
    ):
        """
        Initialize the SampleVarianceEstimator.
        
        Parameters
        ----------
        eps : float, optional
            Small constant for numerical stability, by default 1e-12.
        jit_compile : bool, optional
            Whether to use JAX just-in-time compilation, by default True.
        estimator_type : str, optional
            Type of estimator to use ('function' for gene expression, 'density' for cell density),
            by default 'function'.
        """
        self.eps = eps
        self.jit_compile = jit_compile
        self.estimator_type = estimator_type
        
        if estimator_type not in ['function', 'density']:
            raise ValueError("estimator_type must be either 'function' or 'density'")
        
        # Will be populated during fit
        self.group_predictors = {}
        self.group_centroids = {}
        self._predict_variance_jit = None
    
    def fit(
        self, 
        X: np.ndarray,
        Y: np.ndarray = None, 
        grouping_vector: np.ndarray = None,
        min_cells: int = 10,
        ls_factor: float = 10.0,
        estimator_kwargs: Dict = None
    ):
        """
        Fit estimators for each group in the data and store only their predictors.
        
        Parameters
        ----------
        X : np.ndarray
            Cell states. Shape (n_cells, n_features).
        Y : np.ndarray, optional
            Gene expression values. Shape (n_cells, n_genes). 
            Required for function estimator, not used for density estimator.
        grouping_vector : np.ndarray
            Vector specifying which group each cell belongs to. Shape (n_cells,).
        min_cells : int
            Minimum number of cells for group to train an estimator. Default is 10.
        ls_factor : float, optional
            Multiplication factor to apply to length scale when it's automatically inferred, 
            by default 10.0. Only used when ls is not explicitly provided in estimator_kwargs.
        estimator_kwargs : Dict, optional
            Additional arguments to pass to the estimator constructor
            (FunctionEstimator or DensityEstimator).
            
        Returns
        -------
        self
            The fitted instance.
        """
        if estimator_kwargs is None:
            estimator_kwargs = {}
        
        # Add ls_factor to estimator_kwargs if ls is not already specified
        if 'ls' not in estimator_kwargs:
            estimator_kwargs['ls_factor'] = ls_factor
        
        # Get unique groups
        unique_groups = np.unique(grouping_vector)
        
        logger.info(f"Found {len(unique_groups):,} unique groups for variance estimation")
        
        # Organize data by groups
        group_indices = {
            group_id: np.where(grouping_vector == group_id)[0]
            for group_id in unique_groups
        }
        
        # Train estimators for each group and store only their predictors
        logger.info(f"Training group-specific {self.estimator_type} estimators...")
        
        for group_id, indices in group_indices.items():
            if len(indices) >= min_cells:  # Only train if we have enough data points
                logger.info(f"Training estimator for group {group_id} with {len(indices)} cells")
                X_subset = X[indices]
                
                if self.estimator_type == 'function':
                    if Y is None:
                        raise ValueError("Y must be provided for function estimator type")
                    Y_subset = Y[indices]
                    
                    # Create and train function estimator
                    estimator = mellon.FunctionEstimator(**estimator_kwargs)
                    estimator.fit(X_subset, Y_subset)
                
                else:  # density estimator
                    # Configure density estimator defaults
                    density_defaults = {
                        'd_method': 'fractal',
                        'predictor_with_uncertainty': True,
                        'optimizer': 'advi',
                    }
                    density_defaults.update(estimator_kwargs)
                    
                    # Create and train density estimator
                    estimator = mellon.DensityEstimator(**density_defaults)
                    estimator.fit(X_subset)
                
                # Store only the predictor function, not the full estimator
                self.group_predictors[group_id] = estimator.predict
                
                # Immediately delete the estimator to free memory
                del estimator
            else:
                logger.warning(f"Skipping group {group_id} (only {len(indices):,} cells < min_cells={min_cells:,})")
        
        return self
    
    def predict(self, X_new: np.ndarray, diag: bool = False) -> np.ndarray:
        """
        Predict empirical variance for new points using JAX.
        
        Parameters
        ----------
        X_new : np.ndarray
            New cell states to predict. Shape (n_cells, n_features).
        diag : bool, optional
            If True (default is False), compute the variance for each cell state.
            If False, compute the full covariance matrix between all pairs of cells.
            
        Returns
        -------
        np.ndarray
            If diag=True: 
                For function estimators: Empirical variance for each new point. Shape (n_cells, n_genes).
                For density estimators: Empirical variance for each new point. Shape (n_cells, 1).
            If diag=False: 
                For function estimators: Full covariance matrix. Shape (n_cells, n_cells, n_genes).
                For density estimators: Full covariance matrix. Shape (n_cells, n_cells, 1).
        """
        # Check if group_predictors exists (model was initialized)
        if not hasattr(self, 'group_predictors'):
            error_msg = "Model not initialized correctly. Make sure to initialize SampleVarianceEstimator properly."
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        # Check if group_predictors has any entries (model was fitted successfully)
        if not self.group_predictors:
            error_msg = "No group predictors available. Sample variance estimation failed."
            logger.error(error_msg)
            raise RuntimeError(error_msg)
            
        n_cells = len(X_new)
        
        # Get the shape of the output from the first predictor
        # This assumes all predictors produce outputs of the same shape
        first_predictor = list(self.group_predictors.values())[0]
        
        if self.estimator_type == 'function':
            test_pred = first_predictor([X_new[0]])
            n_genes = test_pred.shape[1] if len(test_pred.shape) > 1 else 1
            
            # Check if we're processing too many genes at once (>100) in a specific case
            if n_genes > 100 and len(self.group_predictors) > 1 and hasattr(self, '_called_from_differential') and self._called_from_differential:
                logger.warning(
                    f"Computing sample variance for {n_genes} genes may require significant memory. "
                    f"If you encounter memory issues, consider running differential expression analysis "
                    f"on smaller gene sets (max 100 genes) at a time."
                )
        else:  # density estimator
            test_pred = first_predictor([X_new[0]])
            # For density, we'll reshape to have a singleton dimension for consistency
            n_genes = 1
        
        # If we have no predictors, raise an error
        if not self.group_predictors:
            error_msg = "No group predictors available. Sample variance estimation failed."
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Get list of predictors
        predictors_list = list(self.group_predictors.values())
        
        if diag:
            # Regular diagonal variance computation (per cell)
            # Compile the prediction function if we're using JIT and haven't already
            if self.jit_compile and self._predict_variance_jit is None:
                # Define our variance computation function
                def compute_variance_from_predictions(X, predictions_list):
                    # Stack the predictions
                    stacked = jnp.stack(predictions_list, axis=0)
                    # Compute variance across groups (axis 0)
                    return jnp.var(stacked, axis=0)
                
                # JIT compile the function
                self._predict_variance_jit = jax.jit(compute_variance_from_predictions)
            
            # Get predictions from each group predictor
            all_group_predictions = []
            for predictor in predictors_list:
                if self.estimator_type == 'function':
                    group_predictions = predictor(X_new)
                else:  # density estimator
                    group_predictions = predictor(X_new, normalize=True)
                    # Reshape to have shape (n_cells, 1) for consistency
                    group_predictions = np.reshape(group_predictions, (-1, 1))
                
                all_group_predictions.append(group_predictions)
            
            # Convert to JAX arrays
            all_group_predictions_jax = [jnp.array(pred) for pred in all_group_predictions]
            
            # Use the JIT-compiled function if available
            if self.jit_compile and self._predict_variance_jit is not None:
                batch_variance = self._predict_variance_jit(X_new, all_group_predictions_jax)
                return np.array(batch_variance)
            else:
                # Stack predictions and compute variance using JAX
                stacked_predictions = jnp.stack(all_group_predictions_jax, axis=0)
                batch_variance = jnp.var(stacked_predictions, axis=0)
                # Convert back to numpy for compatibility
                return np.array(batch_variance)
        
        else:
            # Full covariance matrix computation (between all pairs of cells)
            # Initialize the result matrix
            cov_matrix = np.zeros((n_cells, n_cells, n_genes))
            
            # Get predictions from each group predictor
            group_predictions = []
            for predictor in predictors_list:
                # Get predictions for all cells at once
                if self.estimator_type == 'function':
                    pred = predictor(X_new)
                else:  # density estimator
                    pred = predictor(X_new, normalize=True)
                    # Reshape to have shape (n_cells, 1) for consistency
                    pred = np.reshape(pred, (-1, 1))
                
                group_predictions.append(jnp.array(pred))
            
            # Stack predictions across groups - shape (n_groups, n_cells, n_genes)
            stacked_predictions = jnp.stack(group_predictions, axis=0)
            
            # Calculate covariance between each pair of cells across groups
            # First, center the data for each gene by subtracting the mean across groups
            means = jnp.mean(stacked_predictions, axis=0, keepdims=True)  # (1, n_cells, n_genes)
            centered = stacked_predictions - means  # (n_groups, n_cells, n_genes)
            
            # Reshape for matrix multiplication
            centered_reshaped = jnp.moveaxis(centered, 1, 0)  # (n_cells, n_groups, n_genes)
            
            # Calculate covariance for each gene
            n_groups = centered.shape[0]
            
            # Process each gene individually to avoid memory issues
            for g in range(n_genes):
                gene_centered = centered_reshaped[:, :, g]  # (n_cells, n_groups)
                # Calculate covariance as dot product divided by n_groups
                gene_cov = (gene_centered @ gene_centered.T) / n_groups
                cov_matrix[:, :, g] = np.array(gene_cov)
            
            return cov_matrix


def compute_weighted_mean_fold_change(
    fold_change: np.ndarray,
    log_density_condition1: np.ndarray = None,
    log_density_condition2: np.ndarray = None,
    log_density_diff: np.ndarray = None
) -> np.ndarray:
    """
    Compute weighted mean fold change using density differences as weights.
    
    This utility function computes weighted mean fold changes from expression and density log fold changes.
    
    Parameters
    ----------
    fold_change : np.ndarray
        Expression fold change for each cell and gene. Shape (n_cells, n_genes).
    log_density_condition1 : np.ndarray or pandas.Series, optional
        Log density for condition 1. Shape (n_cells,). Can be omitted if log_density_diff is provided.
    log_density_condition2 : np.ndarray or pandas.Series, optional
        Log density for condition 2. Shape (n_cells,). Can be omitted if log_density_diff is provided.
    log_density_diff : np.ndarray, optional
        Pre-computed log density difference. If provided, log_density_condition1 and 
        log_density_condition2 are ignored. Shape (n_cells,).
        
    Returns
    -------
    np.ndarray
        Weighted mean log fold change for each gene. Shape (n_genes,).
    """
    if log_density_diff is None:
        if log_density_condition1 is None or log_density_condition2 is None:
            raise ValueError("Either log_density_diff or both log_density_condition1 and log_density_condition2 must be provided")
        
        # Convert pandas Series to numpy arrays if needed
        if hasattr(log_density_condition1, 'to_numpy'):
            log_density_condition1 = log_density_condition1.to_numpy()
        if hasattr(log_density_condition2, 'to_numpy'):
            log_density_condition2 = log_density_condition2.to_numpy()
            
        # Calculate the density difference for each cell
        log_density_diff = log_density_condition2 - log_density_condition1
    elif hasattr(log_density_diff, 'to_numpy'):
        # Convert pandas Series to numpy arrays if needed
        log_density_diff = log_density_diff.to_numpy()
    
    # Convert to numpy array if it's a list
    if isinstance(fold_change, list):
        fold_change = np.array(fold_change)
    
    # Create a weights array with shape (n_cells, 1) for broadcasting
    # Apply np.exp(np.abs(...)) to the log_density_diff as part of the function's logic
    weights = np.exp(np.abs(log_density_diff.reshape(-1, 1)))
    
    # Weight the fold changes by density difference
    weighted_fold_change = fold_change * weights
    
    return np.sum(weighted_fold_change, axis=0) / np.sum(weights)


class DifferentialExpression:
    """
    Compute differential expression between two conditions.
    
    This class analyzes the differences in gene expression between two conditions
    (e.g., control vs. treatment) using imputation, Mahalanobis distance, and 
    log fold change analysis.
    
    Attributes
    ----------
    function_predictor1 : Callable
        Function predictor for condition 1.
    function_predictor2 : Callable
        Function predictor for condition 2.
    variance_predictor1 : Callable, optional
        Variance predictor for condition 1. If provided, will be used for uncertainty calculation.
    variance_predictor2 : Callable, optional
        Variance predictor for condition 2. If provided, will be used for uncertainty calculation.
    mahalanobis_distances : np.ndarray
        Mahalanobis distances for each gene.
    """
    
    def __init__(
        self,
        n_landmarks: Optional[int] = None,
        use_sample_variance: Optional[bool] = None,
        eps: float = 1e-12,
        jit_compile: bool = False,
        function_predictor1: Optional[Any] = None,
        function_predictor2: Optional[Any] = None,
        variance_predictor1: Optional[Any] = None,
        variance_predictor2: Optional[Any] = None,
        random_state: Optional[int] = None,
        batch_size: int = 500,
        mahalanobis_batch_size: Optional[int] = None,
    ):
        """
        Initialize DifferentialExpression.
        
        Parameters
        ----------
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
        function_predictor1 : Any, optional
            Precomputed function predictor for condition 1, typically from FunctionEstimator.predict
        function_predictor2 : Any, optional
            Precomputed function predictor for condition 2, typically from FunctionEstimator.predict
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
            Number of cells to process at once during prediction to manage memory usage.
            If None or 0, all samples will be processed at once. Default is 500.
        mahalanobis_batch_size : int, optional
            Number of genes to process in each batch during Mahalanobis distance computation.
            Smaller values use less memory but are slower. If None, uses batch_size.
            Increase for faster computation if you have sufficient memory.
        """
        self.n_landmarks = n_landmarks
        self.eps = eps
        self.jit_compile = jit_compile
        self.random_state = random_state
        
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
            
            # If user explicitly enabled sample variance but no variance predictors are provided, log a warning
            if self.use_sample_variance and variance_predictor1 is None and variance_predictor2 is None:
                logger.warning(
                    "Sample variance estimation was explicitly enabled (use_sample_variance=True) "
                    "but no variance predictors were provided. "
                    "You will need to provide sample indices in the fit() method."
                )
        self.batch_size = batch_size
        
        # For Mahalanobis distance computation, use a separate batch size parameter if provided
        self.mahalanobis_batch_size = mahalanobis_batch_size if mahalanobis_batch_size is not None else batch_size
        
        # Function estimators or predictors
        self.expression_estimator_condition1 = None
        self.expression_estimator_condition2 = None
        self.function_predictor1 = function_predictor1
        self.function_predictor2 = function_predictor2
        
        # Variance predictors
        self.variance_predictor1 = variance_predictor1
        self.variance_predictor2 = variance_predictor2
        
        # Mahalanobis distances
        self.mahalanobis_distances = None
        
    def fit(
        self, 
        X_condition1: np.ndarray,
        y_condition1: np.ndarray, 
        X_condition2: np.ndarray,
        y_condition2: np.ndarray,
        sigma: float = 1.0,
        ls: Optional[float] = None,
        ls_factor: float = 10.0,
        landmarks: Optional[np.ndarray] = None,
        sample_estimator_ls: Optional[float] = None,
        condition1_sample_indices: Optional[np.ndarray] = None,
        condition2_sample_indices: Optional[np.ndarray] = None,
        **function_kwargs
    ):
        """
        Fit function estimators for both conditions.
        
        This method only creates the estimators and does not compute fold changes.
        Call predict() to compute fold changes on any set of points.
        
        Parameters
        ----------
        X_condition1 : np.ndarray
            Cell states for the first condition. Shape (n_cells1, n_features).
        y_condition1 : np.ndarray
            Gene expression values for the first condition. Shape (n_cells1, n_genes).
        X_condition2 : np.ndarray
            Cell states for the second condition. Shape (n_cells2, n_features).
        y_condition2 : np.ndarray
            Gene expression values for the second condition. Shape (n_cells2, n_genes).
        sigma : float, optional
            Noise level for function estimator, by default 1.0.
        ls : float, optional
            Length scale for the GP kernel. If None, it will be estimated, by default None.
        ls_factor : float, optional
            Multiplication factor to apply to length scale when it's automatically inferred, 
            by default 10.0. Only used when ls is None.
        landmarks : np.ndarray, optional
            Pre-computed landmarks to use. If provided, n_landmarks will be ignored.
            Shape (n_landmarks, n_features).
        sample_estimator_ls : float, optional
            Length scale for the sample-specific variance estimators. If None, will use
            the same value as ls or it will be estimated, by default None.
        condition1_sample_indices : np.ndarray, optional
            Sample indices for first condition. Used for sample variance estimation.
            Unique values in this array define different sample groups.
        condition2_sample_indices : np.ndarray, optional
            Sample indices for second condition. Used for sample variance estimation.
            Unique values in this array define different sample groups.
        **function_kwargs : dict
            Additional arguments to pass to the FunctionEstimator.
            
        Returns
        -------
        self
            The fitted instance.
        """
        # Create or use function predictors
        if self.function_predictor1 is None or self.function_predictor2 is None:
            # Configure function estimator defaults
            estimator_defaults = {
                'sigma': sigma,
                'optimizer': 'advi',
                'predictor_with_uncertainty': True,
            }
            
            # Update defaults with user-provided values
            estimator_defaults.update(function_kwargs)
            
            # Use provided landmarks if available, otherwise compute them if requested
            if landmarks is not None:
                logger.info(f"Using provided landmarks with shape {landmarks.shape}")
                estimator_defaults['landmarks'] = landmarks
                estimator_defaults['gp_type'] = 'fixed'
            elif self.n_landmarks is not None:
                # Use mellon's compute_landmarks function to get properly distributed landmarks
                # Pass the random_state parameter directly to ensure reproducible results
                X_combined = np.vstack([X_condition1, X_condition2])
                computed_landmarks = compute_landmarks(
                    X_combined, 
                    gp_type='fixed', 
                    n_landmarks=self.n_landmarks,
                    random_state=self.random_state
                )
                estimator_defaults['landmarks'] = computed_landmarks
                estimator_defaults['gp_type'] = 'fixed'
                # Store computed landmarks for future use
                self.computed_landmarks = computed_landmarks
                
            # If ls is provided, use it directly
            if ls is not None:
                estimator_defaults['ls'] = ls
            else:
                # When ls is not provided, pass ls_factor to the estimator
                estimator_defaults['ls_factor'] = ls_factor
                
            # Fit expression estimators for both conditions
            logger.info("Fitting expression estimator for condition 1...")
            self.expression_estimator_condition1 = mellon.FunctionEstimator(**estimator_defaults)
            self.expression_estimator_condition1.fit(X_condition1, y_condition1)
            self.function_predictor1 = self.expression_estimator_condition1.predict
            
            # Update ls for condition 2 based on condition 1 if not provided
            if ls is None and 'ls' not in function_kwargs:
                # Get ls from condition 1 and use it for condition 2
                ls_cond1 = self.function_predictor1.cov_func.ls
                estimator_defaults['ls'] = ls_cond1
                # We already applied ls_factor in condition 1, so we don't need to pass it again
                if 'ls_factor' in estimator_defaults:
                    del estimator_defaults['ls_factor']
            
            logger.info("Fitting expression estimator for condition 2...")
            self.expression_estimator_condition2 = mellon.FunctionEstimator(**estimator_defaults)
            self.expression_estimator_condition2.fit(X_condition2, y_condition2)
            self.function_predictor2 = self.expression_estimator_condition2.predict
        
        # Check if sample indices are provided
        have_sample_indices = (condition1_sample_indices is not None or condition2_sample_indices is not None)
        
        # Auto-enable sample variance if sample indices are provided
        if have_sample_indices:
            if self.use_sample_variance is None or self.use_sample_variance_explicit is False:
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
            
            # Set up function estimator parameters for sample-specific models
            sample_estimator_kwargs = estimator_defaults.copy() if 'estimator_defaults' in locals() else function_kwargs.copy()
            
            # Use specific length scale if provided
            if sample_estimator_ls is not None:
                sample_estimator_kwargs['ls'] = sample_estimator_ls
            
            # Fit variance estimator for condition 1
            if condition1_sample_indices is not None:
                logger.info("Fitting sample-specific variance estimator for condition 1 using provided indices...")
                condition1_variance_estimator = SampleVarianceEstimator(
                    eps=self.eps
                )
                # Set a flag to indicate this estimator is called from DifferentialExpression
                condition1_variance_estimator._called_from_differential = True
                
                # Check gene count and warn if too many
                if y_condition1.shape[1] > 100:
                    logger.warning(
                        f"Fitting sample variance estimator with {y_condition1.shape[1]:,} genes. "
                        f"This may require significant memory. Consider running with fewer genes "
                        f"(max 100 genes) at a time for better memory efficiency."
                    )
                
                condition1_variance_estimator.fit(
                    X=X_condition1, 
                    Y=y_condition1, 
                    grouping_vector=condition1_sample_indices,
                    ls_factor=ls_factor,
                    estimator_kwargs=sample_estimator_kwargs
                )
                self.variance_predictor1 = condition1_variance_estimator.predict
            
            # Fit variance estimator for condition 2
            if condition2_sample_indices is not None:
                logger.info("Fitting sample-specific variance estimator for condition 2 using provided indices...")
                condition2_variance_estimator = SampleVarianceEstimator(
                    eps=self.eps
                )
                # Set a flag to indicate this estimator is called from DifferentialExpression
                condition2_variance_estimator._called_from_differential = True
                
                # Check gene count and warn if too many
                if y_condition2.shape[1] > 100:
                    logger.warning(
                        f"Fitting sample variance estimator with {y_condition2.shape[1]:,} genes. "
                        f"This may require significant memory. Consider running with fewer genes "
                        f"(max 100 genes) at a time for better memory efficiency."
                    )
                
                condition2_variance_estimator.fit(
                    X=X_condition2, 
                    Y=y_condition2, 
                    grouping_vector=condition2_sample_indices,
                    ls_factor=ls_factor,
                    estimator_kwargs=sample_estimator_kwargs
                )
                self.variance_predictor2 = condition2_variance_estimator.predict
        
        # The fit method now only creates estimators and doesn't compute fold changes
        logger.info("Function estimators fitted. Call predict() to compute fold changes.")
        
        return self
        
    def compute_mahalanobis_distances(
        self, 
        X: np.ndarray, 
        fold_change=None,
        use_landmarks: bool = True,
        landmarks_override: Optional[np.ndarray] = None,
        progress: bool = True
    ):
        """
        Compute Mahalanobis distances for each gene using efficient matrix preparation and batching.
        
        Parameters
        ----------
        X : np.ndarray
            Cell states. Shape (n_cells, n_features).
        fold_change : np.ndarray, optional
            Pre-computed fold change matrix. If None, will compute it.
            Shape (n_cells, n_genes).
        use_landmarks : bool, optional
            Whether to use landmarks for covariance calculation if available, by default True.
        landmarks_override : np.ndarray, optional
            Explicitly provided landmarks to use instead of automatically detected ones, 
            by default None.
        progress : bool, optional
            Whether to show progress bars for gene-wise operations, by default True.
            
        Returns
        -------
        np.ndarray
            Array of Mahalanobis distances for each gene.
        """

        if self.function_predictor1 is None or self.function_predictor2 is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Determine landmarks to use
        landmarks = None
        has_landmarks = False
        
        # Use explicit landmarks if provided
        if landmarks_override is not None:
            landmarks = landmarks_override
            has_landmarks = True
            logger.info(f"Using explicitly provided landmarks with shape {landmarks.shape}")
        # Otherwise check for landmarks from function predictors if enabled
        elif use_landmarks:
            # Check function predictor for landmarks
            if hasattr(self.function_predictor1, 'landmarks') and self.function_predictor1.landmarks is not None:
                landmarks = self.function_predictor1.landmarks
                has_landmarks = True
                logger.info(f"Using landmarks from function_predictor1 with shape {landmarks.shape}")
            # Check estimator for landmarks
            elif (hasattr(self.expression_estimator_condition1, 'landmarks') and 
                  self.expression_estimator_condition1.landmarks is not None):
                landmarks = self.expression_estimator_condition1.landmarks
                has_landmarks = True
                logger.info(f"Using landmarks from expression_estimator_condition1 with shape {landmarks.shape}")
        
        # Determine which points to use for computation
        if has_landmarks and landmarks is not None:
            # For landmark-based approximation, use the actual landmarks
            logger.info(f"Using {len(landmarks):,} landmarks for Mahalanobis computation")
            
            # Get covariance matrices
            cov1 = self.function_predictor1.covariance(landmarks, diag=False)
            cov2 = self.function_predictor2.covariance(landmarks, diag=False)
            
            # We need to use the function predictors to get fold changes at landmark points
            landmarks_pred1 = self.function_predictor1(landmarks)
            landmarks_pred2 = self.function_predictor2(landmarks)
            fold_change_subset = landmarks_pred2 - landmarks_pred1
            
            # Points for sample variance computation
            variance_points = landmarks
        else:
            # Use all points
            logger.info(f"No landmarks used, computing with all {len(X)} points")
            
            # Get covariance matrices
            cov1 = self.function_predictor1.covariance(X, diag=False)
            cov2 = self.function_predictor2.covariance(X, diag=False)
            
            # Use the provided fold_change if available
            if fold_change is not None:
                fold_change_subset = fold_change
            # If provided fold_change is not available, compute it directly
            else:
                condition1_imputed = self.function_predictor1(X)
                condition2_imputed = self.function_predictor2(X)
                fold_change_subset = condition2_imputed - condition1_imputed
                
            # Points for sample variance computation
            variance_points = X
        
        # Average the covariance matrices
        combined_cov = (cov1 + cov2) / 2
        
        # For sample variance, use diag=False to get full covariance matrices
        # Initialize variable to store gene-specific covariance matrices if needed
        gene_specific_covariance = None
        
        if self.use_sample_variance:
            # Add empirical adjustments from sample variance
            
            # Create functions for computing sample variance
            if self.variance_predictor1 is not None:
                try:
                    # Important: use diag=False to get full covariance matrix
                    variance1 = self.variance_predictor1(variance_points, diag=False)
                    if self.variance_predictor2 is not None:
                        variance2 = self.variance_predictor2(variance_points, diag=False)
                        # Add the covariance matrices for complete variance representation
                        combined_variance = variance1 + variance2
                        
                        # Check if we have gene-specific covariance matrices (shape has 3 dimensions)
                        if len(combined_variance.shape) == 3:
                            # We have per-gene covariance matrices with shape (points, points, genes)
                            gene_specific_covariance = combined_variance
                            logger.info(f"Using gene-specific covariance matrices with shape {gene_specific_covariance.shape}")
                        else:
                            # Add the sample variance to the combined covariance from function predictors
                            combined_cov += combined_variance
                            logger.info("Added sample variance covariance matrix to function predictor covariance")
                    else:
                        # Only add variance1 if variance2 is not available
                        if len(variance1.shape) == 3:
                            # We have per-gene covariance matrices
                            gene_specific_covariance = variance1
                            logger.info(f"Using gene-specific covariance matrices from variance1 with shape {gene_specific_covariance.shape}")
                        else:
                            combined_cov += variance1
                            logger.info("Added variance1 covariance matrix to function predictor covariance")
                except Exception as e:
                    error_msg = f"Error computing sample variance from variance_predictor1: {e}."
                    logger.error(error_msg)
                    raise RuntimeError(error_msg) from e
            elif self.variance_predictor2 is not None:
                try:
                    # Important: use diag=False to get full covariance matrix
                    variance2 = self.variance_predictor2(variance_points, diag=False)
                    # Check if we have gene-specific covariance matrices
                    if len(variance2.shape) == 3:
                        # We have per-gene covariance matrices
                        gene_specific_covariance = variance2
                        logger.info(f"Using gene-specific covariance matrices from variance2 with shape {gene_specific_covariance.shape}")
                    else:
                        # Add variance2 to the combined covariance
                        combined_cov += variance2
                        logger.info("Added variance2 covariance matrix to function predictor covariance")
                except Exception as e:
                    error_msg = f"Error computing sample variance from variance_predictor2: {e}."
                    logger.error(error_msg)
                    raise RuntimeError(error_msg) from e
        
        # Choose different approaches based on whether we have gene-specific covariance matrices
        if gene_specific_covariance is not None:
            # Use gene-specific covariance matrices
            logger.info("Computing Mahalanobis distances with gene-specific covariance matrices...")
            
            # Transpose fold_change to get shape (n_genes, n_points) for easier gene-wise processing
            fold_change_transposed = fold_change_subset.T
            
            # For gene-specific covariance, we don't use the combined_cov (shared covariance)
            # since each gene gets its own covariance matrix
            # We still need a dummy placeholder matrix for the compute_mahalanobis_distances function
            prepared_matrix = {
                'is_diagonal': False,
                'diag_values': None,
                'chol': None,
                'matrix_inv': None
            }
            
            try:
                # Compute Mahalanobis distances for all genes using gene-specific covariance
                logger.info(f"Computing Mahalanobis distances for {fold_change_transposed.shape[0]:,} genes with gene-specific covariance matrices...")
                
                # Compute all distances using our utility function with gene-specific covariance
                # This will handle the Cholesky decomposition for each gene individually
                mahalanobis_distances = compute_mahalanobis_distances(
                    diff_values=fold_change_transposed,
                    prepared_matrix=prepared_matrix,  # This is just a placeholder
                    batch_size=self.mahalanobis_batch_size,  
                    jit_compile=self.jit_compile,
                    gene_covariances=gene_specific_covariance,  # Pass gene-specific covariance matrices
                    progress=progress  # Pass progress parameter to control tqdm display
                )
                
                logger.info(f"Successfully computed Mahalanobis distances for {len(mahalanobis_distances):,} genes using gene-specific covariance")
            
            except Exception as e:
                logger.warning(f"Error with gene-specific computation: {e}. Falling back to shared covariance.")
                # Fall back to combined covariance if gene-specific fails
                gene_specific_covariance = None  # Reset to trigger fallback
        
        # If gene-specific approach failed or wasn't available, use the traditional approach
        if gene_specific_covariance is None:
            # Prepare the matrix (compute Cholesky decomposition once)
            logger.info("Preparing covariance matrix for efficient Mahalanobis distance computation...")
            prepared_matrix = prepare_mahalanobis_matrix(
                covariance_matrix=combined_cov,
                diag_adjustments=None,  # No need for separate diagonal adjustments, they're already in combined_cov
                eps=self.eps,
                jit_compile=self.jit_compile
            )
            
            # Transpose fold_change to get shape (n_genes, n_points) for easier gene-wise processing
            fold_change_transposed = fold_change_subset.T
            
            # Compute Mahalanobis distances for all genes at once using batched computation
            logger.info(f"Computing Mahalanobis distances for {fold_change_transposed.shape[0]:,} genes with shared covariance...")
            
            # Use the mahalanobis_batch_size from the class instance
            batch_size = self.mahalanobis_batch_size
            
            try:
                # Compute all distances using our utility function that handles batching internally
                mahalanobis_distances = compute_mahalanobis_distances(
                    diff_values=fold_change_transposed,
                    prepared_matrix=prepared_matrix,
                    batch_size=batch_size,
                    jit_compile=self.jit_compile,
                    progress=progress  # Pass progress parameter to control tqdm display
                )
                
                logger.info(f"Successfully computed Mahalanobis distances for {len(mahalanobis_distances):,} genes")
            
            except Exception as e:
                error_msg = (f"Failed to compute Mahalanobis distances: {str(e)}. "
                           f"Try manually reducing batch_size or disable Mahalanobis "
                           f"distance calculation with compute_mahalanobis=False")
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e

        return mahalanobis_distances
    
    def predict(
        self, 
        X_new: np.ndarray, 
        compute_mahalanobis: bool = False,
        progress: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Predict gene expression and differential metrics for new points.
        
        This method computes fold changes and related metrics for the provided points.
        It uses internal batching for efficient computation with large datasets.
        
        Parameters
        ----------
        X_new : np.ndarray
            New cell states. Shape (n_cells, n_features).
        compute_mahalanobis : bool, optional
            Whether to compute Mahalanobis distances. This can be computationally expensive,
            so it's optional in the predict method. Default is False.
        progress : bool, optional
            Whether to show progress bars for gene-wise operations, by default True.
            
        Returns
        -------
        dict
            Dictionary containing the predictions:
            - 'condition1_imputed': Imputed expression for condition 1
            - 'condition2_imputed': Imputed expression for condition 2
            - 'fold_change': Fold change between conditions
            - 'fold_change_zscores': Z-scores for the fold changes
            - 'mean_log_fold_change': Mean log fold change across all cells
            - 'mahalanobis_distances': Only if compute_mahalanobis is True
        """
        if self.function_predictor1 is None or self.function_predictor2 is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get batch size for internal batching (from DifferentialExpression class attribute)
        batch_size = getattr(self, 'batch_size', None)
        
        # Define batch processing functions for the heavyweight operations
        def predict_condition1(X_batch):
            return self.function_predictor1(X_batch)
            
        def predict_condition2(X_batch):
            return self.function_predictor2(X_batch)
        
        # Functions for computing uncertainties using batches
        def get_uncertainty1(X_batch):
            return self.function_predictor1.covariance(X_batch, diag=True)
                
        def get_uncertainty2(X_batch):
            return self.function_predictor2.covariance(X_batch, diag=True)
            
        # Functions for computing empirical variances using batches
        # When using SampleVarianceEstimator, just define functions here but don't call them yet
        if self.variance_predictor1 is not None:
            def get_variance1(X_batch):
                # Return zeros with same shape as imputed values
                return np.zeros_like(predict_condition1(X_batch))
        else:
            def get_variance1(X_batch):
                # Return zeros with same shape as imputed values
                return np.zeros_like(predict_condition1(X_batch))
                
        if self.variance_predictor2 is not None:
            def get_variance2(X_batch):
                # Return zeros with same shape as imputed values
                return np.zeros_like(predict_condition2(X_batch))
        else:
            def get_variance2(X_batch):
                # Return zeros with same shape as imputed values
                return np.zeros_like(predict_condition2(X_batch))
        
        # Apply batched processing to each expensive operation
        condition1_imputed = apply_batched(
            predict_condition1, X_new, batch_size=batch_size,
            desc="Predicting condition 1" if progress else None
        )
        
        condition2_imputed = apply_batched(
            predict_condition2, X_new, batch_size=batch_size,
            desc="Predicting condition 2" if progress else None
        )
        
        # Get uncertainties from function predictors
        condition1_uncertainty = apply_batched(
            get_uncertainty1, X_new, batch_size=batch_size,
            desc="Computing uncertainty (condition 1)" if progress else None
        )
        
        condition2_uncertainty = apply_batched(
            get_uncertainty2, X_new, batch_size=batch_size,
            desc="Computing uncertainty (condition 2)" if progress else None
        )
        
        # Compute fold change
        fold_change = condition2_imputed - condition1_imputed
        
        # Compute fold change statistics
        lfc_stds = np.std(fold_change, axis=0)
        
        # Ensure uncertainties have the right shape for broadcasting
        if len(condition1_uncertainty.shape) == 1:
            # Reshape to (n_samples, 1) for broadcasting with fold_change
            condition1_uncertainty = condition1_uncertainty[:, np.newaxis]
        if len(condition2_uncertainty.shape) == 1:
            # Reshape to (n_samples, 1) for broadcasting with fold_change
            condition2_uncertainty = condition2_uncertainty[:, np.newaxis]
            
        # Convert uncertainties to numpy arrays if needed
        condition1_uncertainty = np.asarray(condition1_uncertainty)
        condition2_uncertainty = np.asarray(condition2_uncertainty)
        
        # Combined uncertainty - function predictor uncertainties
        variance = condition1_uncertainty + condition2_uncertainty
        
        # Compute bidirectionality
        bidirectionality = np.minimum(
            np.quantile(fold_change, 0.95, axis=0),
            -np.quantile(fold_change, 0.05, axis=0)
        )
        
        # Compute mean log fold change
        mean_log_fold_change = np.mean(fold_change, axis=0)
        
        result = {
            'condition1_imputed': condition1_imputed,
            'condition2_imputed': condition2_imputed,
            'fold_change': fold_change,
            'lfc_stds': lfc_stds,
            'bidirectionality': bidirectionality
        }
        
        # Compute Mahalanobis distances if requested
        if compute_mahalanobis:
            logger.info("Computing Mahalanobis distances...")
            
            # Pass the progress parameter to control tqdm display
            mahalanobis_distances = self.compute_mahalanobis_distances(
                X=X_new, 
                fold_change=fold_change,
                progress=progress  # Use the progress parameter here
            )
            
            result['mahalanobis_distances'] = mahalanobis_distances
            
            # Use sample variance from mahalanobis calculation for z-scores
            # This will get the full covariance matrix with diag=False
            if self.use_sample_variance:
                # Z-scores will be computed after this
                pass
            else:
                # If we're not using sample variance, just use the function predictor uncertainties
                stds = np.sqrt(variance + self.eps)
                fold_change_zscores = fold_change / stds
                result['fold_change_zscores'] = fold_change_zscores
                result['mean_log_fold_change'] = mean_log_fold_change
        else:
            # If we don't compute mahalanobis, we still need the fold change z-scores
            if self.use_sample_variance:
                # Get empirical variances if enabled
                try:
                    # Let's call compute_mahalanobis_distances and only use its variance calculations
                    # This will get the variance with the whole landmark logic
                    # Pass the progress parameter to control tqdm display
                    temp_result = self.compute_mahalanobis_distances(
                        X=X_new, 
                        fold_change=fold_change,
                        progress=progress  # Use the progress parameter here
                    )
                    # The actual distances will be discarded, we just want the side effect of computing the variance
                except Exception as e:
                    error_msg = f"Error in sample variance computation: {e}."
                    logger.error(error_msg)
                    raise RuntimeError(error_msg) from e
            
            # Compute z-scores using function predictor uncertainties
            stds = np.sqrt(variance + self.eps)
            fold_change_zscores = fold_change / stds
            result['fold_change_zscores'] = fold_change_zscores
            result['mean_log_fold_change'] = mean_log_fold_change
        
        return result