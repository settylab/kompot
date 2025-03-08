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
        log_fold_change_threshold: float = 1.7,
        pvalue_threshold: float = 1e-3,
        n_landmarks: Optional[int] = None,
        jit_compile: bool = False,
        density_predictor1: Optional[Any] = None,
        density_predictor2: Optional[Any] = None,
        random_state: Optional[int] = None,
        batch_size: int = None,
    ):
        """
        Initialize DifferentialAbundance.
        
        Parameters
        ----------
        log_fold_change_threshold : float, optional
            Threshold for considering a log fold change significant, by default 1.7.
        pvalue_threshold : float, optional
            Threshold for considering a p-value significant, by default 1e-3.
        n_landmarks : int, optional
            Number of landmarks to use for approximation. If None, use all points, by default None.
        jit_compile : bool, optional
            Whether to use JAX just-in-time compilation, by default False.
        density_predictor1 : Any, optional
            Precomputed density predictor for condition 1, typically from DensityEstimator.predict
        density_predictor2 : Any, optional
            Precomputed density predictor for condition 2, typically from DensityEstimator.predict
        random_state : int, optional
            Random seed for reproducible landmark selection when n_landmarks is specified.
            Controls the random selection of points when using approximation, by default None.
        batch_size : int, optional
            Number of samples to process at once during prediction to manage memory usage.
            If None or 0, all samples will be processed at once. Default is 0.
        """
        self.log_fold_change_threshold = log_fold_change_threshold
        self.pvalue_threshold = pvalue_threshold
        self.n_landmarks = n_landmarks
        self.jit_compile = jit_compile
        self.random_state = random_state
        self.batch_size = batch_size
        
        # Store random_state for reproducible landmark selection if specified
        # We don't need to set np.random.seed here anymore as we'll pass the 
        # random_state directly to compute_landmarks
        
        # Store condition sizes and indices
        self.n_condition1 = None
        self.n_condition2 = None
        self.condition1_indices = None
        self.condition2_indices = None
        
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
        
    def fit(
        self, 
        X_condition1: np.ndarray, 
        X_condition2: np.ndarray,
        landmarks: Optional[np.ndarray] = None,
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
        **density_kwargs : dict
            Additional arguments to pass to the DensityEstimator.
            
        Returns
        -------
        self
            The fitted instance.
        """
        # Store original condition sizes for indexing
        self.n_condition1 = len(X_condition1)
        self.n_condition2 = len(X_condition2)
        self.condition1_indices = np.arange(self.n_condition1)
        self.condition2_indices = np.arange(self.n_condition1, self.n_condition1 + self.n_condition2)
        
        if self.density_predictor1 is None or self.density_predictor2 is None:
            # Configure density estimator defaults
            estimator_defaults = {
                'd_method': 'fractal',
                'predictor_with_uncertainty': True,
                'optimizer': 'advi',
            }
            
            # Update defaults with user-provided values
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
        
        return self
    
    def predict(
        self, 
        X_new: np.ndarray,
        log_fold_change_threshold: Optional[float] = None,
        pvalue_threshold: Optional[float] = None
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
            
        Returns
        -------
        dict
            Dictionary containing the predictions:
            - 'log_density_condition1': Log density for condition 1
            - 'log_density_condition2': Log density for condition 2
            - 'log_fold_change': Log fold change between conditions
            - 'log_fold_change_uncertainty': Uncertainty in the log fold change
            - 'log_fold_change_zscore': Z-scores for the log fold change
            - 'log_fold_change_pvalue': P-values for the log fold change
            - 'log_fold_change_direction': Direction of change ('up', 'down', or 'neutral')
        """
        if self.density_predictor1 is None or self.density_predictor2 is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Use provided thresholds if specified, otherwise use class defaults
        if log_fold_change_threshold is None:
            log_fold_change_threshold = self.log_fold_change_threshold
        if pvalue_threshold is None:
            pvalue_threshold = self.pvalue_threshold
        
        # Get batch size
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
        
        # Apply batched processing to the expensive operations
        log_density_condition1 = apply_batched(
            compute_density1, 
            X_new, 
            batch_size=batch_size, 
            desc="Computing density (condition 1)"
        )
        
        log_density_condition2 = apply_batched(
            compute_density2, 
            X_new, 
            batch_size=batch_size,
            desc="Computing density (condition 2)"
        )
        
        log_density_uncertainty_condition1 = apply_batched(
            compute_uncertainty1, 
            X_new, 
            batch_size=batch_size,
            desc="Computing uncertainty (condition 1)"
        )
        
        log_density_uncertainty_condition2 = apply_batched(
            compute_uncertainty2, 
            X_new, 
            batch_size=batch_size,
            desc="Computing uncertainty (condition 2)"
        )
        
        # The rest of the computation is lightweight and can be done all at once
        # Compute log fold change and uncertainty
        log_fold_change = log_density_condition2 - log_density_condition1
        log_fold_change_uncertainty = log_density_uncertainty_condition1 + log_density_uncertainty_condition2
        
        # Compute z-scores
        sd = np.sqrt(log_fold_change_uncertainty + 1e-16)
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
    Compute local sample variances of gene expressions.
    
    This class manages the computation of empirical variance by fitting function estimators
    for each group in the data and computing the variance between their predictions.
    
    Attributes
    ----------
    group_predictors : Dict
        Dictionary of prediction functions for each group.
    """
    
    def __init__(
        self,
        eps: float = 1e-12,
        jit_compile: bool = True,
        batch_size: int = 500,
    ):
        """
        Initialize the SampleVarianceEstimator.
        
        Parameters
        ----------
        eps : float, optional
            Small constant for numerical stability, by default 1e-12.
        jit_compile : bool, optional
            Whether to use JAX just-in-time compilation, by default True.
        batch_size : int, optional
            Number of samples to process at once during prediction to manage memory usage.
            If None or 0, all samples will be processed at once. Default is 500.
        """
        self.eps = eps
        self.jit_compile = jit_compile
        self.batch_size = batch_size
        
        # Will be populated during fit
        self.group_predictors = {}
        self.group_centroids = {}
        self._predict_variance_jit = None
    
    def fit(
        self, 
        X: np.ndarray,
        Y: np.ndarray, 
        grouping_vector: np.ndarray,
        min_cells: int = 10,
        function_estimator_kwargs: Dict = None
    ):
        """
        Fit function estimators for each group in the data and store only their predictors.
        
        Parameters
        ----------
        X : np.ndarray
            Cell states. Shape (n_cells, n_features).
        Y : np.ndarray
            Gene expression values. Shape (n_cells, n_genes).
        grouping_vector : np.ndarray
            Vector specifying which group each cell belongs to. Shape (n_cells,).
        min_cells : int
            Minimum number of cells for group to train a function estimator. Default is 10.
        function_estimator_kwargs : Dict, optional
            Additional arguments to pass to FunctionEstimator constructor.
            
        Returns
        -------
        self
            The fitted instance.
        """
        if function_estimator_kwargs is None:
            function_estimator_kwargs = {}
        
        # Get unique groups
        unique_groups = np.unique(grouping_vector)
        
        logger.info(f"Found {len(unique_groups):,} unique groups for variance estimation")
        
        # Organize data by groups
        group_indices = {
            group_id: np.where(grouping_vector == group_id)[0]
            for group_id in unique_groups
        }
        
        # Train function estimators for each group and store only their predictors
        logger.info("Training group-specific function estimators...")
        
        for group_id, indices in group_indices.items():
            if len(indices) >= min_cells:  # Only train if we have enough data points
                logger.info(f"Training estimator for group {group_id} with {len(indices)} cells")
                X_subset = X[indices]
                Y_subset = Y[indices]
                
                # Create and train estimator
                estimator = mellon.FunctionEstimator(**function_estimator_kwargs)
                estimator.fit(X_subset, Y_subset)
                
                # Store only the predictor function, not the full estimator
                self.group_predictors[group_id] = estimator.predict
                
                # Immediately delete the estimator to free memory
                del estimator
            else:
                logger.warning(f"Skipping group {group_id} (only {len(indices):,} cells < min_cells={min_cells:,})")
        
        return self
    
    def predict(self, X_new: np.ndarray) -> np.ndarray:
        """
        Predict empirical variance for new points using JAX.
        
        This method uses internal batching to handle large datasets efficiently.
        
        Parameters
        ----------
        X_new : np.ndarray
            New cell states to predict. Shape (n_cells, n_features).
            
        Returns
        -------
        np.ndarray
            Empirical variance for each new point. Shape (n_cells, n_genes).
        """
        if not self.group_predictors:
            raise ValueError("Model not fitted. Call fit() first.")
            
        n_cells = len(X_new)
        
        # Get the shape of the output from the first predictor
        # This assumes all predictors produce outputs of the same shape
        first_predictor = list(self.group_predictors.values())[0]
        test_pred = first_predictor([X_new[0]])
        n_genes = test_pred.shape[1] if len(test_pred.shape) > 1 else 1
        
        # Convert input to JAX array to ensure compatibility with JAX functions
        X_new_jax = jnp.array(X_new)
        
        # If we have no predictors, return zeros
        if not self.group_predictors:
            variance = np.zeros((n_cells, n_genes))
            logger.warning("No group predictors available. Returning zeros for variance.")
            return variance
        
        # Compile the prediction function if we're using JIT and haven't already
        if self.jit_compile and self._predict_variance_jit is None:
            # Define our variance computation function
            def compute_variance_from_predictors(X, predictors_list):
                # Map each predictor to get predictions
                predictions = [predictor(X) for predictor in predictors_list]
                # Stack the predictions
                stacked = jnp.stack(predictions, axis=0)
                # Compute variance across groups (axis 0)
                return jnp.var(stacked, axis=0)
            
            # JIT compile the function
            self._predict_variance_jit = jax.jit(compute_variance_from_predictors)
        
        # Get list of predictors
        predictors_list = list(self.group_predictors.values())
        
        # Define a function for batched processing
        def process_batch(X_batch):
            # Use the JIT-compiled function if available
            if self.jit_compile and self._predict_variance_jit is not None:
                batch_variance = self._predict_variance_jit(X_batch, predictors_list)
                return np.array(batch_variance)
            else:
                # Get predictions from each group predictor
                all_group_predictions = []
                for predictor in predictors_list:
                    group_predictions = predictor(X_batch)
                    all_group_predictions.append(group_predictions)
                
                # Stack predictions and compute variance using JAX
                stacked_predictions = jnp.stack(all_group_predictions, axis=0)
                batch_variance = jnp.var(stacked_predictions, axis=0)
                # Convert back to numpy for compatibility
                return np.array(batch_variance)
        
        # Apply batched processing
        batch_size = self.batch_size
        variance = apply_batched(
            process_batch, 
            X_new_jax, 
            batch_size=batch_size, 
            desc="Computing variance across groups"
        )
        
        return variance


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
                
            # Fit expression estimators for both conditions
            logger.info("Fitting expression estimator for condition 1...")
            self.expression_estimator_condition1 = mellon.FunctionEstimator(**estimator_defaults)
            self.expression_estimator_condition1.fit(X_condition1, y_condition1)
            self.function_predictor1 = self.expression_estimator_condition1.predict
            
            # Update ls for condition 2 based on condition 1 if not provided
            if ls is None and 'ls' not in function_kwargs:
                # Get ls from condition 1 and use it for condition 2
                ls_cond1 = self.function_predictor1.cov_func.ls
                estimator_defaults['ls'] = ls_cond1 * 1.0  # Can scale if needed
            
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
                condition1_variance_estimator.fit(
                    X_condition1, 
                    y_condition1, 
                    condition1_sample_indices,
                    function_estimator_kwargs=sample_estimator_kwargs
                )
                self.variance_predictor1 = condition1_variance_estimator.predict
            
            # Fit variance estimator for condition 2
            if condition2_sample_indices is not None:
                logger.info("Fitting sample-specific variance estimator for condition 2 using provided indices...")
                condition2_variance_estimator = SampleVarianceEstimator(
                    eps=self.eps
                )
                condition2_variance_estimator.fit(
                    X_condition2, 
                    y_condition2, 
                    condition2_sample_indices,
                    function_estimator_kwargs=sample_estimator_kwargs
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
        landmarks_override: Optional[np.ndarray] = None
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
        
        # Average the covariance matrices
        combined_cov = (cov1 + cov2) / 2
        
        # Determine diagonal adjustments for variance
        diag_adjust = None
        
        # Add empirical adjustments if needed
        if self.use_sample_variance:
            if has_landmarks and landmarks is not None:
                # Use variance predictors if available for landmarks
                if self.variance_predictor1 is not None and self.variance_predictor2 is not None:
                    landmarks_variance1 = self.variance_predictor1(landmarks)
                    landmarks_variance2 = self.variance_predictor2(landmarks)
                    diag_adjust = (landmarks_variance1 + landmarks_variance2) / 2
                    logger.info("Using predicted variance from landmarks")
            else:
                # If we're not using landmarks, estimate variance at the given points
                if self.variance_predictor1 is not None and self.variance_predictor2 is not None:
                    variance1 = self.variance_predictor1(X)
                    variance2 = self.variance_predictor2(X)
                    diag_adjust = variance1 + variance2
                    logger.info("Using predicted variance from all points")
        
        # Process diag_adjust to get correct shape if needed
        if diag_adjust is not None:
            # Ensure diag_adjust is the right shape
            if len(diag_adjust.shape) > 1 and diag_adjust.shape[1] > 1:
                # If diag_adjust is per-cell and per-gene, take mean across cells
                diag_adjust_avg = np.mean(diag_adjust, axis=0)
                # If still multi-dimensional, take the first column (first gene)
                if len(diag_adjust_avg.shape) > 1:
                    diag_adjust_avg = diag_adjust_avg[:, 0]
            else:
                diag_adjust_avg = diag_adjust
            
            # Use the processed adjustment
            diag_adjust = diag_adjust_avg
        
        # Prepare the matrix (compute Cholesky decomposition once)
        logger.info("Preparing covariance matrix for efficient Mahalanobis distance computation...")
        prepared_matrix = prepare_mahalanobis_matrix(
            covariance_matrix=combined_cov,
            diag_adjustments=diag_adjust,
            eps=self.eps,
            jit_compile=self.jit_compile
        )
        
        # Transpose fold_change to get shape (n_genes, n_points) for easier gene-wise processing
        fold_change_transposed = fold_change_subset.T
        
        # Compute Mahalanobis distances for all genes at once using batched computation
        logger.info(f"Computing Mahalanobis distances for {fold_change_transposed.shape[0]} genes...")
        
        # Use the mahalanobis_batch_size from the class instance
        batch_size = self.mahalanobis_batch_size
        
        try:
            # Compute all distances using our utility function that handles batching internally
            mahalanobis_distances = compute_mahalanobis_distances(
                diff_values=fold_change_transposed,
                prepared_matrix=prepared_matrix,
                batch_size=batch_size,
                jit_compile=self.jit_compile
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
        compute_mahalanobis: bool = False
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
        
        # Get batch size for internal batching
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
        # The variance predictor returns variance per cell, which is added to uncertainty
        if self.variance_predictor1 is not None:
            def get_variance1(X_batch):
                return self.variance_predictor1(X_batch)
        else:
            def get_variance1(X_batch):
                # Return zeros with same shape as imputed values
                return np.zeros_like(predict_condition1(X_batch))
                
        if self.variance_predictor2 is not None:
            def get_variance2(X_batch):
                return self.variance_predictor2(X_batch)
        else:
            def get_variance2(X_batch):
                # Return zeros with same shape as imputed values
                return np.zeros_like(predict_condition2(X_batch))
        
        # Apply batched processing to each expensive operation
        condition1_imputed = apply_batched(
            predict_condition1, X_new, batch_size=batch_size,
            desc="Predicting condition 1"
        )
        
        condition2_imputed = apply_batched(
            predict_condition2, X_new, batch_size=batch_size,
            desc="Predicting condition 2"
        )
        
        # Get uncertainties from function predictors
        condition1_uncertainty = apply_batched(
            get_uncertainty1, X_new, batch_size=batch_size,
            desc="Computing uncertainty (condition 1)"
        )
        
        condition2_uncertainty = apply_batched(
            get_uncertainty2, X_new, batch_size=batch_size,
            desc="Computing uncertainty (condition 2)"
        )
        
        # Get empirical variances if enabled
        if self.use_sample_variance:
            condition1_variance = apply_batched(
                get_variance1, X_new, batch_size=batch_size,
                desc="Computing empirical variance (condition 1)"
            )
            
            condition2_variance = apply_batched(
                get_variance2, X_new, batch_size=batch_size,
                desc="Computing empirical variance (condition 2)"
            )
        else:
            # Skip empirical variance calculation if not needed
            # Create simple zeros arrays instead of using zeros_like which can fail with inhomogeneous shapes
            condition1_variance = 0
            condition2_variance = 0
        
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
        
        # Combined uncertainty - add empirical variance if enabled
        variance = condition1_uncertainty + condition2_uncertainty
        
        # Add empirical variance component if enabled
        if self.use_sample_variance:
            # Convert to numpy arrays if needed
            condition1_variance = np.asarray(condition1_variance)
            condition2_variance = np.asarray(condition2_variance)
            
            # Ensure proper shape for broadcasting
            if len(condition1_variance.shape) == 1:
                condition1_variance = condition1_variance[:, np.newaxis]
            if len(condition2_variance.shape) == 1:
                condition2_variance = condition2_variance[:, np.newaxis]
                
            variance += condition1_variance + condition2_variance
            
        # Compute z-scores
        stds = np.sqrt(variance + self.eps)
        fold_change_zscores = fold_change / stds
        
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
            'fold_change_zscores': fold_change_zscores,
            'mean_log_fold_change': mean_log_fold_change,
            'lfc_stds': lfc_stds,
            'bidirectionality': bidirectionality
        }
        
        # Compute Mahalanobis distances if requested
        if compute_mahalanobis:
            logger.info("Computing Mahalanobis distances...")
            
            # Simply call our existing method with the fold change we already computed
            mahalanobis_distances = self.compute_mahalanobis_distances(
                X=X_new, 
                fold_change=fold_change
            )
            
            result['mahalanobis_distances'] = mahalanobis_distances
        
        return result