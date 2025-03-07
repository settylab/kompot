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
from .utils import compute_mahalanobis_distance, find_landmarks

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
        """
        self.log_fold_change_threshold = log_fold_change_threshold
        self.pvalue_threshold = pvalue_threshold
        self.n_landmarks = n_landmarks
        self.jit_compile = jit_compile
        self.random_state = random_state
        
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
            
            # Prepare landmarks if requested
            if self.n_landmarks is not None:
                # Use mellon's compute_landmarks function to get properly distributed landmarks
                # Pass the random_state parameter directly to ensure reproducible results
                X_combined = np.vstack([X_condition1, X_condition2])
                landmarks = compute_landmarks(
                    X_combined, 
                    n_landmarks=self.n_landmarks,
                    random_state=self.random_state
                )
                estimator_defaults['landmarks'] = landmarks
                
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
    
    def predict(self, X_new: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict log density and log fold change for new points.
        
        This method now computes all fold changes and related metrics, which were
        previously computed in the fit method.
        
        Parameters
        ----------
        X_new : np.ndarray
            New cell states to predict. Shape (n_cells, n_features).
            
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
        
        # Compute log densities and uncertainties
        log_density_condition1 = self.density_predictor1(X_new, normalize=True)
        log_density_condition2 = self.density_predictor2(X_new, normalize=True)
        
        log_density_uncertainty_condition1 = self.density_predictor1.uncertainty(X_new)
        log_density_uncertainty_condition2 = self.density_predictor2.uncertainty(X_new)
        
        # Compute log fold change and uncertainty
        log_fold_change = log_density_condition2 - log_density_condition1
        log_fold_change_uncertainty = log_density_uncertainty_condition1 + log_density_uncertainty_condition2
        
        # Compute z-scores
        sd = np.sqrt(log_fold_change_uncertainty + 1e-16)
        log_fold_change_zscore = log_fold_change / sd
        
        # Compute p-values
        log_fold_change_pvalue = np.minimum(
            normal.logcdf(log_fold_change_zscore), 
            normal.logcdf(-log_fold_change_zscore)
        ) + np.log(2)
        
        # Determine direction of change based on thresholds
        log_fold_change_direction = np.full(len(log_fold_change), 'neutral', dtype=object)
        significant = (np.abs(log_fold_change) > self.log_fold_change_threshold) & \
                     (log_fold_change_pvalue < np.log(self.pvalue_threshold))
        
        log_fold_change_direction[significant & (log_fold_change > 0)] = 'up'
        log_fold_change_direction[significant & (log_fold_change < 0)] = 'down'
        
        # Store predictions for the current points
        # This is to maintain compatibility with code that accesses these attributes
        if hasattr(self, 'condition1_indices') and self.condition1_indices is not None:
            # If called after fit(), we'll update the class-level attributes for backward compatibility
            # Only update attributes if we're predicting on the original training points
            if len(X_new) == (self.n_condition1 + self.n_condition2):
                self.log_density_condition1 = log_density_condition1
                self.log_density_condition2 = log_density_condition2
                self.log_density_uncertainty_condition1 = log_density_uncertainty_condition1
                self.log_density_uncertainty_condition2 = log_density_uncertainty_condition2
                self.log_fold_change = log_fold_change
                self.log_fold_change_uncertainty = log_fold_change_uncertainty
                self.log_fold_change_zscore = log_fold_change_zscore
                self.log_fold_change_pvalue = log_fold_change_pvalue
                self.log_fold_change_direction = log_fold_change_direction
        
        # Compute mean log fold change
        mean_log_fold_change = np.mean(log_fold_change)
        
        result = {
            'log_density_condition1': log_density_condition1,
            'log_density_condition2': log_density_condition2,
            'log_fold_change': log_fold_change,
            'log_fold_change_uncertainty': log_fold_change_uncertainty,
            'log_fold_change_zscore': log_fold_change_zscore,
            'log_fold_change_pvalue': log_fold_change_pvalue,
            'log_fold_change_direction': log_fold_change_direction,
            'mean_log_fold_change': mean_log_fold_change
        }
        
        return result
        
    def get_condition1_results(self) -> Dict[str, np.ndarray]:
        """
        Return results for condition 1 cells only.
        
        Returns
        -------
        dict
            Dictionary containing results specific to condition 1 cells.
        """
        if self.n_condition1 is None or self.condition1_indices is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        return {
            'log_density_condition1': self.log_density_condition1[:self.n_condition1],
            'log_density_condition2': self.log_density_condition2[:self.n_condition1],
            'log_fold_change': self.log_fold_change[:self.n_condition1],
            'log_fold_change_uncertainty': self.log_fold_change_uncertainty[:self.n_condition1],
            'log_fold_change_zscore': self.log_fold_change_zscore[:self.n_condition1],
            'log_fold_change_pvalue': self.log_fold_change_pvalue[:self.n_condition1],
            'log_fold_change_direction': self.log_fold_change_direction[:self.n_condition1],
        }
    
    def get_condition2_results(self) -> Dict[str, np.ndarray]:
        """
        Return results for condition 2 cells only.
        
        Returns
        -------
        dict
            Dictionary containing results specific to condition 2 cells.
        """
        if self.n_condition2 is None or self.condition2_indices is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        return {
            'log_density_condition1': self.log_density_condition1[self.n_condition1:],
            'log_density_condition2': self.log_density_condition2[self.n_condition1:],
            'log_fold_change': self.log_fold_change[self.n_condition1:],
            'log_fold_change_uncertainty': self.log_fold_change_uncertainty[self.n_condition1:],
            'log_fold_change_zscore': self.log_fold_change_zscore[self.n_condition1:],
            'log_fold_change_pvalue': self.log_fold_change_pvalue[self.n_condition1:],
            'log_fold_change_direction': self.log_fold_change_direction[self.n_condition1:],
        }


class EmpiricVarianceEstimator:
    """
    Compute empirical variance for differential expression.
    
    This class manages the computation of empirical variance using sample indices for both conditions.
    It can train function estimators for each sample group or use nearest-neighbor lookup for 
    empirical variance calculation.
    
    Attributes
    ----------
    condition1_sample_indices : np.ndarray
        Sample indices for the first condition.
    condition2_sample_indices : np.ndarray
        Sample indices for the second condition.
    X_condition1 : np.ndarray
        Cell states for the first condition. Shape (n_cells, n_features).
    X_condition2 : np.ndarray
        Cell states for the second condition. Shape (n_cells, n_features).
    y_condition1 : np.ndarray
        Gene expression values for the first condition. Shape (n_cells, n_genes).
    y_condition2 : np.ndarray
        Gene expression values for the second condition. Shape (n_cells, n_genes).
    """
    
    def __init__(
        self,
        condition1_sample_indices: Optional[np.ndarray] = None,
        condition2_sample_indices: Optional[np.ndarray] = None,
        n_neighbors: int = 5,
        use_subset_estimators: bool = True,
        eps: float = 1e-12,
    ):
        """
        Initialize the EmpiricVarianceEstimator.
        
        Parameters
        ----------
        condition1_sample_indices : np.ndarray, optional
            Sample indices for first condition. Unique values define sample groups.
        condition2_sample_indices : np.ndarray, optional
            Sample indices for second condition. Unique values define sample groups.
        n_neighbors : int, optional
            Number of nearest neighbors to use for variance estimation, by default 5.
        use_subset_estimators : bool, optional
            Whether to train separate estimators for each sample group, by default True.
            If False, will use nearest neighbor lookup instead.
        eps : float, optional
            Small constant for numerical stability, by default 1e-12.
        """
        self.condition1_sample_indices = condition1_sample_indices
        self.condition2_sample_indices = condition2_sample_indices
        self.n_neighbors = n_neighbors
        self.use_subset_estimators = use_subset_estimators
        self.eps = eps
        
        # Will be populated during fit
        self.X_condition1 = None
        self.X_condition2 = None
        self.y_condition1 = None
        self.y_condition2 = None
        
        # Sample group organization
        self.condition1_sample_groups = {}
        self.condition2_sample_groups = {}
        
        # For subset-based estimators approach
        self.condition1_sample_estimators = {}
        self.condition2_sample_estimators = {}
        self.condition1_error_squares = None
        self.condition2_error_squares = None
        
        # For NN-based approach
        self.condition1_nn = None
        self.condition2_nn = None
        self.condition1_raw_errors = None
        self.condition2_raw_errors = None
        
    def fit(
        self, 
        X_condition1: np.ndarray,
        y_condition1: np.ndarray, 
        X_condition2: np.ndarray,
        y_condition2: np.ndarray,
        function_predictor1: Callable,
        function_predictor2: Callable,
        function_estimator_kwargs: Dict = None
    ):
        """
        Fit empirical variance estimators for both conditions.
        
        Parameters
        ----------
        X_condition1 : np.ndarray
            Cell states for the first condition. Shape (n_cells, n_features).
        y_condition1 : np.ndarray
            Gene expression values for the first condition. Shape (n_cells, n_genes).
        X_condition2 : np.ndarray
            Cell states for the second condition. Shape (n_cells, n_features).
        y_condition2 : np.ndarray
            Gene expression values for the second condition. Shape (n_cells, n_genes).
        function_predictor1 : Callable
            Function predictor for the first condition to compute prediction errors.
        function_predictor2 : Callable
            Function predictor for the second condition to compute prediction errors.
        function_estimator_kwargs : Dict, optional
            Additional arguments to pass to FunctionEstimator constructor.
            
        Returns
        -------
        self
            The fitted instance.
        """
        # Store the data for later use
        self.X_condition1 = X_condition1
        self.X_condition2 = X_condition2
        self.y_condition1 = y_condition1
        self.y_condition2 = y_condition2
        
        if function_estimator_kwargs is None:
            function_estimator_kwargs = {}
        
        # Check if sample indices are provided
        if self.condition1_sample_indices is None or self.condition2_sample_indices is None:
            logger.warning("No sample indices provided. Empirical variance estimation will be limited.")
            return self
        
        # Validate sample indices
        if len(self.condition1_sample_indices) != len(X_condition1):
            raise ValueError("condition1_sample_indices length must match X_condition1 length")
        if len(self.condition2_sample_indices) != len(X_condition2):
            raise ValueError("condition2_sample_indices length must match X_condition2 length")
            
        # Get unique sample groups
        unique_samples1 = np.unique(self.condition1_sample_indices)
        unique_samples2 = np.unique(self.condition2_sample_indices)
        
        logger.info(f"Found {len(unique_samples1)} unique sample groups for condition 1")
        logger.info(f"Found {len(unique_samples2)} unique sample groups for condition 2")
        
        # Organize data by sample groups
        self.condition1_sample_groups = {
            sample_id: np.where(self.condition1_sample_indices == sample_id)[0]
            for sample_id in unique_samples1
        }
        
        self.condition2_sample_groups = {
            sample_id: np.where(self.condition2_sample_indices == sample_id)[0]
            for sample_id in unique_samples2
        }
        
        # Compute predictions and raw errors for each condition
        logger.info("Computing prediction errors for condition 1...")
        condition1_preds = function_predictor1(X_condition1)
        self.condition1_raw_errors = (y_condition1 - condition1_preds) ** 2
        
        logger.info("Computing prediction errors for condition 2...")
        condition2_preds = function_predictor2(X_condition2)
        self.condition2_raw_errors = (y_condition2 - condition2_preds) ** 2
        
        # If using subset estimators, train function estimators for each sample group
        if self.use_subset_estimators:
            logger.info("Training sample-specific function estimators for empirical variance...")
            
            # Train estimators for condition 1 sample groups
            for sample_id, indices in self.condition1_sample_groups.items():
                if len(indices) > 10:  # Only train if we have enough data points
                    logger.info(f"Training estimator for condition 1, sample group {sample_id} with {len(indices)} cells")
                    X_subset = X_condition1[indices]
                    errors_subset = self.condition1_raw_errors[indices]
                    
                    # Create and train estimator
                    estimator = mellon.FunctionEstimator(**function_estimator_kwargs)
                    estimator.fit(X_subset, errors_subset)
                    self.condition1_sample_estimators[sample_id] = estimator
                else:
                    logger.warning(f"Skipping sample group {sample_id} for condition 1 (only {len(indices)} cells)")
            
            # Train estimators for condition 2 sample groups
            for sample_id, indices in self.condition2_sample_groups.items():
                if len(indices) > 10:  # Only train if we have enough data points
                    logger.info(f"Training estimator for condition 2, sample group {sample_id} with {len(indices)} cells")
                    X_subset = X_condition2[indices]
                    errors_subset = self.condition2_raw_errors[indices]
                    
                    # Create and train estimator
                    estimator = mellon.FunctionEstimator(**function_estimator_kwargs)
                    estimator.fit(X_subset, errors_subset)
                    self.condition2_sample_estimators[sample_id] = estimator
                else:
                    logger.warning(f"Skipping sample group {sample_id} for condition 2 (only {len(indices)} cells)")
        else:
            # Initialize nearest neighbor models for fast lookup
            logger.info("Building nearest neighbor models for empirical variance estimation...")
            self.condition1_nn = NearestNeighbors(n_neighbors=min(self.n_neighbors, len(X_condition1)))
            self.condition1_nn.fit(X_condition1)
            
            self.condition2_nn = NearestNeighbors(n_neighbors=min(self.n_neighbors, len(X_condition2)))
            self.condition2_nn.fit(X_condition2)
            
        return self
        
    def predict_variance(self, X_new: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict empirical variance for new points.
        
        Parameters
        ----------
        X_new : np.ndarray
            New cell states to predict. Shape (n_cells, n_features).
            
        Returns
        -------
        tuple
            (condition1_error_squares, condition2_error_squares) - Empirical error squares
            for each condition. Each has shape (n_cells, n_genes).
        """
        if self.X_condition1 is None or self.X_condition2 is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        n_cells = len(X_new)
        n_genes = self.y_condition1.shape[1] if len(self.y_condition1.shape) > 1 else 1
        
        # Initialize output arrays
        error_squares1 = np.zeros((n_cells, n_genes))
        error_squares2 = np.zeros((n_cells, n_genes))
        
        if self.use_subset_estimators and (self.condition1_sample_estimators or self.condition2_sample_estimators):
            # Use trained sample estimators for prediction
            logger.info("Using sample-specific estimators for variance prediction...")
            
            # First, find the closest sample group for each point in X_new for condition 1
            if self.condition1_sample_estimators:
                sample_centroids1 = {
                    sample_id: np.mean(self.X_condition1[indices], axis=0)
                    for sample_id, indices in self.condition1_sample_groups.items()
                    if sample_id in self.condition1_sample_estimators
                }
                
                # Compute distances to each centroid
                for i, x in enumerate(X_new):
                    # Find closest sample group
                    min_dist = float('inf')
                    closest_sample = None
                    
                    for sample_id, centroid in sample_centroids1.items():
                        dist = np.sum((x - centroid) ** 2)
                        if dist < min_dist:
                            min_dist = dist
                            closest_sample = sample_id
                    
                    # Use the estimator for the closest sample group
                    if closest_sample is not None and closest_sample in self.condition1_sample_estimators:
                        error_squares1[i] = self.condition1_sample_estimators[closest_sample].predict([x])[0]
            
            # Same for condition 2
            if self.condition2_sample_estimators:
                sample_centroids2 = {
                    sample_id: np.mean(self.X_condition2[indices], axis=0)
                    for sample_id, indices in self.condition2_sample_groups.items()
                    if sample_id in self.condition2_sample_estimators
                }
                
                # Compute distances to each centroid
                for i, x in enumerate(X_new):
                    # Find closest sample group
                    min_dist = float('inf')
                    closest_sample = None
                    
                    for sample_id, centroid in sample_centroids2.items():
                        dist = np.sum((x - centroid) ** 2)
                        if dist < min_dist:
                            min_dist = dist
                            closest_sample = sample_id
                    
                    # Use the estimator for the closest sample group
                    if closest_sample is not None and closest_sample in self.condition2_sample_estimators:
                        error_squares2[i] = self.condition2_sample_estimators[closest_sample].predict([x])[0]
        
        elif self.condition1_nn is not None and self.condition2_nn is not None:
            # Use nearest neighbor lookup
            logger.info("Using nearest neighbor lookup for variance prediction...")
            
            # Find k nearest neighbors for each point in X_new for condition 1
            distances1, indices1 = self.condition1_nn.kneighbors(X_new)
            
            # Calculate weighted average of errors using inverse distance weighting
            for i in range(n_cells):
                # Add small constant to avoid division by zero
                weights1 = 1.0 / (distances1[i] + self.eps)
                weights1 = weights1 / np.sum(weights1)
                
                # Compute weighted average of errors from nearest neighbors
                for j, (idx, w) in enumerate(zip(indices1[i], weights1)):
                    error_squares1[i] += w * self.condition1_raw_errors[idx]
            
            # Same for condition 2
            distances2, indices2 = self.condition2_nn.kneighbors(X_new)
            
            for i in range(n_cells):
                weights2 = 1.0 / (distances2[i] + self.eps)
                weights2 = weights2 / np.sum(weights2)
                
                for j, (idx, w) in enumerate(zip(indices2[i], weights2)):
                    error_squares2[i] += w * self.condition2_raw_errors[idx]
        
        else:
            logger.warning("No suitable variance estimation method available. Using zeros.")
        
        return error_squares1, error_squares2
    
    def get_sample_variance_summary(self) -> Dict[str, Dict]:
        """
        Get a summary of variance statistics per sample group.
        
        Returns
        -------
        dict
            Dictionary with sample variance statistics for both conditions.
        """
        if self.condition1_raw_errors is None or self.condition2_raw_errors is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        summary = {
            "condition1": {},
            "condition2": {}
        }
        
        # Compute statistics for condition 1 sample groups
        for sample_id, indices in self.condition1_sample_groups.items():
            errors = self.condition1_raw_errors[indices]
            summary["condition1"][sample_id] = {
                "n_cells": len(indices),
                "mean_error": np.mean(errors),
                "median_error": np.median(errors),
                "std_error": np.std(errors),
            }
        
        # Compute statistics for condition 2 sample groups
        for sample_id, indices in self.condition2_sample_groups.items():
            errors = self.condition2_raw_errors[indices]
            summary["condition2"][sample_id] = {
                "n_cells": len(indices),
                "mean_error": np.mean(errors),
                "median_error": np.median(errors),
                "std_error": np.std(errors),
            }
            
        return summary


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
            
        # Calculate the density difference (weight) for each cell
        log_density_diff = np.exp(
            np.abs(log_density_condition2 - log_density_condition1)
        )
    elif hasattr(log_density_diff, 'to_numpy'):
        # Convert pandas Series to numpy arrays if needed
        log_density_diff = log_density_diff.to_numpy()
    
    # Create a weights array with shape (n_cells, 1) for broadcasting
    weights = log_density_diff.reshape(-1, 1)
    
    # Weight the fold changes by density difference
    weighted_fold_change = fold_change * weights
    
    # Compute the weighted mean
    return np.sum(weighted_fold_change, axis=0) / np.sum(log_density_diff)


class DifferentialExpression:
    """
    Compute differential expression between two conditions.
    
    This class analyzes the differences in gene expression between two conditions
    (e.g., control vs. treatment) using imputation, Mahalanobis distance, and 
    log fold change analysis.
    
    Attributes
    ----------
    condition1_imputed : np.ndarray
        Imputed gene expression for the first condition.
    condition2_imputed : np.ndarray
        Imputed gene expression for the second condition.
    condition1_uncertainty : np.ndarray
        Uncertainty in the first condition's imputation.
    condition2_uncertainty : np.ndarray
        Uncertainty in the second condition's imputation.
    condition1_error_squares : np.ndarray
        Squared prediction errors for the first condition (when use_empirical_variance=True).
    condition2_error_squares : np.ndarray
        Squared prediction errors for the second condition (when use_empirical_variance=True).
    error_squares_estimator1 : mellon.FunctionEstimator
        Function estimator trained to predict squared errors for condition 1 (when use_empirical_variance=True).
    error_squares_estimator2 : mellon.FunctionEstimator
        Function estimator trained to predict squared errors for condition 2 (when use_empirical_variance=True).
    fold_change : np.ndarray
        Log fold change between conditions (condition2 - condition1).
    fold_change_zscores : np.ndarray
        Z-scores for the fold changes.
    mahalanobis_distances : np.ndarray
        Mahalanobis distances for each gene.
    """
    
    def __init__(
        self,
        n_landmarks: Optional[int] = None,
        use_empirical_variance: bool = False,
        eps: float = 1e-12,
        jit_compile: bool = False,
        function_predictor1: Optional[Any] = None,
        function_predictor2: Optional[Any] = None,
        random_state: Optional[int] = None,
        batch_size: int = 100,
        # Sample-specific empirical variance parameters
        condition1_sample_indices: Optional[np.ndarray] = None,
        condition2_sample_indices: Optional[np.ndarray] = None,
        use_sample_specific_variance: bool = False,
        sample_variance_n_neighbors: int = 5,
        sample_variance_use_estimators: bool = True,
    ):
        """
        Initialize DifferentialExpression.
        
        Parameters
        ----------
        n_landmarks : int, optional
            Number of landmarks to use for approximation. If None, use all points, by default None.
        use_empirical_variance : bool, optional
            Whether to use empirical variance for uncertainty estimation, by default False.
            When True, the class will train additional function estimators to model and predict
            the squared prediction errors at each point, which improves variance estimation
            for fold change significance testing.
        eps : float, optional
            Small constant for numerical stability, by default 1e-12.
        jit_compile : bool, optional
            Whether to use JAX just-in-time compilation, by default False.
        function_predictor1 : Any, optional
            Precomputed function predictor for condition 1, typically from FunctionEstimator.predict
        function_predictor2 : Any, optional
            Precomputed function predictor for condition 2, typically from FunctionEstimator.predict
        random_state : int, optional
            Random seed for reproducible landmark selection when n_landmarks is specified.
            Controls the random selection of points when using approximation, by default None.
        batch_size : int, optional
            Number of genes to process in each batch during Mahalanobis distance computation.
            Smaller values use less memory but are slower, by default 100. Increase for
            faster computation if you have sufficient memory.
        condition1_sample_indices : np.ndarray, optional
            Sample indices for first condition. Used for sample-specific variance estimation.
            Unique values in this array define different sample groups.
        condition2_sample_indices : np.ndarray, optional
            Sample indices for second condition. Used for sample-specific variance estimation.
            Unique values in this array define different sample groups.
        use_sample_specific_variance : bool, optional
            Whether to use sample-specific variance estimation based on the provided
            sample indices, by default False. When True, overrides use_empirical_variance.
        sample_variance_n_neighbors : int, optional
            Number of nearest neighbors to use for sample-specific variance estimation,
            by default 5. Only used when use_sample_specific_variance=True.
        sample_variance_use_estimators : bool, optional
            Whether to train separate estimators for each sample group (True) or use
            nearest neighbor lookup (False), by default True.
        """
        self.n_landmarks = n_landmarks
        self.use_empirical_variance = use_empirical_variance
        self.eps = eps
        self.jit_compile = jit_compile
        self.random_state = random_state
        self.batch_size = batch_size
        
        # Sample-specific empirical variance parameters
        self.condition1_sample_indices = condition1_sample_indices
        self.condition2_sample_indices = condition2_sample_indices
        self.use_sample_specific_variance = use_sample_specific_variance
        self.sample_variance_n_neighbors = sample_variance_n_neighbors
        self.sample_variance_use_estimators = sample_variance_use_estimators
        
        # If sample-specific variance is enabled, make sure empirical variance is also enabled
        if self.use_sample_specific_variance:
            self.use_empirical_variance = True
        
        # Store condition sizes and indices
        self.n_condition1 = None
        self.n_condition2 = None
        self.mean_log_fold_change = None
        
        # These will be populated after fitting
        self.condition1_imputed = None
        self.condition2_imputed = None
        self.condition1_uncertainty = None
        self.condition2_uncertainty = None
        self.condition1_error_squares = None
        self.condition2_error_squares = None
        self.fold_change = None
        self.fold_change_zscores = None
        self.mahalanobis_distances = None
        self.lfc_stds = None
        self.bidirectionality = None
        
        # Function estimators or predictors
        self.expression_estimator_condition1 = None
        self.expression_estimator_condition2 = None
        self.function_predictor1 = function_predictor1
        self.function_predictor2 = function_predictor2
        
        # Error squares estimators (for traditional empirical variance)
        self.error_squares_estimator1 = None
        self.error_squares_estimator2 = None
        
        # Sample-specific empirical variance estimator
        self.empiric_variance_estimator = None
        
    def fit(
        self, 
        X_condition1: np.ndarray,
        y_condition1: np.ndarray, 
        X_condition2: np.ndarray,
        y_condition2: np.ndarray,
        sigma: float = 1.0,
        ls: Optional[float] = None,
        sample_estimator_ls: Optional[float] = None,
        compute_differential_abundance: bool = True,  # Added for backward compatibility
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
        sample_estimator_ls : float, optional
            Length scale for the sample-specific variance estimators. If None, will use
            the same value as ls or it will be estimated, by default None.
        compute_differential_abundance : bool, optional
            Whether to compute differential abundance for weighted fold change. This parameter
            is included for backward compatibility, by default True.
        **function_kwargs : dict
            Additional arguments to pass to the FunctionEstimator.
            
        Returns
        -------
        self
            The fitted instance.
        """
        # Store original condition sizes for indexing
        self.n_condition1 = len(X_condition1)
        self.n_condition2 = len(X_condition2)
        
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
            
            # Prepare landmarks if requested
            if self.n_landmarks is not None:
                # Use mellon's compute_landmarks function to get properly distributed landmarks
                # Pass the random_state parameter directly to ensure reproducible results
                X_combined = np.vstack([X_condition1, X_condition2])
                landmarks = compute_landmarks(
                    X_combined, 
                    gp_type='fixed', 
                    n_landmarks=self.n_landmarks,
                    random_state=self.random_state
                )
                estimator_defaults['landmarks'] = landmarks
                estimator_defaults['gp_type'] = 'fixed'
                
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
            
            # No longer automatically compute density estimators for weighted fold change
            # If the user wants weighted fold change, they should provide their own density estimators
            # or call the standalone compute_weighted_mean_fold_change function
            
            # Handle empirical variance - either sample-specific or traditional
            if self.use_empirical_variance:
                if self.use_sample_specific_variance:
                    # Use the sample-specific variance approach
                    logger.info("Using sample-specific empirical variance estimation...")
                    
                    # Create error variance estimator
                    # Define parameters for sample-specific estimator
                    sample_estimator_params = {
                        'condition1_sample_indices': self.condition1_sample_indices,
                        'condition2_sample_indices': self.condition2_sample_indices,
                        'n_neighbors': self.sample_variance_n_neighbors,
                        'use_subset_estimators': self.sample_variance_use_estimators,
                        'eps': self.eps
                    }
                    
                    # Create estimator
                    self.empiric_variance_estimator = EmpiricVarianceEstimator(**sample_estimator_params)
                    
                    # Create function estimator parameters for sample-specific models
                    sample_estimator_kwargs = estimator_defaults.copy()
                    
                    # Use specific length scale if provided
                    if sample_estimator_ls is not None:
                        sample_estimator_kwargs['ls'] = sample_estimator_ls
                        
                    # Fit the empirical variance estimator
                    self.empiric_variance_estimator.fit(
                        X_condition1, y_condition1,
                        X_condition2, y_condition2,
                        self.function_predictor1, self.function_predictor2,
                        function_estimator_kwargs=sample_estimator_kwargs
                    )
                    
                    # Compute error squares for training data
                    self.condition1_error_squares, self.condition2_error_squares = \
                        self.empiric_variance_estimator.predict_variance(
                            np.vstack([X_condition1, X_condition2])
                        )
                    
                    # Split the error squares back into the respective conditions
                    self.condition1_error_squares = self.condition1_error_squares[:self.n_condition1]
                    self.condition2_error_squares = self.condition2_error_squares[self.n_condition1:]
                    
                    # Log variance summary statistics
                    if self.condition1_sample_indices is not None and self.condition2_sample_indices is not None:
                        variance_summary = self.empiric_variance_estimator.get_sample_variance_summary()
                        logger.info(f"Sample variance summary: {len(variance_summary['condition1'])} groups in condition 1, "
                                   f"{len(variance_summary['condition2'])} groups in condition 2")
                        
                else:
                    # Use the traditional error squares estimators approach
                    logger.info("Using traditional empirical variance estimation...")
                    
                    # Compute predictions and errors for condition 1
                    logger.info("Computing error squares for condition 1...")
                    condition1_imputed = self.function_predictor1(X_condition1)
                    condition1_errors = (y_condition1 - condition1_imputed) ** 2
                    
                    # Fit error squares estimator for condition 1
                    logger.info("Fitting error squares estimator for condition 1...")
                    self.error_squares_estimator1 = mellon.FunctionEstimator(**estimator_defaults)
                    self.error_squares_estimator1.fit(X_condition1, condition1_errors)
                    
                    # Compute predictions and errors for condition 2
                    logger.info("Computing error squares for condition 2...")
                    condition2_imputed = self.function_predictor2(X_condition2)
                    condition2_errors = (y_condition2 - condition2_imputed) ** 2
                    
                    # Fit error squares estimator for condition 2
                    logger.info("Fitting error squares estimator for condition 2...")
                    self.error_squares_estimator2 = mellon.FunctionEstimator(**estimator_defaults)
                    self.error_squares_estimator2.fit(X_condition2, condition2_errors)
                    
                    # Store the error squares for the training data
                    self.condition1_error_squares = self.error_squares_estimator1(X_condition1)
                    self.condition2_error_squares = self.error_squares_estimator2(X_condition2)
        
        # The fit method now only creates estimators and doesn't compute fold changes
        logger.info("Function estimators fitted. Call predict() to compute fold changes.")
        
        return self
        
    def _compute_mahalanobis_distances(self, X: np.ndarray, fold_change=None):
        """
        Compute Mahalanobis distances for each gene.
        
        Parameters
        ----------
        X : np.ndarray
            Cell states. Shape (n_cells, n_features).
        fold_change : np.ndarray, optional
            Pre-computed fold change matrix. If None, will try to use self.fold_change.
            Shape (n_cells, n_genes).
        """
        # Get covariance matrices
        # Determine if we have landmarks
        has_landmarks = False
        landmarks = None
        
        # Check function predictor for landmarks
        if hasattr(self.function_predictor1, 'landmarks') and self.function_predictor1.landmarks is not None:
            has_landmarks = True
            landmarks = self.function_predictor1.landmarks
        # Check estimator for landmarks
        elif (hasattr(self.expression_estimator_condition1, 'landmarks') and 
              self.expression_estimator_condition1.landmarks is not None):
            has_landmarks = True
            landmarks = self.expression_estimator_condition1.landmarks
              
        if has_landmarks and landmarks is not None:
            # For landmark-based approximation, use the actual landmarks
            cov1 = self.function_predictor1.covariance(landmarks, diag=False)
            cov2 = self.function_predictor2.covariance(landmarks, diag=False)
            
            # We need to use the function predictors to get fold changes at landmark points
            landmarks_pred1 = self.function_predictor1(landmarks)
            landmarks_pred2 = self.function_predictor2(landmarks)
            fold_change_subset = landmarks_pred2 - landmarks_pred1
        else:
            # Use all points
            cov1 = self.function_predictor1.covariance(X, diag=False)
            cov2 = self.function_predictor2.covariance(X, diag=False)
            
            # Use the provided fold_change if available
            if fold_change is not None:
                fold_change_subset = fold_change
            # Otherwise, use self.fold_change if it exists and is not None
            elif hasattr(self, 'fold_change') and self.fold_change is not None:
                fold_change_subset = self.fold_change
            # If neither is available, compute it directly (which shouldn't happen with our changes)
            else:
                # This branch should only be executed in rare cases where this method is called directly
                condition1_imputed = self.function_predictor1(X)
                condition2_imputed = self.function_predictor2(X)
                fold_change_subset = condition2_imputed - condition1_imputed
        
        # Average the covariance matrices
        combined_cov = (cov1 + cov2) / 2
        
        # Add empirical adjustments if needed
        if self.use_empirical_variance:
            if has_landmarks and landmarks is not None:
                # First check if we're using sample-specific variance
                if self.use_sample_specific_variance and hasattr(self, 'empiric_variance_estimator') and self.empiric_variance_estimator is not None:
                    # Use the sample-specific empirical variance estimator for landmarks
                    landmarks_error1, landmarks_error2 = self.empiric_variance_estimator.predict_variance(landmarks)
                    diag_adjust = (landmarks_error1 + landmarks_error2) / 2
                # Then check if we have traditional error square estimators
                elif hasattr(self, 'error_squares_estimator1') and self.error_squares_estimator1 is not None:
                    # Predict error squares at landmark points using the trained estimators
                    landmarks_error1 = self.error_squares_estimator1(landmarks)
                    landmarks_error2 = self.error_squares_estimator2(landmarks)
                    diag_adjust = (landmarks_error1 + landmarks_error2) / 2
                else:
                    # Fallback if estimators are not available
                    diag_adjust = np.ones(len(landmarks)) * np.mean(
                        self.condition1_error_squares + self.condition2_error_squares
                    ) / 2
            else:
                diag_adjust = (self.condition1_error_squares + self.condition2_error_squares) / 2
        else:
            diag_adjust = None
            
        # Compute Mahalanobis distance for each gene using batched computation
        n_genes = fold_change_subset.shape[1]
        mahalanobis_distances = np.zeros(n_genes)

        # Use the batch size from the class instance
        batch_size = self.batch_size
        logger.info(f"Using batch size of {batch_size} for Mahalanobis distance computation")

        # Convert to jax arrays for faster processing
        jax_combined_cov = jnp.array(combined_cov)
        
        # Compute the Cholesky decomposition once for all genes
        # Add small constant to diagonal for numerical stability
        adjusted_cov = jax_combined_cov + jnp.eye(jax_combined_cov.shape[0]) * self.eps

        # Try to compute Mahalanobis distances with batching
        try:
            logger.info("Computing Cholesky decomposition...")
            chol_sigma = jnp.linalg.cholesky(adjusted_cov)
            
            # Define optimized computation functions
            def _compute_mahalanobis_single(diffs, chol):
                """Compute Mahalanobis distance with precomputed Cholesky decomposition."""
                # Solve the triangular system
                right_term = jax.scipy.linalg.solve_triangular(chol, diffs, lower=True)
                # Compute the squared distance
                return jnp.sum(right_term**2)
            
            # JIT compile if enabled
            if self.jit_compile:
                compute_fn = jax.jit(_compute_mahalanobis_single)
            else:
                compute_fn = _compute_mahalanobis_single
            
            # Vectorize the function to handle multiple genes at once
            compute_mahalanobis_vmap = jax.vmap(compute_fn, in_axes=(1, None))
            
            # Process genes in batches
            successful = False
            
            try:
                for start_idx in tqdm(range(0, n_genes, batch_size), desc="Computing Mahalanobis distances"):
                    end_idx = min(start_idx + batch_size, n_genes)
                    batch_diffs = jnp.array(fold_change_subset[:, start_idx:end_idx])
                    
                    # Compute distances for this batch
                    batch_results = compute_mahalanobis_vmap(batch_diffs, chol_sigma)
                    
                    # Store results
                    mahalanobis_distances[start_idx:end_idx] = np.sqrt(np.array(batch_results))
                
                successful = True
                logger.info(f"Computed Mahalanobis distances for {n_genes} genes using batched processing")
            
            except Exception as e:
                # Check if it's a memory error (XlaRuntimeError with RESOURCE_EXHAUSTED)
                if "RESOURCE_EXHAUSTED" in str(e) or "Out of memory" in str(e):
                    # If we hit a JAX-specific out-of-memory error, try with a smaller batch size
                    reduced_batch_size = max(1, batch_size // 5)  # Try with 20% of original batch size
                    logger.warning(f"JAX out of memory error with batch size {batch_size}. Trying with reduced batch size {reduced_batch_size}.")
                    
                    try:
                        for start_idx in tqdm(range(0, n_genes, reduced_batch_size), desc="Computing with reduced batch size"):
                            end_idx = min(start_idx + reduced_batch_size, n_genes)
                            batch_diffs = jnp.array(fold_change_subset[:, start_idx:end_idx])
                            
                            # Compute distances for this batch
                            batch_results = compute_mahalanobis_vmap(batch_diffs, chol_sigma)
                            
                            # Store results
                            mahalanobis_distances[start_idx:end_idx] = np.sqrt(np.array(batch_results))
                        
                        successful = True
                        logger.info(f"Successfully completed with reduced batch size {reduced_batch_size}")
                        logger.info(f"SUGGESTION: For future runs with similar data, use batch_size={reduced_batch_size}")
                    except Exception as inner_e:
                        error_msg = (f"Failed with reduced batch size: {str(inner_e)}. "
                                    f"Try manually setting batch_size={max(1, reduced_batch_size // 2)} or "
                                    f"disable Mahalanobis distance calculation with compute_mahalanobis=False")
                        logger.error(error_msg)
                        raise RuntimeError(error_msg) from inner_e
                else:
                    # Log if it's not a memory error
                    error_msg = f"Batched computation error (not memory-related): {str(e)}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg) from e
            
            # If processing failed, provide helpful suggestions rather than falling back to per-gene
            if not successful:
                error_msg = ("Failed to compute Mahalanobis distances. Try manually reducing batch_size "
                            "or disable Mahalanobis distance calculation with compute_mahalanobis=False")
                logger.error(error_msg)
                raise RuntimeError(error_msg)
        
        except Exception as e:
            error_msg = (f"Cholesky decomposition failed: {e}. "
                        f"Try disable Mahalanobis distance calculation with compute_mahalanobis=False")
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
            
        self.mahalanobis_distances = mahalanobis_distances
    
    def predict(
        self, 
        X_new: np.ndarray, 
        cell_condition_labels: Optional[np.ndarray] = None,
        compute_mahalanobis: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Predict gene expression and differential metrics for new points.
        
        This method computes fold changes and related metrics for the provided points.
        
        Parameters
        ----------
        X_new : np.ndarray
            New cell states. Shape (n_cells, n_features).
        cell_condition_labels : np.ndarray, optional
            Optional array specifying which cells belong to which condition. If provided,
            should be an array of integers where 0 indicates condition1 and 1 indicates condition2.
            This allows for condition-specific analysis without relying on the order of cells.
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
              
            If cell_condition_labels is provided, also includes:
            - 'condition1_cells_mean_log_fold_change': Mean log fold change for condition 1 cells
            - 'condition2_cells_mean_log_fold_change': Mean log fold change for condition 2 cells
        """
        if self.function_predictor1 is None or self.function_predictor2 is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Predict expression for both conditions
        condition1_imputed = self.function_predictor1(X_new)
        condition2_imputed = self.function_predictor2(X_new)
        
        # Compute uncertainties
        condition1_uncertainty = self.function_predictor1.covariance(X_new, diag=True)
        condition2_uncertainty = self.function_predictor2.covariance(X_new, diag=True)
        
        # Compute fold change
        fold_change = condition2_imputed - condition1_imputed
        
        # Compute fold change statistics
        lfc_stds = np.std(fold_change, axis=0)
        
        # Compute empirical error squares if needed for variance adjustment
        if self.use_empirical_variance:
            logger.info("Computing empirical error squares for prediction...")
            
            if self.use_sample_specific_variance and hasattr(self, 'empiric_variance_estimator') and self.empiric_variance_estimator is not None:
                # Use the sample-specific empirical variance estimator
                logger.info("Using sample-specific empirical variance estimator for prediction")
                error_squares1, error_squares2 = self.empiric_variance_estimator.predict_variance(X_new)
            # Traditional approach with global estimators 
            elif hasattr(self, 'error_squares_estimator1') and self.error_squares_estimator1 is not None:
                logger.info("Using trained error squares estimators for prediction")
                error_squares1 = self.error_squares_estimator1(X_new)
                error_squares2 = self.error_squares_estimator2(X_new)
            # Otherwise, if we already have error squares computed from before but no estimators
            elif hasattr(self, 'condition1_error_squares') and self.condition1_error_squares is not None:
                # Fall back to base uncertainty as we don't have estimators
                logger.warning("Error squares estimators not available, using base uncertainty.")
                error_squares1 = condition1_uncertainty
                error_squares2 = condition2_uncertainty
            else:
                # Fall back to base uncertainty
                error_squares1 = condition1_uncertainty
                error_squares2 = condition2_uncertainty
                logger.warning("Empirical variance requested but error squares not available. Using base uncertainty.")
        
        # Compute fold change z-scores
        variance_base = condition1_uncertainty + condition2_uncertainty
        
        # Apply empirical variance adjustment if needed
        if self.use_empirical_variance:
            if len(error_squares1.shape) == 1:
                # Reshape for broadcasting
                error_squares1 = error_squares1[:, np.newaxis]
                error_squares2 = error_squares2[:, np.newaxis]
            diag_adjustments = (error_squares1 + error_squares2) / 2
            variance = variance_base + diag_adjustments
        else:
            variance = variance_base
            
        # Ensure variance has the right shape for broadcasting
        if len(variance.shape) == 1:
            # Reshape to (n_samples, 1) for broadcasting with fold_change
            variance = variance[:, np.newaxis]
            
        stds = np.sqrt(variance + self.eps)
        fold_change_zscores = fold_change / stds
        
        # Compute bidirectionality
        bidirectionality = np.minimum(
            np.quantile(fold_change, 0.95, axis=0),
            -np.quantile(fold_change, 0.05, axis=0)
        )
        
        # Compute mean log fold change
        mean_log_fold_change = np.mean(fold_change, axis=0)
        
        # Store predictions for the current points
        # This is to maintain compatibility with code that accesses these attributes
        if hasattr(self, 'condition1_indices') and self.condition1_indices is not None:
            # If called after fit(), we'll update the class-level attributes for backward compatibility
            # Only update attributes if we're predicting on the original training points
            if len(X_new) == (self.n_condition1 + self.n_condition2):
                self.condition1_imputed = condition1_imputed
                self.condition2_imputed = condition2_imputed
                self.condition1_uncertainty = condition1_uncertainty
                self.condition2_uncertainty = condition2_uncertainty
                self.fold_change = fold_change
                self.fold_change_zscores = fold_change_zscores
                self.mean_log_fold_change = mean_log_fold_change
                self.lfc_stds = lfc_stds
                self.bidirectionality = bidirectionality
        
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
            # Pass the already computed fold_change to avoid recomputing it
            # This gets around the issue when fold_change instance attribute isn't set yet
            self._compute_mahalanobis_distances(X_new, fold_change)
            if hasattr(self, 'mahalanobis_distances'):
                result['mahalanobis_distances'] = self.mahalanobis_distances
            
        # If condition labels are provided, compute condition-specific metrics
        if cell_condition_labels is not None:
            # Validate cell_condition_labels
            if len(cell_condition_labels) != len(X_new):
                raise ValueError("cell_condition_labels must have same length as X_new")
            
            # Compute condition-specific metrics
            condition1_mask = cell_condition_labels == 0
            condition2_mask = cell_condition_labels == 1
            
            # Compute mean fold change for each condition
            if np.any(condition1_mask):
                result['condition1_cells_mean_log_fold_change'] = np.mean(fold_change[condition1_mask], axis=0)
            
            if np.any(condition2_mask):
                result['condition2_cells_mean_log_fold_change'] = np.mean(fold_change[condition2_mask], axis=0)
        
        # Weighted fold change computation is now handled by standalone compute_weighted_mean_fold_change function
        
        return result
        
    def get_condition1_results(self) -> Dict[str, np.ndarray]:
        """
        Return results for condition 1 cells only.
        
        Returns
        -------
        dict
            Dictionary containing results specific to condition 1 cells.
        """
        if self.n_condition1 is None or self.condition1_indices is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        return {
            'condition1_imputed': self.condition1_imputed[:self.n_condition1],
            'condition2_imputed': self.condition2_imputed[:self.n_condition1],
            'fold_change': self.fold_change[:self.n_condition1],
            'fold_change_zscores': self.fold_change_zscores[:self.n_condition1],
        }
    
    def get_condition2_results(self) -> Dict[str, np.ndarray]:
        """
        Return results for condition 2 cells only.
        
        Returns
        -------
        dict
            Dictionary containing results specific to condition 2 cells.
        """
        if self.n_condition2 is None or self.condition2_indices is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        return {
            'condition1_imputed': self.condition1_imputed[self.n_condition1:],
            'condition2_imputed': self.condition2_imputed[self.n_condition1:],
            'fold_change': self.fold_change[self.n_condition1:],
            'fold_change_zscores': self.fold_change_zscores[self.n_condition1:],
        }