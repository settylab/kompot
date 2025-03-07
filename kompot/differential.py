"""Differential analysis for gene expression and abundance."""

import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple, List, Optional, Union, Dict, Any
import logging
from scipy.stats import norm as normal
from scipy.sparse import csr_matrix
from tqdm.auto import tqdm

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
    fold_change : np.ndarray
        Log fold change between conditions (condition2 - condition1).
    fold_change_zscores : np.ndarray
        Z-scores for the fold changes.
    mahalanobis_distances : np.ndarray
        Mahalanobis distances for each gene.
    weighted_mean_log_fold_change : np.ndarray
        Weighted mean log fold change, using density differences as weights.
    """
    
    def __init__(
        self,
        n_landmarks: Optional[int] = None,
        use_empirical_variance: bool = False,
        compute_weighted_fold_change: bool = True,
        differential_abundance: Optional["DifferentialAbundance"] = None,
        precomputed_densities: Optional[Dict[str, np.ndarray]] = None,
        eps: float = 1e-12,
        jit_compile: bool = False,
        function_predictor1: Optional[Any] = None,
        function_predictor2: Optional[Any] = None,
        density_predictor1: Optional[Any] = None,
        density_predictor2: Optional[Any] = None,
        random_state: Optional[int] = None,
        batch_size: int = 100,
    ):
        """
        Initialize DifferentialExpression.
        
        Parameters
        ----------
        n_landmarks : int, optional
            Number of landmarks to use for approximation. If None, use all points, by default None.
        use_empirical_variance : bool, optional
            Whether to use empirical variance for uncertainty estimation, by default False.
        compute_weighted_fold_change : bool, optional
            Whether to compute weighted mean log fold change, by default True.
        differential_abundance : DifferentialAbundance, optional
            Pre-computed differential abundance object. If None, a new one will be created
            when needed, by default None.
        precomputed_densities : Dict[str, np.ndarray], optional
            Pre-computed densities for both conditions. Only needed for wheighted log-fold change. If provided, should contain:
            - 'log_density_condition1': Log density for condition 1
            - 'log_density_condition2': Log density for condition 2
        eps : float, optional
            Small constant for numerical stability, by default 1e-12.
        jit_compile : bool, optional
            Whether to use JAX just-in-time compilation, by default False.
        function_predictor1 : Any, optional
            Precomputed function predictor for condition 1, typically from FunctionEstimator.predict
        function_predictor2 : Any, optional
            Precomputed function predictor for condition 2, typically from FunctionEstimator.predict
        density_predictor1 : Any, optional
            Precomputed density predictor for condition 1, typically from DensityEstimator.predict
        density_predictor2 : Any, optional
            Precomputed density predictor for condition 2, typically from DensityEstimator.predict
        random_state : int, optional
            Random seed for reproducible landmark selection when n_landmarks is specified.
            Controls the random selection of points when using approximation, by default None.
        batch_size : int, optional
            Number of genes to process in each batch during Mahalanobis distance computation.
            Smaller values use less memory but are slower, by default 100. Increase for
            faster computation if you have sufficient memory.
        """
        self.n_landmarks = n_landmarks
        self.use_empirical_variance = use_empirical_variance
        self.compute_weighted_fold_change = compute_weighted_fold_change
        self.eps = eps
        self.jit_compile = jit_compile
        self.random_state = random_state
        self.batch_size = batch_size
        
        # Store random_state for reproducible landmark selection if specified
        # We don't need to set np.random.seed here anymore as we'll pass the 
        # random_state directly to compute_landmarks
        
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
        self.weighted_mean_log_fold_change = None
        self.lfc_stds = None
        self.bidirectionality = None
        
        # Function estimators or predictors
        self.expression_estimator_condition1 = None
        self.expression_estimator_condition2 = None
        self.function_predictor1 = function_predictor1
        self.function_predictor2 = function_predictor2
        
        # Differential abundance and density information
        self.differential_abundance = differential_abundance
        self.precomputed_densities = precomputed_densities
        self.density_predictor1 = density_predictor1
        self.density_predictor2 = density_predictor2
        
    def fit(
        self, 
        X_condition1: np.ndarray,
        y_condition1: np.ndarray, 
        X_condition2: np.ndarray,
        y_condition2: np.ndarray,
        sigma: float = 1.0,
        ls: Optional[float] = None,
        compute_differential_abundance: bool = None,
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
        compute_differential_abundance : bool, optional
            Whether to compute differential abundance if not already provided. If None,
            will compute only if compute_weighted_fold_change is True and neither
            differential_abundance nor precomputed_densities were provided.
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
        
        # Determine if we need to compute/use density information for weighting
        use_density_information = self.compute_weighted_fold_change
        have_density_predictors = self.density_predictor1 is not None and self.density_predictor2 is not None
        
        if compute_differential_abundance is None:
            compute_differential_abundance = (
                use_density_information and 
                self.differential_abundance is None and 
                self.precomputed_densities is None and
                not have_density_predictors
            )
        
        if compute_differential_abundance:
            # Compute differential abundance to get density information for weighting
            self.differential_abundance = DifferentialAbundance(
                n_landmarks=self.n_landmarks,
                jit_compile=self.jit_compile,
                random_state=self.random_state
            )
            self.differential_abundance.fit(X_condition1, X_condition2)
        
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
                # We need to compute error squares for landmarks
                # This is a bit of a simplification, ideally we'd predict the error squares at landmarks
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
        density_predictions: Optional[Dict[str, np.ndarray]] = None,
        cell_condition_labels: Optional[np.ndarray] = None,
        compute_mahalanobis: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Predict gene expression and differential metrics for new points.
        
        This method now computes all fold changes and related metrics, which were
        previously computed in the fit method.
        
        Parameters
        ----------
        X_new : np.ndarray
            New cell states. Shape (n_cells, n_features).
        density_predictions : Dict[str, np.ndarray], optional
            Pre-computed density predictions for the new points. If provided and 
            compute_weighted_fold_change is True, will be used to compute weighted
            mean log fold change. Should contain:
            - 'log_density_condition1': Log density for condition 1
            - 'log_density_condition2': Log density for condition 2
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
            - 'weighted_mean_log_fold_change': Only if compute_weighted_fold_change is True
              and density information is available
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
            # These computations are only supported for predictions when
            # we've already computed error squares in a previous call to predict
            if hasattr(self, 'condition1_error_squares') and self.condition1_error_squares is not None:
                # Use the existing error squares models to predict for new points
                # We'd need to have methods to predict error squares for new points
                # For now, use a simpler approach - use the original condition uncertainty
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
        
        # Compute weighted fold change if needed and possible
        if self.compute_weighted_fold_change:
            density_data = None
            
            # Try to get density information from various sources
            if density_predictions is not None:
                density_data = density_predictions
            elif self.density_predictor1 is not None and self.density_predictor2 is not None:
                density_data = {
                    'log_density_condition1': self.density_predictor1(X_new, normalize=True),
                    'log_density_condition2': self.density_predictor2(X_new, normalize=True)
                }
            elif self.differential_abundance is not None:
                # Use the differential abundance predictor
                da_preds = self.differential_abundance.predict(X_new)
                density_data = {
                    'log_density_condition1': da_preds['log_density_condition1'],
                    'log_density_condition2': da_preds['log_density_condition2']
                }
            elif self.precomputed_densities is not None:
                # Try to use precomputed densities, but warn if shapes don't match
                if len(self.precomputed_densities['log_density_condition1']) == len(X_new):
                    density_data = self.precomputed_densities
                else:
                    logger.warning("Precomputed densities shape doesn't match X_new. Skipping weighted fold change.")
            
            if density_data is not None:
                # Calculate log density difference directly
                log_density_diff = np.exp(
                    np.abs(density_data['log_density_condition2'] - density_data['log_density_condition1'])
                )
                
                # Use the standalone utility function to compute the weighted mean with pre-computed difference
                result['weighted_mean_log_fold_change'] = compute_weighted_mean_fold_change(
                    fold_change,
                    log_density_diff=log_density_diff
                )
                
                # Update class attribute for backward compatibility
                if hasattr(self, 'condition1_indices') and self.condition1_indices is not None:
                    if len(X_new) == (self.n_condition1 + self.n_condition2):
                        self.weighted_mean_log_fold_change = result['weighted_mean_log_fold_change']
        
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