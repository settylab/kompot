"""Differential analysis for gene expression and abundance."""

import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple, List, Optional, Union, Dict, Any
import logging
from scipy.stats import norm as normal
from scipy.sparse import csr_matrix

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
        
        # Set random seed for reproducible landmark selection if specified
        if random_state is not None:
            np.random.seed(random_state)
        
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
        self.density_estimator_condition1 = None
        self.density_estimator_condition2 = None
        self.density_predictor1 = density_predictor1
        self.density_predictor2 = density_predictor2
        
    def fit(
        self, 
        X_condition1: np.ndarray, 
        X_condition2: np.ndarray,
        **density_kwargs
    ):
        """
        Fit density estimators and compute differential abundance metrics.
        
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
        # Combine data for predictions
        X_all = np.vstack([X_condition1, X_condition2])
        
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
                # Note: compute_landmarks has its own internal random seed, 
                # we've already set np.random.seed in the constructor if random_state was provided
                landmarks = compute_landmarks(X_all, gp_type='fixed', n_landmarks=self.n_landmarks)
                estimator_defaults['landmarks'] = landmarks
                estimator_defaults['gp_type'] = 'fixed'
                
            # Fit density estimators for both conditions
            logger.info("Fitting density estimator for condition 1...")
            self.density_estimator_condition1 = mellon.DensityEstimator(**estimator_defaults)
            self.density_estimator_condition1.fit(X_condition1)
            self.density_predictor1 = self.density_estimator_condition1.predict
            
            logger.info("Fitting density estimator for condition 2...")
            self.density_estimator_condition2 = mellon.DensityEstimator(**estimator_defaults)
            self.density_estimator_condition2.fit(X_condition2)
            self.density_predictor2 = self.density_estimator_condition2.predict
        
        # Compute log densities and uncertainties
        logger.info("Computing log densities and uncertainties...")
        self.log_density_condition1 = self.density_predictor1(X_all, normalize=True)
        self.log_density_condition2 = self.density_predictor2(X_all, normalize=True)
        
        self.log_density_uncertainty_condition1 = self.density_predictor1.uncertainty(X_all)
        self.log_density_uncertainty_condition2 = self.density_predictor2.uncertainty(X_all)
        
        # Compute log fold change and uncertainty
        self.log_fold_change = self.log_density_condition2 - self.log_density_condition1
        self.log_fold_change_uncertainty = self.log_density_uncertainty_condition1 + self.log_density_uncertainty_condition2
        
        # Compute z-scores
        sd = np.sqrt(self.log_fold_change_uncertainty + 1e-16)
        self.log_fold_change_zscore = self.log_fold_change / sd
        
        # Compute p-values
        self.log_fold_change_pvalue = np.minimum(
            normal.logcdf(self.log_fold_change_zscore), 
            normal.logcdf(-self.log_fold_change_zscore)
        ) + np.log(2)
        
        # Determine direction of change based on thresholds
        self.log_fold_change_direction = np.full(len(self.log_fold_change), 'neutral', dtype=object)
        significant = (np.abs(self.log_fold_change) > self.log_fold_change_threshold) & \
                     (self.log_fold_change_pvalue < np.log(self.pvalue_threshold))
        
        self.log_fold_change_direction[significant & (self.log_fold_change > 0)] = 'up'
        self.log_fold_change_direction[significant & (self.log_fold_change < 0)] = 'down'
        
        return self
    
    def predict(self, X_new: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict log density and log fold change for new points.
        
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
        
        return {
            'log_density_condition1': log_density_condition1,
            'log_density_condition2': log_density_condition2,
            'log_fold_change': log_fold_change,
            'log_fold_change_uncertainty': log_fold_change_uncertainty,
            'log_fold_change_zscore': log_fold_change_zscore,
            'log_fold_change_pvalue': log_fold_change_pvalue,
            'log_fold_change_direction': log_fold_change_direction
        }


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
            Pre-computed densities for both conditions. If provided, should contain:
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
        """
        self.n_landmarks = n_landmarks
        self.use_empirical_variance = use_empirical_variance
        self.compute_weighted_fold_change = compute_weighted_fold_change
        self.eps = eps
        self.jit_compile = jit_compile
        self.random_state = random_state
        
        # Set random seed for reproducible landmark selection if specified
        if random_state is not None:
            np.random.seed(random_state)
        
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
        compute_mahalanobis: bool = True,
        compute_differential_abundance: bool = None,
        **function_kwargs
    ):
        """
        Fit expression estimators and compute differential expression metrics.
        
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
        compute_mahalanobis : bool, optional
            Whether to compute Mahalanobis distances, by default True.
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
        # Combine data for predictions
        X_all = np.vstack([X_condition1, X_condition2])
        
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
                jit_compile=self.jit_compile
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
                # Note: compute_landmarks has its own internal random seed, 
                # we've already set np.random.seed in the constructor if random_state was provided
                landmarks = compute_landmarks(X_all, gp_type='fixed', n_landmarks=self.n_landmarks)
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
        
        # Predict expression for all points
        logger.info("Predicting gene expression for both conditions...")
        self.condition1_imputed = self.function_predictor1(X_all)
        self.condition2_imputed = self.function_predictor2(X_all)
        
        # Compute uncertainties
        logger.info("Computing uncertainties...")
        # Ensure covariance is computed with with_uncertainty=True
        self.condition1_uncertainty = self.function_predictor1.covariance(X_all, diag=True)
        self.condition2_uncertainty = self.function_predictor2.covariance(X_all, diag=True)
        
        # Compute fold change
        self.fold_change = self.condition2_imputed - self.condition1_imputed
        
        # Compute fold change statistics
        self.lfc_stds = np.std(self.fold_change, axis=0)
        self.bidirectionality = np.minimum(
            np.quantile(self.fold_change, 0.95, axis=0),
            -np.quantile(self.fold_change, 0.05, axis=0)
        )
        
        # Compute empirical error squares if needed
        if self.use_empirical_variance:
            logger.info("Computing empirical error squares...")
            # For condition 1
            mask1 = np.arange(len(X_condition1))
            error_squares1 = np.square(
                self.condition1_imputed[mask1] - y_condition1
            )
            fest1 = mellon.FunctionEstimator(
                **{**estimator_defaults, 'ls': estimator_defaults.get('ls', 1.0) * 2.0}
            )
            fest1.fit(X_condition1, error_squares1)
            self.condition1_error_squares = np.clip(
                fest1.predict(X_all),
                self.eps,
                None
            )
            
            # For condition 2
            mask2 = np.arange(len(X_condition2))
            error_squares2 = np.square(
                self.condition2_imputed[mask2] - y_condition2
            )
            fest2 = mellon.FunctionEstimator(
                **{**estimator_defaults, 'ls': estimator_defaults.get('ls', 1.0) * 2.0}
            )
            fest2.fit(X_condition2, error_squares2)
            self.condition2_error_squares = np.clip(
                fest2.predict(X_all),
                self.eps,
                None
            )
        
        # Compute fold change z-scores
        variance_base = self.condition1_uncertainty + self.condition2_uncertainty
        if self.use_empirical_variance and self.condition1_error_squares is not None:
            diag_adjustments = (self.condition1_error_squares + self.condition2_error_squares) / 2
            variance = variance_base + diag_adjustments
        else:
            variance = variance_base
            
        # Ensure variance has the right shape for broadcasting
        if len(variance.shape) == 1:
            # Reshape to (n_samples, 1) for broadcasting with fold_change
            variance = variance[:, np.newaxis]
            
        stds = np.sqrt(variance + self.eps)
        self.fold_change_zscores = self.fold_change / stds
        
        # Compute Mahalanobis distances if requested
        if compute_mahalanobis:
            logger.info("Computing Mahalanobis distances...")
            self._compute_mahalanobis_distances(X_all)

        # Compute weighted mean log fold change if needed
        if self.compute_weighted_fold_change:
            if self.differential_abundance is not None:
                # Use differential_abundance object
                log_density_condition1 = self.differential_abundance.log_density_condition1
                log_density_condition2 = self.differential_abundance.log_density_condition2
            elif self.precomputed_densities is not None:
                # Use precomputed densities
                log_density_condition1 = self.precomputed_densities['log_density_condition1']
                log_density_condition2 = self.precomputed_densities['log_density_condition2']
            elif self.density_predictor1 is not None and self.density_predictor2 is not None:
                # Use density predictors
                log_density_condition1 = self.density_predictor1(X_all, normalize=True)
                log_density_condition2 = self.density_predictor2(X_all, normalize=True)
            else:
                # Skip weighted fold change calculation
                logger.warning("Cannot compute weighted fold change: no density information available")
                return self
                
            # Compute weights from density differences
            log_density_diff = np.exp(
                np.abs(log_density_condition2 - log_density_condition1)
            )
            
            # Weight the fold changes by density difference
            weighted_fold_change = self.fold_change * log_density_diff[:, np.newaxis]
            self.weighted_mean_log_fold_change = np.sum(weighted_fold_change, axis=0) / np.sum(log_density_diff)
        
        return self
        
    def _compute_mahalanobis_distances(self, X: np.ndarray):
        """
        Compute Mahalanobis distances for each gene.
        
        Parameters
        ----------
        X : np.ndarray
            Cell states. Shape (n_cells, n_features).
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
            fold_change_subset = self.fold_change
        
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
            
        # Compute Mahalanobis distance for each gene
        n_genes = fold_change_subset.shape[1]
        mahalanobis_distances = np.zeros(n_genes)
        
        for gene in range(n_genes):
            gene_diff = fold_change_subset[:, gene]
            mahalanobis_distances[gene] = compute_mahalanobis_distance(
                gene_diff, 
                combined_cov,
                diag_adjustments=diag_adjust[:, gene] if diag_adjust is not None else None,
                eps=self.eps,
                jit_compile=self.jit_compile
            )
            
        self.mahalanobis_distances = mahalanobis_distances
        
    def predict(
        self, 
        X_new: np.ndarray, 
        density_predictions: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Predict gene expression and differential metrics for new points.
        
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
            
        Returns
        -------
        dict
            Dictionary containing the predictions:
            - 'condition1_imputed': Imputed expression for condition 1
            - 'condition2_imputed': Imputed expression for condition 2
            - 'fold_change': Fold change between conditions
            - 'fold_change_zscores': Z-scores for the fold changes
            - 'weighted_mean_log_fold_change': Only if compute_weighted_fold_change is True
              and density information is available
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
        
        # Compute fold change z-scores
        variance_base = condition1_uncertainty + condition2_uncertainty
        
        # Ensure variance has the right shape for broadcasting
        if len(variance_base.shape) == 1:
            # Reshape to (n_samples, 1) for broadcasting with fold_change
            variance_base = variance_base[:, np.newaxis]
            
        stds = np.sqrt(variance_base + self.eps)
        fold_change_zscores = fold_change / stds
        
        result = {
            'condition1_imputed': condition1_imputed,
            'condition2_imputed': condition2_imputed,
            'fold_change': fold_change,
            'fold_change_zscores': fold_change_zscores
        }
        
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
            
            if density_data is not None:
                log_density_diff = np.exp(
                    np.abs(
                        density_data['log_density_condition2'] - 
                        density_data['log_density_condition1']
                    )
                )
                
                # Weight the fold changes by density difference
                weighted_fold_change = fold_change * log_density_diff[:, np.newaxis]
                result['weighted_mean_log_fold_change'] = np.sum(weighted_fold_change, axis=0) / np.sum(log_density_diff)
        
        return result