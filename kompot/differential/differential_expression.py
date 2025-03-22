"""Differential expression analysis."""

import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple, List, Optional, Union, Dict, Any, Callable
import logging
from scipy.stats import norm as normal
import mellon
from mellon.parameters import compute_landmarks

from ..utils import (
    compute_mahalanobis_distance, 
    compute_mahalanobis_distances, 
    find_landmarks
)
from ..batch_utils import apply_batched, is_jax_memory_error
from .sample_variance_estimator import SampleVarianceEstimator

logger = logging.getLogger("kompot")


class DifferentialExpression:
    """
    Compute differential expression between two conditions.
    
    This class analyzes the differences in gene expression between two conditions
    (e.g., control to treatment) using imputation, Mahalanobis distance, and 
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
        eps: float = 1e-8,  # Increased default epsilon for better numerical stability
        jit_compile: bool = False,
        function_predictor1: Optional[Any] = None,
        function_predictor2: Optional[Any] = None,
        variance_predictor1: Optional[Any] = None,
        variance_predictor2: Optional[Any] = None,
        random_state: Optional[int] = None,
        batch_size: int = 500,
        store_arrays_on_disk: Optional[bool] = None,
        disk_storage_dir: Optional[str] = None,
        max_memory_ratio: float = 0.8,
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
            Small constant for numerical stability, by default 1e-8.
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
            Number of cells to process at once during prediction and Mahalanobis distance computation
            to manage memory usage. If None or 0, all samples will be processed at once. Default is 500.
        store_arrays_on_disk : bool, optional
            Whether to store large arrays on disk instead of in memory, by default None.
            If None, it will be determined based on disk_storage_dir (True if provided, False otherwise).
            This is useful for very large datasets with many genes, where covariance
            matrices would otherwise exceed available memory.
        disk_storage_dir : str, optional
            Directory to store arrays on disk. If provided and store_arrays_on_disk is None,
            store_arrays_on_disk will be set to True. If store_arrays_on_disk is False and
            this is provided, a warning will be logged and disk storage will not be used.
        max_memory_ratio : float, optional
            Maximum fraction of available memory that arrays should occupy before
            triggering warnings or enabling disk storage, by default 0.8 (80%).
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
                logger.debug("Sample variance estimation automatically enabled due to presence of variance predictors")
        else:
            self.use_sample_variance = use_sample_variance
        self.batch_size = batch_size
        
        # Determine store_arrays_on_disk based on disk_storage_dir if not explicitly set
        if store_arrays_on_disk is None:
            self.store_arrays_on_disk = disk_storage_dir is not None
        else:
            self.store_arrays_on_disk = store_arrays_on_disk
            
        # Log warning if store_arrays_on_disk is False but disk_storage_dir is provided
        if not self.store_arrays_on_disk and disk_storage_dir is not None:
            logger.warning(
                f"Disk storage directory provided ({disk_storage_dir}) but store_arrays_on_disk is False. "
                f"Arrays will NOT be stored on disk."
            )
            
        self.disk_storage_dir = disk_storage_dir
        self.max_memory_ratio = max_memory_ratio
        
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
        
    def __del__(self):
        """Cleanup method for object deletion."""
        # No cleanup needed as we no longer manage disk storage directly
        pass
        
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

        # Check if sample indices are provided
        have_sample_indices = (condition1_sample_indices is not None or condition2_sample_indices is not None)
        
        # Auto-enable sample variance if sample indices are provided
        if have_sample_indices:
            if self.use_sample_variance is None or self.use_sample_variance_explicit is False:
                self.use_sample_variance = True
                logger.debug("Sample variance estimation automatically enabled due to provided sample indices")
        
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

        # Create or use function predictors
        if self.function_predictor1 is None or self.function_predictor2 is None:
            # Configure function estimator defaults
            estimator_defaults = {
                'sigma': sigma,
                'optimizer': 'advi',
                'predictor_with_uncertainty': True,
            }
            
            # Update defaults with user-provided values, but filter out parameters that
            # aren't meant for FunctionEstimator (like 'progress')
            mellon_function_kwargs = {k: v for k, v in function_kwargs.items() 
                                     if k not in ['progress']}
            estimator_defaults.update(mellon_function_kwargs)
            
            # Use provided landmarks if available, otherwise compute them if requested
            if landmarks is not None:
                logger.info(f"Using provided landmarks with shape {landmarks.shape}")
                estimator_defaults['landmarks'] = landmarks
                estimator_defaults['gp_type'] = 'fixed'
                # Store provided landmarks for future use
                self.computed_landmarks = landmarks
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
        
        
        # Handle sample-specific variance if enabled and sample indices are provided
        if self.use_sample_variance and have_sample_indices:
            logger.debug("Setting up sample variance estimation...")
            
            # Set up function estimator parameters for sample-specific models
            sample_estimator_kwargs = estimator_defaults.copy() if 'estimator_defaults' in locals() else function_kwargs.copy()
            
            # Use specific length scale if provided
            if sample_estimator_ls is not None:
                sample_estimator_kwargs['ls'] = sample_estimator_ls
            else:
                try:
                    sample_estimator_kwargs['ls'] = self.function_predictor1.cov_func.ls
                except AttributeError as e:
                    logger.warning(
                        f"Could not extract ls for sample variance from predictor: {e} "
                        "This could lead to inflated Mahalanobis distances."
                    )
            
            # No memory analysis needed during fit - SampleVarianceEstimator will analyze memory 
            # during predict() only when actually needed (when diag=False)
            
            # Now fit variance estimators with the memory analysis information
            
            # Fit variance estimator for condition 1
            if condition1_sample_indices is not None:
                logger.debug("Fitting sample-specific variance estimator for condition 1 using provided indices...")
                
                condition1_variance_estimator = SampleVarianceEstimator(
                    eps=self.eps,
                    store_arrays_on_disk=self.store_arrays_on_disk,
                    disk_storage_dir=self.disk_storage_dir
                )
                # Set a flag to indicate this estimator is called from DifferentialExpression
                condition1_variance_estimator._called_from_differential = True
                
                # Don't set up disk storage here - SampleVarianceEstimator will create it when needed in predict()
                
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
                logger.debug("Fitting sample-specific variance estimator for condition 2 using provided indices...")
                
                condition2_variance_estimator = SampleVarianceEstimator(
                    eps=self.eps,
                    store_arrays_on_disk=self.store_arrays_on_disk,
                    disk_storage_dir=self.disk_storage_dir
                )
                # Set a flag to indicate this estimator is called from DifferentialExpression
                condition2_variance_estimator._called_from_differential = True
                
                # Don't set up disk storage here - SampleVarianceEstimator will create it when needed in predict()
                
                condition2_variance_estimator.fit(
                    X=X_condition2, 
                    Y=y_condition2, 
                    grouping_vector=condition2_sample_indices,
                    ls_factor=ls_factor,
                    estimator_kwargs=sample_estimator_kwargs
                )
                self.variance_predictor2 = condition2_variance_estimator.predict
        
        # The fit method now only creates estimators and doesn't compute fold changes
        logger.debug("Function estimators fitted. Call predict() to compute fold changes.")
        
        return self
        
    def compute_mahalanobis_distances(
        self, 
        X: np.ndarray, 
        fold_change=None,
        use_landmarks: bool = True,
        landmarks_override: Optional[np.ndarray] = None,
        progress: bool = True
    ) -> np.ndarray:
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
            logger.debug(f"Using explicitly provided landmarks with shape {landmarks.shape}")
        # Otherwise check for landmarks from function predictors if enabled
        elif use_landmarks:
            # Check function predictor for landmarks
            if hasattr(self.function_predictor1, 'landmarks') and self.function_predictor1.landmarks is not None:
                landmarks = self.function_predictor1.landmarks
                has_landmarks = True
                logger.debug(f"Using landmarks from function_predictor1 with shape {landmarks.shape}")
            # Check estimator for landmarks
            elif (hasattr(self.expression_estimator_condition1, 'landmarks') and 
                  self.expression_estimator_condition1.landmarks is not None):
                landmarks = self.expression_estimator_condition1.landmarks
                has_landmarks = True
                logger.debug(f"Using landmarks from expression_estimator_condition1 with shape {landmarks.shape}")
        
        # Determine which points to use for computation
        if has_landmarks and landmarks is not None:
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
            logger.info(f"No landmarks used, computing covariance between all {len(X):,} points.")
            
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
                            # Need to add combined_cov to each gene's covariance slice
                            gene_specific_covariance = combined_variance
                            # Check if combined_variance is a JAX array, if not, ensure combined_cov is numpy array
                            if not isinstance(combined_variance, jax.Array):
                                combined_cov_to_add = np.asarray(combined_cov)
                            else:
                                combined_cov_to_add = combined_cov
                            for g in range(combined_variance.shape[2]):
                                gene_specific_covariance[:, :, g] = combined_variance[:, :, g] + combined_cov_to_add
                            logger.debug(f"Using gene-specific covariance matrices with shape {gene_specific_covariance.shape}")
                        else:
                            # Add the sample variance to the combined covariance from function predictors
                            combined_cov += combined_variance
                            logger.debug("Added sample variance covariance matrix to function predictor covariance")
                    else:
                        # Only add variance1 if variance2 is not available
                        if len(variance1.shape) == 3:
                            # We have per-gene covariance matrices
                            # Need to add combined_cov to each gene's covariance slice
                            gene_specific_covariance = variance1
                            # Check if variance1 is a JAX array, if not, ensure combined_cov is numpy array
                            if not isinstance(variance1, jax.Array):
                                combined_cov_to_add = np.asarray(combined_cov)
                            else:
                                combined_cov_to_add = combined_cov
                            for g in range(variance1.shape[2]):
                                gene_specific_covariance[:, :, g] = variance1[:, :, g] + combined_cov_to_add
                            logger.debug(f"Using gene-specific covariance matrices from variance1 with shape {gene_specific_covariance.shape}")
                        else:
                            combined_cov += variance1
                            logger.debug("Added variance1 covariance matrix to function predictor covariance")
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
                        # Need to add combined_cov to each gene's covariance slice
                        gene_specific_covariance = variance2
                        # Check if variance2 is a JAX array, if not, ensure combined_cov is numpy array
                        if not isinstance(variance2, jax.Array):
                            combined_cov_to_add = np.asarray(combined_cov)
                        else:
                            combined_cov_to_add = combined_cov
                        for g in range(variance2.shape[2]):
                            gene_specific_covariance[:, :, g] = variance2[:, :, g] + combined_cov_to_add
                        logger.debug(f"Using gene-specific covariance matrices from variance2 with shape {gene_specific_covariance.shape}")
                    else:
                        # Add variance2 to the combined covariance
                        combined_cov += variance2
                        logger.debug("Added variance2 covariance matrix to function predictor covariance")
                except Exception as e:
                    error_msg = f"Error computing sample variance from variance_predictor2: {e}."
                    logger.error(error_msg)
                    raise RuntimeError(error_msg) from e
        
        # Transpose fold_change to get shape (n_genes, n_points) for easier gene-wise processing
        fold_change_transposed = fold_change_subset.T
        
        # Choose the approach based on whether we have gene-specific covariance matrices
        try:
            if gene_specific_covariance is not None:
                # Use gene-specific covariance matrices (3D tensor)
                logger.debug(f"Computing Mahalanobis distances for {fold_change_transposed.shape[0]:,} genes with gene-specific covariance matrices...")
                
                # Compute all distances using the unified utility function with gene-specific covariance
                logger.info(f"Using batch_size={self.batch_size} for Mahalanobis distance computation")
                mahalanobis_distances = compute_mahalanobis_distances(
                    diff_values=fold_change_transposed,
                    covariance=gene_specific_covariance,
                    batch_size=self.batch_size,  
                    jit_compile=self.jit_compile,
                    eps=self.eps,
                    progress=progress
                )
                
                logger.info(f"Successfully computed Mahalanobis distances for {len(mahalanobis_distances):,} genes using gene-specific covariance")
            else:
                logger.debug(f"Computing Mahalanobis distances for {fold_change_transposed.shape[0]:,} genes with shared covariance...")
                
                # Compute all distances using the unified utility function with the combined covariance matrix
                logger.info(f"Using batch_size={self.batch_size} for Mahalanobis distance computation")
                mahalanobis_distances = compute_mahalanobis_distances(
                    diff_values=fold_change_transposed,
                    covariance=combined_cov,
                    batch_size=self.batch_size,
                    jit_compile=self.jit_compile,
                    eps=self.eps,
                    progress=progress
                )
                
                logger.debug(f"Successfully computed Mahalanobis distances for {len(mahalanobis_distances):,} genes")
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
            - 'lfc_stds': Z-scores for the fold changes
            - 'bidirectionality': Bidirectionality scores
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
            # Always add mean_log_fold_change to the result
            result['mean_log_fold_change'] = mean_log_fold_change
            
            # Use sample variance from mahalanobis calculation for z-scores
            # This will get the full covariance matrix with diag=False
            if self.use_sample_variance:
                # Even with sample variance, we need to provide a default z-score
                # Compute basic z-scores using the function predictor uncertainties
                stds = np.sqrt(variance + self.eps)
                fold_change_zscores = fold_change / stds
                result['fold_change_zscores'] = fold_change_zscores
            else:
                # If we're not using sample variance, just use the function predictor uncertainties
                stds = np.sqrt(variance + self.eps)
                fold_change_zscores = fold_change / stds
                result['fold_change_zscores'] = fold_change_zscores
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