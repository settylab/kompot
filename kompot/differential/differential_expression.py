"""Differential expression analysis."""

import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.stats as jax_stats
from functools import partial
from typing import Tuple, List, Optional, Union, Dict, Any, Callable
import logging
from scipy.stats import norm as normal
import mellon
from mellon.parameters import compute_landmarks
from tqdm.auto import tqdm

from ..utils import (
    compute_mahalanobis_distance,
    compute_mahalanobis_distances,
    find_landmarks
)
from ..batch_utils import apply_batched, is_jax_memory_error
from .expression_model import ExpressionModel
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
        use_empirical_variance: bool = True,
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
        model1: Optional["ExpressionModel"] = None,
        model2: Optional["ExpressionModel"] = None,
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
        use_empirical_variance : bool, optional
            Whether to estimate per-gene empirical variance from GP residuals.
            When True, the expression GP is fitted with ``obs_variance=True``,
            which computes leverage-corrected squared residuals and smooths them
            with a second GP to produce an input-dependent noise surface.
            This captures gene-specific heteroscedastic noise without requiring
            biological replicates. By default False.
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
        model1 : ExpressionModel, optional
            Pre-fitted ExpressionModel for condition 1. If provided, function_predictor1
            and variance_predictor1 are ignored.
        model2 : ExpressionModel, optional
            Pre-fitted ExpressionModel for condition 2. If provided, function_predictor2
            and variance_predictor2 are ignored.
        """
        self.n_landmarks = n_landmarks
        self.use_empirical_variance = use_empirical_variance
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

        # ExpressionModel instances for each condition
        if model1 is not None:
            self.model1 = model1
        elif function_predictor1 is not None:
            self.model1 = ExpressionModel(
                use_empirical_variance=use_empirical_variance,
                eps=eps,
                batch_size=batch_size,
                function_predictor=function_predictor1,
                variance_predictor=variance_predictor1,
            )
        else:
            self.model1 = None

        if model2 is not None:
            self.model2 = model2
        elif function_predictor2 is not None:
            self.model2 = ExpressionModel(
                use_empirical_variance=use_empirical_variance,
                eps=eps,
                batch_size=batch_size,
                function_predictor=function_predictor2,
                variance_predictor=variance_predictor2,
            )
        else:
            self.model2 = None

        # Mahalanobis distances
        self.mahalanobis_distances = None
        
    # ------------------------------------------------------------------
    # Backward-compatible properties delegating to model1/model2
    # ------------------------------------------------------------------

    @property
    def function_predictor1(self):
        """Function predictor for condition 1."""
        return self.model1.predictor if self.model1 else None

    @function_predictor1.setter
    def function_predictor1(self, value):
        if value is not None and self.model1 is None:
            self.model1 = ExpressionModel(
                use_empirical_variance=self.use_empirical_variance,
                eps=self.eps,
                batch_size=self.batch_size,
            )
        if self.model1 is not None:
            self.model1._predictor = value

    @property
    def function_predictor2(self):
        """Function predictor for condition 2."""
        return self.model2.predictor if self.model2 else None

    @function_predictor2.setter
    def function_predictor2(self, value):
        if value is not None and self.model2 is None:
            self.model2 = ExpressionModel(
                use_empirical_variance=self.use_empirical_variance,
                eps=self.eps,
                batch_size=self.batch_size,
            )
        if self.model2 is not None:
            self.model2._predictor = value

    @property
    def variance_predictor1(self):
        """Sample variance predictor for condition 1."""
        return self.model1._sample_variance_predictor if self.model1 else None

    @variance_predictor1.setter
    def variance_predictor1(self, value):
        if value is not None and self.model1 is None:
            self.model1 = ExpressionModel(eps=self.eps, batch_size=self.batch_size)
        if self.model1 is not None:
            self.model1._sample_variance_predictor = value

    @property
    def variance_predictor2(self):
        """Sample variance predictor for condition 2."""
        return self.model2._sample_variance_predictor if self.model2 else None

    @variance_predictor2.setter
    def variance_predictor2(self, value):
        if value is not None and self.model2 is None:
            self.model2 = ExpressionModel(eps=self.eps, batch_size=self.batch_size)
        if self.model2 is not None:
            self.model2._sample_variance_predictor = value

    @property
    def empirical_variance_predictor1(self):
        """Empirical (obs) variance predictor for condition 1."""
        if self.model1 and self.model1.has_empirical_variance:
            return self.model1._obs_variance_func
        return None

    @empirical_variance_predictor1.setter
    def empirical_variance_predictor1(self, value):
        pass  # Managed by ExpressionModel; kept for backward compat

    @property
    def empirical_variance_predictor2(self):
        """Empirical (obs) variance predictor for condition 2."""
        if self.model2 and self.model2.has_empirical_variance:
            return self.model2._obs_variance_func
        return None

    @empirical_variance_predictor2.setter
    def empirical_variance_predictor2(self, value):
        pass  # Managed by ExpressionModel; kept for backward compat

    @property
    def expression_estimator_condition1(self):
        """Mellon FunctionEstimator for condition 1."""
        return self.model1._estimator if self.model1 else None

    @expression_estimator_condition1.setter
    def expression_estimator_condition1(self, value):
        if self.model1 is not None:
            self.model1._estimator = value

    @property
    def expression_estimator_condition2(self):
        """Mellon FunctionEstimator for condition 2."""
        return self.model2._estimator if self.model2 else None

    @expression_estimator_condition2.setter
    def expression_estimator_condition2(self, value):
        if self.model2 is not None:
            self.model2._estimator = value

    def __del__(self):
        """Cleanup method for object deletion."""
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
        allow_single_condition_variance: bool = False,
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

        # Compute shared landmarks (needs both conditions)
        if self.function_predictor1 is None or self.function_predictor2 is None:
            if landmarks is not None:
                logger.info(f"Using provided landmarks with shape {landmarks.shape}")
                self.computed_landmarks = landmarks
            elif self.n_landmarks is not None and self.n_landmarks > 0:
                X_combined = np.vstack([X_condition1, X_condition2])
                landmarks = compute_landmarks(
                    X_combined,
                    gp_type='fixed',
                    n_landmarks=self.n_landmarks,
                    random_state=self.random_state
                )
                self.computed_landmarks = landmarks

        # -- Fit model1 --
        if self.model1 is None:
            self.model1 = ExpressionModel(
                n_landmarks=self.n_landmarks,
                use_empirical_variance=self.use_empirical_variance,
                eps=self.eps,
                random_state=self.random_state,
                batch_size=self.batch_size,
                store_arrays_on_disk=self.store_arrays_on_disk,
                disk_storage_dir=self.disk_storage_dir,
            )

        if self.model1.predictor is None:
            logger.info("Fitting expression estimator for condition 1...")
            self.model1.fit(
                X_condition1, y_condition1,
                sigma=sigma, ls=ls, ls_factor=ls_factor,
                landmarks=landmarks,
                sample_indices=condition1_sample_indices if self.use_sample_variance else None,
                sample_estimator_ls=sample_estimator_ls,
                allow_single_condition_variance=allow_single_condition_variance,
                **function_kwargs,
            )

        # Extract ls from model1 for model2 consistency
        ls_for_model2 = ls
        if ls is None and 'ls' not in function_kwargs and self.model1.ls is not None:
            ls_for_model2 = self.model1.ls

        # -- Fit model2 --
        if self.model2 is None:
            self.model2 = ExpressionModel(
                n_landmarks=self.n_landmarks,
                use_empirical_variance=self.use_empirical_variance,
                eps=self.eps,
                random_state=self.random_state,
                batch_size=self.batch_size,
                store_arrays_on_disk=self.store_arrays_on_disk,
                disk_storage_dir=self.disk_storage_dir,
            )

        if self.model2.predictor is None:
            logger.info("Fitting expression estimator for condition 2...")
            self.model2.fit(
                X_condition2, y_condition2,
                sigma=sigma, ls=ls_for_model2, ls_factor=ls_factor,
                landmarks=landmarks,
                sample_indices=condition2_sample_indices if self.use_sample_variance else None,
                sample_estimator_ls=sample_estimator_ls,
                allow_single_condition_variance=allow_single_condition_variance,
                **function_kwargs,
            )

        # Validate empirical variance for pre-fitted predictors
        if self.use_empirical_variance:
            if not hasattr(self.function_predictor1, 'obs_variance'):
                raise ValueError(
                    "use_empirical_variance=True requires predictors fitted with "
                    "obs_variance=True. Pre-computed predictors must support obs_variance()."
                )

        # Handle single-condition variance fallback
        if self.use_sample_variance and have_sample_indices and allow_single_condition_variance:
            has_sv1 = self.model1.has_sample_variance
            has_sv2 = self.model2.has_sample_variance
            if has_sv1 and not has_sv2:
                logger.info("Using condition 1 variance estimator for both conditions")
                self.model2._sample_variance_predictor = self.model1._sample_variance_predictor
            elif has_sv2 and not has_sv1:
                logger.info("Using condition 2 variance estimator for both conditions")
                self.model1._sample_variance_predictor = self.model2._sample_variance_predictor
            elif not has_sv1 and not has_sv2:
                if condition1_sample_indices is not None or condition2_sample_indices is not None:
                    raise ValueError("Both variance estimators failed to fit. Cannot proceed with sample variance estimation.")

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
            Whether to show tqdm.auto progress bars during Mahalanobis distance computation. 
            When True, displays progress bars for gene-wise operations. When False, progress 
            bars are disabled. Default is True.
            
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
            logger.debug(f"Using {len(landmarks):,} landmarks for Mahalanobis computation")
            
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
            logger.debug(f"No landmarks used, computing covariance between all {len(X):,} points.")
            
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
        del cov1, cov2
        
        # For sample variance, use diag=False to get full covariance matrices
        # Initialize variable to store gene-specific covariance matrices if needed
        gene_specific_covariance = None
        
        if self.use_sample_variance:
            # Add empirical adjustments from sample variance
            
            # Create functions for computing sample variance
            if self.variance_predictor1 is not None:
                try:
                    # Important: use diag=False to get full covariance matrix
                    variance1 = self.variance_predictor1(variance_points, diag=False, progress=progress)
                    if self.variance_predictor2 is not None:
                        variance2 = self.variance_predictor2(variance_points, diag=False, progress=progress)
                        # Add the covariance matrices for complete variance representation
                        combined_variance = variance1 + variance2
                        del variance1, variance2
                        
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
                            for g in tqdm(range(combined_variance.shape[2]), 
                                         desc="Processing gene-specific covariance matrices", 
                                         disable=not progress):
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
                            for g in tqdm(range(variance1.shape[2]), 
                                         desc="Processing gene-specific covariance matrices (variance1)", 
                                         disable=not progress):
                                gene_specific_covariance[:, :, g] = variance1[:, :, g] + combined_cov_to_add
                            logger.debug(f"Using gene-specific covariance matrices from variance1 with shape {gene_specific_covariance.shape}")
                        else:
                            combined_cov += variance1
                            logger.debug("Added variance1 covariance matrix to function predictor covariance")
                        del variance1
                except Exception as e:
                    error_msg = f"Error computing sample variance from variance_predictor1: {e}."
                    logger.error(error_msg)
                    raise RuntimeError(error_msg) from e
            elif self.variance_predictor2 is not None:
                try:
                    # Important: use diag=False to get full covariance matrix
                    variance2 = self.variance_predictor2(variance_points, diag=False, progress=progress)
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
                        for g in tqdm(range(variance2.shape[2]), 
                                     desc="Processing gene-specific covariance matrices (variance2)", 
                                     disable=not progress):
                            gene_specific_covariance[:, :, g] = variance2[:, :, g] + combined_cov_to_add
                        logger.debug(f"Using gene-specific covariance matrices from variance2 with shape {gene_specific_covariance.shape}")
                    else:
                        # Add variance2 to the combined covariance
                        combined_cov += variance2
                        logger.debug("Added variance2 covariance matrix to function predictor covariance")
                    del variance2
                except Exception as e:
                    error_msg = f"Error computing sample variance from variance_predictor2: {e}."
                    logger.error(error_msg)
                    raise RuntimeError(error_msg) from e
        
        # Compute empirical variance at variance_points if enabled
        empirical_diag_var = None
        if self.use_empirical_variance and self.empirical_variance_predictor1 is not None:
            logger.debug("Computing empirical variance at evaluation points...")
            emp_var1 = np.maximum(apply_batched(
                lambda X: self.empirical_variance_predictor1(X),
                variance_points, batch_size=self.batch_size,
                desc="Empirical variance (condition 1)" if progress else None,
            ), self.eps)
            emp_var2 = np.maximum(apply_batched(
                lambda X: self.empirical_variance_predictor2(X),
                variance_points, batch_size=self.batch_size,
                desc="Empirical variance (condition 2)" if progress else None,
            ), self.eps)
            # combined_emp_var shape: (n_points, n_genes)
            combined_emp_var = emp_var1 + emp_var2
            del emp_var1, emp_var2
            # diagonal_variance needs shape (n_genes, n_points)
            empirical_diag_var = combined_emp_var.T
            logger.debug(f"Empirical diagonal variance shape: {empirical_diag_var.shape}")

        # Transpose fold_change to get shape (n_genes, n_points) for easier gene-wise processing
        fold_change_transposed = fold_change_subset.T

        # Choose the approach based on whether we have gene-specific covariance matrices
        try:
            if gene_specific_covariance is not None:
                # Use gene-specific covariance matrices (3D tensor)
                logger.debug(f"Computing Mahalanobis distances for {fold_change_transposed.shape[0]:,} genes with gene-specific covariance matrices...")

                # Note: batch_size is not used for gene-specific covariance (processes one gene at a time)
                # Memory is dominated by the covariance tensor: (n_points, n_points, n_genes)
                logger.debug(f"Gene-specific covariance: batch_size is not used (processes genes sequentially)")
                mahalanobis_distances = compute_mahalanobis_distances(
                    diff_values=fold_change_transposed,
                    covariance=gene_specific_covariance,
                    batch_size=None,  # Ignored for gene-specific covariance
                    jit_compile=self.jit_compile,
                    eps=self.eps,
                    progress=progress,
                    diagonal_variance=empirical_diag_var,
                )

                logger.debug(f"Successfully computed Mahalanobis distances for {len(mahalanobis_distances):,} genes using gene-specific covariance")
            else:
                logger.debug(f"Computing Mahalanobis distances for {fold_change_transposed.shape[0]:,} genes with shared covariance...")

                # Compute all distances using the unified utility function with the combined covariance matrix
                logger.debug(f"Using batch_size={self.batch_size} for Mahalanobis distance computation")
                mahalanobis_distances = compute_mahalanobis_distances(
                    diff_values=fold_change_transposed,
                    covariance=combined_cov,
                    batch_size=self.batch_size,
                    jit_compile=self.jit_compile,
                    eps=self.eps,
                    progress=progress,
                    diagonal_variance=empirical_diag_var,
                )

                logger.debug(f"Successfully computed Mahalanobis distances for {len(mahalanobis_distances):,} genes")
        except Exception as e:
            # Provide context-appropriate error message
            if gene_specific_covariance is not None:
                error_msg = (f"Failed to compute Mahalanobis distances with gene-specific covariance: {str(e)}. "
                           f"Try using store_arrays_on_disk=True or reduce n_landmarks to control memory usage.")
            else:
                error_msg = (f"Failed to compute Mahalanobis distances: {str(e)}. "
                           f"Try reducing batch_size, using store_arrays_on_disk=True, or disable Mahalanobis "
                           f"distance calculation with compute_mahalanobis=False")
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

        # Determine degrees of freedom for chi2 distribution
        # This is the dimension of the vector used to compute Mahalanobis distance
        if has_landmarks and landmarks is not None:
            degrees_of_freedom = landmarks.shape[1]  # Number of features/landmarks
        else:
            degrees_of_freedom = X.shape[1]  # Number of features
        
        # Store degrees of freedom as instance variable for use in predict method
        self._last_mahalanobis_dof = degrees_of_freedom
        
        return mahalanobis_distances
    
    def predict(
        self, 
        X_new: np.ndarray, 
        compute_mahalanobis: bool = False,
        progress: bool = True,
        use_landmarks: bool = True,
        landmarks_override: Optional[np.ndarray] = None
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
            Whether to show tqdm.auto progress bars during computation. When True, displays 
            progress bars for all batch processing operations including prediction, uncertainty 
            computation, and Mahalanobis distance calculations. When False, all progress bars 
            are disabled. Default is True.
        use_landmarks : bool, optional
            Whether to use landmarks for Mahalanobis distance calculation if available, by default True.
            Setting to False will force computation using all provided points, which can be more accurate
            for small datasets or subsets.
        landmarks_override : np.ndarray, optional
            Explicitly provided landmarks to use instead of the ones from the fitted model.
            Shape (n_landmarks, n_features). Used when custom landmarks are needed for a specific
            prediction, such as when analyzing a subset of data.
            
        Returns
        -------
        dict
            Dictionary containing the predictions:
            - 'condition1_imputed': Imputed expression for condition 1
            - 'condition2_imputed': Imputed expression for condition 2
            - 'condition1_std': Posterior standard deviation for condition 1
            - 'condition2_std': Posterior standard deviation for condition 2
            - 'fold_change': Fold change between conditions
            - 'mean_log_fold_change': Mean log fold change across all cells
            - 'mahalanobis_distances': Only if compute_mahalanobis is True
        """
        if self.model1 is None or self.model1.predictor is None:
            raise ValueError("Model not fitted. Call fit() first.")
        if self.model2 is None or self.model2.predictor is None:
            raise ValueError("Model not fitted. Call fit() first.")

        batch_size = getattr(self, 'batch_size', None)

        # Imputed expression via ExpressionModel
        condition1_imputed = self.model1.predict(X_new, batch_size=batch_size, progress=progress)
        condition2_imputed = self.model2.predict(X_new, batch_size=batch_size, progress=progress)

        # Compute total variance once per condition, derive std from it
        total_var1 = self.model1.total_variance(X_new, diag=True, batch_size=batch_size, progress=progress)
        total_var2 = self.model2.total_variance(X_new, diag=True, batch_size=batch_size, progress=progress)
        condition1_std = np.sqrt(total_var1 + self.eps)
        condition2_std = np.sqrt(total_var2 + self.eps)

        # Fold change
        fold_change = condition2_imputed - condition1_imputed

        # Combined variance for z-scores
        total_variance = total_var1 + total_var2
        del total_var1, total_var2

        # Compute mean log fold change
        mean_log_fold_change = np.mean(fold_change, axis=0)

        # Compute z-scores
        fold_change_zscores = fold_change / np.sqrt(total_variance + self.eps)
        del total_variance

        result = {
            'condition1_imputed': condition1_imputed,
            'condition2_imputed': condition2_imputed,
            'condition1_std': condition1_std,
            'condition2_std': condition2_std,
            'fold_change': fold_change,
            'fold_change_zscores': fold_change_zscores,
            'mean_log_fold_change': mean_log_fold_change,
        }

        # Compute Mahalanobis distances if requested
        if compute_mahalanobis:
            logger.debug("Computing Mahalanobis distances...")

            mahalanobis_distances = self.compute_mahalanobis_distances(
                X=X_new,
                fold_change=fold_change,
                use_landmarks=use_landmarks,
                landmarks_override=landmarks_override,
                progress=progress,
            )
            result['mahalanobis_distances'] = mahalanobis_distances

            if hasattr(self, '_last_mahalanobis_dof'):
                logger.debug(f"Computing ptp with {self._last_mahalanobis_dof} degrees of freedom...")
                mahalanobis_squared = jnp.array(mahalanobis_distances) ** 2
                ptp = jax_stats.chi2.sf(mahalanobis_squared, df=self._last_mahalanobis_dof)
                result['ptp'] = np.array(ptp)

        return result