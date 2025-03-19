"""Sample variance estimation for differential analysis."""

import numpy as np
import jax
import jax.numpy as jnp
import logging
from typing import Dict, Optional
from mellon import FunctionEstimator, DensityEstimator

from ..memory_utils import (
    DiskStorage,
    analyze_covariance_memory_requirements,
    DASK_AVAILABLE
)

# Try to import dask if available
if DASK_AVAILABLE:
    try:
        import dask.array as da
        import dask
    except ImportError:
        pass

logger = logging.getLogger("kompot")


class SampleVarianceEstimator:
    """
    Compute local sample variances of gene expressions or density.

    This class manages the computation of empirical variance by fitting function estimators
    or density estimators for each group in the data and computing the variance between their
    predictions. Bessel's correction is applied to the variance calculation to ensure
    unbiased estimation, especially important when the number of samples is small.

    Attributes
    ----------
    group_predictors : Dict
        Dictionary of prediction functions for each group.
    estimator_type : str
        Type of estimator used ('function' for gene expression, 'density' for cell density).
    disk_storage : DiskStorage, optional
        Storage manager for offloading large arrays to disk, if enabled.
    """
    
    def __init__(
        self,
        eps: float = 1e-12,
        jit_compile: bool = True,
        estimator_type: str = 'function',
        store_arrays_on_disk: bool = False,
        disk_storage_dir: Optional[str] = None
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
        store_arrays_on_disk : bool, optional
            Whether to store large arrays on disk instead of in memory, by default False.
            Useful for very large datasets where covariance matrices would exceed available memory.
        disk_storage_dir : str, optional
            Directory to store arrays on disk. If None and store_arrays_on_disk is True,
            a temporary directory will be created and cleaned up when this object is deleted.
        """
        self.eps = eps
        self.jit_compile = jit_compile
        self.estimator_type = estimator_type
        self.store_arrays_on_disk = store_arrays_on_disk
        
        if estimator_type not in ['function', 'density']:
            raise ValueError("estimator_type must be either 'function' or 'density'")
        
        # Will be populated during fit
        self.group_predictors = {}
        self.group_centroids = {}
        self._predict_variance_jit = None
        
        # Define covariance computation function that will be JIT-compiled if needed
        def compute_cov_slice(gene_centered, n_groups):
            # Apply Bessel's correction (divide by n-1 instead of n)
            # Only apply correction if we have more than 1 group
            divisor = jnp.maximum(1, n_groups - 1)
            # Calculate covariance as dot product divided by divisor for Bessel's correction
            return (gene_centered @ gene_centered.T) / divisor
            
        # Store the function as instance attribute
        self._compute_cov_slice = compute_cov_slice
        
        # JIT-compile if requested
        if jit_compile:
            self._compute_cov_slice_jit = jax.jit(compute_cov_slice)
        else:
            self._compute_cov_slice_jit = None
        
        # Initialize disk storage if requested
        self._disk_storage = None
        if store_arrays_on_disk:
            self._disk_storage = DiskStorage(storage_dir=disk_storage_dir)
            logger.info(f"Disk storage for large arrays enabled in SampleVarianceEstimator. "
                       f"Arrays will be stored in {self._disk_storage.storage_dir}")
            
    def __del__(self):
        """Clean up disk storage when the object is deleted."""
        if hasattr(self, '_disk_storage') and self._disk_storage is not None:
            self._disk_storage.cleanup()
    
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
                logger.info(f"Training estimator for group {group_id} with {len(indices):,} cells")
                X_subset = X[indices]
                
                if self.estimator_type == 'function':
                    if Y is None:
                        raise ValueError("Y must be provided for function estimator type")
                    Y_subset = Y[indices]
                    
                    # Create and train function estimator
                    estimator = FunctionEstimator(**estimator_kwargs)
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
                    estimator = DensityEstimator(**density_defaults)
                    estimator.fit(X_subset)
                
                # Store only the predictor function, not the full estimator
                self.group_predictors[group_id] = estimator.predict
                
                # Immediately delete the estimator to free memory
                del estimator
            else:
                logger.warning(f"Skipping group {group_id} (only {len(indices):,} cells < min_cells={min_cells:,})")
        
        return self
    
    def predict(self, X_new: np.ndarray, diag: bool = False, progress: bool = True) -> np.ndarray:
        """
        Predict empirical variance for new points using JAX.
        
        This method computes the variance with Bessel's correction (using n-1 instead of n
        in the denominator) to provide an unbiased estimate of the population variance.
        This correction is particularly important when the number of samples (groups) is small.
        
        Parameters
        ----------
        X_new : np.ndarray
            New cell states to predict. Shape (n_cells, n_features).
        diag : bool, optional
            If True (default is False), compute the variance for each cell state.
            If False, compute the full covariance matrix between all pairs of cells.
        progress : bool, optional
            Whether to show progress bars for gene-wise operations, by default True.
            
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
            
            # Check memory requirements when processing many genes
            if n_genes > 10 and len(self.group_predictors) > 1 and hasattr(self, '_called_from_differential') and self._called_from_differential:
                # Use the specialized covariance memory analysis function with debug log level
                # when disk storage is already enabled
                log_level = "debug" if self.store_arrays_on_disk else "info"
                
                analysis = analyze_covariance_memory_requirements(
                    n_points=n_cells,
                    n_genes=n_genes,
                    max_memory_ratio=0.8,  # Use standard 80% threshold
                    analysis_name="Sample Variance Estimation",
                    store_arrays_on_disk=self.store_arrays_on_disk,
                    log_level=log_level
                )
                
                # Issue appropriate warning based on memory usage
                if analysis['status'] == 'critical':
                    if self.store_arrays_on_disk:
                        logger.debug(
                            f"High memory usage detected ({analysis['memory_ratio']:.2f}x of available memory). "
                            f"Will use disk storage for large covariance arrays."
                        )
                    # The warning is already logged by analyze_covariance_memory_requirements
                elif analysis['status'] == 'warning':
                    # The warning is already logged by analyze_covariance_memory_requirements
                    pass
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
                    # Get number of groups
                    n_groups = stacked.shape[0]
                    # Apply Bessel's correction for unbiased variance estimate
                    # Use ddof=1 for Bessel's correction (divide by n-1 instead of n)
                    # Only apply correction if we have more than 1 group
                    return jnp.var(stacked, axis=0, ddof=1) if n_groups > 1 else jnp.var(stacked, axis=0)
                
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
                # Apply Bessel's correction for unbiased variance estimate
                n_groups = stacked_predictions.shape[0]
                # Only apply correction if we have more than 1 group
                if n_groups > 1:
                    # Use ddof=1 for Bessel's correction (divide by n-1 instead of n)
                    batch_variance = jnp.var(stacked_predictions, axis=0, ddof=1)
                else:
                    batch_variance = jnp.var(stacked_predictions, axis=0)
                # Convert back to numpy for compatibility
                return np.array(batch_variance)
        
        else:
            # Full covariance matrix computation (between all pairs of cells)
            # Check if we should use disk storage for this operation using the specialized function
            analysis = analyze_covariance_memory_requirements(
                n_points=n_cells,
                n_genes=n_genes,
                max_memory_ratio=0.8,  # Standard 80% threshold
                analysis_name="Full Covariance Matrix"
            )
            
            # Determine if we should use disk storage based on the analysis
            use_disk_storage = self.store_arrays_on_disk and analysis['should_use_disk']
            if use_disk_storage and self._disk_storage is None:
                self._disk_storage = DiskStorage()
                logger.info(f"Automatically enabling disk storage at {self._disk_storage.storage_dir}")
                
            # Define the covariance shape for disk-backed matrix
            covariance_shape = (n_cells, n_cells, n_genes)
            
            if use_disk_storage:
                logger.info(f"Using gene-by-gene disk storage for covariance matrix (shape={covariance_shape})")
            
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
            
            if use_disk_storage:
                # Check if dask is available for better performance
                if DASK_AVAILABLE:
                    logger.debug(f"Using dask for disk-backed covariance matrix (shape={covariance_shape})")
                    
                    # Create dask delayed functions to compute each gene slice
                    gene_arrays = []
                    
                    # Don't use progress bar for covariance computation as it's usually very fast
                    # and creates unnecessary visual noise
                    range_func = range(n_genes)
                    
                    # Process each gene slice
                    for g in range_func:
                        # Extract the data for this gene
                        gene_centered = centered_reshaped[:, :, g]  # (n_cells, n_groups)
                        
                        # Create a delayed computation for this gene slice
                        @dask.delayed
                        def compute_gene_slice(gene_centered, g, n_groups):
                            # Choose the right computation function
                            if self._compute_cov_slice_jit is not None:
                                gene_cov = self._compute_cov_slice_jit(gene_centered, n_groups)
                            else:
                                gene_cov = self._compute_cov_slice(gene_centered, n_groups)
                            
                            # Convert to numpy and store to disk for caching 
                            gene_cov_np = np.array(gene_cov)
                            gene_key = f"gene_cov_{g}"
                            self._disk_storage.store_array(gene_cov_np, gene_key)
                            
                            return gene_cov_np
                        
                        # Create a delayed version of the computation
                        delayed_result = compute_gene_slice(gene_centered, g, n_groups)
                        
                        # Convert the delayed computation to a dask array
                        gene_array = da.from_delayed(
                            delayed_result,
                            shape=(n_cells, n_cells),
                            dtype=np.float64
                        )
                        
                        # Add to our list of arrays
                        gene_arrays.append(gene_array)
                    
                    # Stack the gene arrays along the last axis
                    dask_covariance = da.stack(gene_arrays, axis=2)
                    
                    logger.debug(f"Created dask array for covariance with shape {dask_covariance.shape}")
                    return dask_covariance
                
                else:
                    # Without dask, warn the user and use numpy memory mapping
                    logger.warning(
                        "Dask is not available for disk-backed arrays. "
                        "Install dask for better performance: pip install dask"
                    )
                    
                    # Use numpy memory mapping as a fallback
                    import tempfile
                    import os
                    
                    # Create a memory-mapped array
                    filename = os.path.join(self._disk_storage.storage_dir, 'covariance_matrix.npy')
                    mmap_array = np.lib.format.open_memmap(
                        filename, 
                        mode='w+',
                        dtype=np.float64,
                        shape=covariance_shape
                    )
                    
                    # Don't use progress bar for covariance computation as it's usually very fast
                    # and creates unnecessary visual noise
                    range_func = range(n_genes)
                        
                    # Process gene-by-gene
                    for g in range_func:
                        gene_centered = centered_reshaped[:, :, g]  # (n_cells, n_groups)
                        
                        # Compute covariance slice
                        if self._compute_cov_slice_jit is not None:
                            gene_cov = self._compute_cov_slice_jit(gene_centered, n_groups)
                        else:
                            gene_cov = self._compute_cov_slice(gene_centered, n_groups)
                        
                        # Store in the memory-mapped array
                        mmap_array[:, :, g] = np.array(gene_cov)
                    
                    # Return the memory-mapped array
                    return mmap_array
            else:
                # In-memory version
                cov_matrix = np.zeros(covariance_shape)
                
                # Don't use progress bar for covariance computation as it's usually very fast
                # and creates unnecessary visual noise
                range_func = range(n_genes)
                
                for g in range_func:
                    gene_centered = centered_reshaped[:, :, g]  # (n_cells, n_groups)
                    
                    # Compute covariance slice using JIT if available
                    if self._compute_cov_slice_jit is not None:
                        gene_cov = self._compute_cov_slice_jit(gene_centered, n_groups)
                    else:
                        gene_cov = self._compute_cov_slice(gene_centered, n_groups)
                    
                    cov_matrix[:, :, g] = np.array(gene_cov)
                
                return cov_matrix