"""Sample variance estimation for differential analysis."""

import numpy as np
import jax
import jax.numpy as jnp
import logging
from typing import Dict, Optional
from mellon import FunctionEstimator, DensityEstimator
from tqdm.auto import tqdm

from ..memory_utils import DiskStorage
from ..resource_estimation import analyze_covariance_memory_requirements

# Try to import dask for parallel computation
try:
    import dask
    import dask.array as da
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

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
    n_groups : int
        Number of unique groups found during fit. Must be at least 2 for variance calculation.
    """
    
    def __init__(
        self,
        eps: float = 1e-8,  # Increased default epsilon for better numerical stability
        jit_compile: bool = True,
        estimator_type: str = 'function',
        store_arrays_on_disk: Optional[bool] = None,
        disk_storage_dir: Optional[str] = None,
        dask_num_workers: Optional[int] = None
    ):
        """
        Initialize the SampleVarianceEstimator.
        
        Parameters
        ----------
        eps : float, optional
            Small constant for numerical stability, by default 1e-8.
        jit_compile : bool, optional
            Whether to use JAX just-in-time compilation, by default True.
        estimator_type : str, optional
            Type of estimator to use ('function' for gene expression, 'density' for cell density),
            by default 'function'.
        store_arrays_on_disk : bool, optional
            Whether to store large arrays on disk instead of in memory, by default None.
            If None, it will be determined based on disk_storage_dir (True if provided, False otherwise).
            Useful for very large datasets where covariance matrices would exceed available memory.
        disk_storage_dir : str, optional
            Directory to store arrays on disk. If provided and store_arrays_on_disk is None,
            store_arrays_on_disk will be set to True. If store_arrays_on_disk is False and
            this is provided, a warning will be logged and disk storage will not be used.
        dask_num_workers : int, optional
            Number of parallel Dask workers to use for covariance computation. If None (default),
            Dask uses all available CPU cores for maximum speed. Set to a smaller number (e.g., 2-4)
            to limit CPU utilization at the cost of slower computation. Only applies when Dask is
            available and disk storage is enabled. Note: This primarily controls CPU usage, not memory,
            since the main memory footprint comes from the shared centered data array rather than
            per-worker allocations.
        """
        self.eps = eps
        self.jit_compile = jit_compile
        self.estimator_type = estimator_type
        self.dask_num_workers = dask_num_workers
        
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
        
        if estimator_type not in ['function', 'density']:
            raise ValueError("estimator_type must be either 'function' or 'density'")
        
        # Will be populated during fit
        self.group_predictors = {}
        self.group_centroids = {}
        self._predict_variance_jit = None
        self.n_groups = 0  # Will be set during fit
        
        # Define covariance computation function that will be JIT-compiled if needed
        def compute_cov_slice(gene_centered, n_groups):
            # Apply Bessel's correction (divide by n-1 instead of n)
            # Only apply correction if we have more than 1 group
            divisor = n_groups - 1
            # Calculate covariance as dot product divided by divisor for Bessel's correction
            return (gene_centered @ gene_centered.T) / divisor

            
        # Store the function as instance attribute
        self._compute_cov_slice = compute_cov_slice
        
        # JIT-compile if requested
        if jit_compile:
            self._compute_cov_slice_jit = jax.jit(compute_cov_slice)
        else:
            self._compute_cov_slice_jit = None
        
        # Don't initialize disk storage until needed (lazy initialization in predict method)
        self._disk_storage = None
        self._memory_analyzed = False  # Track if we've done memory analysis already
    def __del__(self):
        """Clean up disk storage when the object is deleted."""
        if hasattr(self, '_disk_storage') and self._disk_storage is not None:
            self._disk_storage.cleanup()
            self._disk_storage = None
    
    def fit(
        self, 
        X: np.ndarray,
        Y: np.ndarray = None, 
        grouping_vector: np.ndarray = None,
        min_cells: int = 2,
        ls_factor: float = 10.0,
        estimator_kwargs: Dict = None
    ):
        """
        Fit estimators for each group in the data and store only their predictors.
        
        At least 2 groups with sufficient cells (>= min_cells) are required for
        variance calculation. If fewer than 2 valid groups are found, a ValueError
        will be raised.
        
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
            Minimum number of cells for group to train an estimator. Default is 2.
            Groups with fewer cells will be skipped.
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
            
        Raises
        ------
        ValueError
            If fewer than 2 groups have sufficient cells to compute variance.
        """
        # Check if Y is provided for function estimator
        if self.estimator_type == 'function' and Y is None:
            raise ValueError("Y must be provided for function estimator type")
            
        if estimator_kwargs is None:
            estimator_kwargs = {}
        
        # Add ls_factor to estimator_kwargs if ls is not already specified
        if 'ls' not in estimator_kwargs:
            estimator_kwargs['ls_factor'] = ls_factor
        
        # Get unique groups
        unique_groups = np.unique(grouping_vector)
        potential_n_groups = len(unique_groups)
        
        logger.info(f"Found {potential_n_groups:,} unique groups for variance estimation")
        
        # Organize data by groups
        group_indices = {
            group_id: np.where(grouping_vector == group_id)[0]
            for group_id in unique_groups
        }
        
        # Keep track of how many groups actually have enough cells
        valid_groups = 0
        
        # Filter out groups with too few cells before training
        valid_group_indices = {}
        for group_id, indices in group_indices.items():
            if len(indices) >= min_cells:
                valid_groups += 1
                valid_group_indices[group_id] = indices
            else:
                logger.warning(f"Skipping group {group_id} (only {len(indices):,} cells < min_cells={min_cells:,})")
        
        # Set and validate that we have at least 2 valid groups
        self.n_groups = valid_groups
        if self.n_groups < 2:
            error_msg = f"At least 2 groups with sufficient cells (>= {min_cells}) are required, but only {self.n_groups} valid group(s) found"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        logger.info(f"{self.n_groups} groups have sufficient cells (>= {min_cells}) for variance estimation")
        
        # Train estimators for each valid group and store only their predictors
        logger.debug(f"Training group-specific {self.estimator_type} estimators...")
        
        for group_id, indices in valid_group_indices.items():
            logger.info(f"Training estimator for group {group_id} with {len(indices):,} cells")
            X_subset = X[indices]
            
            if self.estimator_type == 'function':
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
            Whether to show a progress bar during covariance computation. Default True.
            
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
            
            # We'll only analyze memory requirements when actually needed (in diag=False case)
            # No need to check here
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
                    # Apply Bessel's correction for unbiased variance estimate
                    # Use ddof=1 for Bessel's correction (divide by n-1 instead of n)
                    # We already validated that n_groups >= 2 in the fit method
                    return jnp.var(stacked, axis=0, ddof=1)
                
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
                # Use ddof=1 for Bessel's correction (divide by n-1 instead of n)
                # We already validated that n_groups >= 2 in the fit method
                batch_variance = jnp.var(stacked_predictions, axis=0, ddof=1)
                # Convert back to numpy for compatibility
                return np.array(batch_variance)
        
        else:
            
            # Use disk storage if enabled
            use_disk_storage = self.store_arrays_on_disk
            if use_disk_storage and self._disk_storage is None:
                # Lazy initialization of disk storage when actually needed
                # Pass dimensions so DiskStorage can estimate and check space requirements
                self._disk_storage = DiskStorage(
                    storage_dir=self.disk_storage_dir,
                    n_cells=n_cells,
                    n_genes=n_genes
                )
                logger.debug(f"Initializing disk storage for covariance matrix at {self._disk_storage.storage_dir}")
                
            # Define the covariance shape for disk-backed matrix
            covariance_shape = (n_cells, n_cells, n_genes)
            
            if use_disk_storage:
                logger.info(f"Using gene-by-gene disk storage for covariance matrix (shape={covariance_shape})")
            else:
                # Full covariance matrix computation (between all pairs of cells)
                # Only analyze memory requirements if not already done
                if not hasattr(self, '_memory_analyzed') or not self._memory_analyzed:
                    log_level = "debug" if self.store_arrays_on_disk else "info"
                    analysis = analyze_covariance_memory_requirements(
                        n_points=n_cells,
                        n_genes=n_genes,
                        max_memory_ratio=0.8,  # Standard 80% threshold
                        analysis_name="Full Covariance Matrix",
                        store_arrays_on_disk=self.store_arrays_on_disk,
                        log_level=log_level
                    )
                    # Mark that we've done the analysis so we don't do it again
                    self._memory_analyzed = True
                else:
                    logger.debug("Skipping memory analysis - already performed")
            
                
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
        
        if use_disk_storage:
            # Disk-backed version - use parallel computation with Dask if available
            if DASK_AVAILABLE:
                worker_info = f" with {self.dask_num_workers} workers" if self.dask_num_workers else " (all cores)"
                logger.info(f"Using Dask for parallel disk-backed covariance computation{worker_info} (shape={covariance_shape})")

                # Convert JAX to numpy once to avoid repeated conversions
                import gc
                centered_reshaped_np = np.asarray(centered_reshaped)
                del centered_reshaped
                gc.collect()

                # Create dask delayed functions to compute each gene slice
                @dask.delayed
                def compute_gene_cov(gene_data, n_groups):
                    """Compute covariance for a single gene."""
                    divisor = n_groups - 1
                    return np.dot(gene_data, gene_data.T) / divisor

                # Create delayed computations for all genes
                gene_arrays = []
                for g in range(n_genes):
                    gene_centered = centered_reshaped_np[:, :, g]  # (n_cells, n_groups)
                    delayed_result = compute_gene_cov(gene_centered, self.n_groups)

                    # Convert to dask array
                    gene_array = da.from_delayed(
                        delayed_result,
                        shape=(n_cells, n_cells),
                        dtype=np.float64
                    )
                    gene_arrays.append(gene_array)

                # Stack along gene axis
                # Note: Dask uses lazy evaluation - actual computation happens when array values are accessed
                # Set global pool size if num_workers is specified
                # This affects the threaded scheduler used by Dask delayed/array operations
                if self.dask_num_workers is not None:
                    dask.config.set(pool=dask.config.get("pool", {}).copy())
                    dask.config.set({"pool.num-workers": self.dask_num_workers})
                    logger.info(f"Configured Dask to use {self.dask_num_workers} workers (limits parallelism and memory)")

                dask_covariance = da.stack(gene_arrays, axis=2)

                # Clean up
                del centered_reshaped_np
                gc.collect()

                logger.info(f"Created Dask array for covariance with shape {dask_covariance.shape}")
                return dask_covariance

            else:
                # Fallback: numpy memory mapping (sequential, slower)
                logger.warning(
                    "Dask not available - using sequential numpy computation. "
                    "Install dask for parallel processing: pip install dask[array]"
                )
                logger.info("Using memory-mapped arrays for disk-backed covariance computation")

                import tempfile
                import os
                import gc

                # Create a memory-mapped array
                filename = os.path.join(self._disk_storage.storage_dir, 'covariance_matrix.npy')
                mmap_array = np.lib.format.open_memmap(
                    filename,
                    mode='w+',
                    dtype=np.float64,
                    shape=covariance_shape
                )

                # Convert JAX array to numpy ONCE before the loop
                centered_reshaped_np = np.asarray(centered_reshaped)
                del centered_reshaped
                gc.collect()

                # Process gene-by-gene sequentially
                gene_iterator = tqdm(range(n_genes), desc="Computing covariance", disable=not progress)
                for g in gene_iterator:
                    gene_centered_np = centered_reshaped_np[:, :, g]

                    # Compute covariance
                    divisor = self.n_groups - 1
                    gene_cov = np.dot(gene_centered_np, gene_centered_np.T) / divisor

                    # Store in memory-mapped array
                    mmap_array[:, :, g] = gene_cov
                    del gene_cov

                    # Periodic GC
                    if g % 50 == 0 and g > 0:
                        gc.collect(0)

                # Final cleanup
                del centered_reshaped_np
                gc.collect()

                return mmap_array
        else:
            # In-memory version - allocate full covariance matrix
            cov_matrix = np.zeros(covariance_shape)

            # For non-JIT path, convert to numpy once for efficiency
            # For JIT path, keep as JAX array for compilation
            if self._compute_cov_slice_jit is None:
                # No JIT - convert to numpy once to avoid repeated JAX slicing overhead
                import gc
                centered_reshaped_np = np.asarray(centered_reshaped)
                del centered_reshaped
                gc.collect()

                gene_iterator = tqdm(range(n_genes), desc="Computing covariance", disable=not progress)
                for g in gene_iterator:
                    gene_centered_np = centered_reshaped_np[:, :, g]  # (n_cells, n_groups)
                    gene_cov = self._compute_cov_slice(gene_centered_np, self.n_groups)
                    cov_matrix[:, :, g] = np.asarray(gene_cov)
                    del gene_cov

                del centered_reshaped_np
            else:
                # JIT path - keep as JAX array for optimal JIT performance
                gene_iterator = tqdm(range(n_genes), desc="Computing covariance", disable=not progress)
                for g in gene_iterator:
                    gene_centered = centered_reshaped[:, :, g]  # (n_cells, n_groups)
                    gene_cov = self._compute_cov_slice_jit(gene_centered, self.n_groups)
                    cov_matrix[:, :, g] = np.asarray(gene_cov)
                    del gene_cov

                del centered_reshaped

            return cov_matrix