"""Utilities for memory-efficient batch processing of large datasets."""

import numpy as np
import logging
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Union, TypeVar, cast
from tqdm.auto import tqdm

logger = logging.getLogger("kompot")

# Type for the function to be decorated
F = TypeVar('F', bound=Callable[..., Any])

def is_jax_memory_error(error: Exception) -> bool:
    """
    Check if an exception is a JAX memory error.
    
    JAX-specific memory errors typically contain 'RESOURCE_EXHAUSTED' or 'Out of memory'
    in the error message.
    
    Parameters
    ----------
    error : Exception
        The exception to check
        
    Returns
    -------
    bool
        True if the error is a JAX memory error, False otherwise
    """
    error_str = str(error)
    return any(msg in error_str.lower() for msg in [
        "resource_exhausted", 
        "resource exhausted",
        "out of memory", 
        "memory"
    ])


def merge_batch_results(results: List[Any]) -> Any:
    """
    Merge results from batched processing.
    
    This function handles different types of results and merges them appropriately:
    - Dictionaries: merged by key
    - NumPy arrays: concatenated along axis 0
    - Lists: flattened
    
    Parameters
    ----------
    results : List[Any]
        List of results from batched processing
        
    Returns
    -------
    Any
        Merged results
    """
    if not results:
        return {}
    
    # If results are dictionaries, merge them by key
    if isinstance(results[0], dict):
        merged = {}
        # Get keys from first result
        keys = results[0].keys()
        
        for key in keys:
            # Check if all results have this key
            if all(key in res for res in results):
                # Get arrays for this key from all results
                arrays = [res[key] for res in results]
                # Check if elements are arrays or can be combined
                if all(isinstance(arr, (np.ndarray, list)) for arr in arrays):
                    try:
                        # Try to concatenate arrays
                        if isinstance(arrays[0], np.ndarray):
                            if len(arrays[0].shape) == 0:  # It's a scalar
                                merged[key] = arrays[0]  # Just use the first one
                            else:
                                merged[key] = np.concatenate(arrays, axis=0)
                        else:  # It's a list
                            merged[key] = [item for sublist in arrays for item in sublist]
                    except ValueError:
                        # If concatenation fails, just collect results in a list
                        merged[key] = arrays
                else:
                    # Non-array values, just collect them
                    merged[key] = arrays
        
        return merged
    
    # If results are numpy arrays, concatenate them
    elif isinstance(results[0], np.ndarray):
        return np.concatenate(results, axis=0)
    
    # Otherwise, return the list of results
    return results


def batch_process(default_batch_size: int = 500):
    """
    Decorator for batch processing data in predict methods.
    
    This decorator handles memory-efficient batch processing with automatic fallback
    to smaller batch sizes if memory errors occur. It's particularly useful for
    methods that process large datasets and might encounter memory limitations,
    especially when using JAX.
    
    Parameters
    ----------
    default_batch_size : int, optional
        Default batch size to use if not specified in the instance, by default 500
        
    Returns
    -------
    Callable
        Decorated function with batch processing capabilities
    
    Examples
    --------
    >>> @batch_process(default_batch_size=100)
    >>> def predict(self, X_new):
    >>>     # Your prediction code here
    >>>     return result
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(self, X_new, *args, **kwargs):
            # Get batch size from instance or use default
            batch_size = getattr(self, 'batch_size', default_batch_size)
            
            # If batch_size is set to None or 0, process all data at once
            if batch_size is None or batch_size <= 0:
                return func(self, X_new, *args, **kwargs)
            
            # For very small inputs, process all at once
            if len(X_new) <= batch_size:
                return func(self, X_new, *args, **kwargs)
            
            # Try processing in batches
            n_samples = len(X_new)
            original_batch_size = batch_size
            
            # Track which reduction factors we've tried to avoid repeated attempts
            tried_reduction_factors = set()
            
            # We'll collect results for each batch here
            batch_results = []
            
            # Track which samples we've successfully processed
            processed_indices = set()
            
            # First attempt with specified batch size
            logger.info(f"Processing data in batches of size {batch_size}")
            for start_idx in tqdm(range(0, n_samples, batch_size), 
                                desc=f"Processing data (batch size={batch_size})"):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = list(range(start_idx, end_idx))
                
                # Skip batch if all indices have been processed
                if all(idx in processed_indices for idx in batch_indices):
                    continue
                    
                # Filter out already processed indices
                remaining_indices = [idx for idx in batch_indices if idx not in processed_indices]
                if not remaining_indices:
                    continue
                
                batch_X = X_new[remaining_indices]
                
                try:
                    # Process this batch
                    result = func(self, batch_X, *args, **kwargs)
                    batch_results.append(result)
                    processed_indices.update(remaining_indices)
                except Exception as e:
                    # Check if it's a memory-related error
                    if is_jax_memory_error(e):
                        logger.warning(f"Memory error with batch size {batch_size}. Will try with smaller batches.")
                        # Continue with next batch size - we'll handle remaining indices later
                    else:
                        # If it's not a memory error, re-raise
                        raise
            
            # If we haven't processed all samples, try with smaller batch sizes
            if len(processed_indices) < n_samples:
                # Define reduction factors to try
                reduction_factors = [2, 5, 10, 20, 50, 100, 200]
                
                for reduction_factor in reduction_factors:
                    # Skip reduction factors we've already tried
                    if reduction_factor in tried_reduction_factors:
                        continue
                        
                    tried_reduction_factors.add(reduction_factor)
                    current_batch_size = max(1, original_batch_size // reduction_factor)
                    
                    # Skip if we've already tried this batch size through another reduction factor
                    if any(current_batch_size == original_batch_size // factor 
                          for factor in tried_reduction_factors if factor != reduction_factor):
                        continue
                    
                    logger.info(f"Retrying with reduced batch size: {current_batch_size}")
                    
                    remaining_indices = list(set(range(n_samples)) - processed_indices)
                    remaining_indices.sort()  # Keep original order
                    
                    if not remaining_indices:
                        break  # All done
                        
                    for i in range(0, len(remaining_indices), current_batch_size):
                        batch_indices = remaining_indices[i:i + current_batch_size]
                        if not batch_indices:
                            continue
                            
                        batch_X = X_new[batch_indices]
                        
                        try:
                            result = func(self, batch_X, *args, **kwargs)
                            batch_results.append(result)
                            processed_indices.update(batch_indices)
                        except Exception as e:
                            # Only continue batch size reduction if it's a memory error
                            if is_jax_memory_error(e):
                                # This batch size is still too large, break and try smaller
                                logger.warning(f"Memory error with batch size {current_batch_size}. "
                                              f"Trying smaller batch size.")
                                break
                            else:
                                # Not a memory error, re-raise
                                raise
                    
                    # Check if we've processed all indices
                    remaining_indices = list(set(range(n_samples)) - processed_indices)
                    if not remaining_indices:
                        break  # All done
                
                # If we still have remaining indices, process one by one as a last resort
                remaining_indices = list(set(range(n_samples)) - processed_indices)
                if remaining_indices:
                    logger.warning(f"Processing {len(remaining_indices)} remaining samples one by one. This may be slow.")
                    
                    for idx in tqdm(remaining_indices, desc="Processing individual samples"):
                        try:
                            result = func(self, X_new[idx:idx+1], *args, **kwargs)
                            batch_results.append(result)
                            processed_indices.add(idx)
                        except Exception as e:
                            logger.error(f"Failed to process sample {idx}: {str(e)}")
                            if not is_jax_memory_error(e):
                                raise
            
            # Check if we've successfully processed all samples
            if len(processed_indices) < n_samples:
                remaining = n_samples - len(processed_indices)
                logger.warning(f"Failed to process {remaining} samples ({remaining/n_samples:.1%} of total)")
                
            # If we have results, combine them
            if batch_results:
                return merge_batch_results(batch_results)
            else:
                # No results at all
                logger.error("No successful batch processing. Returning empty result.")
                return {}
                
        return cast(F, wrapper)
    return decorator


def apply_batched(
    func: Callable,
    X: np.ndarray,
    batch_size: int = 500,
    axis: int = 0,
    show_progress: bool = True,
    desc: Optional[str] = None
) -> Any:
    """
    Apply a function to data in batches.
    
    This is a utility function for applying any function to data in a batched manner,
    automatically handling batch size reduction if memory errors occur.
    
    Parameters
    ----------
    func : Callable
        Function to apply to batches of data
    X : np.ndarray
        Input data array
    batch_size : int, optional
        Batch size to use, by default 500
    axis : int, optional
        Axis along which to batch the data, by default 0
    show_progress : bool, optional
        Whether to show a progress bar, by default True
    desc : str, optional
        Description for the progress bar, by default None
        
    Returns
    -------
    Any
        Combined results from batched processing
        
    Examples
    --------
    >>> # Apply a function to data in batches
    >>> result = apply_batched(
    >>>     lambda x: np.mean(x, axis=1), 
    >>>     large_array, 
    >>>     batch_size=1000
    >>> )
    """
    # If batch_size is None or 0, or input is small, process all at once
    if batch_size is None or batch_size <= 0 or X.shape[axis] <= batch_size:
        return func(X)
    
    n_samples = X.shape[axis]
    original_batch_size = batch_size
    
    # Track which reduction factors we've tried to avoid repeated attempts
    tried_reduction_factors = set()
    
    # We'll collect results for each batch here
    batch_results = []
    
    # Track which samples we've successfully processed
    processed_indices = set()
    
    # Define a progress iterator
    progress_iter = tqdm(
        range(0, n_samples, batch_size),
        desc=desc or f"Processing (batch_size={batch_size})",
        disable=not show_progress
    )
    
    # First attempt with specified batch size
    for start_idx in progress_iter:
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = list(range(start_idx, end_idx))
        
        # Skip batch if all indices have been processed
        if all(idx in processed_indices for idx in batch_indices):
            continue
            
        # Filter out already processed indices
        remaining_indices = [idx for idx in batch_indices if idx not in processed_indices]
        if not remaining_indices:
            continue
        
        # Create slice for batch extraction
        batch_slice = [slice(None)] * X.ndim
        batch_slice[axis] = remaining_indices
        batch_X = X[tuple(batch_slice)]
        
        try:
            # Process this batch
            result = func(batch_X)
            batch_results.append(result)
            processed_indices.update(remaining_indices)
        except Exception as e:
            # Check if it's a memory-related error
            if is_jax_memory_error(e):
                logger.warning(f"Memory error with batch size {batch_size}. Will try smaller batches.")
            else:
                # If it's not a memory error, re-raise
                raise
    
    # If we haven't processed all samples, try with smaller batch sizes
    if len(processed_indices) < n_samples:
        # Define reduction factors to try
        reduction_factors = [2, 5, 10, 20, 50, 100, 200]
        
        for reduction_factor in reduction_factors:
            # Skip reduction factors we've already tried
            if reduction_factor in tried_reduction_factors:
                continue
                
            tried_reduction_factors.add(reduction_factor)
            current_batch_size = max(1, original_batch_size // reduction_factor)
            
            # Skip if we've already tried this batch size through another reduction factor
            if any(current_batch_size == original_batch_size // factor 
                  for factor in tried_reduction_factors if factor != reduction_factor):
                continue
            
            logger.info(f"Retrying with reduced batch size: {current_batch_size}")
            
            remaining_indices = list(set(range(n_samples)) - processed_indices)
            remaining_indices.sort()  # Keep original order
            
            if not remaining_indices:
                break  # All done
                
            progress_iter = tqdm(
                range(0, len(remaining_indices), current_batch_size),
                desc=f"Processing (batch_size={current_batch_size})",
                disable=not show_progress
            )
            
            for i in progress_iter:
                batch_indices = remaining_indices[i:i + current_batch_size]
                if not batch_indices:
                    continue
                    
                # Create slice for batch extraction
                batch_slice = [slice(None)] * X.ndim
                batch_slice[axis] = batch_indices
                batch_X = X[tuple(batch_slice)]
                
                try:
                    result = func(batch_X)
                    batch_results.append(result)
                    processed_indices.update(batch_indices)
                except Exception as e:
                    # Only continue batch size reduction if it's a memory error
                    if is_jax_memory_error(e):
                        # This batch size is still too large, break and try smaller
                        logger.warning(f"Memory error with batch size {current_batch_size}. "
                                      f"Trying smaller batch size.")
                        break
                    else:
                        # Not a memory error, re-raise
                        raise
            
            # Check if we've processed all indices
            remaining_indices = list(set(range(n_samples)) - processed_indices)
            if not remaining_indices:
                break  # All done
        
        # If we still have remaining indices, process one by one as a last resort
        remaining_indices = list(set(range(n_samples)) - processed_indices)
        if remaining_indices:
            logger.warning(f"Processing {len(remaining_indices)} remaining samples one by one. This may be slow.")
            
            progress_iter = tqdm(
                remaining_indices,
                desc="Processing individual samples",
                disable=not show_progress
            )
            
            for idx in progress_iter:
                try:
                    # Create slice for single sample extraction
                    single_slice = [slice(None)] * X.ndim
                    single_slice[axis] = slice(idx, idx+1)
                    single_X = X[tuple(single_slice)]
                    
                    result = func(single_X)
                    batch_results.append(result)
                    processed_indices.add(idx)
                except Exception as e:
                    logger.error(f"Failed to process sample {idx}: {str(e)}")
                    if not is_jax_memory_error(e):
                        raise
    
    # Check if we've successfully processed all samples
    if len(processed_indices) < n_samples:
        remaining = n_samples - len(processed_indices)
        logger.warning(f"Failed to process {remaining} samples ({remaining/n_samples:.1%} of total)")
        
    # If we have results, combine them
    if batch_results:
        return merge_batch_results(batch_results)
    else:
        # No results at all
        logger.error("No successful batch processing. Returning empty result.")
        return {}