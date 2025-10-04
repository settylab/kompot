"""Utilities for memory-efficient batch processing of large datasets."""

import numpy as np
import jax.numpy as jnp
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


def merge_batch_results(results: List[Any], concat_axis: int = 0) -> Any:
    """
    Merge results from batched processing.

    This function handles different types of results and merges them appropriately:
    - Dictionaries: merged by key
    - NumPy or JAX arrays: concatenated along specified axis
    - Lists: flattened

    Parameters
    ----------
    results : List[Any]
        List of results from batched processing
    concat_axis : int, optional
        Axis along which to concatenate arrays, by default 0

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
        # Get all unique keys from all results
        all_keys = set()
        for res in results:
            all_keys.update(res.keys())

        for key in all_keys:
            # Collect values for this key from all results where it exists
            values = [res[key] for res in results if key in res]
            if not values:
                continue

            # Handle arrays (numpy or jax)
            if all(isinstance(val, (np.ndarray, jnp.ndarray)) for val in values):
                # Determine if we're working with JAX arrays
                is_jax = isinstance(values[0], jnp.ndarray)
                concat_fn = jnp.concatenate if is_jax else np.concatenate

                try:
                    # Handle scalar arrays (arrays with shape () or (1,))
                    if all(len(val.shape) == 0 for val in values):
                        # For true scalars (shape ()), create a new array
                        array_vals = [val.item() for val in values]
                        merged[key] = jnp.array(array_vals) if is_jax else np.array(array_vals)
                    else:
                        # Standard case: concatenate arrays along specified axis
                        merged[key] = concat_fn(values, axis=concat_axis)
                except ValueError as e:
                    logger.warning(f"Failed to concatenate arrays for key '{key}': {e}")
                    # Always return arrays - don't fall back to list
                    merged[key] = values
            elif all(isinstance(val, list) for val in values):
                # Handle lists - flatten them
                merged[key] = [item for sublist in values for item in sublist]
            else:
                # For other types, just keep the list
                merged[key] = values

        return merged

    # If results are arrays, concatenate them
    elif all(isinstance(res, (np.ndarray, jnp.ndarray)) for res in results):
        # Check if we're working with JAX arrays
        is_jax = isinstance(results[0], jnp.ndarray)
        concat_fn = jnp.concatenate if is_jax else np.concatenate

        # Handle scalar arrays
        if all(len(res.shape) == 0 for res in results):
            # For true scalars (shape ()), create a new array
            array_vals = [res.item() for res in results]
            return jnp.array(array_vals) if is_jax else np.array(array_vals)
        else:
            # Standard case: concatenate along specified axis
            return concat_fn(results, axis=concat_axis)

    # If results are lists, flatten them
    elif all(isinstance(res, list) for res in results):
        return [item for sublist in results for item in sublist]

    # For other types, just return the list
    return results


def batch_process(default_batch_size: int = 500):
    """
    Decorator for batch processing data in predict methods.

    This decorator handles memory-efficient batch processing with automatic fallback
    to smaller batch sizes if memory errors occur. It's particularly useful for
    methods that process large datasets and might encounter memory limitations.

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

            # Use apply_batched to handle the batching
            return apply_batched(
                func=lambda x: func(self, x, *args, **kwargs),
                X=X_new,
                batch_size=batch_size,
                axis=0,
                show_progress=False,
                desc=None,
                concat_axis=0
            )
        return cast(F, wrapper)
    return decorator


def batched(
    batch_size: Optional[int] = None,
    axis: int = 0,
    desc: Optional[str] = None,
    show_progress: bool = True,
    concat_axis: int = 0
) -> Callable[[F], F]:
    """
    Decorator that automatically applies batched processing to a function.

    This decorator wraps a function to process its input in batches. It handles
    memory errors by automatically reducing the batch size and retrying.

    Parameters
    ----------
    batch_size : int, optional
        Number of samples to process per batch. If None, processes all at once.
    axis : int, optional
        Axis along which to split the input, by default 0
    desc : str, optional
        Description for the progress bar, by default None
    show_progress : bool, optional
        Whether to show a progress bar, by default True
    concat_axis : int, optional
        Axis along which to concatenate results, by default 0

    Returns
    -------
    Callable
        Decorated function that processes input in batches

    Examples
    --------
    >>> @batched(batch_size=100)
    >>> def process_data(X):
    >>>     return np.mean(X, axis=1)
    >>> result = process_data(large_array)
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(X: Any, *args, **kwargs) -> Any:
            return apply_batched(
                func=lambda x: func(x, *args, **kwargs),
                X=X,
                batch_size=batch_size,
                axis=axis,
                show_progress=show_progress,
                desc=desc,
                concat_axis=concat_axis
            )
        return cast(F, wrapper)
    return decorator


def apply_batched(
    func: Callable[[Any], Any],
    X: Any,
    batch_size: Optional[int] = None,
    axis: int = 0,
    show_progress: bool = True,
    desc: Optional[str] = None,
    concat_axis: int = 0
) -> Any:
    """
    Apply a function to data in batches with automatic memory error handling.

    This function splits the input along the specified axis and processes it in batches.
    If memory errors occur, it automatically reduces the batch size and retries.
    All samples must be processed successfully - partial success is not supported.

    Parameters
    ----------
    func : Callable
        Function to apply to each batch. Should accept input with same structure as X.
    X : Any
        Input data to process. Can be numpy array, JAX array, or other indexable type.
    batch_size : int, optional
        Number of samples per batch. If None/0 or >= input size, processes all at once.
    axis : int, optional
        Axis along which to split the input, by default 0
    show_progress : bool, optional
        Whether to show a progress bar, by default True
    desc : str, optional
        Description for the progress bar, by default None
    concat_axis : int, optional
        Axis along which to concatenate result arrays, by default 0

    Returns
    -------
    Any
        Combined results from batched processing

    Raises
    ------
    RuntimeError
        If processing fails even with smallest batch size
    Exception
        Any non-memory errors are raised immediately

    Examples
    --------
    >>> # Apply a function to data in batches
    >>> result = apply_batched(
    >>>     lambda x: np.mean(x, axis=1),
    >>>     large_array,
    >>>     batch_size=1000
    >>> )
    """
    # If batch_size is None or 0, or input is small, try to process all at once
    # If that fails due to memory error, fall back to using a default batch size of 500
    if batch_size is None or batch_size <= 0 or X.shape[axis] <= batch_size:
        try:
            return func(X)
        except Exception as e:
            if is_jax_memory_error(e):
                # Fall back to batched processing with smaller batch size
                if batch_size is None or batch_size <= 0:
                    logger.warning(f"Memory error encountered with batch_size=None. Falling back to batch_size=500")
                    batch_size = 500
                else:
                    # Reduce the current batch size to enable batching
                    new_batch_size = max(1, batch_size // 2)
                    logger.warning(f"Memory error detected with batch_size={batch_size}. Falling back to batch_size={new_batch_size}")
                    batch_size = new_batch_size
                # Continue with batched processing below
            else:
                # If it's not a memory error, re-raise
                raise

    n_samples = X.shape[axis]

    # Try processing with adaptive batch sizing for memory errors
    while batch_size >= 1:
        try:
            # Process first batch to determine output shape/type and pre-allocate if possible
            first_batch_size = min(batch_size, n_samples)
            batch_slice = [slice(None)] * X.ndim
            batch_slice[axis] = slice(0, first_batch_size)
            first_batch_X = X[tuple(batch_slice)]

            first_result = func(first_batch_X)

            # Check if we can pre-allocate (numpy/jax arrays only)
            can_preallocate = isinstance(first_result, (np.ndarray, jnp.ndarray))

            if first_batch_size == n_samples:
                # Already processed everything in first batch
                return first_result

            if can_preallocate:
                # Pre-allocate output array
                output_shape = list(first_result.shape)
                output_shape[concat_axis] = n_samples
                output = np.empty(output_shape, dtype=first_result.dtype)

                # Fill in first batch
                output_slice = [slice(None)] * output.ndim
                output_slice[concat_axis] = slice(0, first_batch_size)
                output[tuple(output_slice)] = first_result

                # Define a progress iterator starting from second batch
                progress_iter = tqdm(
                    range(first_batch_size, n_samples, batch_size),
                    desc=desc or f"Processing (batch_size={batch_size})",
                    disable=not show_progress,
                    initial=1,
                    total=(n_samples + batch_size - 1) // batch_size
                )

                # Process remaining batches with pre-allocated output
                for start_idx in progress_iter:
                    end_idx = min(start_idx + batch_size, n_samples)

                    # Create slice for batch extraction
                    batch_slice = [slice(None)] * X.ndim
                    batch_slice[axis] = slice(start_idx, end_idx)
                    batch_X = X[tuple(batch_slice)]

                    # Process this batch
                    result = func(batch_X)

                    # Write directly to output
                    output_slice = [slice(None)] * output.ndim
                    output_slice[concat_axis] = slice(start_idx, end_idx)
                    output[tuple(output_slice)] = result

                return output
            else:
                # Use list accumulation for non-array outputs
                batch_results = [first_result]

                progress_iter = tqdm(
                    range(first_batch_size, n_samples, batch_size),
                    desc=desc or f"Processing (batch_size={batch_size})",
                    disable=not show_progress
                )

                for start_idx in progress_iter:
                    end_idx = min(start_idx + batch_size, n_samples)

                    # Create slice for batch extraction
                    batch_slice = [slice(None)] * X.ndim
                    batch_slice[axis] = slice(start_idx, end_idx)
                    batch_X = X[tuple(batch_slice)]

                    # Process this batch
                    result = func(batch_X)
                    batch_results.append(result)

                return merge_batch_results(batch_results, concat_axis=concat_axis)

        except Exception as e:
            if is_jax_memory_error(e):
                # Reduce batch size and retry
                new_batch_size = max(1, batch_size // 2)
                if new_batch_size == batch_size:
                    # Can't reduce further (already at 1)
                    logger.error(f"Memory error even with batch_size=1. Cannot process data.")
                    raise RuntimeError(f"Out of memory even with smallest batch size (1). Error: {str(e)}") from e
                logger.info(f"Memory error with batch_size={batch_size}. Retrying with batch_size={new_batch_size}")
                batch_size = new_batch_size
            else:
                # Non-memory error, raise immediately
                raise

    # Should never reach here
    raise RuntimeError("Failed to process data with adaptive batch sizing")
