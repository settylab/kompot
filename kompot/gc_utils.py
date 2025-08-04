"""
Garbage collection utilities for memory optimization in performance-critical sections.

This module provides utilities to optimize memory usage and garbage collection 
in computationally intensive operations, following best practices for CPython
memory management.
"""

import gc
import logging
from contextlib import contextmanager
from typing import Optional, Union, List, Any
import weakref

logger = logging.getLogger("kompot")


@contextmanager
def no_gc(generation: Optional[int] = None):
    """
    Context manager to disable garbage collection during performance-critical operations.
    
    In CPython, most objects are freed immediately when their reference count hits zero.
    The garbage collector only kicks in to break reference cycles. By disabling GC
    during tight loops and manually collecting at safe points, we can improve
    performance and reduce memory pressure.
    
    Parameters
    ----------
    generation : int, optional
        Specific generation to collect after re-enabling GC (0, 1, or 2).
        If None, performs a full collection. Generation 0 is fastest and
        collects newly created objects.
        
    Examples
    --------
    >>> with no_gc():
    ...     # Performance-critical loop
    ...     for i in range(large_number):
    ...         do_heavy_computation()
    
    >>> with no_gc(generation=0):
    ...     # Process large batch with cleanup of new objects only
    ...     process_large_batch()
    """
    was_enabled = gc.isenabled()
    if was_enabled:
        gc.disable()
        logger.debug("Disabled garbage collection for performance-critical section")
    
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()
            # Perform controlled cleanup
            if generation is not None:
                collected = gc.collect(generation)
                logger.debug(f"Re-enabled GC and collected {collected} objects from generation {generation}")
            else:
                collected = gc.collect()
                logger.debug(f"Re-enabled GC and collected {collected} objects")


def tune_gc_thresholds(gen0: int = 2000, gen1: int = 10, gen2: int = 10):
    """
    Tune garbage collection thresholds to reduce automatic GC frequency.
    
    By raising the trigger thresholds, GC runs less frequently during
    intensive computations. The default CPython thresholds are typically
    (700, 10, 10).
    
    Parameters
    ----------
    gen0 : int, optional
        Threshold for generation 0 (newly created objects), by default 2000
    gen1 : int, optional  
        Threshold for generation 1, by default 10
    gen2 : int, optional
        Threshold for generation 2 (oldest objects), by default 10
        
    Returns
    -------
    tuple
        Previous thresholds for restoration if needed
    """
    old_thresholds = gc.get_threshold()
    gc.set_threshold(gen0, gen1, gen2)
    logger.debug(f"Updated GC thresholds from {old_thresholds} to ({gen0}, {gen1}, {gen2})")
    return old_thresholds


def restore_gc_thresholds(thresholds: tuple):
    """
    Restore garbage collection thresholds.
    
    Parameters
    ----------
    thresholds : tuple
        Previous thresholds returned by tune_gc_thresholds()
    """
    gc.set_threshold(*thresholds)
    logger.debug(f"Restored GC thresholds to {thresholds}")


def explicit_cleanup(containers: Union[List[Any], Any], 
                    collect_generation: int = 0):
    """
    Explicitly clean up large containers and run targeted garbage collection.
    
    This function clears containers and removes references, then runs GC
    on a specific generation for efficient memory reclamation.
    
    Parameters
    ----------
    containers : list or single container
        Container(s) to clean up (lists, dicts, arrays, etc.)
    collect_generation : int, optional
        GC generation to collect (0=fastest, 2=full collection), by default 0
        
    Examples
    --------
    >>> large_results = compute_big_table()
    >>> process(large_results)
    >>> explicit_cleanup(large_results)
    
    >>> # Multiple containers
    >>> explicit_cleanup([results1, results2, cache_dict])
    """
    # Handle single container vs list of containers
    if isinstance(containers, list) and len(containers) > 0:
        # Check if this is a list of containers or a single list container
        # If all elements are clearable containers, treat as list of containers
        # If it's just a list with non-container elements, treat as single container
        if all(hasattr(item, 'clear') or hasattr(item, '__del__') for item in containers 
               if item is not None):
            container_list = containers
        else:
            # This is a single list container
            container_list = [containers]
    else:
        # Single container or empty list
        container_list = [containers] if containers is not None else []
    
    cleaned_count = 0
    for container in container_list:
        if container is not None:
            try:
                # Clear container if it has a clear method
                if hasattr(container, 'clear') and callable(getattr(container, 'clear')):
                    container.clear()
                    cleaned_count += 1
                # For other objects, we can't really "clean" them, just trigger GC
                else:
                    cleaned_count += 1
            except Exception as e:
                logger.warning(f"Failed to clean container: {e}")
    
    # Run targeted garbage collection
    if cleaned_count > 0:
        collected = gc.collect(collect_generation)
        logger.debug(f"Cleaned {cleaned_count} containers and collected {collected} objects from generation {collect_generation}")


class WeakContainer:
    """
    Container that holds weak references to prevent reference cycles.
    
    Use this for back-references or when you need to break potential cycles
    in your data structures while still maintaining access to objects.
    
    Examples
    --------
    >>> # Instead of strong back-reference that creates cycles
    >>> parent.child = child
    >>> child.parent = parent  # Creates cycle
    
    >>> # Use weak reference for back-pointer
    >>> parent.child = child
    >>> child.parent_ref = WeakContainer(parent)  # No cycle
    >>> parent_obj = child.parent_ref.get()  # Access when needed
    """
    
    def __init__(self, obj):
        """
        Initialize with a weak reference to the object.
        
        Parameters
        ----------
        obj : Any
            Object to hold a weak reference to
        """
        try:
            self._ref = weakref.ref(obj)
        except TypeError:
            # Some objects can't have weak references
            logger.warning(f"Cannot create weak reference to {type(obj)}, storing strong reference")
            self._ref = lambda: obj
    
    def get(self):
        """
        Get the referenced object, if it still exists.
        
        Returns
        -------
        object or None
            The referenced object, or None if it has been garbage collected
        """
        return self._ref()
    
    def is_alive(self) -> bool:
        """
        Check if the referenced object is still alive.
        
        Returns
        -------
        bool
            True if object exists, False if garbage collected
        """
        return self.get() is not None


@contextmanager 
def memory_efficient_loop(tune_thresholds: bool = True,
                         generation_cleanup: int = 0,
                         cleanup_frequency: int = 100):
    """
    Context manager for memory-efficient processing of large loops.
    
    Combines GC disabling, threshold tuning, and periodic cleanup for
    optimal memory management during intensive operations.
    
    Parameters
    ----------
    tune_thresholds : bool, optional
        Whether to tune GC thresholds for the operation, by default True
    generation_cleanup : int, optional
        GC generation to use for cleanup (0=fastest), by default 0  
    cleanup_frequency : int, optional
        How often to perform cleanup during the operation, by default 100
        
    Examples
    --------
    >>> with memory_efficient_loop() as cleanup_fn:
    ...     for i, item in enumerate(large_dataset):
    ...         result = process_item(item)
    ...         # Periodic cleanup
    ...         if i % 50 == 0:
    ...             cleanup_fn([temp_arrays, cache])
    """
    # Store original thresholds
    original_thresholds = None
    if tune_thresholds:
        original_thresholds = tune_gc_thresholds()
    
    # Disable GC
    was_enabled = gc.isenabled()
    if was_enabled:
        gc.disable()
        logger.debug("Started memory-efficient loop with GC disabled")
    
    def cleanup_function(containers=None):
        """Function to perform cleanup during the loop."""
        if containers:
            explicit_cleanup(containers, generation_cleanup)
        else:
            gc.collect(generation_cleanup)
    
    try:
        yield cleanup_function
    finally:
        # Re-enable GC and cleanup
        if was_enabled:
            gc.enable()
            gc.collect(generation_cleanup)
            
        # Restore original thresholds
        if tune_thresholds and original_thresholds:
            restore_gc_thresholds(original_thresholds)
            
        logger.debug("Completed memory-efficient loop and restored GC settings")


def get_memory_stats() -> dict:
    """
    Get current garbage collection statistics.
    
    Returns
    -------
    dict
        Dictionary containing GC statistics including counts by generation
        and current thresholds
    """
    stats = {
        'enabled': gc.isenabled(),
        'thresholds': gc.get_threshold(),
        'counts': gc.get_count(),
        'stats': gc.get_stats() if hasattr(gc, 'get_stats') else None
    }
    return stats


def log_memory_stats(level=logging.DEBUG):
    """
    Log current memory and GC statistics.
    
    Parameters
    ---------- 
    level : int, optional
        Logging level to use, by default logging.DEBUG
    """
    stats = get_memory_stats()
    logger.log(level, f"GC Stats - Enabled: {stats['enabled']}, "
                     f"Thresholds: {stats['thresholds']}, "
                     f"Counts: {stats['counts']}")