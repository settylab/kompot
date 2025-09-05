"""Utilities for memory management and estimation."""

import numpy as np
import psutil
import os
import tempfile
import logging
import importlib.util
from typing import Tuple, Union, Optional, Dict, Any, List

logger = logging.getLogger("kompot")

# Check if dask is available
DASK_AVAILABLE = importlib.util.find_spec("dask") is not None
if DASK_AVAILABLE:
    try:
        import dask.array as da
        logger.debug("Dask is available for disk-backed array operations")
    except ImportError:
        DASK_AVAILABLE = False
        logger.debug("Dask import failed despite being detected")

def get_dask_array(array_shape: tuple, chunk_size: Optional[int] = None) -> Union['da.Array', None]:
    """
    Create a Dask array with the given shape if Dask is available.
    
    Parameters
    ----------
    array_shape : tuple
        Shape of the array to create
    chunk_size : int, optional
        Chunk size to use for the array, if None will calculate a reasonable default
    
    Returns
    -------
    dask.array.Array or None
        Dask array if available, otherwise None
    """
    if not DASK_AVAILABLE:
        return None
        
    import dask.array as da
    
    # Calculate reasonable chunk size if not provided
    if chunk_size is None:
        # For 3D arrays, chunk along the gene dimension (shape[2]) for best performance
        if len(array_shape) == 3:
            # Use small chunks for first two dimensions (cells, cells) and larger for gene dimension
            chunks = (min(100, array_shape[0]), 
                     min(100, array_shape[1]), 
                     1)  # Process one gene at a time
        else:
            # For other dimensions, use a simple heuristic (chunk size around 100MB)
            # First estimate element size based on float64
            elem_size = 8  # bytes (assuming float64)
            total_size = np.prod(array_shape) * elem_size
            
            # Target chunk size ~100MB
            target_chunk_size = 100 * 1024 * 1024  # 100 MB in bytes
            
            # Calculate number of chunks
            n_chunks = max(1, int(total_size / target_chunk_size))
            
            # Divide each dimension roughly equally
            dim_factor = n_chunks ** (1 / len(array_shape))
            chunks = tuple(max(1, int(dim / dim_factor)) for dim in array_shape)
    else:
        # Use specified chunk size
        chunks = chunk_size
        
    return da.zeros(array_shape, chunks=chunks)


def human_readable_size(size_in_bytes: int) -> str:
    """
    Convert a size in bytes to a human-readable string with appropriate units.
    
    Parameters
    ----------
    size_in_bytes : int
        Size in bytes
        
    Returns
    -------
    str
        Human-readable size string (e.g., '1.23 GB')
    """
    units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    size = float(size_in_bytes)
    unit_index = 0
    
    while size >= 1024.0 and unit_index < len(units) - 1:
        size /= 1024.0
        unit_index += 1
        
    return f"{size:.2f} {units[unit_index]}"


def array_size(shape: tuple, dtype=np.float64) -> Tuple[str, int]:
    """
    Compute theoretical size of a NumPy array.
    
    Parameters
    ----------
    shape : tuple
        Shape of the array
    dtype : numpy.dtype, optional
        Data type of the array elements, by default np.float64
        
    Returns
    -------
    tuple
        (human_readable_size_string, size_in_bytes)
    """
    dtype_size = np.dtype(dtype).itemsize  # Size of one element in bytes
    num_elements = np.prod(shape)  # Total number of elements
    total_size = num_elements * dtype_size  # Total size in bytes
    return human_readable_size(total_size), total_size


def get_available_memory() -> Tuple[str, int]:
    """
    Get the available system memory.
    
    Returns
    -------
    tuple
        (human_readable_string, size_in_bytes)
    """
    available = psutil.virtual_memory().available
    return human_readable_size(available), available


def memory_requirement_ratio(array_shape: tuple, dtype=np.float64) -> float:
    """
    Calculate the ratio of required memory for an array to available system memory.
    
    Parameters
    ----------
    array_shape : tuple
        Shape of the array
    dtype : numpy.dtype, optional
        Data type of the array elements, by default np.float64
        
    Returns
    -------
    float
        Ratio of required memory to available memory
    """
    _, array_bytes = array_size(array_shape, dtype)
    _, available_bytes = get_available_memory()
    return array_bytes / available_bytes


def analyze_memory_requirements(shapes: List[tuple], max_memory_ratio: float = 0.8, 
                               analysis_name: str = "Memory Analysis",
                               store_arrays_on_disk: bool = False,
                               log_level: str = "info") -> Dict[str, Any]:
    """
    Analyze memory requirements for arrays with specified shapes.
    
    Parameters
    ----------
    shapes : list
        List of array shapes to analyze (tuples of dimensions)
    max_memory_ratio : float, optional
        Maximum fraction of available memory that arrays should occupy before
        triggering warnings or enabling disk storage, by default 0.8 (80%)
    analysis_name : str, optional
        Name to identify this analysis in log messages, by default "Memory Analysis"
    store_arrays_on_disk : bool, optional
        Whether disk storage is already enabled (suppresses warnings), by default False
    log_level : str, optional
        Level to log basic memory information at ('debug', 'info', etc.), by default "info"
        
    Returns
    -------
    dict
        Dictionary with memory analysis results including:
        - array_sizes: List of dictionaries with details for each array
        - total_size: Human-readable total size string
        - total_bytes: Total size in bytes
        - available_memory: Human-readable available memory string
        - available_bytes: Available memory in bytes
        - memory_ratio: Ratio of required memory to available memory
        - status: 'ok', 'warning', or 'critical' based on memory ratio
    """
    # Calculate individual array sizes
    array_sizes = []
    total_bytes = 0
    
    for i, shape in enumerate(shapes):
        size_str, size_bytes = array_size(shape)
        array_sizes.append({
            'index': i,
            'shape': shape,
            'size_str': size_str,
            'size_bytes': size_bytes
        })
        total_bytes += size_bytes
        
    # Calculate total size and memory ratio
    total_size_str = human_readable_size(total_bytes)
    avail_str, avail_bytes = get_available_memory()
    memory_ratio = total_bytes / avail_bytes
    
    # Determine status based on memory ratio
    if memory_ratio > max_memory_ratio:
        status = 'critical'
    elif memory_ratio > max_memory_ratio * 0.5:
        status = 'warning'
    else:
        status = 'ok'
    
    # Create result dictionary
    result = {
        'array_sizes': array_sizes,
        'total_size': total_size_str,
        'total_bytes': total_bytes,
        'available_memory': avail_str,
        'available_bytes': avail_bytes,
        'memory_ratio': memory_ratio,
        'status': status
    }
    
    # Map log level string to the corresponding logging method
    log_func = {
        'debug': logger.debug,
        'info': logger.info,
        'warning': logger.warning,
        'error': logger.error,
        'critical': logger.critical
    }.get(log_level.lower(), logger.info)
    
    # Log the analysis results
    header = f"{analysis_name} - Memory Requirement Analysis"
    log_func(f"{header}:")
    log_func(f"  - Arrays to allocate: {len(shapes)}")
    log_func(f"  - Total memory required: {total_size_str}")
    log_func(f"  - Available memory: {avail_str}")
    log_func(f"  - Memory usage ratio: {memory_ratio:.2f}x")
    
    for arr in array_sizes:
        logger.debug(f"  - Array shape {arr['shape']}: {arr['size_str']}")
    
    # Log warnings based on status, but only if disk storage is not already enabled
    if not store_arrays_on_disk and status == 'critical':
        logger.warning(
            f"CRITICAL: Memory usage ({total_size_str}) exceeds {max_memory_ratio*100:.0f}% of "
            f"available memory ({avail_str}).\n"
            f"Suggestions to reduce memory usage:\n"
            f"1. Use landmark approximation with fewer landmarks\n"
            f"2. Process fewer genes at once\n"
            f"3. Use store_arrays_on_disk=True to offload arrays to disk"
            + (f"\n4. Install dask for better disk-backed storage (pip install dask)" if not DASK_AVAILABLE else "")
            + f"\n" + (f"5" if not DASK_AVAILABLE else f"4") + f". Increase system memory"
        )
    elif not store_arrays_on_disk and status == 'warning':
        logger.warning(
            f"WARNING: High memory usage detected. Arrays will use {memory_ratio:.2f}x "
            f"({total_size_str}) of available memory ({avail_str})."
            + (" Consider using store_arrays_on_disk=True" + 
               (" with dask installed" if DASK_AVAILABLE else "") + 
               ".")
        )
        
    return result


def analyze_covariance_memory_requirements(n_points: int, n_genes: int, 
                                         max_memory_ratio: float = 0.8,
                                         analysis_name: str = "Covariance Matrix Memory Analysis",
                                         store_arrays_on_disk: bool = False,
                                         log_level: str = "info") -> Dict[str, Any]:
    """
    Analyze memory requirements specifically for covariance matrices.
    
    Parameters
    ----------
    n_points : int
        Number of points (cells or landmarks)
    n_genes : int
        Number of genes
    max_memory_ratio : float, optional
        Maximum fraction of available memory that arrays should occupy, by default 0.8
    analysis_name : str, optional
        Name to identify this analysis in log messages
    store_arrays_on_disk : bool, optional
        Whether disk storage is already enabled (suppresses warnings), by default False
    log_level : str, optional
        Level to log memory information at ('debug', 'info', etc.), by default "info"
        Set to "debug" when disk storage is already enabled
        
    Returns
    -------
    dict
        Dictionary with memory analysis results (same as analyze_memory_requirements)
        plus 'should_use_disk' boolean indicating if disk storage is recommended
    """
    # Calculate shape for the covariance matrix
    covariance_shape = (n_points, n_points, n_genes)
    
    # Use debug log level when disk storage is already enabled to reduce verbosity
    if store_arrays_on_disk:
        log_level = "debug"
    
    # Get general memory analysis
    analysis = analyze_memory_requirements(
        [covariance_shape], 
        max_memory_ratio=max_memory_ratio,
        analysis_name=analysis_name,
        store_arrays_on_disk=store_arrays_on_disk,
        log_level=log_level
    )
    
    # Add recommendation about disk storage
    analysis['should_use_disk'] = analysis['status'] in ['warning', 'critical']
    
    return analysis


# For backwards compatibility only - will be removed in future versions
# and replaced completely with dask arrays
class DiskBackedCovarianceMatrix:
    """
    DEPRECATED: Use dask arrays directly for better performance.
    """
    pass


class DiskStorage:
    """
    Manage on-disk storage of large arrays.
    
    This class provides utilities to store and retrieve arrays from disk,
    helping to manage memory usage for large-scale analyses.
    
    Attributes
    ----------
    storage_dir : str
        Directory where arrays are stored
    array_registry : dict
        Registry of stored arrays with their metadata
    use_dask : bool
        Whether to use dask for lazy loading arrays
    namespace : str
        Namespace prefix for array keys to prevent collisions when sharing storage
    """
    
    # Class variable to track references to shared directories
    _shared_dirs = {}
    
    def __init__(self, storage_dir: Optional[str] = None, use_dask: bool = True, namespace: Optional[str] = None):
        """
        Initialize disk storage manager.
        
        Parameters
        ----------
        storage_dir : str, optional
            Directory to store arrays. If None, a temporary directory is created.
        use_dask : bool, optional
            Whether to use dask for lazy loading when available, by default True.
        namespace : str, optional
            Namespace prefix for array keys. If None, a random UUID is generated.
            Use this to prevent key collisions when sharing storage between objects.
        """
        # Generate a unique ID for this instance
        import uuid
        self._instance_id = str(uuid.uuid4())[:8]
        
        if storage_dir is None:
            self.storage_dir = tempfile.mkdtemp(prefix=f"kompot_arrays_{self._instance_id}_")
            self._temp_dir = True
            # Register this as the owner of the temporary directory
            DiskStorage._shared_dirs[self.storage_dir] = self._instance_id
        else:
            self.storage_dir = storage_dir
            os.makedirs(storage_dir, exist_ok=True)
            self._temp_dir = False
            # Register this instance as a user of the shared directory
            if self.storage_dir not in DiskStorage._shared_dirs:
                DiskStorage._shared_dirs[self.storage_dir] = self._instance_id
            
        # Set namespace for array keys to prevent collisions
        self.namespace = namespace if namespace is not None else f"ns_{self._instance_id}"
            
        # Determine if we should use dask
        self.use_dask = use_dask and DASK_AVAILABLE
        
        self.array_registry = {}
        if self.use_dask:
            logger.info(f"Initialized disk storage at {self.storage_dir} with dask support (namespace: {self.namespace})")
        else:
            logger.info(f"Initialized disk storage at {self.storage_dir} (namespace: {self.namespace})")
        
    def __del__(self):
        """Clean up temporary directory if it was created by this instance."""
        self.cleanup()
        
    def cleanup(self):
        """Remove temporary storage directory if it was created by this instance."""
        try:
            # Check if Python is shutting down
            import sys
            if sys.meta_path is None:
                # Python is shutting down, don't try to import or log
                return
                
            # Only perform cleanup if this instance owns the directory and it exists
            if (hasattr(self, '_temp_dir') and self._temp_dir and 
                os.path.exists(self.storage_dir) and 
                self.storage_dir in DiskStorage._shared_dirs and 
                DiskStorage._shared_dirs[self.storage_dir] == self._instance_id):
                
                import shutil
                shutil.rmtree(self.storage_dir)
                logger.info(f"Removed temporary storage directory {self.storage_dir}")
                # Remove from shared directory tracking
                DiskStorage._shared_dirs.pop(self.storage_dir, None)
            elif self.storage_dir in DiskStorage._shared_dirs:
                # Just unregister this instance from the shared directory
                logger.debug(f"Unregistered from shared directory {self.storage_dir}")
                # Only remove if this instance is the registered owner
                if DiskStorage._shared_dirs[self.storage_dir] == self._instance_id:
                    DiskStorage._shared_dirs.pop(self.storage_dir, None)
        except ImportError:
            # Python is shutting down or module not available
            pass
        except Exception as e:
            try:
                logger.warning(f"Failed to clean up disk storage: {str(e)}")
            except:
                # Even logging might fail during shutdown
                pass
                
    def store_array(self, array: Union[np.ndarray, 'da.Array'], key: str) -> str:
        """
        Store an array to disk.
        
        Parameters
        ----------
        array : np.ndarray or dask.array.Array
            Array to store
        key : str
            Identifier for the array
            
        Returns
        -------
        str
            Path to the stored array file
        """
        # Add namespace to key to prevent collisions when sharing storage
        namespaced_key = f"{self.namespace}_{key}"
        file_path = os.path.join(self.storage_dir, f"{namespaced_key}.npy")
        
        # Use file locking to prevent concurrent writes
        import filelock
        lock_path = f"{file_path}.lock"
        lock = filelock.FileLock(lock_path, timeout=60)
        
        try:
            with lock:
                # Handle dask arrays by computing them first
                if hasattr(array, 'compute') and callable(getattr(array, 'compute')):
                    # This is a dask array
                    logger.debug(f"Computing dask array before saving to disk: {namespaced_key}")
                    array = array.compute()
                
                # Save the array
                np.save(file_path, array)
                
                # Save metadata
                self.array_registry[key] = {
                    'path': file_path,
                    'shape': array.shape,
                    'dtype': str(array.dtype),
                    'size_bytes': array.nbytes,
                    'size_human': human_readable_size(array.nbytes),
                    'namespaced_key': namespaced_key
                }
                
                logger.debug(f"Stored array '{key}' to disk: {self.array_registry[key]['size_human']}")
        except filelock.Timeout:
            logger.warning(f"Timeout while trying to acquire lock for {namespaced_key}")
            raise
        finally:
            # Make sure to remove the lock file after use
            if os.path.exists(lock_path):
                try:
                    os.remove(lock_path)
                except:
                    pass
        
        return file_path
        
    def load_array(self, key: str, lazy: bool = None) -> Union[np.ndarray, 'da.Array']:
        """
        Load an array from disk, with option for lazy loading.
        
        Parameters
        ----------
        key : str
            Identifier for the array
        lazy : bool, optional
            Whether to use lazy loading with dask. If None, uses instance default.
            
        Returns
        -------
        np.ndarray or dask.array.Array
            Loaded array, either as NumPy array or Dask array
        """
        if key not in self.array_registry:
            raise KeyError(f"Array with key '{key}' not found in registry")
            
        file_path = self.array_registry[key]['path']
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Array file not found at {file_path}")
        
        # Determine whether to use lazy loading
        use_lazy = self.use_dask if lazy is None else (lazy and DASK_AVAILABLE)
        
        if use_lazy:
            import dask.array as da
            logger.debug(f"Lazy loading array '{key}' from disk: {self.array_registry[key]['size_human']}")
            # Use da.from_array instead of da.from_npy_stack for single .npy files
            return da.from_array(np.load(file_path), chunks='auto')
        else:
            logger.debug(f"Loading array '{key}' from disk: {self.array_registry[key]['size_human']}")
            return np.load(file_path)
        
    def remove_array(self, key: str):
        """
        Remove an array from disk.
        
        Parameters
        ----------
        key : str
            Identifier for the array
        """
        if key not in self.array_registry:
            logger.warning(f"Array with key '{key}' not found in registry")
            return
            
        file_path = self.array_registry[key]['path']
        
        # Use file locking to prevent concurrent access issues
        import filelock
        lock_path = f"{file_path}.lock"
        lock = filelock.FileLock(lock_path, timeout=10)
        
        try:
            with lock:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.debug(f"Removed array file {file_path}")
        except filelock.Timeout:
            logger.warning(f"Timeout while trying to acquire lock for removal of {key}")
        except Exception as e:
            logger.warning(f"Error removing array file {file_path}: {str(e)}")
        finally:
            # Clean up the lock file
            if os.path.exists(lock_path):
                try:
                    os.remove(lock_path)
                except:
                    pass
            
        del self.array_registry[key]
        
    @property
    def total_storage_used(self) -> Tuple[str, int]:
        """
        Get the total storage space used by all arrays.
        
        Returns
        -------
        tuple
            (human_readable_size, size_in_bytes)
        """
        total_bytes = sum(info['size_bytes'] for info in self.array_registry.values())
        return human_readable_size(total_bytes), total_bytes
        
    def list_arrays(self) -> Dict[str, Dict[str, Any]]:
        """
        Get a list of all stored arrays and their metadata.
        
        Returns
        -------
        dict
            Dictionary of array metadata keyed by array identifier
        """
        return self.array_registry.copy()
        
    def as_dask_array(self, shape: tuple = None, dtype = np.float64) -> 'da.Array':
        """
        Create a dask array representing all stored arrays in a single tensor.
        
        Parameters
        ----------
        shape : tuple, optional
            Shape of the result array. If None, will try to infer from stored arrays.
        dtype : numpy.dtype, optional
            Data type of the array, by default np.float64
            
        Returns
        -------
        dask.array.Array
            A dask array representing the stored data
        """
        if not self.use_dask:
            raise ImportError("Dask is not available. Install dask or set use_dask=False.")
        
        import dask.array as da
        
        # If shape is not provided, try to infer from stored arrays
        if shape is None:
            # For 3D covariance matrices, we expect (cells, cells, genes)
            # where each stored array is a 2D slice (cells, cells) for a single gene
            if all(len(info['shape']) == 2 for info in self.array_registry.values()):
                # All stored arrays are 2D, assume they're slices of a 3D tensor
                # Get shape from first array
                first_key = next(iter(self.array_registry.keys()))
                first_shape = self.array_registry[first_key]['shape']
                shape = (first_shape[0], first_shape[1], len(self.array_registry))
            else:
                raise ValueError("Cannot infer shape from stored arrays. Please provide shape explicitly.")
        
        # Create a list of dask arrays, one for each gene slice
        if len(shape) == 3:
            # We're dealing with a 3D matrix where each gene is a separate slice
            gene_arrays = []
            
            # For each gene (third dimension), create or load the appropriate slice
            for g in range(shape[2]):
                gene_key = f"gene_{g}"
                
                if gene_key in self.array_registry:
                    # Load the actual numpy array for consistency (no lazy loading here)
                    # This ensures the exact same numeric values
                    numpy_array = np.load(self.array_registry[gene_key]['path'])
                    
                    # Create a dask array from the numpy array
                    gene_array = da.from_array(numpy_array, chunks='auto')
                    
                    # Make sure it has the right shape
                    if gene_array.shape != (shape[0], shape[1]):
                        gene_array = gene_array.reshape((shape[0], shape[1]))
                else:
                    # Create zeros for missing slices
                    gene_array = da.zeros((shape[0], shape[1]), dtype=dtype)
                
                gene_arrays.append(gene_array)
                
            # Stack all gene arrays along the third dimension
            return da.stack(gene_arrays, axis=2)
        else:
            # For simpler cases, just create a zeros array
            return da.zeros(shape, dtype=dtype)