"""Utilities for memory management and estimation."""

import numpy as np
import psutil
import os
import tempfile
import logging
from typing import Tuple, Union, Optional, Dict, Any, List

logger = logging.getLogger("kompot")


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
                               analysis_name: str = "Memory Analysis") -> Dict[str, Any]:
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
    
    # Log the analysis results
    header = f"{analysis_name} - Memory Requirement Analysis"
    logger.info(f"{header}:")
    logger.info(f"  - Arrays to allocate: {len(shapes)}")
    logger.info(f"  - Total memory required: {total_size_str}")
    logger.info(f"  - Available memory: {avail_str}")
    logger.info(f"  - Memory usage ratio: {memory_ratio:.2f}x")
    
    for arr in array_sizes:
        logger.debug(f"  - Array shape {arr['shape']}: {arr['size_str']}")
    
    # Log warnings based on status
    if status == 'critical':
        logger.warning(
            f"CRITICAL: Memory usage ({total_size_str}) exceeds {max_memory_ratio*100:.0f}% of "
            f"available memory ({avail_str}).\n"
            f"Suggestions to reduce memory usage:\n"
            f"1. Use landmark approximation with fewer landmarks\n"
            f"2. Process fewer genes at once\n"
            f"3. Use store_arrays_on_disk=True to offload arrays to disk\n"
            f"4. Increase system memory"
        )
    elif status == 'warning':
        logger.warning(
            f"WARNING: High memory usage detected. Arrays will use {memory_ratio:.2f}x "
            f"({total_size_str}) of available memory ({avail_str})."
        )
        
    return result


def analyze_covariance_memory_requirements(n_points: int, n_genes: int, 
                                         max_memory_ratio: float = 0.8,
                                         analysis_name: str = "Covariance Matrix Memory Analysis") -> Dict[str, Any]:
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
        
    Returns
    -------
    dict
        Dictionary with memory analysis results (same as analyze_memory_requirements)
        plus 'should_use_disk' boolean indicating if disk storage is recommended
    """
    # Calculate shape for the covariance matrix
    covariance_shape = (n_points, n_points, n_genes)
    
    # Get general memory analysis
    analysis = analyze_memory_requirements(
        [covariance_shape], 
        max_memory_ratio=max_memory_ratio,
        analysis_name=analysis_name
    )
    
    # Add recommendation about disk storage
    analysis['should_use_disk'] = analysis['status'] in ['warning', 'critical']
    
    return analysis


class DiskBackedCovarianceMatrix:
    """
    Efficient disk-backed covariance matrix that loads slices on demand.
    
    This class provides an interface similar to a numpy array, but stores
    gene slices of the covariance matrix on disk, loading only what is needed.
    
    Attributes
    ----------
    disk_storage : DiskStorage
        Storage manager for slices
    shape : tuple
        Shape of the full matrix (cells, cells, genes)
    """
    
    def __init__(self, disk_storage, shape, gene_keys=None):
        """
        Initialize disk-backed covariance matrix.
        
        Parameters
        ----------
        disk_storage : DiskStorage
            Storage manager for the array slices
        shape : tuple
            Shape of the full covariance matrix (cells, cells, genes)
        gene_keys : dict, optional
            Dictionary mapping gene indices to storage keys
        """
        self.disk_storage = disk_storage
        self.shape = shape
        self.gene_keys = gene_keys or {}
        
    def __getitem__(self, key):
        """
        Get a slice of the covariance matrix.
        
        Parameters
        ----------
        key : tuple or int
            Index or slice into the matrix
            
        Returns
        -------
        np.ndarray
            The requested slice of data
        """
        # Handle different slicing patterns
        if isinstance(key, tuple) and len(key) == 3:
            # Full 3D slicing (cells, cells, genes)
            cell1_slice, cell2_slice, gene_slice = key
            
            # If requesting a single gene, just load that slice
            if isinstance(gene_slice, int):
                gene_key = self.gene_keys.get(gene_slice)
                if gene_key:
                    gene_cov = self.disk_storage.load_array(gene_key)
                    return gene_cov[cell1_slice, cell2_slice]
                else:
                    raise KeyError(f"Gene index {gene_slice} not found in storage")
            else:
                # Need to load multiple gene slices - do it one by one
                if isinstance(gene_slice, slice):
                    gene_indices = range(*gene_slice.indices(self.shape[2]))
                else:
                    gene_indices = gene_slice
                    
                result = []
                for g in gene_indices:
                    gene_key = self.gene_keys.get(g)
                    if gene_key:
                        gene_cov = self.disk_storage.load_array(gene_key)
                        result.append(gene_cov[cell1_slice, cell2_slice])
                    else:
                        raise KeyError(f"Gene index {g} not found in storage")
                        
                # Stack the gene slices along the last axis
                return np.stack(result, axis=-1)
        elif isinstance(key, int):
            # Single gene slice
            gene_key = self.gene_keys.get(key)
            if gene_key:
                return self.disk_storage.load_array(gene_key)
            else:
                raise KeyError(f"Gene index {key} not found in storage")
                
        # Other slicing patterns
        raise NotImplementedError(f"Slicing pattern {key} not supported")
        
    def __array__(self):
        """
        Get the full array (should be avoided as it loads everything into memory).
        
        Returns
        -------
        np.ndarray
            Full covariance matrix
        """
        logger.warning(
            f"Converting DiskBackedCovarianceMatrix to full numpy array. "
            f"This will load the entire matrix ({human_readable_size(np.prod(self.shape) * 8)}) into memory."
        )
        
        result = np.zeros(self.shape)
        for g in range(self.shape[2]):
            gene_key = self.gene_keys.get(g)
            if gene_key:
                result[:, :, g] = self.disk_storage.load_array(gene_key)
                
        return result


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
    """
    
    def __init__(self, storage_dir: Optional[str] = None):
        """
        Initialize disk storage manager.
        
        Parameters
        ----------
        storage_dir : str, optional
            Directory to store arrays. If None, a temporary directory is created.
        """
        if storage_dir is None:
            self.storage_dir = tempfile.mkdtemp(prefix="kompot_arrays_")
            self._temp_dir = True
        else:
            self.storage_dir = storage_dir
            os.makedirs(storage_dir, exist_ok=True)
            self._temp_dir = False
            
        self.array_registry = {}
        logger.info(f"Initialized disk storage at {self.storage_dir}")
        
    def __del__(self):
        """Clean up temporary directory if it was created by this instance."""
        self.cleanup()
        
    def cleanup(self):
        """Remove temporary storage directory if it was created by this instance."""
        if hasattr(self, '_temp_dir') and self._temp_dir and os.path.exists(self.storage_dir):
            import shutil
            try:
                shutil.rmtree(self.storage_dir)
                logger.info(f"Removed temporary storage directory {self.storage_dir}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary directory {self.storage_dir}: {str(e)}")
                
    def store_array(self, array: np.ndarray, key: str) -> str:
        """
        Store an array to disk.
        
        Parameters
        ----------
        array : np.ndarray
            Array to store
        key : str
            Identifier for the array
            
        Returns
        -------
        str
            Path to the stored array file
        """
        file_path = os.path.join(self.storage_dir, f"{key}.npy")
        np.save(file_path, array)
        
        # Save metadata
        self.array_registry[key] = {
            'path': file_path,
            'shape': array.shape,
            'dtype': str(array.dtype),
            'size_bytes': array.nbytes,
            'size_human': human_readable_size(array.nbytes)
        }
        
        logger.info(f"Stored array '{key}' to disk: {self.array_registry[key]['size_human']}")
        return file_path
        
    def load_array(self, key: str) -> np.ndarray:
        """
        Load an array from disk.
        
        Parameters
        ----------
        key : str
            Identifier for the array
            
        Returns
        -------
        np.ndarray
            Loaded array
        """
        if key not in self.array_registry:
            raise KeyError(f"Array with key '{key}' not found in registry")
            
        file_path = self.array_registry[key]['path']
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Array file not found at {file_path}")
            
        logger.info(f"Loading array '{key}' from disk: {self.array_registry[key]['size_human']}")
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
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Removed array file {file_path}")
            
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