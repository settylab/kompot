"""Utility functions for Kompot package."""

import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple, List, Optional, Union, Callable, Dict, Any
from anndata import AnnData

# Import _sanitize_name from anndata.functions
try:
    from .anndata.functions import _sanitize_name
except (ImportError, AttributeError):
    # Define locally if import fails
    def _sanitize_name(name):
        """Convert a string to a valid column/key name by replacing invalid characters."""
        # Replace spaces, slashes, and other common problematic characters
        return str(name).replace(' ', '_').replace('/', '_').replace('\\', '_').replace('-', '_').replace('.', '_')
from scipy.linalg import solve_triangular

import pynndescent
import igraph as ig
import leidenalg as la
import logging

logger = logging.getLogger("kompot")

# Define standard colors for consistent use throughout the package
KOMPOT_COLORS = {
    # Direction colors for differential abundance
    "direction": {
        "up": "#d73027",     # red
        "down": "#4575b4",   # blue
        "neutral": "#d3d3d3" # light gray
    },
    # Additional color palettes can be added here
}


def get_run_from_history(
    adata: AnnData, 
    run_id: Optional[int] = None, 
    history_key: str = 'kompot_run_history',
    analysis_type: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Get run information from run history based on run_id.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object containing run history
    run_id : int, optional
        Run ID to retrieve. Negative indices count from the end.
        If None, returns None.
    history_key : str, optional
        Key in adata.uns where the run history is stored.
        Default is 'kompot_run_history' for the global history.
        For analysis-specific history, use either:
        - 'kompot_da.run_history' for differential abundance runs
        - 'kompot_de.run_history' for differential expression runs
        - Or set analysis_type instead for automatic lookup
        This is only used if analysis_type is None.
    analysis_type : str, optional
        Type of analysis to look up: "da", "de", or None.
        If provided, only looks in the specific analysis type's history
        and ignores history_key.
        
    Returns
    -------
    dict or None
        The run information dict if found, or None if not found or run_id is None
        
    Notes
    -----
    The run history is always stored in fixed locations:
    - adata.uns['kompot_da'] for differential abundance runs
    - adata.uns['kompot_de'] for differential expression runs
    - adata.uns['kompot_run_history'] for combined runs
    """
    if run_id is None:
        return None
    
    # Use specific analysis history if provided
    if analysis_type is not None:
        if analysis_type == "da":
            history_key = "kompot_da.run_history"
        elif analysis_type == "de":
            history_key = "kompot_de.run_history"
        elif analysis_type == "combined":
            history_key = "kompot_run_history"
        else:
            logger.warning(f"Unknown analysis_type: {analysis_type}. Using provided history_key: {history_key}")
    
    # Handle case where history_key is specified as 'storage_key.run_history'
    if '.' in history_key:
        parts = history_key.split('.')
        storage_key = parts[0]
        subkey = parts[1]
        if storage_key in adata.uns and subkey in adata.uns[storage_key]:
            history = adata.uns[storage_key][subkey]
        else:
            logger.warning(f"Run history at {storage_key}.{subkey} not found.")
            return None
    
    # Direct access to specified history key
    elif history_key in adata.uns:
        history = adata.uns[history_key]
    
    # Not found
    else:
        logger.warning(f"No run history found at {history_key}.")
        return None
    
    # If history is empty
    if len(history) == 0:
        logger.warning(f"Run history at {history_key} is empty.")
        return None
        
    # Handle negative indices (e.g., -1 for latest run)
    if run_id < 0 and len(history) >= abs(run_id):
        adjusted_run_id = len(history) + run_id
    else:
        adjusted_run_id = run_id
    
    # Find the requested run
    if 0 <= adjusted_run_id < len(history):
        return history[adjusted_run_id]
    else:
        logger.warning(f"Run ID {run_id} not found in {history_key}. Using default or latest run.")
        return None


def get_environment_info() -> Dict[str, str]:
    """
    Get information about the current execution environment.
    
    Returns
    -------
    Dict[str, str]
        Dictionary with environment information
    """
    from datetime import datetime
    import platform
    import getpass
    import socket
    import os
    
    try:
        hostname = socket.gethostname()
    except:
        hostname = "unknown"
        
    try:
        username = getpass.getuser()
    except:
        username = "unknown"
        
    env_info = {
        "timestamp": datetime.now().isoformat(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "hostname": hostname,
        "username": username,
        "pid": os.getpid()
    }
    
    # Try to get package version if available
    try:
        from kompot.version import __version__
        env_info["kompot_version"] = __version__
    except ImportError:
        try:
            # Alternative way to get version
            import pkg_resources
            env_info["kompot_version"] = pkg_resources.get_distribution("kompot").version
        except:
            env_info["kompot_version"] = "unknown"
        
    return env_info


def generate_output_field_names(
    result_key: str,
    condition1: str,
    condition2: str,
    analysis_type: str = "da",
    with_sample_suffix: bool = False,
    sample_suffix: str = "_sample_var"
) -> Dict[str, str]:
    """
    Generate standardized field names for analysis outputs.
    
    Parameters
    ----------
    result_key : str
        Base key for results (e.g., "kompot_da", "kompot_de")
    condition1 : str
        Name of the first condition
    condition2 : str
        Name of the second condition
    analysis_type : str, optional
        Type of analysis: "da" for differential abundance or "de" for differential expression
        By default "da"
    with_sample_suffix : bool, optional
        Whether to include sample variance suffix in field names, by default False
    sample_suffix : str, optional
        Suffix to add for sample variance variants, by default "_sample_var"
        
    Returns
    -------
    Dict[str, str]
        Dictionary mapping field types to their standardized names
    """
    # Sanitize condition names
    cond1_safe = _sanitize_name(condition1)
    cond2_safe = _sanitize_name(condition2)
    
    # Apply suffix when sample variance is used
    suffix = sample_suffix if with_sample_suffix else ""
    
    # Basic fields for both analysis types
    field_names = {}
    
    if analysis_type == "da":
        # Differential abundance field names
        field_names.update({
            "lfc_key": f"{result_key}_log_fold_change_{cond1_safe}_vs_{cond2_safe}{suffix}",
            "zscore_key": f"{result_key}_log_fold_change_zscore_{cond1_safe}_vs_{cond2_safe}{suffix}",
            "pval_key": f"{result_key}_neg_log10_fold_change_pvalue_{cond1_safe}_vs_{cond2_safe}{suffix}",
            "direction_key": f"{result_key}_log_fold_change_direction_{cond1_safe}_vs_{cond2_safe}{suffix}",
            "density_key_1": f"{result_key}_log_density_{cond1_safe}{suffix}",
            "density_key_2": f"{result_key}_log_density_{cond2_safe}{suffix}"
        })
    elif analysis_type == "de":
        # Differential expression field names
        field_names.update({
            "mahalanobis_key": f"{result_key}_mahalanobis_{cond1_safe}_vs_{cond2_safe}{suffix}",
            "mean_lfc_key": f"{result_key}_mean_lfc_{cond1_safe}_vs_{cond2_safe}{suffix}",
            "weighted_lfc_key": f"{result_key}_weighted_lfc_{cond1_safe}_vs_{cond2_safe}{suffix}",
            "lfc_std_key": f"{result_key}_lfc_std_{cond1_safe}_vs_{cond2_safe}{suffix}",
            "bidirectionality_key": f"{result_key}_bidirectionality_{cond1_safe}_vs_{cond2_safe}{suffix}",
            "imputed_key_1": f"{result_key}_imputed_{cond1_safe}{suffix}",
            "imputed_key_2": f"{result_key}_imputed_{cond2_safe}{suffix}",
            "fold_change_key": f"{result_key}_fold_change_{cond1_safe}_vs_{cond2_safe}{suffix}"
        })
    else:
        raise ValueError(f"Unknown analysis_type: {analysis_type}. Use 'da' or 'de'.")
    
    return field_names


def detect_output_field_overwrite(
    adata: AnnData, 
    result_key: str, 
    output_patterns: List[str],
    location: str = "obs",
    result_type: str = "results",
    with_sample_suffix: bool = False,
    sample_suffix: str = "_sample_var",
    analysis_type: str = "da"
) -> Tuple[bool, List[str], Optional[Dict[str, Any]]]:
    """
    Detects if we would overwrite existing output fields in an AnnData object.
    This function scans AnnData object for output fields that match the given patterns
    and looks through run history to find previous runs that might have created them.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object to check for existing fields
    result_key : str
        Key under which results are stored (used for field generation, not storage location)
    output_patterns : List[str]
        Patterns of output field names to check for (e.g., ["lfc_key", "pval_key"])
    location : str, optional
        Location to check for field patterns (e.g., "obs", "var", "layers"), by default "obs"
    result_type : str, optional
        Description of the results for warning/error messages, by default "results"
    with_sample_suffix : bool, optional
        Whether to also check for patterns with sample suffix, by default False
    sample_suffix : str, optional
        Suffix to add when checking for sample variance variants, by default "_sample_var"
    analysis_type : str, optional
        Type of analysis ("da" or "de"), determines where run history is stored, by default "da"
        
    Returns
    -------
    Tuple[bool, List[str], Optional[Dict[str, Any]]]
        - Boolean indicating if any fields would be overwritten
        - List of field names that would be overwritten
        - Previous run info if found in run history, otherwise None
        
    Notes
    -----
    Run history is stored in adata.uns["kompot_da"] or adata.uns["kompot_de"] based on analysis_type.
    This function will look in those fixed locations rather than using result_key for storage location.
    """
    existing_fields = []
    
    # Get the object to check for patterns based on location
    if location == "obs":
        obj_to_check = adata.obs
    elif location == "var":
        obj_to_check = adata.var
    elif location == "layers":
        obj_to_check = adata.layers
    else:
        raise ValueError(f"Unknown location: {location}. Use 'obs', 'var', or 'layers'")
    
    # Check for patterns in the specified location
    if hasattr(obj_to_check, 'columns'):  # DataFrame-like (obs or var)
        for pattern in output_patterns:
            for column in obj_to_check.columns:
                if column.startswith(pattern):
                    existing_fields.append(f"{location}:{column}")
                    logger.debug(f"Found existing field to be overwritten: {location}:{column}")
                    break
                    
            # Also check with sample suffix if requested
            if with_sample_suffix:
                for column in obj_to_check.columns:
                    if column.startswith(pattern + sample_suffix):
                        existing_fields.append(f"{location}:{column}")
                        logger.debug(f"Found existing field with sample suffix to be overwritten: {location}:{column}")
                        break
    
    else:  # dict-like (layers)
        for pattern in output_patterns:
            for key in obj_to_check.keys():
                if key.startswith(pattern):
                    existing_fields.append(f"{location}:{key}")
                    logger.debug(f"Found existing field to be overwritten: {location}:{key}")
                    break
                    
            # Also check with sample suffix if requested
            if with_sample_suffix:
                for key in obj_to_check.keys():
                    if key.startswith(pattern + sample_suffix):
                        existing_fields.append(f"{location}:{key}")
                        logger.debug(f"Found existing field with sample suffix to be overwritten: {location}:{key}")
                        break
    
    # Infer analysis_type from result_key if not provided
    if analysis_type is None:
        if "da" in result_key:
            analysis_type = "da"
        elif "de" in result_key:
            analysis_type = "de"
    
    # Look for matching run in run history
    previous_run = None
    
    # Look in the fixed location determined by analysis_type
    # This is simpler now - we just need to get the latest run from the appropriate fixed location
    previous_run = get_run_from_history(adata, run_id=-1, analysis_type=analysis_type)
    
    # If no previous run in fixed location, try the global history
    if previous_run is None and 'kompot_run_history' in adata.uns:
        # Look for most recent run with matching analysis_type in global history
        matching_runs = []
        for i, run in enumerate(adata.uns['kompot_run_history']):
            if run.get('analysis_type') == analysis_type:
                matching_runs.append((i, run))
        
        if matching_runs:
            # Get the most recent matching run
            previous_run = matching_runs[-1][1]
    
    # Return a tuple with detection results
    return (len(existing_fields) > 0, existing_fields, previous_run)


def build_graph(X: np.ndarray, n_neighbors: int = 15) -> Tuple[List[Tuple[int, int]], pynndescent.NNDescent]:
    """
    Build a graph from a dataset using approximate nearest neighbors.
    
    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n_samples, n_features).
    n_neighbors : int, optional
        Number of neighbors for graph construction, by default 15.
        
    Returns
    -------
    Tuple[List[Tuple[int, int]], pynndescent.NNDescent]
        A tuple containing:
        - edges: List of (source, target) tuples defining the graph
        - index: The nearest neighbor index for future queries
    """
    # Build the nearest neighbor index
    index = pynndescent.NNDescent(X, n_neighbors=n_neighbors, random_state=42)
    
    # Query for nearest neighbors
    indices, _ = index.query(X, k=n_neighbors)
    
    # Convert to edges
    n_obs = X.shape[0]
    edges = []
    for i in range(n_obs):
        for j in indices[i]:
            if i != j:  # Avoid self-loops
                edges.append((i, j))
    
    return edges, index


def prepare_mahalanobis_matrix(
    covariance_matrix: np.ndarray,
    diag_adjustments: Optional[np.ndarray] = None,
    eps: float = 1e-12,
    jit_compile: bool = True,
):
    """
    Prepare a covariance matrix for Mahalanobis distance computation.
    
    This function processes the covariance matrix, adds diagonal adjustments if needed,
    and computes the Cholesky decomposition for efficient distance calculations.
    
    Parameters
    ----------
    covariance_matrix : np.ndarray
        The covariance matrix.
    diag_adjustments : np.ndarray, optional
        Additional diagonal adjustments for the covariance matrix, by default None.
    eps : float, optional
        Small constant for numerical stability, by default 1e-12.
    jit_compile : bool, optional
        Whether to use JAX just-in-time compilation, by default True.
        
    Returns
    -------
    dict
        A dictionary containing:
        - 'chol': The Cholesky decomposition of the adjusted covariance matrix if successful
        - 'matrix_inv': The inverse of the adjusted covariance matrix (fallback)
        - 'is_diagonal': Whether using a diagonal approximation
        - 'diag_values': Diagonal values for fast computation if is_diagonal is True
    """
    # Convert to JAX arrays
    cov = jnp.array(covariance_matrix)
    
    # Check if matrix is diagonal or near-diagonal (which would allow much faster calculation)
    is_diagonal = False
    diag_values = None
    
    if covariance_matrix.shape[0] > 10:
        diag_sum = np.sum(np.diag(covariance_matrix))
        total_sum = np.sum(np.abs(covariance_matrix))
        diag_ratio = diag_sum / total_sum if total_sum > 0 else 0
        
        # If matrix is nearly diagonal, use faster diagonal approximation
        if diag_ratio > 0.95:
            logger.info("Using fast diagonal matrix approximation")
            is_diagonal = True
            diag_values = jnp.diag(cov)
            
            # Add diagonal adjustments if provided
            if diag_adjustments is not None:
                diag_adj = jnp.array(diag_adjustments)
                diag_values = diag_values + diag_adj
                
            # Ensure numerical stability
            diag_values = jnp.clip(diag_values, eps, None)
            
            return {
                'is_diagonal': is_diagonal,
                'diag_values': diag_values,
                'chol': None,
                'matrix_inv': None
            }
    
    # For non-diagonal matrices, prepare for Cholesky
    # Adjust covariance matrix if needed
    if diag_adjustments is not None:
        diag_adj = jnp.array(diag_adjustments)
        cov = cov + jnp.diag(diag_adj)
    
    # Ensure numerical stability by clipping diagonal elements
    diag_clipped = jnp.clip(jnp.diag(cov), eps, None)
    cov = cov.at[jnp.arange(cov.shape[0]), jnp.arange(cov.shape[0])].set(diag_clipped)
    
    try:
        # Compute Cholesky decomposition once
        chol = jnp.linalg.cholesky(cov)
        
        return {
            'is_diagonal': False,
            'diag_values': None,
            'chol': chol,
            'matrix_inv': None
        }
    except Exception as e:
        # Fallback to matrix inverse if Cholesky fails
        logger.warning(
            f"Cholesky decomposition failed: {e}. Using pseudoinverse. "
            f"This might be slow. Consider raising eps ({eps}) to enable Cholesky decomposition instead."
        )
        try:
            matrix_inv = jnp.linalg.pinv(cov)
            return {
                'is_diagonal': False,
                'diag_values': None,
                'chol': None,
                'matrix_inv': matrix_inv
            }
        except Exception as e2:
            # Last resort - diagonal approximation
            logger.error(f"Matrix inverse also failed: {e2}. Using diagonal approximation.")
            diag_values = jnp.clip(jnp.diag(cov), eps, None)
            return {
                'is_diagonal': True,
                'diag_values': diag_values,
                'chol': None,
                'matrix_inv': None
            }

def compute_mahalanobis_distances(
    diff_values: np.ndarray,
    prepared_matrix: dict,
    batch_size: int = 500,
    jit_compile: bool = True,
    gene_covariances: Optional[np.ndarray] = None,
    progress: bool = True,
) -> np.ndarray:
    """
    Compute Mahalanobis distances for multiple difference vectors efficiently.
    
    This function takes preprocessed matrix information and computes the Mahalanobis
    distance for each provided difference vector. It supports batched computation for
    memory efficiency.
    
    Parameters
    ----------
    diff_values : np.ndarray
        The difference vectors for which to compute Mahalanobis distances.
        Shape should be (n_samples, n_features) or (n_features, n_samples).
    prepared_matrix : dict
        A dictionary from prepare_mahalanobis_matrix containing matrix information.
    batch_size : int, optional
        Number of vectors to process at once, by default 500.
    jit_compile : bool, optional
        Whether to use JAX just-in-time compilation, by default True.
    gene_covariances : np.ndarray, optional
        Gene-specific covariance matrices with shape (n_points, n_points, n_genes).
        If provided, will use gene-specific covariance for each gene.
    progress : bool, optional
        Whether to show a progress bar for gene-specific calculations, by default True.
        
    Returns
    -------
    np.ndarray
        Array of Mahalanobis distances for each input vector.
    """
    from .batch_utils import apply_batched
    from tqdm.auto import tqdm
    
    # Convert input to JAX array
    diffs = jnp.array(diff_values)
    
    # Handle different input shapes - we want (n_genes, n_points) for gene-wise processing
    if len(diffs.shape) == 1:
        # Single vector, reshape to (1, n_features)
        diffs = diffs.reshape(1, -1)
    
    # Special case: gene-specific covariance matrices
    if gene_covariances is not None:
        logger.info(f"Computing Mahalanobis distances using gene-specific covariance matrices")
        n_genes = diffs.shape[0]
        mahalanobis_distances = np.zeros(n_genes)
        
        # Get the combined covariance from prepared_matrix for broadcasting
        combined_cov = None
        if prepared_matrix['chol'] is not None:
            combined_cov = jnp.dot(prepared_matrix['chol'], prepared_matrix['chol'].T)
        elif prepared_matrix['matrix_inv'] is not None:
            combined_cov = jnp.linalg.pinv(prepared_matrix['matrix_inv'])
        
        # Process each gene separately to save memory, with progress bar
        gene_iterator = tqdm(range(n_genes), desc="Computing gene-specific Mahalanobis distances") if progress else range(n_genes)
        for g in gene_iterator:
            # Get the gene-specific difference vector and covariance
            gene_diff = diffs[g]
            gene_cov = gene_covariances[:, :, g]
            
            # Add the combined_cov to gene_cov if it exists
            if combined_cov is not None:
                gene_cov = gene_cov + combined_cov
            
            # We need to handle the Cholesky decomposition for each gene separately
            # to ensure numerical stability and proper memory management
            
            # Prepare covariance matrix for this gene individually (compute Cholesky once per gene)
            gene_prepared_matrix = prepare_mahalanobis_matrix(
                covariance_matrix=gene_cov,
                eps=1e-10,  # Use slightly larger eps for numerical stability
                jit_compile=jit_compile
            )
            
            # Compute Mahalanobis distance for this gene
            if gene_prepared_matrix['is_diagonal']:
                # Diagonal case
                diag_values = gene_prepared_matrix['diag_values']
                weighted_diff = gene_diff / jnp.sqrt(diag_values)
                mahalanobis_distances[g] = float(jnp.sqrt(jnp.sum(weighted_diff**2)))
            
            elif gene_prepared_matrix['chol'] is not None:
                # Cholesky case - most efficient
                chol = gene_prepared_matrix['chol']
                solved = jax.scipy.linalg.solve_triangular(chol, gene_diff, lower=True)
                mahalanobis_distances[g] = float(jnp.sqrt(jnp.sum(solved**2)))
            
            elif gene_prepared_matrix['matrix_inv'] is not None:
                # Matrix inverse case
                matrix_inv = gene_prepared_matrix['matrix_inv']
                mahalanobis_distances[g] = float(jnp.sqrt(jnp.dot(gene_diff, jnp.dot(matrix_inv, gene_diff))))
            
            else:
                # Fallback to Euclidean distance
                mahalanobis_distances[g] = float(jnp.sqrt(jnp.sum(gene_diff**2)))
            
            # Free memory
            del gene_prepared_matrix
            
        return mahalanobis_distances
    
    # Regular case with shared covariance matrix
    # Check if we're using diagonal approximation
    if prepared_matrix['is_diagonal']:
        # Diagonal case - much faster computation
        diag_values = prepared_matrix['diag_values']
        
        # Define computation for diagonal case
        def compute_diagonal_batch(batch_diffs):
            # For diagonal matrix, Mahalanobis is just a weighted Euclidean distance
            weighted_diffs = batch_diffs / jnp.sqrt(diag_values)
            return jnp.sqrt(jnp.sum(weighted_diffs**2, axis=1))
        
        # JIT compile if enabled
        if jit_compile:
            diag_compute_fn = jax.jit(compute_diagonal_batch)
        else:
            diag_compute_fn = compute_diagonal_batch
        
        # Process in batches using apply_batched - respect progress parameter
        desc = "Computing diagonal Mahalanobis distances" if progress else None
        return apply_batched(
            diag_compute_fn,
            diffs,
            batch_size=batch_size,
            desc=desc
        )
    
    # Non-diagonal case - use Cholesky if available
    if prepared_matrix['chol'] is not None:
        chol = prepared_matrix['chol']
        
        # Define computation function using Cholesky
        def compute_cholesky_batch(batch_diffs):
            # Triangular solve for each diff vector
            solved = jax.vmap(lambda d: jax.scipy.linalg.solve_triangular(chol, d, lower=True))(batch_diffs)
            
            # Compute squared distances
            squared_distances = jnp.sum(solved**2, axis=1)
            return jnp.sqrt(squared_distances)
        
        # JIT compile if enabled
        if jit_compile:
            chol_compute_fn = jax.jit(compute_cholesky_batch)
        else:
            chol_compute_fn = compute_cholesky_batch
        
        # Process in batches using apply_batched - respect progress parameter
        desc = "Computing Cholesky Mahalanobis distances" if progress else None
        return apply_batched(
            chol_compute_fn,
            diffs,
            batch_size=batch_size,
            desc=desc
        )
    
    # Fallback to matrix inverse
    if prepared_matrix['matrix_inv'] is not None:
        matrix_inv = prepared_matrix['matrix_inv']
        
        # Define computation function using inverse matrix
        def compute_inverse_batch(batch_diffs):
            # Apply matrix multiplication for each vector: d' * inv(cov) * d
            products = jax.vmap(lambda d: jnp.dot(d, jnp.dot(matrix_inv, d)))(batch_diffs)
            return jnp.sqrt(products)
        
        # JIT compile if enabled
        if jit_compile:
            inv_compute_fn = jax.jit(compute_inverse_batch)
        else:
            inv_compute_fn = compute_inverse_batch
        
        # Process in batches using apply_batched - respect progress parameter
        desc = "Computing inverse Mahalanobis distances" if progress else None
        return apply_batched(
            inv_compute_fn,
            diffs,
            batch_size=batch_size,
            desc=desc
        )
    
    # If all else fails, use a simple approximation
    logger.error("No valid matrix information found. Using simple Euclidean distance.")
    return np.sqrt(np.sum(diff_values**2, axis=1))

def compute_mahalanobis_distance(
    diff_values: np.ndarray,
    covariance_matrix: np.ndarray,
    diag_adjustments: Optional[np.ndarray] = None,
    eps: float = 1e-12,
    jit_compile: bool = True,
) -> float:
    """
    Compute the Mahalanobis distance for a vector given a covariance matrix.
    
    This is a convenience function for computing a single Mahalanobis distance.
    For multiple vectors, use prepare_mahalanobis_matrix and compute_mahalanobis_distances
    for better performance.
    
    Parameters
    ----------
    diff_values : np.ndarray
        The difference vector for which to compute the Mahalanobis distance.
    covariance_matrix : np.ndarray
        The covariance matrix.
    diag_adjustments : np.ndarray, optional
        Additional diagonal adjustments for the covariance matrix, by default None.
    eps : float, optional
        Small constant for numerical stability, by default 1e-12.
    jit_compile : bool, optional
        Whether to use JAX just-in-time compilation, by default True.
        
    Returns
    -------
    float
        The Mahalanobis distance.
    """
    # Prepare the matrix
    prepared_matrix = prepare_mahalanobis_matrix(
        covariance_matrix=covariance_matrix,
        diag_adjustments=diag_adjustments,
        eps=eps,
        jit_compile=jit_compile
    )
    
    # Compute distance for single vector
    distances = compute_mahalanobis_distances(
        diff_values=diff_values,
        prepared_matrix=prepared_matrix,
        batch_size=1,
        jit_compile=jit_compile
    )
    
    # Return the single distance
    if len(distances) > 1:
        return float(distances[0])
    else:
        return float(distances.item())


def find_optimal_resolution(
    edges: List[Tuple[int, int]],
    n_obs: int,
    n_clusters: int,
    tol: float = 0.1,
    max_iter: int = 10
) -> Tuple[float, any]:
    """
    Find an optimal resolution for Leiden clustering to achieve a target number of clusters.
    
    Parameters
    ----------
    edges : List[Tuple[int, int]]
        List of edges defining the graph.
    n_obs : int
        Number of observations (nodes) in the graph.
    n_clusters : int
        Desired number of clusters.
    tol : float, optional
        Tolerance for the deviation from the target number of clusters, by default 0.1.
    max_iter : int, optional
        Maximum number of iterations for the search, by default 10.
        
    Returns
    -------
    Tuple[float, any]
        A tuple containing:
        - optimal_resolution: The resolution value that best approximates the desired number of clusters
        - best_partition: The clustering partition at the optimal resolution
    """
    # Create igraph object
    G_igraph = ig.Graph(edges=edges, directed=False)
    G_igraph.vs["name"] = [str(i) for i in range(n_obs)]

    # Initial heuristic
    initial_resolution = n_obs / n_clusters
    resolution = initial_resolution
    lower, upper = 0.01, 1000.0

    best_partition = None
    
    for iteration in range(max_iter):
        partition = la.find_partition(
            G_igraph,
            la.RBConfigurationVertexPartition,
            n_iterations=-1,
            resolution_parameter=resolution,
        )
        current_clusters = len(set(partition.membership))
        percent_diff = (current_clusters - n_clusters) / n_clusters

        if abs(percent_diff) <= tol:
            logger.info(
                f"Converged at iteration {iteration + 1}: resolution={resolution}, clusters={current_clusters}"
            )
            return resolution, partition

        logger.info(
            f"Iteration {iteration + 1}: resolution={resolution}, clusters={current_clusters}"
        )

        # Adjust resolution logarithmically
        if current_clusters < n_clusters:
            lower = resolution
        else:
            upper = resolution

        resolution = np.sqrt(lower * upper)
        best_partition = partition

    logger.warning(
        f"Did not fully converge within {max_iter} iterations. Using resolution={resolution}."
    )
    return resolution, best_partition


def find_landmarks(
    X: np.ndarray,
    n_clusters: int = 200,
    n_neighbors: int = 15,
    tol: float = 0.1,
    max_iter: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identify landmark points representing clusters in the dataset.
    
    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n_samples, n_features).
    n_clusters : int, optional
        Desired number of clusters/landmarks, by default 200.
    n_neighbors : int, optional
        Number of neighbors for graph construction, by default 15.
    tol : float, optional
        Tolerance for the deviation from the target number of clusters, by default 0.1.
    max_iter : int, optional
        Maximum number of iterations for resolution search, by default 10.
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - landmarks: Matrix of shape (n_clusters, n_features) containing landmark coordinates
        - landmark_indices: Indices of landmarks in the original dataset
    """
    # Build graph
    edges, index = build_graph(X, n_neighbors=n_neighbors)
    n_obs = X.shape[0]

    # Find optimal resolution and clustering
    optimal_resolution, partition = find_optimal_resolution(
        edges, n_obs, n_clusters, tol=tol, max_iter=max_iter
    )
    clusters = np.array(partition.membership)
    cluster_ids = np.unique(clusters)

    # Compute centroids
    centroids = np.array([X[clusters == c].mean(axis=0) for c in cluster_ids])

    # Find the nearest data point to each centroid
    landmark_indices, _ = index.query(centroids, k=1)
    landmark_indices = landmark_indices.flatten()
    landmarks = X[landmark_indices]

    logger.info(
        f"Found {len(cluster_ids)} clusters at resolution={optimal_resolution}, creating landmarks..."
    )
    
    return landmarks, landmark_indices