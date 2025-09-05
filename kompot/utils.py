"""Utility functions for Kompot package."""

import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple, List, Optional, Union, Dict, Any
from anndata import AnnData

# Import _sanitize_name from anndata.utils
try:
    from .anndata.utils import _sanitize_name, generate_output_field_names
except (ImportError, AttributeError):
    # Define locally if import fails
    def _sanitize_name(name):
        """Convert a string to a valid column/key name by replacing invalid characters."""
        # Replace spaces, slashes, and other common problematic characters
        return str(name).replace(' ', '_').replace('/', '_').replace('\\', '_').replace('-', '_').replace('.', '_')
from scipy.linalg import solve_triangular

import pynndescent
import igraph as ig
import logging

from .memory_utils import DASK_AVAILABLE
if DASK_AVAILABLE:
    try:
        import dask.array as da
        import dask
    except ImportError:
        pass


logger = logging.getLogger("kompot")

# Define standard colors for consistent use throughout the package
KOMPOT_COLORS = {
    # Direction colors for differential abundance
    "direction": {
        "up": "#d73027",     # red
        "down": "#4575b4",   # blue
        "neutral": "#d3d3d3" # light gray
    }
}


# The functions validate_field_run_id and get_run_from_history have been moved to anndata.utils



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



def compute_mahalanobis_distances(
    diff_values: np.ndarray,
    covariance: Union[np.ndarray, jnp.ndarray, 'da.Array'],
    batch_size: int = 500,
    jit_compile: bool = True,
    eps: float = 1e-8,  # Increased default epsilon for better numerical stability
    progress: bool = True,
) -> np.ndarray:
    """
    Compute Mahalanobis distances for multiple difference vectors efficiently.
    
    This function computes the Mahalanobis distance for each provided difference vector
    using the provided covariance matrix or tensor. It handles both single covariance
    matrix and gene-specific covariance tensors.
    
    Parameters
    ----------
    diff_values : np.ndarray
        The difference vectors for which to compute Mahalanobis distances.
        Shape should be (n_samples, n_features) or (n_features, n_samples).
    covariance : np.ndarray, jnp.ndarray, or dask.array.Array
        Covariance matrix or tensor:
        - If 2D shape (n_points, n_points): shared covariance for all vectors
        - If 3D shape (n_points, n_points, n_genes): gene-specific covariance matrices
        - Can be a dask array for lazy/distributed computation
    batch_size : int, optional
        Number of vectors to process at once, by default 500.
    jit_compile : bool, optional
        Whether to use JAX just-in-time compilation, by default True.
    eps : float, optional
        Small constant for numerical stability, by default 1e-8.
    progress : bool, optional
        Whether to show a progress bar for calculations, by default True.
        
    Returns
    -------
    np.ndarray
        Array of Mahalanobis distances for each input vector.
    """
    from .batch_utils import apply_batched
    from tqdm.auto import tqdm
    from .memory_utils import DASK_AVAILABLE
    
    # Check if covariance is a Dask array
    is_dask = False
    if DASK_AVAILABLE:
        import dask.array as da
        is_dask = isinstance(covariance, da.Array)
    
    # Convert inputs to JAX arrays if not using Dask
    if not is_dask:
        diffs = jnp.array(diff_values)
    else:
        diffs = diff_values
    
    # Handle different input shapes - we want (n_genes, n_points) for gene-wise processing
    if len(diffs.shape) == 1:
        # Single vector, reshape to (1, n_features)
        diffs = diffs.reshape(1, -1)
    
    # Determine if we have gene-specific covariance matrices (3D tensor)
    is_gene_specific = hasattr(covariance, 'shape') and len(covariance.shape) == 3
    
    if is_gene_specific:
        logger.info(f"Computing Mahalanobis distances using gene-specific covariance matrices")
        n_genes = diffs.shape[0]
        n_points = covariance.shape[1]  # Shape is (n_points, n_points, n_genes)

        # Verify tensor dimensions
        if covariance.shape[2] != n_genes:
            logger.warning(
                f"Gene dimension mismatch: covariance has {covariance.shape[2]} genes, "
                f"but diff values has {n_genes} genes. Using genes from diff values."
            )
            # If there's a mismatch, truncate to the shorter dimension
            min_genes = min(covariance.shape[2], n_genes)
            n_genes = min_genes

        # Create a custom function that can be mapped over gene dimensions
        def compute_gene_mahalanobis(g):
            # Extract the difference vector and covariance matrix for this gene
            gene_diff = diffs[g]
            gene_cov = covariance[:, :, g]
            
            # Convert to numpy arrays to ensure consistent handling with JAX version
            gene_diff_np = np.array(gene_diff)
            gene_cov_np = np.array(gene_cov)
            
            # Add a small diagonal term for numerical stability (same as JAX version)
            gene_cov_reg = gene_cov_np + np.eye(gene_cov_np.shape[0]) * eps
            
            try:
                # Try Cholesky decomposition (fast and accurate for positive definite matrices)
                # Use numpy.linalg.cholesky for consistency with JAX version
                L = np.linalg.cholesky(gene_cov_reg)
                
                # Use scipy.linalg.solve_triangular with lower=True, just like JAX version
                from scipy.linalg import solve_triangular
                solved = solve_triangular(L, gene_diff_np, lower=True)
                
                # Compute the Mahalanobis distance exactly as in JAX version
                mahal_dist = float(np.sqrt(np.sum(solved**2)))
                
                return mahal_dist
            except np.linalg.LinAlgError:
                # If Cholesky fails, the matrix is not positive definite
                logger.warning(f"Gene {g}: Cholesky decomposition failed. Matrix is not positive definite. Using NaN.")
                return np.nan
        
        # Handle dask arrays specifically
        if is_dask:
            import dask.array as da
                        
            # Apply the function to each gene in parallel with dask
            # We map the function over the genes and then compute the result
            if progress:
                logger.info(f"Computing Mahalanobis distances for {n_genes:,} genes using dask")
            
            distances = []
            for g in range(n_genes):
                distances.append(dask.delayed(compute_gene_mahalanobis)(g))
                
            # Compute the delayed values to get actual distances
            # Use progress bar if requested
            if progress:
                try:
                    from tqdm.dask import TqdmCallback
                    # Use TqdmCallback for efficient progress tracking
                    with TqdmCallback(desc="Computing Mahalanobis distances"):
                        mahalanobis_distances = np.array(dask.compute(*distances))
                except ImportError:
                    # Fall back to standard compute if tqdm.dask is not available
                    logger.info("tqdm.dask not available, computing without progress bar")
                    mahalanobis_distances = np.array(dask.compute(*distances))
            else:
                # Compute all at once without progress bar
                mahalanobis_distances = np.array(dask.compute(*distances))
            
            return mahalanobis_distances
            
        # For JAX arrays, proceed with the original approach
        cov = jnp.array(covariance)
        mahalanobis_distances = np.zeros(n_genes)
        
        # Process each gene separately to save memory, with progress bar
        gene_iterator = tqdm(range(n_genes), desc="Computing gene-specific Mahalanobis distances") if progress else range(n_genes)
        for g in gene_iterator:
            mahalanobis_distances[g] = compute_gene_mahalanobis(g)
            
        return mahalanobis_distances

    cov = jnp.array(covariance)
    
    # Case: shared covariance matrix (2D matrix)
    # First check for dimension mismatch
    if len(diffs) > 0 and diffs.shape[1] != cov.shape[0]:
        logger.warning(
            f"Dimension mismatch: covariance matrix has shape {cov.shape}, "
            f"but diff vectors have shape {diffs.shape}. Unable to compute distances."
        )
        # Return NaN values to indicate calculation failures
        return np.full(len(diffs), np.nan)
    
    # Try diagonal approximation first if the matrix is large enough
    if cov.shape[0] > 10:
        diag_values = jnp.diag(cov)
        diag_sum = jnp.sum(diag_values)
        total_sum = jnp.sum(jnp.abs(cov))
        diag_ratio = diag_sum / total_sum if total_sum > 0 else 0
        
        # If matrix is nearly diagonal, use faster diagonal approximation
        if diag_ratio > 0.95:
            logger.info("Using fast diagonal matrix approximation")
            
            # Ensure numerical stability
            diag_values = jnp.clip(diag_values, eps, None)
            
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
            distances = apply_batched(
                diag_compute_fn,
                diffs,
                batch_size=batch_size,
                desc=desc
            )
            
            # Post-process to handle NaN and Inf values
            invalid_mask = np.isnan(distances) | np.isinf(distances)
            if np.any(invalid_mask):
                n_invalid = np.sum(invalid_mask)
                logger.warning(f"Found {n_invalid} NaN or Inf Mahalanobis distances in diagonal computation. "
                             f"These will be kept as NaN.")
                
            return distances
    
    # Add a small diagonal term for numerical stability
    cov_stable = cov + jnp.eye(cov.shape[0]) * eps
    
    # Try Cholesky decomposition (should work for positive definite matrices)
    try:
        logger.debug("Computing Cholesky decomposition of covariance matrix")
        chol = jnp.linalg.cholesky(cov_stable)
        
        # Define computation function using Cholesky decomposition
        def compute_cholesky_batch(batch_diffs):
            try:
                # Solve the triangular system for each vector
                solved = jax.vmap(lambda d: jax.scipy.linalg.solve_triangular(chol, d, lower=True))(batch_diffs)
                # Compute the distance as the L2 norm of the solved vector
                return jnp.sqrt(jnp.sum(solved**2, axis=1))
            except Exception as e:
                logger.error(f"Error in Cholesky solution: {e}. Returning NaN values.")
                return jnp.full(batch_diffs.shape[0], np.nan)
        
        # JIT compile if enabled
        if jit_compile:
            chol_compute_fn = jax.jit(compute_cholesky_batch)
        else:
            chol_compute_fn = compute_cholesky_batch
        
        # Process in batches using apply_batched - respect progress parameter
        desc = "Computing Cholesky Mahalanobis distances" if progress else None
        distances = apply_batched(
            chol_compute_fn,
            diffs,
            batch_size=batch_size,
            desc=desc
        )
        
        # Post-process to handle NaN and Inf values
        invalid_mask = np.isnan(distances) | np.isinf(distances)
        if np.any(invalid_mask):
            n_invalid = np.sum(invalid_mask)
            logger.warning(f"Found {n_invalid} NaN or Inf Mahalanobis distances in Cholesky computation. "
                         f"These will be kept as NaN.")
            
        return distances
    except Exception as e:
        logger.warning(f"Cholesky decomposition failed: {e}. Matrix is not positive definite. Returning NaN values.")
        # Return NaNs to indicate calculation failures
        return np.full(len(diffs), np.nan)

def compute_mahalanobis_distance(
    diff_values: np.ndarray,
    covariance_matrix: np.ndarray,
    eps: float = 1e-8,  # Increased default epsilon for better numerical stability
    jit_compile: bool = True,
) -> float:
    """
    Compute the Mahalanobis distance for a vector given a covariance matrix.
    
    This is a convenience function for computing a single Mahalanobis distance.
    For multiple vectors, use compute_mahalanobis_distances for better performance.
    
    Parameters
    ----------
    diff_values : np.ndarray
        The difference vector for which to compute the Mahalanobis distance.
    covariance_matrix : np.ndarray
        The covariance matrix.
    eps : float, optional
        Small constant for numerical stability, by default 1e-10.
    jit_compile : bool, optional
        Whether to use JAX just-in-time compilation, by default True.
        
    Returns
    -------
    float
        The Mahalanobis distance.
    """
    # Ensure diff_values is a single vector
    if len(diff_values.shape) > 1 and diff_values.shape[0] > 1:
        # Multiple vectors - take just the first one for single distance calculation
        diff = diff_values[0]
    else:
        diff = diff_values
    
    # Compute distance using the new unified function
    distances = compute_mahalanobis_distances(
        diff_values=diff,
        covariance=covariance_matrix,
        batch_size=1,
        jit_compile=jit_compile,
        eps=eps
    )
    
    # Return the single distance
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
        partition = G_igraph.community_leiden(
            objective_function="modularity",
            weights=None,
            resolution=resolution,
            beta=0.01,
            n_iterations=2
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
