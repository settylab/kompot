"""Utility functions for Kompot package."""

import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple, List, Optional, Union, Callable
from scipy.linalg import solve_triangular

import pynndescent
import igraph as ig
import leidenalg as la
import logging

logger = logging.getLogger("kompot")


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
        
    Returns
    -------
    np.ndarray
        Array of Mahalanobis distances for each input vector.
    """
    # Convert input to JAX array
    diffs = jnp.array(diff_values)
    
    # Handle different input shapes - we want (n_samples, n_features)
    if len(diffs.shape) == 1:
        # Single vector, reshape to (1, n_features)
        diffs = diffs.reshape(1, -1)
    elif len(diffs.shape) == 2 and diffs.shape[0] < diffs.shape[1]:
        # More features than samples, likely (n_features, n_samples)
        diffs = diffs.T
    
    # Get number of samples
    n_samples = diffs.shape[0]
    
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
        
        # Process in batches
        results = []
        for i in range(0, n_samples, batch_size):
            batch = diffs[i:i+batch_size]
            batch_results = diag_compute_fn(batch)
            results.append(np.array(batch_results))
        
        return np.concatenate(results) if len(results) > 1 else results[0]
    
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
        
        # Process in batches
        results = []
        for i in range(0, n_samples, batch_size):
            batch = diffs[i:i+batch_size]
            batch_results = chol_compute_fn(batch)
            results.append(np.array(batch_results))
        
        return np.concatenate(results) if len(results) > 1 else results[0]
    
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
        
        # Process in batches
        results = []
        for i in range(0, n_samples, batch_size):
            batch = diffs[i:i+batch_size]
            batch_results = inv_compute_fn(batch)
            results.append(np.array(batch_results))
        
        return np.concatenate(results) if len(results) > 1 else results[0]
    
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
    return float(distances[0]) if len(distances) > 1 else float(distances)


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