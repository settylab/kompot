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


def compute_mahalanobis_distance(
    diff_values: np.ndarray,
    covariance_matrix: np.ndarray,
    diag_adjustments: Optional[np.ndarray] = None,
    eps: float = 1e-12,
    jit_compile: bool = True,
) -> float:
    """
    Compute the Mahalanobis distance for a vector given a covariance matrix.
    
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
    # Start timing and debug logging
    import time
    start_time = time.time()
    
    # Log matrix dimensions for debugging
    logger.debug(f"Mahalanobis input shapes: diff={diff_values.shape}, cov={covariance_matrix.shape}, "
                f"diag_adjustments={diag_adjustments.shape if diag_adjustments is not None else None}")
    
    # Check for potential batch-based optimization
    if diff_values.shape[0] > 5000:
        logger.warning(f"Large vector size ({diff_values.shape}) may cause slow performance. Consider batching.")
    
    if covariance_matrix.shape[0] > 500:
        logger.warning(f"Large covariance matrix ({covariance_matrix.shape}) may cause slow Cholesky decomposition.")
    
    # Check if matrix is diagonal or near-diagonal (which would allow much faster calculation)
    if covariance_matrix.shape[0] > 10:
        diag_sum = np.sum(np.diag(covariance_matrix))
        total_sum = np.sum(np.abs(covariance_matrix))
        diag_ratio = diag_sum / total_sum if total_sum > 0 else 0
        logger.debug(f"Diagonal ratio: {diag_ratio:.4f} (1.0 means perfectly diagonal)")
        
        # If matrix is nearly diagonal, use faster diagonal approximation
        if diag_ratio > 0.95:
            logger.info("Using fast diagonal matrix approximation")
            # For diagonal matrix, Mahalanobis is just a weighted Euclidean distance
            cov_diag = np.diag(covariance_matrix)
            if diag_adjustments is not None:
                cov_diag = cov_diag + diag_adjustments
            # Avoid division by zero
            cov_diag = np.clip(cov_diag, eps, None)
            weighted_diff = diff_values / np.sqrt(cov_diag)
            result = np.sqrt(np.sum(weighted_diff**2))
            logger.debug(f"Diagonal approx took {time.time() - start_time:.4f}s")
            return float(result)
    
    # Define the JAX implementation with triangular solve
    def _compute_mahalanobis_jax(diff, cov, diag_adj=None):
        # Start timing for JAX operations
        jax_start = time.time()
        
        # Adjust covariance matrix if needed
        if diag_adj is not None:
            diag_adj = jnp.array(diag_adj)
            cov = cov + jnp.diag(diag_adj)
        
        # Ensure numerical stability
        diag_clipped = jnp.clip(jnp.diag(cov), eps, None)
        cov = cov.at[jnp.arange(cov.shape[0]), jnp.arange(cov.shape[0])].set(diag_clipped)
        
        # Cholesky decomposition
        logger.debug(f"JAX matrix preparation took {time.time() - jax_start:.4f}s")
        chol_start = time.time()
        chol = jnp.linalg.cholesky(cov)
        logger.debug(f"JAX Cholesky decomposition took {time.time() - chol_start:.4f}s")
        
        # Triangular solve: solve L*y = diff (forward substitution)
        solve_start = time.time()
        y = jax.scipy.linalg.solve_triangular(chol, diff, lower=True)
        logger.debug(f"JAX triangular solve took {time.time() - solve_start:.4f}s")
        
        # The squared Mahalanobis distance is the squared L2 norm of y
        final_start = time.time()
        mahalanobis_squared = jnp.sum(y**2)
        result = jnp.sqrt(mahalanobis_squared)
        logger.debug(f"JAX final calculation took {time.time() - final_start:.4f}s")
        
        # Return the Mahalanobis distance (square root of the squared distance)
        return result
    
    # Define the NumPy implementation as a fallback
    def _compute_mahalanobis_numpy(diff, cov, diag_adj=None):
        # Create a copy to avoid modifying the original
        numpy_start = time.time()
        cov = np.array(cov, copy=True)
        
        # Adjust covariance matrix if needed
        if diag_adj is not None:
            cov = cov + np.diag(diag_adj)
        
        # Ensure numerical stability
        diag_clipped = np.clip(np.diag(cov), eps, None)
        np.fill_diagonal(cov, diag_clipped)
        logger.debug(f"NumPy matrix preparation took {time.time() - numpy_start:.4f}s")
        
        # Compare with original approach to see if direct matrix inversion is faster for small matrices
        if cov.shape[0] < 100:
            try:
                inv_start = time.time()
                cov_inv = np.linalg.inv(cov)
                inv_time = time.time() - inv_start
                logger.debug(f"NumPy direct inversion took {inv_time:.4f}s")
                
                calc_start = time.time()
                result_direct = np.sqrt(np.dot(diff, np.dot(cov_inv, diff)))
                calc_time = time.time() - calc_start
                logger.debug(f"NumPy direct calculation took {calc_time:.4f}s")
                
                logger.debug(f"NumPy direct approach total: {inv_time + calc_time:.4f}s")
                return float(result_direct)
            except np.linalg.LinAlgError:
                logger.debug("Direct inversion failed, using Cholesky decomposition")
        
        # Cholesky decomposition
        chol_start = time.time()
        try:
            chol = np.linalg.cholesky(cov)
            logger.debug(f"NumPy Cholesky decomposition took {time.time() - chol_start:.4f}s")
            
            # Solve the triangular system for improved efficiency
            solve_start = time.time()
            y = solve_triangular(chol, diff, lower=True)
            logger.debug(f"NumPy triangular solve took {time.time() - solve_start:.4f}s")
            
            # The squared Mahalanobis distance is the squared L2 norm of y
            final_start = time.time()
            mahalanobis_squared = np.sum(y**2)
            result = np.sqrt(mahalanobis_squared)
            logger.debug(f"NumPy final calculation took {time.time() - final_start:.4f}s")
            
            return float(result)
        except np.linalg.LinAlgError as e:
            logger.warning(f"Cholesky decomposition failed: {e}. Using pseudoinverse.")
            # Fallback to pseudoinverse
            pinv_start = time.time()
            cov_inv = np.linalg.pinv(cov)
            logger.debug(f"NumPy pseudoinverse took {time.time() - pinv_start:.4f}s")
            
            final_start = time.time()
            result = np.sqrt(np.dot(diff, np.dot(cov_inv, diff)))
            logger.debug(f"NumPy final calculation with pinv took {time.time() - final_start:.4f}s")
            
            return float(result)
    
    # Convert inputs to appropriate array types
    diff = np.asarray(diff_values)
    cov = np.asarray(covariance_matrix)
    
    try:
        if jit_compile:
            # Try to use the JAX implementation with JIT compilation
            logger.debug("Using JAX JIT implementation")
            jit_start = time.time()
            jit_fn = jax.jit(_compute_mahalanobis_jax)
            # Convert to JAX arrays
            jax_diff = jnp.array(diff)
            jax_cov = jnp.array(cov)
            jax_diag_adj = jnp.array(diag_adjustments) if diag_adjustments is not None else None
            
            # Compute using JAX and convert result back to numpy
            result = jit_fn(jax_diff, jax_cov, jax_diag_adj)
            numpy_result = np.array(result)
            logger.debug(f"JAX total computation took {time.time() - jit_start:.4f}s")
            return float(numpy_result)
        else:
            # Fall back to NumPy implementation if JIT is disabled
            logger.debug("Using NumPy implementation")
            numpy_start = time.time()
            result = _compute_mahalanobis_numpy(diff, cov, diag_adjustments)
            logger.debug(f"NumPy total computation took {time.time() - numpy_start:.4f}s")
            return result
    except Exception as e:
        # Fall back to NumPy implementation if there's any JAX error
        logger.warning(f"JAX computation failed: {e}. Falling back to NumPy implementation.")
        numpy_start = time.time()
        result = _compute_mahalanobis_numpy(diff, cov, diag_adjustments)
        logger.debug(f"NumPy fallback computation took {time.time() - numpy_start:.4f}s")
        return result


def build_graph(X: np.ndarray, n_neighbors: int = 15) -> Tuple[List[Tuple[int, int]], any]:
    """
    Build a graph from data points using pynndescent.
    
    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n_samples, n_features).
    n_neighbors : int, optional
        Number of neighbors for graph construction, by default 15.
        
    Returns
    -------
    Tuple[List[Tuple[int, int]], any]
        A tuple containing:
        - edges: List of tuples representing the edges
        - index: The pynndescent nearest neighbor index
    """
    # Compute nearest neighbors
    index = pynndescent.NNDescent(X, n_neighbors=n_neighbors, metric="euclidean")
    indices, _ = index.neighbor_graph

    n_samples = X.shape[0]
    sources = np.repeat(np.arange(n_samples), n_neighbors)
    targets = indices.flatten()
    edges = list(zip(sources, targets))
    
    # Remove duplicates
    edges = list(set(tuple(sorted(edge)) for edge in edges))
    
    return edges, index


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