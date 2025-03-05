"""Utility functions for Kompot package."""

import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple, List, Optional, Union, Callable

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
    jit_compile: bool = False,
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
        Whether to use JAX just-in-time compilation, by default False.
        
    Returns
    -------
    float
        The Mahalanobis distance.
    """
    # Convert inputs to JAX arrays
    diff_values = jnp.array(diff_values)
    covariance_matrix = jnp.array(covariance_matrix)
    
    def _compute_mahalanobis(diff, cov, diag_adj=None):
        # Adjust covariance matrix if needed
        if diag_adj is not None:
            diag_adj = jnp.array(diag_adj)
            cov = cov + jnp.diag(diag_adj)
        
        # Ensure numerical stability
        diag_clipped = jnp.clip(jnp.diag(cov), eps, None)
        cov = cov.at[jnp.arange(cov.shape[0]), jnp.arange(cov.shape[0])].set(diag_clipped)
        
        # Cholesky decomposition
        chol = jnp.linalg.cholesky(cov)
        
        # Solve the system and calculate squared Mahalanobis distance
        solved = jnp.linalg.solve(chol, diff)
        mahalanobis_squared = jnp.sum(solved**2)
        
        # Return the Mahalanobis distance (square root of the squared distance)
        return jnp.sqrt(mahalanobis_squared)
    
    # JIT-compile the function if requested
    if jit_compile:
        _compute_mahalanobis = jax.jit(_compute_mahalanobis)
    
    return _compute_mahalanobis(diff_values, covariance_matrix, diag_adjustments)


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