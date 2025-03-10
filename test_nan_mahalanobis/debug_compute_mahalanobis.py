#!/usr/bin/env python3
"""
Debug script to identify exactly where NaNs are introduced in compute_mahalanobis_distances

This script creates a modified version of the compute_mahalanobis_distances function
with additional debugging and instrumentation.
"""

import sys
import logging
import numpy as np
import jax
import jax.numpy as jnp
from kompot.utils import prepare_mahalanobis_matrix
from scipy.linalg import solve_triangular

# Set up logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("compute_mahalanobis_debug")

def analyze_vector(v, name):
    """Analyze properties of a vector"""
    logger.debug(f"{name} - Shape: {v.shape}")
    logger.debug(f"{name} - Min: {np.min(v):.6e}, Max: {np.max(v):.6e}")
    logger.debug(f"{name} - Has NaNs: {np.isnan(v).any()}, Has Infs: {np.isinf(v).any()}")
    if np.isnan(v).any() or np.isinf(v).any():
        prob_idxs = np.where(np.isnan(v) | np.isinf(v))[0]
        logger.debug(f"{name} - Problem indices: {prob_idxs}")

def instrument_compute_mahalanobis(
    diff_values: np.ndarray,
    covariance_matrix: np.ndarray,
    eps: float = 1e-10,
):
    """
    Instrumented version of prepare_mahalanobis_matrix and compute_mahalanobis_distances
    to debug NaN issues
    """
    logger.info("=== Starting instrumented Mahalanobis calculation ===")
    
    # First analyze inputs
    logger.info(f"diff_values shape: {diff_values.shape}")
    logger.info(f"covariance_matrix shape: {covariance_matrix.shape}")
    
    # Check for NaNs or Infs in inputs
    if np.isnan(diff_values).any() or np.isinf(diff_values).any():
        logger.warning("diff_values contains NaNs or Infs")
        nan_count = np.isnan(diff_values).sum()
        inf_count = np.isinf(diff_values).sum()
        logger.warning(f"NaN count: {nan_count}, Inf count: {inf_count}")
    
    if np.isnan(covariance_matrix).any() or np.isinf(covariance_matrix).any():
        logger.warning("covariance_matrix contains NaNs or Infs")
        nan_count = np.isnan(covariance_matrix).sum()
        inf_count = np.isinf(covariance_matrix).sum()
        logger.warning(f"NaN count: {nan_count}, Inf count: {inf_count}")
    
    # Check eigenvalues of covariance matrix
    try:
        eigenvalues = np.linalg.eigvalsh(covariance_matrix)
        min_eigen = np.min(eigenvalues)
        max_eigen = np.max(eigenvalues)
        logger.info(f"Eigenvalue range: [{min_eigen:.6e}, {max_eigen:.6e}]")
        logger.info(f"Condition number: {max_eigen/min_eigen if min_eigen > 0 else 'infinite'}")
        
        # Show histogram of eigenvalues
        bins = [0, 1e-15, 1e-10, 1e-5, 1e-2, 1.0, np.inf]
        hist, _ = np.histogram(np.abs(eigenvalues), bins=bins)
        logger.info(f"Eigenvalue distribution:")
        for i in range(len(bins)-1):
            logger.info(f"  {bins[i]:.1e} - {bins[i+1]:.1e}: {hist[i]}")
        
        # Check rank
        rank = np.sum(np.abs(eigenvalues) > eps)
        logger.info(f"Matrix rank (eigenvalues > {eps:.1e}): {rank}/{len(eigenvalues)}")
        
        if rank < len(eigenvalues):
            logger.warning("Matrix is rank deficient - this could cause numerical issues")
            
    except Exception as e:
        logger.error(f"Error analyzing eigenvalues: {e}")
    
    # Now prepare the matrix - first step toward Cholesky
    logger.info("=== Preparing matrix for Mahalanobis distance ===")
    
    # Ensure diagonal elements are positive
    diag_clipped = np.clip(np.diag(covariance_matrix), eps, None)
    cov = covariance_matrix.copy()
    np.fill_diagonal(cov, diag_clipped)
    
    # Add a small regularization to the diagonal
    logger.info(f"Adding regularization with eps={eps}")
    reg_cov = cov + np.eye(cov.shape[0]) * eps * 10
    
    # Try Cholesky decomposition with increasing regularization until it succeeds
    success = False
    regularization_factor = 1
    
    while not success and regularization_factor <= 1e6:
        try:
            logger.info(f"Attempting Cholesky with regularization factor {regularization_factor}")
            chol = np.linalg.cholesky(reg_cov)
            success = True
            logger.info("Cholesky decomposition succeeded")
        except np.linalg.LinAlgError as e:
            logger.warning(f"Cholesky failed: {e}")
            regularization_factor *= 10
            reg_cov = cov + np.eye(cov.shape[0]) * eps * regularization_factor
    
    if not success:
        logger.error("Cholesky decomposition failed even with high regularization")
        # Fall back to pseudoinverse
        logger.info("Falling back to pseudoinverse")
        try:
            matrix_inv = np.linalg.pinv(cov)
            logger.info("Pseudoinverse computation succeeded")
            # Compute distances using inverse
            distances = np.zeros(diff_values.shape[0])
            for i in range(diff_values.shape[0]):
                d = diff_values[i]
                # Get the Mahalanobis distance
                md = np.sqrt(d @ matrix_inv @ d)
                distances[i] = md
                # Check for NaN
                if np.isnan(md) or np.isinf(md):
                    logger.warning(f"NaN/Inf in distance {i}: {md}")
                    # Detailed analysis of this calculation
                    prod = matrix_inv @ d
                    quad = d @ prod
                    logger.warning(f"  Matrix-vector product range: [{np.min(prod):.6e}, {np.max(prod):.6e}]")
                    logger.warning(f"  Quadratic form: {quad:.6e}")
                    logger.warning(f"  Square root of: {quad:.6e} -> {md}")
            
            logger.info(f"Distances: {distances}")
            return distances
        except Exception as e:
            logger.error(f"Pseudoinverse also failed: {e}")
            return np.full(diff_values.shape[0], np.nan)
    
    # If Cholesky succeeded, compute distances
    distances = np.zeros(diff_values.shape[0])
    
    for i in range(diff_values.shape[0]):
        d = diff_values[i]
        
        try:
            # First solve the triangular system
            solved = solve_triangular(chol, d, lower=True)
            
            # Analyze the solved vector
            analyze_vector(solved, f"Solved vector {i}")
            
            # Compute sum of squares
            squared_dist = np.sum(solved**2)
            logger.debug(f"Squared distance {i}: {squared_dist:.6e}")
            
            # Take square root
            md = np.sqrt(squared_dist)
            distances[i] = md
            
            # Check for NaN
            if np.isnan(md) or np.isinf(md):
                logger.warning(f"NaN/Inf detected in final distance {i}: {md}")
        except Exception as e:
            logger.error(f"Error computing distance {i}: {e}")
            distances[i] = np.nan
    
    logger.info(f"Final distances: {distances}")
    return distances

def create_test_case_1():
    """Create a test case that's likely to produce NaNs"""
    n_dims = 5
    n_points = 3
    
    # Create a nearly singular covariance matrix
    cov = np.eye(n_dims)
    cov[0, 0] = 1e-14  # Very small variance
    cov[1, 2] = 0.999999999  # Strong correlation
    cov[2, 1] = 0.999999999
    
    # Create diff vectors
    diff = np.ones((n_points, n_dims))
    # Make one vector have extreme values
    diff[1, 0] = 1e10
    
    return diff, cov

def create_test_case_2():
    """Create another test case with zero eigenvalues"""
    n_dims = 10
    n_points = 5
    
    # Create a rank-deficient matrix
    # This matrix will have some zero eigenvalues
    v1 = np.random.randn(n_dims)
    v2 = np.random.randn(n_dims)
    v3 = np.random.randn(n_dims)
    
    # Create a low-rank matrix (rank 3)
    cov = np.outer(v1, v1) + np.outer(v2, v2) + np.outer(v3, v3)
    
    # Create diff vectors
    diff = np.random.randn(n_points, n_dims)
    
    return diff, cov

def create_test_case_3():
    """Create a test case based on a simulated gene expression case"""
    # Simulate cell states
    n_cells = 50
    n_dims = 8
    n_genes = 5
    
    # Create cell states
    X1 = np.random.normal(0, 1, (n_cells, n_dims))
    X2 = np.random.normal(0.2, 1.1, (n_cells, n_dims))
    
    # Make one dimension have very small variance
    X1[:, 0] = 0.001 + np.random.normal(0, 1e-5, n_cells)
    
    # Gene expressions
    expr1 = np.zeros((n_cells, n_genes))
    expr2 = np.zeros((n_cells, n_genes))
    
    # Normal gene
    expr1[:, 0] = 5 + np.random.normal(0, 1, n_cells)
    expr2[:, 0] = 7 + np.random.normal(0, 1, n_cells)
    
    # Gene with tiny difference
    expr1[:, 1] = 2 + np.random.normal(0, 0.1, n_cells)
    expr2[:, 1] = 2 + 1e-10 + np.random.normal(0, 0.1, n_cells)
    
    # Gene with extreme values
    expr1[:, 2] = 1e6 + np.random.normal(0, 1e3, n_cells)
    expr2[:, 2] = 1e6 + 1e3 + np.random.normal(0, 1e3, n_cells)
    
    # Gene perfectly correlated with the small-variance dimension
    expr1[:, 3] = X1[:, 0] * 5 + 2
    expr2[:, 3] = X2[:, 0] * 5 + 2
    
    # Identical gene in both conditions
    same = 3 + np.random.normal(0, 0.5, n_cells)
    expr1[:, 4] = same
    expr2[:, 4] = same
    
    # Compute fold changes
    fold_changes = expr2 - expr1
    
    # Now compute covariance matrix for Mahalanobis distance
    cov1 = np.cov(X1, rowvar=False)
    cov2 = np.cov(X2, rowvar=False)
    combined_cov = (cov1 + cov2) / 2
    
    # Transpose fold_changes for the Mahalanobis calculation
    # as done in the kompot code
    fold_changes_t = fold_changes.T
    
    return fold_changes_t, combined_cov, X1, X2, expr1, expr2

def main():
    """Run the tests"""
    logger.info("=== Test Case 1: Nearly Singular Matrix ===")
    diff1, cov1 = create_test_case_1()
    distances1 = instrument_compute_mahalanobis(diff1, cov1)
    logger.info(f"Test 1 results: {distances1}")
    
    logger.info("\n=== Test Case 2: Rank-Deficient Matrix ===")
    diff2, cov2 = create_test_case_2()
    distances2 = instrument_compute_mahalanobis(diff2, cov2)
    logger.info(f"Test 2 results: {distances2}")
    
    logger.info("\n=== Test Case 3: Simulated Gene Expression ===")
    diff3, cov3, X1, X2, expr1, expr2 = create_test_case_3()
    
    # First analyze the fold change matrix
    logger.info(f"Fold changes shape: {diff3.shape}")
    for i in range(diff3.shape[0]):
        logger.info(f"Gene {i} fold change - min: {np.min(diff3[i]):.6e}, max: {np.max(diff3[i]):.6e}")
    
    # Then compute Mahalanobis distances
    distances3 = instrument_compute_mahalanobis(diff3, cov3)
    logger.info(f"Test 3 results: {distances3}")
    
    # Check for NaNs
    if np.isnan(distances3).any() or np.isinf(distances3).any():
        nan_indices = np.where(np.isnan(distances3))[0]
        inf_indices = np.where(np.isinf(distances3))[0]
        logger.info(f"NaN indices: {nan_indices}")
        logger.info(f"Inf indices: {inf_indices}")
        
        # Analyze problematic indices
        problem_indices = np.concatenate([nan_indices, inf_indices])
        for idx in problem_indices:
            logger.info(f"Problem gene {idx}:")
            logger.info(f"  Fold change: {diff3[idx]}")
            logger.info(f"  Min: {np.min(diff3[idx]):.6e}, Max: {np.max(diff3[idx]):.6e}")
            logger.info(f"  Variance: {np.var(diff3[idx]):.6e}")
    
if __name__ == "__main__":
    main()