#!/usr/bin/env python3
"""
Test script to reproduce and debug NaN Mahalanobis distances issue
"""

import sys
import logging
import numpy as np
from kompot.differential import DifferentialExpression
import jax.numpy as jnp
from kompot.utils import prepare_mahalanobis_matrix, compute_mahalanobis_distances

# Set up logging to see detailed information
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("kompot_test")

def analyze_matrix_properties(matrix, name):
    """Analyze numerical properties of a matrix"""
    logger.info(f"\n{'='*20} {name} Analysis {'='*20}")
    logger.info(f"Matrix shape: {matrix.shape}")
    logger.info(f"Contains NaN: {np.isnan(matrix).any()}")
    logger.info(f"Contains Inf: {np.isinf(matrix).any()}")
    
    try:
        # Check eigenvalues
        eigenvalues = np.linalg.eigvalsh(matrix)
        min_eig = np.min(eigenvalues)
        max_eig = np.max(eigenvalues)
        logger.info(f"Eigenvalue range: [{min_eig:.6e}, {max_eig:.6e}]")
        logger.info(f"Condition number: {max_eig/min_eig if min_eig > 0 else 'infinite'}")
        logger.info(f"Number of near-zero eigenvalues (< 1e-10): {np.sum(eigenvalues < 1e-10)}")
    except Exception as e:
        logger.error(f"Error computing eigenvalues: {e}")

def create_toy_dataset(n_cells=100, n_dims=5, n_genes=20, ill_conditioned=False):
    """Create a toy dataset designed to test Mahalanobis distance calculation
    
    Parameters:
    ----------
    n_cells : int
        Number of cells per condition
    n_dims : int
        Number of dimensions for cell states
    n_genes : int
        Number of genes to simulate
    ill_conditioned : bool
        If True, create an ill-conditioned dataset to trigger numerical issues
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create cell states for each condition - slightly different distributions
    X1 = np.random.normal(0, 1, (n_cells, n_dims))
    X2 = np.random.normal(0.5, 1.2, (n_cells, n_dims))
    
    if ill_conditioned:
        # Make one dimension have very small variance in one condition
        # This creates an ill-conditioned covariance matrix
        X1[:, 0] = np.random.normal(0, 1e-8, n_cells)
        
        # Make another dimension perfectly correlated with another
        # This creates a singular or near-singular covariance matrix
        X1[:, 1] = X1[:, 2] * 1.000001 + np.random.normal(0, 1e-10, n_cells)
    
    # Create gene expression values with clear differences between conditions
    expr1 = np.zeros((n_cells, n_genes))
    expr2 = np.zeros((n_cells, n_genes))
    
    for i in range(n_genes):
        # Normal genes with different expressions
        if i < n_genes - 5:
            expr1[:, i] = np.random.normal(5 + i*0.5, 1, n_cells)
            expr2[:, i] = np.random.normal(5 + i*0.5 + 2, 1, n_cells)
        else:
            # Create some problematic genes with tiny differences or extreme values
            if i == n_genes - 5:
                # Very small difference between conditions
                expr1[:, i] = np.random.normal(10, 1, n_cells)
                expr2[:, i] = np.random.normal(10 + 1e-10, 1, n_cells)
            elif i == n_genes - 4:
                # Extremely large values
                expr1[:, i] = np.random.normal(1e6, 1e4, n_cells)
                expr2[:, i] = np.random.normal(1e6 + 1e4, 1e4, n_cells)
            elif i == n_genes - 3:
                # Very small values
                expr1[:, i] = np.random.normal(1e-8, 1e-9, n_cells)
                expr2[:, i] = np.random.normal(1e-8 + 1e-9, 1e-9, n_cells)
            elif i == n_genes - 2:
                # Identical values in both conditions
                same_vals = np.random.normal(5, 1, n_cells)
                expr1[:, i] = same_vals
                expr2[:, i] = same_vals
            else:
                # Perfectly correlated with cell state
                expr1[:, i] = X1[:, 0] * 10 + 5
                expr2[:, i] = X2[:, 0] * 10 + 5
    
    return X1, expr1, X2, expr2

def test_mahalanobis_direct():
    """Test the Mahalanobis distance computation directly to find potential issues"""
    logger.info("Testing direct Mahalanobis distance computation")
    
    # Create a covariance matrix with known issues
    n_dims = 5
    cov1 = np.eye(n_dims)
    # Make the matrix slightly ill-conditioned
    cov1[0, 0] = 1e-10  # Very small variance in first dimension
    cov1[1, 2] = 0.9999  # Strong correlation between dims 1 and 2
    cov1[2, 1] = 0.9999
    
    # Create another matrix with different conditioning
    cov2 = np.eye(n_dims) * 2
    
    # Average them as done in the code
    combined_cov = (cov1 + cov2) / 2
    
    # Analyze the matrices
    analyze_matrix_properties(cov1, "Covariance Matrix 1")
    analyze_matrix_properties(cov2, "Covariance Matrix 2")
    analyze_matrix_properties(combined_cov, "Combined Covariance Matrix")
    
    # Create a diff vector with some extreme values
    diff_values = np.ones((10, n_dims))
    diff_values[5, 0] = 1e10  # Extreme value in problem dimension
    
    # Test prepare_mahalanobis_matrix
    prepared_matrix = prepare_mahalanobis_matrix(
        covariance_matrix=combined_cov,
        eps=1e-10,
        jit_compile=False
    )
    
    # Check which method was used
    logger.info(f"Prepared matrix uses: " + 
                ("diagonal" if prepared_matrix['is_diagonal'] else
                 "Cholesky" if prepared_matrix['chol'] is not None else
                 "matrix inverse" if prepared_matrix['matrix_inv'] is not None else
                 "unknown"))
    
    # Test the distance computation
    try:
        distances = compute_mahalanobis_distances(
            diff_values=diff_values,
            prepared_matrix=prepared_matrix,
            batch_size=5,
            jit_compile=False
        )
        logger.info(f"Computed distances: {distances}")
        logger.info(f"Contains NaN: {np.isnan(distances).any()}")
        logger.info(f"Contains Inf: {np.isinf(distances).any()}")
        
        if np.isnan(distances).any() or np.isinf(distances).any():
            problem_indices = np.where(np.isnan(distances) | np.isinf(distances))[0]
            logger.info(f"Problem indices: {problem_indices}")
            logger.info(f"Corresponding diff values: {diff_values[problem_indices]}")
    except Exception as e:
        logger.error(f"Error computing distances: {e}")

def test_differential_expression():
    """Test the DifferentialExpression class with a toy dataset"""
    logger.info("Testing DifferentialExpression with toy dataset")
    
    # Create a normal dataset and an ill-conditioned one
    X1, expr1, X2, expr2 = create_toy_dataset(ill_conditioned=False)
    X1_ill, expr1_ill, X2_ill, expr2_ill = create_toy_dataset(ill_conditioned=True)
    
    # Test with normal dataset first
    logger.info("\n" + "="*20 + " Normal Dataset " + "="*20)
    diff_expr = DifferentialExpression(jit_compile=False, mahalanobis_batch_size=5)
    diff_expr.fit(X1, expr1, X2, expr2)
    
    # Predict and specifically request Mahalanobis distance computation
    results = diff_expr.predict(X1, compute_mahalanobis=True)
    
    # Check if Mahalanobis distances have NaNs
    mahalanobis_distances = results.get('mahalanobis_distances', None)
    if mahalanobis_distances is not None:
        logger.info(f"Normal dataset - Mahalanobis distances shape: {mahalanobis_distances.shape}")
        logger.info(f"Normal dataset - NaNs in Mahalanobis: {np.isnan(mahalanobis_distances).any()}")
        logger.info(f"Normal dataset - Infs in Mahalanobis: {np.isinf(mahalanobis_distances).any()}")
        
        if np.isnan(mahalanobis_distances).any() or np.isinf(mahalanobis_distances).any():
            problem_indices = np.where(np.isnan(mahalanobis_distances) | np.isinf(mahalanobis_distances))[0]
            logger.info(f"Problem gene indices: {problem_indices}")
    
    # Now test with ill-conditioned dataset
    logger.info("\n" + "="*20 + " Ill-conditioned Dataset " + "="*20)
    diff_expr_ill = DifferentialExpression(jit_compile=False, mahalanobis_batch_size=5)
    diff_expr_ill.fit(X1_ill, expr1_ill, X2_ill, expr2_ill)
    
    # Predict and request Mahalanobis distance computation
    results_ill = diff_expr_ill.predict(X1_ill, compute_mahalanobis=True)
    
    # Check if Mahalanobis distances have NaNs
    mahalanobis_distances_ill = results_ill.get('mahalanobis_distances', None)
    if mahalanobis_distances_ill is not None:
        logger.info(f"Ill-conditioned dataset - Mahalanobis distances shape: {mahalanobis_distances_ill.shape}")
        logger.info(f"Ill-conditioned dataset - NaNs in Mahalanobis: {np.isnan(mahalanobis_distances_ill).any()}")
        logger.info(f"Ill-conditioned dataset - Infs in Mahalanobis: {np.isinf(mahalanobis_distances_ill).any()}")
        
        if np.isnan(mahalanobis_distances_ill).any() or np.isinf(mahalanobis_distances_ill).any():
            problem_indices = np.where(np.isnan(mahalanobis_distances_ill) | np.isinf(mahalanobis_distances_ill))[0]
            logger.info(f"Problem gene indices: {problem_indices}")

def main():
    """Run all tests"""
    logger.info("Starting NaN Mahalanobis distances debug tests")
    
    # First test direct Mahalanobis computation
    test_mahalanobis_direct()
    
    # Then test in the context of DifferentialExpression
    test_differential_expression()
    
    logger.info("Completed all tests")

if __name__ == "__main__":
    main()