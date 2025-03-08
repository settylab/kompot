"""Tests for the utils module."""

import numpy as np
import pytest
from kompot.utils import compute_mahalanobis_distance, find_landmarks, get_run_from_history


def test_compute_mahalanobis_distance():
    """Test the compute_mahalanobis_distance function."""
    # Define a simple test case
    diff_values = np.array([1.0, 2.0, 3.0])
    covariance_matrix = np.eye(3)  # Identity matrix for covariance
    
    # Compute the distance
    distance = compute_mahalanobis_distance(diff_values, covariance_matrix)
    
    # For identity covariance, this should be the same as Euclidean distance
    expected_distance = np.sqrt(np.sum(diff_values**2))
    assert np.isclose(distance, expected_distance)
    
    # Test with diagonal adjustments
    diag_adjustments = np.array([0.5, 0.5, 0.5])
    distance_with_adj = compute_mahalanobis_distance(
        diff_values, covariance_matrix, diag_adjustments
    )
    
    # With adjustments, the distance should change
    assert distance_with_adj != distance


def test_find_landmarks():
    """Test the find_landmarks function."""
    # Create a simple dataset with some structure
    np.random.seed(42)
    X = np.vstack([
        np.random.randn(50, 5),              # Cluster 1
        np.random.randn(50, 5) + 5,          # Cluster 2
        np.random.randn(50, 5) - 5,          # Cluster 3
    ])
    
    # Find landmarks
    n_landmarks = 10
    landmarks, landmark_indices = find_landmarks(X, n_clusters=n_landmarks, max_iter=3)
    
    # Check shapes
    assert landmarks.shape[0] <= n_landmarks  # Could be less due to optimization
    assert landmarks.shape[1] == X.shape[1]
    assert len(landmark_indices) == landmarks.shape[0]
    
    # Check that landmark indices are valid
    assert all(0 <= idx < X.shape[0] for idx in landmark_indices)
    
    # Check that landmarks are actual data points
    for i, idx in enumerate(landmark_indices):
        assert np.allclose(landmarks[i], X[idx])


def test_get_run_from_history():
    """Test the get_run_from_history function with different scenarios."""
    # Skip if anndata not installed
    try:
        import anndata
    except ImportError:
        pytest.skip("anndata not installed, skipping test")
    
    # Create an AnnData object with run history
    adata = anndata.AnnData(X=np.random.randn(10, 10))
    
    # Test with None run_id
    assert get_run_from_history(adata, None) is None
    
    # Test with no kompot_run_history in adata
    assert get_run_from_history(adata, 0) is None
    
    # Add run history
    adata.uns['kompot_run_history'] = [
        {'run_id': 0, 'timestamp': '2023-01-01', 'name': 'run_0'},
        {'run_id': 1, 'timestamp': '2023-01-02', 'name': 'run_1'},
        {'run_id': 2, 'timestamp': '2023-01-03', 'name': 'run_2'}
    ]
    
    # Test with valid run_id
    result = get_run_from_history(adata, 1)
    assert result is not None
    assert result['run_id'] == 1
    assert result['name'] == 'run_1'
    
    # Test with negative run_id (counting from end)
    result = get_run_from_history(adata, -1)
    assert result is not None
    assert result['run_id'] == 2  # Last run
    assert result['name'] == 'run_2'
    
    result = get_run_from_history(adata, -2)
    assert result is not None
    assert result['run_id'] == 1  # Second to last run
    assert result['name'] == 'run_1'
    
    # Test with out-of-bounds run_id
    assert get_run_from_history(adata, 5) is None
    assert get_run_from_history(adata, -5) is None