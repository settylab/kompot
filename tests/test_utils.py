"""Tests for the utils module."""

import numpy as np
import pytest
from kompot.utils import compute_mahalanobis_distance, build_graph, find_landmarks


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


def test_build_graph():
    """Test the build_graph function."""
    # Create a simple dataset
    X = np.random.randn(100, 5)
    
    # Build the graph
    edges, index = build_graph(X, n_neighbors=10)
    
    # Check that edges is a list of tuples
    assert isinstance(edges, list)
    assert all(isinstance(edge, tuple) for edge in edges)
    
    # Check that each edge connects valid node indices
    n_samples = X.shape[0]
    assert all(0 <= i < n_samples and 0 <= j < n_samples for i, j in edges)
    
    # Check that the index is a valid pynndescent object
    assert hasattr(index, 'query')


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