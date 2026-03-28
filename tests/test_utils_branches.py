"""Tests for kompot/utils.py targeting uncovered lines.

Covers KOMPOT_COLORS, compute_mahalanobis_distances (2D, 3D, diagonal,
Cholesky failures, JIT paths), compute_mahalanobis_distance, and
_sanitize_name.
"""

import pytest
import numpy as np


class TestUtils:
    """Tests for utils.py targeting uncovered lines."""

    def test_kompot_colors_structure(self):
        """Line 18, 27-31: KOMPOT_COLORS is accessible and has expected structure."""
        from kompot.utils import KOMPOT_COLORS
        assert "direction" in KOMPOT_COLORS
        assert "up" in KOMPOT_COLORS["direction"]
        assert "down" in KOMPOT_COLORS["direction"]
        assert "neutral" in KOMPOT_COLORS["direction"]

    def test_mahalanobis_dimension_mismatch(self):
        """Lines 299-300: dimension mismatch returns NaN."""
        from kompot.utils import compute_mahalanobis_distances
        diffs = np.random.randn(5, 3)
        cov = np.eye(4)  # Wrong shape
        result = compute_mahalanobis_distances(diffs, cov, progress=False)
        assert np.all(np.isnan(result))

    def test_mahalanobis_cholesky_failure(self):
        """Lines 406-409: Cholesky failure returns NaN."""
        from kompot.utils import compute_mahalanobis_distances
        diffs = np.random.randn(3, 3)
        # Create a non-positive definite matrix
        cov = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
        result = compute_mahalanobis_distances(diffs, cov, progress=False, eps=0)
        assert np.all(np.isnan(result))

    def test_mahalanobis_diagonal_approximation(self):
        """Lines 283: diagonal matrix fast path."""
        from kompot.utils import compute_mahalanobis_distances
        n = 15
        diffs = np.random.randn(5, n)
        cov = np.diag(np.ones(n) * 2.0)  # Pure diagonal -> ratio > 0.95
        result = compute_mahalanobis_distances(diffs, cov, progress=False)
        assert result.shape == (5,)
        assert not np.any(np.isnan(result))

    def test_mahalanobis_diagonal_with_invalid(self):
        """Lines 365-366: NaN in diagonal computation."""
        from kompot.utils import compute_mahalanobis_distances
        n = 15
        diffs = np.random.randn(5, n)
        diffs[0, :] = np.nan  # Insert NaN
        cov = np.diag(np.ones(n) * 2.0)
        result = compute_mahalanobis_distances(diffs, cov, progress=False)
        assert np.any(np.isnan(result))

    def test_mahalanobis_gene_specific_3d(self):
        """Lines 140-141, 147, 202-205: gene-specific 3D covariance."""
        from kompot.utils import compute_mahalanobis_distances
        n_points = 5
        n_genes = 3
        diffs = np.random.randn(n_genes, n_points)
        cov = np.zeros((n_points, n_points, n_genes))
        for g in range(n_genes):
            cov[:, :, g] = np.eye(n_points)
        result = compute_mahalanobis_distances(diffs, cov, progress=False)
        assert result.shape == (n_genes,)

    def test_mahalanobis_gene_specific_cholesky_fail(self):
        """Lines 209-236: gene-specific with non-PD matrix."""
        from kompot.utils import compute_mahalanobis_distances
        n_points = 5
        n_genes = 2
        diffs = np.random.randn(n_genes, n_points)
        cov = np.zeros((n_points, n_points, n_genes))
        cov[:, :, 0] = np.eye(n_points)
        cov[:, :, 1] = -np.eye(n_points)  # Not positive definite
        result = compute_mahalanobis_distances(diffs, cov, progress=False, eps=0)
        # gene 0 should be OK, gene 1 should be NaN
        assert not np.isnan(result[0])
        assert np.isnan(result[1])

    def test_mahalanobis_single_vector(self):
        """Line 152: single vector input."""
        from kompot.utils import compute_mahalanobis_distances
        diff = np.array([1.0, 0.0, 0.0])
        cov = np.eye(3)
        result = compute_mahalanobis_distances(diff, cov, progress=False)
        assert result.shape == (1,)
        assert abs(result[0] - 1.0) < 0.1

    def test_compute_mahalanobis_distance_single(self):
        """Lines 379-381: convenience function for single distance."""
        from kompot.utils import compute_mahalanobis_distance
        diff = np.array([1.0, 0.0])
        cov = np.eye(2)
        result = compute_mahalanobis_distance(diff, cov)
        assert isinstance(result, float)
        assert abs(result - 1.0) < 0.1

    def test_sanitize_name_fallback(self):
        """Line 18: _sanitize_name fallback."""
        from kompot.utils import _sanitize_name
        assert _sanitize_name("hello world/foo") == "hello_world_foo"


class TestUtilsDeepCoverage:
    """Deeper coverage for utils.py."""

    def test_dask_import_branch(self):
        """Lines 27-31: DASK_AVAILABLE import branch."""
        from kompot.utils import KOMPOT_COLORS
        # Just verify the module loaded properly
        assert KOMPOT_COLORS is not None

    def test_mahalanobis_no_jit(self):
        """Lines 283, 285: jit_compile=False path."""
        from kompot.utils import compute_mahalanobis_distances
        n = 15
        diffs = np.random.randn(5, n)
        cov = np.diag(np.ones(n) * 2.0)
        result = compute_mahalanobis_distances(
            diffs, cov, jit_compile=False, progress=False
        )
        assert result.shape == (5,)

    def test_mahalanobis_diagonal_variance(self):
        """Lines 317-369: diagonal_variance factor trick."""
        from kompot.utils import compute_mahalanobis_distances
        n_points = 5
        n_genes = 3
        diffs = np.random.randn(n_genes, n_points)
        cov = np.eye(n_points).astype(np.float64)
        diag_var = np.abs(np.random.randn(n_genes, n_points)).astype(np.float64)
        result = compute_mahalanobis_distances(
            diffs, cov, progress=False, diagonal_variance=diag_var
        )
        assert result.shape == (n_genes,)
        assert not np.any(np.isnan(result))

    def test_mahalanobis_gene_specific_with_diag_var(self):
        """Lines 183-184: gene-specific 3D cov with diagonal_variance."""
        from kompot.utils import compute_mahalanobis_distances
        n_points = 5
        n_genes = 2
        diffs = np.random.randn(n_genes, n_points)
        cov = np.zeros((n_points, n_points, n_genes))
        for g in range(n_genes):
            cov[:, :, g] = np.eye(n_points)
        diag_var = np.abs(np.random.randn(n_genes, n_points))
        result = compute_mahalanobis_distances(
            diffs, cov, progress=False, diagonal_variance=diag_var
        )
        assert result.shape == (n_genes,)

    def test_mahalanobis_cholesky_no_jit(self):
        """Line 387: Cholesky path without JIT."""
        from kompot.utils import compute_mahalanobis_distances
        diffs = np.random.randn(3, 5)
        cov = np.eye(5)
        result = compute_mahalanobis_distances(
            diffs, cov, jit_compile=False, progress=False
        )
        assert result.shape == (3,)
