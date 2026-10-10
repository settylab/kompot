"""The landmark snap in ``find_landmarks`` returns the cell exactly nearest each centroid.

``find_landmarks`` clusters the cells and snaps each cluster centroid to a cell.
Through 0.8.0 the snap queried the approximate pynndescent index built for the
graph, which on large, high-dimensional data often returns a nearby cell
instead of the nearest one. The snap is now an exact, chunked brute-force
search with a defined tie-break.
"""

import numpy as np
import pytest

import kompot.utils as kutils
from kompot.utils import find_landmarks


def _brute_force_nearest(X, queries):
    """Reference: direct float64 squared distances, first index on ties."""
    X = np.asarray(X, dtype=np.float64)
    return np.array(
        [int(np.argmin(((X - q) ** 2).sum(axis=1))) for q in np.asarray(queries)]
    )


class _FixedPartition:
    def __init__(self, membership):
        self.membership = list(membership)


def _clustered(n, d, k, seed):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(k, d)) * 3.0
    labels = rng.integers(0, k, n)
    X = centres[labels] + rng.normal(size=(n, d))
    return X, labels


class TestFindLandmarksExactSnap:
    def test_landmarks_are_exact_nearest_to_centroids(self, monkeypatch):
        """Each landmark is the cell exactly nearest its cluster centroid.

        20 000 cells in 50 dimensions with 100 clusters: here the approximate
        index query misses the exact nearest cell for roughly one centroid in
        eight (13 of 100 with pynndescent 0.6.0), so this fails on the
        approximate snap. The clustering is fixed to the generating labels so
        that the centroids are known.
        """
        X, labels = _clustered(20_000, 50, 100, seed=0)
        monkeypatch.setattr(
            kutils,
            "find_optimal_resolution",
            lambda *a, **k: (1.0, _FixedPartition(labels)),
        )

        landmarks, indices = find_landmarks(X, n_clusters=100, random_state=0)

        cluster_ids = np.unique(labels)
        centroids = np.array([X[labels == c].mean(axis=0) for c in cluster_ids])
        expected = _brute_force_nearest(X, centroids)
        missed = int(np.sum(indices != expected))
        assert missed == 0, f"{missed} of {len(expected)} landmarks are not the nearest cell"
        np.testing.assert_array_equal(landmarks, X[expected])

    def test_real_clustering_path_snaps_exactly(self, monkeypatch):
        """Through the real Leiden step, each landmark is the nearest cell to its centroid."""
        captured = {}
        real = kutils.find_optimal_resolution

        def spy(*args, **kwargs):
            resolution, partition = real(*args, **kwargs)
            captured["membership"] = np.array(partition.membership)
            return resolution, partition

        monkeypatch.setattr(kutils, "find_optimal_resolution", spy)
        X, _ = _clustered(3_000, 20, 25, seed=3)
        _, indices = find_landmarks(X, n_clusters=25, random_state=0)

        membership = captured["membership"]
        centroids = np.array(
            [X[membership == c].mean(axis=0) for c in np.unique(membership)]
        )
        np.testing.assert_array_equal(indices, _brute_force_nearest(X, centroids))

    def test_approximate_snap_is_an_explicit_opt_out(self, monkeypatch):
        """``exact_snap=False`` keeps the pre-0.9.0 index query and skips the exact search."""

        def boom(*args, **kwargs):
            raise AssertionError("exact search must not run when exact_snap=False")

        monkeypatch.setattr(kutils, "_exact_nearest_indices", boom)
        rng = np.random.default_rng(0)
        X = rng.normal(size=(300, 5))
        landmarks, indices = find_landmarks(
            X,
            n_clusters=10,
            random_state=0,
            exact_snap=False,
            knn_method="nndescent_query",
        )
        assert len(indices) == landmarks.shape[0]
        np.testing.assert_array_equal(landmarks, X[indices])

    def test_seeded_landmarks_are_deterministic(self):
        X, _ = _clustered(2_000, 10, 15, seed=4)
        first = find_landmarks(X, n_clusters=15, random_state=0)
        second = find_landmarks(X, n_clusters=15, random_state=0)
        np.testing.assert_array_equal(first[1], second[1])
        np.testing.assert_array_equal(first[0], second[0])


class TestExactNearestIndices:
    @pytest.mark.parametrize("dtype", [np.float64, np.float32])
    @pytest.mark.parametrize("chunk_size", [None, 1, 7, 512])
    def test_matches_brute_force(self, dtype, chunk_size):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(2_000, 12)).astype(dtype)
        queries = rng.normal(size=(40, 12)) * 0.3
        result = kutils._exact_nearest_indices(X, queries, chunk_size=chunk_size)
        np.testing.assert_array_equal(result, _brute_force_nearest(X, queries))
        assert result.dtype.kind == "i"

    def test_ties_go_to_the_lowest_index(self):
        """Duplicate rows are equally near; the lowest index wins, across chunks too."""
        rng = np.random.default_rng(2)
        base = rng.normal(size=(50, 6))
        X = np.concatenate([base, base, base])  # row i == row i+50 == row i+100
        perm = rng.permutation(len(X))
        X = X[perm]
        queries = base[:20] + 1e-3 * rng.normal(size=(20, 6))

        for chunk_size in (None, 1, 13, 64):
            result = kutils._exact_nearest_indices(X, queries, chunk_size=chunk_size)
            for j, q in enumerate(queries):
                d2 = ((X - q) ** 2).sum(axis=1)
                tied = np.flatnonzero(d2 == d2.min())
                assert len(tied) == 3
                assert result[j] == tied.min()

    def test_exact_ties_on_a_grid(self):
        """Integer coordinates make many distances exactly equal."""
        rng = np.random.default_rng(3)
        X = rng.integers(0, 3, size=(4_000, 3)).astype(np.float64)
        queries = rng.integers(0, 3, size=(30, 3)).astype(np.float64) + 0.5
        for chunk_size in (None, 37):
            result = kutils._exact_nearest_indices(X, queries, chunk_size=chunk_size)
            np.testing.assert_array_equal(result, _brute_force_nearest(X, queries))

    def test_deterministic_and_chunk_invariant(self):
        rng = np.random.default_rng(5)
        X = rng.normal(size=(5_000, 30))
        queries = rng.normal(size=(25, 30)) * 0.2
        reference = kutils._exact_nearest_indices(X, queries)
        for chunk_size in (None, 3, 100, 4_999, 5_000, 10_000):
            np.testing.assert_array_equal(
                kutils._exact_nearest_indices(X, queries, chunk_size=chunk_size),
                reference,
            )

    def test_far_from_origin(self):
        """Large coordinates relative to the spread stress the rounding bound."""
        rng = np.random.default_rng(6)
        X = 1e4 + rng.normal(size=(3_000, 40))
        queries = X[rng.choice(3_000, 20, replace=False)] + 0.05 * rng.normal(
            size=(20, 40)
        )
        result = kutils._exact_nearest_indices(X, queries, chunk_size=128)
        np.testing.assert_array_equal(result, _brute_force_nearest(X, queries))

    def test_near_duplicate_cells(self):
        """Cells closer together than the matrix-product score can resolve.

        The fast score ranks these copies by rounding noise alone, so the result
        matches brute force only if the shortlist keeps every near-tie and the
        shortlist is then ranked by the direct sum.
        """
        rng = np.random.default_rng(0)
        base = rng.normal(size=(500, 20)) * 50
        X = np.concatenate(
            [base, base + 1e-12 * rng.normal(size=base.shape), base * (1 + 1e-15)]
        )
        X = X[rng.permutation(len(X))]
        queries = base[:40] + 1e-10 * rng.normal(size=(40, 20))
        result = kutils._exact_nearest_indices(X, queries, chunk_size=97)
        np.testing.assert_array_equal(result, _brute_force_nearest(X, queries))

    def test_query_at_the_midpoint_of_two_cells(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(300, 10))
        pairs = rng.choice(300, (40, 2), replace=False)
        queries = (X[pairs[:, 0]] + X[pairs[:, 1]]) / 2
        result = kutils._exact_nearest_indices(X, queries, chunk_size=11)
        np.testing.assert_array_equal(result, _brute_force_nearest(X, queries))

    def test_tiny_spread_far_from_origin(self):
        """Score cancellation: a spread of 1e-3 at an offset of 1e7."""
        rng = np.random.default_rng(8)
        X = 1e7 + 1e-3 * rng.normal(size=(3_000, 64))
        queries = X[:20] + 1e-5 * rng.normal(size=(20, 64))
        result = kutils._exact_nearest_indices(X, queries, chunk_size=500)
        np.testing.assert_array_equal(result, _brute_force_nearest(X, queries))

    def test_rejects_bad_input(self):
        X = np.zeros((10, 3))
        with pytest.raises(ValueError):
            kutils._exact_nearest_indices(X, np.zeros((2, 4)))
        with pytest.raises(ValueError):
            kutils._exact_nearest_indices(np.zeros((0, 3)), np.zeros((2, 3)))
        with pytest.raises(ValueError):
            kutils._exact_nearest_indices(X, np.full((1, 3), np.nan))
        assert kutils._exact_nearest_indices(X, np.zeros((0, 3))).shape == (0,)
