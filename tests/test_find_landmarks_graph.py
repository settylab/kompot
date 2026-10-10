"""The kNN graph and row-order handling behind ``find_landmarks``.

Through 0.9.0 ``build_graph`` built a pynndescent index and then queried it
for every training point. On large data with low intrinsic dimension and
varying density that query takes orders of magnitude longer than the build.
The default graph is now the one the build produces, ``knn_method="exact"``
gives a deterministic kd-tree kNN, ``order_invariant=True`` makes the
landmarks independent of row order, and the edge list is built without a
Python loop.
"""

import numpy as np
import pynndescent
import pytest

import kompot.utils as kutils
from kompot.utils import build_graph, find_landmarks


def _mixture(n, d, k, seed):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(k, d)) * 4.0
    labels = rng.integers(0, k, n)
    return centres[labels] + rng.normal(size=(n, d))


def _brute_force_knn(X, k):
    """Reference: full float64 distance matrix, k smallest per row in distance order."""
    sq = ((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2)
    return np.argsort(sq, axis=1, kind="stable")[:, :k]


def _loop_edges(indices):
    """The pre-0.10 edge construction, kept verbatim as the reference."""
    edges = []
    for i in range(indices.shape[0]):
        for j in indices[i]:
            if i != j:
                edges.append((i, j))
    return edges


class TestExactGraph:
    def test_exact_graph_matches_brute_force(self):
        """``knn_method="exact"`` equals a brute-force kNN, edge for edge.

        3 000 points in 30 dimensions with k = 15: here the pynndescent graph
        misses some true neighbors, so this fails on an approximate graph.
        """
        rng = np.random.default_rng(0)
        X = rng.normal(size=(3000, 30))
        k = 15

        edges, index = build_graph(X, n_neighbors=k, knn_method="exact")

        expected = np.asarray(_loop_edges(_brute_force_knn(X, k)))
        np.testing.assert_array_equal(np.asarray(edges), expected)
        assert index is None

    def test_exact_graph_does_not_depend_on_thread_count(self, monkeypatch):
        X = _mixture(4000, 12, 10, seed=1)
        monkeypatch.setattr(kutils, "_available_cpus", lambda: 1)
        one, _ = build_graph(X, n_neighbors=15, knn_method="exact")
        monkeypatch.setattr(kutils, "_available_cpus", lambda: 4)
        four, _ = build_graph(X, n_neighbors=15, knn_method="exact")
        np.testing.assert_array_equal(one, four)

    def test_default_graph_is_the_nndescent_construction_graph(self):
        """By default the graph is the index's own neighbor graph, with no re-query.

        On 3 000 points in 30 dimensions the construction graph and a query of
        the index differ, so this fails on the 0.9.0 re-query.
        """
        X = np.random.default_rng(0).normal(size=(3000, 30))

        edges, index = build_graph(X, n_neighbors=15, random_state=42)

        ref = pynndescent.NNDescent(X, n_neighbors=15, random_state=42)
        assert isinstance(index, pynndescent.NNDescent)
        np.testing.assert_array_equal(
            np.asarray(edges), np.asarray(_loop_edges(ref.neighbor_graph[0]))
        )

    def test_nndescent_query_reproduces_the_old_graph(self):
        """``knn_method="nndescent_query"`` returns exactly the graph 0.9.0 built."""
        X = _mixture(2000, 10, 8, seed=2)

        edges, index = build_graph(
            X, n_neighbors=15, random_state=42, knn_method="nndescent_query"
        )

        old_index = pynndescent.NNDescent(X, n_neighbors=15, random_state=42)
        old_indices, _ = old_index.query(X, k=15)
        assert isinstance(index, pynndescent.NNDescent)
        np.testing.assert_array_equal(
            np.asarray(edges), np.asarray(_loop_edges(old_indices))
        )

    def test_rejects_unknown_method_and_caps_neighbors(self):
        X = np.random.default_rng(3).normal(size=(10, 3))
        with pytest.raises(ValueError, match="knn_method"):
            build_graph(X, knn_method="annoy")
        with pytest.raises(ValueError, match="n_neighbors"):
            build_graph(X, n_neighbors=0)
        # More neighbors than samples: every other sample is a neighbor.
        edges, _ = build_graph(X, n_neighbors=15, knn_method="exact")
        assert len(edges) == 10 * 9


class TestVectorisedEdges:
    def test_edges_equal_the_loop(self):
        """Same edges, same order, as the loop, including rows without a self entry."""
        rng = np.random.default_rng(4)
        n, k = 500, 7
        indices = rng.integers(0, n, size=(n, k))
        # Self at the front of most rows, somewhere else in some, absent in a few.
        indices[:400, 0] = np.arange(400)
        indices[400:450, 3] = np.arange(400, 450)

        got = kutils._knn_edges(indices)

        expected = np.asarray(_loop_edges(indices))
        assert got.shape == expected.shape
        np.testing.assert_array_equal(got, expected)

    def test_padding_entries_are_dropped(self):
        """pynndescent pads rows with -1 when it finds fewer neighbors than asked for."""
        indices = np.array([[0, 1, -1], [1, 0, -1], [2, -1, -1]])
        got = kutils._knn_edges(indices)
        np.testing.assert_array_equal(got, [[0, 1], [1, 0]])


class TestOrderInvariant:
    def test_landmarks_do_not_depend_on_row_order(self):
        """Three row permutations give the same landmarks, as rows of the original."""
        X = _mixture(3000, 8, 12, seed=5)
        ref_landmarks, ref_idx = find_landmarks(
            X, n_clusters=40, random_state=0, order_invariant=True
        )
        rng = np.random.default_rng(6)
        for _ in range(3):
            perm = rng.permutation(X.shape[0])
            landmarks, idx = find_landmarks(
                X[perm], n_clusters=40, random_state=0, order_invariant=True
            )
            # idx indexes X[perm]; perm[idx] are the same rows of X.
            assert sorted(perm[idx].tolist()) == sorted(ref_idx.tolist())
            np.testing.assert_array_equal(landmarks, X[perm][idx])

    def test_indices_point_into_the_input(self):
        """The returned indices refer to the caller's row order, not the sorted one.

        On input already in the canonical order the sort is the identity, so
        that run's landmarks are the reference; a shuffled copy must give the
        same landmark coordinates, found at the returned rows of the shuffled
        input.
        """
        X = _mixture(1500, 5, 6, seed=7)
        X_sorted = X[np.lexsort(X.T[::-1])]
        X_shuffled = X_sorted[np.random.default_rng(8).permutation(len(X_sorted))]

        ref_landmarks, _ = find_landmarks(
            X_sorted, n_clusters=15, random_state=0, order_invariant=True
        )
        landmarks, idx = find_landmarks(
            X_shuffled, n_clusters=15, random_state=0, order_invariant=True
        )

        np.testing.assert_array_equal(landmarks, X_shuffled[idx])

        def rows(a):
            return a[np.lexsort(a.T[::-1])]

        np.testing.assert_array_equal(rows(landmarks), rows(ref_landmarks))

    def test_requires_a_seed(self):
        X = _mixture(200, 4, 3, seed=8)
        with pytest.raises(ValueError, match="random_state"):
            find_landmarks(X, n_clusters=5, order_invariant=True)

    def test_uses_the_exact_graph(self, monkeypatch):
        """``order_invariant`` selects the exact graph and refuses any other."""
        X = _mixture(200, 4, 3, seed=8)
        for method in ("nndescent", "nndescent_query"):
            with pytest.raises(ValueError, match="knn_method='exact'"):
                find_landmarks(
                    X,
                    n_clusters=5,
                    random_state=0,
                    order_invariant=True,
                    knn_method=method,
                )

        seen = []
        real = kutils.build_graph

        def spy(*args, **kwargs):
            seen.append(kwargs.get("knn_method"))
            return real(*args, **kwargs)

        monkeypatch.setattr(kutils, "build_graph", spy)
        find_landmarks(X, n_clusters=5, random_state=0, order_invariant=True)
        find_landmarks(X, n_clusters=5, random_state=0)
        assert seen == ["exact", "nndescent"]


def test_approximate_snap_needs_a_pynndescent_index():
    """``exact_snap=False`` uses the pynndescent index, which the exact graph does not build."""
    X = _mixture(200, 4, 3, seed=9)
    with pytest.raises(ValueError, match="knn_method='nndescent_query'"):
        find_landmarks(
            X, n_clusters=5, random_state=0, exact_snap=False, knn_method="exact"
        )
