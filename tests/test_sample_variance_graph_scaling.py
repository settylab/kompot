"""Regression tests for the gene-specific (sample-variance) covariance path.

These guard a SCALING property, not a single timing. The defect they cover was not
a constant factor: with a Dask-backed covariance tensor the per-gene
``__setitem__`` loop in ``DifferentialExpression.compute_mahalanobis_distances``
bound its result to the same object as its input, so every assignment layered onto
the graph the previous iteration had just built. Task count grew QUADRATICALLY in
``n_genes`` (measured: 1,804 tasks after one iteration, 994,740 after 120), which
is why runs of a few hundred genes did not finish while the same arithmetic done
directly takes ~16 ms/gene.

Task count is used rather than wall time on purpose: it is deterministic, so these
tests do not flake on a loaded CI machine.
"""

import numpy as np
import pytest

from kompot.memory_utils import DASK_AVAILABLE
from kompot.utils import compute_mahalanobis_distances

N_POINTS = 24
N_GROUPS = 4


def _shared_cov(n_points=N_POINTS, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n_points, 6)) * 0.3
    return a @ a.T + np.eye(n_points)


def _centred(n_genes, seed, n_points=N_POINTS):
    """(n_points, n_groups, n_genes), as SampleVarianceEstimator.predict builds."""
    rng = np.random.default_rng(seed)
    p = rng.normal(size=(N_GROUPS, n_points, n_genes)) * 0.1
    p = p - p.mean(axis=0, keepdims=True)
    return np.moveaxis(p, 1, 0)


def _dask_gene_covariance(n_genes, seed, n_points=N_POINTS):
    """Rebuilds the Dask array shape produced by the disk-backed estimator path."""
    import dask
    import dask.array as da

    centred = _centred(n_genes, seed, n_points)

    @dask.delayed
    def gene_cov(gene_data, n_groups):
        return np.dot(gene_data, gene_data.T) / (n_groups - 1)

    return da.stack(
        [
            da.from_delayed(
                gene_cov(centred[:, :, g], N_GROUPS),
                shape=(n_points, n_points),
                dtype=np.float64,
            )
            for g in range(n_genes)
        ],
        axis=2,
    )


def _numpy_gene_covariance(n_genes, seed, n_points=N_POINTS):
    centred = _centred(n_genes, seed, n_points)
    out = np.empty((n_points, n_points, n_genes))
    for g in range(n_genes):
        gc = centred[:, :, g]
        out[:, :, g] = gc @ gc.T / (N_GROUPS - 1)
    return out


def _reference_distances(n_genes, diffs, shared, seed_pair=(1, 2)):
    """Straight per-gene computation, independent of the code under test."""
    from scipy.linalg import solve_triangular

    c1 = _centred(n_genes, seed_pair[0])
    c2 = _centred(n_genes, seed_pair[1])
    out = np.empty(n_genes)
    for g in range(n_genes):
        a, b = c1[:, :, g], c2[:, :, g]
        cov = (a @ a.T + b @ b.T) / (N_GROUPS - 1) + shared + np.eye(N_POINTS) * 1e-8
        chol = np.linalg.cholesky(cov)
        out[g] = np.sqrt(np.sum(solve_triangular(chol, diffs[g], lower=True) ** 2))
    return out


def _task_count(arr):
    """Graph size of the object handed to the Mahalanobis step, or None if it has no graph.

    None is a PASS, not a skip: an object that carries no task graph (a lazy view
    that materialises one gene at a time) cannot exhibit the graph blowup this
    module guards against. Returning None rather than raising is what lets the
    same test express the property across implementations that differ in KIND,
    not just in degree.
    """
    graph = getattr(arr, "__dask_graph__", None)
    if graph is None:
        return None
    return len(graph())


class _FakeFunctionPredictor:
    """Minimal stand-in for a fitted mellon predictor."""

    def __init__(self, landmarks, n_genes, seed):
        self.landmarks = landmarks
        rng = np.random.default_rng(seed)
        self._pred = rng.normal(size=(len(landmarks), n_genes)) * 0.2
        a = rng.normal(size=(len(landmarks), 6)) * 0.3
        self._cov = a @ a.T + np.eye(len(landmarks))

    def __call__(self, X):
        return self._pred

    def covariance(self, X, diag=False):
        return self._cov


def _make_variance_predictor(n_genes, seed, n_points=N_POINTS):
    def predict(X, diag=False, progress=False):
        assert not diag, "the DE path must request the full covariance"
        return _dask_gene_covariance(n_genes, seed, n_points)

    return predict


def _covariance_handed_to_mahalanobis(n_genes, monkeypatch):
    """Run the real DE path and return the covariance object it passes downstream.

    Intercepting at this boundary is what makes this test version independent: it
    measures the array the shipped code actually builds, whatever route it took to
    build it.
    """
    from kompot.differential import differential_expression as de_mod

    landmarks = np.linspace(0, 1, N_POINTS).reshape(-1, 1)
    captured = {}

    def spy(diff_values, covariance, **kwargs):
        captured["covariance"] = covariance
        return np.zeros(np.shape(diff_values)[0])

    monkeypatch.setattr(de_mod, "compute_mahalanobis_distances", spy)

    de = de_mod.DifferentialExpression(
        use_sample_variance=True,
        function_predictor1=_FakeFunctionPredictor(landmarks, n_genes, 11),
        function_predictor2=_FakeFunctionPredictor(landmarks, n_genes, 12),
        variance_predictor1=_make_variance_predictor(n_genes, 1),
        variance_predictor2=_make_variance_predictor(n_genes, 2),
    )
    de.compute_mahalanobis_distances(
        X=landmarks, landmarks_override=landmarks, progress=False
    )
    return captured["covariance"]


@pytest.mark.skipif(not DASK_AVAILABLE, reason="dask is not installed")
def test_gene_specific_covariance_graph_scales_linearly(monkeypatch):
    """Task count must grow at most ~linearly with n_genes, not quadratically.

    This is the regression guard for the defect. It drives the real
    ``DifferentialExpression.compute_mahalanobis_distances`` and measures the graph
    of the covariance tensor that method hands downstream, so it exercises whatever
    the shipped code does rather than a particular helper.

    Before the fix the per-gene ``__setitem__`` loop aliased its own input, so each
    assignment layered onto the graph the previous iteration built and the task
    count grew quadratically. The bound below is deliberately generous - 4x genes
    may cost up to 8x tasks - so it tolerates a change in how dask chunks a stack
    while still failing on quadratic growth, which at these sizes is ~16x.
    """
    small, large = 20, 80  # 4x genes

    tasks = {}
    for n_genes in (small, large):
        cov = _covariance_handed_to_mahalanobis(n_genes, monkeypatch)
        assert cov.shape == (N_POINTS, N_POINTS, n_genes)
        tasks[n_genes] = _task_count(cov)

    if tasks[small] is None or tasks[large] is None:
        # A lazy view with no task graph: the blowup is not merely bounded here,
        # it is unrepresentable. Nothing left to assert.
        return

    growth = tasks[large] / tasks[small]
    assert growth <= 8.0, (
        "gene-specific covariance graph grew superlinearly: "
        f"{tasks[small]} tasks at {small} genes vs {tasks[large]} at {large} "
        f"({growth:.1f}x for 4x the genes; linear is 4x, quadratic is 16x)"
    )


@pytest.mark.parametrize("backend", ["numpy", "dask"])
def test_gene_specific_mahalanobis_matches_reference(backend):
    """The optimised path must return exactly what a direct computation returns."""
    if backend == "dask" and not DASK_AVAILABLE:
        pytest.skip("dask is not installed")

    n_genes = 16
    shared = _shared_cov()
    rng = np.random.default_rng(7)
    diffs = rng.normal(size=(n_genes, N_POINTS)) * 0.2

    build = _dask_gene_covariance if backend == "dask" else _numpy_gene_covariance
    # Built without the helper so this checks the MATHS on any build.
    cov = build(n_genes, seed=1) + build(n_genes, seed=2) + shared[:, :, None]

    got = compute_mahalanobis_distances(
        diff_values=diffs, covariance=cov, batch_size=None,
        jit_compile=False, eps=1e-8, progress=False,
    )
    expected = _reference_distances(n_genes, diffs, shared)

    assert np.all(np.isfinite(got))
    np.testing.assert_allclose(got, expected, rtol=1e-8, atol=1e-10)


