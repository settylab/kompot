"""Regression guards for the sample-variance cost defects fixed in 0.9.0.

Three distinct failures, all of which were silent — the code returned a
plausible answer at rc 0 and only a measurement disagreed:

* ``settylab/kompot#25`` — ``kompot.de(dry_run=True)`` priced a run with zero
  null genes while the run it was pricing would add 2 000.
* ``settylab/kompot#26`` — the per-gene covariance tensors were summed into a
  third dense tensor before use, so ``store_arrays_on_disk=True`` did not keep
  them out of memory, and ``compute_mahalanobis_distances`` materialised the
  whole tensor again into a variable it never read.
* The estimator charged that third tensor, and charged disk for the Dask path
  that writes nothing.
"""

import io
import contextlib
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import kompot
from kompot.utils import LazyGeneCovariance


def _adata(n_cells=240, n_genes=12, n_samples=3, n_dims=6, seed=0):
    rng = np.random.default_rng(seed)
    obs = pd.DataFrame(
        {
            "condition": np.where(np.arange(n_cells) < n_cells // 2, "A", "B"),
            "donor": [f"d{i % n_samples}" for i in range(n_cells)],
        },
        index=[f"c{i}" for i in range(n_cells)],
    )
    adata = ad.AnnData(
        X=rng.normal(size=(n_cells, n_genes)),
        obs=obs,
        var=pd.DataFrame(index=[f"g{i}" for i in range(n_genes)]),
    )
    adata.obsm["DM_EigenVectors"] = rng.normal(size=(n_cells, n_dims))
    return adata


def _plan(adata, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return kompot.de(
            adata,
            groupby="condition",
            condition1="A",
            condition2="B",
            obsm_key="DM_EigenVectors",
            dry_run=True,
            **kwargs,
        )


def _named(plan, needle):
    return [r for r in plan.requirements if needle in r.name]


# --------------------------------------------------------------------------
# settylab/kompot#25 — the dry run must price the run it is pricing
# --------------------------------------------------------------------------


def test_dry_run_resolves_auto_null_genes():
    """A default dry run must account for the 2 000 null genes the run adds.

    ``null_genes="auto"`` used to reach the estimator unresolved, where it
    matched neither the int nor the list branch and silently counted zero.
    """
    adata = _adata()
    auto = _plan(adata)  # FDRSettings default -> "auto"
    explicit = _plan(adata, fdr=kompot.FDRSettings(null_genes=2000))

    assert auto.total_memory_required == explicit.total_memory_required
    fold_change = _named(auto, "Fold change")[0]
    assert fold_change.shape == (adata.n_obs, adata.n_vars + 2000)


def test_dry_run_auto_null_genes_is_zero_with_sample_col():
    """With ``sample_col`` set, ``"auto"`` resolves to 0, so no null genes."""
    adata = _adata()
    auto = _plan(adata, sample_col="donor")
    explicit = _plan(adata, sample_col="donor", fdr=kompot.FDRSettings(null_genes=0))

    assert auto.total_memory_required == explicit.total_memory_required
    assert _named(auto, "Fold change")[0].shape == (adata.n_obs, adata.n_vars)


def test_estimator_refuses_unresolved_null_genes():
    """The estimator must refuse a sentinel it cannot price, not count zero."""
    from kompot.resource_estimation import estimate_differential_expression_resources

    adata = _adata()
    with pytest.raises(ValueError, match="must be resolved"):
        estimate_differential_expression_resources(
            adata, "A", "B", "condition", null_genes="auto"
        )


# --------------------------------------------------------------------------
# settylab/kompot#26 — the per-gene tensor is assembled one gene at a time
# --------------------------------------------------------------------------


def test_lazy_gene_covariance_matches_eager_sum():
    """The view must be numerically identical to the dense sum it replaces."""
    rng = np.random.default_rng(1)
    n_points, n_genes = 5, 4
    a = rng.normal(size=(n_points, n_points, n_genes))
    b = rng.normal(size=(n_points, n_points, n_genes))
    base = rng.normal(size=(n_points, n_points))

    view = LazyGeneCovariance([a, b], base=base)
    eager = a + b + base[:, :, None]

    assert view.shape == (n_points, n_points, n_genes)
    assert view.ndim == 3
    for g in range(n_genes):
        np.testing.assert_allclose(view[:, :, g], eager[:, :, g], rtol=0, atol=0)


def test_lazy_gene_covariance_does_not_mutate_its_terms():
    """A NumPy basic slice is a view; the accumulation must copy first."""
    rng = np.random.default_rng(2)
    a = rng.normal(size=(4, 4, 3))
    base = rng.normal(size=(4, 4))
    before = a.copy()

    view = LazyGeneCovariance([a], base=base)
    for g in range(3):
        view.gene(g)
        view.gene(g)  # twice: a second call must not accumulate again

    np.testing.assert_array_equal(a, before)
    np.testing.assert_allclose(view.gene(0), before[:, :, 0] + base)


def test_lazy_gene_covariance_rejects_bad_input():
    with pytest.raises(ValueError, match="at least one term"):
        LazyGeneCovariance([])
    with pytest.raises(ValueError, match="share a shape"):
        LazyGeneCovariance([np.zeros((3, 3, 2)), np.zeros((3, 3, 4))])
    with pytest.raises(ValueError, match="3-D"):
        LazyGeneCovariance([np.zeros((3, 3))])
    with pytest.raises(ValueError, match="base has shape"):
        LazyGeneCovariance([np.zeros((3, 3, 2))], base=np.zeros((4, 4)))
    with pytest.raises(TypeError, match="only"):
        LazyGeneCovariance([np.zeros((3, 3, 2))])[0]


def test_mahalanobis_accepts_lazy_view_and_matches_dense():
    """The consumer must give the same distances for a view and a dense tensor."""
    from kompot.utils import compute_mahalanobis_distances

    rng = np.random.default_rng(3)
    n_points, n_genes = 8, 5
    base = np.eye(n_points) * 2.0

    def spd():
        m = rng.normal(size=(n_points, n_points))
        return m @ m.T / n_points + np.eye(n_points)

    a = np.stack([spd() for _ in range(n_genes)], axis=2)
    b = np.stack([spd() for _ in range(n_genes)], axis=2)
    diffs = rng.normal(size=(n_genes, n_points))

    dense = a + b + base[:, :, None]
    view = LazyGeneCovariance([a, b], base=base)

    from_dense = compute_mahalanobis_distances(
        diffs, dense, jit_compile=False, progress=False
    )
    from_view = compute_mahalanobis_distances(
        diffs, view, jit_compile=False, progress=False
    )
    np.testing.assert_allclose(from_view, from_dense, rtol=1e-12, atol=0)


def test_de_with_sample_variance_produces_finite_scores():
    """End-to-end: the lazy path still yields a usable sample-variance column."""
    adata = _adata(n_cells=200, n_genes=8, n_samples=3)
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        kompot.de(
            adata,
            groupby="condition",
            condition1="A",
            condition2="B",
            obsm_key="DM_EigenVectors",
            sample_col="donor",
            gp=kompot.GPSettings(n_landmarks=40, random_state=0),
            fdr=kompot.FDRSettings(null_genes=0),
            output=kompot.OutputSettings(progress=False),
        )
    col = "kompot_de_A_to_B_mahalanobis_sample_var"
    assert col in adata.var
    assert np.isfinite(adata.var[col].to_numpy()).all()


# --------------------------------------------------------------------------
# The plan must describe the code that now runs
# --------------------------------------------------------------------------


def test_plan_no_longer_charges_a_third_covariance_tensor():
    """In-memory mode holds variance1 + variance2, and nothing else.

    The expected figure is re-derived from the allocation, not copied from a
    previous run: two (n_landmarks, n_landmarks, n_genes) float64 tensors.
    """
    adata = _adata(n_cells=200, n_genes=10, n_samples=3)
    plan = _plan(
        adata,
        sample_col="donor",
        gp=kompot.GPSettings(n_landmarks=40),
        storage=kompot.StorageSettings(store_arrays_on_disk=False),
    )

    assert _named(plan, "Combined sample covariances") == []
    sv = _named(plan, "Sample covariances")
    assert len(sv) == 1
    assert sv[0].resource_type == "memory"
    assert sv[0].size_bytes == 2 * 40 * 40 * adata.n_vars * 8


def test_plan_disk_column_matches_the_selected_backend():
    """Disk is charged only on the path that actually writes files.

    With dask the tensor is a lazy graph and nothing reaches
    ``disk_storage_dir``; without dask it is written as memory-mapped ``.npy``.
    """
    from kompot import resource_estimation

    adata = _adata(n_cells=200, n_genes=10, n_samples=3)
    kwargs = dict(
        sample_col="donor",
        gp=kompot.GPSettings(n_landmarks=40),
        storage=kompot.StorageSettings(store_arrays_on_disk=True),
    )
    expected = 2 * 40 * 40 * adata.n_vars * 8

    original = resource_estimation.DASK_AVAILABLE
    try:
        resource_estimation.DASK_AVAILABLE = True
        with_dask = _plan(adata, **kwargs)
        resource_estimation.DASK_AVAILABLE = False
        without_dask = _plan(adata, **kwargs)
    finally:
        resource_estimation.DASK_AVAILABLE = original

    assert with_dask.total_disk_required == 0
    assert _named(with_dask, "Sample covariances") == []

    disk_reqs = [r for r in without_dask.requirements if r.resource_type == "disk"]
    assert len(disk_reqs) == 1
    assert disk_reqs[0].size_bytes == expected
    assert without_dask.total_disk_required == expected


def test_plan_charges_the_per_gene_working_set():
    """Whatever the backend, one gene's matrices are held at a time."""
    adata = _adata(n_cells=200, n_genes=10, n_samples=3)
    for on_disk in (False, True):
        plan = _plan(
            adata,
            sample_col="donor",
            gp=kompot.GPSettings(n_landmarks=40),
            storage=kompot.StorageSettings(store_arrays_on_disk=on_disk),
        )
        working = _named(plan, "Per-gene covariance working set")
        assert len(working) == 1, on_disk
        assert working[0].resource_type == "memory"
        # two sample-variance terms + the shared posterior covariance
        assert working[0].size_bytes == 3 * 40 * 40 * 8


# --------------------------------------------------------------------------
# The restructured branches in DifferentialExpression.compute_mahalanobis
# --------------------------------------------------------------------------


def _spd(rng, n):
    m = rng.normal(size=(n, n))
    return m @ m.T / n + np.eye(n)


@pytest.mark.parametrize("n_terms", [1, 2])
def test_lazy_view_matches_dense_for_one_or_two_predictors(n_terms):
    """Both predictors, or only one, must give the dense arithmetic's answer.

    The fix replaced a nested if/else over (variance_predictor1,
    variance_predictor2) x (gene-specific, shared) with a flat list of terms.
    The single-predictor arms are reachable through ``ModelSettings`` and are
    easy to leave untested.
    """
    from kompot.utils import compute_mahalanobis_distances

    rng = np.random.default_rng(11)
    n_points, n_genes = 7, 4
    base = _spd(rng, n_points)
    terms = [
        np.stack([_spd(rng, n_points) for _ in range(n_genes)], axis=2)
        for _ in range(n_terms)
    ]
    diffs = rng.normal(size=(n_genes, n_points))

    dense = sum(terms) + base[:, :, None]
    view = LazyGeneCovariance(terms, base=base)

    from_view = compute_mahalanobis_distances(
        diffs, view, jit_compile=False, progress=False
    )
    from_dense = compute_mahalanobis_distances(
        diffs, dense, jit_compile=False, progress=False
    )
    np.testing.assert_allclose(from_view, from_dense, rtol=1e-12, atol=0)


def test_lazy_view_refuses_a_shared_term():
    """A 2-D sample variance is folded into ``base`` by the caller.

    It must never be handed to the view as a term; if it is, that is a bug in
    the caller and the view should say so rather than broadcast silently.
    """
    rng = np.random.default_rng(12)
    gene_specific = np.stack([_spd(rng, 5) for _ in range(3)], axis=2)
    shared = _spd(rng, 5)

    with pytest.raises(ValueError, match="share a shape"):
        LazyGeneCovariance([gene_specific, shared])
