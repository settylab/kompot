"""Tests for kompot.plot.dotplot."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

try:
    from anndata import AnnData

    _has_anndata = True
except ImportError:
    _has_anndata = False

pytestmark = pytest.mark.skipif(
    not _has_anndata, reason="anndata is required for these tests"
)


def _make_adata(n_obs: int = 90, n_vars: int = 12, seed: int = 0) -> "AnnData":
    """Minimal AnnData with an LFC layer, an expression layer, and a
    Mahalanobis-like ranking column in var."""
    rng = np.random.default_rng(seed)
    expr = np.clip(rng.normal(size=(n_obs, n_vars)) + 0.5, 0.0, None)
    var_names = [f"G{i}" for i in range(n_vars)]
    obs_names = [f"C{i}" for i in range(n_obs)]
    var = pd.DataFrame(
        {
            "kompot_de_cond_mahalanobis": rng.uniform(0.0, 10.0, size=n_vars),
            "kompot_de_cond_is_de": rng.uniform(size=n_vars) > 0.5,
        },
        index=var_names,
    )
    obs = pd.DataFrame(
        {"celltype": pd.Categorical(rng.choice(list("ABC"), size=n_obs))},
        index=obs_names,
    )
    adata = AnnData(X=expr, obs=obs, var=var)
    adata.layers["lfc"] = rng.normal(size=(n_obs, n_vars)).astype(float)
    adata.layers["logcounts"] = expr.astype(float)
    return adata


def test_dotplot_import_and_exported():
    import kompot.plot as kp

    assert hasattr(kp, "dotplot")
    assert "dotplot" in kp.__all__


def test_dotplot_reuses_heatmap_primitives():
    """Sanity-check that the heatmap.utils reuse surface is wired up.

    This is load-bearing: if someone renames the helpers without updating
    dotplot, the import will fail here long before the plot does.
    """
    import sys
    import kompot.plot  # noqa: F401 — force module registration

    dp_mod = sys.modules["kompot.plot.dotplot"]

    from kompot.plot.heatmap.utils import (
        _get_expression_matrix,
        _infer_score_key,
        _prepare_gene_list,
        _setup_colormap_normalization,
    )

    assert dp_mod._get_expression_matrix is _get_expression_matrix
    assert dp_mod._infer_score_key is _infer_score_key
    assert dp_mod._prepare_gene_list is _prepare_gene_list
    assert dp_mod._setup_colormap_normalization is _setup_colormap_normalization


def test_dotplot_standalone_returns_figure():
    from kompot.plot import dotplot

    adata = _make_adata()
    genes = list(adata.var_names[:5])
    fig = dotplot(
        adata,
        genes=genes,
        groupby="celltype",
        lfc_layer="lfc",
        expr_layer="logcounts",
        return_fig=True,
    )
    assert isinstance(fig, plt.Figure)
    # main + colorbar + size legend (>= 3; matplotlib may add a colorbar ax)
    assert len(fig.axes) >= 3
    ax_main = fig.axes[0]
    # main axis renders one scatter collection with 5 * 3 = 15 offsets
    assert len(ax_main.collections) >= 1
    assert ax_main.collections[0].get_offsets().shape[0] == len(genes) * 3
    plt.close(fig)


def test_dotplot_standalone_without_return_fig_returns_none():
    from kompot.plot import dotplot

    adata = _make_adata()
    out = dotplot(
        adata,
        genes=list(adata.var_names[:4]),
        groupby="celltype",
        lfc_layer="lfc",
        expr_layer="logcounts",
        return_fig=False,
    )
    assert out is None
    plt.close("all")


def test_dotplot_embeds_into_provided_axes():
    from kompot.plot import dotplot

    adata = _make_adata()
    fig = plt.figure(figsize=(6, 3), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.14])
    ax_main = fig.add_subplot(gs[0, 0])
    inner = gs[0, 1].subgridspec(2, 1, height_ratios=[1.0, 1.0])
    ax_size = fig.add_subplot(inner[0, 0])
    ax_cbar = fig.add_subplot(inner[1, 0])

    figs_before = list(plt.get_fignums())
    axes_before = list(fig.axes)

    out = dotplot(
        adata,
        genes=list(adata.var_names[:4]),
        groupby="celltype",
        lfc_layer="lfc",
        expr_layer="logcounts",
        axes=(ax_main, ax_cbar, ax_size),
    )
    # caller owns the figure; nothing returned
    assert out is None
    # no new Figure was created by the function
    assert plt.get_fignums() == figs_before
    # provided axes are still present and the main axis now has a scatter
    for ax in axes_before:
        assert ax in fig.axes
    assert len(ax_main.collections) >= 1
    assert ax_main.collections[0].get_offsets().shape[0] == 4 * 3
    plt.close(fig)


def test_dotplot_rejects_bad_axes_length():
    from kompot.plot import dotplot

    adata = _make_adata()
    fig, (ax1, ax2) = plt.subplots(1, 2)
    with pytest.raises(ValueError, match="3-tuple"):
        dotplot(
            adata,
            genes=list(adata.var_names[:3]),
            groupby="celltype",
            lfc_layer="lfc",
            expr_layer="logcounts",
            axes=(ax1, ax2),
        )
    plt.close(fig)


def test_dotplot_auto_picks_top_n_by_score():
    from kompot.plot import dotplot

    adata = _make_adata()
    n = 4
    fig = dotplot(
        adata,
        genes=None,
        groupby="celltype",
        lfc_layer="lfc",
        expr_layer="logcounts",
        score_key="kompot_de_cond_mahalanobis",
        n_top=n,
        return_fig=True,
    )
    assert isinstance(fig, plt.Figure)
    ax_main = fig.axes[0]
    ytl = [t.get_text() for t in ax_main.get_yticklabels()]
    assert len(ytl) == n
    expected = list(
        adata.var["kompot_de_cond_mahalanobis"]
        .sort_values(ascending=False)
        .head(n)
        .index
    )
    # order matters — top-ranked gene sits at the top of the y-axis
    assert ytl == expected
    plt.close(fig)


def test_dotplot_auto_pick_honors_filter_key():
    from kompot.plot import dotplot

    adata = _make_adata()
    eligible = adata.var.index[adata.var["kompot_de_cond_is_de"].astype(bool)]
    assert len(eligible) >= 3
    fig = dotplot(
        adata,
        genes=None,
        groupby="celltype",
        lfc_layer="lfc",
        expr_layer="logcounts",
        score_key="kompot_de_cond_mahalanobis",
        filter_key="kompot_de_cond_is_de",
        n_top=min(3, len(eligible)),
        return_fig=True,
    )
    ax_main = fig.axes[0]
    ytl = [t.get_text() for t in ax_main.get_yticklabels()]
    assert set(ytl).issubset(set(eligible))
    plt.close(fig)


def test_dotplot_auto_pick_without_score_raises():
    from kompot.plot import dotplot

    # Strip Mahalanobis-like columns so field inference has nothing to latch
    # onto — confirms we raise a helpful error rather than silently picking
    # random columns.
    adata = _make_adata()
    adata.var = adata.var.drop(
        columns=[c for c in adata.var.columns if "mahalanobis" in c]
    )
    with pytest.raises(ValueError, match="Mahalanobis"):
        dotplot(
            adata,
            genes=None,
            groupby="celltype",
            lfc_layer="lfc",
            expr_layer="logcounts",
        )
    plt.close("all")


def test_dotplot_missing_lfc_layer_raises():
    from kompot.plot import dotplot

    adata = _make_adata()
    with pytest.raises(ValueError, match="fold-change layer"):
        dotplot(
            adata,
            genes=list(adata.var_names[:3]),
            groupby="celltype",
            expr_layer="logcounts",
        )
    plt.close("all")


def test_dotplot_symmetric_color_scale_about_zero():
    from kompot.plot import dotplot

    adata = _make_adata()
    # skew the LFC layer strongly positive and stick a few extreme outliers
    # in so the 98th-percentile cap matters.
    rng = np.random.default_rng(1)
    lfc = rng.normal(loc=2.0, scale=0.5, size=adata.shape)
    lfc[:4, 0] = 100.0
    adata.layers["lfc"] = lfc

    fig = dotplot(
        adata,
        genes=list(adata.var_names[:5]),
        groupby="celltype",
        lfc_layer="lfc",
        expr_layer="logcounts",
        vabs_pct=98.0,
        return_fig=True,
    )
    vmin, vmax = fig.axes[0].collections[0].get_clim()
    assert vmax > 0
    assert vmin == pytest.approx(-vmax)
    # clip should be below the 100.0 outliers (98th-pct cap)
    assert vmax < 100.0
    plt.close(fig)


def test_dotplot_vabs_min_floors_scale():
    from kompot.plot import dotplot

    adata = _make_adata()
    adata.layers["lfc"] = np.zeros(adata.shape, dtype=float)
    fig = dotplot(
        adata,
        genes=list(adata.var_names[:3]),
        groupby="celltype",
        lfc_layer="lfc",
        expr_layer="logcounts",
        vabs_min=0.5,
        return_fig=True,
    )
    vmin, vmax = fig.axes[0].collections[0].get_clim()
    assert vmax == pytest.approx(0.5)
    assert vmin == pytest.approx(-0.5)
    plt.close(fig)


def test_dotplot_vmax_overrides_percentile():
    from kompot.plot import dotplot

    adata = _make_adata()
    fig = dotplot(
        adata,
        genes=list(adata.var_names[:3]),
        groupby="celltype",
        lfc_layer="lfc",
        expr_layer="logcounts",
        vmax=2.5,
        return_fig=True,
    )
    vmin, vmax = fig.axes[0].collections[0].get_clim()
    assert vmax == pytest.approx(2.5)
    assert vmin == pytest.approx(-2.5)
    plt.close(fig)


def test_dotplot_min_cells_drops_rare_categories():
    from kompot.plot import dotplot

    adata = _make_adata()
    # Force exactly one 'C' cell so min_cells=5 drops the category.
    ct = adata.obs["celltype"].astype(str).copy()
    ct.iloc[:] = "A"
    ct.iloc[:30] = "B"
    ct.iloc[0] = "C"
    adata.obs["celltype"] = pd.Categorical(ct, categories=["A", "B", "C"])

    fig = dotplot(
        adata,
        genes=list(adata.var_names[:3]),
        groupby="celltype",
        lfc_layer="lfc",
        expr_layer="logcounts",
        min_cells=5,
        return_fig=True,
    )
    ax_main = fig.axes[0]
    xtl = [t.get_text() for t in ax_main.get_xticklabels()]
    assert xtl == ["A", "B"]
    plt.close(fig)


def test_dotplot_categories_order_filters_and_orders():
    from kompot.plot import dotplot

    adata = _make_adata()
    fig = dotplot(
        adata,
        genes=list(adata.var_names[:3]),
        groupby="celltype",
        lfc_layer="lfc",
        expr_layer="logcounts",
        categories_order=["C", "A"],
        return_fig=True,
    )
    xtl = [t.get_text() for t in fig.axes[0].get_xticklabels()]
    assert xtl == ["C", "A"]
    plt.close(fig)


def test_dotplot_missing_gene_raises():
    from kompot.plot import dotplot

    adata = _make_adata()
    with pytest.raises(KeyError):
        dotplot(
            adata,
            genes=["G0", "NOT_A_GENE"],
            groupby="celltype",
            lfc_layer="lfc",
            expr_layer="logcounts",
        )
    plt.close("all")
