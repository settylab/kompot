"""Dotplot of kompot differential expression results.

Ax-embeddable fold-change-per-group dotplot. Color encodes the mean of a
per-cell log fold-change layer within each ``groupby`` category; size
encodes the fraction of cells in each category whose expression exceeds a
threshold. Gene selection is either an explicit list or auto-picked by
top-N Mahalanobis from a kompot DE run.

Unlike :func:`scanpy.pl.DotPlot`, this function does not build its own
``GridSpec`` and composes cleanly into externally-provided axes. Pass
``axes=(main, cbar, size_legend)`` to embed the dotplot into a composite
figure, or leave ``axes=None`` for a standalone figure.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence, Tuple

import numpy as np
from anndata import AnnData

logger = logging.getLogger("kompot")

try:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator
except ImportError as e:
    raise ImportError(
        "matplotlib is required for plotting: pip install matplotlib"
    ) from e


def _resolve_layer(adata: AnnData, layer: Optional[str]) -> np.ndarray:
    """Dense (n_obs, n_vars) matrix from ``adata.layers[layer]`` or ``adata.X``."""
    if layer is not None:
        if layer not in adata.layers:
            raise KeyError(
                f"layer '{layer}' not found in adata.layers "
                f"(available: {list(adata.layers)})"
            )
        mat = adata.layers[layer]
    else:
        mat = adata.X
    if hasattr(mat, "toarray"):
        mat = mat.toarray()
    return np.asarray(mat)


def _group_mean_matrix(
    values: np.ndarray,
    obs_labels: np.ndarray,
    categories: Sequence[str],
) -> np.ndarray:
    """Per-category mean of an ``(n_obs, n_genes)`` matrix."""
    out = np.full((len(categories), values.shape[1]), np.nan, dtype=float)
    for j, c in enumerate(categories):
        mask = obs_labels == c
        if mask.any():
            out[j] = values[mask].mean(axis=0)
    return out


def _group_frac_matrix(
    values: np.ndarray,
    obs_labels: np.ndarray,
    categories: Sequence[str],
    threshold: float,
) -> np.ndarray:
    """Per-category fraction of cells with ``values > threshold``."""
    pos = (values > threshold).astype(float)
    out = np.zeros((len(categories), values.shape[1]), dtype=float)
    for j, c in enumerate(categories):
        mask = obs_labels == c
        if mask.any():
            out[j] = pos[mask].mean(axis=0)
    return out


def _infer_lfc_layer(adata: AnnData, run_id: int) -> Optional[str]:
    """Fold-change layer name from kompot DE ``run_info``."""
    try:
        from ..anndata.utils import get_run_from_history
    except ImportError:
        return None
    try:
        run_info = get_run_from_history(adata, run_id, analysis_type="de")
    except Exception:
        return None
    if run_info is None:
        return None
    layer_keys = (
        run_info.get("smoothed_layer_keys")
        or run_info.get("imputed_layer_keys")
        or {}
    )
    fc = layer_keys.get("fold_change")
    if fc is None and "field_names" in run_info:
        fc = run_info["field_names"].get("fold_change_key")
    return fc


def _infer_score_key(adata: AnnData, run_id: int) -> Optional[str]:
    """Mahalanobis-ranking column from kompot DE ``run_info``."""
    try:
        from .field_inference import infer_fields_from_run_info
    except ImportError:
        return None
    try:
        fields = infer_fields_from_run_info(
            adata,
            analysis_type="de",
            run_id=run_id,
            required_fields=["mahalanobis_key"],
            strict=False,
        )
    except Exception:
        return None
    return fields.get("mahalanobis_key") if fields else None


def dotplot(
    adata: AnnData,
    genes: Optional[Sequence[str]],
    groupby: str,
    *,
    lfc_layer: Optional[str] = None,
    expr_layer: Optional[str] = None,
    score_key: Optional[str] = None,
    filter_key: Optional[str] = None,
    n_top: int = 15,
    categories_order: Optional[Sequence[str]] = None,
    min_cells: int = 0,
    expr_threshold: float = 0.0,
    vabs_pct: float = 98.0,
    vabs_min: float = 0.0,
    vmax: Optional[float] = None,
    cmap: str = "RdBu_r",
    size_exponent: float = 1.5,
    dot_max: float = 60.0,
    dot_edge_color: str = "white",
    dot_edge_lw: float = 0.2,
    axes: Optional[Tuple[Axes, Axes, Axes]] = None,
    figsize: Tuple[float, float] = (7.5, 3.2),
    cbar_title: str = "mean LFC",
    size_title: str = "fraction\nexpressing",
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    gene_label_fontsize: float = 6.5,
    category_label_fontsize: float = 5.5,
    italic_genes: bool = True,
    run_id: int = -1,
    return_fig: bool = False,
    save: Optional[str] = None,
) -> Optional[Figure]:
    """Kompot fold-change dotplot across groups.

    Each tile encodes two quantities for a (gene, group) pair:

    * **color** — mean of the per-cell LFC layer ``lfc_layer`` over cells
      in the group (i.e. the per-cell kompot fold-change averaged within
      the category). Symmetric diverging scale keyed on the
      ``vabs_pct``-th percentile of ``|LFC|`` by default.
    * **size** — fraction of cells in the group whose
      ``expr_layer`` value exceeds ``expr_threshold`` (default 0).

    Parameters
    ----------
    adata : AnnData
        AnnData with kompot DE results.
    genes : sequence of str or None
        Explicit gene list (in display order, top-first). If ``None``,
        the top ``n_top`` genes are picked by the Mahalanobis column
        inferred from run history (or ``score_key`` if given).
    groupby : str
        Column in ``adata.obs`` used for the column axis of the dotplot.
    lfc_layer : str, optional
        Layer in ``adata.layers`` holding per-cell LFC (e.g.
        ``"kompot_de_<c1>_to_<c2>_fold_change"``). If ``None``, inferred
        from the latest kompot DE run.
    expr_layer : str, optional
        Layer used to compute fraction-expressed. Defaults to ``adata.X``.
    score_key : str, optional
        Column in ``adata.var`` used to rank genes when ``genes is None``.
        Defaults to the Mahalanobis column inferred from run history.
    filter_key : str, optional
        Boolean column in ``adata.var`` restricting auto-pick candidates
        (e.g. ``"kompot_de_<c1>_to_<c2>_is_de"``).
    n_top : int, default 15
        Number of genes selected when ``genes is None``.
    categories_order : sequence of str, optional
        Subset/order of ``groupby`` categories to display. Categories not
        present in ``adata.obs[groupby]`` are dropped.
    min_cells : int, default 0
        Drop categories with fewer than this many cells.
    expr_threshold : float, default 0.0
        Threshold used for the fraction-expressed calculation.
    vabs_pct : float, default 98.0
        Percentile of ``|LFC|`` setting the symmetric color limits.
    vabs_min : float, default 0.0
        Floor for the symmetric color limit (``max(pct, floor)``). Useful
        when most tiles are near-zero and the percentile rounds down.
    vmax : float, optional
        Override the color limit. If provided, ``vabs_pct`` and
        ``vabs_min`` are ignored and the scale spans ``[-vmax, vmax]``.
    cmap : str, default ``"RdBu_r"``
        Diverging colormap name.
    size_exponent : float, default 1.5
        Exponent applied to the fraction-expressed before scaling to a
        scatter ``s`` value. Higher compresses low fractions.
    dot_max : float, default 60.0
        Target scatter ``s`` (area in pt²) for a ``frac=1.0`` dot.
    dot_edge_color : str, default ``"white"``
    dot_edge_lw : float, default 0.2
    axes : 3-tuple of :class:`matplotlib.axes.Axes`, optional
        ``(main, cbar, size_legend)``. If ``None`` a standalone figure is
        built with the three axes laid out in a constrained grid.
    figsize : tuple, default ``(7.5, 3.2)``
        Size of the standalone figure; ignored when ``axes`` is given.
    cbar_title, size_title : str
        Titles for the colorbar and size legend.
    title, xlabel, ylabel : str, optional
        Main-axis annotations. Defaults leave these empty.
    gene_label_fontsize, category_label_fontsize : float
        Font sizes for the y-axis (genes) and x-axis (categories).
    italic_genes : bool, default True
        Italicize gene y-labels.
    run_id : int, default -1
        Kompot DE run index used when inferring ``lfc_layer`` /
        ``score_key``.
    return_fig : bool, default False
        If ``True`` and ``axes is None``, return the created ``Figure``.
    save : str, optional
        If given, ``fig.savefig(save, bbox_inches="tight")`` is called.

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure if ``axes is None`` and ``return_fig`` is ``True``,
        otherwise ``None``.

    Examples
    --------
    Standalone figure, explicit genes::

        import kompot
        kompot.plot.dotplot(
            adata,
            genes=["Hbb-bh1", "Hba-x", "Tal1"],
            groupby="celltype.mapped",
            lfc_layer="kompot_de_WT_to_Tal1_fold_change",
            expr_layer="logcounts",
            return_fig=True,
        )

    Auto-pick top-20 Mahalanobis hits restricted to ``is_de=True``::

        kompot.plot.dotplot(
            adata, genes=None, groupby="celltype.mapped",
            filter_key="kompot_de_WT_to_Tal1_is_de",
            n_top=20, return_fig=True,
        )

    Embed into a composite figure::

        fig = plt.figure(figsize=(8, 4), layout="constrained")
        gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.14])
        ax_main = fig.add_subplot(gs[0, 0])
        inner = gs[0, 1].subgridspec(2, 1)
        ax_size = fig.add_subplot(inner[0, 0])
        ax_cbar = fig.add_subplot(inner[1, 0])
        kompot.plot.dotplot(
            adata, genes=top_genes, groupby="celltype.mapped",
            lfc_layer="kompot_de_WT_to_Tal1_fold_change",
            axes=(ax_main, ax_cbar, ax_size),
        )
    """
    # ---- resolve gene list --------------------------------------
    if genes is None:
        if score_key is None:
            score_key = _infer_score_key(adata, run_id)
        if score_key is None:
            raise ValueError(
                "genes=None requires a Mahalanobis ranking column, but no "
                "`score_key` was provided and none could be inferred from "
                "run history. Pass `genes=[...]` or `score_key=...`."
            )
        if score_key not in adata.var.columns:
            raise KeyError(
                f"score_key '{score_key}' not found in adata.var.columns"
            )
        candidates = adata.var.index
        if filter_key is not None:
            if filter_key not in adata.var.columns:
                raise KeyError(
                    f"filter_key '{filter_key}' not found in adata.var.columns"
                )
            mask = adata.var[filter_key].astype(bool).values
            candidates = adata.var.index[mask]
        if len(candidates) == 0:
            raise ValueError(
                f"no genes remain after applying filter_key='{filter_key}'"
            )
        ranked = (
            adata.var.loc[candidates, score_key]
            .astype(float)
            .dropna()
            .sort_values(ascending=False)
        )
        genes = list(ranked.head(n_top).index)
    else:
        genes = list(genes)

    if not genes:
        raise ValueError("no genes to plot")
    missing = [g for g in genes if g not in adata.var_names]
    if missing:
        raise KeyError(
            f"{len(missing)} gene(s) not in adata.var_names "
            f"(first few: {missing[:5]})"
        )

    # ---- resolve layers -----------------------------------------
    if lfc_layer is None:
        lfc_layer = _infer_lfc_layer(adata, run_id)
    if lfc_layer is None:
        raise ValueError(
            "Could not infer a per-cell fold-change layer from run history. "
            "Pass `lfc_layer=` explicitly (e.g. "
            "'kompot_de_<cond1>_to_<cond2>_fold_change')."
        )
    sub = adata[:, genes]
    lfc_values = _resolve_layer(sub, lfc_layer)
    expr_values = _resolve_layer(sub, expr_layer)

    # ---- resolve groups -----------------------------------------
    if groupby not in adata.obs.columns:
        raise KeyError(f"groupby '{groupby}' not in adata.obs.columns")
    obs_series = adata.obs[groupby]
    all_cats = list(obs_series.astype("category").cat.categories)
    if categories_order is None:
        cats = all_cats
    else:
        cats = [c for c in categories_order if c in all_cats]
    obs_labels = obs_series.astype(str).values
    cats_str = [str(c) for c in cats]
    if min_cells and min_cells > 0:
        cats_str = [c for c in cats_str if int((obs_labels == c).sum()) >= min_cells]
    if not cats_str:
        raise ValueError(
            "no categories remain after applying categories_order / min_cells"
        )

    lfc_mat = _group_mean_matrix(lfc_values, obs_labels, cats_str)
    frac_mat = _group_frac_matrix(expr_values, obs_labels, cats_str, expr_threshold)

    # rows = genes, cols = categories (fig-3/fig-4 swap_axes style)
    lfc = lfc_mat.T
    frac = frac_mat.T

    # ---- color scale --------------------------------------------
    if vmax is None:
        finite = lfc[np.isfinite(lfc)]
        if finite.size:
            vabs = float(np.nanpercentile(np.abs(finite), vabs_pct))
        else:
            vabs = 0.0
        vabs = max(vabs, float(vabs_min))
    else:
        vabs = float(vmax)
    if vabs <= 0:
        vabs = 1e-12  # avoid Normalize(vmin=vmax) error; caller warned below
        logger.warning(
            "dotplot: degenerate color scale (|LFC| ≈ 0 everywhere). "
            "Set vabs_min or vmax to force a meaningful range."
        )

    # ---- dot sizes ----------------------------------------------
    frac_clip = np.clip(frac, 0.0, 1.0)
    sizes = np.clip((frac_clip ** size_exponent) * dot_max, 0.0, dot_max)

    # ---- axes ---------------------------------------------------
    standalone = axes is None
    if standalone:
        fig = plt.figure(figsize=figsize, layout="constrained")
        gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.14])
        ax_main = fig.add_subplot(gs[0, 0])
        inner = gs[0, 1].subgridspec(2, 1, height_ratios=[1.0, 1.0])
        ax_size = fig.add_subplot(inner[0, 0])
        ax_cbar = fig.add_subplot(inner[1, 0])
    else:
        if len(axes) != 3:
            raise ValueError(
                "`axes` must be a 3-tuple (main, cbar, size_legend)"
            )
        ax_main, ax_cbar, ax_size = axes
        fig = ax_main.figure

    # ---- main ---------------------------------------------------
    n_genes, n_cats = lfc.shape
    xs, ys = np.meshgrid(np.arange(n_cats), np.arange(n_genes))
    ax_main.scatter(
        xs.ravel(), ys.ravel(),
        s=sizes.ravel(), c=lfc.ravel(),
        cmap=cmap, vmin=-vabs, vmax=vabs,
        edgecolors=dot_edge_color, linewidths=dot_edge_lw,
    )
    ax_main.set_xticks(np.arange(n_cats))
    ax_main.set_xticklabels(
        cats_str,
        rotation=90, fontsize=category_label_fontsize, ha="center",
    )
    ax_main.set_yticks(np.arange(n_genes))
    gene_label_kwargs = {"fontsize": gene_label_fontsize}
    if italic_genes:
        gene_label_kwargs["fontstyle"] = "italic"
    ax_main.set_yticklabels(genes, **gene_label_kwargs)
    ax_main.set_xlim(-0.5, n_cats - 0.5)
    ax_main.set_ylim(n_genes - 0.5, -0.5)
    ax_main.tick_params(axis="both", length=0, pad=1)
    for spine in ("top", "right"):
        ax_main.spines[spine].set_visible(False)
    ax_main.grid(False)
    if title is not None:
        ax_main.set_title(title, pad=2, fontsize=7.5)
    if xlabel is not None:
        ax_main.set_xlabel(xlabel)
    if ylabel is not None:
        ax_main.set_ylabel(ylabel)

    # ---- colorbar -----------------------------------------------
    sm = ScalarMappable(norm=Normalize(vmin=-vabs, vmax=vabs), cmap=cmap)
    cb = fig.colorbar(sm, cax=ax_cbar, orientation="vertical")
    cb.locator = MaxNLocator(nbins=3)
    cb.update_ticks()
    cb.ax.tick_params(labelsize=category_label_fontsize, length=2)
    for spine in cb.ax.spines.values():
        spine.set_visible(False)
    if cbar_title:
        ax_cbar.set_title(
            cbar_title, fontsize=category_label_fontsize, pad=3, loc="center",
        )

    # ---- size legend --------------------------------------------
    size_fractions = (0.2, 0.4, 0.6, 0.8, 1.0)
    n_leg = len(size_fractions)
    ax_size.set_xlim(0, 1)
    ax_size.set_ylim(-0.5, n_leg - 0.5)
    ax_size.set_xticks([])
    ax_size.set_yticks([])
    for spine in ax_size.spines.values():
        spine.set_visible(False)
    # Boost legend dot sizes relative to main so a narrow size-legend axis
    # still renders readable dots; matches the fig-4 panel-N rendering.
    legend_scale = 3.5
    for i, frac_ in enumerate(size_fractions):
        y = n_leg - 1 - i
        s_val = (float(frac_) ** size_exponent) * dot_max * legend_scale
        ax_size.scatter(
            [0.3], [y], s=s_val, c="0.5",
            linewidths=0.3, edgecolors=dot_edge_color, zorder=2,
        )
        ax_size.text(
            0.55, y, f"{int(round(frac_ * 100))}%",
            ha="left", va="center",
            fontsize=category_label_fontsize, color="0.2",
        )
    if size_title:
        ax_size.set_title(
            size_title, fontsize=category_label_fontsize, pad=3, loc="center",
        )

    # ---- save / return ------------------------------------------
    if save is not None:
        fig.savefig(save, bbox_inches="tight")

    if standalone and return_fig:
        return fig
    return None
