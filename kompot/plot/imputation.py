"""Plotting functions for expression imputation results."""

import logging
import numpy as np
from typing import Optional, List, Union

logger = logging.getLogger("kompot")

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    import matplotlib.gridspec as gridspec
except ImportError:
    raise ImportError("matplotlib is required for plotting: pip install matplotlib")

try:
    import scanpy as sc

    _has_scanpy = True
except ImportError:
    _has_scanpy = False


def plot_imputation(
    adata,
    genes: Optional[List[str]] = None,
    n_top_genes: int = 6,
    embedding_key: str = "X_umap",
    result_key: str = "kompot_impute",
    condition: Optional[str] = None,
    layer: Optional[str] = None,
    show_obs_variance: bool = True,
    cmap: str = "Spectral_r",
    cmap_std: str = "magma",
    figsize_per_panel: tuple = (3.5, 3.0),
    title: Optional[str] = None,
    return_fig: bool = False,
    **kwargs,
) -> Optional[Figure]:
    """Plot raw vs. GP-imputed expression with uncertainty.

    Shows a grid with rows:
    1. Raw expression
    2. GP-imputed expression
    3. Epistemic std (GP posterior, shared across genes) or total std if no obs_variance
    4. Aleatoric std (sqrt of obs_variance, per-gene) -- only if available

    Uses ``scanpy.pl.embedding`` internally when available, falling back to
    manual scatter plots otherwise.

    Parameters
    ----------
    adata : AnnData
        AnnData object with imputation results.
    genes : list of str, optional
        Genes to plot. If *None*, selects top genes by mean imputed value.
    n_top_genes : int
        Number of genes when ``genes`` is *None*. Max 8.
    embedding_key : str
        Key in ``adata.obsm`` for 2-D coordinates.
    result_key : str
        Base key used in ``impute_expression()``.
    condition : str, optional
        Condition label in the layer names. If *None*, auto-detected from
        available layers.
    layer : str, optional
        Layer with raw expression. If *None*, uses ``adata.X``.
    show_obs_variance : bool
        Show obs_variance row if available.
    cmap : str
        Colormap for expression values.
    cmap_std : str
        Colormap for uncertainty panels.
    figsize_per_panel : tuple
        Size of each subplot (width, height).
    title : str, optional
        Overall figure title.
    return_fig : bool
        If *True*, return the Figure instead of calling ``plt.show()``.
    **kwargs
        Extra keyword arguments forwarded to ``scanpy.pl.embedding``
        (e.g. ``s``, ``size``, ``vmin``, ``vmax``).

    Returns
    -------
    Figure or None
    """
    # --- Validate embedding ---
    if embedding_key not in adata.obsm:
        raise ValueError(
            f"'{embedding_key}' not found in adata.obsm. "
            f"Available: {list(adata.obsm.keys())}"
        )
    coords = adata.obsm[embedding_key]
    if coords.shape[1] < 2:
        raise ValueError(
            f"Embedding must have >= 2 dimensions, got {coords.shape[1]}"
        )

    # --- Auto-detect condition label ---
    if condition is None:
        condition = _detect_condition_label(adata, result_key)

    imputed_key = f"{result_key}_{condition}_imputed"
    std_key = f"{result_key}_{condition}_std"
    obs_var_key = f"{result_key}_{condition}_obs_variance"

    if imputed_key not in adata.layers:
        raise ValueError(
            f"No imputation layers found with prefix '{result_key}'. "
            f"Run kompot.impute_expression() first or check result_key/condition."
        )

    # --- Cell mask (non-NaN rows in imputed layer) ---
    imputed_full = adata.layers[imputed_key]
    mask = ~np.all(np.isnan(imputed_full), axis=1)
    adata_sub = adata[mask].copy()

    # --- Select genes ---
    if genes is None:
        mean_imp = np.nanmean(np.abs(imputed_full[mask]), axis=0)
        top_idx = np.argsort(mean_imp)[::-1][:n_top_genes]
        genes = [adata.var_names[i] for i in top_idx]
    n_genes = min(len(genes), 8)
    genes = genes[:n_genes]

    # --- Determine rows ---
    has_std = std_key in adata.layers
    has_obs_var = show_obs_variance and obs_var_key in adata.layers
    n_rows = 2 + int(has_std) + int(has_obs_var)
    row_labels = ["Raw", "GP imputed"]
    if has_std:
        row_labels.append("Epistemic std" if has_obs_var else "Total std")
    if has_obs_var:
        row_labels.append("Aleatoric std")

    # --- Prepare temporary obs columns for sc.pl.embedding ---
    _tmp_keys = []
    basis = embedding_key.replace("X_", "")

    for gene in genes:
        gene_idx = list(adata.var_names).index(gene)

        # Imputed
        key_imp = f"_kompot_plot_imputed_{gene}"
        adata_sub.obs[key_imp] = np.asarray(imputed_full[mask, gene_idx])
        _tmp_keys.append(key_imp)

        # Std: show epistemic component when obs_var is also shown
        if has_std:
            key_std = f"_kompot_plot_std_{gene}"
            total_std = np.asarray(adata.layers[std_key][mask, gene_idx])
            if has_obs_var:
                obs_var_vals = np.asarray(
                    adata.layers[obs_var_key][mask, gene_idx]
                )
                # epistemic std = sqrt(total_var - obs_var)
                epistemic_var = np.maximum(total_std**2 - obs_var_vals, 0)
                adata_sub.obs[key_std] = np.sqrt(epistemic_var)
            else:
                adata_sub.obs[key_std] = total_std
            _tmp_keys.append(key_std)

        # Obs variance -> show as std (same units as expression)
        if has_obs_var:
            key_ov = f"_kompot_plot_obsvar_{gene}"
            adata_sub.obs[key_ov] = np.sqrt(
                np.asarray(adata.layers[obs_var_key][mask, gene_idx])
            )
            _tmp_keys.append(key_ov)

    # --- Build figure ---
    fw, fh = figsize_per_panel
    label_width = 0.6  # inches for row labels
    fig_w = label_width + fw * n_genes
    fig_h = fh * n_rows
    fig = plt.figure(figsize=(fig_w, fig_h))

    # GridSpec with left margin for row labels
    left_frac = label_width / fig_w
    gs = gridspec.GridSpec(
        n_rows,
        n_genes,
        figure=fig,
        left=left_frac,
        right=0.98,
        bottom=0.02,
        top=0.92,
        hspace=0.15,
        wspace=0.05,
    )

    sc_kwargs = dict(
        show=False,
        frameon=False,
        colorbar_loc=None,
        **kwargs,
    )

    for col, gene in enumerate(genes):
        gene_idx = list(adata.var_names).index(gene)

        # --- Row 0: Raw expression ---
        ax = fig.add_subplot(gs[0, col])
        if _has_scanpy:
            sc.pl.embedding(
                adata_sub,
                basis=basis,
                color=gene,
                layer=layer,
                color_map=cmap,
                title=gene,
                ax=ax,
                **sc_kwargs,
            )
        else:
            _scatter_fallback(
                ax,
                adata_sub.obsm[embedding_key][:, :2],
                _get_raw(adata_sub, gene, layer),
                cmap=cmap,
                n_obs=adata_sub.n_obs,
            )
            ax.set_title(gene, fontsize=10, fontweight="bold")
            ax.axis("off")

        # --- Row 1: GP imputed ---
        ax = fig.add_subplot(gs[1, col])
        key_imp = f"_kompot_plot_imputed_{gene}"
        if _has_scanpy:
            sc.pl.embedding(
                adata_sub,
                basis=basis,
                color=key_imp,
                color_map=cmap,
                title="",
                ax=ax,
                **sc_kwargs,
            )
        else:
            _scatter_fallback(
                ax,
                adata_sub.obsm[embedding_key][:, :2],
                np.asarray(adata_sub.obs[key_imp]),
                cmap=cmap,
                n_obs=adata_sub.n_obs,
            )
            ax.axis("off")

        row_idx = 2

        # --- Row 2: Total std ---
        if has_std:
            ax = fig.add_subplot(gs[row_idx, col])
            key_std = f"_kompot_plot_std_{gene}"
            if _has_scanpy:
                sc.pl.embedding(
                    adata_sub,
                    basis=basis,
                    color=key_std,
                    color_map=cmap_std,
                    title="",
                    ax=ax,
                    **sc_kwargs,
                )
            else:
                _scatter_fallback(
                    ax,
                    adata_sub.obsm[embedding_key][:, :2],
                    np.asarray(adata_sub.obs[key_std]),
                    cmap=cmap_std,
                    n_obs=adata_sub.n_obs,
                )
                ax.axis("off")
            row_idx += 1

        # --- Row 3: Obs variance ---
        if has_obs_var:
            ax = fig.add_subplot(gs[row_idx, col])
            key_ov = f"_kompot_plot_obsvar_{gene}"
            if _has_scanpy:
                sc.pl.embedding(
                    adata_sub,
                    basis=basis,
                    color=key_ov,
                    color_map=cmap_std,
                    title="",
                    ax=ax,
                    **sc_kwargs,
                )
            else:
                _scatter_fallback(
                    ax,
                    adata_sub.obsm[embedding_key][:, :2],
                    np.asarray(adata_sub.obs[key_ov]),
                    cmap=cmap_std,
                    n_obs=adata_sub.n_obs,
                )
                ax.axis("off")

    # --- Vertical row labels on the left margin ---
    for row, label in enumerate(row_labels):
        # Center of each row in figure coordinates
        row_top = gs[row, 0].get_position(fig).y1
        row_bottom = gs[row, 0].get_position(fig).y0
        y_center = (row_top + row_bottom) / 2
        fig.text(
            left_frac * 0.5,
            y_center,
            label,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            rotation=90,
        )

    # --- Title ---
    if title is None:
        title = f"Expression Imputation ({condition})"
    fig.suptitle(title, fontsize=13)

    # Store grid dimensions for programmatic access
    fig._kompot_nrows = n_rows
    fig._kompot_ncols = n_genes

    if return_fig:
        return fig
    plt.show()
    return None


def _get_raw(adata, gene, layer):
    """Get raw expression values for a gene, handling sparse matrices."""
    if layer is not None:
        raw = adata[:, gene].layers[layer]
    else:
        raw = adata[:, gene].X
    if hasattr(raw, "toarray"):
        return np.asarray(raw.toarray()).ravel()
    return np.asarray(raw).ravel()


def _scatter_fallback(ax, xy, values, cmap, n_obs=1):
    """Manual scatter plot fallback when scanpy is unavailable."""
    # Mimic scanpy's default point size heuristic
    s = 120000 / max(n_obs, 1)
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=values,
        s=s,
        cmap=cmap,
        rasterized=True,
    )
    ax.axis("off")


def _detect_condition_label(adata, result_key: str) -> str:
    """Auto-detect the condition label from imputation layer names."""
    prefix = f"{result_key}_"
    suffix = "_imputed"
    for key in adata.layers:
        if key.startswith(prefix) and key.endswith(suffix):
            return key[len(prefix) : -len(suffix)]
    raise ValueError(
        f"No imputation layers found with prefix '{prefix}'. "
        f"Available layers: {list(adata.layers.keys())}"
    )
