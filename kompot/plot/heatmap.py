"""Heatmap function for visualizing gene expression across conditions."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Optional, Union, List, Tuple, Dict, Any, Sequence
from anndata import AnnData
import pandas as pd
import seaborn as sns
import logging

from ..utils import get_run_from_history

logger = logging.getLogger("kompot")


def _infer_heatmap_keys(adata: AnnData, run_id: Optional[int] = None, lfc_key: Optional[str] = None,
                       score_key: Optional[str] = "kompot_de_mahalanobis"):
    """
    Infer heatmap keys from AnnData object.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object with differential expression results
    run_id : int, optional
        Run ID to use. If None, uses latest run.
    lfc_key : str, optional
        Log fold change key. If provided, will be returned as is.
    score_key : str, optional
        Score key. If "kompot_de_mahalanobis", might be replaced with a run-specific key.
        
    Returns
    -------
    tuple
        (lfc_key, score_key) with the inferred keys
    """
    inferred_lfc_key = lfc_key
    inferred_score_key = score_key
    
    # If keys already provided, use them
    if inferred_lfc_key is not None and not (score_key == "kompot_de_mahalanobis" and 
                                        not any(k == score_key for k in adata.var.columns)):
        return inferred_lfc_key, inferred_score_key
    
    # Try to get key from specific run if requested
    run_info = get_run_from_history(adata, run_id)
    if run_info is not None and run_info.get('expression_key'):
        de_key = run_info['expression_key']
        
        # Check if we have run_info for this key
        if de_key in adata.uns and 'run_info' in adata.uns[de_key]:
            key_run_info = adata.uns[de_key]['run_info']
            if inferred_lfc_key is None and 'lfc_key' in key_run_info:
                inferred_lfc_key = key_run_info['lfc_key']
                logger.info(f"Using lfc_key '{inferred_lfc_key}' from run {run_id}")
            
            # Try to use mahalanobis key from run info if available
            if score_key == "kompot_de_mahalanobis" and 'mahalanobis_key' in key_run_info and key_run_info['mahalanobis_key']:
                inferred_score_key = key_run_info['mahalanobis_key']
                logger.info(f"Using score_key '{inferred_score_key}' from run {run_id}")
    
    # If still no lfc_key, try latest run
    if (inferred_lfc_key is None or score_key == "kompot_de_mahalanobis") and 'kompot_latest_run' in adata.uns and adata.uns['kompot_latest_run'].get('expression_key'):
        # Get the latest DE run key
        de_key = adata.uns['kompot_latest_run']['expression_key']
        
        # Check if we have run_info for this key
        if de_key in adata.uns and 'run_info' in adata.uns[de_key]:
            run_info = adata.uns[de_key]['run_info']
            if inferred_lfc_key is None and 'lfc_key' in run_info:
                inferred_lfc_key = run_info['lfc_key']
                logger.info(f"Using lfc_key '{inferred_lfc_key}' from latest run information")
            
            # Try to use mahalanobis key from run info if available
            if score_key == "kompot_de_mahalanobis" and 'mahalanobis_key' in run_info and run_info['mahalanobis_key']:
                inferred_score_key = run_info['mahalanobis_key']
                logger.info(f"Using score_key '{inferred_score_key}' from latest run information")
    
    # If still not found, try to infer from column names
    if inferred_lfc_key is None:
        lfc_keys = [k for k in adata.var.columns if 'kompot_de_' in k and 'lfc' in k.lower()]
        if len(lfc_keys) == 1:
            inferred_lfc_key = lfc_keys[0]
        elif len(lfc_keys) > 1:
            # If multiple keys found, try to find the mean or avg one
            mean_keys = [k for k in lfc_keys if 'mean' in k.lower() or 'avg' in k.lower()]
            if mean_keys:
                inferred_lfc_key = mean_keys[0]
            else:
                inferred_lfc_key = lfc_keys[0]
        else:
            raise ValueError("Could not infer lfc_key. Please specify manually.")
    
    return inferred_lfc_key, inferred_score_key


def heatmap(
    adata: AnnData,
    var_names: Optional[Union[List[str], Sequence[str]]] = None,
    groupby: str = None,
    n_top_genes: int = 20,
    lfc_key: Optional[str] = None,
    score_key: str = "kompot_de_mahalanobis",
    layer: Optional[str] = None,
    standard_scale: Optional[Union[str, int]] = None,
    cmap: Union[str, mcolors.Colormap] = "viridis",
    dendrogram: bool = False,
    swap_axes: bool = False,
    figsize: Optional[Tuple[float, float]] = None,
    show_gene_labels: bool = True,
    show_group_labels: bool = True,
    gene_labels_size: int = 10,
    group_labels_size: int = 12,
    colorbar_title: str = "Expression",
    colorbar_kwargs: Optional[Dict[str, Any]] = None,
    title: Optional[str] = None,
    sort_genes: bool = True,
    center: Optional[float] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
    return_fig: bool = False,
    save: Optional[str] = None,
    run_id: Optional[int] = None,
    **kwargs
) -> Union[None, Tuple[plt.Figure, plt.Axes, Optional[plt.Axes]]]:
    """
    Create a heatmap of gene expression from Kompot differential expression results.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object containing expression data
    var_names : list, optional
        List of genes to include in the heatmap. If None, will use top genes
        based on score_key and lfc_key
    groupby : str, optional
        Key in adata.obs for grouping cells. If None, no grouping is performed
    n_top_genes : int, optional
        Number of top genes to include if var_names is None
    lfc_key : str, optional
        Key in adata.var for log fold change values.
        If None, will try to infer from kompot_de_ keys.
    score_key : str, optional
        Key in adata.var for significance scores.
        Default is "kompot_de_mahalanobis"
    layer : str, optional
        Layer in AnnData to use for expression values. If None, uses .X
    standard_scale : str or int, optional
        Whether to scale the expression values ('var', 'group' or 0 for rows, 1 for columns)
    cmap : str or colormap, optional
        Colormap to use for the heatmap
    dendrogram : bool, optional
        Whether to show dendrograms for clustering
    swap_axes : bool, optional
        Whether to swap the axes (genes as columns, groups as rows)
    figsize : tuple, optional
        Figure size as (width, height) in inches
    show_gene_labels : bool, optional
        Whether to show gene labels
    show_group_labels : bool, optional
        Whether to show group labels
    gene_labels_size : int, optional
        Font size for gene labels
    group_labels_size : int, optional
        Font size for group labels
    colorbar_title : str, optional
        Title for the colorbar
    colorbar_kwargs : dict, optional
        Additional parameters for colorbar
    title : str, optional
        Title for the heatmap
    sort_genes : bool, optional
        Whether to sort genes by score
    center : float, optional
        Value to center the colormap at
    vmin : float, optional
        Minimum value for colormap
    vmax : float, optional
        Maximum value for colormap
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure
    return_fig : bool, optional
        If True, returns the figure and axes
    save : str, optional
        Path to save figure. If None, figure is not saved
    run_id : int, optional
        Specific run ID to use for fetching field names from run history.
        Negative indices count from the end (-1 is the latest run). If None, 
        uses the latest run information.
    **kwargs :
        Additional parameters passed to sns.heatmap
        
    Returns
    -------
    If return_fig is True, returns (fig, main_ax, [dendrogram_ax])
    """
    # Set default colorbar kwargs
    colorbar_kwargs = colorbar_kwargs or {}
    
    # If var_names not provided, get top genes based on DE results
    if var_names is None:
        # Infer keys using the helper function
        lfc_key, score_key = _infer_heatmap_keys(adata, run_id, lfc_key, score_key)
        
        # Get top genes based on score
        de_data = pd.DataFrame({
            'gene': adata.var_names,
            'lfc': adata.var[lfc_key] if lfc_key in adata.var else np.zeros(adata.n_vars),
            'score': adata.var[score_key] if score_key in adata.var else np.zeros(adata.n_vars)
        })
        
        if sort_genes:
            de_data = de_data.sort_values('score', ascending=False)
            
        # Get top genes
        var_names = de_data.head(n_top_genes)['gene'].tolist()
    
    # Get expression data for the selected genes
    if layer is not None and layer in adata.layers:
        expr_matrix = adata[:, var_names].layers[layer].toarray() if hasattr(adata.layers[layer], 'toarray') else adata[:, var_names].layers[layer]
    else:
        expr_matrix = adata[:, var_names].X.toarray() if hasattr(adata.X, 'toarray') else adata[:, var_names].X
    
    # Create dataframe with expression data
    expr_df = pd.DataFrame(expr_matrix, index=adata.obs_names, columns=var_names)
    
    # Group by condition if provided
    if groupby is not None and groupby in adata.obs:
        # Calculate mean expression per group
        grouped_expr = expr_df.groupby(adata.obs[groupby]).mean()
    else:
        grouped_expr = expr_df
    
    # Scale data if requested
    if standard_scale == 'group' or standard_scale == 1:
        # Scale by group (cols)
        grouped_expr = (grouped_expr - grouped_expr.mean(axis=0)) / grouped_expr.std(axis=0)
    elif standard_scale == 'var' or standard_scale == 0:
        # Scale by gene (rows)
        grouped_expr = (grouped_expr.T - grouped_expr.T.mean(axis=0)) / grouped_expr.T.std(axis=0)
        grouped_expr = grouped_expr.T
    
    # Swap axes if requested
    if swap_axes:
        grouped_expr = grouped_expr.T
    
    # Set up figure and axes
    if ax is None:
        # Calculate appropriate figsize if not provided
        if figsize is None:
            # Base size on number of genes and groups
            n_rows, n_cols = grouped_expr.shape
            figsize = (8 + n_cols * 0.3, 6 + n_rows * 0.3)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Create the heatmap
    dendrogram_ax = None
    if dendrogram:
        # With dendrogram - use clustermap
        grid = sns.clustermap(
            grouped_expr,
            cmap=cmap,
            center=center,
            vmin=vmin,
            vmax=vmax,
            xticklabels=show_group_labels if not swap_axes else show_gene_labels,
            yticklabels=show_gene_labels if not swap_axes else show_group_labels,
            cbar_kws={"label": colorbar_title, **colorbar_kwargs},
            **kwargs
        )
        # Extract main axes and adjust label sizes
        ax = grid.ax_heatmap
        if title:
            plt.suptitle(title)
        fig = grid.fig
    else:
        # Without dendrogram - use regular heatmap
        sns.heatmap(
            grouped_expr,
            cmap=cmap,
            center=center,
            vmin=vmin,
            vmax=vmax,
            ax=ax,
            xticklabels=show_group_labels if not swap_axes else show_gene_labels,
            yticklabels=show_gene_labels if not swap_axes else show_group_labels,
            cbar_kws={"label": colorbar_title, **colorbar_kwargs},
            **kwargs
        )
        if title:
            ax.set_title(title)
    
    # Adjust label sizes
    if swap_axes:
        if show_gene_labels:
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=gene_labels_size)
        if show_group_labels:
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=group_labels_size)
    else:
        if show_gene_labels:
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=gene_labels_size)
        if show_group_labels:
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=group_labels_size)
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
    
    # Return figure and axes if requested
    if return_fig:
        return fig, ax, dendrogram_ax
    elif save is None:
        # Only show if not saving and not returning
        plt.show()