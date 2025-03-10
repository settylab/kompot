"""Heatmap and related plotting functions."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Optional, Union, List, Tuple, Dict, Any, Sequence, Literal
from anndata import AnnData
import pandas as pd
import seaborn as sns
import logging

from ..utils import get_run_from_history, KOMPOT_COLORS
try:
    import scanpy as sc
    _has_scanpy = True
except ImportError:
    _has_scanpy = False

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
    
    # Try to get key from specific run if requested - specifically from kompot_de first
    run_info = get_run_from_history(adata, run_id, analysis_type="de")
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
    
    # If still no lfc_key, try global history
    if inferred_lfc_key is None:
        run_info = get_run_from_history(adata, run_id, history_key="kompot_run_history")
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


def direction_barplot(
    adata: AnnData,
    category_column: str,
    direction_column: Optional[str] = None,
    condition1: Optional[str] = None,
    condition2: Optional[str] = None,
    normalize: Literal["index", "columns", None] = "index",
    figsize: Tuple[float, float] = (12, 6),
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    colors: Optional[Dict[str, str]] = None,
    rotation: float = 90,
    legend_title: str = "Direction",
    legend_loc: str = "best",
    stacked: bool = True,
    sort_by: Optional[str] = None,
    ascending: bool = False,
    ax: Optional[plt.Axes] = None,
    return_fig: bool = False,
    save: Optional[str] = None,
    run_id: int = -1,
    **kwargs
) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """
    Create a barplot showing the direction of change distribution across categories.
    
    This function creates a stacked or grouped barplot showing the distribution of 
    up/down/neutral changes across different categories (like cell types).
    
    Parameters
    ----------
    adata : AnnData
        AnnData object containing differential abundance results
    category_column : str
        Column in adata.obs to use for grouping (e.g., "cell_type")
    direction_column : str, optional
        Column in adata.obs containing direction information.
        If None, will try to infer from "kompot_da_log_fold_change_direction" pattern.
    condition1 : str, optional
        Name of condition 1 (denominator in fold change)
    condition2 : str, optional
        Name of condition 2 (numerator in fold change)
    normalize : str or None, optional
        How to normalize the data. Options:
        - "index": normalize across rows (sum to 100% for each category)
        - "columns": normalize across columns (sum to 100% for each direction)
        - None: use raw counts
    figsize : tuple, optional
        Figure size as (width, height) in inches
    title : str, optional
        Plot title. If None and conditions provided, uses "Direction of Change by {category_column}\n{condition1} vs {condition2}"
    xlabel : str, optional
        Label for x-axis. If None, uses the category_column
    ylabel : str, optional
        Label for y-axis. Defaults to "Percentage (%)" when normalize="index", otherwise "Count"
    colors : dict, optional
        Dictionary mapping direction values to colors. Default is {"up": "#d73027", "down": "#4575b4", "neutral": "#d3d3d3"}
    rotation : float, optional
        Rotation angle for x-tick labels
    legend_title : str, optional
        Title for the legend
    legend_loc : str, optional
        Location for the legend
    stacked : bool, optional
        Whether to create a stacked (True) or grouped (False) bar plot
    sort_by : str, optional
        Direction category to sort by (e.g., "up", "down"). If None, uses the order in the data
    ascending : bool, optional
        Whether to sort in ascending order
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure
    return_fig : bool, optional
        If True, returns the figure and axes
    save : str, optional
        Path to save figure. If None, figure is not saved
    run_id : int, optional
        Specific run ID to use for fetching data from run history.
        Negative indices count from the end (-1 is the latest run). 
    **kwargs : 
        Additional parameters passed to pandas.DataFrame.plot
        
    Returns
    -------
    If return_fig is True, returns (fig, ax)
    """
    # Default colors if not provided
    if colors is None:
        colors = {
            "up": KOMPOT_COLORS["direction"]["up"],
            "down": KOMPOT_COLORS["direction"]["down"],
            "neutral": KOMPOT_COLORS["direction"]["neutral"]
        }
    
    # Default ylabel based on normalization method
    if ylabel is None:
        ylabel = "Percentage (%)" if normalize == "index" else "Count"
    
    # Get run information if available
    run_info = get_run_from_history(adata, run_id)
    
    # Log run information - always use positive run index for logging
    if run_info is not None:
        # Get the actual run index for logging (convert negative to positive)
        if run_id < 0:
            if 'kompot_da' in adata.uns and 'run_history' in adata.uns['kompot_da']:
                actual_run_id = len(adata.uns['kompot_da']['run_history']) + run_id
            elif 'kompot_run_history' in adata.uns:
                actual_run_id = len(adata.uns['kompot_run_history']) + run_id
            else:
                actual_run_id = run_id
        else:
            actual_run_id = run_id
            
        logger.info(f"Using run {actual_run_id} for direction_barplot")
            
        # Extract conditions from run info if available for better logging
        run_conditions = None
        if 'params' in run_info and 'condition_key' in run_info['params'] and 'conditions' in run_info['params']:
            cond_key = run_info['params']['condition_key']
            conditions = run_info['params']['conditions']
            if len(conditions) == 2:
                run_conditions = conditions
                logger.info(f"Run compares conditions: {conditions[0]} vs {conditions[1]} (key: {cond_key})")
    
    # Infer direction column if not provided
    if direction_column is None:
        # Try to get direction column from run info
        if run_info is not None:
            if 'abundance_key' in run_info:
                result_key = run_info['abundance_key']
                if result_key in adata.uns and 'run_info' in adata.uns[result_key]:
                    key_run_info = adata.uns[result_key]['run_info']
                    if 'direction_key' in key_run_info:
                        direction_column = key_run_info['direction_key']
                        logger.info(f"Using direction column '{direction_column}' from run info")
        
        # If not found in run info, look for columns matching pattern
        if direction_column is None:
            direction_cols = [col for col in adata.obs.columns if "kompot_da_log_fold_change_direction" in col]
            if not direction_cols:
                raise ValueError("Could not find direction column. Please specify direction_column.")
            elif len(direction_cols) == 1:
                direction_column = direction_cols[0]
                logger.info(f"Using direction column: {direction_column}")
            else:
                # If conditions provided, try to find matching column
                if condition1 and condition2:
                    for col in direction_cols:
                        if f"{condition1}_vs_{condition2}" in col:
                            direction_column = col
                            logger.info(f"Using direction column matching conditions: {direction_column}")
                            break
                        elif f"{condition2}_vs_{condition1}" in col:
                            direction_column = col
                            logger.warning(f"Found direction column with reversed conditions: {col}")
                            break
                    if direction_column is None:
                        direction_column = direction_cols[0]
                        logger.warning(f"Multiple direction columns found, using the first one: {direction_column}")
                else:
                    direction_column = direction_cols[0]
                    logger.warning(f"Multiple direction columns found, using the first one: {direction_column}")
    
    # Extract condition names from direction column if not provided
    if (condition1 is None or condition2 is None) and "_vs_" in direction_column:
        parts = direction_column.split("_vs_")
        if len(parts) >= 2:
            cond_parts = parts[-2:]
            # Remove any trailing parts after condition names
            condition1 = cond_parts[0].split("_")[-1]
            condition2 = cond_parts[1].split("_")[0]
    
    # Create the crosstab
    crosstab = pd.crosstab(
        adata.obs[category_column],
        adata.obs[direction_column],
        normalize=normalize
    )
    
    # If normalize is "index", multiply by 100 for percentage
    if normalize == "index":
        crosstab = crosstab * 100
    
    # Order columns consistently
    if "up" in crosstab.columns and "down" in crosstab.columns:
        # Keep only the columns that exist in our data
        ordered_cols = [col for col in ["up", "down", "neutral"] if col in crosstab.columns]
        crosstab = crosstab[ordered_cols]
    
    # Sort by specified direction if requested
    if sort_by is not None and sort_by in crosstab.columns:
        crosstab = crosstab.sort_values(by=sort_by, ascending=ascending)
    
    # Create figure if ax not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Color mapping - only use colors for directions that exist in our data
    plot_colors = [colors.get(col, "gray") for col in crosstab.columns]
    
    # Create the plot
    crosstab.plot(
        kind="bar",
        stacked=stacked,
        color=plot_colors,
        ax=ax,
        **kwargs
    )
    
    # Remove grid by default
    ax.grid(False)
    
    # Set labels and title
    ax.set_xlabel(xlabel if xlabel is not None else category_column)
    ax.set_ylabel(ylabel)
    
    # Set title if provided or can be inferred
    if title is None and condition1 and condition2:
        title = f"Direction of Change by {category_column}\n{condition1} vs {condition2}"
    if title:
        ax.set_title(title)
    
    # Rotate tick labels
    plt.xticks(rotation=rotation)
    
    # Set legend
    plt.legend(title=legend_title, loc=legend_loc, frameon=False, bbox_to_anchor=(1.05, 1), borderaxespad=0)
    
    # Adjust layout to accommodate legend
    if legend_loc == 'best' or legend_loc.startswith('right'):
        plt.tight_layout(rect=[0, 0, 0.85, 1])
    else:
        plt.tight_layout()
    
    # Save figure if path provided
    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight")
    
    # Return figure and axes if requested
    if return_fig:
        return fig, ax
    elif save is None:
        # Only show if not saving and not returning
        plt.show()


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