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
from .volcano import _extract_conditions_from_key, _infer_da_keys
try:
    import scanpy as sc
    _has_scanpy = True
except ImportError:
    _has_scanpy = False

logger = logging.getLogger("kompot")


def _infer_direction_key(adata: AnnData, run_id: int = -1, direction_column: Optional[str] = None):
    """
    Infer direction column for direction barplots from AnnData object.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object with differential abundance results
    run_id : int, optional
        Run ID to use. Default is -1 (latest run).
    direction_column : str, optional
        Direction column to use. If provided, will be returned as is.
        
    Returns
    -------
    tuple
        (direction_column, condition1, condition2) with the inferred values
    """
    # If direction column already provided, just check if it exists
    if direction_column is not None:
        if direction_column in adata.obs.columns:
            # Try to extract conditions from the column name
            conditions = _extract_conditions_from_key(direction_column)
            if conditions:
                condition1, condition2 = conditions
                return direction_column, condition1, condition2
            else:
                return direction_column, None, None
        else:
            logger.warning(f"Provided direction_column '{direction_column}' not found in adata.obs")
            direction_column = None
            
    # Get run info from specified run_id - specifically from kompot_da
    run_info = get_run_from_history(adata, run_id, analysis_type="da")
    condition1 = None
    condition2 = None
    
    # Get condition information
    if run_info is not None and 'params' in run_info:
        params = run_info['params']
        if 'conditions' in params and len(params['conditions']) == 2:
            condition1 = params['conditions'][0]
            condition2 = params['conditions'][1]
    
    # First try to get direction column from run info field_names
    if run_info is not None and 'field_names' in run_info:
        field_names = run_info['field_names']
        if 'direction_key' in field_names:
            direction_column = field_names['direction_key']
            # Check that column exists
            if direction_column not in adata.obs.columns:
                direction_column = None
    
    # If not found in field_names, try the older method with abundance_key
    if direction_column is None and run_info is not None:
        if 'abundance_key' in run_info:
            result_key = run_info['abundance_key']
            if result_key in adata.uns and 'run_info' in adata.uns[result_key]:
                key_run_info = adata.uns[result_key]['run_info']
                if 'direction_key' in key_run_info:
                    direction_column = key_run_info['direction_key']
    
    # If still not found, look for columns matching pattern
    if direction_column is None:
        direction_cols = [col for col in adata.obs.columns if "kompot_da_log_fold_change_direction" in col]
        if not direction_cols:
            return None, condition1, condition2
        elif len(direction_cols) == 1:
            direction_column = direction_cols[0]
        else:
            # If conditions provided, try to find matching column
            if condition1 and condition2:
                for col in direction_cols:
                    if f"{condition1}_vs_{condition2}" in col:
                        direction_column = col
                        break
                    elif f"{condition2}_vs_{condition1}" in col:
                        direction_column = col
                        # Keep this warning as it's informative about reversed condition order
                        logger.warning(f"Found direction column with reversed conditions: {col}")
                        break
                if direction_column is None:
                    direction_column = direction_cols[0]
                    # Keep this warning as it's important information about ambiguity
                    logger.warning(f"Multiple direction columns found, using the first one: {direction_column}")
            else:
                direction_column = direction_cols[0]
                # Keep this warning as it's important information about ambiguity
                logger.warning(f"Multiple direction columns found, using the first one: {direction_column}")
                
    # If we found a direction column but not conditions, try to extract them from the column name
    if direction_column is not None and (condition1 is None or condition2 is None):
        conditions = _extract_conditions_from_key(direction_column)
        if conditions:
            condition1, condition2 = conditions
            
    return direction_column, condition1, condition2


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
    
    # If keys already provided, just return them
    if inferred_lfc_key is not None and not (score_key == "kompot_de_mahalanobis" and 
                                        not any(k == score_key for k in adata.var.columns)):
        return inferred_lfc_key, inferred_score_key
    
    # Use the specified run_id to get keys from the run history
    if run_id is not None:
        # Get run info from kompot_de for the specific run_id
        run_info = get_run_from_history(adata, run_id, analysis_type="de")
        
        if run_info is not None and 'field_names' in run_info:
            field_names = run_info['field_names']
            
            # Get lfc_key from field_names
            if inferred_lfc_key is None and 'mean_lfc_key' in field_names:
                inferred_lfc_key = field_names['mean_lfc_key']
                # Check that column exists
                if inferred_lfc_key not in adata.var.columns:
                    inferred_lfc_key = None
            
            # Get score_key from field_names if needed
            if score_key == "kompot_de_mahalanobis" and 'mahalanobis_key' in field_names:
                inferred_score_key = field_names['mahalanobis_key']
                # Check that column exists
                if inferred_score_key not in adata.var.columns:
                    inferred_score_key = None
    
    # For backwards compatibility - if run_id is None, try to use latest run 
    elif inferred_lfc_key is None:
        # Try to infer from column names
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
    
    # If lfc_key still not found, raise error
    if inferred_lfc_key is None:
        raise ValueError("Could not infer lfc_key from the specified run. Please specify manually.")
    
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
        If None, will try to infer from the run specified by run_id.
    condition1 : str, optional
        Name of condition 1 (denominator in fold change).
        If None, will try to infer from the run_id.
    condition2 : str, optional
        Name of condition 2 (numerator in fold change).
        If None, will try to infer from the run_id.
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
        This determines which differential abundance run's data is used.
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
    
    # Calculate the actual (positive) run ID for logging
    actual_run_id = None
    if run_id < 0:
        if 'kompot_da' in adata.uns and 'run_history' in adata.uns['kompot_da']:
            actual_run_id = len(adata.uns['kompot_da']['run_history']) + run_id
        elif 'kompot_run_history' in adata.uns:
            actual_run_id = len(adata.uns['kompot_run_history']) + run_id
        else:
            actual_run_id = run_id
    else:
        actual_run_id = run_id
    
    # Use the helper function to infer the direction column and conditions
    direction_column, inferred_condition1, inferred_condition2 = _infer_direction_key(adata, run_id, direction_column)
    
    # Use the inferred conditions if not explicitly provided
    if condition1 is None:
        condition1 = inferred_condition1
    if condition2 is None:
        condition2 = inferred_condition2
    
    # Log which run is being used and the conditions
    conditions_str = f": comparing {condition1} vs {condition2}" if condition1 and condition2 else ""
    logger.info(f"Using DA run {actual_run_id} for direction_barplot{conditions_str}")
    
    # Raise error if direction column not found
    if direction_column is None:
        raise ValueError("Could not find direction column. Please specify direction_column.")
    
    # Log the plot type and conditions first, then fields
    if condition1 and condition2:
        logger.info(f"Creating direction barplot for {condition1} vs {condition2}")
    else:
        logger.info(f"Creating direction barplot")
    
    # Log the fields being used in the plot
    logger.info(f"Using fields - category_column: '{category_column}', direction_column: '{direction_column}'")
    
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
        
        # Calculate the actual (positive) run ID for logging
        actual_run_id = None
        if run_id is not None:
            if run_id < 0:
                if 'kompot_de' in adata.uns and 'run_history' in adata.uns['kompot_de']:
                    actual_run_id = len(adata.uns['kompot_de']['run_history']) + run_id
                else:
                    actual_run_id = run_id
            else:
                actual_run_id = run_id
        
        # Get condition information from the run specified by run_id
        condition1 = None
        condition2 = None
        run_info = get_run_from_history(adata, run_id, analysis_type="de")
        
        if run_info is not None and 'params' in run_info:
            params = run_info['params']
            if 'conditions' in params and len(params['conditions']) == 2:
                condition1 = params['conditions'][0]
                condition2 = params['conditions'][1]
        
        # Try to extract from key name if still not found
        if (condition1 is None or condition2 is None) and lfc_key is not None:
            conditions = _extract_conditions_from_key(lfc_key)
            if conditions:
                condition1, condition2 = conditions
        
        # Log which run is being used
        if run_id is not None:
            conditions_str = f": comparing {condition1} vs {condition2}" if condition1 and condition2 else ""
            logger.info(f"Using DE run {actual_run_id} for heatmap{conditions_str}")
        else:
            logger.info("Using latest available DE run for heatmap")
            
        # Log the fields being used
        logger.info(f"Using fields for heatmap - lfc_key: '{lfc_key}', score_key: '{score_key}'")
        
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
    
    # Log the plot type first
    logger.info(f"Creating heatmap with {len(var_names)} genes/features")
    
    # Log the data sources being used for the heatmap
    if layer is not None and layer in adata.layers:
        logger.info(f"Using expression data from layer: '{layer}'")
        expr_matrix = adata[:, var_names].layers[layer].toarray() if hasattr(adata.layers[layer], 'toarray') else adata[:, var_names].layers[layer]
    else:
        logger.info(f"Using expression data from adata.X")
        expr_matrix = adata[:, var_names].X.toarray() if hasattr(adata.X, 'toarray') else adata[:, var_names].X
    
    # Create dataframe with expression data
    expr_df = pd.DataFrame(expr_matrix, index=adata.obs_names, columns=var_names)
    
    # Group by condition if provided
    if groupby is not None and groupby in adata.obs:
        # Calculate mean expression per group
        logger.info(f"Grouping expression by '{groupby}' ({adata.obs[groupby].nunique()} groups)")
        grouped_expr = expr_df.groupby(adata.obs[groupby]).mean()
    else:
        logger.info(f"No grouping applied (showing all {len(expr_df)} cells)")
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