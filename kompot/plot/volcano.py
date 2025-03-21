"""Volcano plot functions for visualizing differential expression and abundance results."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
from typing import Optional, Union, List, Tuple, Dict, Any, Literal
from anndata import AnnData
import pandas as pd
import warnings
import logging
import sys

from ..utils import get_run_from_history, KOMPOT_COLORS

try:
    import scanpy as sc
    _has_scanpy = True
except (ImportError, TypeError):
    # Catch both ImportError (if scanpy isn't installed) 
    # and TypeError for metaclass conflicts
    _has_scanpy = False
    
# Get the pre-configured logger
logger = logging.getLogger("kompot")


def _extract_conditions_from_key(key: str) -> Optional[Tuple[str, str]]:
    """
    Extract condition names from a key name containing 'vs'.
    
    Parameters
    ----------
    key : str
        Key name, possibly containing 'vs' between condition names
        
    Returns
    -------
    tuple or None
        (condition1, condition2) if found, None otherwise
    """
    if key is None:
        return None
        
    # Try to extract from key name, assuming format like "kompot_de_mean_lfc_Old_vs_Young"
    key_parts = key.split('_')
    if len(key_parts) >= 2 and 'vs' in key_parts:
        vs_index = key_parts.index('vs')
        if vs_index > 0 and vs_index < len(key_parts) - 1:
            condition1 = key_parts[vs_index-1]
            condition2 = key_parts[vs_index+1]
            return condition1, condition2
    return None


def _infer_de_keys(adata: AnnData, run_id: int = -1, lfc_key: Optional[str] = None, 
                   score_key: Optional[str] = None):
    """
    Infer differential expression keys from AnnData object.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object with differential expression results
    run_id : int, optional
        Run ID to use. Default is -1 (the latest run).
    lfc_key : str, optional
        Log fold change key. If provided, will be returned as is.
    score_key : str, optional
        Score key. If provided, will be returned as is unless the default
        value needs to be replaced with a run-specific key.
        
    Returns
    -------
    tuple
        (lfc_key, score_key) with the inferred keys
    """
    inferred_lfc_key = lfc_key
    inferred_score_key = score_key
    
    # If keys already provided, just return them
    if inferred_lfc_key is not None and inferred_score_key is not None:
        return inferred_lfc_key, inferred_score_key
    
    # Get run info from specified run_id - specifically from kompot_de
    run_info = get_run_from_history(adata, run_id, analysis_type="de")
    
    # If the run_info is None but a run_id was specified, log this
    if run_info is None and run_id is not None:
        logger.warning(f"Could not find run information for run_id={run_id}, analysis_type=de")
    
    if run_info is not None and 'field_names' in run_info:
        field_names = run_info['field_names']
        
        # Get lfc_key from field_names
        if inferred_lfc_key is None and 'mean_lfc_key' in field_names:
            inferred_lfc_key = field_names['mean_lfc_key']
            # Check that column exists
            if inferred_lfc_key not in adata.var.columns:
                inferred_lfc_key = None
                logger.warning(f"Found mean_lfc_key '{inferred_lfc_key}' in run info, but column not in adata.var")
        
        # Get score_key from field_names
        if inferred_score_key is None and 'mahalanobis_key' in field_names:
            inferred_score_key = field_names['mahalanobis_key']
            # Check that column exists
            if inferred_score_key not in adata.var.columns:
                logger.warning(f"Found mahalanobis_key '{inferred_score_key}' in run info, but column not in adata.var")
                inferred_score_key = None
    
    # If lfc_key still not found, raise error
    if inferred_lfc_key is None:
        raise ValueError("Could not infer lfc_key from the specified run. Please specify manually.")
    
    return inferred_lfc_key, inferred_score_key


def _infer_da_keys(adata: AnnData, run_id: int = -1, lfc_key: Optional[str] = None, 
                  pval_key: Optional[str] = None):
    """
    Infer differential abundance keys from AnnData object.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object with differential abundance results
    run_id : int, optional
        Run ID to use. Default is -1 (the latest run).
    lfc_key : str, optional
        Log fold change key. If provided, will be returned as is.
    pval_key : str, optional
        P-value key. If provided, will be returned as is.
        
    Returns
    -------
    tuple
        (lfc_key, pval_key) with the inferred keys, and a tuple of (lfc_threshold, pval_threshold)
    """
    inferred_lfc_key = lfc_key
    inferred_pval_key = pval_key
    lfc_threshold = None
    pval_threshold = None
    
    # If both keys already provided, just check for thresholds and return
    if inferred_lfc_key is not None and inferred_pval_key is not None:
        # Get run info to check for thresholds
        run_info = get_run_from_history(adata, run_id, analysis_type="da")
        if run_info is not None and 'params' in run_info:
            params = run_info['params']
            lfc_threshold = params.get('log_fold_change_threshold')
            pval_threshold = params.get('pvalue_threshold')
            
        return inferred_lfc_key, inferred_pval_key, (lfc_threshold, pval_threshold)
    
    # Get run info from specified run_id - specifically from kompot_da
    run_info = get_run_from_history(adata, run_id, analysis_type="da")
    
    if run_info is not None:
        # Check for thresholds in params
        if 'params' in run_info:
            params = run_info['params']
            lfc_threshold = params.get('log_fold_change_threshold')
            pval_threshold = params.get('pvalue_threshold')
        
        # Get field names directly from the run_info
        if 'field_names' in run_info:
            field_names = run_info['field_names']
            
            # Get lfc_key from field_names
            if inferred_lfc_key is None and 'lfc_key' in field_names:
                inferred_lfc_key = field_names['lfc_key']
                # Check that column exists
                if inferred_lfc_key not in adata.obs.columns:
                    logger.warning(f"Found lfc_key '{inferred_lfc_key}' in run info, but column not in adata.obs")
                    inferred_lfc_key = None
            
            # Get pval_key from field_names
            if inferred_pval_key is None and 'pval_key' in field_names:
                inferred_pval_key = field_names['pval_key']
                # Check that column exists
                if inferred_pval_key not in adata.obs.columns:
                    logger.warning(f"Found pval_key '{inferred_pval_key}' in run info, but column not in adata.obs")
                    inferred_pval_key = None
    
    # If keys still not found, raise error
    if inferred_lfc_key is None:
        raise ValueError("Could not infer lfc_key from the specified run. Please specify manually.")
    
    if inferred_pval_key is None:
        raise ValueError("Could not infer pval_key from the specified run. Please specify manually.")
    
    return inferred_lfc_key, inferred_pval_key, (lfc_threshold, pval_threshold)


def volcano_de(
    adata: AnnData,
    lfc_key: str = None,
    score_key: str = None,
    condition1: Optional[str] = None,
    condition2: Optional[str] = None,
    n_top_genes: int = 10,
    highlight_genes: Optional[List[str]] = None,
    show_names: bool = True,
    figsize: Tuple[float, float] = (10, 8),
    title: Optional[str] = None,
    xlabel: Optional[str] = "Log Fold Change",
    ylabel: Optional[str] = "Mahalanobis Distance",
    n_x_ticks: int = 3,
    n_y_ticks: int = 3,
    color_up: str = KOMPOT_COLORS["direction"]["up"],
    color_down: str = KOMPOT_COLORS["direction"]["down"],
    color_background: str = "gray",
    alpha_background: float = 0.4,
    point_size: float = 5,
    font_size: float = 9,
    text_offset: Tuple[float, float] = (2, 2),
    text_kwargs: Optional[Dict[str, Any]] = None,
    grid: bool = True,
    grid_kwargs: Optional[Dict[str, Any]] = None,
    ax: Optional[plt.Axes] = None,
    legend_loc: str = "best",
    legend_fontsize: Optional[float] = None,
    legend_title_fontsize: Optional[float] = None,
    show_legend: bool = True,
    sort_key: Optional[str] = None,
    return_fig: bool = False,
    save: Optional[str] = None,
    run_id: int = -1,
    legend_ncol: Optional[int] = None,
    **kwargs
) -> Union[None, Tuple[plt.Figure, plt.Axes]]:
    """
    Create a volcano plot from Kompot differential expression results.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object containing differential expression results in .var
    lfc_key : str, optional
        Key in adata.var for log fold change values.
        If None, will try to infer from ``kompot_de_`` keys.
    score_key : str, optional
        Key in adata.var for significance scores.
        Default is ``"kompot_de_mahalanobis"``
    condition1 : str, optional
        Name of condition 1 (negative log fold change)
    condition2 : str, optional
        Name of condition 2 (positive log fold change)
    n_top_genes : int, optional
        Total number of top genes to highlight and label, selected by highest Mahalanobis distance (default: 10).
        Ignored if `highlight_genes` is provided.
    highlight_genes : list of str, optional
        A list of specific gene names to highlight on the plot. If provided, this will override the `n_top_genes` parameter.
    show_names : bool, optional
        Whether to display gene names (default: True)
    figsize : tuple, optional
        Figure size as (width, height) in inches
    title : str, optional
        Plot title. If None and conditions provided, uses "{condition2} vs {condition1}"
    xlabel : str, optional
        Label for x-axis
    ylabel : str, optional
        Label for y-axis
    n_x_ticks : int, optional
        Number of ticks to display on the x-axis (default: 3)
    n_y_ticks : int, optional
        Number of ticks to display on the y-axis (default: 3)
    color_up : str, optional
        Color for up-regulated genes
    color_down : str, optional
        Color for down-regulated genes
    color_background : str, optional
        Color for background genes
    alpha_background : float, optional
        Alpha value for background genes
    point_size : float, optional
        Size of points for background genes
    font_size : float, optional
        Font size for gene labels
    text_offset : tuple, optional
        Offset (x, y) in points for gene labels from their points
    text_kwargs : dict, optional
        Additional parameters for text labels
    grid : bool, optional
        Whether to show grid lines
    grid_kwargs : dict, optional
        Additional parameters for grid
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure
    legend_loc : str, optional
        Location for the legend ('best', 'upper right', 'lower left', etc., or 'none' to hide)
    legend_fontsize : float, optional
        Font size for the legend text. If None, uses matplotlib defaults.
    legend_title_fontsize : float, optional
        Font size for the legend title. If None, uses matplotlib defaults.
    show_legend : bool, optional
        Whether to show the legend (default: True)
    legend_ncol : int, optional
        Number of columns in the legend. If None, automatically determined.
    sort_key : str, optional
        Key to sort genes by. If None, sorts by score_key
    return_fig : bool, optional
        If True, returns the figure and axes
    save : str, optional
        Path to save figure. If None, figure is not saved
    run_id : int, optional
        Specific run ID to use for fetching field names from run history.
        Negative indices count from the end (-1 is the latest run). If None, 
        uses the latest run information.
    **kwargs : 
        Additional parameters passed to plt.scatter
        
    Returns
    -------
    If return_fig is True, returns (fig, ax)
    """
    # Set default text and grid kwargs
    default_text_kwargs = {'ha': 'left', 'va': 'bottom', 'xytext': text_offset, 'textcoords': 'offset points'}
    text_kwargs = {**default_text_kwargs, **(text_kwargs or {})}
    grid_kwargs = grid_kwargs or {'alpha': 0.3}
    
    # Infer keys using helper function - this will get the right keys but won't do any logging
    lfc_key, score_key = _infer_de_keys(adata, run_id, lfc_key, score_key)
    
    # Calculate the actual (positive) run ID for logging - use same logic as volcano_da
    if run_id < 0:
        if 'kompot_de' in adata.uns and 'run_history' in adata.uns['kompot_de']:
            actual_run_id = len(adata.uns['kompot_de']['run_history']) + run_id
        else:
            actual_run_id = run_id
    else:
        actual_run_id = run_id
    
    # Only try to get conditions if they were not explicitly provided
    if condition1 is None or condition2 is None:
        # Try to extract from key name
        conditions = _extract_conditions_from_key(lfc_key)
        if conditions:
            condition1, condition2 = conditions
        else:
            # If not in key, try getting from run info
            run_info = get_run_from_history(adata, run_id, analysis_type="de")
            if run_info is not None and 'params' in run_info:
                params = run_info['params']
                if 'conditions' in params and len(params['conditions']) == 2:
                    condition1 = params['conditions'][0]
                    condition2 = params['conditions'][1]
    
    # Log which run and fields are being used
    conditions_str = f": comparing {condition1} vs {condition2}" if condition1 and condition2 else ""
    logger.info(f"Using DE run {actual_run_id}{conditions_str}")
    logger.info(f"Using fields for DE plot - lfc_key: '{lfc_key}', score_key: '{score_key}'")
    
    # Update axis labels
    if condition1 and condition2 and xlabel == "Log Fold Change":
        # Adjust for new key format where condition1 is the baseline/denominator
        xlabel = f"Log Fold Change: {condition2} / {condition1}"
                
    # Create figure if ax not provided - adjust figsize if legend is outside
    if ax is None:
        # If legend is outside and not explicitly placed elsewhere, adjust figsize
        if show_legend and (legend_loc == 'best' or legend_loc == 'center left'):
            # Increase width to accommodate legend
            adjusted_figsize = (figsize[0] * 1.3, figsize[1])
            fig, ax = plt.subplots(figsize=adjusted_figsize)
        else:
            fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Extract data for all genes
    x = adata.var[lfc_key].values
    y = adata.var[score_key].values
    
    # Plot all genes as background
    ax.scatter(x, y, alpha=alpha_background, s=point_size, c=color_background, 
              label="All genes", **kwargs)
    
    # Determine key to sort genes by
    sort_key = sort_key or score_key
    
    # Create a DataFrame with the relevant information and sort
    de_data = pd.DataFrame({
        'gene': adata.var_names,
        'lfc': adata.var[lfc_key],
        'score': adata.var[score_key],
        'sort_val': adata.var[sort_key]
    })
    
    # Determine which genes to highlight
    if highlight_genes is not None:
        # Filter for user-specified genes to highlight
        valid_genes = [g for g in highlight_genes if g in adata.var_names]
        if len(valid_genes) < len(highlight_genes):
            missing_genes = set(highlight_genes) - set(valid_genes)
            logger.warning(f"{len(missing_genes)} genes not found in the dataset: {', '.join(missing_genes)}")
        
        # Filter dataframe to only include requested genes
        top_genes = de_data[de_data['gene'].isin(valid_genes)]
        logger.info(f"Highlighting {len(top_genes)} user-specified genes")
    else:
        # Sort all genes by score (mahalanobis distance) and select top genes
        top_genes = de_data.sort_values('sort_val', ascending=False).head(n_top_genes)
        logger.info(f"Highlighting top {len(top_genes)} genes by {sort_key or score_key}")
    
    # Split into up and down regulated for display purposes
    top_up = top_genes[top_genes['lfc'] > 0]
    top_down = top_genes[top_genes['lfc'] < 0]
    
    # Plot up-regulated genes
    if len(top_up) > 0:
        ax.scatter(
            top_up['lfc'].values, 
            top_up['score'].values, 
            alpha=1, s=point_size*3, c=color_up, 
            label=f"Higher in {condition2}" if condition2 else "Up-regulated"
        )
        
        # Label top up-regulated genes
        if show_names:
            for _, gene_row in top_up.iterrows():
                ax.annotate(
                    gene_row['gene'],
                    (gene_row['lfc'], gene_row['score']),
                    fontsize=font_size, **text_kwargs
                )
    
    # Plot down-regulated genes
    if len(top_down) > 0:
        ax.scatter(
            top_down['lfc'].values,
            top_down['score'].values,
            alpha=1, s=point_size*3, c=color_down,
            label=f"Higher in {condition1}" if condition1 else "Down-regulated"
        )
        
        # Label top down-regulated genes
        if show_names:
            for _, gene_row in top_down.iterrows():
                ax.annotate(
                    gene_row['gene'],
                    (gene_row['lfc'], gene_row['score']),
                    fontsize=font_size, **text_kwargs
                )
    
    # Add formatting
    ax.axvline(x=0, color="black", linestyle="--", alpha=0.3)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    # Set the number of ticks on each axis
    if n_x_ticks > 0:
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(n_x_ticks))
    
    if n_y_ticks > 0:
        from matplotlib.ticker import MaxNLocator
        ax.yaxis.set_major_locator(MaxNLocator(n_y_ticks))
    
    # Set title if provided or can be inferred
    if title is None and condition1 and condition2:
        title = f"Volcano Plot: {condition1} vs {condition2}"
    if title:
        ax.set_title(title, fontsize=14)
    
    # Add legend with appropriate styling
    if show_legend and legend_loc != 'none':
        # Default to bbox_to_anchor outside the plot if legend_loc is not explicitly specified
        if legend_loc == 'best':
            legend = ax.legend(
                bbox_to_anchor=(1.05, 1), 
                loc='upper left', 
                fontsize=legend_fontsize,
                frameon=False,
                ncol=legend_ncol or 1
            )
            # Adjust figure layout to accommodate legend
            plt.tight_layout(rect=[0, 0, 0.85, 1])
        else:
            legend = ax.legend(
                loc=legend_loc, 
                fontsize=legend_fontsize,
                frameon=False,
                ncol=legend_ncol or 1
            )
    
    if grid:
        ax.grid(**grid_kwargs)
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
    
    # Return figure and axes if requested
    if return_fig:
        return fig, ax
    elif save is None:
        # Only show if not saving and not returning
        plt.show()


def volcano_da(
    adata: AnnData,
    lfc_key: Optional[str] = None,
    pval_key: Optional[str] = None, 
    group_key: Optional[str] = None,
    log_transform_pval: bool = True,
    lfc_threshold: Optional[float] = None,
    pval_threshold: Optional[float] = 0.05,
    color: Optional[Union[str, List[str]]] = None,
    alpha_background: float = 1.0,  # No alpha by default
    highlight_subset: Optional[Union[np.ndarray, List[bool]]] = None,
    highlight_color: str = KOMPOT_COLORS["direction"]["up"],
    figsize: Tuple[float, float] = (10, 8),
    title: Optional[str] = "Differential Abundance Volcano Plot",
    xlabel: Optional[str] = "Log Fold Change",
    ylabel: Optional[str] = "-log10(p-value)",
    n_x_ticks: int = 3,
    n_y_ticks: int = 3,
    legend_loc: str = "best",
    legend_fontsize: Optional[float] = None,
    legend_title_fontsize: Optional[float] = None,
    show_legend: bool = True,
    grid: bool = True,
    grid_kwargs: Optional[Dict[str, Any]] = None,
    ax: Optional[plt.Axes] = None,
    palette: Optional[Union[str, List[str], Dict[str, str]]] = None,
    save: Optional[str] = None,
    show: bool = None,
    return_fig: bool = False,
    run_id: int = -1,
    legend_ncol: Optional[int] = None,
    update_direction: bool = False,
    direction_column: Optional[str] = None,
    show_thresholds: bool = True,
    show_colorbar: bool = True,  # Whether to show colorbar for numeric columns
    cmap: Optional[Union[str, Colormap]] = None,  # Colormap for numeric values
    vcenter: Optional[float] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    **kwargs
) -> Union[None, Tuple[plt.Figure, plt.Axes]]:
    """
    Create a volcano plot for differential abundance results.
    
    This function visualizes cells in a 2D volcano plot with log fold change on the x-axis
    and significance (-log10 p-value) on the y-axis. Cells can be colored by any column
    in adata.obs.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object containing differential abundance results
    lfc_key : str, optional
        Key in adata.obs for log fold change values.
        If None, will try to infer from ``kompot_da_`` keys.
    pval_key : str, optional
        Key in adata.obs for p-values.
        If None, will try to infer from ``kompot_da_`` keys.
    group_key : str, optional
        Key in adata.obs to group cells by (for coloring)
    log_transform_pval : bool, optional
        Whether to -log10 transform p-values for the y-axis
    lfc_threshold : float, optional
        Log fold change threshold for significance (for drawing threshold lines)
    pval_threshold : float, optional
        P-value threshold for significance (for drawing threshold lines)
    color : str or list of str, optional
        Keys in adata.obs for coloring cells. Requires scanpy.
    alpha_background : float, optional
        Alpha value for background cells (below threshold). Default is 1.0 (no transparency)
    highlight_subset : array or list, optional
        Boolean mask to highlight specific cells
    highlight_color : str, optional
        Color for highlighted cells
    figsize : tuple, optional
        Figure size as (width, height) in inches
    title : str, optional
        Plot title
    xlabel : str, optional
        Label for x-axis
    ylabel : str, optional
        Label for y-axis
    n_x_ticks : int, optional
        Number of ticks to display on the x-axis (default: 3)
    n_y_ticks : int, optional
        Number of ticks to display on the y-axis (default: 3)
    legend_loc : str, optional
        Location for the legend ('best', 'upper right', 'lower left', etc., or 'none' to hide)
    legend_fontsize : float, optional
        Font size for the legend text. If None, uses matplotlib defaults.
    legend_title_fontsize : float, optional
        Font size for the legend title. If None, uses matplotlib defaults.
    show_legend : bool, optional
        Whether to show the legend (default: True)
    grid : bool, optional
        Whether to show grid lines
    grid_kwargs : dict, optional
        Additional parameters for grid
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure
    palette : str, list, or dict, optional
        Color palette to use for categorical coloring
    legend_ncol : int, optional
        Number of columns in the legend. If None, automatically determined based on the
        number of categories.
    save : str, optional
        Path to save figure. If None, figure is not saved
    show : bool, optional
        Whether to show the plot
    return_fig : bool, optional
        If True, returns the figure and axes
    run_id : int, optional
        Specific run ID to use for fetching field names from run history.
        Negative indices count from the end (-1 is the latest run). If None, 
        uses the latest run information.
    update_direction : bool, optional
        Whether to update the direction column based on the provided thresholds
        before plotting (default: False)
    direction_column : str, optional
        Direction column to update if update_direction=True. If None, infers
        from run_id.
    show_thresholds : bool, optional
        Whether to display horizontal and vertical threshold lines (default: True).
        Set to False to hide threshold lines.
    show_colorbar : bool, optional
        Whether to display colorbar for numeric color columns (default: True).
        Set to False to hide colorbar.
    condition1 : str, optional
        Name of condition 1 (denominator in fold change)
    condition2 : str, optional
        Name of condition 2 (numerator in fold change)
    **kwargs : 
        Additional parameters passed to plt.scatter
        
    Returns
    -------
    If return_fig is True, returns (fig, ax)
    """
    # Set default grid kwargs
    grid_kwargs = grid_kwargs or {'alpha': 0.3}
    
    # Infer keys using helper function
    lfc_key, pval_key, thresholds = _infer_da_keys(adata, run_id, lfc_key, pval_key)
    
    # Calculate the actual (positive) run ID for logging
    if run_id < 0:
        if 'kompot_da' in adata.uns and 'run_history' in adata.uns['kompot_da']:
            actual_run_id = len(adata.uns['kompot_da']['run_history']) + run_id
        else:
            actual_run_id = run_id
    else:
        actual_run_id = run_id
    
    # Extract the threshold values
    auto_lfc_threshold, auto_pval_threshold = thresholds
    
    # Track which values needed inference for logging
    needed_column_inference = lfc_key is None or pval_key is None
    needed_threshold_inference = False
    
    # Use run thresholds if available and not explicitly overridden
    if lfc_threshold is None and auto_lfc_threshold is not None:
        lfc_threshold = auto_lfc_threshold
        needed_threshold_inference = True
    
    if pval_threshold is None and auto_pval_threshold is not None:
        pval_threshold = auto_pval_threshold
        needed_threshold_inference = True
        
    # Update direction column if requested
    if update_direction:
        from ..differential.utils import update_direction_column as update_dir
        logger.info(f"Updating direction column with new thresholds before plotting")
        update_dir(
            adata=adata,
            lfc_threshold=lfc_threshold,
            pval_threshold=pval_threshold,
            direction_column=direction_column,
            lfc_key=lfc_key,
            pval_key=pval_key,
            run_id=run_id,
            inplace=True
        )
    
    # Get condition information from the run specified by run_id
    run_info = get_run_from_history(adata, run_id, analysis_type="da")
    condition1 = None
    condition2 = None
    
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
    
    # Log appropriate information based on what needed to be inferred
    if needed_column_inference:
        conditions_str = f": comparing {condition1} vs {condition2}" if condition1 and condition2 else ""
        logger.info(f"Inferred DA columns from run {actual_run_id}{conditions_str}")
        logger.info(f"Using fields for DA plot - lfc_key: '{lfc_key}', pval_key: '{pval_key}'")
    
    if needed_threshold_inference:
        logger.info(f"Using inferred thresholds - lfc_threshold: {lfc_threshold}, pval_threshold: {pval_threshold}")
    
    # Update axis labels with condition information if not explicitly set
    if condition1 and condition2 and xlabel == "Log Fold Change":
        # Adjust for new key format where condition1 is the baseline/denominator
        xlabel = f"Log Fold Change: {condition2} / {condition1}"
    
    # Create figure if ax not provided - adjust figsize if legend is outside
    if ax is None:
        # If legend is outside and not explicitly placed elsewhere, adjust figsize
        if show_legend and legend_loc == 'best':
            # Increase width to accommodate legend
            adjusted_figsize = (figsize[0] * 1.3, figsize[1])
            fig, ax = plt.subplots(figsize=adjusted_figsize)
        else:
            fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Extract data
    x = adata.obs[lfc_key].values
    
    # Handle p-values - check if they're already negative log10 transformed
    if 'neg_log10' in pval_key.lower() or pval_key.lower().startswith('neg_log10') or '-log10' in pval_key.lower():
        # Already negative log10 transformed - use as is (values should be positive)
        y = adata.obs[pval_key].values
        ylabel = ylabel or "-log10(p-value)"
        log_transform_pval = False  # Override since already transformed
    elif log_transform_pval:
        y = -np.log10(adata.obs[pval_key].values)
        ylabel = ylabel or "-log10(p-value)"
    else:
        y = adata.obs[pval_key].values
        ylabel = ylabel or "p-value"
    
    # Define significance thresholds for coloring
    if pval_threshold is not None:
        if 'neg_log10' in pval_key.lower() or pval_key.lower().startswith('neg_log10') or '-log10' in pval_key.lower():
            # Key indicates values are already negative log10 transformed (higher = more significant)
            # Convert pval_threshold to -log10 scale if it's a raw p-value (between 0 and 1)
            if 0 < pval_threshold < 1:
                y_threshold = -np.log10(pval_threshold)  # Convert to -log10 scale
            else:
                # Assume it's already on -log10 scale
                y_threshold = pval_threshold
        elif log_transform_pval:
            y_threshold = -np.log10(pval_threshold)
        else:
            y_threshold = pval_threshold
    else:
        y_threshold = None
    
    # Define masks for significant cells
    if pval_threshold is not None and lfc_threshold is not None:
        # Both thresholds provided
        significant = (y > y_threshold) & (np.abs(x) > lfc_threshold)
    elif pval_threshold is not None:
        # Only p-value threshold provided
        significant = y > y_threshold
    elif lfc_threshold is not None:
        # Only LFC threshold provided
        significant = np.abs(x) > lfc_threshold
    else:
        # No thresholds provided
        significant = np.ones(len(x), dtype=bool)
    
    # Apply custom highlight mask if provided
    if highlight_subset is not None:
        significant = highlight_subset
    
    # First plot all cells as background
    scatter_kwargs = {'s': 10}  # Default point size
    scatter_kwargs.update(kwargs)
    
    ax.scatter(
        x, y, 
        alpha=alpha_background, 
        c="lightgray", 
        label="Non-significant",
        **scatter_kwargs
    )
    
    # Color significant cells
    if color is not None:
        if not _has_scanpy:
            warnings.warn(
                "Scanpy is required for coloring cells by obs columns. "
                "Falling back to default coloring. Install scanpy to use this feature."
            )
            # Default coloring without scanpy
            ax.scatter(
                x[significant], y[significant], 
                alpha=1, 
                c=highlight_color, 
                label="Significant",
                **scatter_kwargs
            )
        else:
            # We'll handle coloring manually instead of using scanpy's scatter
            # Use matplotlib directly instead of seaborn
            from matplotlib.colors import ListedColormap, Normalize
            
            # Get the significant indices
            sig_indices = np.where(significant)[0]
            
            if isinstance(color, str):
                color = [color]
                
            for c in color:
                if c not in adata.obs:
                    warnings.warn(f"Color key '{c}' not found in adata.obs. Skipping.")
                    continue
                
                # Check if the color column is categorical or string
                if not pd.api.types.is_categorical_dtype(adata.obs[c]):
                    # If column contains string data, convert to categorical with warning
                    if pd.api.types.is_string_dtype(adata.obs[c]) or pd.api.types.is_object_dtype(adata.obs[c]):
                        warnings.warn(f"Color column '{c}' contains string data but is not categorical. "
                                     f"Converting to categorical for proper coloring.")
                        adata.obs[c] = adata.obs[c].astype('category')
                
                # Get the color values for the significant points
                color_values = adata.obs[c].values[sig_indices]
                
                # Check if the color column is categorical
                if pd.api.types.is_categorical_dtype(adata.obs[c]):
                    categories = adata.obs[c].cat.categories
                    
                    # Check if colors are stored in adata.uns with f"{color}_colors" format
                    colors_key = f"{c}_colors"
                    if colors_key in adata.uns and len(adata.uns[colors_key]) == len(categories):
                        # Use stored colors from adata.uns
                        stored_colors = adata.uns[colors_key]
                        color_dict = dict(zip(categories, stored_colors))
                        logger.debug(f"Using colors from adata.uns['{colors_key}']")
                    # Otherwise, use palette or generate colors
                    elif isinstance(palette, str):
                        # Use matplotlib colormaps instead of seaborn
                        cmap = plt.cm.get_cmap(palette, len(categories))
                        colors = [cmap(i/len(categories)) for i in range(len(categories))]
                        color_dict = dict(zip(categories, colors))
                        # Store colors in adata.uns for future use
                        adata.uns[colors_key] = colors
                        logger.debug(f"Created and stored colors in adata.uns['{colors_key}']")
                    elif isinstance(palette, dict):
                        color_dict = palette
                    else:
                        # Use default palette - tab10 equivalent
                        cmap = plt.cm.get_cmap('tab10', len(categories))
                        colors = [cmap(i/len(categories)) for i in range(len(categories))]
                        color_dict = dict(zip(categories, colors))
                        # Store colors in adata.uns for future use
                        adata.uns[colors_key] = colors
                        logger.debug(f"Created and stored colors in adata.uns['{colors_key}']")
                    
                    # Plot each category separately
                    for cat in categories:
                        cat_mask = color_values == cat
                        if np.sum(cat_mask) > 0:
                            cat_color = color_dict.get(cat, highlight_color)
                            ax.scatter(
                                x[sig_indices][cat_mask], 
                                y[sig_indices][cat_mask],
                                alpha=1,
                                c=[cat_color],
                                label=f"{cat}",
                                **scatter_kwargs
                            )
                    
                    # Add legend for categorical data
                    if show_legend and legend_loc != 'none':
                        # Count number of categories to determine if we need multicolumn layout
                        num_categories = len([c for c in categories if c in color_values])
                        
                        # Use provided legend_ncol if specified, otherwise auto-determine
                        if legend_ncol is not None:
                            ncol = legend_ncol
                        # Determine if we need a multicolumn layout (more than 10 categories)
                        elif num_categories > 10:
                            ncol = max(2, min(5, num_categories // 10))  # Use 2-5 columns based on count
                        else:
                            ncol = 1
                            
                        # Default to bbox_to_anchor outside the plot if legend_loc is not explicitly specified
                        if legend_loc == 'best':
                            legend = ax.legend(
                                bbox_to_anchor=(1.05, 1), 
                                loc='upper left', 
                                title=c, 
                                frameon=False,
                                fontsize=legend_fontsize,
                                ncol=ncol
                            )
                        else:
                            legend = ax.legend(
                                loc=legend_loc, 
                                title=c, 
                                frameon=False, 
                                fontsize=legend_fontsize,
                                ncol=ncol
                            )
                        
                        # Set frame properties only if it's explicitly needed
                        # legend.get_frame().set_facecolor('white')
                        # legend.get_frame().set_alpha(0.8)
                        
                        # Set legend title font size if specified
                        if legend_title_fontsize is not None and legend.get_title():
                            legend.get_title().set_fontsize(legend_title_fontsize)
                            
                        # If legend is outside, adjust the figure layout
                        if legend_loc == 'best':
                            plt.tight_layout(rect=[0, 0, 0.85, 1])
                else:
                    # For numeric columns, use a colormap
                    # Default to Spectral_r if no palette specified
                    scatter_kwargs_color = scatter_kwargs.copy()
                    use_cmap = cmap if 'cmap' in kwargs else (palette if isinstance(palette, str) else "Spectral_r")
                    scatter_kwargs_color['cmap'] = use_cmap
                    
                    scatter = ax.scatter(
                        x[sig_indices],
                        y[sig_indices],
                        alpha=1,
                        c=color_values,
                        **scatter_kwargs_color
                    )
                    # Only add colorbar if show_colorbar is True
                    if show_colorbar:
                        plt.colorbar(scatter, ax=ax, label=c)
    else:
        # Default coloring without color key
        ax.scatter(
            x[significant], y[significant], 
            alpha=1, 
            c=highlight_color, 
            label="Significant",
            **scatter_kwargs
        )
    
    # Add threshold lines if requested
    if show_thresholds:
        if lfc_threshold is not None:
            ax.axvline(x=lfc_threshold, color="black", linestyle="--", alpha=0.5)
            ax.axvline(x=-lfc_threshold, color="black", linestyle="--", alpha=0.5)
        
        if pval_threshold is not None:
            if 'neg_log10' in pval_key.lower() or pval_key.lower().startswith('neg_log10') or '-log10' in pval_key.lower():
                # For negative log10 p-values, convert if needed
                if 0 < pval_threshold < 1:
                    ax.axhline(y=-np.log10(pval_threshold), color="black", linestyle="--", alpha=0.5)
                else:
                    ax.axhline(y=pval_threshold, color="black", linestyle="--", alpha=0.5)
            elif log_transform_pval:
                ax.axhline(y=-np.log10(pval_threshold), color="black", linestyle="--", alpha=0.5)
            else:
                ax.axhline(y=pval_threshold, color="black", linestyle="--", alpha=0.5)
    
    # Add center line if requested (unchanged from previous behavior)
    if show_thresholds:
        ax.axvline(x=0, color="black", linestyle="-", alpha=0.3)
    
    # Add labels and title
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    # Set the number of ticks on each axis
    if n_x_ticks > 0:
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(n_x_ticks))
    
    if n_y_ticks > 0:
        from matplotlib.ticker import MaxNLocator
        ax.yaxis.set_major_locator(MaxNLocator(n_y_ticks))
        
    if title:
        ax.set_title(title, fontsize=14)
    
    # Add legend for non-categorical coloring
    if color is None and show_legend and legend_loc != 'none':
        # Default to bbox_to_anchor outside the plot if legend_loc is not explicitly specified
        if legend_loc == 'best':
            legend = ax.legend(
                bbox_to_anchor=(1.05, 1), 
                loc='upper left', 
                fontsize=legend_fontsize,
                frameon=False
            )
            # Adjust figure layout to accommodate legend
            plt.tight_layout(rect=[0, 0, 0.85, 1])
        else:
            legend = ax.legend(
                loc=legend_loc, 
                fontsize=legend_fontsize,
                frameon=False
            )
    
    # Add grid
    if grid:
        ax.grid(**grid_kwargs)
    
    # Don't use tight_layout as it may interfere with multi-panel plots
    # Instead, use proper spacing when in a multi-plot context
    if ax.get_figure().get_axes() == [ax]:  # Only adjust if this is the only plot
        fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    
    # Save figure if path provided
    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight")
    
    # Show or return
    if return_fig:
        return fig, ax
    elif show or (show is None and save is None):
        plt.show()


def multi_volcano_da(
    adata: AnnData,
    groupby: str,
    lfc_key: Optional[str] = None,
    pval_key: Optional[str] = None,
    log_transform_pval: bool = True,
    lfc_threshold: Optional[float] = None,
    pval_threshold: Optional[float] = 0.05, 
    color: Optional[Union[str, List[str]]] = None,
    alpha_background: float = 1.0,  # No alpha by default
    highlight_subset: Optional[Union[np.ndarray, List[bool]]] = None,
    highlight_color: str = KOMPOT_COLORS["direction"]["up"],
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = "Differential Abundance Volcano Plot",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = "-log10(p-value)",
    n_x_ticks: int = 3,
    n_y_ticks: int = 0,  # By default do not show y-ticks
    legend_loc: str = "bottom",  # Default to bottom placement
    legend_fontsize: Optional[float] = None,
    legend_title_fontsize: Optional[float] = None,
    show_legend: Optional[bool] = None,
    grid: bool = True,
    grid_kwargs: Optional[Dict[str, Any]] = None,
    palette: Optional[Union[str, List[str], Dict[str, str]]] = None,
    show_thresholds: bool = False,
    plot_width_factor: float = 10.0,  # Default width factor - plots are 10x wider than tall
    share_y: bool = True,  # Share y-axis by default
    layout_config: Optional[Dict[str, float]] = None,  # Configuration for layout spacing
    save: Optional[str] = None,
    show: bool = None,
    return_fig: bool = False,
    run_id: int = -1,
    update_direction: bool = False,
    direction_column: Optional[str] = None,
    cmap: Optional[Union[str, Colormap]] = None,  # Use standard matplotlib name
    vcenter: Optional[float] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    **kwargs
) -> Union[None, Tuple[plt.Figure, List[plt.Axes]]]:
    """
    Create multiple volcano plots for differential abundance results, one per group.
    
    This function creates a panel of volcano plots, one for each unique value in the groupby column.
    Each plot is wider than tall (by default 10x wider than tall) and is aligned with other plots.
    Only the bottom plot shows x-axis labels and ticks, only the middle plot shows the y-axis label,
    and y-axis ticks are hidden for all plots. Group labels are placed to the right of each plot,
    aligned with the plot edge. Each plot has a box outline by default, and points are drawn 
    with full opacity (no transparency). If the color and groupby columns are identical, the 
    legend is hidden. Vertical lines (both threshold and center line at 0) are hidden by 
    default but can be enabled with show_thresholds=True.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object containing differential abundance results
    groupby : str
        Column in adata.obs to group cells by (for separating into multiple plots)
    lfc_key : str, optional
        Key in adata.obs for log fold change values.
        If None, will try to infer from ``kompot_da_`` keys.
    pval_key : str, optional
        Key in adata.obs for p-values.
        If None, will try to infer from ``kompot_da_`` keys.
    log_transform_pval : bool, optional
        Whether to -log10 transform p-values for the y-axis
    lfc_threshold : float, optional
        Log fold change threshold for significance (for drawing threshold lines)
    pval_threshold : float, optional
        P-value threshold for significance (for drawing threshold lines)
    color : str or list of str, optional
        Keys in adata.obs for coloring cells. Requires scanpy.
        If identical to groupby, the legend will be hidden.
    alpha_background : float, optional
        Alpha value for background cells (below threshold). Default is 1.0 (no transparency)
    highlight_subset : array or list, optional
        Boolean mask to highlight specific cells
    highlight_color : str, optional
        Color for highlighted cells
    figsize : tuple, optional
        Figure size as (width, height) in inches. If None, it will be calculated
        automatically based on the number of groups and layout parameters.
    title : str, optional
        Plot title
    xlabel : str, optional
        Label for x-axis (only shown on bottom plot). If None, it will be automatically 
        generated based on condition names extracted from lfc_key if available.
    ylabel : str, optional
        Label for y-axis (only shown on middle plot)
    n_x_ticks : int, optional
        Number of ticks to display on the x-axis (default: 3)
    n_y_ticks : int, optional
        Number of ticks to display on the y-axis (default: 0, no y-ticks)
    legend_loc : str, optional
        Location for the legend ('bottom', 'right', 'best', 'upper right', etc.)
    legend_fontsize : float, optional
        Font size for the legend text
    legend_title_fontsize : float, optional
        Font size for the legend title
    show_legend : bool, optional
        Whether to show the legend. If None (default), legend will be shown except when 
        color column is identical to groupby column. If explicitly set to True or False, 
        this setting will override the automatic behavior.
    grid : bool, optional
        Whether to show grid lines
    grid_kwargs : dict, optional
        Additional parameters for grid
    palette : str, list, or dict, optional
        Color palette to use for categorical coloring
    show_thresholds : bool, optional
        Whether to display threshold lines on the plots (default: False)
    show_colorbar : bool, optional
        Whether to display colorbars in individual volcano plots (default: False in multi_volcano_da)
    plot_width_factor : float, optional
        Width factor for each volcano plot. Higher values make plots wider relative to their height.
        Default is 10.0 (plots are 10x wider than tall). This is maintained regardless of
        the number of groups.
    share_y : bool, optional
        Whether to use the same y-axis limits for all plots (default: True)
    layout_config : dict, optional
        Configuration for controlling plot layout spacing. Keys include:
        - 'unit_size': Base unit size in inches (default: 0.15)
        - 'title_height': Height for title area in units (default: 2)
        - 'legend_bottom_margin': Distance from bottom of figure to legend/colorbar in units (default: 3)
        - 'legend_plot_gap': Gap between last plot and legend/colorbar in units (default: 3)
        - 'legend_height': Minimum height for legend/colorbar area in units (default: 3)
        - 'plot_height': Height for each plot in units (default: 4)
        - 'plot_width': Width for each plot in units (default: plot_width_factor * plot_height)
        - 'label_width': Width for group labels in units (default: 4)
        - 'top_margin': Top margin in units (default: 1)
        - 'plot_spacing': Spacing between plots in units (default: 0.2)
        - 'y_label_width': Width for y-axis label in units (default: 2)
        - 'y_label_offset': Offset of y-axis label from plots in units (default: 0.5)
    save : str, optional
        Path to save figure. If None, figure is not saved
    show : bool, optional
        Whether to show the plot
    return_fig : bool, optional
        If True, returns the figure and axes
    run_id : int, optional
        Specific run ID to use for fetching field names from run history
    update_direction : bool, optional
        Whether to update the direction column based on the provided thresholds
        before plotting (default: False). This is only applied once to the full dataset,
        not to individual group subsets.
    direction_column : str, optional
        Direction column to update if update_direction=True. If None, infers
        from run_id.
    cmap : str or matplotlib.cm.Colormap, optional
        Colormap to use for numeric color values. If not provided, automatically selects 
        'RdBu_r' with vcenter=0 for columns containing 'log_fold_change' or 'lfc',
        otherwise defaults to "Spectral_r".
    vcenter : float, optional
        Value to center the colormap at. Only applies to diverging colormaps.
        If not specified but a column containing 'log_fold_change' or 'lfc' is used
        for coloring, defaults to 0.
    vmin : float, optional
        Minimum value for the colormap. If not provided, uses the minimum value in the data.
    vmax : float, optional
        Maximum value for the colormap. If not provided, uses the maximum value in the data.
    **kwargs : 
        Additional parameters passed to plt.scatter
        
    Returns
    -------
    If return_fig is True, returns (fig, axes_list)
    """
    # Get the unique groups
    if groupby not in adata.obs.columns:
        raise ValueError(f"Group column '{groupby}' not found in adata.obs")
    
    groups = adata.obs[groupby].unique()
    n_groups = len(groups)
    
    if n_groups == 0:
        raise ValueError(f"No groups found in column '{groupby}'")
        
    # Sort groups if they're strings or numbers
    if all(isinstance(g, (str, int, float)) for g in groups):
        groups = sorted(groups)
    
    # Infer keys using helper function
    lfc_key, pval_key, thresholds = _infer_da_keys(adata, run_id, lfc_key, pval_key)
    
    # Extract the threshold values
    auto_lfc_threshold, auto_pval_threshold = thresholds
    
    # Try to extract conditions from the key name for better labeling
    condition_names = _extract_conditions_from_key(lfc_key)
    condition1, condition2 = condition_names if condition_names else (None, None)
    
    # Track which values needed inference for logging
    needed_column_inference = lfc_key is None or pval_key is None
    needed_threshold_inference = False
    
    # Use run thresholds if available and not explicitly overridden
    if lfc_threshold is None and auto_lfc_threshold is not None:
        lfc_threshold = auto_lfc_threshold
        needed_threshold_inference = True
    
    if pval_threshold is None and auto_pval_threshold is not None:
        pval_threshold = auto_pval_threshold
        needed_threshold_inference = True
    
    # Log appropriate information based on what needed to be inferred
    if needed_column_inference:
        logger.info(f"Inferred columns for multi-volcano plot: lfc_key='{lfc_key}', pval_key='{pval_key}'")
    
    if needed_threshold_inference:
        logger.info(f"Using inferred thresholds - lfc_threshold: {lfc_threshold}, pval_threshold: {pval_threshold}")
    
    logger.info(f"Creating volcano plots for groups: {', '.join(map(str, groups))}")
    
    # Update direction for the entire dataset if requested (do this only once!)
    if update_direction:
        from ..differential.utils import update_direction_column as update_dir
        logger.info(f"Updating direction column with new thresholds before plotting")
        update_dir(
            adata=adata,
            lfc_threshold=lfc_threshold,
            pval_threshold=pval_threshold,
            direction_column=direction_column,
            lfc_key=lfc_key,
            pval_key=pval_key,
            run_id=run_id,
            inplace=True
        )
    
    # Set up default layout config using a unit-based system
    default_layout = {
        'unit_size': 0.15,                       # Base unit size in inches
        'title_height': 2,                       # Height for title area in units
        'legend_bottom_margin': 3,               # Distance from bottom of figure to legend/colorbar
        'legend_plot_gap': 3,                    # Gap between last plot and legend/colorbar
        'legend_height': 3,                      # Minimum height for legend/colorbar area
        'plot_height': 4,                        # Height for each plot in units
        'plot_width': plot_width_factor * 4,     # Width for each plot in units
        'label_width': 4,                        # Width for group labels in units
        'top_margin': 1,                         # Top margin in units
        'plot_spacing': 0.2,                     # Spacing between plots in units
        'y_label_width': 2,                      # Width for y-axis label in units
        'y_label_offset': 0.5,                   # Offset of y-axis label from plots
    }
    
    # Update with user-provided config if any
    if layout_config:
        default_layout.update(layout_config)
    
    # Store the layout for later use
    layout = default_layout
    unit = layout['unit_size']  # Base unit size in inches
    
    # Calculate figure dimensions based on the unit system
    total_width_units = layout['y_label_width'] + layout['y_label_offset'] + layout['plot_width'] + layout['label_width']
    plot_area_height = (layout['plot_height'] * n_groups) + (layout['plot_spacing'] * (n_groups - 1))
    
    # Calculate legend area dynamically
    legend_height = layout['legend_height']  # Start with minimum height
    
    # Total height calculation
    total_height_units = (
        layout['title_height'] +      # Title area
        plot_area_height +            # All plots with spacing
        layout['legend_plot_gap'] +   # Gap between plots and legend
        legend_height +               # Legend area
        layout['top_margin'] +        # Top margin
        layout['legend_bottom_margin'] # Bottom margin
    )
    
    # Convert to inches
    width_inches = total_width_units * unit
    height_inches = total_height_units * unit
    
    # Use provided figsize if specified, otherwise use calculated dimensions
    if figsize:
        fig = plt.figure(figsize=figsize)
    else:
        fig = plt.figure(figsize=(width_inches, height_inches))
    
    # Calculate positions in figure coordinates (0-1)
    fig_width, fig_height = fig.get_size_inches()
    
    # Calculate the main plot grid area in normalized coordinates (0-1)
    plot_area_height_norm = plot_area_height * unit / fig_height
    
    # Calculate the bottom position of the plot area
    legend_area_height_norm = (legend_height + layout['legend_bottom_margin']) * unit / fig_height
    main_bottom = legend_area_height_norm + (layout['legend_plot_gap'] * unit / fig_height)
    main_top = main_bottom + plot_area_height_norm
    
    # Calculate left position for the main plot area (after y-label)
    y_label_width_norm = (layout['y_label_width'] + layout['y_label_offset']) * unit / fig_width
    
    # Define the grid layout
    # Use 3 columns: [main plot column, group label column]
    
    # Calculate height ratios for each plot + spacing
    height_ratios = []
    for i in range(n_groups):
        height_ratios.append(layout['plot_height'])
        if i < n_groups - 1:
            height_ratios.append(layout['plot_spacing'])
    
    # Set up GridSpec with alternating plot and spacing rows
    total_rows = 2 * n_groups - 1 if n_groups > 1 else 1
    gs = fig.add_gridspec(
        total_rows, 2,
        left=y_label_width_norm,  # Start after y-label area
        right=0.95,              # Fixed right margin
        bottom=main_bottom,      # Bottom of plot area
        top=main_top,            # Top of plot area
        height_ratios=height_ratios,
        width_ratios=[0.85, 0.15],  # [plot, label]
        wspace=0.0  # No spacing between columns
    )
    
    # Create all the axes at once
    axes = []
    group_label_axes = []
    
    # Store the first axes as reference for x and y sharing
    shared_x = None
    shared_y = None
    
    # Create all plot and label axes first
    for i, group in enumerate(groups):
        # Calculate row index (accounting for spacing rows)
        row_idx = i * 2 if i > 0 else 0
        
        # Create plot axis
        if i == 0:
            plot_ax = fig.add_subplot(gs[row_idx, 0])  # Main plot column
            shared_x = plot_ax  # First plot becomes the x-axis reference
            if share_y:
                shared_y = plot_ax  # Also use as y-axis reference if sharing
        else:
            if share_y:
                plot_ax = fig.add_subplot(gs[row_idx, 0], sharex=shared_x, sharey=shared_y)
            else:
                plot_ax = fig.add_subplot(gs[row_idx, 0], sharex=shared_x)
        
        axes.append(plot_ax)  # Store for later reference
        
        # Create a label axis in the labels column
        label_ax = fig.add_subplot(gs[row_idx, 1])
        label_ax.axis('off')  # Hide axis elements
        label_ax.text(
            0.1,  # Left-aligned within the cell
            0.5,  # Vertically centered
            f"{group}",
            ha='left',
            va='center',
            fontsize=12,
            transform=label_ax.transAxes  # Use axis coordinates (0-1)
        )
        group_label_axes.append(label_ax)
    
    # Create a single y-label axis if needed
    if ylabel and n_groups > 0:
        # Calculate the center position of the plot area
        y_center = (main_top + main_bottom) / 2
        
        # Position the y-label to the left of the plot area
        y_label_left = y_label_width_norm - (layout['y_label_width'] * unit / fig_width)
        y_label_width = layout['y_label_width'] * unit / fig_width
        y_label_height = 0.2  # Fixed height in normalized coordinates
        
        # Create a dedicated axis for the y-label
        y_label_ax = fig.add_axes([y_label_left, y_center - (y_label_height/2), y_label_width, y_label_height])
        y_label_ax.axis('off')
        
        # Add the y-label text
        y_label_ax.text(
            0.5, 0.5, ylabel,
            ha='center', va='center',
            fontsize=12, rotation=90,
            transform=y_label_ax.transAxes
        )
    
    # Add overall title at the top if provided
    if title:
        # Create a dedicated title axis at the top
        title_top = 1.0
        title_height = layout['title_height'] * unit / fig_height
        title_ax = fig.add_axes([0, 1.0 - title_height, 1, title_height])
        title_ax.axis('off')  # Hide axis elements
        
        # Add title text to this axis
        title_ax.text(
            0.5, 0.5, title,
            ha='center', va='center',
            fontsize=14, fontweight='bold',
            transform=title_ax.transAxes
        )
    
    # Set up legend and colorbar tracking variables
    all_handles = []
    all_labels = []
    colorbar_needed = False
    first_color_mappable = None
    colorbar_label = None
    
    # Calculate global color scale limits if color is provided and is numeric
    global_vmin = None
    global_vmax = None
    
    if color is not None and isinstance(color, str) and color in adata.obs and not pd.api.types.is_categorical_dtype(adata.obs[color]):
        # For numeric color values, calculate global min/max across all data points for consistent coloring
        if vmin is None:
            global_vmin = np.nanmin(adata.obs[color].values)
        if vmax is None:
            global_vmax = np.nanmax(adata.obs[color].values)
    
    # Draw volcano plots in each axis
    for i, group in enumerate(groups):
        plot_ax = axes[i]
        
        # Create a mask for the current group
        mask = adata.obs[groupby] == group
        
        # Extract data directly from the masked anndata
        # No need to create a copy - use a view
        x = adata.obs[lfc_key].values[mask]
        
        # Handle p-values - check if they're already negative log10 transformed
        if 'neg_log10' in pval_key.lower() or pval_key.lower().startswith('neg_log10') or '-log10' in pval_key.lower():
            # Already negative log10 transformed - use as is (values should be positive)
            y = adata.obs[pval_key].values[mask]
            y_label = "-log10(p-value)"
            log_transform_pval_now = False  # Override since already transformed
        elif log_transform_pval:
            y = -np.log10(adata.obs[pval_key].values[mask])
            y_label = "-log10(p-value)"
            log_transform_pval_now = True
        else:
            y = adata.obs[pval_key].values[mask]
            y_label = "p-value"
            log_transform_pval_now = False
        
        # Define significance threshold for y-axis
        if pval_threshold is not None:
            if log_transform_pval_now:
                y_threshold = -np.log10(pval_threshold)
            elif 'neg_log10' in pval_key.lower() or pval_key.lower().startswith('neg_log10') or '-log10' in pval_key.lower():
                # Convert threshold if it's in raw p-value format (between 0 and 1)
                if 0 < pval_threshold < 1:
                    y_threshold = -np.log10(pval_threshold)
                else:
                    y_threshold = pval_threshold
            else:
                y_threshold = pval_threshold
        else:
            y_threshold = None
            
        # Define masks for significant cells
        if pval_threshold is not None and lfc_threshold is not None:
            significant = (y > y_threshold) & (np.abs(x) > lfc_threshold)
        elif pval_threshold is not None:
            significant = y > y_threshold
        elif lfc_threshold is not None:
            significant = np.abs(x) > lfc_threshold
        else:
            significant = np.ones(len(x), dtype=bool)
        
        # Apply custom highlight mask if provided for this subset
        if highlight_subset is not None:
            group_highlight = highlight_subset[mask]
            significant = group_highlight
        
        # Scatter plot parameters
        scatter_kwargs = {'s': 10}  # Default point size
        
        # Filter out cmap from kwargs if it exists to prevent conflicts
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'cmap'}
        scatter_kwargs.update(filtered_kwargs)
        
        # First plot all cells as background
        plot_ax.scatter(
            x, y, 
            alpha=alpha_background, 
            c="lightgray", 
            label="Non-significant",
            **scatter_kwargs
        )
        
        # Color significant cells
        if color is not None and _has_scanpy:
            # Extract the color values just for this group
            if isinstance(color, str):
                color_values = [color]
            else:
                color_values = color
                
            # Plot colored points for each color column
            for c in color_values:
                if c not in adata.obs:
                    warnings.warn(f"Color key '{c}' not found in adata.obs. Skipping.")
                    continue
                
                # Get the color values for this subset
                color_array = adata.obs[c].values[mask]
                
                # Check if the color column is categorical
                if pd.api.types.is_categorical_dtype(adata.obs[c]):
                    categories = adata.obs[c].cat.categories
                    
                    # Check if colors are stored in adata.uns
                    colors_key = f"{c}_colors"
                    if colors_key in adata.uns and len(adata.uns[colors_key]) == len(categories):
                        # Use stored colors from adata.uns
                        stored_colors = adata.uns[colors_key]
                        color_dict = dict(zip(categories, stored_colors))
                    elif isinstance(palette, str):
                        # Use matplotlib colormaps
                        cmap = plt.cm.get_cmap(palette, len(categories))
                        colors = [cmap(i/len(categories)) for i in range(len(categories))]
                        color_dict = dict(zip(categories, colors))
                    elif isinstance(palette, dict):
                        color_dict = palette
                    else:
                        # Use default palette
                        cmap = plt.cm.get_cmap('tab10', len(categories))
                        colors = [cmap(i/len(categories)) for i in range(len(categories))]
                        color_dict = dict(zip(categories, colors))
                    
                    # Plot each category separately
                    for cat in categories:
                        cat_mask = color_array == cat
                        if np.sum(cat_mask) > 0 and np.sum(cat_mask & significant) > 0:
                            cat_color = color_dict.get(cat, highlight_color)
                            plot_ax.scatter(
                                x[cat_mask & significant], 
                                y[cat_mask & significant],
                                alpha=1,
                                c=[cat_color],
                                label=f"{cat}",
                                **scatter_kwargs
                            )
                else:
                    # For numeric columns, determine the appropriate colormap and settings
                    
                    # Get data for coloring
                    data_for_color = color_array[significant]
                    
                    # Prepare scatter plot kwargs with color settings
                    scatter_kwargs_color = scatter_kwargs.copy()
                    
                    # Handle colormap selection
                    use_cmap = None
                    if cmap is not None:
                        use_cmap = cmap
                    elif any(term in c.lower() for term in ['lfc', 'log_fold_change']):
                        # For log fold change data, use diverging colormap with center at 0
                        use_cmap = "RdBu_r"
                        # Set default vcenter if not provided
                        if vcenter is None:
                            vcenter = 0
                    else:
                        # Default colormap for other numeric data
                        use_cmap = palette if isinstance(palette, str) else "Spectral_r"
                    
                    # Remove cmap from scatter_kwargs if it exists to prevent conflicts
                    if 'cmap' in scatter_kwargs:
                        scatter_kwargs_color.pop('cmap', None)
                    
                    scatter_kwargs_color['cmap'] = use_cmap
                    
                    # Handle color scaling with vmin, vmax, and vcenter
                    if vcenter is not None:
                        # Create diverging norm centered at vcenter
                        from matplotlib.colors import TwoSlopeNorm
                        
                        # Use global min/max if available, otherwise use local values
                        use_vmin = vmin if vmin is not None else (global_vmin if global_vmin is not None else np.min(data_for_color))
                        use_vmax = vmax if vmax is not None else (global_vmax if global_vmax is not None else np.max(data_for_color))
                        
                        # Ensure the bounds make sense for a diverging colormap
                        vrange = max(abs(use_vmin - vcenter), abs(use_vmax - vcenter))
                        use_vmin = vcenter - vrange
                        use_vmax = vcenter + vrange
                        
                        # Create normalized colormap centered at vcenter
                        scatter_kwargs_color['norm'] = TwoSlopeNorm(vmin=use_vmin, vcenter=vcenter, vmax=use_vmax)
                    else:
                        # Use regular vmin/vmax if provided, otherwise use global values if available
                        if vmin is not None:
                            scatter_kwargs_color['vmin'] = vmin
                        elif global_vmin is not None:
                            scatter_kwargs_color['vmin'] = global_vmin
                            
                        if vmax is not None:
                            scatter_kwargs_color['vmax'] = vmax
                        elif global_vmax is not None:
                            scatter_kwargs_color['vmax'] = global_vmax
                    
                    # Draw the scatter plot
                    scatter = plot_ax.scatter(
                        x[significant],
                        y[significant],
                        alpha=1,
                        c=data_for_color,
                        **scatter_kwargs_color
                    )
                    
                    # Store first mappable for colorbar
                    if first_color_mappable is None:
                        first_color_mappable = scatter
                        colorbar_needed = True
                        colorbar_label = c if isinstance(color, str) else None
        else:
            # Default coloring without color key
            plot_ax.scatter(
                x[significant], y[significant], 
                alpha=1, 
                c=highlight_color, 
                label="Significant",
                **scatter_kwargs
            )
        
        # Add threshold lines if requested
        if show_thresholds:
            if lfc_threshold is not None:
                plot_ax.axvline(x=lfc_threshold, color="black", linestyle="--", alpha=0.5)
                plot_ax.axvline(x=-lfc_threshold, color="black", linestyle="--", alpha=0.5)
            
            if pval_threshold is not None:
                if log_transform_pval_now:
                    plot_ax.axhline(y=-np.log10(pval_threshold), color="black", linestyle="--", alpha=0.5)
                elif 'neg_log10' in pval_key.lower() or pval_key.lower().startswith('neg_log10') or '-log10' in pval_key.lower():
                    # For negative log10 p-values, convert if needed
                    if 0 < pval_threshold < 1:
                        plot_ax.axhline(y=-np.log10(pval_threshold), color="black", linestyle="--", alpha=0.5)
                    else:
                        plot_ax.axhline(y=pval_threshold, color="black", linestyle="--", alpha=0.5)
                else:
                    plot_ax.axhline(y=pval_threshold, color="black", linestyle="--", alpha=0.5)
            
            # Add center line
            plot_ax.axvline(x=0, color="black", linestyle="-", alpha=0.3)
        
        # Add grid if requested
        if grid:
            plot_ax.grid(**(grid_kwargs or {'alpha': 0.3}))
        
        # Add a box around the plot
        for spine in plot_ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
        
        # Configure ticks
        if n_y_ticks == 0:
            plot_ax.yaxis.set_ticks([])
        else:
            from matplotlib.ticker import MaxNLocator
            plot_ax.yaxis.set_major_locator(MaxNLocator(n_y_ticks))
        
        # Only show x-axis labels on the bottom plot
        if i == n_groups - 1:
            # Customize x-axis label based on condition names if available
            if xlabel is None and condition1 and condition2:
                actual_xlabel = f"Log Fold Change: {condition1} vs {condition2}"
            else:
                actual_xlabel = xlabel or "Log Fold Change"
                
            plot_ax.set_xlabel(actual_xlabel, fontsize=12)
            if n_x_ticks > 0:
                from matplotlib.ticker import MaxNLocator
                plot_ax.xaxis.set_major_locator(MaxNLocator(n_x_ticks))
        else:
            # Hide x-tick labels for all but the bottom plot
            plt.setp(plot_ax.get_xticklabels(), visible=False)
            plot_ax.xaxis.set_ticks([])
        
        # Empty y label (handled separately)
        plot_ax.set_ylabel("")
        
        # Collect legend handles and labels, avoiding duplicates
        handles, labels = plot_ax.get_legend_handles_labels()
        for h, l in zip(handles, labels):
            if l not in all_labels:
                all_handles.append(h)
                all_labels.append(l)
    
    # Determine whether to show legend
    if show_legend is not None:
        # Explicitly set by user, honor their preference
        show_legend_now = show_legend
    else:
        # Automatically determine based on content
        show_legend_now = True
        
        # Check if color and groupby are identical - don't show legend in that case
        if isinstance(color, str) and color == groupby:
            logger.info(f"Color column '{color}' is identical to groupby column - not showing legend")
            show_legend_now = False
    
    # Create a separate area for legend/colorbar at the bottom
    if show_legend_now and (len(all_handles) > 0 or colorbar_needed):
        # Position the legend area at the bottom with fixed distance from the last plot
        # and properly scaled for its content
        
        # Determine legend height based on the number of items (for legend) or fixed size (for colorbar)
        if colorbar_needed and first_color_mappable is not None:
            actual_legend_height = max(layout['legend_height'], 3)  # Minimum 3 units for colorbar
        elif len(all_handles) > 0:
            # Scale legend height based on number of items (with min and max limits)
            item_count = len(all_handles)
            rows = (item_count + 3) // 4  # 4 items per row, rounded up
            actual_legend_height = max(layout['legend_height'], min(8, 2 + rows))
        else:
            actual_legend_height = layout['legend_height']  # Default
            
        # Use fixed gap between plots and legend
        legend_gap = layout['legend_plot_gap']
        
        # Get the bottom of the last plot
        last_plot_bottom = axes[-1].get_position().y0
        
        # Calculate the legend area in figure coordinates
        # Position legend directly below the last plot with the specified gap
        # For colorbars, use the standard gap
        # For legends, add an additional 2 units of space
        if colorbar_needed and first_color_mappable is not None:
            additional_gap = 0  # No additional gap for colorbars
        else:
            additional_gap = 3  # Additional 2 units for legends
            
        legend_top = last_plot_bottom - ((legend_gap + additional_gap) * unit / fig_height)
        legend_height = actual_legend_height * unit / fig_height
        legend_bottom = legend_top - legend_height
        
        # Calculate the left position to align with the plot area (not the whole figure)
        # This centers the legend/colorbar under the plots
        plot_left = y_label_width_norm  # The left edge of the plot area
        plot_right = 0.95 - 0.15  # The right edge of the plot area (minus the label width)
        plot_width = plot_right - plot_left
        
        # Center the legend/colorbar under the main plot (not label)
        legend_width = plot_width * 0.85  # Make it slightly narrower than the plot
        legend_left = plot_left + (plot_width - legend_width) / 2
        
        # Create a separate axes for the legend/colorbar
        # Position is [left, bottom, width, height]
        legend_ax = fig.add_axes([legend_left, legend_bottom, legend_width, legend_height])
        
        # Handle colorbar if needed
        if colorbar_needed and first_color_mappable is not None:
            legend_ax.axis('off')  # Hide the axis itself
            
            # Create horizontal colorbar
            cbar = fig.colorbar(
                first_color_mappable, 
                ax=legend_ax,
                orientation='horizontal',
                fraction=0.6,  # Controls height of the colorbar
                aspect=30      # Controls width-to-height ratio
            )
            
            # Set label
            if colorbar_label:
                cbar.set_label(colorbar_label, fontsize=legend_fontsize or 10)
            
            # Remove grid from colorbar
            cbar.ax.grid(False)
            
            # Limit number of ticks
            cbar.ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        
        # Add legend if we have handles and no colorbar (don't show both)
        elif len(all_handles) > 0:
            legend_ax.axis('off')  # Hide the axis itself
            
            # Calculate number of columns based on the number of labels
            ncols = min(4, len(all_labels))
            
            # Create the legend without a box, top-aligned
            # Use custom bbox transform to ensure it's positioned at the very top
            legend = legend_ax.legend(
                all_handles, all_labels,
                loc='upper center',  # Center horizontally
                ncol=ncols,
                fontsize=legend_fontsize or 10,
                title=color if isinstance(color, str) and not colorbar_needed else None,
                title_fontsize=legend_title_fontsize,
                frameon=False,
                bbox_to_anchor=(0.5, 0.0),  # Anchor at top-center
                bbox_transform=legend_ax.transAxes  # Use axes coordinates
            )
            
            # Explicitly position the legend at the top of the legend axis
            legend._set_loc(8)  # 8 is the code for 'center top'
    
    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.1)
    
    # Save figure if path provided
    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight")
    
    # Return figure and axes if requested
    if return_fig:
        return fig, axes
    elif show or (show is None and save is None):
        plt.show()