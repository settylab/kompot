"""Volcano plot functions for visualizing differential expression and abundance results."""

import numpy as np
import matplotlib.pyplot as plt
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
except ImportError:
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
    show_names: bool = True,
    figsize: Tuple[float, float] = (10, 8),
    title: Optional[str] = None,
    xlabel: Optional[str] = "Log Fold Change",
    ylabel: Optional[str] = "Mahalanobis Distance",
    color_up: str = KOMPOT_COLORS["direction"]["up"],
    color_down: str = KOMPOT_COLORS["direction"]["down"],
    color_background: str = "gray",
    alpha_background: float = 0.4,
    point_size: float = 5,
    label_top_genes: bool = True,
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
        If None, will try to infer from kompot_de_ keys.
    score_key : str, optional
        Key in adata.var for significance scores.
        Default is "kompot_de_mahalanobis"
    condition1 : str, optional
        Name of condition 1 (negative log fold change)
    condition2 : str, optional
        Name of condition 2 (positive log fold change)
    n_top_genes : int, optional
        Number of top genes to highlight and label (default: 10)
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
    label_top_genes : bool, optional
        Whether to label top genes (default: True)
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
    
    # Split into up and down regulated and sort by score
    up_genes = de_data[de_data['lfc'] > 0].sort_values('sort_val', ascending=False)
    down_genes = de_data[de_data['lfc'] < 0].sort_values('sort_val', ascending=False)
    
    # Select top genes
    top_up = up_genes.head(n_top_genes)
    top_down = down_genes.head(n_top_genes)
    
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
    alpha_background: float = 0.4,
    point_size: float = 10,
    highlight_subset: Optional[Union[np.ndarray, List[bool]]] = None,
    highlight_color: str = KOMPOT_COLORS["direction"]["up"],
    figsize: Tuple[float, float] = (10, 8),
    title: Optional[str] = "Differential Abundance Volcano Plot",
    xlabel: Optional[str] = "Log Fold Change",
    ylabel: Optional[str] = "-log10(p-value)",
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
        If None, will try to infer from kompot_da_ keys.
    pval_key : str, optional
        Key in adata.obs for p-values.
        If None, will try to infer from kompot_da_ keys.
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
        Alpha value for background cells (below threshold)
    point_size : float, optional
        Size of points
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
    
    # Use run thresholds if available and not explicitly overridden
    if lfc_threshold is None and auto_lfc_threshold is not None:
        lfc_threshold = auto_lfc_threshold
        logger.debug(f"Using automatically detected lfc_threshold: {lfc_threshold}")
    
    if pval_threshold is None and auto_pval_threshold is not None:
        pval_threshold = auto_pval_threshold
        logger.debug(f"Using automatically detected pval_threshold: {pval_threshold}")
    
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
    
    # Log which run and fields are being used
    conditions_str = f": comparing {condition1} vs {condition2}" if condition1 and condition2 else ""
    logger.info(f"Using DA run {actual_run_id}{conditions_str}")
    logger.info(f"Using fields for DA plot - lfc_key: '{lfc_key}', pval_key: '{pval_key}'")
    if lfc_threshold is not None or pval_threshold is not None:
        logger.info(f"Using thresholds - lfc_threshold: {lfc_threshold}, pval_threshold: {pval_threshold}")
    
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
    ax.scatter(
        x, y, 
        alpha=alpha_background, 
        s=point_size, 
        c="lightgray", 
        label="Non-significant",
        **kwargs
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
                s=point_size, 
                c=highlight_color, 
                label="Significant",
                **kwargs
            )
        else:
            # We'll handle coloring manually instead of using scanpy's scatter
            import seaborn as sns
            from matplotlib.colors import ListedColormap
            
            # Get the significant indices
            sig_indices = np.where(significant)[0]
            
            if isinstance(color, str):
                color = [color]
                
            for c in color:
                if c not in adata.obs:
                    warnings.warn(f"Color key '{c}' not found in adata.obs. Skipping.")
                    continue
                    
                # Get the color values for the significant points
                color_values = adata.obs[c].values[sig_indices]
                
                # Check if the color column is categorical
                if pd.api.types.is_categorical_dtype(adata.obs[c]):
                    categories = adata.obs[c].cat.categories
                    
                    # If palette is a string, convert it to a color map
                    if isinstance(palette, str):
                        colors = sns.color_palette(palette, n_colors=len(categories))
                        color_dict = dict(zip(categories, colors))
                    elif isinstance(palette, dict):
                        color_dict = palette
                    else:
                        # Use default palette
                        colors = sns.color_palette("tab10", n_colors=len(categories))
                        color_dict = dict(zip(categories, colors))
                    
                    # Plot each category separately
                    for cat in categories:
                        cat_mask = color_values == cat
                        if np.sum(cat_mask) > 0:
                            cat_color = color_dict.get(cat, highlight_color)
                            ax.scatter(
                                x[sig_indices][cat_mask], 
                                y[sig_indices][cat_mask],
                                alpha=1,
                                s=point_size,
                                c=[cat_color],
                                label=f"{cat}",
                                **kwargs
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
                    scatter = ax.scatter(
                        x[sig_indices],
                        y[sig_indices],
                        alpha=1,
                        s=point_size,
                        c=color_values,
                        cmap=palette if isinstance(palette, str) else "viridis",
                        **kwargs
                    )
                    plt.colorbar(scatter, ax=ax, label=c)
    else:
        # Default coloring without color key
        ax.scatter(
            x[significant], y[significant], 
            alpha=1, 
            s=point_size, 
            c=highlight_color, 
            label="Significant",
            **kwargs
        )
    
    # Add threshold lines
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
    
    # Add center line
    ax.axvline(x=0, color="black", linestyle="-", alpha=0.3)
    
    # Add labels and title
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
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
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight")
    
    # Show or return
    if return_fig:
        return fig, ax
    elif show or (show is None and save is None):
        plt.show()