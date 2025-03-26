"""Volcano plot functions for differential expression."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
from typing import Optional, Union, List, Tuple, Dict, Any
from anndata import AnnData
import pandas as pd
import warnings
import logging

from ...utils import KOMPOT_COLORS, get_run_from_history
from .utils import _extract_conditions_from_key, _infer_de_keys

try:
    import scanpy as sc
    _has_scanpy = True
except (ImportError, TypeError):
    # Catch both ImportError (if scanpy isn't installed) 
    # and TypeError for metaclass conflicts
    _has_scanpy = False

# Get the pre-configured logger
logger = logging.getLogger("kompot")

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
    conditions_str = f": comparing {condition1} to {condition2}" if condition1 and condition2 else ""
    logger.info(f"Using DE run {actual_run_id}{conditions_str}")
    logger.info(f"Using fields for DE plot - lfc_key: '{lfc_key}', score_key: '{score_key}'")
    
    # Update axis labels
    if condition1 and condition2 and xlabel == "Log Fold Change":
        # Adjust for new key format where condition1 is the baseline/denominator
        xlabel = f"Log Fold Change: {condition1} to {condition2}"
                
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