"""Volcano plot functions for visualizing differential expression and abundance results."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union, List, Tuple, Dict, Any, Literal
from anndata import AnnData
import pandas as pd
import warnings
import logging

from ..utils import get_run_from_history, KOMPOT_COLORS

try:
    import scanpy as sc
    _has_scanpy = True
except ImportError:
    _has_scanpy = False
    
logger = logging.getLogger("kompot")


def _infer_de_keys(adata: AnnData, run_id: Optional[int] = None, lfc_key: Optional[str] = None, 
                   score_key: Optional[str] = "kompot_de_mahalanobis"):
    """
    Infer differential expression keys from AnnData object.
    
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
            
            # Also try to get an appropriate score_key if using default
            if score_key == "kompot_de_mahalanobis" and 'mahalanobis_key' in key_run_info and key_run_info['mahalanobis_key']:
                inferred_score_key = key_run_info['mahalanobis_key']
                logger.info(f"Using score_key '{inferred_score_key}' from run {run_id}")
    
    # If still no lfc_key, try latest run
    if inferred_lfc_key is None and 'kompot_latest_run' in adata.uns and adata.uns['kompot_latest_run'].get('expression_key'):
        # Get the latest DE run key
        de_key = adata.uns['kompot_latest_run']['expression_key']
        
        # Check if we have run_info for this key
        if de_key in adata.uns and 'run_info' in adata.uns[de_key]:
            run_info = adata.uns[de_key]['run_info']
            if 'lfc_key' in run_info:
                inferred_lfc_key = run_info['lfc_key']
                logger.info(f"Using lfc_key '{inferred_lfc_key}' from latest run information")
            
            # Try to use mahalanobis key from run info if still using default
            if score_key == "kompot_de_mahalanobis" and 'mahalanobis_key' in run_info and run_info['mahalanobis_key']:
                inferred_score_key = run_info['mahalanobis_key']
                logger.info(f"Using score_key '{inferred_score_key}' from latest run information")
    
    # If still no lfc_key, try to find from column names
    if inferred_lfc_key is None:
        # Look for kompot LFC keys
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


def _infer_da_keys(adata: AnnData, run_id: Optional[int] = None, lfc_key: Optional[str] = None, 
                  pval_key: Optional[str] = None):
    """
    Infer differential abundance keys from AnnData object.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object with differential abundance results
    run_id : int, optional
        Run ID to use. If None, uses latest run.
    lfc_key : str, optional
        Log fold change key. If provided, will be returned as is.
    pval_key : str, optional
        P-value key. If provided, will be returned as is.
        
    Returns
    -------
    tuple
        (lfc_key, pval_key) with the inferred keys
    """
    inferred_lfc_key = lfc_key
    inferred_pval_key = pval_key
    
    # If both keys already provided, use them
    if inferred_lfc_key is not None and inferred_pval_key is not None:
        return inferred_lfc_key, inferred_pval_key
        
    # Try to get keys from specific run if requested
    run_info = get_run_from_history(adata, run_id)
    if run_info is not None and run_info.get('abundance_key'):
        da_key = run_info['abundance_key']
        
        # Check if we have run_info for this key
        if da_key in adata.uns and 'run_info' in adata.uns[da_key]:
            key_run_info = adata.uns[da_key]['run_info']
            if inferred_lfc_key is None and 'lfc_key' in key_run_info:
                inferred_lfc_key = key_run_info['lfc_key']
                logger.info(f"Using lfc_key '{inferred_lfc_key}' from run {run_id}")
            if inferred_pval_key is None and 'pval_key' in key_run_info:
                inferred_pval_key = key_run_info['pval_key']
                logger.info(f"Using pval_key '{inferred_pval_key}' from run {run_id}")
    
    # If keys are still missing, try latest run
    if (inferred_lfc_key is None or inferred_pval_key is None) and 'kompot_latest_run' in adata.uns and adata.uns['kompot_latest_run'].get('abundance_key'):
        # Get the latest DA run key
        da_key = adata.uns['kompot_latest_run']['abundance_key']
        
        # Check if we have run_info for this key
        if da_key in adata.uns and 'run_info' in adata.uns[da_key]:
            run_info = adata.uns[da_key]['run_info']
            if inferred_lfc_key is None and 'lfc_key' in run_info:
                inferred_lfc_key = run_info['lfc_key']
                logger.info(f"Using lfc_key '{inferred_lfc_key}' from latest run information")
            if inferred_pval_key is None and 'pval_key' in run_info:
                inferred_pval_key = run_info['pval_key']
                logger.info(f"Using pval_key '{inferred_pval_key}' from latest run information")
    
    # If lfc_key still not found, look for it in the data
    if inferred_lfc_key is None:
        # Look for kompot LFC keys in obs
        lfc_keys = [k for k in adata.obs.columns if 'kompot_da_' in k and 'lfc' in k.lower()]
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
    
    # If pval_key still not found, look for it in the data
    if inferred_pval_key is None:
        # First, look for the new negative log10 p-value naming convention
        standardized_neg_log10_keys = [
            k for k in adata.obs.columns if 'kompot_da_' in k and 'neg_log10_fold_change_pvalue' in k.lower()
        ]
        
        if len(standardized_neg_log10_keys) >= 1:
            # Use the standardized name if available
            inferred_pval_key = standardized_neg_log10_keys[0]
        else:
            # Fall back to alternative log10 naming patterns
            log10_pval_keys = [
                k for k in adata.obs.columns if 'kompot_da_' in k and 
                ('log10_pval' in k.lower() or 'log10_p_val' in k.lower() or 
                 'log10_p-val' in k.lower() or '-log10_pval' in k.lower() or
                 '-log10_p_val' in k.lower() or '-log10_p-val' in k.lower())
            ]
            
            if len(log10_pval_keys) >= 1:
                # Prefer log10 keys if available
                inferred_pval_key = log10_pval_keys[0]
            else:
                # Look for regular p-value keys
                pval_keys = [
                    k for k in adata.obs.columns if 'kompot_da_' in k and 
                    ('pval' in k.lower() or 'p_val' in k.lower() or 'p-val' in k.lower()) and
                    not any(prefix in k.lower() for prefix in ['log10', '-log10'])
                ]
                
                if len(pval_keys) == 1:
                    inferred_pval_key = pval_keys[0]
                elif len(pval_keys) > 1:
                    # Just use the first one
                    inferred_pval_key = pval_keys[0]
                else:
                    # As a last resort, try any key that might be a p-value
                    any_pval_keys = [
                        k for k in adata.obs.columns if 'kompot_da_' in k and 
                        any(term in k.lower() for term in ['pvalue', 'p_value', 'p-value', 'pval'])
                    ]
                    
                    if len(any_pval_keys) >= 1:
                        inferred_pval_key = any_pval_keys[0]
                    else:
                        raise ValueError("Could not infer pval_key. Please specify manually.")
    
    return inferred_lfc_key, inferred_pval_key


def volcano_de(
    adata: AnnData,
    lfc_key: str = None,
    score_key: str = "kompot_de_mahalanobis",
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
    text_kwargs: Optional[Dict[str, Any]] = None,
    grid: bool = True,
    grid_kwargs: Optional[Dict[str, Any]] = None,
    ax: Optional[plt.Axes] = None,
    sort_key: Optional[str] = None,
    return_fig: bool = False,
    save: Optional[str] = None,
    run_id: Optional[int] = None,
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
    text_kwargs : dict, optional
        Additional parameters for text labels
    grid : bool, optional
        Whether to show grid lines
    grid_kwargs : dict, optional
        Additional parameters for grid
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure
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
    text_kwargs = text_kwargs or {'ha': 'left', 'va': 'bottom'}
    grid_kwargs = grid_kwargs or {'alpha': 0.3}
    
    # Infer keys using helper function
    lfc_key, score_key = _infer_de_keys(adata, run_id, lfc_key, score_key)
    
    # Extract conditions from lfc_key if not provided
    if condition1 is None or condition2 is None:
        # Try to extract from key name, assuming format like "kompot_de_mean_lfc_Old_vs_Young"
        key_parts = lfc_key.split('_')
        if len(key_parts) >= 2 and 'vs' in key_parts:
            vs_index = key_parts.index('vs')
            if vs_index > 0 and vs_index < len(key_parts) - 1:
                condition1 = key_parts[vs_index-1]
                condition2 = key_parts[vs_index+1]
                
    # Create figure if ax not provided
    if ax is None:
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
            label=f"Up in {condition2}" if condition2 else "Up-regulated"
        )
        
        # Label top up-regulated genes
        if show_names:
            for _, gene_row in top_up.iterrows():
                ax.text(
                    gene_row['lfc'], gene_row['score'], gene_row['gene'],
                    fontsize=font_size, **text_kwargs
                )
    
    # Plot down-regulated genes
    if len(top_down) > 0:
        ax.scatter(
            top_down['lfc'].values,
            top_down['score'].values,
            alpha=1, s=point_size*3, c=color_down,
            label=f"Up in {condition1}" if condition1 else "Down-regulated"
        )
        
        # Label top down-regulated genes
        if show_names:
            for _, gene_row in top_down.iterrows():
                ax.text(
                    gene_row['lfc'], gene_row['score'], gene_row['gene'],
                    fontsize=font_size, **text_kwargs
                )
    
    # Add formatting
    ax.axvline(x=0, color="black", linestyle="--", alpha=0.3)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    # Set title if provided or can be inferred
    if title is None and condition1 and condition2:
        title = f"Volcano Plot: {condition2} vs {condition1}"
    if title:
        ax.set_title(title, fontsize=14)
    
    ax.legend()
    
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
    grid: bool = True,
    grid_kwargs: Optional[Dict[str, Any]] = None,
    ax: Optional[plt.Axes] = None,
    palette: Optional[Union[str, List[str], Dict[str, str]]] = None,
    save: Optional[str] = None,
    show: bool = None,
    return_fig: bool = False,
    run_id: Optional[int] = None,
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
        Location for the legend
    grid : bool, optional
        Whether to show grid lines
    grid_kwargs : dict, optional
        Additional parameters for grid
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure
    palette : str, list, or dict, optional
        Color palette to use for categorical coloring
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
    **kwargs : 
        Additional parameters passed to plt.scatter
        
    Returns
    -------
    If return_fig is True, returns (fig, ax)
    """
    # Set default grid kwargs
    grid_kwargs = grid_kwargs or {'alpha': 0.3}
    
    # Infer keys using helper function
    lfc_key, pval_key = _infer_da_keys(adata, run_id, lfc_key, pval_key)
    
    # Create figure if ax not provided
    if ax is None:
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
            # Use scanpy's coloring functionality
            # Pass custom palette if provided
            color_args = {"palette": palette} if palette else {}
            
            # Extract coordinates for significant cells
            coords = pd.DataFrame({
                "x": x[significant],
                "y": y[significant]
            }, index=adata.obs_names[significant])
            
            # Create temporary anndata with significant cells and their coordinates
            temp_adata = AnnData(
                X=coords.values,
                obs=adata.obs.loc[significant],
                var=pd.DataFrame(index=["x", "y"])
            )
            
            # Use scanpy to plot with colors
            if isinstance(color, str):
                color = [color]
                
            for c in color:
                if c not in temp_adata.obs:
                    warnings.warn(f"Color key '{c}' not found in adata.obs. Skipping.")
                    continue
                    
                # Use scanpy to generate colors
                sc.pl.scatter(
                    temp_adata, 
                    x=0, y=1,  # x and y are columns 0 and 1 in temp_adata
                    color=c,
                    ax=ax,
                    show=False,
                    size=point_size,
                    **color_args
                )
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
    
    # Add legend if not using scanpy coloring
    if color is None or not _has_scanpy:
        ax.legend(loc=legend_loc)
    
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