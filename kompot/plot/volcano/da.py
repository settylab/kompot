"""Volcano plot functions for differential abundance."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
from typing import Optional, Union, List, Tuple, Dict, Any
from anndata import AnnData
import pandas as pd
import warnings
import logging

from ...utils import KOMPOT_COLORS
from ...anndata.utils import get_run_from_history
from .utils import _extract_conditions_from_key, _infer_da_keys

try:
    import scanpy as sc
    _has_scanpy = True
except (ImportError, TypeError):
    # Catch both ImportError (if scanpy isn't installed) 
    # and TypeError for metaclass conflicts
    _has_scanpy = False

# Get the pre-configured logger
logger = logging.getLogger("kompot")

def volcano_da(
    adata: AnnData,
    lfc_key: Optional[str] = None,
    ptp_key: Optional[str] = None, 
    group_key: Optional[str] = None,
    log_transform_ptp: bool = True,
    lfc_threshold: Optional[float] = None,
    ptp_threshold: Optional[float] = None,
    color: Optional[Union[str, List[str]]] = None,
    alpha_background: float = 1.0,  # No alpha by default
    highlight_subset: Optional[Union[np.ndarray, List[bool]]] = None,
    highlight_color: str = KOMPOT_COLORS["direction"]["up"],
    figsize: Tuple[float, float] = (10, 8),
    title: Optional[str] = "Differential Abundance Volcano Plot",
    xlabel: Optional[str] = "Log Fold Change",
    ylabel: Optional[str] = "-log10(ptp)",
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
    and significance (-log10 PTP (Posterior Tail Probability)) on the y-axis. Cells can be colored by any column
    in adata.obs.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object containing differential abundance results
    lfc_key : str, optional
        Key in adata.obs for log fold change values.
        If None, will try to infer from ``kompot_da_`` keys.
    ptp_key : str, optional
        Key in adata.obs for PTPs (Posterior Tail Probabilities). Posterior Tail Probability is a significance measure score similar to p-value.
        If None, will try to infer from ``kompot_da_`` keys.
    group_key : str, optional
        Key in adata.obs to group cells by (for coloring)
    log_transform_ptp : bool, optional
        Whether to -log10 transform PTPs (Posterior Tail Probabilities) for the y-axis
    lfc_threshold : float, optional
        Log fold change threshold for significance (for drawing threshold lines)
    ptp_threshold : float, optional
        PTP (Posterior Tail Probability) threshold for significance (for drawing threshold lines)
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
    lfc_key, ptp_key, thresholds = _infer_da_keys(adata, run_id, lfc_key, ptp_key)
    
    # Calculate the actual (positive) run ID for logging
    if run_id < 0:
        # Use get_run_history to get the deserialized run history
        from ...anndata.utils import get_run_history
        run_history = get_run_history(adata, "da")
        if run_history is not None:
            actual_run_id = len(run_history) + run_id
        else:
            actual_run_id = run_id
    else:
        actual_run_id = run_id
    
    # Extract the threshold values
    auto_lfc_threshold, auto_ptp_threshold = thresholds
    
    # Track which values needed inference for logging
    needed_column_inference = lfc_key is None or ptp_key is None
    needed_threshold_inference = False
    
    # Use run thresholds if available and not explicitly overridden
    if lfc_threshold is None and auto_lfc_threshold is not None:
        lfc_threshold = auto_lfc_threshold
        needed_threshold_inference = True
    
    if ptp_threshold is None and auto_ptp_threshold is not None:
        ptp_threshold = auto_ptp_threshold
        needed_threshold_inference = True
        
    # Update direction column if requested
    if update_direction:
        from ...differential.utils import update_direction_column as update_dir
        logger.info(f"Updating direction column with new thresholds before plotting")
        update_dir(
            adata=adata,
            lfc_threshold=lfc_threshold,
            ptp_threshold=ptp_threshold,
            direction_column=direction_column,
            lfc_key=lfc_key,
            ptp_key=ptp_key,
            run_id=run_id,
            inplace=True
        )
    
    # Get condition information from the run specified by run_id
    run_info = get_run_from_history(adata, run_id, analysis_type="da")
    condition1 = None
    condition2 = None

    if run_info is not None and 'params' in run_info:
        params = run_info['params']
        if 'condition1' in params and 'condition2' in params:
            condition1 = params['condition1']
            condition2 = params['condition2']
    
    # Try to extract from key name if still not found
    if (condition1 is None or condition2 is None) and lfc_key is not None:
        conditions = _extract_conditions_from_key(lfc_key)
        if conditions:
            condition1, condition2 = conditions
    
    # Log appropriate information based on what needed to be inferred
    if needed_column_inference:
        conditions_str = f": comparing {condition1} to {condition2}" if condition1 and condition2 else ""
        logger.info(f"Inferred DA columns from run {actual_run_id}{conditions_str}")
        logger.info(f"Using fields for DA plot - lfc_key: '{lfc_key}', ptp_key: '{ptp_key}'")
    
    if needed_threshold_inference:
        logger.info(f"Using inferred thresholds - lfc_threshold: {lfc_threshold}, ptp_threshold: {ptp_threshold}")
    
    # Update axis labels with condition information if not explicitly set
    if condition1 and condition2 and xlabel == "Log Fold Change":
        # Adjust for new key format where condition1 is the baseline/denominator
        xlabel = f"Log Fold Change: {condition1} to {condition2}"
    
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
    
    # Handle PTPs (Posterior Tail Probabilities) - check if they're already negative log10 transformed
    if 'neg_log10' in ptp_key.lower() or ptp_key.lower().startswith('neg_log10') or '-log10' in ptp_key.lower():
        # Already negative log10 transformed - use as is (values should be positive)
        y = adata.obs[ptp_key].values
        ylabel = ylabel or "-log10(PTP (Posterior Tail Probability))"
        log_transform_ptp = False  # Override since already transformed
    elif log_transform_ptp:
        y = -np.log10(adata.obs[ptp_key].values)
        ylabel = ylabel or "-log10(PTP (Posterior Tail Probability))"
    else:
        y = adata.obs[ptp_key].values
        ylabel = ylabel or "PTP (Posterior Tail Probability)"
    
    # Define significance thresholds for coloring
    if ptp_threshold is not None:
        if 'neg_log10' in ptp_key.lower() or ptp_key.lower().startswith('neg_log10') or '-log10' in ptp_key.lower():
            # Key indicates values are already negative log10 transformed (higher = more significant)
            # Convert ptp_threshold to -log10 scale if it's a raw PTP (Posterior Tail Probability) (between 0 and 1)
            if 0 < ptp_threshold < 1:
                y_threshold = -np.log10(ptp_threshold)  # Convert to -log10 scale
            else:
                # Assume it's already on -log10 scale
                y_threshold = ptp_threshold
        elif log_transform_ptp:
            y_threshold = -np.log10(ptp_threshold)
        else:
            y_threshold = ptp_threshold
    else:
        y_threshold = None
    
    # Define masks for significant cells
    if ptp_threshold is not None and lfc_threshold is not None:
        # Both thresholds provided
        significant = (y > y_threshold) & (np.abs(x) > lfc_threshold)
    elif ptp_threshold is not None:
        # Only PTP (Posterior Tail Probability) threshold provided
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
        
        if ptp_threshold is not None:
            if 'neg_log10' in ptp_key.lower() or ptp_key.lower().startswith('neg_log10') or '-log10' in ptp_key.lower():
                # For negative PTPs (Posterior Tail Probabilities), convert if needed
                if 0 < ptp_threshold < 1:
                    ax.axhline(y=-np.log10(ptp_threshold), color="black", linestyle="--", alpha=0.5)
                else:
                    ax.axhline(y=ptp_threshold, color="black", linestyle="--", alpha=0.5)
            elif log_transform_ptp:
                ax.axhline(y=-np.log10(ptp_threshold), color="black", linestyle="--", alpha=0.5)
            else:
                ax.axhline(y=ptp_threshold, color="black", linestyle="--", alpha=0.5)
    
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
