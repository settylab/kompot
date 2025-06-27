"""Multiple volcano plots for differential abundance."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
from typing import Optional, Union, List, Tuple, Dict, Any, Literal
from anndata import AnnData
import pandas as pd
import warnings
import logging
from scipy import stats

from ...utils import KOMPOT_COLORS
from ...anndata.utils import get_run_from_history
from .utils import _extract_conditions_from_key, _infer_da_keys
from .da import volcano_da

try:
    import scanpy as sc
    _has_scanpy = True
except (ImportError, TypeError):
    # Catch both ImportError (if scanpy isn't installed) 
    # and TypeError for metaclass conflicts
    _has_scanpy = False

# Get the pre-configured logger
logger = logging.getLogger("kompot")

def multi_volcano_da(
    adata: AnnData,
    groupby: str,
    lfc_key: Optional[str] = None,
    ptp_key: Optional[str] = None,
    log_transform_ptp: bool = True,
    lfc_threshold: Optional[float] = None,
    ptp_threshold: Optional[float] = None, 
    color: Optional[Union[str, List[str]]] = None,
    alpha_background: float = 1.0,  # No alpha by default
    highlight_subset: Optional[Union[np.ndarray, List[bool]]] = None,
    highlight_color: str = KOMPOT_COLORS["direction"]["up"],
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = "Differential Abundance Volcano Plot",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = "-log10(PTP (Posterior Tail Probability))",
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
    background_plot: Optional[Literal["kde", "violin"]] = None,  # Background plot type
    background_alpha: float = 0.5,  # Alpha value for the background plot
    background_color: str = "#E6E6E6",  # Light gray color for the background plot
    background_edgecolor: str = "#808080",  # Medium gray for the outline
    background_height_factor: float = 0.6,  # Height of background plot as fraction of y-axis range
    background_kwargs: Optional[Dict[str, Any]] = None,  # Additional kwargs for the background plot
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
    ptp_key : str, optional
        Key in adata.obs for PTPs (Posterior Tail Probabilities). Posterior Tail Probability is a significance measure score similar to p-value.
        If None, will try to infer from ``kompot_da_`` keys.
    log_transform_ptp : bool, optional
        Whether to -log10 transform PTPs (Posterior Tail Probabilities) for the y-axis
    lfc_threshold : float, optional
        Log fold change threshold for significance (for drawing threshold lines)
    ptp_threshold : float, optional
        PTP (Posterior Tail Probability) threshold for significance (for drawing threshold lines)
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
    background_plot : str, optional
        Type of background density plot to display. Options are "kde" or "violin".
        If None (default), no background density plot is shown.
    background_alpha : float, optional
        Alpha (transparency) value for the background density plot (default: 0.5)
    background_color : str, optional
        Color for the background density plot (default: "#E6E6E6", light gray)
    background_edgecolor : str, optional
        Color for the outline of the background density plot (default: "#808080", medium gray)
    background_height_factor : float, optional
        Controls the height of the background plot as a fraction of the y-axis range (default: 0.6).
        Higher values make the KDE/violin taller, lower values make it shorter.
    background_kwargs : dict, optional
        Additional parameters for the background density plot. Options include:
        - For KDE: "bw_method" (bandwidth method), "show_2d_kde" (bool), "contour_levels" (int),
          "contour_cmap" (colormap name), "contour_alpha" (float)
        - For violin: "showmeans" (bool), "showmedians" (bool), "showextrema" (bool)
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
    lfc_key, ptp_key, thresholds = _infer_da_keys(adata, run_id, lfc_key, ptp_key)
    
    # Get global y-values for consistent KDE/violin sizing directly from the full dataset
    if 'neg_log10' in ptp_key.lower() or ptp_key.lower().startswith('neg_log10') or '-log10' in ptp_key.lower():
        # Already negative log10 transformed - use as is
        global_y_values = adata.obs[ptp_key].values
    elif log_transform_ptp:
        global_y_values = -np.log10(adata.obs[ptp_key].values)
    else:
        global_y_values = adata.obs[ptp_key].values
    
    # Calculate global min/max and range
    global_y_min = np.nanmin(global_y_values)
    global_y_max = np.nanmax(global_y_values)
    global_y_range = global_y_max - global_y_min
    
    # Extract the threshold values
    auto_lfc_threshold, auto_ptp_threshold = thresholds
    
    # Try to extract conditions from the key name for better labeling
    condition_names = _extract_conditions_from_key(lfc_key)
    condition1, condition2 = condition_names if condition_names else (None, None)
    
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
    
    # Log appropriate information based on what needed to be inferred
    if needed_column_inference:
        logger.info(f"Inferred columns for multi-volcano plot: lfc_key='{lfc_key}', ptp_key='{ptp_key}'")
    
    if needed_threshold_inference:
        logger.info(f"Using inferred thresholds - lfc_threshold: {lfc_threshold}, ptp_threshold: {ptp_threshold}")
    
    logger.info(f"Creating volcano plots for groups: {', '.join(map(str, groups))}")
    
    # Update direction for the entire dataset if requested (do this only once!)
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
        
        # Handle PTPs (Posterior Tail Probabilities) - check if they're already negative log10 transformed
        if 'neg_log10' in ptp_key.lower() or ptp_key.lower().startswith('neg_log10') or '-log10' in ptp_key.lower():
            # Already negative log10 transformed - use as is (values should be positive)
            y = adata.obs[ptp_key].values[mask]
            y_label = "-log10(PTP (Posterior Tail Probability))"
            log_transform_ptp_now = False  # Override since already transformed
        elif log_transform_ptp:
            y = -np.log10(adata.obs[ptp_key].values[mask])
            y_label = "-log10(PTP (Posterior Tail Probability))"
            log_transform_ptp_now = True
        else:
            y = adata.obs[ptp_key].values[mask]
            y_label = "PTP (Posterior Tail Probability)"
            log_transform_ptp_now = False
        
        # Define significance threshold for y-axis
        if ptp_threshold is not None:
            if log_transform_ptp_now:
                y_threshold = -np.log10(ptp_threshold)
            elif 'neg_log10' in ptp_key.lower() or ptp_key.lower().startswith('neg_log10') or '-log10' in ptp_key.lower():
                # Convert threshold if it's in raw PTP (Posterior Tail Probability) format (between 0 and 1)
                if 0 < ptp_threshold < 1:
                    y_threshold = -np.log10(ptp_threshold)
                else:
                    y_threshold = ptp_threshold
            else:
                y_threshold = ptp_threshold
        else:
            y_threshold = None
            
        # Define masks for significant cells
        if ptp_threshold is not None and lfc_threshold is not None:
            significant = (y > y_threshold) & (np.abs(x) > lfc_threshold)
        elif ptp_threshold is not None:
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
        
        # Initialize background plot kwargs
        bg_kwargs = background_kwargs or {}
        
        # Add background density plot if requested
        if background_plot is not None:
            # Set default KDE/violin parameters if not provided
            if 'bw_method' not in bg_kwargs:
                bg_kwargs['bw_method'] = 'scott'  # Default bandwidth method
                
            # Get axis limits
            x_min, x_max = np.min(x), np.max(x)
            x_range = x_max - x_min
            y_min, y_max = np.min(y), np.max(y)
            y_range = y_max - y_min
            
            # Calculate plot height based on the height factor and global y range for consistency
            plot_height = background_height_factor * global_y_range  # Height for KDE/violin
            
            # Create the density plot based on the requested type
            if background_plot == "kde":
                # Position KDE at the bottom of the plot, using global range for consistent spacing
                bottom_pos = global_y_min
                
                # Create X-axis KDE (at bottom of plot)
                x_grid = np.linspace(x_min - 0.2*x_range, x_max + 0.2*x_range, 1000)
                
                # Compute KDE for x values
                kde_x = stats.gaussian_kde(x, bw_method=bg_kwargs.get('bw_method'))
                x_density = kde_x(x_grid)
                
                # Scale the density to fit in the plot
                x_density_scaled = x_density / np.max(x_density) * plot_height
                
                # Plot X-axis KDE at the bottom of the plot
                plot_ax.fill_between(
                    x_grid, 
                    bottom_pos,  # Below the data points
                    bottom_pos + x_density_scaled, 
                    color=background_color, 
                    alpha=background_alpha,
                    edgecolor=background_edgecolor,
                    linewidth=1.0,
                    zorder=0  # Ensure it's behind the points
                )
                
                # Only add 2D KDE if explicitly requested
                if bg_kwargs.get('show_2d_kde', False):
                    try:
                        # Create grid for 2D KDE
                        y_grid = np.linspace(y_min - 0.2*y_range, y_max + 0.2*y_range, 100)
                        xx, yy = np.meshgrid(x_grid, y_grid)
                        positions = np.vstack([xx.ravel(), yy.ravel()])
                        
                        # Calculate 2D KDE
                        values = np.vstack([x, y])
                        kernel = stats.gaussian_kde(values, bw_method=bg_kwargs.get('bw_method'))
                        density = np.reshape(kernel(positions).T, xx.shape)
                        
                        # Plot 2D KDE as contour with low alpha
                        contour_levels = bg_kwargs.get('contour_levels', 5)
                        contour_alpha = bg_kwargs.get('contour_alpha', background_alpha * 0.5)
                        plot_ax.contourf(
                            xx, 
                            yy, 
                            density, 
                            levels=contour_levels, 
                            cmap=bg_kwargs.get('contour_cmap', 'Blues'),
                            alpha=contour_alpha,
                            zorder=0  # Ensure it's behind the points
                        )
                    except Exception as e:
                        logger.warning(f"Failed to create 2D KDE contour plot: {e}. Skipping.")
                
            elif background_plot == "violin":
                # Calculate the center position for the violin plot
                y_center = (global_y_max + global_y_min) / 2  # Center of y-axis
                
                # Create the violin plot centered on the y-axis
                violin_parts = plot_ax.violinplot(
                    dataset=[x],  # Just the x values
                    positions=[y_center],  # Position at the center of y-axis
                    vert=False,  # Horizontal orientation
                    showmeans=bg_kwargs.get('showmeans', False),
                    showextrema=bg_kwargs.get('showextrema', False),
                    showmedians=bg_kwargs.get('showmedians', False),
                    widths=plot_height  # Height of violin
                )
                
                # Set violin colors and alpha
                for pc in violin_parts['bodies']:
                    pc.set_color(background_color)
                    pc.set_alpha(background_alpha)
                    pc.set_edgecolor(background_edgecolor)
                    pc.set_linewidth(1.0)
                
                # Remove the stat markers if we're not showing them
                if not bg_kwargs.get('showmeans', False) and not bg_kwargs.get('showmedians', False):
                    # Hide the lines that mark mean, median, etc.
                    for line_type in ['cmeans', 'cmins', 'cmaxes', 'cbars']:
                        if line_type in violin_parts:
                            violin_parts[line_type].set_visible(False)
        
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
            
            if ptp_threshold is not None:
                if log_transform_ptp_now:
                    plot_ax.axhline(y=-np.log10(ptp_threshold), color="black", linestyle="--", alpha=0.5)
                elif 'neg_log10' in ptp_key.lower() or ptp_key.lower().startswith('neg_log10') or '-log10' in ptp_key.lower():
                    # For negative log10 PTPs (Posterior Tail Probabilities), convert if needed
                    if 0 < ptp_threshold < 1:
                        plot_ax.axhline(y=-np.log10(ptp_threshold), color="black", linestyle="--", alpha=0.5)
                    else:
                        plot_ax.axhline(y=ptp_threshold, color="black", linestyle="--", alpha=0.5)
                else:
                    plot_ax.axhline(y=ptp_threshold, color="black", linestyle="--", alpha=0.5)
            
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
                actual_xlabel = f"Log Fold Change: {condition1} to {condition2}"
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