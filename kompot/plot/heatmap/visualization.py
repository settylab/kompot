"""Visualization functions for heatmap plotting."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Optional, Union, List, Tuple, Dict, Any, Sequence, Literal, Callable, Set
from anndata import AnnData
import pandas as pd
import logging
from scipy.cluster.hierarchy import linkage, dendrogram as scipy_dendrogram
import scipy.spatial.distance as ssd
from matplotlib.gridspec import GridSpec

logger = logging.getLogger("kompot")



def _setup_colormap_normalization(data, center, vmin, vmax, cmap):
    """
    Set up colormap normalization based on parameters.
    
    Parameters
    ----------
    data : numpy.ndarray
        Data to normalize
    center : float or None
        Value to center the colormap at
    vmin : float or None
        Minimum value for colormap
    vmax : float or None
        Maximum value for colormap
    cmap : str or colormap
        Colormap to use
        
    Returns
    -------
    tuple
        (norm, cmap_obj, vmin, vmax)
    """
    if center is not None:
        # Use diverging normalization
        vmin = np.nanmin(data) if vmin is None else vmin
        vmax = np.nanmax(data) if vmax is None else vmax
        # Ensure vmin and vmax are equidistant from center
        max_distance = max(abs(vmin - center), abs(vmax - center))
        vmin = center - max_distance
        vmax = center + max_distance
        norm = mcolors.TwoSlopeNorm(vcenter=center, vmin=vmin, vmax=vmax)
    else:
        # Use standard normalization
        vmin = np.nanmin(data) if vmin is None else vmin
        vmax = np.nanmax(data) if vmax is None else vmax
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        
    # Get colormap object
    if isinstance(cmap, str):
        try:
            # Use the newer API if available
            cmap_obj = plt.colormaps[cmap]
        except (AttributeError, KeyError):
            # Fall back to older API for compatibility
            cmap_obj = plt.cm.get_cmap(cmap)
    else:
        cmap_obj = cmap
    
    return norm, cmap_obj, vmin, vmax


def _draw_diagonal_split_cell(
    ax,
    x,
    y,
    w,
    h,
    val1,
    val2,
    cmap,
    vmin,
    vmax,
    alpha=1.0,
    edgecolor="none",
    linewidth=0,
):
    """
    Draw a cell split diagonally with two different values.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw on
    x, y : float
        The bottom-left coordinates of the cell
    w, h : float
        The width and height of the cell
    val1 : float
        The value for the lower-left triangle (first condition)
    val2 : float
        The value for the upper-right triangle (second condition)
    cmap : str or colormap
        The colormap to use
    vmin, vmax : float
        The minimum and maximum values for the colormap
    alpha : float, optional
        The opacity of the cell
    edgecolor : str, optional
        The color of the cell border
    linewidth : float, optional
        The width of the cell border
    """
    # Use provided colormap and normalize with the provided vmin/vmax
    # No need to create a new normalization object - use the one passed in
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    
    # Get colormap object if it's a string
    if isinstance(cmap, str):
        try:
            # Use the newer API if available
            cmap_obj = plt.colormaps[cmap]
        except (AttributeError, KeyError):
            # Fall back to older API for compatibility
            cmap_obj = plt.cm.get_cmap(cmap)
    else:
        cmap_obj = cmap  # Already a colormap object

    # Handle NaN values to prevent black triangles
    if np.isnan(val1):
        # Use a very light gray for NaN in lower triangle
        facecolor1 = (0.9, 0.9, 0.9, 0.5)  # Light gray with transparency
    else:
        facecolor1 = cmap_obj(norm(val1))

    if np.isnan(val2):
        # Use a very light gray for NaN in upper triangle
        facecolor2 = (0.9, 0.9, 0.9, 0.5)  # Light gray with transparency
    else:
        facecolor2 = cmap_obj(norm(val2))

    # Create triangles
    lower_triangle = mpatches.Polygon(
        [[x, y], [x + w, y], [x, y + h]],
        facecolor=facecolor1,
        alpha=alpha,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )

    upper_triangle = mpatches.Polygon(
        [[x + w, y], [x + w, y + h], [x, y + h]],
        facecolor=facecolor2,
        alpha=alpha,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )

    # Add to axes
    ax.add_patch(lower_triangle)
    ax.add_patch(upper_triangle)


def _draw_split_dot_cell(
    ax, 
    x, 
    y, 
    w, 
    h, 
    val1, 
    val2, 
    cmap, 
    vmin, 
    vmax, 
    cell_count1=None, 
    cell_count2=None,
    global_max_count=None,  # Global maximum count for scaling
    max_size_factor=0.9,  # Maximum size of the circle as a factor of the tile size
    alpha=1.0, 
    edgecolor="none", 
    linewidth=0
):
    """
    Draw a cell with a split dot showing two different values, with dot halves sized based on respective cell counts.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw on
    x, y : float
        The bottom-left coordinates of the cell
    w, h : float
        The width and height of the cell
    val1 : float
        The value for the left half of the dot (first condition)
    val2 : float
        The value for the right half of the dot (second condition)
    cmap : str or colormap
        The colormap to use
    vmin, vmax : float
        The minimum and maximum values for the colormap
    cell_count1 : int or float, optional
        Number of cells in first condition, determines the left half dot size
    cell_count2 : int or float, optional
        Number of cells in second condition, determines the right half dot size
    global_max_count : int or float, optional
        Global maximum count to use for consistent scaling across all dots
    max_size_factor : float, optional
        Maximum fraction of the tile that the dot can occupy
    alpha : float, optional
        The opacity of the cell
    edgecolor : str, optional
        The color of the dot border
    linewidth : float, optional
        The width of the dot border
    """
    # Use provided colormap and normalize with the provided vmin/vmax
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    
    # Get colormap object if it's a string
    if isinstance(cmap, str):
        try:
            # Use the newer API if available
            cmap_obj = plt.colormaps[cmap]
        except (AttributeError, KeyError):
            # Fall back to older API for compatibility
            cmap_obj = plt.cm.get_cmap(cmap)
    else:
        cmap_obj = cmap  # Already a colormap object

    # Handle NaN values 
    if np.isnan(val1):
        # Use a very light gray for NaN in left half
        facecolor1 = (0.9, 0.9, 0.9, 0.5)  # Light gray with transparency
    else:
        facecolor1 = cmap_obj(norm(val1))

    if np.isnan(val2):
        # Use a very light gray for NaN in right half
        facecolor2 = (0.9, 0.9, 0.9, 0.5)  # Light gray with transparency
    else:
        facecolor2 = cmap_obj(norm(val2))
    
    # Set defaults for missing counts
    cell_count1 = cell_count1 or 0
    cell_count2 = cell_count2 or 0
    
    # Determine max radius based on tile dimensions
    max_radius = min(w, h) * max_size_factor / 2
    
    # Calculate radius for each half based on their respective cell counts
    # The radius scales with the square root of the count (area ~ radius^2)
    if cell_count1 == 0 and cell_count2 == 0:
        # Default size if no counts available
        radius1 = max_radius * 0.3  # Small default size
        radius2 = max_radius * 0.3
    else:
        # Use either the global max count (if provided) or the local max count
        if global_max_count is not None:
            max_count = global_max_count
            # Cap the cell counts at the maximum if specified
            cell_count1 = min(cell_count1, global_max_count)
            cell_count2 = min(cell_count2, global_max_count)
        else:
            # Normalize by local max count as fallback
            max_count = max(cell_count1, cell_count2, 1)
            
        # Scale factor to ensure maximum radius is used for the largest count
        scale_factor = max_radius / np.sqrt(max_count)
        
        # Calculate each radius proportionally
        radius1 = np.sqrt(cell_count1) * scale_factor
        radius2 = np.sqrt(cell_count2) * scale_factor
    
    # Center of the circle
    center_x = x + w / 2
    center_y = y + h / 2
    
    # Create the left half dot with its specific radius
    left_half = mpatches.Wedge(
        (center_x, center_y),       # Center coordinates
        radius1,                    # Radius based on cell_count1
        90, 270,                    # Start and end angles (left half)
        facecolor=facecolor1,
        alpha=alpha,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )
    
    # Create the right half dot with its specific radius
    right_half = mpatches.Wedge(
        (center_x, center_y),       # Center coordinates
        radius2,                    # Radius based on cell_count2
        270, 90,                    # Start and end angles (right half)
        facecolor=facecolor2,
        alpha=alpha,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )
    
    # Add both halves to the axes
    ax.add_patch(left_half)
    ax.add_patch(right_half)

def _draw_fold_change_cell(ax, x, y, w, h, val1, val2, cmap, vmin, vmax, alpha=1.0, edgecolor="none", linewidth=0):
    """
    Draw a cell colored by the fold change between two values.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw on
    x, y : float
        The bottom-left coordinates of the cell
    w, h : float
        The width and height of the cell
    val1 : float
        The value for the first condition, or the pre-calculated fold change value
    val2 : float
        The value for the second condition (not used if val1 is the fold change)
    cmap : str or colormap
        The colormap to use
    vmin, vmax : float
        The minimum and maximum values for the colormap
    alpha : float, optional
        The opacity of the cell
    edgecolor : str, optional
        The color of the cell border
    linewidth : float, optional
        The width of the cell border
    """
    # Check if the value is NaN
    if np.isnan(val1):
        # Use a very light gray for NaN
        facecolor = (0.9, 0.9, 0.9, 0.5)  # Light gray with transparency
    else:
        # Use provided colormap and normalize with the provided vmin/vmax
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        
        # Get colormap object if it's a string
        if isinstance(cmap, str):
            try:
                # Use the newer API if available
                cmap_obj = plt.colormaps[cmap]
            except (AttributeError, KeyError):
                # Fall back to older API for compatibility
                cmap_obj = plt.cm.get_cmap(cmap)
        else:
            cmap_obj = cmap  # Already a colormap object
        
        # Use val1 directly as the fold change (pre-computed)
        fold_change = val1
        facecolor = cmap_obj(norm(fold_change))

    # Create a rectangle for the cell
    rectangle = mpatches.Rectangle(
        (x, y),
        w, h,
        facecolor=facecolor,
        alpha=alpha,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )

    # Add to axes
    ax.add_patch(rectangle)