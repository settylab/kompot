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
    draw_values=False,
    norm=None
):
    """
    Draw a cell split diagonally with two different values, and optionally display these values.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw on.
    x, y : float
        The bottom-left coordinates of the cell.
    w, h : float
        The width and height of the cell.
    val1 : float or str
        The value for the lower-left triangle (first condition).
    val2 : float or str
        The value for the upper-right triangle (second condition).
    cmap : str or colormap
        The colormap to use.
    vmin, vmax : float
        The minimum and maximum values for the colormap.
    alpha : float, optional
        The opacity of the cell.
    edgecolor : str, optional
        The color of the cell border.
    linewidth : float, optional
        The width of the cell border.
    draw_values : bool, optional
        If True, draw the corresponding numerical values on each triangle for debugging.
    norm : matplotlib.colors.Normalize, optional
        Normalization to use. If None, a standard Normalize will be created.
    """
    # Ensure ax is a valid axis object before proceeding
    if ax is None or not hasattr(ax, 'add_patch'):
        raise ValueError("ax must be a valid matplotlib Axes object with add_patch method")
    
    # Convert val1 to float if needed
    if isinstance(val1, str):
        try:
            val1 = float(val1)
        except ValueError:
            val1 = np.nan

    # Convert val2 to float if needed
    if isinstance(val2, str):
        try:
            val2 = float(val2)
        except ValueError:
            val2 = np.nan
            
    # Use provided norm or create a standard one
    if norm is None:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        
    # Get colormap object
    if isinstance(cmap, str):
        try:
            cmap_obj = plt.colormaps[cmap]
        except (AttributeError, KeyError):
            cmap_obj = plt.cm.get_cmap(cmap)
    else:
        cmap_obj = cmap

    # Compute face colors with handling for NaNs
    facecolor1 = (0.9, 0.9, 0.9, 0.5) if np.isnan(val1) else cmap_obj(norm(val1))
    facecolor2 = (0.9, 0.9, 0.9, 0.5) if np.isnan(val2) else cmap_obj(norm(val2))

    # Create lower-left triangle (vertices: bottom-left, bottom-right, top-left)
    lower_triangle = mpatches.Polygon(
        [[x, y], [x + w, y], [x, y + h]],
        facecolor=facecolor1,
        alpha=alpha,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )

    # Create upper-right triangle (vertices: bottom-right, top-right, top-left)
    upper_triangle = mpatches.Polygon(
        [[x + w, y], [x + w, y + h], [x, y + h]],
        facecolor=facecolor2,
        alpha=alpha,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )

    # Add triangles to axes
    ax.add_patch(lower_triangle)
    ax.add_patch(upper_triangle)
    
    # Optionally draw the values for debugging
    if draw_values:
        # Calculate approximate centroids for each triangle
        lower_cx, lower_cy = x + w / 3, y + h / 3
        upper_cx, upper_cy = x + 2 * w / 3, y + 2 * h / 3
        
        text_val1 = "NaN" if np.isnan(val1) else f"{val1:.2f}"
        text_val2 = "NaN" if np.isnan(val2) else f"{val2:.2f}"
        
        # Lower triangle text
        ax.text(lower_cx, lower_cy, text_val1,
                ha="center", va="center", color="black", fontsize=8,
                bbox=dict(facecolor='white', edgecolor='none', pad=1, alpha=0.7))
        
        # Upper triangle text
        ax.text(upper_cx, upper_cy, text_val2,
                ha="center", va="center", color="black", fontsize=8,
                bbox=dict(facecolor='white', edgecolor='none', pad=1, alpha=0.7))


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
    linewidth=0,
    draw_values=False,
    norm=None
):
    """
    Draw a cell with a split dot showing two different values, with dot halves sized
    based on respective cell counts, and optionally annotate the halves with their values.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw on.
    x, y : float
        The bottom-left coordinates of the cell.
    w, h : float
        The width and height of the cell.
    val1 : float or str
        The value for the left half of the dot (first condition).
    val2 : float or str
        The value for the right half of the dot (second condition).
    cmap : str or colormap
        The colormap to use.
    vmin, vmax : float
        The minimum and maximum values for the colormap.
    cell_count1 : int or float, optional
        Number of cells in the first condition, determines the left half dot size.
    cell_count2 : int or float, optional
        Number of cells in the second condition, determines the right half dot size.
    global_max_count : int or float, optional
        Global maximum count to use for consistent scaling across all dots.
    max_size_factor : float, optional
        Maximum fraction of the tile that the dot can occupy.
    alpha : float, optional
        The opacity of the cell.
    edgecolor : str, optional
        The color of the cell border.
    linewidth : float, optional
        The width of the dot border.
    draw_values : bool, optional
        If True, draw the corresponding numerical values on the left and right halves for debugging.
    norm : matplotlib.colors.Normalize, optional
        Normalization to use. If None, a standard Normalize will be created.
    """
    # Ensure ax is a valid axis object before proceeding
    if ax is None or not hasattr(ax, 'add_patch'):
        raise ValueError("ax must be a valid matplotlib Axes object with add_patch method")
    
    # Convert val1 to float if needed
    if isinstance(val1, str):
        try:
            val1 = float(val1)
        except ValueError:
            val1 = np.nan

    # Convert val2 to float if needed
    if isinstance(val2, str):
        try:
            val2 = float(val2)
        except ValueError:
            val2 = np.nan
    
    # Handle cell counts if they're strings
    if isinstance(cell_count1, str):
        try:
            cell_count1 = float(cell_count1)
        except ValueError:
            cell_count1 = 0
    
    if isinstance(cell_count2, str):
        try:
            cell_count2 = float(cell_count2)
        except ValueError:
            cell_count2 = 0
            
    if isinstance(global_max_count, str):
        try:
            global_max_count = float(global_max_count)
        except ValueError:
            global_max_count = None
            
    # Use provided norm or create a standard one
    if norm is None:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        
    # Get colormap object
    if isinstance(cmap, str):
        try:
            cmap_obj = plt.colormaps[cmap]
        except (AttributeError, KeyError):
            cmap_obj = plt.cm.get_cmap(cmap)
    else:
        cmap_obj = cmap  # Already a colormap object

    # Determine face colors for left and right halves with NaN handling
    facecolor1 = (0.9, 0.9, 0.9, 0.5) if np.isnan(val1) else cmap_obj(norm(val1))
    facecolor2 = (0.9, 0.9, 0.9, 0.5) if np.isnan(val2) else cmap_obj(norm(val2))
    
    # Set defaults for missing counts
    cell_count1 = cell_count1 or 0
    cell_count2 = cell_count2 or 0
    
    # Determine maximum radius based on tile dimensions
    max_radius = min(w, h) * max_size_factor / 2
    
    # Calculate radius for each half based on their respective cell counts
    if cell_count1 == 0 and cell_count2 == 0:
        # Default size if no counts available
        radius1 = max_radius * 0.3  # Small default size
        radius2 = max_radius * 0.3
    else:
        # Use either the global max count (if provided) or the local max count
        if global_max_count is not None:
            max_count = global_max_count
            cell_count1 = min(cell_count1, global_max_count)
            cell_count2 = min(cell_count2, global_max_count)
        else:
            max_count = max(cell_count1, cell_count2, 1)
            
        scale_factor = max_radius / np.sqrt(max_count)
        radius1 = np.sqrt(cell_count1) * scale_factor
        radius2 = np.sqrt(cell_count2) * scale_factor
    
    # Center of the circle
    center_x = x + w / 2
    center_y = y + h / 2
    
    # Create the left half dot using a Wedge (covers 180° on the left side)
    left_half = mpatches.Wedge(
        (center_x, center_y),  # Center coordinates
        radius1,               # Radius based on cell_count1
        90, 270,               # Wedge angles for left half
        facecolor=facecolor1,
        alpha=alpha,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )
    
    # Create the right half dot using a Wedge (covers 180° on the right side)
    right_half = mpatches.Wedge(
        (center_x, center_y),
        radius2,
        270, 90,               # Wedge angles for right half
        facecolor=facecolor2,
        alpha=alpha,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )
    
    # Add both halves to the axes
    ax.add_patch(left_half)
    ax.add_patch(right_half)
    
    # Optionally draw the values on each half for debugging.
    if draw_values:
        left_text = "NaN" if np.isnan(val1) else f"{val1:.2f}"
        right_text = "NaN" if np.isnan(val2) else f"{val2:.2f}"
        # Position left text at half the left radius offset from center, right text at half the right radius offset
        left_x = center_x - (radius1 / 2)
        right_x = center_x + (radius2 / 2)
        
        ax.text(left_x, center_y, left_text,
                ha="center", va="center", fontsize=8,
                bbox=dict(facecolor='white', edgecolor='none', pad=1, alpha=0.7))
        ax.text(right_x, center_y, right_text,
                ha="center", va="center", fontsize=8,
                bbox=dict(facecolor='white', edgecolor='none', pad=1, alpha=0.7))

def _draw_fold_change_cell(ax, x, y, w, h, lfc, cmap, vmin, vmax, alpha=1.0, edgecolor="none", linewidth=0, draw_values=False, norm=None):
    """
    Draw a cell colored by the fold change between two values and optionally display the value.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw on.
    x, y : float
        The bottom-left coordinates of the cell.
    w, h : float
        The width and height of the cell.
    lfc : float or str
        The log fold change value.
    cmap : str or colormap
        The colormap to use.
    vmin, vmax : float
        The minimum and maximum values for the colormap.
    alpha : float, optional
        The opacity of the cell.
    edgecolor : str, optional
        The color of the cell border.
    linewidth : float, optional
        The width of the cell border.
    draw_values : bool, optional
        If True, draw the fold change value at the center of the cell for debugging.
    norm : matplotlib.colors.Normalize, optional
        Normalization to use. If None, a standard Normalize will be created.
    """
    # Ensure ax is a valid axis object before proceeding
    if ax is None or not hasattr(ax, 'add_patch'):
        raise ValueError("ax must be a valid matplotlib Axes object with add_patch method")
    
    if isinstance(lfc, str):
        try:
            # Try to convert string to float
            lfc = float(lfc)
        except ValueError:
            # If conversion fails, use NaN
            lfc = np.nan
    
    # Determine the face color based on lfc
    if np.isnan(lfc):
        # Use a very light gray for NaN
        facecolor = (0.9, 0.9, 0.9, 0.5)  # Light gray with transparency
    else:
        # Use provided norm or create a standard one
        if norm is None:
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            
        # Get colormap object if cmap is a string, falling back if necessary
        if isinstance(cmap, str):
            try:
                cmap_obj = plt.colormaps[cmap]
            except (AttributeError, KeyError):
                cmap_obj = plt.cm.get_cmap(cmap)
        else:
            cmap_obj = cmap
        facecolor = cmap_obj(norm(lfc))

    # Create a rectangle for the cell
    rectangle = mpatches.Rectangle(
        (x, y),
        w, h,
        facecolor=facecolor,
        alpha=alpha,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )

    # Add rectangle to axes
    ax.add_patch(rectangle)
    
    # Optionally draw the fold change value at the center of the cell for debugging
    if draw_values:
        text_val = "NaN" if np.isnan(lfc) else f"{lfc:.2f}"
        ax.text(x + w / 2, y + h / 2, text_val,
                ha="center", va="center", color="black", fontsize=8,
                bbox=dict(facecolor='white', edgecolor='none', pad=1, alpha=0.7))