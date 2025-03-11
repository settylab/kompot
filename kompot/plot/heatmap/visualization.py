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