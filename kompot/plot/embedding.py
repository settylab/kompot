"""Functions for plotting embeddings with group filtering."""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Dict, List, Union, Tuple, Any
from anndata import AnnData
import logging
import warnings

# Get the pre-configured logger
logger = logging.getLogger("kompot")

try:
    import scanpy as sc
    _has_scanpy = True
except ImportError:
    _has_scanpy = False


def embedding(
    adata: AnnData,
    basis: str,
    groups: Optional[Union[Dict[str, Union[str, List[str]]], str, List[str]]] = None,
    background_color: Optional[str] = "lightgrey",
    matplotlib_scatter_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Any:
    """
    Plot embeddings with group filtering capabilities.
    
    This function wraps scanpy's plotting.embedding function but adds the ability to filter
    cells based on observation column values. Selected cells are plotted normally using scanpy,
    while non-selected cells can be displayed in a different color in the background.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object containing the embedding coordinates.
    basis : str
        Key for the embedding coordinates. Same as scanpy's basis parameter.
    groups : Dict[str, Union[str, List[str]]] or str or List[str], optional
        If a dictionary: keys are column names in adata.obs and values are lists or individual
        allowed values. Only cells matching ALL conditions will be highlighted.
        If a string: Same as scanpy's groups parameter for categorical groupby.
        If None: all cells are shown normally.
    background_color : str, optional
        Color for non-selected cells. If None, background cells are not shown.
        Default is "lightgrey".
    matplotlib_scatter_kwargs : Dict[str, Any], optional
        Additional keyword arguments to pass to matplotlib's scatter function
        when plotting background cells. Common options include 'alpha', 's' (size),
        'edgecolors', and 'zorder'. Defaults match scanpy's styling with 
        {'zorder': 0, 'edgecolors': 'none', 'linewidths': 0, 'alpha': 0.7}.
    **kwargs : 
        All other parameters are passed directly to scanpy.pl.embedding.
        See scanpy.pl.embedding documentation for details on available parameters.
        
    Returns
    -------
    Whatever scanpy.pl.embedding returns based on your kwargs.
    If return_fig=True, returns the figure or (figure, axes) depending on scanpy version.
    Otherwise returns None.
    
    Notes
    -----
    This function requires scanpy. If scanpy is not available, it will raise a warning.
    See scanpy.pl.embedding documentation for full details of base plotting parameters.
    """
    # Check if scanpy is available
    if not _has_scanpy:
        warnings.warn(
            "Scanpy is required for plotting embeddings. Install scanpy to use this function."
        )
        return None
    
    # Process kwargs with special handling for show and return_fig
    user_show = kwargs.pop('show', None)
    user_return_fig = kwargs.pop('return_fig', False)
    
    # We need return_fig=True for our implementation regardless of user setting
    # And we'll handle the showing ourselves
    kwargs['show'] = False
    kwargs['return_fig'] = True
    
    # Calculate point size using scanpy's formula if not provided
    # Scanpy uses size = 120000 / n_cells if size is not specified
    user_size = kwargs.get('size', None)
    if user_size is None:
        # Calculate the point size based on the total number of cells
        total_points = adata.n_obs
        point_size = 120000 / total_points
        kwargs['size'] = point_size
    
    # Extract marker style to match background and foreground
    marker = kwargs.get('marker', '.')
    
    # Default matplotlib_scatter_kwargs with styling to match scanpy's defaults
    # Set zorder=0 to ensure background is behind, and match scanpy's default styling
    default_bg_kwargs = {
        'zorder': 0,       # Keep in background
        'edgecolors': 'none',  # No edges like scanpy
        'linewidths': 0,   # No edge width
        'alpha': 0.7       # Slight transparency (can be overridden)
    }
    
    if matplotlib_scatter_kwargs is None:
        matplotlib_scatter_kwargs = default_bg_kwargs
    else:
        # Merge defaults with user-provided kwargs
        # User settings take precedence over defaults
        for k, v in default_bg_kwargs.items():
            if k not in matplotlib_scatter_kwargs:
                matplotlib_scatter_kwargs[k] = v
    
    # Add point size to bg_kwargs if specified by user or calculated
    if user_size is not None and 's' not in matplotlib_scatter_kwargs:
        matplotlib_scatter_kwargs['s'] = user_size
    elif user_size is None and 's' not in matplotlib_scatter_kwargs:
        matplotlib_scatter_kwargs['s'] = point_size
    
    # Add marker to match foreground
    if 'marker' not in matplotlib_scatter_kwargs:
        matplotlib_scatter_kwargs['marker'] = marker
    
    # Handle different format for basis key between stored value and scanpy parameter
    basis_key = basis
    if not basis.startswith('X_') and f'X_{basis}' in adata.obsm:
        basis_key = f'X_{basis}'
    elif basis.startswith('X_') and basis not in adata.obsm:
        if basis[2:] in adata.obsm:
            basis_key = basis[2:]
    
    # Check if the basis exists
    if basis_key not in adata.obsm:
        available_bases = list(adata.obsm.keys())
        raise ValueError(f"Basis '{basis}' not found in adata.obsm. Available bases: {available_bases}")
    
    # Process groups - handle different formats
    if groups is None:
        # No filtering, use scanpy directly with all cells
        mask = np.ones(adata.n_obs, dtype=bool)
    elif isinstance(groups, dict):
        # Dictionary-based filtering
        mask = np.ones(adata.n_obs, dtype=bool)
        
        # Apply each filter condition
        for column, values in groups.items():
            if column not in adata.obs.columns:
                logger.warning(f"Column '{column}' not found in adata.obs. Skipping this filter.")
                continue
                
            # Convert single value to list for consistent handling
            if not isinstance(values, (list, tuple, np.ndarray)):
                values = [values]
                
            # Update mask to include only cells that match this condition
            column_mask = adata.obs[column].isin(values)
            mask = mask & column_mask
        
        # Log how many cells were selected
        n_selected = np.sum(mask)
        logger.info(f"Selected {n_selected:,} cells out of {adata.n_obs:,} total cells.")
        
        if n_selected == 0:
            logger.warning("No cells match the filtering criteria. Check your group filters.")
            return None
    else:
        # Groups is a string or list - pass directly to scanpy
        kwargs['groups'] = groups
        mask = np.ones(adata.n_obs, dtype=bool)
    
    # Create subset for scanpy to plot
    selected_adata = adata[mask]
    
    # Call scanpy embedding function with the subset
    result = sc.pl.embedding(
        selected_adata,
        basis=basis.replace("X_", ""),  # Scanpy doesn't want the X_ prefix
        **kwargs
    )
    
    # Add background points if requested and there are filtered cells
    has_filtered_cells = not np.all(mask)
    if background_color is not None and has_filtered_cells:
        # Get figure and axes from scanpy's result
        if isinstance(result, dict):
            # Multi-panel case where result is a dict of axes
            fig = next(iter(result.values())).figure
            axes_dict = result
            
            # Add background to each axis
            for ax in axes_dict.values():
                # Add background cells
                ax.scatter(
                    adata[~mask].obsm[basis_key][:, 0],
                    adata[~mask].obsm[basis_key][:, 1],
                    c=background_color,
                    **matplotlib_scatter_kwargs
                )
        else:
            # Single panel or figure with axes
            fig = result
            
            # Find the axes (might be different depending on scanpy version)
            if hasattr(fig, 'axes'):
                axes = fig.axes
            elif hasattr(fig, 'get_axes'):
                axes = fig.get_axes()
            else:
                axes = [plt.gca()]
            
            # Add background to each axis
            for ax in axes:
                # Add background cells
                ax.scatter(
                    adata[~mask].obsm[basis_key][:, 0],
                    adata[~mask].obsm[basis_key][:, 1],
                    c=background_color,
                    **matplotlib_scatter_kwargs
                )
    
    # Handle showing based on user preference
    if user_show is None:
        # Default behavior is to show if not returning the figure
        if not user_return_fig:
            plt.show()
    elif user_show:
        plt.show()
    
    # Return according to user preference
    if user_return_fig:
        return result
    else:
        return None