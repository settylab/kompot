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
    mgroups: Optional[List[Dict[str, Union[str, List[str]]]]] = None,
    ncols: Optional[int] = None,
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
    mgroups : List[Dict[str, Union[str, List[str]]]], optional
        List of groups dictionaries to create multiple panels. Each element is treated 
        as a separate groups argument in its own subplot. Cannot be used with multiple colors.
        If provided, title argument should align with the number of groups in mgroups.
    ncols : int, optional
        Number of columns for panel layout when using mgroups. Default is 4 or less
        depending on the number of panels.
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
    
    # Handle mgroups parameter (multiple groups in subplots)
    if mgroups is not None:
        # Check if color is a list, which is incompatible with mgroups
        if 'color' in kwargs and isinstance(kwargs['color'], (list, tuple, np.ndarray)):
            raise ValueError("Cannot use multiple colors (list of color values) with mgroups parameter.")
        
        # Extract relevant parameters for subplot creation
        user_return_fig = kwargs.get('return_fig', False)
        user_show = kwargs.get('show', None)
        
        # Get or create titles for each subplot
        titles = kwargs.pop('title', None)
        if titles is None:
            # Generate default titles based on group definitions
            titles = []
            for i, group_dict in enumerate(mgroups):
                if isinstance(group_dict, dict):
                    # Create descriptive title based on group filtering
                    parts = []
                    for col, vals in group_dict.items():
                        if not isinstance(vals, (list, tuple, np.ndarray)):
                            vals = [vals]
                        parts.append(f"{col}={','.join(str(v) for v in vals)}")
                    titles.append(" & ".join(parts))
                else:
                    # If not a dict, use a simple generic title
                    titles.append(f"Group {i+1}")
        elif isinstance(titles, str):
            # Convert single string title to list with placeholders for other panels
            titles = [titles] + [f"Group {i+1}" for i in range(1, len(mgroups))]
        
        # Ensure titles match number of groups
        if len(titles) < len(mgroups):
            # Add generic titles for any missing
            titles.extend([f"Group {i+1}" for i in range(len(titles), len(mgroups))])
        
        # Create subplots using scanpy's grid spec
        n_panels = len(mgroups)
        
        # Determine number of columns (user-specified or default)
        if ncols is not None:
            n_cols = ncols
        else:
            n_cols = min(4, n_panels)  # Default: up to 4 columns, then wrap
            
        n_rows = (n_panels - 1) // n_cols + 1
        
        # Create figure with appropriate size
        figsize = kwargs.pop('figsize', None)
        if figsize is None:
            # Default sizing similar to scanpy
            figsize = (4 * n_cols, 4 * n_rows)
        
        fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize, 
                               squeeze=False,  # Always return 2D array of axes
                               tight_layout=True)
        axs = axs.flatten()  # Flatten to 1D for easier indexing
        
        # Create each subplot recursively
        for i, (group_dict, title, ax) in enumerate(zip(mgroups, titles, axs)):
            # Skip this call if we've run out of groups
            if i >= len(mgroups):
                ax.axis('off')  # Hide unused axes
                continue
                
            # Create a copy of kwargs for this subplot
            subplot_kwargs = kwargs.copy()
            subplot_kwargs['ax'] = ax
            subplot_kwargs['title'] = title
            subplot_kwargs['show'] = False  # Never show individual subplots
            subplot_kwargs['return_fig'] = False  # Don't return individual figures
            
            # Make the recursive call for this panel
            embedding(
                adata=adata,
                basis=basis,
                groups=group_dict,
                background_color=background_color,
                matplotlib_scatter_kwargs=matplotlib_scatter_kwargs,
                **subplot_kwargs
            )
            
            # Calculate and display cell fraction in title if title looks auto-generated
            if title.startswith("Group ") or "=" in title:
                if isinstance(group_dict, dict):
                    # Calculate mask for this group
                    mask = np.ones(adata.n_obs, dtype=bool)
                    for column, values in group_dict.items():
                        if column not in adata.obs.columns:
                            continue
                        if not isinstance(values, (list, tuple, np.ndarray)):
                            values = [values]
                        column_mask = adata.obs[column].isin(values)
                        mask = mask & column_mask
                    
                    # Add percentage to title
                    cell_fraction = np.sum(mask) / adata.n_obs
                    ax.set_title(f"{title}\n({cell_fraction:.1%} of cells)")
        
        # Hide any unused axes
        for i in range(len(mgroups), len(axs)):
            axs[i].axis('off')
        
        # Handle figure showing based on user preference
        if user_show is None or user_show:
            plt.tight_layout()
            plt.show()
        
        # Return the figure if requested
        if user_return_fig:
            return fig
        else:
            return None
        
    # Single plot case - process kwargs with special handling for show and return_fig
    user_show = kwargs.pop('show', None)
    user_return_fig = kwargs.pop('return_fig', False)
    
    # Pass ncols to scanpy if it was provided but mgroups is not used
    if ncols is not None:
        kwargs['ncols'] = ncols
    
    # We need return_fig=True for our implementation regardless of user setting
    # And we'll handle the showing ourselves
    kwargs['show'] = False
    kwargs['return_fig'] = True
    
    # Calculate point size using scanpy's formula if not provided
    # Scanpy uses size = 120000 / n_cells if size is not specified
    user_size = kwargs.get('s', None)
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
    
    # When ax is provided, we need to modify kwargs for scanpy
    if 'ax' in kwargs:
        # Store the figure for returning later if needed
        ax_fig = kwargs['ax'].figure
        
        # Create a copy of kwargs to avoid modifying the original
        scanpy_kwargs = kwargs.copy()
        
        # When ax is provided, set return_fig=False to avoid scanpy error
        scanpy_kwargs['return_fig'] = False
        
        # Call scanpy embedding function with the subset
        result = sc.pl.embedding(
            selected_adata,
            basis=basis.replace("X_", ""),  # Scanpy doesn't want the X_ prefix
            **scanpy_kwargs
        )
        
        # Store the figure as the result if user wants it returned
        if user_return_fig:
            result = ax_fig
    else:
        # Normal case - no ax provided
        result = sc.pl.embedding(
            selected_adata,
            basis=basis.replace("X_", ""),  # Scanpy doesn't want the X_ prefix
            **kwargs
        )
    
    # Add background points if requested and there are filtered cells
    has_filtered_cells = not np.all(mask)
    if background_color is not None and has_filtered_cells:
        if 'ax' in kwargs:
            # User provided ax parameter - use it directly
            axes = [kwargs['ax']]
            
            # Add background to the specified axis
            for ax in axes:
                ax.scatter(
                    adata[~mask].obsm[basis_key][:, 0],
                    adata[~mask].obsm[basis_key][:, 1],
                    c=background_color,
                    **matplotlib_scatter_kwargs
                )
        elif isinstance(result, dict):
            # Multi-panel case where result is a dict of axes
            axes_dict = result
            
            # Add background to each axis
            for ax in axes_dict.values():
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
                ax.scatter(
                    adata[~mask].obsm[basis_key][:, 0],
                    adata[~mask].obsm[basis_key][:, 1],
                    c=background_color,
                    **matplotlib_scatter_kwargs
                )
    
    # Handle showing based on user preference
    # Don't show automatically if an ax is provided (user is likely building a multi-panel figure)
    if 'ax' in kwargs:
        # Only show if user explicitly requested it
        if user_show:
            plt.show()
    else:
        # Normal showing behavior for standalone plots
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