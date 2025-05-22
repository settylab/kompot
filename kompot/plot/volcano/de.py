"""Volcano plot functions for differential expression."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Colormap, ListedColormap
from typing import Optional, Union, List, Tuple, Dict, Any, Callable
from anndata import AnnData
import pandas as pd
import warnings
import logging

from ...utils import KOMPOT_COLORS
from ...anndata.utils import get_run_from_history
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
    highlight_genes: Optional[Union[List[str], Dict[str, str], List[Dict[str, Any]]]] = None,
    background_color_key: Optional[str] = None,
    background_cmap: Union[str, Colormap] = None,  # Will be auto-selected based on data type
    color_discrete_map: Optional[Dict[str, str]] = None,
    vmin: Optional[Union[float, str]] = None,
    vmax: Optional[Union[float, str]] = None,
    vcenter: Optional[float] = None,
    show_names: Union[bool, List[str]] = True,
    figsize: Tuple[float, float] = (10, 8),
    title: Optional[str] = None,
    xlabel: Optional[str] = "Log Fold Change",
    ylabel: Optional[str] = "Mahalanobis Distance",
    n_x_ticks: int = 3,
    n_y_ticks: int = 3,
    color_up: str = KOMPOT_COLORS["direction"]["up"],
    color_down: str = KOMPOT_COLORS["direction"]["down"],
    color_background: str = "#c0c0c0",  # Medium gray
    alpha_background: float = 1.0,
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
    group: Optional[str] = None,
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
    highlight_genes : list of str, dict of {str: str}, or list of dict, optional
        Can be:
        - A list of specific gene names to highlight on the plot
        - A dictionary where keys are gene names and values are colors
        - A list of dictionaries, each containing:
          - 'genes': list of gene names (required)
          - 'name': group name for the legend (optional)
          - 'color': color for this group (optional)
        If provided, this will override the `n_top_genes` parameter.
    background_color_key : str, optional
        Key in adata.var to use for coloring background genes. Can be continuous or categorical.
    background_cmap : str or Colormap, optional
        Colormap to use for background coloring. Default is for continuous 'Spectral_r'.
    color_discrete_map : dict, optional
        Mapping of category values to colors for categorical background_color_key.
        If not provided, colors will be selected from the colormap.
    vmin : float or str, optional
        Minimum value for colormap normalization. If a string starting with 'p' followed by a number,
        uses that percentile (e.g., 'p5' for 5th percentile).
    vmax : float or str, optional
        Maximum value for colormap normalization. If a string starting with 'p' followed by a number,
        uses that percentile (e.g., 'p95' for 95th percentile).
    vcenter : float, optional
        Center value for diverging colormaps. If provided with vmin/vmax, ensures proper ordering.
    show_names : bool or list of str, optional
        Whether to display gene names, or a list of specific gene names to annotate.
        If True, shows names for all highlighted genes. If False, shows no names.
        If a list, shows names only for genes in the list (default: True)
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
        Color for background genes when not using background_color_key
    alpha_background : float, optional
        Alpha value for background genes (default: 1.0)
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
    group : str, optional
        If provided, use data for a specific group/subset analyzed with the 'groups' parameter
        in compute_differential_expression. Will use the values from adata.varm instead of
        adata.var for Mahalanobis distances, mean fold changes, and weighted mean fold changes.
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
        # Use get_run_history to get the deserialized run history
        from ...anndata.utils import get_run_history
        run_history = get_run_history(adata, "de")
        if run_history is not None:
            actual_run_id = len(run_history) + run_id
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
                if 'condition1' in params and 'condition2' in params:
                    condition1 = params['condition1']
                    condition2 = params['condition2']
    
    # Log which run and fields are being used
    conditions_str = f": comparing {condition1} to {condition2}" if condition1 and condition2 else ""
    logger.info(f"Using DE run {actual_run_id}{conditions_str}")
    
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
    
    # Check if group-specific data is provided and should be used
    x = None
    y = None
    
    if group is not None:
        # Get run information to access field names
        run_info = get_run_from_history(adata, run_id, analysis_type="de")
        
        if not run_info or 'field_names' not in run_info:
            logger.warning(f"Cannot find run information for group-specific data. Make sure you're using the correct run_id.")
            return
        
        # Get varm keys directly from run information
        if 'varm_keys' not in run_info:
            logger.warning(f"No varm keys found in run information. This run may not have used groups.")
            return
            
        lfc_varm_key = run_info['varm_keys']['mean_lfc']
        score_varm_key = run_info['varm_keys']['mahalanobis']
        weighted_lfc_varm_key = run_info['varm_keys']['weighted_lfc']
        
        logger.debug(f"Using varm keys: lfc={lfc_varm_key}, score={score_varm_key}, weighted={weighted_lfc_varm_key}")
        
        # Check if the keys exist in varm and group is available
        lfc_data_available = (
            lfc_varm_key in adata.varm and 
            group in adata.varm[lfc_varm_key].columns
        )
        
        score_data_available = (
            score_varm_key in adata.varm and 
            group in adata.varm[score_varm_key].columns
        )
        
        weighted_lfc_data_available = (
            weighted_lfc_varm_key in adata.varm and
            group in adata.varm[weighted_lfc_varm_key].columns
        )
        
        if lfc_data_available and score_data_available:
            logger.info(f"Using group-specific data for group '{group}' from varm")
            logger.info(f"Using fields for DE plot - lfc_key: '{lfc_varm_key}', score_key: '{score_varm_key}'")
            x = adata.varm[lfc_varm_key][group].values
            y = adata.varm[score_varm_key][group].values
            
            # Log information about weighted mean log fold change
            if weighted_lfc_data_available:
                logger.info(f"Group-specific weighted mean log fold change data found for '{group}'")
            else:
                logger.info(f"Group-specific weighted mean log fold change data not available for '{group}'")
            
            # Update title to indicate group-specific data
            if title is None and condition1 and condition2:
                title = f"Volcano Plot: {condition1} vs {condition2} - Group: {group}"
            elif title is not None and "Group:" not in title:
                title = f"{title} - Group: {group}"
                
            # Log some basic stats about the data
            n_valid = np.sum(~np.isnan(x) & ~np.isnan(y))
            logger.info(f"Found {n_valid:,} valid genes with group-specific metrics for '{group}'")
            
        else:
            missing_keys = []
            if not lfc_data_available:
                missing_keys.append(f"{lfc_varm_key} for {group}")
            if not score_data_available:
                missing_keys.append(f"{score_varm_key} for {group}")
                
            # Check available groups for more helpful error message
            available_lfc_groups = []
            available_score_groups = []
            
            if lfc_varm_key in adata.varm:
                available_lfc_groups = list(adata.varm[lfc_varm_key].columns)
            
            if score_varm_key in adata.varm:
                available_score_groups = list(adata.varm[score_varm_key].columns)
                
            available_groups = set(available_lfc_groups).intersection(set(available_score_groups))
            
            if available_groups:
                group_str = ", ".join(sorted(available_groups))
                logger.warning(f"Group-specific data for '{group}' not found in varm. Missing: {', '.join(missing_keys)}. Available groups: {group_str}. Falling back to default data.")
            else:
                logger.warning(f"Group-specific data for '{group}' not found in varm. Missing: {', '.join(missing_keys)}. No groups available. Falling back to default data.")
    
    # If no group-specific data was found or no group was specified, use regular var data
    if x is None or y is None:
        x = adata.var[lfc_key].values if lfc_key is not None else None
        y = adata.var[score_key].values if score_key is not None else None
        
        # Handle cases where keys are missing
        if x is None or y is None:
            error_msg = []
            if x is None:
                error_msg.append(f"LFC key '{lfc_key}' not found in adata.var")
            if y is None:
                error_msg.append(f"Score key '{score_key}' not found in adata.var")
            
            error_str = " and ".join(error_msg)
            raise ValueError(f"Cannot create volcano plot: {error_str}")
            
        logger.info(f"Using data columns from var - lfc: '{lfc_key}', score: '{score_key}'")
        logger.info(f"Using fields for DE plot - lfc_key: '{lfc_key}', score_key: '{score_key}'")
    
    # Create a DataFrame with all relevant information
    data_dict = {
        'gene': adata.var_names,
        'lfc': x,
        'score': y
    }
    
    # Add sort_val - either from the specified sort_key or use y (score) by default
    if sort_key is not None:
        # If group-specific and sort_key appears to be a weighted_lfc column and group-specific weighted_lfc available
        if group is not None and "weighted" in sort_key.lower() and weighted_lfc_data_available:
            data_dict['sort_val'] = adata.varm[weighted_lfc_varm_key][group].values
            logger.info(f"Using group-specific weighted mean log fold change for sorting")
        else:
            data_dict['sort_val'] = adata.var[sort_key].values
    else:
        data_dict['sort_val'] = y
        
    de_data = pd.DataFrame(data_dict)
    
    # If background_color_key is provided, add it to the dataframe
    if background_color_key is not None and background_color_key in adata.var.columns:
        de_data['bg_color'] = adata.var[background_color_key].values
        
        # Determine if background color is categorical or continuous
        bg_values = adata.var[background_color_key]
        if (isinstance(bg_values.dtype, pd.CategoricalDtype) or 
            bg_values.dtype == 'object' or 
            bg_values.dtype == 'category'):
            bg_color_is_categorical = True
            categories = adata.var[background_color_key].unique()
            logger.info(f"Using categorical coloring for background with {len(categories):,} categories")
            
            # Auto-select appropriate colormap for categorical data if none provided
            if background_cmap is None:
                if len(categories) <= 10:
                    background_cmap = 'tab10'
                elif len(categories) <= 20:
                    background_cmap = 'tab20'
                else:
                    background_cmap = 'Set3'  # More pastel colors for many categories
                logger.info(f"Auto-selected '{background_cmap}' colormap for categorical data")
            
            # Create color map for categorical data
            if color_discrete_map is not None:
                # Use provided mapping
                category_colors = color_discrete_map
            else:
                # Generate colors from colormap for discrete data
                if isinstance(background_cmap, str):
                    base_cmap = mpl.colormaps[background_cmap]
                else:
                    base_cmap = background_cmap
                    
                # Get colors from the discrete colormap
                n_colors = len(categories)
                
                # For categorical colormaps, directly get colors from the colormap's list
                if hasattr(base_cmap, 'colors'):
                    # This works for ListedColormap instances like tab10, tab20, etc.
                    avail_colors = base_cmap.colors
                    # Just take the first n_colors (or cycle if we need more)
                    colors = [avail_colors[i % len(avail_colors)] for i in range(n_colors)]
                else:
                    # Fallback for other colormap types
                    colors = [base_cmap(i/max(1, n_colors-1)) for i in range(n_colors)]
                
                # Create a discrete colormap
                discrete_cmap = ListedColormap(colors)
                
                # Store both the discrete colormap and the category mapping
                cmap = discrete_cmap
                category_colors = {cat: colors[i] for i, cat in enumerate(categories)}
                
            # Map categories to colors
            bg_colors = [category_colors.get(val, 'gray') for val in de_data['bg_color']]
            bg_norm = None
            
        else:
            # Continuous background coloring
            bg_color_is_categorical = False
            logger.info(f"Using continuous coloring for background")
            
            # Auto-select appropriate colormap for continuous data if none provided
            if background_cmap is None:
                background_cmap = 'Spectral_r'  # Default continuous colormap
                logger.info(f"Auto-selected '{background_cmap}' colormap for continuous data")
            
            # Get colormap
            if isinstance(background_cmap, str):
                cmap = mpl.colormaps[background_cmap]
            else:
                cmap = background_cmap
            
            # Process vmin and vmax - handle percentile strings
            bg_values = de_data['bg_color'].values
            
            # Handle percentile strings for vmin
            if isinstance(vmin, str):
                if vmin.startswith('p'):
                    try:
                        percentile = float(vmin[1:])
                        vmin_value = np.nanpercentile(bg_values, percentile)
                        logger.info(f"Using {percentile}th percentile ({vmin_value}) for vmin")
                    except ValueError:
                        logger.warning(f"Invalid percentile format: {vmin}. Using data minimum.")
                        vmin_value = np.nanmin(bg_values)
                else:
                    # Handle non-percentile string values - try to convert to float or use min
                    try:
                        vmin_value = float(vmin)
                    except ValueError:
                        logger.warning(f"Invalid vmin value: {vmin}. Using data minimum.")
                        vmin_value = np.nanmin(bg_values)
            else:
                vmin_value = vmin if vmin is not None else np.nanmin(bg_values)
                
            # Handle percentile strings for vmax
            if isinstance(vmax, str):
                if vmax.startswith('p'):
                    try:
                        percentile = float(vmax[1:])
                        vmax_value = np.nanpercentile(bg_values, percentile)
                        logger.info(f"Using {percentile}th percentile ({vmax_value}) for vmax")
                    except ValueError:
                        logger.warning(f"Invalid percentile format: {vmax}. Using data maximum.")
                        vmax_value = np.nanmax(bg_values)
                else:
                    # Handle non-percentile string values - try to convert to float or use max
                    try:
                        vmax_value = float(vmax)
                    except ValueError:
                        logger.warning(f"Invalid vmax value: {vmax}. Using data maximum.")
                        vmax_value = np.nanmax(bg_values)
            else:
                vmax_value = vmax if vmax is not None else np.nanmax(bg_values)
            
            # Create appropriate normalization for the colormap
            if vcenter is not None:
                # Make sure vmin, vcenter, vmax are in correct order
                v_values = [v for v in [vmin_value, vcenter, vmax_value] if v is not None]
                vmin_value, vmax_value = min(v_values), max(v_values)
                
                # If vcenter is outside the range, adjust it
                if vcenter < vmin_value:
                    vcenter = vmin_value + 1e-16
                elif vcenter > vmax_value:
                    vcenter = vmax_value - 1e-16
                    
                logger.info(f"Using diverging normalization with vmin={vmin_value}, vcenter={vcenter}, vmax={vmax_value}")
                bg_norm = mpl.colors.TwoSlopeNorm(vmin=vmin_value, vcenter=vcenter, vmax=vmax_value)
            else:
                logger.info(f"Using linear normalization with vmin={vmin_value}, vmax={vmax_value}")
                bg_norm = mpl.colors.Normalize(vmin=vmin_value, vmax=vmax_value)
            
            # We'll use the scatter's built-in normalization for continuous colors
            bg_colors = de_data['bg_color'].values
            
    # Determine which genes to highlight
    highlight_groups = []
    
    if highlight_genes is not None:
        if isinstance(highlight_genes, str):
            # it might be just a single gene
            highlight_genes = [highlight_genes]
        # Process highlight_genes based on its type
        if isinstance(highlight_genes, list) and len(highlight_genes) > 0:
            if isinstance(highlight_genes[0], dict):
                # List of dictionaries format
                for i, group in enumerate(highlight_genes):
                    if 'genes' not in group:
                        logger.warning(f"Group {i} missing 'genes' key, skipping")
                        continue
                        
                    # Extract genes and filter for valid ones
                    gene_list = group['genes']
                    valid_genes = [g for g in gene_list if g in adata.var_names]
                        
                    if len(valid_genes) < len(gene_list):
                        missing_genes = set(gene_list) - set(valid_genes)
                        logger.warning(f"Group {i}: {len(missing_genes)} genes not found in the dataset")
                    
                    if not valid_genes:
                        logger.warning(f"Group {i}: No valid genes found, skipping")
                        continue
                    
                    # Use provided color or auto-generate
                    color = group.get('color')
                    name = group.get('name', f"Group {i+1}")
                    
                    # Add group to list
                    highlight_groups.append({
                        'genes': valid_genes,
                        'color': color,
                        'name': name
                    })
                    logger.info(f"Added highlight group '{name}' with {len(valid_genes)} genes")
            elif all(isinstance(item, (str, int)) for item in highlight_genes):
                # If highlight_genes is a list of strings or numbers, interpret as gene names
                valid_genes = [g for g in highlight_genes if g in adata.var_names]
                
                if len(valid_genes) < len(highlight_genes):
                    missing_genes = set(str(g) for g in highlight_genes) - set(str(g) for g in valid_genes)
                    logger.warning(f"{len(missing_genes)} genes not found in the dataset: {', '.join(str(g) for g in missing_genes)}")
                
                # Create a single group without custom colors
                highlight_groups.append({
                    'genes': valid_genes,
                    'name': "Highlighted genes"
                })
                logger.info(f"Highlighting {len(valid_genes)} user-specified genes")
            else:
                # List with mix of types - try to interpret as list of lists
                for i, group in enumerate(highlight_genes):
                    if isinstance(group, list):
                        # This is a list of genes
                        valid_genes = [g for g in group if g in adata.var_names]
                        
                        if len(valid_genes) < len(group):
                            missing_genes = set(str(g) for g in group) - set(str(g) for g in valid_genes)
                            logger.warning(f"Group {i}: {len(missing_genes)} genes not found in the dataset")
                        
                        if not valid_genes:
                            logger.warning(f"Group {i}: No valid genes found, skipping")
                            continue
                        
                        # Add as a group with auto-generated name
                        highlight_groups.append({
                            'genes': valid_genes,
                            'name': f"Group {i+1}"
                        })
                        logger.info(f"Added highlight group {i+1} with {len(valid_genes)} genes")
                
        elif isinstance(highlight_genes, dict):
            # Dictionary format: {gene_name: color}
            gene_list = list(highlight_genes.keys())
            valid_genes = [g for g in gene_list if g in adata.var_names]
            
            if len(valid_genes) < len(gene_list):
                missing_genes = set(gene_list) - set(valid_genes)
                logger.warning(f"{len(missing_genes)} genes not found in the dataset")
            
            # Create a single group with custom colors for each gene
            highlight_groups.append({
                'genes': valid_genes,
                'colors': {g: highlight_genes[g] for g in valid_genes},
                'name': "Highlighted genes"
            })
            logger.info(f"Highlighting {len(valid_genes)} genes with custom colors")
        else:
            # This case shouldn't be triggered anymore since we handle all list types above
            # But keeping for backward compatibility - Simple list of genes
            valid_genes = [g for g in highlight_genes if g in adata.var_names]
            
            if len(valid_genes) < len(highlight_genes):
                # Try to convert to strings for error reporting
                try:
                    missing_genes = set(str(g) for g in highlight_genes) - set(str(g) for g in valid_genes)
                    logger.warning(f"{len(missing_genes)} genes not found in the dataset: {', '.join(missing_genes)}")
                except:
                    logger.warning(f"Some genes not found in the dataset")
            
            # Create a single group without custom colors
            highlight_groups.append({
                'genes': valid_genes,
                'name': "Highlighted genes"
            })
            logger.info(f"Highlighting {len(valid_genes)} user-specified genes")
    else:
        # No highlight_genes provided, use top n_top_genes by score
        top_genes = de_data.sort_values('sort_val', ascending=False).head(n_top_genes)
        highlight_groups.append({
            'genes': top_genes['gene'].tolist(),
            'name': f"Top {n_top_genes} genes"
        })
        logger.info(f"Highlighting top {n_top_genes:,} genes by {sort_key or score_key}")
        
    # Plot background genes
    if background_color_key is not None:
        if bg_color_is_categorical:
            # Create a scatter plot for each category to add to legend
            for category, color in category_colors.items():
                mask = de_data['bg_color'] == category
                if mask.any():  # Only plot if we have points with this category
                    # Need to use color as a string or RGB tuple, not as 'c' parameter
                    # to avoid the single numeric RGB warning
                    ax.scatter(
                        de_data.loc[mask, 'lfc'].values,
                        de_data.loc[mask, 'score'].values,
                        alpha=alpha_background,
                        s=point_size,
                        color=color,  # Use 'color' parameter instead of 'c'
                        label=category,
                        **kwargs
                    )
        else:
            # Continuous coloring
            scatter = ax.scatter(
                de_data['lfc'].values,
                de_data['score'].values,
                alpha=alpha_background,
                s=point_size,
                c=bg_colors,
                cmap=cmap,
                norm=bg_norm,
                **kwargs
            )
            
            # Add colorbar for continuous values - position it in bottom part of right sidebar
            if show_legend:
                # Get the position of the axes
                bbox = ax.get_position()
                
                # Create a small vertical colorbar in the bottom part of the right sidebar
                # This coordinates with the legend placement to create a split sidebar
                
                # Calculate the height to be 20% of the plot height
                cax_height = bbox.height * 0.2  # 20% of plot height
                
                # Place it in the bottom section of the right side, centered vertically
                # Calculate center position vertically in lower third of plot
                sidebar_bottom_center = bbox.y0 + bbox.height * 0.3
                
                cax_rect = [
                    bbox.x0 + bbox.width + 0.01,             # x position (just to the right of the plot)
                    sidebar_bottom_center - cax_height/2,    # y position (centered in lower portion)
                    0.02,                                    # width (thin)
                    cax_height                               # height (20% of plot height)
                ]
                
                # Create a small vertical colorbar
                cax = fig.add_axes(cax_rect)
                cbar = fig.colorbar(scatter, cax=cax, orientation='vertical')
                
                # Adjust label and ticks for better visibility
                cbar.set_label(background_color_key, fontsize=10)
                cbar.ax.tick_params(labelsize=8)
    else:
        # Standard background coloring
        ax.scatter(
            de_data['lfc'].values,
            de_data['score'].values,
            alpha=alpha_background,
            s=point_size,
            c=color_background,
            label="All genes" if show_legend else None,
            **kwargs
        )
    
    # Process each highlight group
    for group in highlight_groups:
        # Get genes for this group
        genes = group['genes']
        group_df = de_data[de_data['gene'].isin(genes)].copy()
        
        # Determine how to color genes in this group
        if 'colors' in group:
            # Dictionary of per-gene colors
            for _, gene_row in group_df.iterrows():
                gene_name = gene_row['gene']
                color = group['colors'].get(gene_name)
                if color is None:
                    # Use default color based on direction
                    color = color_up if gene_row['lfc'] > 0 else color_down
                
                # Plot this gene
                ax.scatter(
                    gene_row['lfc'],
                    gene_row['score'],
                    alpha=1,
                    s=point_size*3,
                    c=color
                )
                
                # Add label if requested
                if show_names is True:
                    ax.annotate(
                        gene_name,
                        (gene_row['lfc'], gene_row['score']),
                        fontsize=font_size,
                        **text_kwargs
                    )
        else:
            # Single color for the whole group (or split by direction)
            group_color = group.get('color')
            
            if group_color is not None:
                # Use single color for the whole group
                ax.scatter(
                    group_df['lfc'].values,
                    group_df['score'].values,
                    alpha=1,
                    s=point_size*3,
                    c=group_color,
                    label=group['name']
                )
                
                # Add labels if requested
                if show_names is True:
                    for _, gene_row in group_df.iterrows():
                        ax.annotate(
                            gene_row['gene'],
                            (gene_row['lfc'], gene_row['score']),
                            fontsize=font_size,
                            **text_kwargs
                        )
            else:
                # Split by direction (up/down regulated)
                up_genes = group_df[group_df['lfc'] > 0]
                down_genes = group_df[group_df['lfc'] < 0]
                
                # Plot up-regulated genes
                if len(up_genes) > 0:
                    ax.scatter(
                        up_genes['lfc'].values,
                        up_genes['score'].values,
                        alpha=1,
                        s=point_size*3,
                        c=color_up,
                        label=f"{group['name']} - Higher in {condition2}" if condition2 else f"{group['name']} - Up-regulated"
                    )
                    
                    # Add labels if requested
                    if show_names is True:
                        for _, gene_row in up_genes.iterrows():
                            ax.annotate(
                                gene_row['gene'],
                                (gene_row['lfc'], gene_row['score']),
                                fontsize=font_size,
                                **text_kwargs
                            )
                
                # Plot down-regulated genes
                if len(down_genes) > 0:
                    ax.scatter(
                        down_genes['lfc'].values,
                        down_genes['score'].values,
                        alpha=1,
                        s=point_size*3,
                        c=color_down,
                        label=f"{group['name']} - Higher in {condition1}" if condition1 else f"{group['name']} - Down-regulated"
                    )
                    
                    # Add labels if requested
                    if show_names is True:
                        for _, gene_row in down_genes.iterrows():
                            ax.annotate(
                                gene_row['gene'],
                                (gene_row['lfc'], gene_row['score']),
                                fontsize=font_size,
                                **text_kwargs
                            )
    
    # If show_names is a list, label those specific genes
    if isinstance(show_names, list):
        genes_to_label = [g for g in show_names if g in adata.var_names]
        
        if genes_to_label:
            # Get data for these genes and add labels
            genes_df = de_data[de_data['gene'].isin(genes_to_label)]
            for _, gene_row in genes_df.iterrows():
                ax.annotate(
                    gene_row['gene'],
                    (gene_row['lfc'], gene_row['score']),
                    fontsize=font_size,
                    **text_kwargs
                )
    
    # Create dummy entries for the legend if no highlighted genes
    if len(highlight_groups) == 0 and show_legend:
        ax.scatter([], [], alpha=1, s=point_size*3, c=color_up, 
                  label=f"Higher in {condition2}" if condition2 else "Up-regulated")
        ax.scatter([], [], alpha=1, s=point_size*3, c=color_down,
                  label=f"Higher in {condition1}" if condition1 else "Down-regulated")
    
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
    
    # Prepare to handle legend placement in coordination with the colorbar
    # Will place them both in a split sidebar when using continuous background colors
    has_continuous_colorbar = (background_color_key is not None and not bg_color_is_categorical)
    
    # Add legend with appropriate styling
    if show_legend and legend_loc != 'none':
        if has_continuous_colorbar:
            # For continuous colorbar case: place legend in top part of right sidebar
            if legend_loc == 'best':
                legend = ax.legend(
                    bbox_to_anchor=(1.05, 0.7),  # Position in top part of sidebar
                    loc='upper left', 
                    fontsize=legend_fontsize,
                    frameon=False,
                    ncol=legend_ncol or 1
                )
            else:
                # If user specified a different location, respect it
                legend = ax.legend(
                    loc=legend_loc, 
                    fontsize=legend_fontsize,
                    frameon=False,
                    ncol=legend_ncol or 1
                )
        else:
            # Standard legend placement without colorbar competition
            if legend_loc == 'best':
                legend = ax.legend(
                    bbox_to_anchor=(1.05, 1), 
                    loc='upper left', 
                    fontsize=legend_fontsize,
                    frameon=False,
                    ncol=legend_ncol or 1
                )
            else:
                legend = ax.legend(
                    loc=legend_loc, 
                    fontsize=legend_fontsize,
                    frameon=False,
                    ncol=legend_ncol or 1
                )
    
    if grid:
        ax.grid(**grid_kwargs)
    
    # Instead of tight_layout, manually adjust the plot's spacing
    # This avoids issues with colorbars and other axes elements
    if has_continuous_colorbar or (show_legend and legend_loc == 'best'):
        # Make room for the right sidebar with legend and/or colorbar
        plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)
    else:
        # Standard spacing when no sidebar is needed
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    
    # Save figure if path provided
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
    
    # Return figure and axes if requested
    if return_fig:
        return fig, ax
    elif save is None:
        # Only show if not saving and not returning
        # Check if the current backend allows for interactive display
        if plt.get_backend().lower() not in ['agg', 'pdf', 'svg', 'ps', 'cairo', 'template']:
            plt.show()