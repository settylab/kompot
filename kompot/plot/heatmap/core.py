"""Core heatmap plotting functions."""

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

from ..volcano import _extract_conditions_from_key
from ...utils import get_run_from_history, KOMPOT_COLORS
from .utils import (_infer_heatmap_keys, _prepare_gene_list, _get_expression_matrix, 
                   _filter_excluded_groups, _apply_scaling, _calculate_figsize, 
                   _setup_colormap_normalization)
from .visualization import _draw_diagonal_split_cell

try:
    import scanpy as sc
    _has_scanpy = True
except ImportError:
    _has_scanpy = False

logger = logging.getLogger("kompot")

def heatmap(
    adata: AnnData,
    var_names: Optional[Union[List[str], Sequence[str]]] = None,
    groupby: str = None,
    n_top_genes: int = 20,
    gene_list: Optional[Union[List[str], Sequence[str]]] = None,
    lfc_key: Optional[str] = None,
    score_key: Optional[str] = None,
    layer: Optional[str] = None,
    standard_scale: Optional[Union[str, int]] = "var",  # Default to gene-wise z-scoring
    cmap: Optional[Union[str, mcolors.Colormap]] = None,
    dendrogram: bool = False,  # Whether to show dendrograms
    cluster_rows: bool = True,  # Whether to cluster rows
    cluster_cols: bool = True,  # Whether to cluster columns
    dendrogram_color: str = "black",  # Default dendrogram color
    figsize: Optional[Tuple[float, float]] = None,
    tile_aspect_ratio: float = 1.0,  # Default aspect ratio of individual tiles (width/height)
    tile_size: float = 0.3,     # Size for each tile in inches (reference dimension, width for square tiles)
    fixed_spacing: bool = True,  # Whether to use fixed spacing for labels, dendrograms, etc.
    show_gene_labels: bool = True,
    show_group_labels: bool = True,
    gene_labels_size: int = 10,
    group_labels_size: int = 12,
    colorbar_title: Optional[str] = None,
    colorbar_kwargs: Optional[Dict[str, Any]] = None,
    title: Optional[str] = None,
    sort_genes: bool = True,
    vcenter: Optional[Union[float, str]] = None,
    vmin: Optional[Union[float, str]] = None,
    vmax: Optional[Union[float, str]] = None,
    ax: Optional[plt.Axes] = None,
    return_fig: bool = False,
    save: Optional[str] = None,
    run_id: Optional[int] = None,
    condition_column: Optional[str] = None,
    observed: bool = True,
    condition1_name: Optional[str] = None,
    condition2_name: Optional[str] = None,
    exclude_groups: Optional[Union[str, List[str]]] = None,
    **kwargs,
) -> Union[
    None, Tuple[plt.Figure, plt.Axes], Tuple[plt.Figure, plt.Axes, Dict[str, plt.Axes]]
]:
    """
    Create a split-cell heatmap visualizing gene expression data for two conditions.
    
    The heatmap displays expression values with diagonally split cells, where the lower-left
    triangle shows values for the first condition and the upper-right triangle shows values
    for the second condition. This creates a compact visualization that highlights
    differences between conditions.
    
    Genes are shown on the y-axis and groups (cell types, clusters, etc.) are shown
    on the x-axis, with a legend and colorbar positioned to the right of the plot.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing expression data
    var_names : list, optional
        List of genes to include in the heatmap. If None, will use top genes
        based on score_key and lfc_key.
    groupby : str, optional
        Key in adata.obs for grouping cells
    n_top_genes : int, optional
        Number of top genes to include if var_names is None
    gene_list : list, optional
        Explicit list of genes to include in the heatmap. 
        Takes precedence over var_names if both are provided.
    lfc_key : str, optional
        Key in adata.var for log fold change values.
        If None, will try to infer from run information.
    score_key : str, optional
        Key in adata.var for significance scores.
        If None, will try to infer from run information.
    layer : str, optional
        Layer in AnnData to use for expression values. If None, uses .X
    standard_scale : str or int, optional
        Whether to scale the expression values ('var', 'group' or 0, 1).
        Default is 'var' for gene-wise z-scoring. When any z-scoring is applied,
        the colormap is automatically centered at 0 (vcenter=0), uses symmetric limits (equal
        positive and negative ranges), and uses a divergent colormap unless
        vcenter, vmin, vmax, or cmap is explicitly specified.
    cmap : str or colormap, optional
        Colormap to use for the heatmap. If None, defaults to "coolwarm" (divergent) when 
        z-scoring is applied, and "viridis" (sequential) otherwise.
    dendrogram : bool, optional
        Whether to show dendrograms for hierarchical clustering
    cluster_rows : bool, optional
        Whether to cluster rows (genes)
    cluster_cols : bool, optional
        Whether to cluster columns (groups)
    dendrogram_color : str, optional
        Color for dendrograms
    figsize : tuple, optional
        Figure size as (width, height) in inches. If None, will be calculated based on
        data dimensions, cell_size, and aspect_ratio.
    tile_aspect_ratio : float, optional
        Aspect ratio of individual tiles (width/height). Default is 1.0 (square tiles).
        Values > 1 create wider tiles, values < 1 create taller tiles.
    tile_size : float, optional
        Base size in inches for each tile when automatically calculating figure size.
        Default is 0.5 inches. For square tiles (tile_aspect_ratio=1), this is the width and height.
        For non-square tiles, this is the width if tile_aspect_ratio > 1, or the height if 
        tile_aspect_ratio < 1.
    fixed_spacing : bool, optional
        If True, uses fixed spacing for dendrograms, labels, and other elements proportional
        to the tile size, creating a more consistent layout. If False, uses the legacy spacing
        approach.
    show_gene_labels : bool, optional
        Whether to show gene labels
    show_group_labels : bool, optional
        Whether to show group labels
    gene_labels_size : int, optional
        Font size for gene labels
    group_labels_size : int, optional
        Font size for group labels
    colorbar_title : str, optional
        Title for the colorbar. If None, will default to "Z-score" when any z-scoring is applied
        (standard_scale="var", standard_scale="group", or standard_scale=0, 1),
        and "Expression" otherwise.
    colorbar_kwargs : dict, optional
        Additional parameters for colorbar customization. Supported keys include:
        - 'label_kwargs': dict with parameters for colorbar label (e.g. fontsize, color)
        - 'locator': A matplotlib Locator instance for tick positions
        - 'formatter': A matplotlib Formatter instance for tick labels
        - Any attribute of matplotlib colorbar instance
    title : str, optional
        Title for the heatmap
    sort_genes : bool, optional
        Whether to sort genes by score
    vcenter : float or str, optional
        Value to center the colormap at. If None and any z-scoring is applied 
        (standard_scale='var', 'group', 0, or 1), the colormap will be centered at 0.
        If None and no z-scoring is applied, a standard (non-centered) colormap will be used.
        Can be specified as a percentile using 'p<number>' format (e.g., 'p50' for median).
    vmin : float or str, optional
        Minimum value for colormap. If None and z-scoring is applied, will use a 
        symmetric limit based on the maximum absolute value of the data.
        Can be specified as a percentile using 'p<number>' format (e.g., 'p5' for 5th percentile).
    vmax : float or str, optional
        Maximum value for colormap. If None and z-scoring is applied, will use a
        symmetric limit based on the maximum absolute value of the data.
        Can be specified as a percentile using 'p<number>' format (e.g., 'p95' for 95th percentile).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure
    return_fig : bool, optional
        If True, returns the figure and axes
    save : str, optional
        Path to save figure. If None, figure is not saved
    run_id : int, optional
        Specific run ID to use for fetching field names from run history.
        -1 (default) is the latest run.
    condition_column : str, optional
        Column in adata.obs containing condition information.
        If None, tries to infer from run_info.
    observed : bool, optional
        Whether to use only observed combinations in groupby operations.
    condition1_name, condition2_name : str, optional
        Display names for the two conditions. If None, tries to infer from run_info.
    exclude_groups : str or list, optional
        Group name(s) to exclude from the heatmap.
    **kwargs : 
        Additional keyword arguments passed to matplotlib

    Returns
    -------
    If return_fig is True and dendrogram is False, returns (fig, ax)
    If return_fig is True and dendrogram is True, returns (fig, ax, dendrogram_axes)
    """
    # Normalize run_id to use -1 (latest run) if None
    effective_run_id = -1 if run_id is None else run_id
    
    # Prepare gene list and get run info
    var_names, lfc_key, score_key, run_info = _prepare_gene_list(
        adata=adata,
        var_names=var_names,
        gene_list=gene_list,
        n_top_genes=n_top_genes,
        lfc_key=lfc_key,
        score_key=score_key,
        sort_genes=sort_genes,
        run_id=effective_run_id,
    )
    
    # Extract key parameters from run_info
    condition1 = None
    condition2 = None
    condition_key = None
    actual_run_id = None
    inferred_layer = None
    
    if run_info is not None and "params" in run_info:
        params = run_info["params"]
        if "conditions" in params and len(params["conditions"]) == 2:
            condition1 = params["conditions"][0]
            condition2 = params["conditions"][1]
        # In Kompot DE runs, the condition column is called "groupby"
        if "groupby" in params:
            condition_key = params["groupby"]
        # Try to infer layer from run_info
        if "layer" in params:
            inferred_layer = params["layer"]
            
    # Use inferred layer if no explicit layer is provided
    if layer is None and inferred_layer is not None:
        layer = inferred_layer
        logger.info(f"Using layer '{layer}' inferred from run information")
    
    # Try to extract from key name if still not found
    if (condition1 is None or condition2 is None) and lfc_key is not None:
        conditions = _extract_conditions_from_key(lfc_key)
        if conditions:
            condition1, condition2 = conditions
    
    # Use condition names for display
    if condition1_name is None:
        condition1_name = condition1
    if condition2_name is None:
        condition2_name = condition2
    
    # If condition_column is None, use the one from run_info
    if condition_column is None and condition_key is not None:
        condition_column = condition_key
        logger.info(f"Using condition_column '{condition_column}' from run information")
    
    # Log which run is being used
    conditions_str = (
        f": comparing {condition1} vs {condition2}"
        if condition1 and condition2
        else ""
    )
    logger.info(f"Using DE run {effective_run_id} for heatmap{conditions_str}")

    # Log the plot type
    logger.info(f"Creating split heatmap with {len(var_names)} genes/features")

    # Validate condition column
    if condition_column is None:
        logger.warning("No condition_column could be inferred. Split heatmap requires a condition column.")
        return None

    # Check that condition column exists
    if condition_column not in adata.obs.columns:
        logger.error(f"Condition column '{condition_column}' not found in adata.obs")
        return None

    # Check for presence of both conditions in data
    if condition1 not in adata.obs[condition_column].unique():
        logger.error(f"Condition '{condition1}' not found in {condition_column}")
        return None
    if condition2 not in adata.obs[condition_column].unique():
        logger.error(f"Condition '{condition2}' not found in {condition_column}")
        return None

    # Get expression data
    expr_matrix = _get_expression_matrix(adata, var_names, layer)

    # Create dataframe with expression, condition column, and groupby column if any
    expr_df = pd.DataFrame(
        expr_matrix,
        index=adata.obs_names,
        columns=var_names
    )
    expr_df[condition_column] = adata.obs[condition_column].values

    # Add groupby column if provided
    if groupby is not None:
        if groupby not in adata.obs.columns:
            logger.error(f"Groupby column '{groupby}' not found in adata.obs")
            return None
        expr_df[groupby] = adata.obs[groupby].values

    # Split by condition
    cond1_df = expr_df[expr_df[condition_column] == condition1].drop(columns=[condition_column])
    cond2_df = expr_df[expr_df[condition_column] == condition2].drop(columns=[condition_column])

    # Group by the groupby column if provided
    if groupby is not None:
        # Get all unique groups for column reference
        all_groups = sorted(adata.obs[groupby].unique())
        
        # Filter out excluded groups if any
        if exclude_groups is not None:
            # Filter expression dataframes
            cond1_df = _filter_excluded_groups(cond1_df, groupby, exclude_groups, all_groups)
            cond2_df = _filter_excluded_groups(cond2_df, groupby, exclude_groups, all_groups)
            
            # Update all_groups list after filtering
            all_groups = sorted(set(cond1_df[groupby].unique()) | set(cond2_df[groupby].unique()))

        # Calculate mean expression per group
        cond1_means = (
            cond1_df.groupby(groupby, observed=observed)
            .mean()
            .reindex(all_groups)
            .loc[lambda df: ~df.index.isnull()]
        )
        cond2_means = (
            cond2_df.groupby(groupby, observed=observed)
            .mean()
            .reindex(all_groups)
            .loc[lambda df: ~df.index.isnull()]
        )

        # Save shape for figsize calculation
        n_groups = len(cond1_means)
        n_genes = len(var_names)

        # Apply scaling if needed
        if standard_scale is not None:
            # Get shared columns to ensure proper alignment
            shared_cols = list(set(cond1_means.columns).intersection(set(cond2_means.columns)))
            
            # Create a MultiIndex DataFrame with both conditions
            # This ensures z-scoring happens across both conditions together
            combined = pd.concat(
                [
                    cond1_means[shared_cols], 
                    cond2_means[shared_cols]
                ], 
                keys=["cond1", "cond2"],
                names=["condition", "group"]
            )
            
            # Apply scaling - set is_split=True to ensure proper handling of hierarchical structure
            # Use log_message=True to let the utility function handle logging
            scaled = _apply_scaling(combined, standard_scale, is_split=True, has_hierarchical_index=True)
            
            # Extract the results
            cond1_means_scaled = scaled.loc["cond1"].copy()
            cond2_means_scaled = scaled.loc["cond2"].copy()
            
            # Copy scaled values back to original dataframes to preserve any columns
            # that might not have been in both conditions
            for col in shared_cols:
                cond1_means[col] = cond1_means_scaled[col]
                cond2_means[col] = cond2_means_scaled[col]

        # Calculate a custom figure size with fixed aspect ratio tiles and consistent spacing
        if figsize is None:
            # Define constants for fixed spacing in multiples of the base unit
            GENE_LABEL_SPACE = 3.5    # Space for gene labels (y-axis)
            GROUP_LABEL_SPACE = 2.0   # Space for group labels (x-axis)
            TITLE_SPACE = 3.0         # Space for title
            LEGEND_SPACE = 5.0        # Space for legend
            COLORBAR_SPACE = 3.0      # Space for colorbar
            ROW_DENDROGRAM_SPACE = 2.5 # Space for row dendrogram
            COL_DENDROGRAM_SPACE = 2.5 # Space for column dendrogram
            
            # Calculate base tile dimensions - determine reference dimension
            if tile_aspect_ratio >= 1.0:
                # Wider or square tiles
                tile_width = tile_size
                tile_height = tile_size / tile_aspect_ratio
            else:
                # Taller tiles
                tile_height = tile_size
                tile_width = tile_size * tile_aspect_ratio
            
            # Space unit for fixed elements - use the larger dimension of the tile for consistent spacing
            base_unit = max(tile_width, tile_height)
            
            # Calculate data area dimensions in base units
            data_width_units = n_groups
            data_height_units = n_genes
            
            # Convert to inches
            data_width_inches = data_width_units * tile_width
            data_height_inches = data_height_units * tile_height
            
            # Calculate space for each component in absolute inches
            # Left area - for gene labels
            left_area_inches = GENE_LABEL_SPACE * base_unit if show_gene_labels else base_unit
            
            # Bottom area - for group labels
            bottom_area_inches = GROUP_LABEL_SPACE * base_unit if show_group_labels else base_unit
            
            # Right area - for column dendrogram and/or sidebar
            right_area_inches = max(LEGEND_SPACE, COLORBAR_SPACE) * base_unit
            if dendrogram and cluster_cols:
                right_area_inches += COL_DENDROGRAM_SPACE * base_unit
            
            # Top area - for title and row dendrogram
            top_area_inches = TITLE_SPACE * base_unit
            if dendrogram and cluster_rows:
                top_area_inches += ROW_DENDROGRAM_SPACE * base_unit
            
            # Calculate final figure dimensions
            width_inches = left_area_inches + data_width_inches + right_area_inches
            height_inches = bottom_area_inches + data_height_inches + top_area_inches
            
            # Cap figure size for very large data
            max_width = 30
            max_height = 30
            if width_inches > max_width or height_inches > max_height:
                # Scale down while preserving aspect ratio
                scale_factor = min(max_width / width_inches, max_height / height_inches)
                width_inches *= scale_factor
                height_inches *= scale_factor
                
                # Scale all dimensions proportionally
                left_area_inches *= scale_factor
                bottom_area_inches *= scale_factor
                right_area_inches *= scale_factor
                top_area_inches *= scale_factor
                data_width_inches *= scale_factor
                data_height_inches *= scale_factor
                base_unit *= scale_factor
            
            figsize = (width_inches, height_inches)
            
            # Store all the calculated dimensions for later use
            fig_dims = {
                # Store original units for reference
                'base_unit': base_unit,
                'tile_width': tile_width,
                'tile_height': tile_height,
                
                # Store absolute dimensions in inches
                'width_inches': width_inches,
                'height_inches': height_inches,
                'left_area_inches': left_area_inches,
                'bottom_area_inches': bottom_area_inches,
                'right_area_inches': right_area_inches,
                'top_area_inches': top_area_inches,
                'data_width_inches': data_width_inches,
                'data_height_inches': data_height_inches,
                
                # Store flag for dendrogram presence
                'has_row_dendrogram': dendrogram and cluster_rows,
                'has_col_dendrogram': dendrogram and cluster_cols
            }
        else:
            # When figsize is explicitly provided, we need to calculate reasonable dimensions
            # Calculate tile dimensions
            if tile_aspect_ratio >= 1.0:
                tile_width = tile_size
                tile_height = tile_size / tile_aspect_ratio
            else:
                tile_height = tile_size
                tile_width = tile_size * tile_aspect_ratio
                
            base_unit = max(tile_width, tile_height)
            
            # Use default proportions when figsize is user-provided
            width_inches, height_inches = figsize
            
            # Approximate reasonable areas based on base_unit
            left_area_inches = min(3.0 * base_unit, width_inches * 0.2)
            right_area_inches = min(5.0 * base_unit, width_inches * 0.3)
            bottom_area_inches = min(2.0 * base_unit, height_inches * 0.15)
            top_area_inches = min(2.0 * base_unit, height_inches * 0.15)
            if dendrogram:
                if cluster_rows:
                    top_area_inches += min(2.0 * base_unit, height_inches * 0.1)
                if cluster_cols:
                    right_area_inches += min(2.0 * base_unit, width_inches * 0.1)
            
            # Calculate data area from the remaining space
            data_width_inches = width_inches - left_area_inches - right_area_inches
            data_height_inches = height_inches - bottom_area_inches - top_area_inches
            
            # Store all dimensions
            fig_dims = {
                'base_unit': base_unit,
                'tile_width': tile_width,
                'tile_height': tile_height,
                
                'width_inches': width_inches,
                'height_inches': height_inches,
                'left_area_inches': left_area_inches, 
                'bottom_area_inches': bottom_area_inches,
                'right_area_inches': right_area_inches,
                'top_area_inches': top_area_inches,
                'data_width_inches': data_width_inches,
                'data_height_inches': data_height_inches,
                
                'has_row_dendrogram': dendrogram and cluster_rows,
                'has_col_dendrogram': dendrogram and cluster_cols
            }
        
        # Create figure if no axes provided
        create_fig = ax is None
        if create_fig:
            # Create figure with the calculated size
            fig = plt.figure(figsize=figsize)
            
            # Calculate plot areas using absolute inches - convert to figure coordinates
            fig_width, fig_height = fig.get_size_inches()
            
            # Calculate main heatmap position
            main_left = fig_dims['left_area_inches'] / fig_width
            main_bottom = fig_dims['bottom_area_inches'] / fig_height
            main_width = fig_dims['data_width_inches'] / fig_width
            main_height = fig_dims['data_height_inches'] / fig_height
            
            # Create main axes for the heatmap
            ax = fig.add_axes([main_left, main_bottom, main_width, main_height])
            
            # Add title area as a separate axes
            title_height = (TITLE_SPACE * fig_dims['base_unit']) / fig_height
            title_bottom = 1.0 - title_height
            title_ax = fig.add_axes([0, title_bottom, 1.0, title_height])
            title_ax.set_axis_off()
            
            # Add dendrogram axes if needed
            dendrogram_axes = {}
            
            if dendrogram:
                # For column dendrogram - next to main plot on right
                if cluster_cols:
                    col_dend_width = (COL_DENDROGRAM_SPACE * fig_dims['base_unit']) / fig_width
                    col_dend_left = main_left + main_width + (0.5 * fig_dims['base_unit'] / fig_width)
                    col_dendrogram_ax = fig.add_axes([
                        col_dend_left, 
                        main_bottom, 
                        col_dend_width, 
                        main_height
                    ])
                    dendrogram_axes['col'] = col_dendrogram_ax
                    col_dendrogram_ax.set_axis_off()
                
                # For row dendrogram - above main plot
                if cluster_rows:
                    row_dend_height = (ROW_DENDROGRAM_SPACE * fig_dims['base_unit']) / fig_height
                    row_dend_bottom = main_bottom + main_height + (0.5 * fig_dims['base_unit'] / fig_height)
                    row_dendrogram_ax = fig.add_axes([
                        main_left, 
                        row_dend_bottom, 
                        main_width, 
                        row_dend_height
                    ])
                    dendrogram_axes['row'] = row_dendrogram_ax
                    row_dendrogram_ax.set_axis_off()
            
            # Add sidebar area for legend and colorbar
            sidebar_left = main_left + main_width
            if fig_dims['has_col_dendrogram']:
                sidebar_left += (COL_DENDROGRAM_SPACE * fig_dims['base_unit'] + fig_dims['base_unit']) / fig_width
            else:
                sidebar_left += (0.5 * fig_dims['base_unit']) / fig_width
                
            sidebar_width = (LEGEND_SPACE * fig_dims['base_unit']) / fig_width
            sidebar_ax = fig.add_axes([sidebar_left, main_bottom, sidebar_width, main_height])
            sidebar_ax.set_axis_off()
            
            # Store all axes in the fig object for later reference
            fig.title_ax = title_ax
            fig.sidebar_ax = sidebar_ax
        else:
            # Use existing axes
            fig = ax.figure
            dendrogram_axes = {}

        # Handle clustering
        if cluster_rows or cluster_cols:
            # Combined data for clustering - impute NaNs for distance calculation
            # Keep original concatenation along columns for row clustering
            combined = pd.concat([cond1_means, cond2_means], axis=1)
            # Fill NaN with column means for clustering purposes only
            combined_for_clustering = combined.fillna(combined.mean())
            
            # Row clustering (genes - now on y-axis/rows after transpose)
            if cluster_rows:
                # Calculate row linkage for genes (rows of transposed data)
                row_dist = ssd.pdist(combined_for_clustering.values)
                row_linkage_matrix = linkage(row_dist, method='average')
                
                if dendrogram and 'row' in dendrogram_axes:
                    # Draw row dendrogram on the top
                    row_dendrogram = scipy_dendrogram(
                        row_linkage_matrix,
                        orientation='top',  # Changed to top orientation for the top-positioned dendrogram
                        ax=dendrogram_axes['row'],
                        color_threshold=-1,  # No color threshold
                        above_threshold_color=dendrogram_color
                    )
                    # Get the leaf order from the dendrogram
                    row_order = row_dendrogram['leaves']
                else:
                    # Just get the leaf order without drawing
                    temp_tree = scipy_dendrogram(
                        row_linkage_matrix,
                        no_plot=True
                    )
                    row_order = temp_tree['leaves']
                
                # Make sure we don't have empty cluster issue
                if len(row_order) != cond1_means.shape[0]:
                    logger.warning(f"Mismatch in row_order length ({len(row_order)}) vs. data rows ({cond1_means.shape[0]})")
                    # Adjust row_order to match the number of rows in the dataframes
                    row_order = row_order[:min(len(row_order), cond1_means.shape[0])]
                    
                # Apply the row ordering - safely handle potential indexing errors
                try:
                    cond1_means = cond1_means.iloc[row_order]
                    cond2_means = cond2_means.iloc[row_order]
                except IndexError as e:
                    logger.error(f"IndexError during row ordering: {e}")
                    # Continue without reordering if there's an error
            
            # Column clustering (groups - will be on x-axis after transpose)
            if cluster_cols:
                # Before clustering, ensure the data structure is appropriate
                n_columns = cond1_means.shape[1]
                
                # For column clustering, we need to concatenate along rows to ensure
                # consistent column dimensions regardless of which columns appear in each condition
                combined_cols = pd.concat([cond1_means, cond2_means], axis=0)
                # Fill NaN values for clustering
                combined_cols_for_clustering = combined_cols.fillna(combined_cols.mean())
                
                # Calculate column linkage properly on the transposed data
                # (since we're clustering the columns)
                col_dist = ssd.pdist(combined_cols_for_clustering.values.T)  # Transpose for column distance
                col_linkage_matrix = linkage(col_dist, method='average')
                
                if dendrogram and 'col' in dendrogram_axes:
                    # Draw column dendrogram on the right side
                    col_dendrogram = scipy_dendrogram(
                        col_linkage_matrix,
                        orientation='right',  # Changed to right for right-side positioning
                        ax=dendrogram_axes['col'],
                        color_threshold=-1,  # No color threshold
                        above_threshold_color=dendrogram_color
                    )
                    # Get the leaf order from the dendrogram
                    col_order = col_dendrogram['leaves']
                else:
                    # Just get the leaf order without drawing
                    temp_tree = scipy_dendrogram(
                        col_linkage_matrix,
                        no_plot=True
                    )
                    col_order = temp_tree['leaves']
                
                # Map leaf indices to column names
                # Get the common column names from the clustering
                columns_ordered = combined_cols.columns[col_order].tolist()
                
                # Convert column names to indices in the original data frame
                col_order_indices = []
                for col_name in columns_ordered:
                    if col_name in cond1_means.columns:
                        col_order_indices.append(list(cond1_means.columns).index(col_name))
                
                # Check if we need to fix column mismatch
                if len(col_order_indices) != n_columns:
                    logger.warning(f"Column order mismatch detected: {len(col_order_indices)} vs {n_columns} - fixing")
                    
                    # Add any missing columns at the end
                    all_cols_set = set(range(n_columns))
                    missing_cols = all_cols_set.difference(set(col_order_indices))
                    col_order_indices.extend(sorted(missing_cols))
                
                # Replace col_order with properly mapped indices
                col_order = col_order_indices
                
                # Apply the column ordering - with robust error handling
                try:
                    # Ensure all indices are within bounds
                    valid_col_order = [i for i in col_order if 0 <= i < n_columns]
                    
                    # If we ended up with fewer indices than columns, add the missing ones
                    if len(valid_col_order) < n_columns:
                        existing = set(valid_col_order)
                        missing = [i for i in range(n_columns) if i not in existing]
                        valid_col_order.extend(missing)
                    
                    # Apply the ordering
                    cond1_means = cond1_means.iloc[:, valid_col_order]
                    cond2_means = cond2_means.iloc[:, valid_col_order]
                except Exception as e:
                    logger.error(f"Error during column ordering: {str(e)}")
                    # Continue without reordering

        # Clear existing content from the axes
        ax.clear()

        # Transpose the data to have genes on y-axis and groups on x-axis
        cond1_means = cond1_means.T
        cond2_means = cond2_means.T

        # Calculate min/max for colormap
        all_data = np.concatenate(
            [
                cond1_means.values.flatten(),
                cond2_means.values.flatten()
            ]
        )
        all_data = all_data[~np.isnan(all_data)]  # Remove NaN values

        # Determine if we're using z-scoring
        is_zscored = standard_scale == "var" or standard_scale == 0 or standard_scale == "group" or standard_scale == 1

        # Set default colormap based on whether we're using z-scoring
        effective_cmap = cmap
        if effective_cmap is None:
            if is_zscored:
                effective_cmap = "coolwarm"  # Default divergent colormap for z-scored data
            else:
                effective_cmap = "viridis"   # Default sequential colormap for raw data
        
        # Determine if we should use a centered colormap
        # When z-scoring, always default to vcenter=0 
        # Otherwise, don't center unless explicitly specified
        if is_zscored and vcenter is None:
            # Default to centering at 0 for z-scored data
            effective_vcenter = 0
        else:
            # Use provided vcenter or None
            effective_vcenter = vcenter
        
        # Process percentile-based limits if specified
        def parse_percentile(value, data):
            """Convert percentile string (e.g., 'p5') to actual value from data."""
            if isinstance(value, str) and value.startswith('p'):
                try:
                    percentile = float(value[1:])
                    if 0 <= percentile <= 100:
                        return np.nanpercentile(data, percentile)
                    else:
                        logger.warning(f"Invalid percentile {percentile}, must be between 0 and 100. Using None instead.")
                        return None
                except ValueError:
                    logger.warning(f"Invalid percentile format '{value}'. Use 'p<number>' (e.g., 'p5'). Using None instead.")
                    return None
            return value
        
        # Parse percentile values if specified
        parsed_vcenter = parse_percentile(effective_vcenter, all_data)
        parsed_vmin = parse_percentile(vmin, all_data)
        parsed_vmax = parse_percentile(vmax, all_data)
        
        # For z-scored data, use symmetric limits by default unless vmin/vmax are explicitly provided
        effective_vmin = parsed_vmin
        effective_vmax = parsed_vmax
        effective_vcenter = parsed_vcenter
        
        if is_zscored and parsed_vmin is None and parsed_vmax is None:
            # Find the maximum absolute value to use for symmetric limits
            abs_max = np.max(np.abs(all_data))
            effective_vmin = -abs_max
            effective_vmax = abs_max
            logger.info(f"Using symmetric colormap limits [-{abs_max:.2f}, {abs_max:.2f}] for z-scored data")
        
        # Ensure vmin, vcenter, and vmax are in the correct order
        # For TwoSlopeNorm: vmin < vcenter < vmax must be true
        if effective_vcenter is not None:
            # If vcenter is defined, ensure vmin < vcenter < vmax
            
            # Handle cases where vmin >= vcenter
            if effective_vmin is not None and effective_vmin >= effective_vcenter:
                original_vmin = effective_vmin
                effective_vmin = effective_vcenter - 1e-6  # Set slightly below vcenter
                logger.warning(
                    f"vmin ({original_vmin:.4f}) must be less than vcenter ({effective_vcenter:.4f}). "
                    f"Setting vmin to {effective_vmin:.4f}."
                )
                
            # Handle cases where vmax <= vcenter
            if effective_vmax is not None and effective_vmax <= effective_vcenter:
                original_vmax = effective_vmax
                effective_vmax = effective_vcenter + 1e-6  # Set slightly above vcenter
                logger.warning(
                    f"vmax ({original_vmax:.4f}) must be greater than vcenter ({effective_vcenter:.4f}). "
                    f"Setting vmax to {effective_vmax:.4f}."
                )
                
        # Ensure vmin < vmax even without vcenter
        if (effective_vmin is not None and effective_vmax is not None and 
            effective_vmin >= effective_vmax):
            # Swap values if vmin >= vmax
            logger.warning(
                f"vmin ({effective_vmin:.4f}) must be less than vmax ({effective_vmax:.4f}). "
                f"Swapping values."
            )
            effective_vmin, effective_vmax = effective_vmax, effective_vmin

        # Set up colormap normalization
        norm, cmap_obj, vmin, vmax = _setup_colormap_normalization(
            all_data, effective_vcenter, effective_vmin, effective_vmax, effective_cmap
        )

        # Use the calculated tile dimensions for each cell based on the base unit
        # Scale to appropriate relative sizes within the axes
        cell_width = 1.0  # Standard width in axes units
        cell_height = fig_dims['tile_height'] / fig_dims['tile_width']  # Preserve aspect ratio
        
        # Draw each cell
        for i, gene in enumerate(cond1_means.index):
            for j, group in enumerate(cond1_means.columns):
                val1 = cond1_means.iloc[i, j]
                val2 = cond2_means.iloc[i, j]
                _draw_diagonal_split_cell(
                    ax, j, i, cell_width, cell_height, 
                    val1, val2, cmap_obj, vmin, vmax, 
                    edgecolor='none', linewidth=0, **kwargs
                )

        # Configure axis limits to show all cells
        ax.set_xlim(0, len(cond1_means.columns))
        ax.set_ylim(0, len(cond1_means.index))
        
        # Remove axis spines
        for spine in ax.spines.values():
            spine.set_visible(False)
            
        # Completely remove any ticks
        ax.tick_params(axis='both', which='both', length=0)
        ax.set_xticks([])
        ax.set_yticks([])

        # Add group labels (now on x-axis)
        if show_group_labels:
            ax.set_xticks(np.arange(len(cond1_means.columns)) + 0.5)
            ax.set_xticklabels(cond1_means.columns, rotation=90, fontsize=group_labels_size, ha='center')
            # Ensure tick labels are outside the data area
            ax.tick_params(axis='x', which='major', pad=5)

        # Add gene labels (now on y-axis)
        if show_gene_labels:
            ax.set_yticks(np.arange(len(cond1_means.index)) + 0.5)
            ax.set_yticklabels(cond1_means.index, fontsize=gene_labels_size, va='center')
            # Ensure tick labels align with the cells by adjusting padding
            ax.tick_params(axis='y', which='major', pad=5)

        # Remove the grid
        ax.grid(False)

        # Set title if provided, or create an informative default title using the dedicated title_ax
        if create_fig:  # Only if we created the figure
            # Choose title content
            if title:
                # Use provided title
                title_text = title
            elif condition1_name and condition2_name:
                # Generate default title
                title_text = f"{condition1_name} vs {condition2_name}\nMean expression by {groupby}"
            else:
                title_text = "Gene Expression Heatmap"
                
            # Add title to the dedicated title axes
            fig.title_ax.text(
                0.5, 0.5, title_text,
                fontsize=18, fontweight='bold',
                horizontalalignment='center',
                verticalalignment='center'
            )

        # Create legend and colorbar in the sidebar area
        if create_fig:  # Only if we created the figure
            # Get the sidebar axes
            sidebar_ax = fig.sidebar_ax
            
            # Create triangle patches for legend with position information
            lower_triangle = mpatches.Polygon(
                [[0, 0], [1, 0], [0, 1]],
                facecolor=cmap_obj(0.7),  # Use a specific color for legend
                label=f"{condition1_name} (lower left)"
            )
            upper_triangle = mpatches.Polygon(
                [[1, 0], [1, 1], [0, 1]],
                facecolor=cmap_obj(0.3),  # Use a different color for contrast
                label=f"{condition2_name} (upper right)"
            )

            legend_elements = [upper_triangle, lower_triangle]
            
            # Calculate legend position within sidebar - use top portion
            legend_height = 0.4  # Use top 40% of sidebar for legend
            
            # Custom handler for the triangular patches
            class HandlerTriangle(HandlerPatch):
                def create_artists(
                    self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
                ):
                    # With transposed plot, we need to update the triangles
                    if "(lower left)" in orig_handle.get_label():
                        # Lower triangle (first condition)
                        verts = [
                            [xdescent, ydescent],
                            [xdescent + width, ydescent],
                            [xdescent, ydescent + height],
                        ]
                    else:  # upper right
                        verts = [
                            [xdescent + width, ydescent],
                            [xdescent + width, ydescent + height],
                            [xdescent, ydescent + height],
                        ]
                    triangle = mpatches.Polygon(
                        verts,
                        closed=True,
                        facecolor=orig_handle.get_facecolor(),
                        edgecolor=orig_handle.get_edgecolor(),
                    )
                    triangle.set_transform(trans)
                    return [triangle]

            # Add the legend at the top of the sidebar
            # Create an axes for the legend in the top portion of the sidebar
            bbox = sidebar_ax.get_position()
            legend_ax = fig.add_axes([
                bbox.x0, 
                bbox.y0 + bbox.height * (1 - legend_height), 
                bbox.width, 
                bbox.height * legend_height
            ])
            legend_ax.set_axis_off()
            
            # Add the legend
            legend = legend_ax.legend(
                handles=legend_elements,
                loc="center",
                title="Conditions",
                frameon=False,
                handler_map={mpatches.Polygon: HandlerTriangle()},
            )
            legend.get_title().set_fontweight('bold')
            
            # Add colorbar in the lower portion of the sidebar
            colorbar_height = 0.5  # Use 50% of sidebar for colorbar
            # Create colorbar axes in the bottom portion of sidebar with a gap
            gap = 0.1  # 10% gap in the middle
            
            # Make the colorbar narrower (reduce size)
            colorbar_width = 0.25  # Reduce from 0.4 to 0.25 (smaller colorbar)
            colorbar_ax = fig.add_axes([
                bbox.x0 + bbox.width * (0.5 - colorbar_width/2),  # Center horizontally 
                bbox.y0,  # Start from bottom of sidebar
                bbox.width * colorbar_width,  # Make narrower for smaller colorbar
                bbox.height * (1 - legend_height - gap) # Fill remaining space minus gap
            ])
            # Don't turn off axis for colorbar - we need to see the ticks and labels
            
            # Create the colorbar in the new axes
            cax = colorbar_ax

        # Create the colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax)
        
        # Style the colorbar
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(axis='y', which='both', length=4, width=1, direction='out')
        cbar.ax.grid(False)
        
        # Make all spines invisible for a cleaner look
        for spine_name, spine in cbar.ax.spines.items():
            spine.set_visible(False)

        # Set colorbar label based on whether data was z-scored
        if colorbar_title is None:
            if is_zscored:
                label_text = "Z-score"
            else:
                label_text = "Expression"
        else:
            label_text = colorbar_title
            
        # Use colorbar_kwargs to override default label settings if provided
        label_kwargs = {'labelpad': 10, 'fontweight': 'bold', 'fontsize': 12}
        if colorbar_kwargs and 'label_kwargs' in colorbar_kwargs:
            label_kwargs.update(colorbar_kwargs.get('label_kwargs', {}))
            
        cbar.set_label(label_text, **label_kwargs)
        
        # Ensure ticks are visible with proper font size
        cbar.ax.tick_params(labelsize=10)
        
        # Set exactly 3 ticks on the colorbar by default
        cbar.ax.yaxis.set_major_locator(plt.MaxNLocator(3))
            
        # Ensure tick labels have proper formatting
        cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
        
        # Apply any additional colorbar formatting from colorbar_kwargs
        if colorbar_kwargs:
            # Apply any tick locator if specified
            if 'locator' in colorbar_kwargs:
                cbar.ax.yaxis.set_major_locator(colorbar_kwargs['locator'])
            
            # Apply any formatter if specified
            if 'formatter' in colorbar_kwargs:
                cbar.ax.yaxis.set_major_formatter(colorbar_kwargs['formatter'])
                
            # Apply any other colorbar properties
            for key, value in colorbar_kwargs.items():
                if key not in ['label_kwargs', 'locator', 'formatter'] and hasattr(cbar, key):
                    setattr(cbar, key, value)

        # Save figure if path provided
        if save:
            plt.savefig(save, dpi=300, bbox_inches="tight")

        # Return figure and axes if requested
        if return_fig:
            if dendrogram and len(dendrogram_axes) > 0:
                return fig, ax, dendrogram_axes
            else:
                return fig, ax
        
    else:
        # Error case - we need a groupby column for the heatmap
        logger.error("No groupby column provided. Split heatmap requires grouping.")
        return None

# No backward compatibility needed