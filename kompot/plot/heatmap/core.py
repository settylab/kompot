"""Core heatmap plotting functions."""

from collections import namedtuple
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
from ...utils import KOMPOT_COLORS
from ...anndata.utils import get_run_from_history
from .utils import (_prepare_gene_list, _get_expression_matrix, 
                   _filter_excluded_groups, _apply_scaling, _calculate_figsize, 
                   _setup_colormap_normalization)
from .visualization import _draw_diagonal_split_cell, _draw_fold_change_cell, _draw_split_dot_cell

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
    genes: Optional[Union[List[str], Sequence[str]]] = None,
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
    show_gene_labels: bool = True,
    show_group_labels: bool = True,
    gene_labels_size: int = 12,
    group_labels_size: int = 12,
    colorbar_title: Optional[str] = None,
    colorbar_kwargs: Optional[Dict[str, Any]] = None,
    n_colorbar_ticks: Optional[int] = 3,  # Control number of ticks in the colorbar
    layout_config: Optional[Dict[str, float]] = None,  # Configuration for layout spacing
    title: Optional[str] = None,
    sort_genes: bool = True,
    vcenter: Optional[Union[float, str]] = None,
    vmin: Optional[Union[float, str]] = None,
    vmax: Optional[Union[float, str]] = None,
    ax: Optional[plt.Axes] = None,
    draw_values: bool = False,
    return_fig: bool = False,
    return_data: bool = False,
    save: Optional[str] = None,
    run_id: Optional[int] = None,
    condition_column: Optional[str] = None,
    observed: bool = True,
    condition1: Optional[str] = None,
    condition2: Optional[str] = None,
    condition1_name: Optional[str] = None,
    condition2_name: Optional[str] = None,
    exclude_groups: Optional[Union[str, List[str]]] = None,
    fold_change_mode: bool = False,  # Whether to use fold change coloring instead of split tiles
    split_dot_mode: bool = False,  # Whether to use split dots instead of split tiles
    max_cell_count: Optional[int] = None,  # Upper limit for cell count used for dot sizing (None = use actual max)
    **kwargs,
) -> Union[
    None, Tuple[plt.Figure, plt.Axes], Tuple[plt.Figure, plt.Axes, Dict[str, plt.Axes]]
]:
    """
    Create a heatmap visualizing gene expression data for two conditions.
    
    By default, the heatmap displays expression values with diagonally split cells, where the lower-left
    triangle shows values for the first condition and the upper-right triangle shows values
    for the second condition. This creates a compact visualization that highlights
    differences between conditions.
    
    When fold_change_mode=True, each cell is a single square colored by the fold change
    (difference between means) between the two conditions, providing a simpler visualization
    focused on the differential expression.
    
    When split_dot_mode=True, the heatmap displays dots split in half vertically, where the
    left half shows values for the first condition and the right half shows values for the
    second condition. The size of each half-dot is determined by the number of cells in that 
    condition for that group, creating a visualization that highlights both expression differences
    and relative group sizes simultaneously.
    
    Genes are shown on the y-axis and groups (cell types, clusters, etc.) are shown
    on the x-axis, with a legend and colorbar positioned to the right of the plot.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing expression data
    var_names : list, optional
        List of genes to include in the heatmap. If None, will use top genes
        based on score_key.
    groupby : str, optional
        Key in adata.obs for grouping cells
    n_top_genes : int, optional
        Number of top genes to include if var_names is None
    genes : list, optional
        Alternative parameter name for specifying genes to include.
        Takes precedence over var_names if provided.
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
        z-scoring is applied, "Reds" in split dot mode, and "viridis" (sequential) otherwise.
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
        tile_aspect_ratio < 1.cell
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
    n_colorbar_ticks : int, optional
        Number of ticks to display in the colorbar. Default is 3. This parameter provides
        a simple way to control tick density, while the colorbar_kwargs['locator'] option
        provides more fine-grained control if needed.
    layout_config : dict, optional
        Configuration for controlling plot layout spacing. Keys include:
        - 'gene_label_space': Space for gene labels (y-axis), default 3.5
        - 'group_label_space': Space for group labels (x-axis), default 2.0
        - 'title_space': Space for title, default 3.0
        - 'base_legend_space': Base space for legend, default 4.0
        - 'legend_name_factor': Factor to adjust legend space based on condition name length, default 0.15
        - 'colorbar_space': Space for colorbar, default 3.0
        - 'row_dendrogram_space': Space for row dendrogram, default 2.5
        - 'col_dendrogram_space': Space for column dendrogram, default 2.5
        - 'legend_fontsize': Base font size for legend, default 12
        - 'legend_fontsize_factor': Factor to reduce font size for long condition names, default 0.25
        - 'colorbar_height': Height proportion of sidebar for colorbar, default 0.5
        - 'colorbar_width': Width proportion for colorbar, default 0.25
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
    draw_values : bool
        Whether to draw the values in the heatmap cells. Default is False.
    return_fig : bool, optional
        If True, returns the figure and axes
    return_data : bool, optional
        If True, returns the expression means and fold-changes used for the heatmap
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
    condition1, condition2 : str, optional
        Names of the two conditions to compare. If None, tries to infer from run_info.
        These must match the values in the condition_column in adata.obs.
    condition1_name, condition2_name : str, optional
        Display names for the two conditions in the plot legend and title.
        If None, defaults to the values of condition1 and condition2.
    exclude_groups : str or list, optional
        Group name(s) to exclude from the heatmap.
    fold_change_mode : bool, optional
        Whether to use fold change coloring instead of split tiles
    split_dot_mode : bool, optional
        Whether to use split dots instead of split tiles. When True, the size of each half-dot
        represents the number of cells in that condition for that group
    max_cell_count : int, optional
        Upper limit for cell count used for dot sizing. If provided, all dots will be scaled
        relative to this maximum value, even if actual cell counts exceed it. This helps maintain
        readable visualization when some groups have much larger cell counts than others.
    **kwargs : 
        Additional keyword arguments passed to matplotlib

    Returns
    -------
    If return_fig is True and dendrogram is False, returns (fig, ax)
    If return_fig is True and dendrogram is True, returns (fig, ax, dendrogram_axes)
    """
    # Normalize run_id to use -1 (latest run) if None
    effective_run_id = -1 if run_id is None else run_id
    
    # If genes parameter is provided, use it (with priority over var_names)
    if genes is not None:
        var_names = genes
        
    # Prepare gene list and get run info
    var_names, score_key, run_info = _prepare_gene_list(
        adata=adata,
        var_names=var_names,
        n_top_genes=n_top_genes,
        score_key=score_key,
        sort_genes=sort_genes,
        run_id=effective_run_id,
    )

    if run_info is not None and "params" in run_info:
        params = run_info["params"]
        if condition_column is None:
            condition_column = params["groupby"]
            logger.info(f"Inferred condition_column='{condition_column}' from run information")
        if condition1 is None or condition2 is None:
            # First check if condition1 and condition2 are directly in params
            if "condition1" in params and "condition2" in params:
                if condition1 is None:
                    condition1 = params["condition1"]
                    logger.info(f"Inferred condition1='{condition1}' from run information")
                if condition2 is None:
                    condition2 = params["condition2"]
                    logger.info(f"Inferred condition2='{condition2}' from run information")
            # Fallback to conditions list if available
            elif "conditions" in params and len(params["conditions"]) == 2:
                if condition1 is None:
                    condition1 = params["conditions"][0]
                    logger.info(f"Inferred condition1='{condition1}' from run information")
                if condition2 is None:
                    condition2 = params["conditions"][1]
                    logger.info(f"Inferred condition2='{condition2}' from run information")
        if layer is None and "layer" in params:
            layer = params["layer"]
            logger.info(f"Inferred layer='{layer}' from run information")
                
    # Handle display names for conditions
    # If display names weren't provided, use the actual condition values
    if condition1_name is None and condition1 is not None:
        condition1_name = condition1
    if condition2_name is None and condition2 is not None:
        condition2_name = condition2
    

    # Log the plot type
    if fold_change_mode:
        logger.info(f"Creating fold change heatmap with {len(var_names)} genes/features")
    elif split_dot_mode:
        logger.info(f"Creating split dot heatmap with {len(var_names)} genes/features")
    else:
        logger.info(f"Creating split heatmap with {len(var_names)} genes/features")

    # Validate condition column
    if condition_column is None:
        logger.warning("No condition_column could be inferred. Split heatmap requires a condition column.")
        return None

    # Check that condition column exists
    if condition_column not in adata.obs.columns:
        logger.error(f"Condition column '{condition_column}' not found in adata.obs")
        return None

    unique_values = adata.obs[condition_column].unique()

    # Check for presence of both conditions in data
    if condition1 is None:
        logger.error(f"No value for condition1 could be inferred. Please provide using condition1 parameter.")
        return None
    if condition2 is None:
        logger.error(f"No value for condition2 could be inferred. Please provide using condition2 parameter.")
        return None

    if groupby is None:
        logger.error("No groupby column provided. Split heatmap requires grouping.")
        return None

    if condition1 not in unique_values:
        logger.error(
            f"Condition '{condition1}' not found in {condition_column}. "
            f"Available values are: {', '.join(map(str, unique_values))}"
        )
        return None
    if condition2 not in unique_values:
        logger.error(
            f"Condition '{condition2}' not found in {condition_column}. "
            f"Available values are: {', '.join(map(str, unique_values))}"
        )
        return None
    
    # Log display names if they differ from the condition values
    if condition1 != condition1_name or condition2 != condition2_name:
        logger.info(
            f"Using display names: '{condition1_name}' for condition1, "
            f"'{condition2_name}' for condition2"
        )
    

    # Get expression data
    expr_matrix = _get_expression_matrix(adata, var_names, layer)

    # Create dataframe with expression, condition column, and groupby column if any
    expr_df = pd.DataFrame(
        expr_matrix,
        index=adata.obs_names,
        columns=var_names
    )
    expr_df[condition_column] = adata.obs[condition_column].values

    if groupby not in adata.obs.columns:
        logger.error(f"Groupby column '{groupby}' not found in adata.obs")
        return None
    expr_df[groupby] = adata.obs[groupby].values

    # Split by condition
    cond1_df = expr_df[expr_df[condition_column] == condition1].drop(columns=[condition_column])
    cond2_df = expr_df[expr_df[condition_column] == condition2].drop(columns=[condition_column])
        
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
    
    # For split_dot_mode, also calculate cell counts for each group
    if split_dot_mode:
        cell_counts1 = (
            cond1_df.groupby(groupby, observed=observed)
            .size()
            .reindex(all_groups)
            .fillna(0)
        )
        cell_counts2 = (
            cond2_df.groupby(groupby, observed=observed)
            .size()
            .reindex(all_groups)
            .fillna(0)
        )
        
        # Handle numeric conversion in case cell_counts are strings
        # Convert cell_counts Series to numeric values
        cell_counts1 = pd.to_numeric(cell_counts1, errors='coerce').fillna(0)
        cell_counts2 = pd.to_numeric(cell_counts2, errors='coerce').fillna(0)
        
        # Calculate the global maximum count across all groups and conditions
        actual_max_count = max(max(cell_counts1), max(cell_counts2))
        
        # Use user-specified max if provided, otherwise use actual max
        if max_cell_count is not None and max_cell_count > 0:
            global_max_count = max_cell_count
            logger.info(f"Using user-specified max count limit of {global_max_count} cells to scale all dots (actual max: {int(actual_max_count)})")
        else:
            global_max_count = actual_max_count
            logger.info(f"Using global max count of {int(global_max_count)} cells to scale all dots")

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
        # Set up default layout config
        default_layout = {
            'gene_label_space': 3.5,       # Space for gene labels (y-axis)
            'group_label_space': 2.0,      # Space for group labels (x-axis)
            'title_space': 3.0,            # Space for title
            'base_legend_space': 4.0,      # Base space for legend
            'legend_name_factor': 0.15,    # Factor for legend space adjustment per condition name length
            'colorbar_space': 3.0,         # Space for colorbar
            'row_dendrogram_space': 2.5,   # Space for row dendrogram
            'col_dendrogram_space': 2.5,   # Space for column dendrogram
            'colorbar_height': 0.5,        # Height proportion for colorbar
            'colorbar_width': 0.25,        # Width proportion for colorbar
            'legend_fontsize': 12,         # Base legend font size
            'legend_fontsize_factor': 0.25 # Factor to reduce font size for long names
        }
        
        # Update with user-provided config if any
        if layout_config:
            default_layout.update(layout_config)
            
        # Store the layout for later use
        layout = default_layout
        
        # Define constants for fixed spacing in multiples of the base unit
        GENE_LABEL_SPACE = layout['gene_label_space']      # Space for gene labels (y-axis)
        GROUP_LABEL_SPACE = layout['group_label_space']    # Space for group labels (x-axis)
        TITLE_SPACE = layout['title_space']                # Space for title
        
        # Dynamically determine legend space based on condition name lengths
        max_condition_name_length = max(len(str(condition1_name or "")), len(str(condition2_name or "")))
        
        # Calculate additional space needed for the legend based on name length
        additional_space = (max_condition_name_length + 5) * layout['legend_name_factor']
            
        LEGEND_SPACE = layout['base_legend_space'] + additional_space
        
        COLORBAR_SPACE = layout['colorbar_space']           # Space for colorbar
        ROW_DENDROGRAM_SPACE = layout['row_dendrogram_space'] # Space for row dendrogram
        COL_DENDROGRAM_SPACE = layout['col_dendrogram_space'] # Space for column dendrogram
        
        # Store these values in the layout for later use
        layout['legend_space'] = LEGEND_SPACE
        
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
        if dendrogram and cluster_rows:
            right_area_inches += COL_DENDROGRAM_SPACE * base_unit
        
        # Top area - for title and row dendrogram
        top_area_inches = TITLE_SPACE * base_unit
        if dendrogram and cluster_cols:
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
            
            # Store flag for dendrogram presence based on which parameters control which dendrograms
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
        
        # Set up default layout config if not already defined
        if not layout_config:
            # Create a new layout config with default values
            default_layout = {
                'gene_label_space': 3.5,       # Space for gene labels (y-axis)
                'group_label_space': 2.0,      # Space for group labels (x-axis)
                'title_space': 3.0,            # Space for title
                'base_legend_space': 4.0,      # Base space for legend
                'legend_name_factor': 0.15,    # Factor for legend space adjustment
                'colorbar_space': 3.0,         # Space for colorbar
                'row_dendrogram_space': 2.5,   # Space for row dendrogram
                'col_dendrogram_space': 2.5,   # Space for column dendrogram
                'colorbar_height': 0.5,        # Height proportion for colorbar
                'colorbar_width': 0.25,        # Width proportion for colorbar
                'legend_fontsize': 12,         # Base legend font size
                'legend_fontsize_factor': 0.25 # Factor to reduce font size for long names
            }
            # Store the layout for later use
            layout = default_layout
        
        # Calculate dynamic legend space similar to non-fixed figsize case
        max_condition_name_length = max(len(str(condition1_name or "")), len(str(condition2_name or "")))
        legend_factor = 1.0 + max(0, (max_condition_name_length * layout['legend_name_factor']))
        
        # Approximate reasonable areas based on base_unit
        left_area_inches = min(layout['gene_label_space'] * base_unit, width_inches * 0.2)
        right_area_inches = min(layout['base_legend_space'] * base_unit * legend_factor, width_inches * 0.3)
        bottom_area_inches = min(layout['group_label_space'] * base_unit, height_inches * 0.15)
        top_area_inches = min(layout['title_space'] * base_unit, height_inches * 0.15)
        if dendrogram:
            if cluster_cols:
                top_area_inches += min(layout['row_dendrogram_space'] * base_unit, height_inches * 0.1)
            if cluster_rows:
                right_area_inches += min(layout['col_dendrogram_space'] * base_unit, width_inches * 0.1)
        
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
            # This is for clustering rows (genes) after transpose - controlled by cluster_rows
            if cluster_rows:
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
            # This is for clustering columns (groups) after transpose - controlled by cluster_cols
            if cluster_cols:
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
        if fig_dims['has_row_dendrogram']:
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
        
        # Column clustering - but using cluster_rows parameter 
        if cluster_cols:
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
        
        # Row clustering - but using cluster_cols parameter
        if cluster_rows:
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

    # Calculate fold changes before any scaling if in fold_change_mode
    fold_changes = cond2_means - cond1_means
    fold_changes = fold_changes.T
    fold_changes = fold_changes.iloc[::-1]


    # Clear existing content from the axes
    ax.clear()

    # Transpose the data to have genes on y-axis and groups on x-axis
    cond1_means = cond1_means.T
    cond2_means = cond2_means.T
    
    # Reverse the row order so genes appear in the correct order when plotted 
    # (since matplotlib plots from bottom to top on the y-axis)
    cond1_means = cond1_means.iloc[::-1]
    cond2_means = cond2_means.iloc[::-1]

    # Calculate min/max for colormap
    if fold_change_mode:
        # For fold change mode, use the fold change values for colormap limits
        all_data = fold_changes.values.flatten()
    else:
        # For split mode, use all expression values
        all_data = np.concatenate(
            [
                cond1_means.values.flatten(),
                cond2_means.values.flatten()
            ]
        )
    all_data = all_data[~np.isnan(all_data)]  # Remove NaN values

    # Determine if we're using z-scoring (but not in fold_change_mode where z-scoring doesn't make sense)
    is_zscored = not fold_change_mode and (standard_scale == "var" or standard_scale == 0 or standard_scale == "group" or standard_scale == 1)
    
    # If in fold_change_mode and standard_scale is specified, warn that it will be ignored
    if fold_change_mode and standard_scale is not None:
        logger.warning("standard_scale is ignored in fold_change_mode as z-scoring is not appropriate for fold changes")

    # Set default colormap based on mode and whether we're using z-scoring
    effective_cmap = cmap
    if effective_cmap is None:
        if fold_change_mode or is_zscored:
            effective_cmap = "coolwarm"  # Default divergent colormap for fold change or z-scored data
        elif split_dot_mode:
            effective_cmap = "Reds" 
        else:
            effective_cmap = "viridis"   # Default sequential colormap for raw data
    
    # Determine if we should use a centered colormap
    # When fold_change_mode or z-scoring, always default to vcenter=0 
    # Otherwise, don't center unless explicitly specified
    if (fold_change_mode or is_zscored) and vcenter is None:
        # Default to centering at 0 for fold change or z-scored data
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
    
    # For z-scored data or fold_change_mode, use symmetric limits by default unless vmin/vmax are explicitly provided
    effective_vmin = parsed_vmin
    effective_vmax = parsed_vmax
    effective_vcenter = parsed_vcenter
    
    if (is_zscored or fold_change_mode) and parsed_vmin is None and parsed_vmax is None:
        # Find the maximum absolute value to use for symmetric limits
        if len(all_data) > 0:
            abs_max = np.max(np.abs(all_data))
            effective_vmin = -abs_max
            effective_vmax = abs_max
            if fold_change_mode:
                logger.info(f"Using symmetric colormap limits [-{abs_max:.2f}, {abs_max:.2f}] for fold change data")
            else:
                logger.info(f"Using symmetric colormap limits [-{abs_max:.2f}, {abs_max:.2f}] for z-scored data")
        else:
            # If there's no data, use default limits
            effective_vmin = -1.0
            effective_vmax = 1.0
            logger.info("No valid data found, using default colormap limits [-1.0, 1.0]")
    
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
    cell_height = 1.0
    
    # Draw each cell
    for i, gene in enumerate(cond1_means.index):
        for j, group in enumerate(cond1_means.columns):
            val1 = cond1_means.iloc[i, j]
            val2 = cond2_means.iloc[i, j]
            if fold_change_mode:
                # For fold change mode, use the pre-computed fold change value
                fc_val = fold_changes.iloc[i, j]
                _draw_fold_change_cell(
                    ax, j, i, cell_width, cell_height, 
                    fc_val, cmap_obj, vmin, vmax, 
                    edgecolor='none', linewidth=0,
                    draw_values=draw_values, norm=norm, **kwargs
                )
            elif split_dot_mode:
                # For split dot mode, get the cell counts for this group
                count1 = cell_counts1[group] if 'cell_counts1' in locals() else None
                count2 = cell_counts2[group] if 'cell_counts2' in locals() else None
                
                # Draw the split dot - pass the global max count for consistent scaling
                _draw_split_dot_cell(
                    ax, j, i, cell_width, cell_height,
                    val1, val2, cmap_obj, vmin, vmax,
                    cell_count1=count1, cell_count2=count2,
                    global_max_count=global_max_count,
                    edgecolor='none', linewidth=0,
                    draw_values=draw_values, norm=norm, **kwargs
                )
            else:
                _draw_diagonal_split_cell(
                    ax, j, i, cell_width, cell_height, 
                    val1, val2, cmap_obj, vmin, vmax, 
                    edgecolor='none', linewidth=0,
                    draw_values=draw_values, norm=norm, **kwargs
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

    # Add gene labels if requested (now on y-axis)
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
            title_text = f"{condition1_name} to {condition2_name}\nMean expression by {groupby}"
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
        
        # Create legend elements based on mode
        if fold_change_mode:
            # For fold change mode, use rectangles to show the fold change direction
            positive_change = mpatches.Rectangle(
                (0, 0), 1, 1, 
                facecolor=cmap_obj(0.7),  # Use a specific color for legend (higher values)
                label=f"Higher in {condition2_name}"
            )
            negative_change = mpatches.Rectangle(
                (0, 0), 1, 1, 
                facecolor=cmap_obj(0.3),  # Use a different color for contrast (lower values)
                label=f"Higher in {condition1_name}"
            )
            neutral_change = mpatches.Rectangle(
                (0, 0), 1, 1, 
                facecolor=cmap_obj(0.5),  # Color for no change (center of colormap)
                label="No change"
            )
            legend_elements = [positive_change, neutral_change, negative_change]
        elif split_dot_mode:
            # For split dot mode, use split dot in the legend
            left_half = mpatches.Wedge(
                (0.5, 0.5), 0.5, 90, 270,
                facecolor=cmap_obj(0.7),
                label=f"{condition1_name} (left half)"
            )
            right_half = mpatches.Wedge(
                (0.5, 0.5), 0.5, 270, 90,
                facecolor=cmap_obj(0.3),
                label=f"{condition2_name} (right half)"
            )
            legend_elements = [right_half, left_half]
            
            # Calculate example counts for the legend - small, medium, large
            # Find the actual range of cell counts in the data
            all_counts = []
            if 'cell_counts1' in locals():
                all_counts.extend(cell_counts1.values)
            if 'cell_counts2' in locals():
                all_counts.extend(cell_counts2.values)
            
            # Filter out zeros and get the min, median, and max counts
            non_zero_counts = [c for c in all_counts if c > 0]
            
            if not non_zero_counts:
                # Default values if no real data available
                small_count, medium_count, large_count = 10, 100, 1000
            else:
                # Use global max count (which might be user-specified) as the upper bound
                max_count = global_max_count if 'global_max_count' in locals() else max(non_zero_counts)
                
                # Round up to a nice number for the legend
                # Round to the nearest power of 10 multiplied by 1, 2, or 5
                magnitude = 10 ** np.floor(np.log10(max_count))
                if max_count / magnitude <= 1.5:
                    large_count = magnitude
                elif max_count / magnitude <= 3.5:
                    large_count = 2 * magnitude
                elif max_count / magnitude <= 7.5:
                    large_count = 5 * magnitude
                else:
                    large_count = 10 * magnitude
                    
                # Create medium and small counts that are evenly spaced on a log scale
                if large_count >= 1000:
                    medium_count = large_count / 10
                    small_count = large_count / 100
                elif large_count >= 100:
                    medium_count = large_count / 5
                    small_count = large_count / 25
                else:
                    medium_count = large_count / 3
                    small_count = large_count / 10
                
                # Make sure all counts are integers
                large_count = int(large_count)
                medium_count = int(medium_count)
                small_count = max(1, int(small_count))
            
            # Format count labels with commas and add "+" if max_cell_count is specified
            if max_cell_count is not None and actual_max_count > max_cell_count and large_count >= max_cell_count:
                large_label = f"{large_count:,}+ cells"
            else:
                large_label = f"{large_count:,} cells"
                
            medium_label = f"{medium_count:,} cells"
            small_label = f"{small_count:,} cells"
            
            # Store important information about dot sizes in a global dictionary
            # that can be accessed from the SizeTextHandler
            import builtins
            if not hasattr(builtins, 'kompot_legend_dot_info'):
                builtins.kompot_legend_dot_info = {}
            
            # Store the actual counts and the cell_width/height
            kompot_legend_dot_info = {
                'small_count': small_count,
                'medium_count': medium_count,
                'large_count': large_count,
                'cell_width': cell_width,
                'cell_height': cell_height,
                'max_size_factor': 0.9,  # Same as in _draw_split_dot_cell
                'global_max_count': global_max_count  # Add global max count for legend
            }
            builtins.kompot_legend_dot_info = kompot_legend_dot_info
            
            # Create size examples for legend
            small_example = mpatches.Rectangle(
                (0, 0), 1, 1, 
                fill=False, edgecolor='none',
                label=small_label
            )
            medium_example = mpatches.Rectangle(
                (0, 0), 1, 1, 
                fill=False, edgecolor='none',
                label=medium_label
            )
            large_example = mpatches.Rectangle(
                (0, 0), 1, 1, 
                fill=False, edgecolor='none',
                label=large_label
            )
            
            # We'll set the cell dimensions later after the SizeTextHandler class is defined
            
            # Create section title without adding it to legend elements
            # We'll add a title directly to the legend instead
            legend_elements.append(small_example)
            legend_elements.append(medium_example)
            legend_elements.append(large_example)
        else:
            # For split mode, use triangles as before
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
        
        # Calculate legend position within sidebar using layout config
        # Use more space for split_dot_mode because it has more legend items
        if split_dot_mode:
            # Use more height for the split_dot legend
            legend_height = 1.0 - (layout['colorbar_height'] if layout else 0.3) * 0.7
        else:
            legend_height = 1.0 - (layout['colorbar_height'] if layout else 0.4)  # Use top portion for legend

        # Custom handler for the triangular patches
        class HandlerTriangle(HandlerPatch):
            def create_artists(
                self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
            ):
                # Enforce square shape
                size = min(width, height)
                x0 = xdescent + (width - size) / 2
                y0 = ydescent + (height - size) / 2
        
                if "(lower left)" in orig_handle.get_label():
                    verts = [
                        [x0, y0],
                        [x0 + size, y0],
                        [x0, y0 + size],
                    ]
                else:  # upper right
                    verts = [
                        [x0 + size, y0],
                        [x0 + size, y0 + size],
                        [x0, y0 + size],
                    ]
        
                triangle = mpatches.Polygon(
                    verts,
                    closed=True,
                    facecolor=orig_handle.get_facecolor(),
                    edgecolor=orig_handle.get_edgecolor(),
                )
                triangle.set_transform(trans)
                return [triangle]
                
        # Custom handler for wedges (half-circle elements)
        class HandlerWedge(HandlerPatch):
            def create_artists(
                self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
            ):
                # Enforce square shape for the circle
                size = min(width, height)
                center_x = xdescent + width / 2
                center_y = ydescent + height / 2
                radius = size / 2
                
                # Create the appropriate half-circle based on the label
                if "(left half)" in orig_handle.get_label():
                    wedge = mpatches.Wedge(
                        (center_x, center_y), radius, 90, 270, 
                        facecolor=orig_handle.get_facecolor(),
                        edgecolor=orig_handle.get_edgecolor()
                    )
                else:  # right half
                    wedge = mpatches.Wedge(
                        (center_x, center_y), radius, 270, 90, 
                        facecolor=orig_handle.get_facecolor(),
                        edgecolor=orig_handle.get_edgecolor()
                    )
                
                wedge.set_transform(trans)
                return [wedge]

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
        
        # Add the legend with adaptive font size
        # Calculate an appropriate font size based on legend space and condition name length
        max_condition_name_length = max(len(str(condition1_name or "")), len(str(condition2_name or "")))
        
        # For split_dot_mode, also consider the length of the cell count labels
        if split_dot_mode:
            # Get the length of the longest cell count label
            if 'large_label' in locals():
                max_cell_label_length = max(
                    len(small_label), 
                    len(medium_label), 
                    len(large_label)
                )
                # Use the longer of the two for font size calculation
                max_text_length = max(max_condition_name_length, max_cell_label_length)
            else:
                max_text_length = max_condition_name_length
        else:
            max_text_length = max_condition_name_length
            
        # Base font size with reduction for very long text
        base_fontsize = layout['legend_fontsize'] if layout else 12
        fontsize_factor = layout['legend_fontsize_factor'] if layout else 0.25
        adaptive_fontsize = max(8, base_fontsize - max(0, max_text_length * fontsize_factor))
        
        # Create the legend with appropriate handler map based on mode
        if fold_change_mode:
            legend = legend_ax.legend(
                handles=legend_elements,
                loc="center",
                title="Fold Change",
                frameon=False,
                prop={'size': adaptive_fontsize},  # Use adaptive font size
                title_fontsize=adaptive_fontsize + 1,  # Make title slightly larger
            )
        elif split_dot_mode:
            # Create a special formatter for dot sizes in the legend
            from matplotlib.legend_handler import HandlerBase
            
            class SizeTextHandler(HandlerBase):
                """Custom handler to create dot icons of different sizes for the legend"""
                
                @classmethod
                def calculate_scale_factor(cls, fontsize):
                    """Calculate an appropriate scale factor based on the fontsize"""
                    return fontsize * 2
                
                def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
                    # Access the global dot information
                    import builtins
                    if not hasattr(builtins, 'kompot_legend_dot_info'):
                        builtins.kompot_legend_dot_info = {}
                    dot_info = builtins.kompot_legend_dot_info
                    
                    # Extract the count from the label
                    label = orig_handle.get_label()
                    
                    # Extract the actual count from the label (e.g., "100 cells" or "500+ cells" -> 100 or 500)
                    count_str = label.split(" ")[0].replace(",", "").replace("+", "")
                    count = int(count_str)
                    
                    # Use the exact same algorithm as the plot function
                    # This ensures exact proportion matching
                    
                    # Get information needed for size calculation
                    cell_width = dot_info.get('cell_width', 1.0)
                    cell_height = dot_info.get('cell_height', 1.0)
                    max_size_factor = dot_info.get('max_size_factor', 0.9)
                    
                    # Get the large, medium, small counts and global max count
                    large_count = dot_info.get('large_count', max(count, 1000))
                    global_max_count = dot_info.get('global_max_count', large_count)
                    
                    # Perform the exact same calculation as in _draw_split_dot_cell
                    # Step 1: Determine max radius based on tile dimensions (same as plot function)
                    max_radius = min(cell_width, cell_height) * max_size_factor / 2
                    
                    # Step 2: Calculate scale factor based on the global max count (same as plot function)
                    # Use global max if available, otherwise fallback to large_count
                    scale_factor = max_radius / np.sqrt(global_max_count if global_max_count is not None else large_count)
                    
                    # Step 3: Calculate dot radius based on count (same as plot function)
                    plot_dot_radius = np.sqrt(count) * scale_factor
                    
                    # Calculate the scale factor based on fontsize for consistent sizing
                    self.plot_to_legend_scale = self.calculate_scale_factor(fontsize)
                    
                    # Now scale to legend coordinates with the dynamic scale factor
                    # This ensures dots are correctly proportioned relative to each other
                    # and sized appropriately for the current font size
                    legend_dot_radius = plot_dot_radius * self.plot_to_legend_scale
                    
                    # Create a dot - do NOT limit the size
                    dot = mpatches.Circle(
                        (xdescent + width/2, ydescent + height/2),
                        radius=legend_dot_radius,
                        facecolor='#AAAAAA',  # Light gray as before
                        edgecolor='#555555',  # Medium gray border
                        linewidth=0.5,
                        alpha=0.8,  # Slightly transparent as before
                        transform=trans
                    )
                    return [dot]
            
            # Create the handler map, specifying how to render each type of legend item
            handler_map = {
                mpatches.Wedge: HandlerWedge(),  # For condition wedges
            }
            
            # Add custom handlers for size examples
            for item in legend_elements:
                if any(label in item.get_label() for label in [small_label, medium_label, large_label]):
                    handler_map[item] = SizeTextHandler()
            
            # For split_dot_mode, create separate legends for conditions and dot sizes
            if split_dot_mode:
                # Separate legend elements for conditions and dot sizes
                condition_elements = [right_half, left_half]
                dot_size_elements = [
                    small_example,
                    medium_example,
                    large_example
                ]
                
                # Create handler maps for each legend
                condition_handler_map = {
                    mpatches.Wedge: HandlerWedge()
                }
                
                dot_size_handler_map = {}
                for item in dot_size_elements:
                    if any(label in item.get_label() for label in [small_label, medium_label, large_label]):
                        dot_size_handler_map[item] = SizeTextHandler()
                
                # Split sidebar into three parts vertically:
                # 1. Top: Conditions legend (25%)
                # 2. Middle: Dot size legend (45%)
                # 3. Bottom: Colorbar (30%) - This will be handled later in the existing code
                
                # Define proportions of the sidebar (excluding colorbar area)
                conditions_height_proportion = 0.25
                dot_sizes_height_proportion = 0.45  # Original proportion
                
                # Get the sidebar position - this is the full area excluding the colorbar
                bbox = sidebar_ax.get_position()
                
                # Calculate heights and positions
                # The colorbar will use the bottom portion as defined elsewhere in the code
                colorbar_height = layout['colorbar_height'] if layout else 0.3
                legend_area_height = bbox.height * (1 - colorbar_height - 0.05)  # 5% buffer
                
                conditions_height = legend_area_height * conditions_height_proportion
                dot_sizes_height = legend_area_height * dot_sizes_height_proportion
                
                # Create conditions legend at the top portion
                conditions_ax = fig.add_axes([
                    bbox.x0, 
                    bbox.y0 + bbox.height - conditions_height, 
                    bbox.width, 
                    conditions_height
                ])
                conditions_ax.set_axis_off()
                
                conditions_legend = conditions_ax.legend(
                    handles=condition_elements,
                    loc="center",
                    title="Conditions",
                    frameon=False,
                    prop={'size': adaptive_fontsize},
                    title_fontsize=adaptive_fontsize + 1,
                    handler_map=condition_handler_map
                )
                conditions_legend.get_title().set_fontweight('bold')
                
                # Create dot sizes legend in the middle portion with appropriate spacing
                dot_sizes_ax = fig.add_axes([
                    bbox.x0, 
                    bbox.y0 + bbox.height - conditions_height - dot_sizes_height, 
                    bbox.width, 
                    dot_sizes_height
                ])
                dot_sizes_ax.set_axis_off()
                
                # Calculate appropriate spacing based on largest dot size
                # First, calculate what the largest dot radius would be in the legend
                
                # Use same calculation as in SizeTextHandler
                max_radius = min(cell_width, cell_height) * kompot_legend_dot_info['max_size_factor'] / 2
                global_max = kompot_legend_dot_info.get('global_max_count', large_count)
                scale_factor = max_radius / np.sqrt(global_max)
                
                # Calculate dynamic scale factor based on the current font size
                dynamic_scale = SizeTextHandler.calculate_scale_factor(adaptive_fontsize)
                largest_dot_radius = np.sqrt(large_count) * scale_factor * dynamic_scale
                
                # Calculate appropriate spacing based on the largest dot's size and the font size
                # This ensures dots don't overlap while keeping proportional to text size
                dynamic_spacing = (largest_dot_radius * .5 / adaptive_fontsize) + 1.0
                
                # Create dot size legend with dynamically calculated spacing
                dot_sizes_legend = dot_sizes_ax.legend(
                    handles=dot_size_elements,
                    loc="center",
                    frameon=False,
                    title="Cell Count",  # Proper title for dot size legend
                    prop={'size': adaptive_fontsize},
                    title_fontsize=adaptive_fontsize + 1,
                    handler_map=dot_size_handler_map,
                    labelspacing=max(1.5, dynamic_spacing)  # Use dynamic spacing with a minimum value
                )
                dot_sizes_legend.get_title().set_fontweight('bold')
                
                # Store main legend (the sidebar_ax variable is used later in the code)
                # The colorbar code will run normally with the remaining space
                legend = conditions_legend
                
                # Set sidebar_ax (base legend area) invisible since we've replaced it
                sidebar_ax.set_visible(False)
            else:
                # Standard legend for other modes
                legend = legend_ax.legend(
                    handles=legend_elements,
                    loc="center",
                    title="Legend",
                    frameon=False,
                    prop={'size': adaptive_fontsize},
                    title_fontsize=adaptive_fontsize + 1,
                    handler_map=handler_map,
                    ncol=1
                )
        else:
            legend = legend_ax.legend(
                handles=legend_elements,
                loc="center",
                title="Conditions",
                frameon=False,
                prop={'size': adaptive_fontsize},  # Use adaptive font size
                title_fontsize=adaptive_fontsize + 1,  # Make title slightly larger
                handler_map={mpatches.Polygon: HandlerTriangle()},
            )
        legend.get_title().set_fontweight('bold')
        
        # Add colorbar in the lower portion of the sidebar
        colorbar_height = layout['colorbar_height'] if layout else 0.5  # Use configured height for sidebar
        # Create colorbar axes in the bottom portion of sidebar with a gap
        gap = 0.1  # 10% gap in the middle
        
        # Make the colorbar narrower (reduce size)
        colorbar_width = layout['colorbar_width'] if layout else 0.25  # Width from layout config
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

    # Set colorbar label based on mode and whether data was z-scored
    if colorbar_title is None:
        if fold_change_mode:
            label_text = "Log-Fold Change"
        elif is_zscored:
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
    
    # Use custom number of ticks if provided, otherwise use default of 3
    if 'locator' not in (colorbar_kwargs or {}):
        cbar.ax.yaxis.set_major_locator(plt.MaxNLocator(n_colorbar_ticks))
        
    # Ensure tick labels have proper formatting
    cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    
    # Apply any additional colorbar formatting from colorbar_kwargs
    if colorbar_kwargs:
        # Apply any tick locator if specified (overrides n_colorbar_ticks)
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

    # Define a single namedtuple to capture all possible outputs.
    HeatmapResult = namedtuple("HeatmapResult", [
        "fig", "ax", "dendrogram_axes", "cond1_means", "cond2_means", "fold_changes"
    ])

    # Early exit: if neither figure nor data is requested, return nothing.
    if not (return_fig or return_data):
        return

    # Set the results based on requested outputs.
    result_fig = fig if return_fig else None
    result_ax = ax if return_fig else None
    result_dendrogram_axes = dendrogram_axes if (return_fig and dendrogram and len(dendrogram_axes) > 0) else None
    result_cond1_means = cond1_means if return_data else None
    result_cond2_means = cond2_means if return_data else None
    result_fold_changes = fold_changes if (return_data and fold_change_mode) else None

    # Return the unified result.
    return HeatmapResult(
        result_fig,
        result_ax,
        result_dendrogram_axes,
        result_cond1_means,
        result_cond2_means,
        result_fold_changes
    )