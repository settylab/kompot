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
    cmap: Union[str, mcolors.Colormap] = "viridis",
    dendrogram: bool = False,  # Whether to show dendrograms
    cluster_rows: bool = True,  # Whether to cluster rows
    cluster_cols: bool = True,  # Whether to cluster columns
    dendrogram_color: str = "black",  # Default dendrogram color
    figsize: Optional[Tuple[float, float]] = None,
    aspect_ratio: float = 0.3,  # Height/width ratio for automatic figsize (small aspect = wide plot)
    cell_size: float = 0.5,     # Base size for each cell in inches
    show_gene_labels: bool = True,
    show_group_labels: bool = True,
    gene_labels_size: int = 10,
    group_labels_size: int = 12,
    colorbar_title: str = "Expression",
    colorbar_kwargs: Optional[Dict[str, Any]] = None,
    title: Optional[str] = None,
    sort_genes: bool = True,
    center: Optional[float] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
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
        Whether to scale the expression values ('var', 'group' or 0, 1)
        Default is 'var' for gene-wise z-scoring
    cmap : str or colormap, optional
        Colormap to use for the heatmap
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
    aspect_ratio : float, optional
        Overall height/width ratio constraint for the figure. Default is 0.3, which tends to
        create wider plots. Higher values create taller plots. Set to 0 to disable this constraint
        and use only cell_size for determining dimensions.
    cell_size : float, optional
        Base size in inches for each cell when automatically calculating figure size.
        Default is 0.5 inches. Individual cells will always be drawn as squares regardless
        of this setting.
    show_gene_labels : bool, optional
        Whether to show gene labels
    show_group_labels : bool, optional
        Whether to show group labels
    gene_labels_size : int, optional
        Font size for gene labels
    group_labels_size : int, optional
        Font size for group labels
    colorbar_title : str, optional
        Title for the colorbar
    colorbar_kwargs : dict, optional
        Additional parameters for colorbar
    title : str, optional
        Title for the heatmap
    sort_genes : bool, optional
        Whether to sort genes by score
    center : float, optional
        Value to center the colormap at
    vmin : float, optional
        Minimum value for colormap
    vmax : float, optional
        Maximum value for colormap
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
    
    if run_info is not None and "params" in run_info:
        params = run_info["params"]
        if "conditions" in params and len(params["conditions"]) == 2:
            condition1 = params["conditions"][0]
            condition2 = params["conditions"][1]
        # In Kompot DE runs, the condition column is called "groupby"
        if "groupby" in params:
            condition_key = params["groupby"]
    
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

        # Apply gene-wise scaling if needed
        if standard_scale is not None:
            # Single scaling to avoid duplicate log messages
            if standard_scale == "var" or standard_scale == 0:
                logger.info("Applying gene-wise z-scoring (standard_scale='var')")
                
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
                
                # Apply gene-wise scaling - set is_split=True to ensure proper handling of hierarchical structure
                scaled = _apply_scaling(combined, standard_scale, is_split=True, has_hierarchical_index=True, log_message=False)
                
                # Extract the results
                cond1_means_scaled = scaled.loc["cond1"].copy()
                cond2_means_scaled = scaled.loc["cond2"].copy()
                
                # Copy scaled values back to original dataframes to preserve any columns
                # that might not have been in both conditions
                for col in shared_cols:
                    cond1_means[col] = cond1_means_scaled[col]
                    cond2_means[col] = cond2_means_scaled[col]
            else:
                # Apply regular scaling
                cond1_means = _apply_scaling(cond1_means, standard_scale, log_message=False)
                cond2_means = _apply_scaling(cond2_means, standard_scale)

        # Calculate a custom figure size based on cell count and aspect ratio
        if figsize is None:
            # After transposition, we'll have:
            # - genes on y-axis (rows)
            # - groups on x-axis (columns)
            
            # Base width on the number of groups (will be columns after transpose)
            data_width = n_groups * cell_size
            # Base height on the number of genes (will be rows after transpose)
            data_height = n_genes * cell_size
            
            # Set relative sizes - keep cells square for proper visualization
            cell_width_factor = 1.0  # Standard width
            cell_height_factor = 1.0  # Standard height for square cells
            
            # Apply the adjustments
            data_width = data_width * cell_width_factor
            data_height = data_height * cell_height_factor
            
            
            # Add space for labels, title, and other elements
            width_inches = data_width + 4  # For labels and sidebar
            # Make height more proportional to gene count - minimal padding for few genes
            height_inches = data_height + min(2, max(0.5, data_height * 0.2))  # Adaptive padding based on gene count
            
            # Cap the size for very large gene/group counts
            max_width = 20
            max_height = 30
            width_inches = min(width_inches, max_width)
            height_inches = min(height_inches, max_height)
            
            # Apply overall aspect ratio constraint if specified, but only to reduce height, never increase it
            if aspect_ratio > 0:
                current_ratio = height_inches / width_inches
                # Only apply if the current ratio is far from target and would make the plot smaller
                if current_ratio > aspect_ratio and abs(current_ratio - aspect_ratio) > 0.1:
                    # Too tall - increase width but don't exceed max
                    width_inches = min(height_inches / aspect_ratio, max_width)
            
            # Add extra space for dendrograms if needed
            if dendrogram:
                if cluster_cols:
                    width_inches += 1.5  # More space for column dendrogram on right
                if cluster_rows:
                    height_inches += 1.2  # More space for row dendrogram on top
                # Extra space for title when dendrograms are present
                height_inches += 0.8
            
            figsize = (width_inches, height_inches)

        # Create figure if no axes provided
        create_fig = ax is None
        if create_fig:
            # Create figure with room for the sidebar and dendrograms
            fig = plt.figure(figsize=figsize)
            
            # Define layout regions - adjust for dendrograms
            # Set reasonable margins
            bottom_margin = 0.1  # Bottom margin for x-axis labels
            left_margin = 0.15  # Standard left margin
            
            # Right margin depends on column dendrogram and sidebar
            # Move column dendrogram to right side (before sidebar)
            if dendrogram and cluster_cols:
                right_margin = 0.65  # Less space on right to make room for column dendrogram
            else:
                right_margin = 0.75  # Standard right margin for just sidebar
                
            # Top margin depends on row dendrogram
            # Create more space at the top to avoid title overlap
            if dendrogram and cluster_rows:
                top_margin = 0.75  # More space on top for row dendrogram + title
            else:
                top_margin = 0.85  # Standard top margin without dendrogram
            
            # Main grid layout for the plot area
            main_grid = GridSpec(1, 1, 
                                left=left_margin, 
                                right=right_margin, 
                                top=top_margin, 
                                bottom=bottom_margin)
            
            # Add the main axes for the heatmap
            ax = fig.add_subplot(main_grid[0, 0])
            
            # Add dendrogram axes if needed - corrected positioning with proper names
            dendrogram_axes = {}
            if dendrogram:
                # For column dendrogram - ON THE RIGHT for groups (after transpose, groups are on y-axis)
                if cluster_cols:
                    # Position the column dendrogram on the right side
                    col_dendrogram_width = 0.08
                    col_dendrogram_left = right_margin + 0.01
                    col_dendrogram_ax = fig.add_axes([col_dendrogram_left, bottom_margin, col_dendrogram_width, top_margin - bottom_margin])
                    dendrogram_axes['col'] = col_dendrogram_ax
                    col_dendrogram_ax.set_axis_off()
                
                # For row dendrogram - ON TOP for genes (after transpose, genes are on x-axis)
                if cluster_rows:
                    # Position dendrogram with proper spacing for title
                    row_dendrogram_ax = fig.add_axes([left_margin, top_margin + 0.02, right_margin - left_margin, 0.1])
                    dendrogram_axes['row'] = row_dendrogram_ax
                    row_dendrogram_ax.set_axis_off()
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

        # Set up colormap normalization
        norm, cmap_obj, vmin, vmax = _setup_colormap_normalization(
            all_data, center, vmin, vmax, cmap
        )

        # Use square cells for better visualization
        # Each cell should be square to maintain proper proportions
        cell_width = 1.0
        cell_height = 1.0
        
        # Do not set the global aspect ratio - let matplotlib handle it based on the figure size
        # This allows for proper alignment with labels and dendrograms
        
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

        # Set title if provided, or create an informative default title
        # Adjust title position based on dendrogram presence
        
        # Set a consistent title position with appropriate spacing
        y_pos = 0.98  # Default position
        
        # Adjust based on dendrogram presence
        if dendrogram and cluster_rows:
            # Lower the title position when row dendrogram is present
            y_pos = 0.94  # More space above for dendrogram
            
        # Choose title content
        if title:
            # Use provided title
            fig.suptitle(title, fontsize=18, fontweight='bold', y=y_pos)
        elif condition1_name and condition2_name:
            # Generate default title
            fig.suptitle(
                f"{condition1_name} vs {condition2_name}\n"
                f"Mean expression by {groupby}", 
                fontsize=18, fontweight='bold', y=y_pos
            )

        # Create sidebar for legend and colorbar - always position it on the right
        # Start sidebar after the column dendrogram if present
        if dendrogram and cluster_cols:
            # Account for column dendrogram on the right
            sidebar_left = right_margin + 0.11  # Add dendrogram width + small gap
        else:
            sidebar_left = right_margin + 0.02  # Standard position
            
        sidebar_width = 0.15  # Make sidebar narrower to save space
        legend_height = 0.25
        colorbar_height = 0.25
        
        # Add legend at top right
        legend_ax = fig.add_axes([sidebar_left, 0.65, sidebar_width, legend_height])
        legend_ax.axis('off')  # Hide the axes

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

        # Custom handler for the triangular patches - make sure they match the display in the plot
        class HandlerTriangle(HandlerPatch):
            def create_artists(
                self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
            ):
                # With transposed plot, we need to update the triangles
                # The condition1 is still lower left, condition2 is upper right
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

        # Add the legend using our custom handler
        legend = legend_ax.legend(
            handles=legend_elements,
            loc="center",
            title="Conditions",
            frameon=False,
            handler_map={mpatches.Polygon: HandlerTriangle()},
        )
        legend.get_title().set_fontweight('bold')
        
        # Add colorbar below the legend
        colorbar_width = 0.03  # Narrower colorbar
        colorbar_left = sidebar_left + (sidebar_width - colorbar_width) / 2  # Center in sidebar
        cax = fig.add_axes([colorbar_left, 0.3, colorbar_width, colorbar_height])

        # Create the colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax)
        
        # Remove colorbar outline and grid
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(axis='both', which='both', length=0)
        cbar.ax.grid(False)
        
        # Make sure there are no spines visible
        for spine in cbar.ax.spines.values():
            spine.set_visible(False)

        # Set colorbar label based on whether data was z-scored
        if standard_scale == "var" or standard_scale == 0:
            cbar.set_label(
                "Z-score" if colorbar_title == "Expression" else colorbar_title, 
                labelpad=10, fontweight='bold'
            )
        else:
            cbar.set_label(colorbar_title, labelpad=10, fontweight='bold')

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