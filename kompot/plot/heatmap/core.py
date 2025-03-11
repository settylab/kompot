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
        Figure size as (width, height) in inches
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
                # Apply scaling to both matrices at once
                combined = pd.concat([cond1_means, cond2_means], keys=["cond1", "cond2"])
                scaled = _apply_scaling(combined, standard_scale, log_message=False)
                # Extract the results
                cond1_means = scaled.loc["cond1"]
                cond2_means = scaled.loc["cond2"]
            else:
                # Apply regular scaling
                cond1_means = _apply_scaling(cond1_means, standard_scale, log_message=False)
                cond2_means = _apply_scaling(cond2_means, standard_scale)

        # With transposed data, n_genes and n_groups are swapped
        # Setup for plotting with transposed data dimensions
        if figsize is None:
            # For transposed view: groups are on x-axis (columns), genes on y-axis (rows)
            figsize = _calculate_figsize(
                n_genes, n_groups, dendrogram, cluster_rows, cluster_cols
            )
            # Add extra width for the sidebar
            figsize = (figsize[0] + 2, figsize[1])

        # Create figure if no axes provided
        create_fig = ax is None
        if create_fig:
            # Create figure with room for the sidebar
            fig = plt.figure(figsize=figsize)
            
            # Main grid layout for the plot area - adjust for transposed view
            main_grid_width = 0.75  # 75% for the main plot area
            main_grid = GridSpec(1, 1, left=0.1, right=main_grid_width, top=0.9, bottom=0.1)
            
            # Add the main axes for the heatmap
            ax = fig.add_subplot(main_grid[0, 0])
            
            # Add dendrogram axes if needed - adjusted for transposed view
            dendrogram_axes = {}
            if dendrogram:
                # For row dendrogram (now on the left for genes)
                if cluster_rows:
                    row_dendrogram_ax = fig.add_axes([0.01, 0.1, 0.08, 0.8])
                    dendrogram_axes['row'] = row_dendrogram_ax
                    row_dendrogram_ax.set_axis_off()
                
                # For column dendrogram (now on top for groups)
                if cluster_cols:
                    col_dendrogram_ax = fig.add_axes([0.1, 0.91, 0.65, 0.08])
                    dendrogram_axes['col'] = col_dendrogram_ax
                    col_dendrogram_ax.set_axis_off()
        else:
            # Use existing axes
            fig = ax.figure
            dendrogram_axes = {}

        # Handle clustering
        if cluster_rows or cluster_cols:
            # Combined data for clustering - impute NaNs for distance calculation
            combined = pd.concat([cond1_means, cond2_means], axis=1)
            # Fill NaN with column means for clustering purposes only
            combined_for_clustering = combined.fillna(combined.mean())
            
            # Row clustering (genes - now on y-axis/rows after transpose)
            if cluster_rows:
                # Calculate row linkage for genes (rows of transposed data)
                row_dist = ssd.pdist(combined_for_clustering.values)
                row_linkage_matrix = linkage(row_dist, method='average')
                
                if dendrogram and 'row' in dendrogram_axes:
                    # Draw row dendrogram
                    row_dendrogram = scipy_dendrogram(
                        row_linkage_matrix,
                        orientation='left',
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
            
            # Column clustering (groups - now on x-axis/columns after transpose)
            if cluster_cols:
                # Calculate column linkage for groups (columns of transposed data)
                col_dist = ssd.pdist(combined_for_clustering.values.T)
                col_linkage_matrix = linkage(col_dist, method='average')
                
                if dendrogram and 'col' in dendrogram_axes:
                    # Draw column dendrogram
                    col_dendrogram = scipy_dendrogram(
                        col_linkage_matrix,
                        orientation='top',
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
                
                # Make sure we don't have empty cluster issue
                if len(col_order) != cond1_means.shape[1]:
                    logger.warning(f"Mismatch in col_order length ({len(col_order)}) vs. data columns ({cond1_means.shape[1]})")
                    # Adjust col_order to match the number of columns in the dataframes
                    col_order = col_order[:min(len(col_order), cond1_means.shape[1])]
                
                # Apply the column ordering - safely handle potential indexing errors
                try:
                    cond1_means = cond1_means.iloc[:, col_order]
                    cond2_means = cond2_means.iloc[:, col_order]
                except IndexError as e:
                    logger.error(f"IndexError during column ordering: {e}")
                    # Continue without reordering if there's an error

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

        # Draw split cells - use normal size cells (1x1)
        for i, gene in enumerate(cond1_means.index):
            for j, group in enumerate(cond1_means.columns):
                val1 = cond1_means.iloc[i, j]
                val2 = cond2_means.iloc[i, j]
                _draw_diagonal_split_cell(
                    ax, j, i, 1, 1, val1, val2, cmap_obj, vmin, vmax, 
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

        # Add gene labels (now on y-axis)
        if show_gene_labels:
            ax.set_yticks(np.arange(len(cond1_means.index)) + 0.5)
            ax.set_yticklabels(cond1_means.index, fontsize=gene_labels_size, va='center')

        # Remove the grid
        ax.grid(False)

        # Set title if provided, or create an informative default title
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        elif condition1_name and condition2_name:
            ax.set_title(
                f"{condition1_name} vs {condition2_name}\n"
                f"Mean expression by {groupby}", 
                fontsize=14, fontweight='bold'
            )

        # Create sidebar for legend and colorbar (to the right of the main plot)
        # Define position of sidebar elements
        sidebar_left = main_grid_width + 0.02  # Start sidebar just after main plot
        sidebar_width = 0.2
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