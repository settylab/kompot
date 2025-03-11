"""Heatmap and related plotting functions."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Optional, Union, List, Tuple, Dict, Any, Sequence, Literal
from anndata import AnnData
import pandas as pd
import logging

from ..utils import get_run_from_history, KOMPOT_COLORS
from .volcano import _extract_conditions_from_key, _infer_da_keys

try:
    import scanpy as sc

    _has_scanpy = True
except ImportError:
    _has_scanpy = False

logger = logging.getLogger("kompot")


def _infer_direction_key(
    adata: AnnData, run_id: int = -1, direction_column: Optional[str] = None
):
    """
    Infer direction column for direction barplots from AnnData object.

    Parameters
    ----------
    adata : AnnData
        AnnData object with differential abundance results
    run_id : int, optional
        Run ID to use. Default is -1 (latest run).
    direction_column : str, optional
        Direction column to use. If provided, will be returned as is.

    Returns
    -------
    tuple
        (direction_column, condition1, condition2) with the inferred values
    """
    # If direction column already provided, just check if it exists
    if direction_column is not None:
        if direction_column in adata.obs.columns:
            # Try to extract conditions from the column name
            conditions = _extract_conditions_from_key(direction_column)
            if conditions:
                condition1, condition2 = conditions
                return direction_column, condition1, condition2
            else:
                return direction_column, None, None
        else:
            logger.warning(
                f"Provided direction_column '{direction_column}' not found in adata.obs"
            )
            direction_column = None

    # Get run info from specified run_id - specifically from kompot_da
    run_info = get_run_from_history(adata, run_id, analysis_type="da")
    condition1 = None
    condition2 = None

    # Get condition information
    if run_info is not None and "params" in run_info:
        params = run_info["params"]
        if "conditions" in params and len(params["conditions"]) == 2:
            condition1 = params["conditions"][0]
            condition2 = params["conditions"][1]

    # First try to get direction column from run info field_names
    if run_info is not None and "field_names" in run_info:
        field_names = run_info["field_names"]
        if "direction_key" in field_names:
            direction_column = field_names["direction_key"]
            # Check that column exists
            if direction_column not in adata.obs.columns:
                direction_column = None

    # If not found in field_names, try the older method with abundance_key
    if direction_column is None and run_info is not None:
        if "abundance_key" in run_info:
            result_key = run_info["abundance_key"]
            if result_key in adata.uns and "run_info" in adata.uns[result_key]:
                key_run_info = adata.uns[result_key]["run_info"]
                if "direction_key" in key_run_info:
                    direction_column = key_run_info["direction_key"]

    # If still not found, look for columns matching pattern
    if direction_column is None:
        direction_cols = [
            col
            for col in adata.obs.columns
            if "kompot_da_log_fold_change_direction" in col
        ]
        if not direction_cols:
            return None, condition1, condition2
        elif len(direction_cols) == 1:
            direction_column = direction_cols[0]
        else:
            # If conditions provided, try to find matching column
            if condition1 and condition2:
                for col in direction_cols:
                    if f"{condition1}_vs_{condition2}" in col:
                        direction_column = col
                        break
                    elif f"{condition2}_vs_{condition1}" in col:
                        direction_column = col
                        # Keep this warning as it's informative about reversed condition order
                        logger.warning(
                            f"Found direction column with reversed conditions: {col}"
                        )
                        break
                if direction_column is None:
                    direction_column = direction_cols[0]
                    # Keep this warning as it's important information about ambiguity
                    logger.warning(
                        f"Multiple direction columns found, using the first one: {direction_column}"
                    )
            else:
                direction_column = direction_cols[0]
                # Keep this warning as it's important information about ambiguity
                logger.warning(
                    f"Multiple direction columns found, using the first one: {direction_column}"
                )

    # If we found a direction column but not conditions, try to extract them from the column name
    if direction_column is not None and (condition1 is None or condition2 is None):
        conditions = _extract_conditions_from_key(direction_column)
        if conditions:
            condition1, condition2 = conditions

    return direction_column, condition1, condition2


def _infer_heatmap_keys(
    adata: AnnData,
    run_id: Optional[int] = None,
    lfc_key: Optional[str] = None,
    score_key: Optional[str] = "kompot_de_mahalanobis",
):
    """
    Infer heatmap keys from AnnData object.

    Parameters
    ----------
    adata : AnnData
        AnnData object with differential expression results
    run_id : int, optional
        Run ID to use. If None, uses latest run.
    lfc_key : str, optional
        Log fold change key. If provided, will be returned as is.
    score_key : str, optional
        Score key. If "kompot_de_mahalanobis", might be replaced with a run-specific key.

    Returns
    -------
    tuple
        (lfc_key, score_key) with the inferred keys
    """
    inferred_lfc_key = lfc_key
    inferred_score_key = score_key

    # If keys already provided, just return them
    if inferred_lfc_key is not None and not (
        score_key == "kompot_de_mahalanobis"
        and not any(k == score_key for k in adata.var.columns)
    ):
        return inferred_lfc_key, inferred_score_key

    # Use the specified run_id to get keys from the run history if lfc_key is not provided
    if inferred_lfc_key is None:
        # Get run info from kompot_de for the specific run_id
        # First, try to use the provided run_id (which may be None)
        run_info = get_run_from_history(adata, run_id, analysis_type="de")

        # If that fails and run_id is None, try using -1 for the latest run
        if run_info is None and run_id is None:
            run_info = get_run_from_history(adata, -1, analysis_type="de")
            if run_info is not None:
                logger.info("Using latest run (run_id=-1) for key inference")

        if run_info is not None and "field_names" in run_info:
            field_names = run_info["field_names"]

            # Get lfc_key from field_names
            if "mean_lfc_key" in field_names:
                inferred_lfc_key = field_names["mean_lfc_key"]
                # Check that column exists
                if inferred_lfc_key not in adata.var.columns:
                    logger.warning(
                        f"Found mean_lfc_key '{inferred_lfc_key}' in run info, but column not in adata.var"
                    )
                    inferred_lfc_key = None

            # Get score_key from field_names if needed
            if (
                score_key == "kompot_de_mahalanobis"
                and "mahalanobis_key" in field_names
            ):
                inferred_score_key = field_names["mahalanobis_key"]
                # Check that column exists
                if inferred_score_key not in adata.var.columns:
                    logger.warning(
                        f"Found mahalanobis_key '{inferred_score_key}' in run info, but column not in adata.var"
                    )
                    inferred_score_key = None

        # For backwards compatibility - if we still don't have an lfc_key, try to infer from column names
        # Try to infer from column names
        lfc_keys = [
            k for k in adata.var.columns if "kompot_de_" in k and "lfc" in k.lower()
        ]
        if len(lfc_keys) == 1:
            inferred_lfc_key = lfc_keys[0]
        elif len(lfc_keys) > 1:
            # If multiple keys found, try to find the mean or avg one
            mean_keys = [
                k for k in lfc_keys if "mean" in k.lower() or "avg" in k.lower()
            ]
            if mean_keys:
                inferred_lfc_key = mean_keys[0]
            else:
                inferred_lfc_key = lfc_keys[0]

    # If lfc_key still not found, raise error
    if inferred_lfc_key is None:
        raise ValueError(
            "Could not infer lfc_key from the specified run. Please specify manually."
        )

    return inferred_lfc_key, inferred_score_key


def direction_barplot(
    adata: AnnData,
    category_column: str,
    direction_column: Optional[str] = None,
    condition1: Optional[str] = None,
    condition2: Optional[str] = None,
    normalize: Literal["index", "columns", None] = "index",
    figsize: Tuple[float, float] = (12, 6),
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    colors: Optional[Dict[str, str]] = None,
    rotation: float = 90,
    legend_title: str = "Direction",
    legend_loc: str = "best",
    stacked: bool = True,
    sort_by: Optional[str] = None,
    ascending: bool = False,
    ax: Optional[plt.Axes] = None,
    return_fig: bool = False,
    save: Optional[str] = None,
    run_id: int = -1,
    **kwargs,
) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """
    Create a barplot showing the direction of change distribution across categories.

    This function creates a stacked or grouped barplot showing the distribution of
    up/down/neutral changes across different categories (like cell types).

    Parameters
    ----------
    adata : AnnData
        AnnData object containing differential abundance results
    category_column : str
        Column in adata.obs to use for grouping (e.g., "cell_type")
    direction_column : str, optional
        Column in adata.obs containing direction information.
        If None, will try to infer from the run specified by run_id.
    condition1 : str, optional
        Name of condition 1 (denominator in fold change).
        If None, will try to infer from the run_id.
    condition2 : str, optional
        Name of condition 2 (numerator in fold change).
        If None, will try to infer from the run_id.
    normalize : str or None, optional
        How to normalize the data. Options:
        - "index": normalize across rows (sum to 100% for each category)
        - "columns": normalize across columns (sum to 100% for each direction)
        - None: use raw counts
    figsize : tuple, optional
        Figure size as (width, height) in inches
    title : str, optional
        Plot title. If None and conditions provided, uses "Direction of Change by {category_column}\n{condition1} vs {condition2}"
    xlabel : str, optional
        Label for x-axis. If None, uses the category_column
    ylabel : str, optional
        Label for y-axis. Defaults to "Percentage (%)" when normalize="index", otherwise "Count"
    colors : dict, optional
        Dictionary mapping direction values to colors. Default is {"up": "#d73027", "down": "#4575b4", "neutral": "#d3d3d3"}
    rotation : float, optional
        Rotation angle for x-tick labels
    legend_title : str, optional
        Title for the legend
    legend_loc : str, optional
        Location for the legend
    stacked : bool, optional
        Whether to create a stacked (True) or grouped (False) bar plot
    sort_by : str, optional
        Direction category to sort by (e.g., "up", "down"). If None, uses the order in the data
    ascending : bool, optional
        Whether to sort in ascending order
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure
    return_fig : bool, optional
        If True, returns the figure and axes
    save : str, optional
        Path to save figure. If None, figure is not saved
    run_id : int, optional
        Specific run ID to use for fetching data from run history.
        Negative indices count from the end (-1 is the latest run).
        This determines which differential abundance run's data is used.
    **kwargs :
        Additional parameters passed to pandas.DataFrame.plot

    Returns
    -------
    If return_fig is True, returns (fig, ax)
    """
    # Default colors if not provided
    if colors is None:
        colors = {
            "up": KOMPOT_COLORS["direction"]["up"],
            "down": KOMPOT_COLORS["direction"]["down"],
            "neutral": KOMPOT_COLORS["direction"]["neutral"],
        }

    # Default ylabel based on normalization method
    if ylabel is None:
        ylabel = "Percentage (%)" if normalize == "index" else "Count"

    # Calculate the actual (positive) run ID for logging
    actual_run_id = None
    if run_id < 0:
        if "kompot_da" in adata.uns and "run_history" in adata.uns["kompot_da"]:
            actual_run_id = len(adata.uns["kompot_da"]["run_history"]) + run_id
        elif "kompot_run_history" in adata.uns:
            actual_run_id = len(adata.uns["kompot_run_history"]) + run_id
        else:
            actual_run_id = run_id
    else:
        actual_run_id = run_id

    # Use the helper function to infer the direction column and conditions
    direction_column, inferred_condition1, inferred_condition2 = _infer_direction_key(
        adata, run_id, direction_column
    )

    # Use the inferred conditions if not explicitly provided
    if condition1 is None:
        condition1 = inferred_condition1
    if condition2 is None:
        condition2 = inferred_condition2

    # Log which run is being used and the conditions
    conditions_str = (
        f": comparing {condition1} vs {condition2}" if condition1 and condition2 else ""
    )
    logger.info(f"Using DA run {actual_run_id} for direction_barplot{conditions_str}")

    # Raise error if direction column not found
    if direction_column is None:
        raise ValueError(
            "Could not find direction column. Please specify direction_column."
        )

    # Log the plot type and conditions first, then fields
    if condition1 and condition2:
        logger.info(f"Creating direction barplot for {condition1} vs {condition2}")
    else:
        logger.info(f"Creating direction barplot")

    # Log the fields being used in the plot
    logger.info(
        f"Using fields - category_column: '{category_column}', direction_column: '{direction_column}'"
    )

    # Create the crosstab
    crosstab = pd.crosstab(
        adata.obs[category_column], adata.obs[direction_column], normalize=normalize
    )

    # If normalize is "index", multiply by 100 for percentage
    if normalize == "index":
        crosstab = crosstab * 100

    # Order columns consistently
    if "up" in crosstab.columns and "down" in crosstab.columns:
        # Keep only the columns that exist in our data
        ordered_cols = [
            col for col in ["up", "down", "neutral"] if col in crosstab.columns
        ]
        crosstab = crosstab[ordered_cols]

    # Sort by specified direction if requested
    if sort_by is not None and sort_by in crosstab.columns:
        crosstab = crosstab.sort_values(by=sort_by, ascending=ascending)

    # Create figure if ax not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Color mapping - only use colors for directions that exist in our data
    plot_colors = [colors.get(col, "gray") for col in crosstab.columns]

    # Create the plot
    crosstab.plot(kind="bar", stacked=stacked, color=plot_colors, ax=ax, **kwargs)

    # Remove grid by default
    ax.grid(False)

    # Set labels and title
    ax.set_xlabel(xlabel if xlabel is not None else category_column)
    ax.set_ylabel(ylabel)

    # Set title if provided or can be inferred
    if title is None and condition1 and condition2:
        title = (
            f"Direction of Change by {category_column}\n{condition1} vs {condition2}"
        )
    if title:
        ax.set_title(title)

    # Rotate tick labels
    plt.xticks(rotation=rotation)

    # Set legend
    plt.legend(
        title=legend_title,
        loc=legend_loc,
        frameon=False,
        bbox_to_anchor=(1.05, 1),
        borderaxespad=0,
    )

    # Adjust layout to accommodate legend
    if legend_loc == "best" or legend_loc.startswith("right"):
        plt.tight_layout(rect=[0, 0, 0.85, 1])
    else:
        plt.tight_layout()

    # Save figure if path provided
    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight")

    # Return figure and axes if requested
    if return_fig:
        return fig, ax
    elif save is None:
        # Only show if not saving and not returning
        plt.show()


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
    import matplotlib.patches as patches
    from matplotlib.colors import Normalize

    # Normalize values
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.cm.get_cmap(cmap) if isinstance(cmap, str) else cmap

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
    lower_triangle = patches.Polygon(
        [[x, y], [x + w, y], [x, y + h]],
        facecolor=facecolor1,
        alpha=alpha,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )

    upper_triangle = patches.Polygon(
        [[x + w, y], [x + w, y + h], [x, y + h]],
        facecolor=facecolor2,
        alpha=alpha,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )

    # Add to axes
    ax.add_patch(lower_triangle)
    ax.add_patch(upper_triangle)

    # No diagonal line between triangles


def diagonal_split_heatmap(
    adata: AnnData,
    var_names: Optional[Union[List[str], Sequence[str]]] = None,
    groupby: str = None,
    n_top_genes: int = 20,
    gene_list: Optional[Union[List[str], Sequence[str]]] = None,
    lfc_key: Optional[str] = None,
    score_key: str = "kompot_de_mahalanobis",
    layer: Optional[str] = None,
    standard_scale: Optional[Union[str, int]] = "var",  # Default to gene-wise z-scoring
    cmap: Union[str, mcolors.Colormap] = "viridis",
    condition1_name: Optional[str] = None,
    condition2_name: Optional[str] = None,
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
    dendrogram: bool = False,
    cluster_rows: bool = True,
    cluster_cols: bool = True,
    dendrogram_color: str = "black",  # Set default dendrogram color to black
    exclude_groups: Optional[Union[str, List[str]]] = None,
    **kwargs,
) -> Union[
    None, Tuple[plt.Figure, plt.Axes], Tuple[plt.Figure, plt.Axes, Dict[str, plt.Axes]]
]:
    """
    Create a heatmap with diagonally split cells to display expression values for two conditions.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing expression data
    var_names : list, optional
        List of genes to include in the heatmap. If None, will use top genes
        based on score_key and lfc_key. This parameter is kept for backwards compatibility,
        gene_list is the preferred way to specify genes directly.
    groupby : str, optional
        Key in adata.obs for grouping cells. If None, no grouping is performed
    n_top_genes : int, optional
        Number of top genes to include if var_names and gene_list are None
    gene_list : list, optional
        Explicit list of genes to include in the heatmap. Takes precedence over var_names if both are provided.
    lfc_key : str, optional
        Key in adata.var for log fold change values.
        If None, will try to infer from kompot_de_ keys.
    score_key : str, optional
        Key in adata.var for significance scores.
        Default is "kompot_de_mahalanobis"
    layer : str, optional
        Layer in AnnData to use for expression values. If None, uses .X
    standard_scale : str or int, optional
        Whether to scale the expression values ('var', 'group' or 0 for rows, 1 for columns)
        Default is 'var' for gene-wise z-scoring
    cmap : str or colormap, optional
        Colormap to use for the heatmap
    condition1_name : str, optional
        Display name for condition 1 (lower-left triangle). If None, tries to infer from run.
    condition2_name : str, optional
        Display name for condition 2 (upper-right triangle). If None, tries to infer from run.
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
        Negative indices count from the end (-1 is the latest run). If None,
        uses the latest run information.
    condition_column : str, optional
        Column in adata.obs containing condition information.
        If None, tries to infer from run_info.
    observed : bool, optional
        Whether to use only observed combinations in groupby operations.
        Default is True, which improves performance with categorical data.
    dendrogram : bool, optional
        Whether to show dendrograms for clustering. Default is False.
    cluster_rows : bool, optional
        Whether to cluster rows (genes). Default is True.
    cluster_cols : bool, optional
        Whether to cluster columns (groups). Default is False.
    dendrogram_color : str, optional
        Color for dendrograms. Default is "black".
    exclude_groups : str or list, optional
        Group name(s) to exclude from the heatmap. Can be a single group name or
        a list of group names. Only applies when groupby is not None.
    **kwargs :
        Additional parameters passed to cell rendering

    Returns
    -------
    If return_fig is True and dendrogram is False, returns (fig, ax)
    If return_fig is True and dendrogram is True, returns (fig, ax, dendrogram_axes)
    where dendrogram_axes is a dictionary with keys 'row' and/or 'col' depending on clustering options
    """
    # Set default colorbar kwargs
    colorbar_kwargs = colorbar_kwargs or {}

    # Initialize run_info variable
    run_info = None

    # If gene_list is provided, use it directly
    if gene_list is not None:
        var_names = gene_list
        logger.info(f"Using provided gene_list with {len(gene_list)} genes/features")

        # We still need run info for condition names and other parameters
        run_info = get_run_from_history(adata, run_id, analysis_type="de")
    # If var_names not provided and no gene_list, get top genes based on DE results
    elif var_names is None:
        # Infer keys using the helper function
        lfc_key, score_key = _infer_heatmap_keys(adata, run_id, lfc_key, score_key)

        # Calculate the actual (positive) run ID for logging
        actual_run_id = None
        if run_id is not None:
            if run_id < 0:
                if "kompot_de" in adata.uns and "run_history" in adata.uns["kompot_de"]:
                    actual_run_id = len(adata.uns["kompot_de"]["run_history"]) + run_id
                else:
                    actual_run_id = run_id
            else:
                actual_run_id = run_id

        # Get condition information from the run specified by run_id
        condition1 = None
        condition2 = None
        condition_key = None
        run_info = get_run_from_history(adata, run_id, analysis_type="de")

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
            logger.info(
                f"Using condition_column '{condition_column}' from run information"
            )

        # Log which run is being used
        if run_id is not None:
            conditions_str = (
                f": comparing {condition1} vs {condition2}"
                if condition1 and condition2
                else ""
            )
            logger.info(
                f"Using DE run {actual_run_id} for diagonal split heatmap{conditions_str}"
            )
        else:
            logger.info("Using latest available DE run for diagonal split heatmap")

        # Log the fields being used
        logger.info(
            f"Using fields for diagonal split heatmap - lfc_key: '{lfc_key}', score_key: '{score_key}'"
        )

        # Get top genes based on score
        de_data = pd.DataFrame(
            {
                "gene": adata.var_names,
                "lfc": (
                    adata.var[lfc_key]
                    if lfc_key in adata.var
                    else np.zeros(adata.n_vars)
                ),
                "score": (
                    adata.var[score_key]
                    if score_key in adata.var
                    else np.zeros(adata.n_vars)
                ),
            }
        )

        if sort_genes:
            de_data = de_data.sort_values("score", ascending=False)

        # Get top genes
        var_names = de_data.head(n_top_genes)["gene"].tolist()

    # Log the plot type first
    logger.info(f"Creating diagonal split heatmap with {len(var_names)} genes/features")

    # Check that condition_column is provided
    if condition_column is None:
        logger.warning(
            "No condition_column could be inferred. Diagonal split requires a condition column."
        )
        return None

    # Check that condition_column exists
    if condition_column not in adata.obs.columns:
        logger.warning(
            f"Condition column '{condition_column}' not found in adata.obs columns."
        )
        return None

    # Get known conditions from run_info if available
    if run_info is not None and "params" in run_info:
        params = run_info["params"]
        run_condition1 = params.get("condition1")
        run_condition2 = params.get("condition2")
    else:
        run_condition1 = None
        run_condition2 = None

    # Get all unique conditions in the data
    all_conditions = adata.obs[condition_column].unique()

    # If we have conditions from run info, use those
    if run_condition1 is not None and run_condition2 is not None:
        # Check if these conditions exist in the data
        if run_condition1 in all_conditions and run_condition2 in all_conditions:
            unique_conditions = np.array([run_condition1, run_condition2])
            logger.info(
                f"Using conditions from run info: {run_condition1} and {run_condition2}"
            )
        else:
            missing = []
            if run_condition1 not in all_conditions:
                missing.append(run_condition1)
            if run_condition2 not in all_conditions:
                missing.append(run_condition2)
            logger.warning(f"Conditions from run info not found in data: {missing}")

            # Fall back to the first two conditions if we have exactly two
            if len(all_conditions) == 2:
                unique_conditions = all_conditions
                logger.info(
                    f"Falling back to available conditions: {unique_conditions[0]} and {unique_conditions[1]}"
                )
            else:
                logger.warning(
                    f"Diagonal split requires exactly 2 known conditions, but found {len(all_conditions)}."
                )
                return None
    else:
        # Without run info, check that there are exactly two conditions
        if len(all_conditions) != 2:
            logger.warning(
                f"Diagonal split requires exactly 2 conditions, but found {len(all_conditions)}."
            )
            return None
        unique_conditions = all_conditions

    # Set condition names, giving priority to explicitly provided names
    if condition1_name is None:
        condition1_name = unique_conditions[0]
    if condition2_name is None:
        condition2_name = unique_conditions[1]

    # Log the conditions
    logger.info(
        f"Using conditions: {condition1_name} (lower-left triangle) and {condition2_name} (upper-right triangle)"
    )

    # Extract layer from run parameters if not explicitly provided
    run_layer = None
    if layer is None and run_info is not None and "params" in run_info:
        run_layer = run_info["params"].get("layer")
        if run_layer is not None:
            logger.info(f"Using layer '{run_layer}' from run information")
            layer = run_layer

    # Log the data sources being used for the heatmap
    if layer is not None and layer in adata.layers:
        logger.info(f"Using expression data from layer: '{layer}'")
        expr_matrix = (
            adata[:, var_names].layers[layer].toarray()
            if hasattr(adata.layers[layer], "toarray")
            else adata[:, var_names].layers[layer]
        )
    else:
        if layer is not None:
            logger.warning(
                f"Requested layer '{layer}' not found, falling back to adata.X"
            )
        logger.info(f"Using expression data from adata.X")
        expr_matrix = (
            adata[:, var_names].X.toarray()
            if hasattr(adata.X, "toarray")
            else adata[:, var_names].X
        )

    # Create dataframe with expression data
    expr_df = pd.DataFrame(expr_matrix, index=adata.obs_names, columns=var_names)

    # Add condition information to expr_df
    expr_df[condition_column] = adata.obs[condition_column].values

    # Group by condition
    if groupby is not None and groupby in adata.obs:
        # Calculate mean expression per group and condition
        logger.info(
            f"Grouping expression by '{groupby}' ({adata.obs[groupby].nunique()} groups)"
        )

        # Add groupby column to expr_df
        expr_df[groupby] = adata.obs[groupby].values

        # Handle group exclusion if specified
        if exclude_groups is not None:
            # Convert single group to list for consistent handling
            if isinstance(exclude_groups, str):
                exclude_groups = [exclude_groups]

            # Check for non-existent groups
            available_groups = adata.obs[groupby].unique()
            non_existent = [g for g in exclude_groups if g not in available_groups]
            if non_existent:
                logger.warning(
                    f"Some groups in exclude_groups do not exist: {', '.join(non_existent)}"
                )
                # Filter to only include existing groups
                exclude_groups = [g for g in exclude_groups if g in available_groups]

            # Filter out the excluded groups
            original_size = len(expr_df)
            expr_df = expr_df[~expr_df[groupby].isin(exclude_groups)]
            filtered_size = len(expr_df)

            if filtered_size < original_size:
                logger.info(
                    f"Excluded {original_size - filtered_size} cells from groups: {', '.join(exclude_groups)}"
                )

                # Check if we have any remaining cells
                if filtered_size == 0:
                    raise ValueError(
                        f"All cells were excluded after filtering out groups: {', '.join(exclude_groups)}. "
                        f"Please check your exclude_groups parameter."
                    )

        # Group by both groupby and condition_column
        grouped_expr = expr_df.groupby(
            [groupby, condition_column], observed=observed
        ).mean()

        # Reshape to have groups as index and genes as columns
        # This creates a hierarchical column structure
        unstacked_expr = grouped_expr.unstack(level=1)

        # Get the condition values for each group
        condition_values = {}

        # Only include groups that are present in the grouped data (after exclusion)
        available_groups = unstacked_expr.index.unique()

        for group in available_groups:
            # Create a dict of {gene: {condition1: value, condition2: value}}
            group_data = {}
            for gene in var_names:
                # Safely extract values for each condition, using NaN for missing data
                cond_values = {}

                try:
                    cond_values[str(condition1_name)] = unstacked_expr.loc[
                        group, (gene, unique_conditions[0])
                    ]
                except KeyError:
                    # Only log if this is not due to exclusion
                    if exclude_groups is None or group not in exclude_groups:
                        logger.debug(
                            f"Missing data for group {group}, gene {gene}, condition {unique_conditions[0]}"
                        )
                    cond_values[str(condition1_name)] = np.nan

                try:
                    cond_values[str(condition2_name)] = unstacked_expr.loc[
                        group, (gene, unique_conditions[1])
                    ]
                except KeyError:
                    # Only log if this is not due to exclusion
                    if exclude_groups is None or group not in exclude_groups:
                        logger.debug(
                            f"Missing data for group {group}, gene {gene}, condition {unique_conditions[1]}"
                        )
                    cond_values[str(condition2_name)] = np.nan

                # Only add if we have at least one valid value
                if not (
                    np.isnan(cond_values[str(condition1_name)])
                    and np.isnan(cond_values[str(condition2_name)])
                ):
                    group_data[gene] = cond_values
                else:
                    logger.warning(
                        f"Missing data for both conditions for group {group}, gene {gene}"
                    )
                    continue
            condition_values[group] = group_data
    else:
        logger.error(
            "Diagonal split heatmap requires a groupby parameter for cell grouping"
        )
        return None

    # Create matrices for clustering (if dendrogram=True) and for z-scoring
    # These matrices will have genes as rows and groups as columns, with concatenated conditions
    clustering_gene_array = None  # Initialize to avoid reference errors

    # Always build the gene_array for either dendrogram or standard_scale
    # Create matrices for clustering and z-scoring
    gene_group_matrix = {}

    # For each gene, create a row with values for all groups
    for gene in var_names:
        gene_row = []
        for group in condition_values:
            if gene in condition_values[group]:
                # Concatenate both condition values for this gene/group
                gene_data = condition_values[group][gene]
                cond1_val = gene_data.get(str(condition1_name), np.nan)
                cond2_val = gene_data.get(str(condition2_name), np.nan)
                gene_row.extend([cond1_val, cond2_val])
            else:
                # If gene not in this group, add NaN values
                gene_row.extend([np.nan, np.nan])

        # Store the row for this gene
        gene_group_matrix[gene] = gene_row

    # Convert to numpy array for easier manipulation
    gene_array = np.array([gene_group_matrix[gene] for gene in var_names])

    # Handle NaN values for clustering
    if cluster_rows or cluster_cols:
        # For clustering, replace NaN with mean of non-NaN values for that gene
        clustering_gene_array = gene_array.copy()
        for i, row in enumerate(clustering_gene_array):
            mask = ~np.isnan(row)
            if np.any(mask):  # If we have any non-NaN values
                row_mean = np.mean(row[mask])
                row[~mask] = row_mean  # Replace NaN with mean
            else:
                row[:] = 0  # If all NaN, replace with 0

    # Now we have matrix for clustering with genes as rows and groups*conditions as columns

    # Scale data if requested
    if standard_scale == "var" or standard_scale == 0:
        # Perform gene-wise z-scoring
        logger.info(
            "Applying gene-wise z-scoring (standard_scale='var') on concatenated condition data"
        )

        # Create a new condition_values dictionary with z-scored values
        z_scored_values = {}

        # Apply z-scoring to each gene (row) in the gene_array
        means = np.nanmean(gene_array, axis=1, keepdims=True)
        stds = np.nanstd(gene_array, axis=1, keepdims=True)
        # Avoid division by zero
        stds[stds == 0] = 1.0
        stds[np.isnan(stds)] = 1.0

        # Z-score the gene array
        z_scored_gene_array = (gene_array - means) / stds

        # Convert back to dictionary structure
        for i, gene in enumerate(var_names):
            gene_row = z_scored_gene_array[i]

            # Split the z-scored row back into the condition_values structure
            for j, group in enumerate(condition_values):
                if group not in z_scored_values:
                    z_scored_values[group] = {}

                if gene in condition_values[group]:
                    # Create a new dict for this gene
                    z_scored_values[group][gene] = {}

                    # Extract the z-scored values for each condition
                    col_idx = j * 2  # Each group has 2 conditions
                    # Handle the first condition
                    if str(condition1_name) in condition_values[group][gene]:
                        z_scored_values[group][gene][str(condition1_name)] = gene_row[
                            col_idx
                        ]

                    # Handle the second condition
                    if str(condition2_name) in condition_values[group][gene]:
                        z_scored_values[group][gene][str(condition2_name)] = gene_row[
                            col_idx + 1
                        ]

        # Replace original values with z-scored values
        condition_values = z_scored_values

    elif standard_scale == "group" or standard_scale == 1:
        logger.info(
            "Applying group-wise z-scoring (standard_scale='group') on concatenated condition data"
        )

        # Transpose to get groups as rows for group-wise z-scoring
        group_gene_array = gene_array.T

        # Apply z-scoring to each group (now rows in transposed array)
        means = np.nanmean(group_gene_array, axis=1, keepdims=True)
        stds = np.nanstd(group_gene_array, axis=1, keepdims=True)
        # Avoid division by zero
        stds[stds == 0] = 1.0
        stds[np.isnan(stds)] = 1.0

        # Z-score the group array
        z_scored_group_array = (group_gene_array - means) / stds

        # Transpose back to have genes as rows
        z_scored_gene_array = z_scored_group_array.T

        # Convert back to dictionary structure
        z_scored_values = {}
        for i, gene in enumerate(var_names):
            gene_row = z_scored_gene_array[i]

            # Split the z-scored row back into the condition_values structure
            for j, group in enumerate(condition_values):
                if group not in z_scored_values:
                    z_scored_values[group] = {}

                if gene in condition_values[group]:
                    # Create a new dict for this gene
                    z_scored_values[group][gene] = {}

                    # Extract the z-scored values for each condition
                    col_idx = j * 2  # Each group has 2 conditions
                    # Handle the first condition
                    if str(condition1_name) in condition_values[group][gene]:
                        z_scored_values[group][gene][str(condition1_name)] = gene_row[
                            col_idx
                        ]

                    # Handle the second condition
                    if str(condition2_name) in condition_values[group][gene]:
                        z_scored_values[group][gene][str(condition2_name)] = gene_row[
                            col_idx + 1
                        ]

        # Replace original values with z-scored values
        condition_values = z_scored_values

    # Determine dimensions
    n_rows = len(var_names)
    n_cols = len(condition_values)

    # Variables to store dendrogram information
    row_linkage = None
    col_linkage = None
    row_order = list(range(n_rows))  # Default order (0 to n_rows-1)
    col_order = list(range(n_cols))  # Default order (0 to n_cols-1)
    dendrogram_axes = {}

    # Always perform clustering if cluster=True (default)
    if cluster_rows or cluster_cols:
        from scipy.cluster.hierarchy import linkage, dendrogram as scipy_dendrogram
        import scipy.spatial.distance as ssd

        logger.info("Computing clustering for heatmap")

        # Use the clustering_gene_array we prepared earlier for row clustering
        if cluster_rows:
            # Compute distance matrix for genes (rows)
            row_dist = ssd.pdist(clustering_gene_array, metric="euclidean")
            row_linkage = linkage(row_dist, method="average")

            # Get the order of rows from dendrogram
            row_dendrogram = scipy_dendrogram(row_linkage, no_plot=True)
            row_order = row_dendrogram["leaves"]

            logger.info(f"Clustered {len(row_order)} genes")

        # Use the clustering_gene_array for column clustering (transpose for columns)
        if cluster_cols:
            # Transpose for column clustering
            # Each column in clustering_gene_array represents a group-condition pair
            # We need to consolidate the data for the same group across both conditions
            n_groups = len(condition_values.keys())
            col_data = np.zeros((n_groups, clustering_gene_array.shape[0]))

            for i, group_idx in enumerate(range(n_groups)):
                # Get columns for this group (2 columns per group, one for each condition)
                col_idx_1 = group_idx * 2
                col_idx_2 = group_idx * 2 + 1

                # Take average of both conditions for clustering purposes
                if col_idx_2 < clustering_gene_array.shape[1]:
                    col_data[i, :] = np.nanmean(
                        [
                            clustering_gene_array[:, col_idx_1],
                            clustering_gene_array[:, col_idx_2],
                        ],
                        axis=0,
                    )
                else:
                    col_data[i, :] = clustering_gene_array[:, col_idx_1]

            # Compute distance matrix for columns (groups)
            col_dist = ssd.pdist(col_data, metric="euclidean")
            col_linkage = linkage(col_dist, method="average")

            # Get the order of columns from dendrogram
            col_dendrogram = scipy_dendrogram(col_linkage, no_plot=True)
            col_order = col_dendrogram["leaves"]

            logger.info(f"Clustered {len(col_order)} groups")

    # Calculate appropriate figsize if not provided
    if figsize is None:
        # Calculate cell size that ensures square tiles
        base_size = 0.5  # Base cell size

        # Base width and height
        width_inches = 6 + n_cols * base_size
        height_inches = 6 + n_rows * base_size

        # Add space for dendrograms if enabled
        if dendrogram:
            if cluster_rows:
                width_inches += 1.5  # Add space for row dendrogram
            if cluster_cols:
                height_inches += 1.5  # Add space for column dendrogram

        # Set the figure size
        figsize = (width_inches, height_inches)

    # Create figure
    fig = plt.figure(figsize=figsize)

    # Create subplot grid based on dendrogram settings
    from matplotlib.gridspec import GridSpec

    if dendrogram:
        # Define heights and widths for grid - column dendrogram at bottom now
        if cluster_rows and cluster_cols:
            # Both row and column dendrograms
            heights = [0.85, 0.15]  # [main, col dendrogram] - flipped order
            widths = [0.15, 0.85]  # [row dendrogram, main]
            gs = GridSpec(2, 2, height_ratios=heights, width_ratios=widths)
            main_ax = fig.add_subplot(gs[0, 1])
            row_dendrogram_ax = fig.add_subplot(gs[0, 0])
            col_dendrogram_ax = fig.add_subplot(gs[1, 1])  # Now at bottom
            dendrogram_axes["row"] = row_dendrogram_ax
            dendrogram_axes["col"] = col_dendrogram_ax
        elif cluster_rows:
            # Only row dendrogram
            widths = [0.15, 0.85]  # [row dendrogram, main]
            gs = GridSpec(1, 2, width_ratios=widths)
            main_ax = fig.add_subplot(gs[0, 1])
            row_dendrogram_ax = fig.add_subplot(gs[0, 0])
            dendrogram_axes["row"] = row_dendrogram_ax
        elif cluster_cols:
            # Only column dendrogram
            heights = [0.85, 0.15]  # [main, col dendrogram] - flipped order
            gs = GridSpec(2, 1, height_ratios=heights)
            main_ax = fig.add_subplot(gs[0, 0])
            col_dendrogram_ax = fig.add_subplot(gs[1, 0])  # Now at bottom
            dendrogram_axes["col"] = col_dendrogram_ax
        else:
            # No dendrograms (shouldn't reach here if dendrogram is True)
            gs = GridSpec(1, 1)
            main_ax = fig.add_subplot(gs[0, 0])
    else:
        # No dendrograms
        if ax is None:
            main_ax = fig.add_subplot(111)
        else:
            main_ax = ax
            fig = ax.figure

    # Set ax to main_ax for the rest of the function
    ax = main_ax

    # Draw dendrograms if enabled
    if dendrogram:
        if cluster_rows and "row" in dendrogram_axes:
            # Plot row dendrogram
            scipy_dendrogram(
                row_linkage,
                ax=dendrogram_axes["row"],
                orientation="left",
                no_labels=True,
                color_threshold=0,
                link_color_func=lambda x: dendrogram_color,  # Use black by default
            )
            # Remove axes from dendrogram
            dendrogram_axes["row"].axis("off")

        if cluster_cols and "col" in dendrogram_axes:
            # Plot column dendrogram (now at bottom)
            scipy_dendrogram(
                col_linkage,
                ax=dendrogram_axes["col"],
                orientation="bottom",
                no_labels=True,
                color_threshold=0,
                link_color_func=lambda x: dendrogram_color,  # Use black by default
            )
            # Remove axes from dendrogram
            dendrogram_axes["col"].axis("off")

    # Set up the axes for the heatmap cells
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)

    # Remove frame and spines
    ax.set_frame_on(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Force square cells by setting aspect ratio
    ax.set_aspect("equal")

    # Determine min/max values for colormap normalization
    if vmin is None or vmax is None:
        all_values = []
        for group_data in condition_values.values():
            for gene_data in group_data.values():
                all_values.extend(list(gene_data.values()))

        if vmin is None:
            vmin = np.nanmin(all_values)
        if vmax is None:
            vmax = np.nanmax(all_values)

    # Create a colormap instance
    cmap_obj = plt.cm.get_cmap(cmap) if isinstance(cmap, str) else cmap

    # Create a grid of diagonally split cells
    groups = list(condition_values.keys())

    # Apply row ordering from dendrogram if available
    ordered_var_names = [var_names[i] for i in row_order]

    # Apply column ordering from dendrogram if available
    ordered_groups = [groups[i] for i in col_order]

    for plot_row_idx, gene in enumerate(ordered_var_names):
        for plot_col_idx, group in enumerate(ordered_groups):
            if gene in condition_values[group]:
                gene_data = condition_values[group][gene]
                val1 = gene_data[str(condition1_name)]
                val2 = gene_data[str(condition2_name)]

                # Draw diagonal split cell with no edgecolor
                # Check for None or NaN values that might cause black triangles
                if val1 is None or (isinstance(val1, (int, float)) and np.isnan(val1)):
                    logger.debug(
                        f"Found NaN or None for lower triangle: {group}, {gene}, {condition1_name}"
                    )

                if val2 is None or (isinstance(val2, (int, float)) and np.isnan(val2)):
                    logger.debug(
                        f"Found NaN or None for upper triangle: {group}, {gene}, {condition2_name}"
                    )

                _draw_diagonal_split_cell(
                    ax,
                    plot_col_idx,
                    n_rows - plot_row_idx - 1,
                    1,
                    1,
                    val1,
                    val2,
                    cmap_obj,
                    vmin,
                    vmax,
                    edgecolor="none",
                    linewidth=0,
                )

    # Set tick positions
    ax.set_xticks(np.arange(0.5, n_cols))
    ax.set_yticks(np.arange(0.5, n_rows))

    # Set tick labels
    if show_group_labels:
        ax.set_xticklabels(ordered_groups, rotation=90, fontsize=group_labels_size)
    else:
        ax.set_xticklabels([])

    if show_gene_labels:
        ax.set_yticklabels(ordered_var_names[::-1], fontsize=gene_labels_size)
    else:
        ax.set_yticklabels([])

    # Adjust grid display
    ax.grid(False)

    # Add axes labels
    ax.set_xlabel(groupby)
    ax.set_ylabel("Genes")

    # Add title if provided
    if title:
        ax.set_title(title)

    # We'll use a completely different approach with tight_layout disabled
    # First, finish setting up the main plot
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Leave less space on the right side

    # Create custom legend with triangle patches matching the heatmap cells
    legend_elements = []

    # Create an upper triangle patch for condition2
    upper_triangle = mpatches.Polygon(
        [[1, 0], [1, 1], [0, 1]],
        facecolor="white",
        edgecolor="black",
        label=f"{condition2_name} (upper right)",
    )
    upper_triangle.triangle_type = "upper"  # custom attribute
    legend_elements.append(upper_triangle)

    # Create a lower triangle patch for condition1
    lower_triangle = mpatches.Polygon(
        [[0, 0], [1, 0], [0, 1]],
        facecolor="white",
        edgecolor="black",
        label=f"{condition1_name} (lower left)",
    )
    lower_triangle.triangle_type = "lower"  # custom attribute
    legend_elements.append(lower_triangle)

    # Calculate positions for the right-side elements (in figure coordinates)
    right_side_width = 0.12  # Narrower sidebar
    right_side_left = 0.87  # Starting position of sidebar
    sidebar_height = 0.8  # Total height of sidebar area
    box_height = sidebar_height / 3  # Height of each box

    # Top box for legend
    legend_ax = fig.add_axes([right_side_left, 0.7, right_side_width, box_height])
    legend_ax.axis("off")

    # Custom legend handler to draw triangles without indexing colors
    from matplotlib.legend_handler import HandlerPatch

    class HandlerTriangle(HandlerPatch):
        def create_artists(
            self,
            legend,
            orig_handle,
            xdescent,
            ydescent,
            width,
            height,
            fontsize,
            trans,
        ):
            # Choose vertices based on custom triangle_type attribute
            if getattr(orig_handle, "triangle_type", "lower") == "lower":
                verts = [
                    [xdescent, ydescent],
                    [xdescent + width, ydescent],
                    [xdescent, ydescent + height],
                ]
            else:  # "upper"
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

    # Add the legend to the top box using our custom handler for Polygon objects
    legend = legend_ax.legend(
        handles=legend_elements,
        loc="center",
        title="Conditions",
        frameon=False,
        handler_map={mpatches.Polygon: HandlerTriangle()},
    )
    # Middle box for colorbar - make it narrower and centered
    colorbar_width = 0.02  # Narrower colorbar
    # Center the colorbar horizontally in its box
    colorbar_left = right_side_left + (right_side_width - colorbar_width) / 2
    cax = fig.add_axes([colorbar_left, 0.4, colorbar_width, box_height])

    # Create the colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.outline.set_visible(False)  # Remove colorbar outline

    # Remove grid from colorbar
    cbar.ax.grid(False)

    # Third box is left empty (bottom section)

    # Set colorbar label based on whether data was z-scored
    if standard_scale == "var" or standard_scale == 0:
        cbar.set_label(
            "Z-score" if colorbar_title == "Expression" else colorbar_title, labelpad=10
        )
    else:
        cbar.set_label(colorbar_title, labelpad=10)

    # Save figure if path provided
    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight")

    # Return figure and axes if requested
    if return_fig:
        if dendrogram and len(dendrogram_axes) > 0:
            return fig, ax, dendrogram_axes
        else:
            return fig, ax
    # Don't automatically show the figure


def heatmap(
    adata: AnnData,
    var_names: Optional[Union[List[str], Sequence[str]]] = None,
    groupby: str = None,
    n_top_genes: int = 20,
    gene_list: Optional[Union[List[str], Sequence[str]]] = None,
    lfc_key: Optional[str] = None,
    score_key: str = "kompot_de_mahalanobis",
    layer: Optional[str] = None,
    standard_scale: Optional[Union[str, int]] = "var",  # Default to gene-wise z-scoring
    cmap: Union[str, mcolors.Colormap] = "viridis",
    dendrogram: bool = False,  # Whether to show dendrograms
    cluster_rows: bool = True,  # Whether to cluster rows
    cluster_cols: bool = True,  # Whether to cluster columns
    dendrogram_color: str = "black",  # Default dendrogram color
    swap_axes: bool = False,
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
    split_by_condition: bool = False,
    condition_column: Optional[str] = None,
    observed: bool = True,
    diagonal_split: bool = True,
    condition1_name: Optional[str] = None,
    condition2_name: Optional[str] = None,
    exclude_groups: Optional[Union[str, List[str]]] = None,
    **kwargs,
) -> Union[
    None,
    Tuple[plt.Figure, plt.Axes, Optional[plt.Axes]],
    Tuple[plt.Figure, plt.Axes, Dict[str, plt.Axes]],
]:
    """
    Create a heatmap of gene expression from Kompot differential expression results.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing expression data
    var_names : list, optional
        List of genes to include in the heatmap. If None, will use top genes
        based on score_key and lfc_key. This parameter is kept for backwards compatibility,
        gene_list is the preferred way to specify genes directly.
    groupby : str, optional
        Key in adata.obs for grouping cells. If None, no grouping is performed
    n_top_genes : int, optional
        Number of top genes to include if var_names and gene_list are None
    gene_list : list, optional
        Explicit list of genes to include in the heatmap. Takes precedence over var_names if both are provided.
    lfc_key : str, optional
        Key in adata.var for log fold change values.
        If None, will try to infer from kompot_de_ keys.
    score_key : str, optional
        Key in adata.var for significance scores.
        Default is "kompot_de_mahalanobis"
    layer : str, optional
        Layer in AnnData to use for expression values. If None, uses .X
    standard_scale : str or int, optional
        Whether to scale the expression values ('var', 'group' or 0 for rows, 1 for columns)
        Default is 'var' for gene-wise z-scoring
    cmap : str or colormap, optional
        Colormap to use for the heatmap
    dendrogram : bool, optional
        Whether to show dendrograms for clustering. Default is False.
    cluster_rows : bool, optional
        Whether to cluster rows (genes). Default is True.
    cluster_cols : bool, optional
        Whether to cluster columns (groups). Default is True.
    dendrogram_color : str, optional
        Color for dendrograms. Default is "black".
    swap_axes : bool, optional
        Whether to swap the axes (genes as columns, groups as rows)
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
        Negative indices count from the end (-1 is the latest run). If None,
        uses the latest run information.
    split_by_condition : bool, optional
        Whether to split each group by condition (e.g., Young/Old) to show contrast.
        Requires condition_column to be provided.
    condition_column : str, optional
        Column in adata.obs containing condition information for splitting.
        If None, tries to infer from run_info.
    observed : bool, optional
        Whether to use only observed combinations in groupby operations.
        Default is True, which improves performance with categorical data.
    diagonal_split : bool, optional
        Whether to use diagonal split cells to show two conditions in each tile.
        Default is True. If True, redirects to diagonal_split_heatmap function.
    condition1_name : str, optional
        Display name for condition 1 in diagonal split (lower-left triangle).
    condition2_name : str, optional
        Display name for condition 2 in diagonal split (upper-right triangle).
    exclude_groups : str or list, optional
        Group name(s) to exclude from the heatmap. Can be a single group name or
        a list of group names. Only applies when groupby is not None.
    **kwargs :
        Additional parameters passed to sns.heatmap or sns.clustermap

    Returns
    -------
    If return_fig is True, returns (fig, main_ax, [dendrogram_ax])
    """
    # Get run info for parameter extraction
    run_info = get_run_from_history(adata, run_id, analysis_type="de")

    # Extract parameters from run_info (only if not explicitly provided)
    if run_info is not None and "params" in run_info:
        params = run_info["params"]

        # Extract condition column if needed
        if (split_by_condition or diagonal_split) and condition_column is None:
            # In Kompot DE runs, the condition column is called "groupby"
            if "groupby" in params:
                condition_key = params["groupby"]
                condition_column = condition_key
                logger.info(
                    f"Using condition_column '{condition_column}' from run information"
                )

        # Extract layer if not explicitly provided
        if layer is None and "layer" in params:
            layer = params["layer"]
            logger.info(f"Using layer '{layer}' from run information")

        # Extract condition names if not provided
        if condition1_name is None and "condition1" in params:
            condition1_name = params["condition1"]
        if condition2_name is None and "condition2" in params:
            condition2_name = params["condition2"]

    # Check if diagonal split is requested
    if diagonal_split:
        return diagonal_split_heatmap(
            adata=adata,
            var_names=var_names,
            groupby=groupby,
            n_top_genes=n_top_genes,
            gene_list=gene_list,
            lfc_key=lfc_key,
            score_key=score_key,
            layer=layer,
            standard_scale=standard_scale,
            cmap=cmap,
            condition1_name=condition1_name,
            condition2_name=condition2_name,
            figsize=figsize,
            show_gene_labels=show_gene_labels,
            show_group_labels=show_group_labels,
            gene_labels_size=gene_labels_size,
            group_labels_size=group_labels_size,
            colorbar_title=colorbar_title,
            colorbar_kwargs=colorbar_kwargs,
            title=title,
            sort_genes=sort_genes,
            center=center,
            vmin=vmin,
            vmax=vmax,
            ax=ax,
            return_fig=return_fig,
            save=save,
            run_id=run_id,
            condition_column=condition_column,
            observed=observed,
            dendrogram=dendrogram,  # Only show dendrograms if requested
            cluster_rows=cluster_rows,  # Always use provided cluster_rows value
            cluster_cols=cluster_cols,  # Always use provided cluster_cols value
            dendrogram_color=dendrogram_color,  # Pass the dendrogram color
            exclude_groups=exclude_groups,  # Pass the exclude_groups parameter
            **kwargs,
        )

    # Set default colorbar kwargs
    colorbar_kwargs = colorbar_kwargs or {}

    # Set defaults for grid removal
    if "linewidths" not in kwargs:
        kwargs["linewidths"] = 0

    # Initialize run_info variable
    run_info = None

    # If gene_list is provided, use it directly
    if gene_list is not None:
        var_names = gene_list
        logger.info(f"Using provided gene_list with {len(gene_list)} genes/features")

        # We still need run info for condition names and other parameters
        run_info = get_run_from_history(adata, run_id, analysis_type="de")
    # If var_names not provided and no gene_list, get top genes based on DE results
    elif var_names is None:
        # Infer keys using the helper function
        lfc_key, score_key = _infer_heatmap_keys(adata, run_id, lfc_key, score_key)

        # Calculate the actual (positive) run ID for logging
        actual_run_id = None
        if run_id is not None:
            if run_id < 0:
                if "kompot_de" in adata.uns and "run_history" in adata.uns["kompot_de"]:
                    actual_run_id = len(adata.uns["kompot_de"]["run_history"]) + run_id
                else:
                    actual_run_id = run_id
            else:
                actual_run_id = run_id

        # Get condition information from the run specified by run_id
        condition1 = None
        condition2 = None
        run_info = get_run_from_history(adata, run_id, analysis_type="de")

        if run_info is not None and "params" in run_info:
            params = run_info["params"]
            if "conditions" in params and len(params["conditions"]) == 2:
                condition1 = params["conditions"][0]
                condition2 = params["conditions"][1]

        # Try to extract from key name if still not found
        if (condition1 is None or condition2 is None) and lfc_key is not None:
            conditions = _extract_conditions_from_key(lfc_key)
            if conditions:
                condition1, condition2 = conditions

        # Log which run is being used
        if run_id is not None:
            conditions_str = (
                f": comparing {condition1} vs {condition2}"
                if condition1 and condition2
                else ""
            )
            logger.info(f"Using DE run {actual_run_id} for heatmap{conditions_str}")
        else:
            logger.info("Using latest available DE run for heatmap")

        # Log the fields being used
        logger.info(
            f"Using fields for heatmap - lfc_key: '{lfc_key}', score_key: '{score_key}'"
        )

        # Get top genes based on score
        de_data = pd.DataFrame(
            {
                "gene": adata.var_names,
                "lfc": (
                    adata.var[lfc_key]
                    if lfc_key in adata.var
                    else np.zeros(adata.n_vars)
                ),
                "score": (
                    adata.var[score_key]
                    if score_key in adata.var
                    else np.zeros(adata.n_vars)
                ),
            }
        )

        if sort_genes:
            de_data = de_data.sort_values("score", ascending=False)

        # Get top genes
        var_names = de_data.head(n_top_genes)["gene"].tolist()

    # Log the plot type first
    logger.info(f"Creating heatmap with {len(var_names)} genes/features")

    # Log the data sources being used for the heatmap
    if layer is not None and layer in adata.layers:
        logger.info(f"Using expression data from layer: '{layer}'")
        expr_matrix = (
            adata[:, var_names].layers[layer].toarray()
            if hasattr(adata.layers[layer], "toarray")
            else adata[:, var_names].layers[layer]
        )
    else:
        if layer is not None:
            logger.warning(
                f"Requested layer '{layer}' not found, falling back to adata.X"
            )
        logger.info(f"Using expression data from adata.X")
        expr_matrix = (
            adata[:, var_names].X.toarray()
            if hasattr(adata.X, "toarray")
            else adata[:, var_names].X
        )

    # Create dataframe with expression data
    expr_df = pd.DataFrame(expr_matrix, index=adata.obs_names, columns=var_names)

    # If condition splitting is requested, check that condition_column is provided
    if split_by_condition and condition_column is None:
        logger.warning(
            "split_by_condition=True but no condition_column provided. Disabling split_by_condition."
        )
        split_by_condition = False

    # Add condition information to expr_df if splitting by condition
    if split_by_condition and condition_column in adata.obs:
        expr_df[condition_column] = adata.obs[condition_column].values

    # Group by condition if provided
    if groupby is not None and groupby in adata.obs:
        # Calculate mean expression per group, optionally splitting by condition
        logger.info(
            f"Grouping expression by '{groupby}' ({adata.obs[groupby].nunique()} groups)"
        )

        # Add groupby column to expr_df (regardless of split_by_condition)
        expr_df[groupby] = adata.obs[groupby].values

        # Handle group exclusion if specified
        if exclude_groups is not None:
            # Convert single group to list for consistent handling
            if isinstance(exclude_groups, str):
                exclude_groups = [exclude_groups]

            # Check for non-existent groups
            available_groups = adata.obs[groupby].unique()
            non_existent = [g for g in exclude_groups if g not in available_groups]
            if non_existent:
                logger.warning(
                    f"Some groups in exclude_groups do not exist: {', '.join(non_existent)}"
                )
                # Filter to only include existing groups
                exclude_groups = [g for g in exclude_groups if g in available_groups]

            # Filter out the excluded groups
            original_size = len(expr_df)
            expr_df = expr_df[~expr_df[groupby].isin(exclude_groups)]
            filtered_size = len(expr_df)

            if filtered_size < original_size:
                logger.info(
                    f"Excluded {original_size - filtered_size} cells from groups: {', '.join(exclude_groups)}"
                )

                # Check if we have any remaining cells
                if filtered_size == 0:
                    raise ValueError(
                        f"All cells were excluded after filtering out groups: {', '.join(exclude_groups)}. "
                        f"Please check your exclude_groups parameter."
                    )

        if split_by_condition and condition_column in adata.obs:
            logger.info(f"Splitting each group by '{condition_column}'")
            # Group by both groupby and condition_column
            grouped_expr = expr_df.groupby(
                [groupby, condition_column], observed=observed
            ).mean()
            # Reshape the MultiIndex DataFrame to have condition-specific columns
            # This creates a hierarchical column structure with gene names and conditions
            grouped_expr = grouped_expr.unstack(level=1)
        else:
            # Group by just groupby
            grouped_expr = expr_df.groupby(groupby, observed=observed).mean()
    else:
        logger.info(f"No grouping applied (showing all {len(expr_df)} cells)")
        grouped_expr = expr_df

    # Scale data if requested
    if standard_scale == "group" or standard_scale == 1:
        # Scale by group (cols)
        logger.info("Applying group-wise z-scoring (standard_scale='group')")
        if split_by_condition:
            # Scale each gene separately across all conditions
            for gene in var_names:
                gene_data = grouped_expr.loc[:, gene]
                # Handle NaN values and zeros
                gene_mean = gene_data.mean(skipna=True)
                gene_std = gene_data.std(skipna=True)
                if gene_std > 0:
                    grouped_expr.loc[:, gene] = (gene_data - gene_mean) / gene_std
                # NaN values remain NaN
        else:
            # Properly handle NaN values
            means = grouped_expr.mean(axis=0, skipna=True)
            stds = grouped_expr.std(axis=0, skipna=True)
            # Avoid division by zero
            stds = stds.replace(0, 1)
            # Apply z-scoring
            grouped_expr = (grouped_expr - means) / stds

    elif standard_scale == "var" or standard_scale == 0:
        # Scale by gene (rows)
        logger.info("Applying gene-wise z-scoring (standard_scale='var')")
        if split_by_condition:
            # Handle hierarchical columns by scaling each group separately
            for group_idx in grouped_expr.index:
                group_data = grouped_expr.loc[group_idx]
                # Handle NaN values and zeros
                group_mean = group_data.mean(skipna=True)
                group_std = group_data.std(skipna=True)
                if group_std > 0:
                    grouped_expr.loc[group_idx] = (group_data - group_mean) / group_std
                # NaN values remain NaN
        else:
            # Transpose, z-score each column (which is a gene), then transpose back
            data_T = grouped_expr.T

            # Properly handle NaN values
            means = data_T.mean(axis=0, skipna=True)
            stds = data_T.std(axis=0, skipna=True)
            # Avoid division by zero
            stds = stds.replace(0, 1)
            # Apply z-scoring
            data_T_zscore = (data_T - means) / stds

            # Transpose back
            grouped_expr = data_T_zscore.T

    # Swap axes if requested
    if swap_axes:
        grouped_expr = grouped_expr.T

    # Only create a new figure if dendrogram=False, or if ax is provided
    # For dendrogram=True, clustermap will create its own figure
    if not dendrogram and ax is None:
        # Calculate appropriate figsize if not provided
        if figsize is None:
            # Get dimensions
            n_rows, n_cols = grouped_expr.shape

            # Calculate cell size that ensures square tiles
            base_size = 0.5  # Base cell size

            # Calculate axes dimensions to ensure square cells
            width_inches = 6 + n_cols * base_size  # Base width for labels and margins
            height_inches = 6 + n_rows * base_size  # Base height for labels and margins

            # Set the figure size
            figsize = (width_inches, height_inches)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
    elif ax is not None:
        fig = ax.figure

    # Create the heatmap
    dendrogram_ax = None
    if dendrogram:
        logger.warning(
            "Dendrogram option is not currently supported. Using standard heatmap instead."
        )

    # Get the data matrix
    data = grouped_expr.values
    rows = grouped_expr.index
    cols = grouped_expr.columns

    # Create figure and axes if not provided
    if ax is None:
        # Calculate appropriate figsize if not provided
        if figsize is None:
            # Calculate aspect ratio that ensures square tiles
            n_rows, n_cols = data.shape
            # Base width includes space for labels and legend
            base_width = 8  # Base width for labels and margins
            base_height = 6  # Base height for labels and margins

            # Calculate cell size that ensures square tiles
            cell_size = min(0.5, 12 / max(n_rows, n_cols))

            # Calculate figsize to maintain square cells
            figsize = (
                base_width + n_cols * cell_size,
                base_height + n_rows * cell_size,
            )

        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Determine color normalization
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

    # Get colormap
    cmap_obj = plt.cm.get_cmap(cmap) if isinstance(cmap, str) else cmap

    # Create the heatmap using imshow
    im = ax.imshow(
        data,
        aspect="equal",  # Use equal aspect to ensure square cells
        cmap=cmap_obj,
        norm=norm,
        origin="upper",
        **{k: v for k, v in kwargs.items() if k not in ["linewidths", "linecolor"]},
    )

    # Force square cells by setting aspect ratio
    ax.set_aspect("equal")

    # Configure axes - remove frame and spines
    ax.set_frame_on(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)

    # Set tick positions
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(rows)))

    # Set tick labels
    if show_group_labels if not swap_axes else show_gene_labels:
        ax.set_xticklabels(
            cols,
            rotation=90,
            fontsize=group_labels_size if not swap_axes else gene_labels_size,
        )
    else:
        ax.set_xticklabels([])

    if show_gene_labels if not swap_axes else show_group_labels:
        ax.set_yticklabels(
            rows, fontsize=gene_labels_size if not swap_axes else group_labels_size
        )
    else:
        ax.set_yticklabels([])

    # Create a grid layout for the colorbar
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)

    # Create the colorbar to the right of the main axis
    cax = divider.append_axes("right", size="5%", pad=0.5)
    cax.set_frame_on(False)

    # Create the colorbar vertically
    default_cbar_kwargs = {"cax": cax, "orientation": "vertical"}
    combined_cbar_kwargs = {**default_cbar_kwargs, **(colorbar_kwargs or {})}
    cbar = plt.colorbar(im, **combined_cbar_kwargs)
    cbar.outline.set_visible(False)  # Remove colorbar outline
    cbar.ax.tick_params(grid_alpha=0)  # Remove grid lines from colorbar
    cbar.set_label(colorbar_title, labelpad=10)  # Add padding to the colorbar label

    # Set title if provided
    if title:
        ax.set_title(title)

    # Adjust label sizes
    if swap_axes:
        if show_gene_labels:
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=gene_labels_size)
        if show_group_labels:
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=group_labels_size)
    else:
        if show_gene_labels:
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=gene_labels_size)
        if show_group_labels:
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=group_labels_size)

    # Save figure if path provided
    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight")

    # Return figure and axes if requested
    if return_fig:
        return fig, ax, dendrogram_ax
    # Don't automatically show the figure
