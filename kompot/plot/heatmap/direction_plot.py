"""Direction bar plotting functions."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union, List, Tuple, Dict, Any, Sequence, Literal, Callable, Set
from anndata import AnnData
import pandas as pd
import logging

from ...utils import get_run_from_history, KOMPOT_COLORS
from ..volcano import _extract_conditions_from_key

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
    """Create a barplot showing the direction of change distribution across categories.

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
        How to normalize the data. Options: "index" (normalize rows), 
        "columns" (normalize columns), or None (raw counts).
    figsize : tuple, optional
        Figure size as (width, height) in inches
    title : str, optional
        Plot title. If None and conditions provided, uses "Direction of Change by {category_column}\\n{condition1} vs {condition2}"
    xlabel : str, optional
        Label for x-axis. If None, uses the category_column
    ylabel : str, optional
        Label for y-axis. Defaults to "Percentage (%)" when normalize="index", otherwise "Count"
    colors : dict, optional
        Dictionary mapping direction values to colors.
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
    
    Returns
    -------
    tuple or None
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

    # Check that both columns exist in the data
    if category_column not in adata.obs.columns:
        raise ValueError(f"Category column '{category_column}' not found in adata.obs")
    if direction_column not in adata.obs.columns:
        raise ValueError(f"Direction column '{direction_column}' not found in adata.obs")

    # Create a crosstab of category vs direction
    crosstab = pd.crosstab(
        adata.obs[category_column], adata.obs[direction_column], normalize=normalize
    )

    # Convert to percentages if normalized by index
    if normalize == "index":
        crosstab = crosstab * 100

    # Reorder columns to have neutral at the top of the stack
    if "neutral" in crosstab.columns and len(crosstab.columns) > 1:
        # Get all columns except neutral
        other_cols = [col for col in crosstab.columns if col != "neutral"]
        # Create new column order with neutral last (appears at top in stacked plot)
        new_col_order = other_cols + ["neutral"]
        # Reorder the dataframe columns
        crosstab = crosstab[new_col_order]

    # Sort by a specific direction if requested
    if sort_by is not None and sort_by in crosstab.columns:
        crosstab = crosstab.sort_values(by=sort_by, ascending=ascending)

    # Create figure if no axes provided
    create_fig = ax is None
    if create_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Create the plot
    crosstab.plot(
        kind="bar",
        stacked=stacked,
        color=[colors.get(c, f"C{i}") for i, c in enumerate(crosstab.columns)],
        ax=ax,
        rot=rotation,
        **kwargs,
    )

    # Remove grid lines
    ax.grid(False)

    # Set title if provided or can be inferred
    if title is None and condition1 and condition2:
        title = f"Direction of Change by {category_column}\n{condition1} vs {condition2}"
    if title is not None:
        ax.set_title(title)

    # Set axis labels
    ax.set_xlabel(xlabel or category_column)
    ax.set_ylabel(ylabel)

    # Set legend with no box and outside the plot
    # Also reverse the order of elements (up first)
    handles, labels = ax.get_legend_handles_labels()
    order = []
    if "up" in labels:
        order.append(labels.index("up"))
    if "neutral" in labels:
        order.append(labels.index("neutral"))
    if "down" in labels:
        order.append(labels.index("down"))
    if not order:  # If none of the expected labels found, keep original order
        order = list(range(len(labels)))
    
    ax.legend(
        [handles[i] for i in order], 
        [labels[i] for i in order], 
        title=legend_title, 
        loc='center left', 
        bbox_to_anchor=(1.0, 0.5),
        frameon=False
    )

    # Adjust layout
    plt.tight_layout()

    # Save figure if requested
    if save is not None:
        plt.savefig(save, bbox_inches="tight", dpi=300)

    # Return figure if requested
    if return_fig:
        return fig, ax

    # Show plot if creating new figure and not returning
    if create_fig and not return_fig:
        plt.show()

    return None