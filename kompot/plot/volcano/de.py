"""Volcano plot functions for differential expression."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Colormap, ListedColormap
from typing import Optional, Union, List, Tuple, Dict, Any
from anndata import AnnData
import pandas as pd
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
    n_top_genes: Optional[int] = None,
    highlight_genes: Optional[Union[List[str], Dict[str, str], List[Dict[str, Any]]]] = None,
    background_color_key: Optional[str] = None,
    background_cmap: Union[str, Colormap] = None,  # Will be auto-selected based on data type
    color_discrete_map: Optional[Dict[str, str]] = None,
    vmin: Optional[Union[float, str]] = None,
    vmax: Optional[Union[float, str]] = None,
    vcenter: Optional[float] = None,
    gene_labels: Union[bool, int, List[str], Dict[str, str]] = 10,
    figsize: Tuple[float, float] = (10, 8),
    title: Optional[str] = None,
    xlabel: Optional[str] = "Log Fold Change",
    ylabel: Optional[str] = None,  # Now auto-inferred from y_axis_type
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
    # New significance-related parameters
    y_axis_type: str = "mahalanobis",  # "mahalanobis", "local_fdr", "tail_fdr", "log10_ptp", or custom column name
    significance_threshold: Optional[Union[float, Dict[str, float]]] = None,
    update_de_classification: bool = False,
    de_column: Optional[str] = None,
    show_thresholds: bool = True,
    **kwargs,
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
        If specified, highlight this number of top genes by score instead of using DE classification.
        Cannot be used together with significance_threshold. If not specified (None), will use
        DE classification from is_de column when available. Ignored if `highlight_genes` is provided.
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
    gene_labels : bool, int, list of str, or dict, optional
        Controls which genes get labeled with their names:
        - True: label all highlighted genes  
        - False: label no genes
        - int: label top N genes by score (default: 10)
        - list of str: label specific genes by name
        - dict: label genes with custom labels (gene_name -> custom_label)
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
        adata.var for Mahalanobis distances, and mean fold changes.
    y_axis_type : str, optional
        Type of values to use for the y-axis: "mahalanobis" (default), "local_fdr", "tail_fdr", 
        "ptp", or a custom column name from adata.var. When using FDR or ptp values, they are 
        -log10 transformed for display.
    significance_threshold : float or dict, optional
        Significance threshold for the y-axis values. Can be:
        - float: Single threshold for the current y_axis_type, shown as horizontal line
        - dict: Multiple thresholds with keys corresponding to y_axis_type values
                (e.g., {"local_fdr": 0.05, "ptp": 0.01}). Cells must pass ALL thresholds.
                No threshold line is drawn when using dict format.
        The interpretation depends on y_axis_type (or dict keys):
        - "mahalanobis": minimum distance for significance
        - "local_fdr"/"tail_fdr": maximum FDR for significance (e.g., 0.05)
        - "ptp": maximum p-value for significance (e.g., 0.01)
        - custom column: threshold applied to the raw column values
    update_de_classification : bool, optional
        Whether to update the differential expression classification column based on the new
        significance threshold. Applicable for FDR and ptp y_axis_types (default: False).
    de_column : str, optional
        Name of the differential expression boolean column to update if update_de_classification=True.
        If None, tries to infer from the score_key.
    show_thresholds : bool, optional
        Whether to show threshold lines on the plot (default: True).
    **kwargs :
        Additional parameters passed to plt.scatter

    Returns
    -------
    If return_fig is True, returns (fig, ax)
    """
    # Set default text and grid kwargs
    default_text_kwargs = {
        "ha": "left",
        "va": "bottom",
        "xytext": text_offset,
        "textcoords": "offset points",
    }
    text_kwargs = {**default_text_kwargs, **(text_kwargs or {})}
    grid_kwargs = grid_kwargs or {"alpha": 0.3}

    # Infer keys using helper function - this will get the right keys but won't do any logging
    lfc_key, score_key = _infer_de_keys(adata, run_id, lfc_key, score_key)

    # Get run info early since we'll need it for multiple purposes
    run_info = get_run_from_history(adata, run_id, analysis_type="de")

    # Parameter validation
    if n_top_genes is not None and significance_threshold is not None:
        raise ValueError(
            "Cannot specify both 'n_top_genes' and 'significance_threshold'. "
            "Use 'n_top_genes' for top gene highlighting or 'significance_threshold' for threshold-based highlighting."
        )

    # Handle various y-axis options
    original_score_key = score_key
    significance_key = None

    def fdr_y_transform(y):
        """Transform FDR values using -log10."""
        return -np.log10(np.maximum(y, 1e-300))  # Avoid log(0)

    y_transform = None
    
    if y_axis_type in ["local_fdr", "tail_fdr"]:
        # FDR-based y-axis
        if run_info and "fdr_keys" in run_info and run_info["fdr_keys"]:
            # Get the appropriate FDR key from run info
            if y_axis_type == "local_fdr":
                significance_key = run_info["fdr_keys"].get("local_fdr_key")
            else:  # tail_fdr
                significance_key = run_info["fdr_keys"].get("tail_fdr_key")
            
            # Check if the FDR key exists in the data
            if significance_key and significance_key in adata.var.columns:
                score_key = significance_key
                y_transform = fdr_y_transform
                logger.info(f"Using {y_axis_type} values for y-axis: {significance_key}")
            elif significance_key:
                logger.warning(
                    f"FDR key '{significance_key}' from run info not found in adata.var. "
                    f"Available FDR columns: {[col for col in adata.var.columns if 'fdr' in col.lower() or 'pvalue' in col.lower()]}"
                )
                significance_key = None
            else:
                logger.warning(f"No {y_axis_type} key found in run info fdr_keys")
        else:
            logger.warning("No FDR keys found in run info or FDR was not computed for this run")
        
        # Fallback to string replacement if run info approach fails
        if significance_key is None and score_key and "mahalanobis" in score_key:
            logger.info("Attempting fallback FDR key inference from score key...")
            if y_axis_type == "local_fdr":
                fallback_key = score_key.replace("mahalanobis", "mahalanobis_local_fdr")
            else:  # tail_fdr  
                fallback_key = score_key.replace("mahalanobis", "mahalanobis_tail_fdr")
            
            if fallback_key in adata.var.columns:
                score_key = fallback_key
                significance_key = fallback_key
                y_transform = fdr_y_transform
                logger.info(f"Using fallback {y_axis_type} key: {fallback_key}")
            else:
                logger.warning(f"Fallback FDR key '{fallback_key}' not found either")
        
        # Final fallback to original score key if nothing worked
        if significance_key is None and y_transform is None:
            logger.info(f"Using original score key: {original_score_key}")
            score_key = original_score_key
            
    elif y_axis_type == "ptp":
        # Posterior tail probability (will be -log10 transformed for display)
        if run_info and "ptp_key" in run_info and run_info["ptp_key"]:
            significance_key = run_info["ptp_key"]
            
            if significance_key and significance_key in adata.var.columns:
                score_key = significance_key
                y_transform = fdr_y_transform  # Same -log10 transform as FDR
                logger.info(f"Using ptp values for y-axis: {significance_key}")
            else:
                logger.warning(f"ptp key '{significance_key}' from run info not found in adata.var")
                significance_key = None
        
        # Fallback to string replacement if run info approach fails
        if significance_key is None and score_key and "mahalanobis" in score_key:
            logger.info("Attempting fallback ptp key inference from score key...")
            fallback_key = score_key.replace("mahalanobis", "ptp")
            
            if fallback_key in adata.var.columns:
                score_key = fallback_key
                significance_key = fallback_key
                y_transform = fdr_y_transform  # Same -log10 transform as FDR
                logger.info(f"Using fallback ptp key: {fallback_key}")
            else:
                logger.warning(f"Fallback ptp key '{fallback_key}' not found either")
        
        # Final fallback to original score key if nothing worked
        if significance_key is None:
            logger.info(f"Using original score key: {original_score_key}")
            score_key = original_score_key
            
    elif y_axis_type not in ["mahalanobis"]:
        # Custom column name - check if it exists directly in adata.var
        if y_axis_type in adata.var.columns:
            score_key = y_axis_type
            significance_key = y_axis_type
            logger.info(f"Using custom column '{y_axis_type}' for y-axis")
        else:
            logger.warning(
                f"Custom y_axis_type '{y_axis_type}' not found in adata.var.columns. "
                f"Falling back to original score_key: {original_score_key}"
            )
            score_key = original_score_key

    # Auto-infer ylabel if not provided
    if ylabel is None:
        if y_axis_type == "local_fdr" and y_transform is not None:
            ylabel = "-log10(Local FDR)"
        elif y_axis_type == "tail_fdr" and y_transform is not None:
            ylabel = "-log10(Tail FDR)"
        elif y_axis_type == "ptp" and y_transform is not None:
            ylabel = "-log10(Posterior Tail Probability)"
        elif y_axis_type == "mahalanobis" or (score_key and "mahalanobis" in score_key.lower()):
            ylabel = "Mahalanobis Distance"
        else:
            ylabel = score_key

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
            # If not in key, try getting from run info (already retrieved above)
            if run_info is not None and "params" in run_info:
                params = run_info["params"]
                if "condition1" in params and "condition2" in params:
                    condition1 = params["condition1"]
                    condition2 = params["condition2"]

    # Log which run and fields are being used
    conditions_str = (
        f": comparing {condition1} to {condition2}" if condition1 and condition2 else ""
    )
    logger.info(f"Using DE run {actual_run_id}{conditions_str}")

    # Update axis labels
    if condition1 and condition2 and xlabel == "Log Fold Change":
        # Adjust for new key format where condition1 is the baseline/denominator
        xlabel = f"Log Fold Change: {condition1} to {condition2}"

    # Create figure if ax not provided - adjust figsize if legend is outside
    if ax is None:
        # If legend is outside and not explicitly placed elsewhere, adjust figsize
        if show_legend and (legend_loc == "best" or legend_loc == "center left"):
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
        # Use run information already retrieved above to access field names
        if not run_info or "field_names" not in run_info:
            logger.warning(
                "Cannot find run information for group-specific data. Make sure you're using the correct run_id."
            )
            return

        # Get varm keys directly from run information
        if "varm_keys" not in run_info:
            logger.warning(
                "No varm keys found in run information. This run may not have used groups."
            )
            return
            
        lfc_varm_key = run_info['varm_keys']['mean_lfc']
        score_varm_key = run_info['varm_keys']['mahalanobis']

        logger.debug(f"Using varm keys: lfc={lfc_varm_key}, score={score_varm_key}")
        
        # Check if the keys exist in varm and group is available
        lfc_data_available = (
            lfc_varm_key in adata.varm and group in adata.varm[lfc_varm_key].columns
        )

        score_data_available = (
            score_varm_key in adata.varm and group in adata.varm[score_varm_key].columns
        )
        
        
        if lfc_data_available and score_data_available:
            logger.info(f"Using group-specific data for group '{group}' from varm")
            logger.info(
                f"Using fields for DE plot - lfc_key: '{lfc_varm_key}', score_key: '{score_varm_key}'"
            )
            x = adata.varm[lfc_varm_key][group].values
            y = adata.varm[score_varm_key][group].values

            # Apply y-axis transformation if needed (for FDR values)
            if y_transform is not None:
                # Note: Group-specific FDR data might not be available in varm
                # For now, we'll warn users if they try to use FDR with groups
                if y_axis_type in ["local_fdr", "tail_fdr"]:
                    logger.warning(
                        f"FDR y-axis options ({y_axis_type}) with group-specific data may not work as expected. "
                        f"Group-specific FDR values are typically not stored in adata.varm."
                    )
                else:
                    y = y_transform(y)
                    logger.info(
                        f"Applied {y_axis_type} transformation to group-specific y-axis data"
                    )

            
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
                logger.warning(
                    f"Group-specific data for '{group}' not found in varm. Missing: {', '.join(missing_keys)}. Available groups: {group_str}. Falling back to default data."
                )
            else:
                logger.warning(
                    f"Group-specific data for '{group}' not found in varm. Missing: {', '.join(missing_keys)}. No groups available. Falling back to default data."
                )

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

    # Apply y-axis transformation if needed (for FDR values)
    if y is not None and y_transform is not None:
        y = y_transform(y)
        logger.info(f"Applied {y_axis_type} transformation to y-axis data")

    # Determine DE column for potential use
    inferred_de_column = None
    if de_column is None:
        # First try to get DE column from run info
        if run_info and "fdr_keys" in run_info and run_info["fdr_keys"]:
            inferred_de_column = run_info["fdr_keys"].get("is_de_key")
            logger.debug(f"Found is_de key from run info: {inferred_de_column}")
        
        # Fallback to string manipulation if run info doesn't have it
        if inferred_de_column is None:
            if significance_key and "mahalanobis" in significance_key:
                inferred_de_column = significance_key.replace("mahalanobis_local_fdr", "is_de").replace(
                    "mahalanobis_tail_fdr", "is_de"
                )
            elif original_score_key and "mahalanobis" in original_score_key:
                inferred_de_column = original_score_key.replace("mahalanobis", "is_de")
    else:
        inferred_de_column = de_column

    # Note: Removed automatic background coloring for DE columns
    # DE highlighting is now handled consistently through the highlight_groups system below

    # Create a DataFrame with all relevant information
    data_dict = {"gene": adata.var_names, "lfc": x, "score": y}

    # Add sort_val - either from the specified sort_key or use y (score) by default
    if sort_key is not None:
        # If group-specific and sort_key appears to be a mean_lfc column and group-specific lfc available
        if group is not None and "mean_lfc" in sort_key.lower() and lfc_data_available:
            data_dict['sort_val'] = adata.varm[lfc_varm_key][group].values
            logger.info(f"Using group-specific mean log fold change for sorting")
        elif sort_key in adata.var.columns:
            data_dict['sort_val'] = adata.var[sort_key].values
            logger.info(f"Using '{sort_key}' for sorting")
        elif sort_key in adata.varm.keys():
            data_dict['sort_val'] = adata.varm[sort_key][group].values
            logger.info(f"Using group-specific '{sort_key}' for sorting")
        else:
            msg = f"sort_key = '{sort_key}' not found in adata.var or adata.varm."
            logger.error(msg)
            raise KeyError(msg)
    else:
        data_dict["sort_val"] = y

    de_data = pd.DataFrame(data_dict)

    # If background_color_key is provided, add it to the dataframe
    if background_color_key is not None and background_color_key in adata.var.columns:
        de_data["bg_color"] = adata.var[background_color_key].values

        # Determine if background color is categorical or continuous
        bg_values = adata.var[background_color_key]
        if (
            isinstance(bg_values.dtype, pd.CategoricalDtype)
            or bg_values.dtype == "object"
            or bg_values.dtype == "category"
            or bg_values.dtype == "bool"
        ):
            bg_color_is_categorical = True
            categories = adata.var[background_color_key].unique()
            logger.info(
                f"Using categorical coloring for background with {len(categories):,} categories"
            )

            # Auto-select appropriate colormap for categorical data if none provided
            if background_cmap is None:
                if len(categories) <= 10:
                    background_cmap = "tab10"
                elif len(categories) <= 20:
                    background_cmap = "tab20"
                else:
                    background_cmap = "Set3"  # More pastel colors for many categories
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
                if hasattr(base_cmap, "colors"):
                    # This works for ListedColormap instances like tab10, tab20, etc.
                    avail_colors = base_cmap.colors
                    # Just take the first n_colors (or cycle if we need more)
                    colors = [avail_colors[i % len(avail_colors)] for i in range(n_colors)]
                else:
                    # Fallback for other colormap types
                    colors = [base_cmap(i / max(1, n_colors - 1)) for i in range(n_colors)]

                # Create a discrete colormap
                discrete_cmap = ListedColormap(colors)

                # Store both the discrete colormap and the category mapping
                cmap = discrete_cmap
                category_colors = {cat: colors[i] for i, cat in enumerate(categories)}

            # Map categories to colors
            bg_colors = [category_colors.get(val, "gray") for val in de_data["bg_color"]]
            bg_norm = None

        else:
            # Continuous background coloring
            bg_color_is_categorical = False
            logger.info("Using continuous coloring for background")

            # Auto-select appropriate colormap for continuous data if none provided
            if background_cmap is None:
                background_cmap = "Spectral_r"  # Default continuous colormap
                logger.info(f"Auto-selected '{background_cmap}' colormap for continuous data")

            # Get colormap
            if isinstance(background_cmap, str):
                cmap = mpl.colormaps[background_cmap]
            else:
                cmap = background_cmap

            # Process vmin and vmax - handle percentile strings
            bg_values = de_data["bg_color"].values

            # Handle percentile strings for vmin
            if isinstance(vmin, str):
                if vmin.startswith("p"):
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
                if vmax.startswith("p"):
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

                logger.info(
                    f"Using diverging normalization with vmin={vmin_value}, vcenter={vcenter}, vmax={vmax_value}"
                )
                bg_norm = mpl.colors.TwoSlopeNorm(vmin=vmin_value, vcenter=vcenter, vmax=vmax_value)
            else:
                logger.info(f"Using linear normalization with vmin={vmin_value}, vmax={vmax_value}")
                bg_norm = mpl.colors.Normalize(vmin=vmin_value, vmax=vmax_value)

            # We'll use the scatter's built-in normalization for continuous colors
            bg_colors = de_data["bg_color"].values

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
                    if "genes" not in group:
                        logger.warning(f"Group {i} missing 'genes' key, skipping")
                        continue

                    # Extract genes and filter for valid ones
                    gene_list = group["genes"]
                    valid_genes = [g for g in gene_list if g in adata.var_names]

                    if len(valid_genes) < len(gene_list):
                        missing_genes = set(gene_list) - set(valid_genes)
                        logger.warning(
                            f"Group {i}: {len(missing_genes)} genes not found in the dataset"
                        )

                    if not valid_genes:
                        logger.warning(f"Group {i}: No valid genes found, skipping")
                        continue

                    # Use provided color or auto-generate
                    color = group.get("color")
                    name = group.get("name", f"Group {i+1}")

                    # Add group to list
                    highlight_groups.append({"genes": valid_genes, "color": color, "name": name})
                    logger.info(f"Added highlight group '{name}' with {len(valid_genes)} genes")
            elif all(isinstance(item, (str, int)) for item in highlight_genes):
                # If highlight_genes is a list of strings or numbers, interpret as gene names
                valid_genes = [g for g in highlight_genes if g in adata.var_names]

                if len(valid_genes) < len(highlight_genes):
                    missing_genes = set(str(g) for g in highlight_genes) - set(
                        str(g) for g in valid_genes
                    )
                    logger.warning(
                        f"{len(missing_genes)} genes not found in the dataset: {', '.join(str(g) for g in missing_genes)}"
                    )

                # Create a single group without custom colors
                highlight_groups.append({"genes": valid_genes, "name": "Highlighted genes"})
                logger.info(f"Highlighting {len(valid_genes)} user-specified genes")
            else:
                # List with mix of types - try to interpret as list of lists
                for i, group in enumerate(highlight_genes):
                    if isinstance(group, list):
                        # This is a list of genes
                        valid_genes = [g for g in group if g in adata.var_names]

                        if len(valid_genes) < len(group):
                            missing_genes = set(str(g) for g in group) - set(
                                str(g) for g in valid_genes
                            )
                            logger.warning(
                                f"Group {i}: {len(missing_genes)} genes not found in the dataset"
                            )

                        if not valid_genes:
                            logger.warning(f"Group {i}: No valid genes found, skipping")
                            continue

                        # Add as a group with auto-generated name
                        highlight_groups.append({"genes": valid_genes, "name": f"Group {i+1}"})
                        logger.info(f"Added highlight group {i+1} with {len(valid_genes)} genes")

        elif isinstance(highlight_genes, dict):
            # Dictionary format: {gene_name: color}
            gene_list = list(highlight_genes.keys())
            valid_genes = [g for g in gene_list if g in adata.var_names]

            if len(valid_genes) < len(gene_list):
                missing_genes = set(gene_list) - set(valid_genes)
                logger.warning(f"{len(missing_genes)} genes not found in the dataset")

            # Create a single group with custom colors for each gene
            highlight_groups.append(
                {
                    "genes": valid_genes,
                    "colors": {g: highlight_genes[g] for g in valid_genes},
                    "name": "Highlighted genes",
                }
            )
            logger.info(f"Highlighting {len(valid_genes)} genes with custom colors")
        else:
            # This case shouldn't be triggered anymore since we handle all list types above
            # But keeping for backward compatibility - Simple list of genes
            valid_genes = [g for g in highlight_genes if g in adata.var_names]

            if len(valid_genes) < len(highlight_genes):
                # Try to convert to strings for error reporting
                try:
                    missing_genes = set(str(g) for g in highlight_genes) - set(
                        str(g) for g in valid_genes
                    )
                    logger.warning(
                        f"{len(missing_genes)} genes not found in the dataset: {', '.join(missing_genes)}"
                    )
                except Exception:
                    logger.warning("Some genes not found in the dataset")

            # Create a single group without custom colors
            highlight_groups.append({"genes": valid_genes, "name": "Highlighted genes"})
            logger.info(f"Highlighting {len(valid_genes)} user-specified genes")
    else:
        # No highlight_genes provided - choose highlighting strategy
        if n_top_genes is not None:
            # Use top N genes by score approach
            top_genes = de_data.sort_values("sort_val", ascending=False).head(n_top_genes)
            highlight_groups.append(
                {"genes": top_genes["gene"].tolist(), "name": f"Top {n_top_genes} genes"}
            )
            logger.info(f"Highlighting top {n_top_genes:,} genes by {sort_key or score_key}")
        else:
            # Use significance threshold or is_de column to determine highlighted genes (default behavior)
            
            # Use significance threshold for gene selection if provided
            if significance_threshold is not None:
                # Initialize variables
                significant_genes = []
                significant_mask = None

                # Handle both float and dictionary threshold formats
                if isinstance(significance_threshold, dict):
                    # Dictionary format: apply multiple thresholds, genes must pass ALL
                    significant_mask = pd.Series(True, index=adata.var_names)
                    threshold_descriptions = []

                    for axis_type, threshold_val in significance_threshold.items():
                        # Get the appropriate column for this axis type
                        if axis_type == "local_fdr":
                            col_key = run_info["fdr_keys"].get("local_fdr_key") if run_info and "fdr_keys" in run_info and run_info["fdr_keys"] else None
                            comparison = '<'
                        elif axis_type == "tail_fdr":
                            col_key = run_info["fdr_keys"].get("tail_fdr_key") if run_info and "fdr_keys" in run_info and run_info["fdr_keys"] else None
                            comparison = '<'
                        elif axis_type == "ptp":
                            col_key = run_info.get("ptp_key") if run_info else None
                            comparison = '<'
                        elif axis_type == "mahalanobis":
                            col_key = score_key if "mahalanobis" in (score_key or "").lower() else None
                            comparison = '>'
                        else:
                            # Custom column
                            col_key = axis_type if axis_type in adata.var.columns else None
                            comparison = '>'  # Assume higher is more significant for custom columns

                        if col_key and col_key in adata.var.columns:
                            col_values = adata.var[col_key]
                            if comparison == '<':
                                axis_mask = col_values < threshold_val
                            else:
                                axis_mask = col_values > threshold_val

                            significant_mask = significant_mask & axis_mask
                            threshold_descriptions.append(f"{axis_type} {comparison} {threshold_val}")
                            logger.info(f"Applied threshold: {axis_type} {comparison} {threshold_val} ({axis_mask.sum()} genes pass)")
                        else:
                            logger.warning(f"Column for axis type '{axis_type}' not found, skipping this threshold")

                    significant_genes = adata.var_names[significant_mask].tolist()
                    threshold_desc = " AND ".join(threshold_descriptions)
                    logger.info(f"Found {len(significant_genes)} genes passing all thresholds: {threshold_desc}")

                else:
                    # Float format: single threshold (original behavior)
                    significance_values_key = None
                    threshold_comparison = None  # '<' for FDR/ptp, '>' for mahalanobis

                    # Determine which column to use for significance values
                    if y_axis_type == "local_fdr":
                        significance_values_key = run_info["fdr_keys"].get("local_fdr_key") if run_info and "fdr_keys" in run_info and run_info["fdr_keys"] else None
                        threshold_comparison = '<'
                    elif y_axis_type == "tail_fdr":
                        significance_values_key = run_info["fdr_keys"].get("tail_fdr_key") if run_info and "fdr_keys" in run_info and run_info["fdr_keys"] else None
                        threshold_comparison = '<'
                    elif y_axis_type == "ptp":
                        significance_values_key = run_info.get("ptp_key") if run_info else None
                        threshold_comparison = '<'
                    elif y_axis_type == "mahalanobis":
                        significance_values_key = score_key  # Use the current score key
                        threshold_comparison = '>'
                    else:
                        # For custom columns, use the current score key and assume higher is more significant
                        significance_values_key = score_key
                        threshold_comparison = '>'

                    # Fallback to score_key if no specific key found
                    if not significance_values_key:
                        significance_values_key = score_key
                        threshold_comparison = '>' if y_axis_type in ['mahalanobis'] else '<'
                        logger.info(f"Using score key '{score_key}' for significance threshold (threshold={significance_threshold})")

                    if significance_values_key and significance_values_key in adata.var.columns:
                        # Select genes based on significance threshold
                        sig_values = adata.var[significance_values_key]
                        logger.info(f"Significance threshold selection: using column '{significance_values_key}' with threshold {threshold_comparison} {significance_threshold}")
                        logger.info(f"Values range: {sig_values.min():.6f} - {sig_values.max():.6f}")

                        if threshold_comparison == '<':
                            significant_mask = sig_values < significance_threshold
                        else:  # '>'
                            significant_mask = sig_values > significance_threshold

                        significant_genes = adata.var_names[significant_mask].tolist()
                        logger.info(f"Found {len(significant_genes)} genes with {y_axis_type} {threshold_comparison} {significance_threshold}")
                    else:
                        logger.warning(f"Cannot use significance threshold: column '{significance_values_key}' not found in adata.var")
                        logger.info(f"Available var columns: {[col for col in adata.var.columns if any(term in col.lower() for term in ['fdr', 'pvalue', 'mahalanobis', 'ptp'])]}")
                        # significant_genes remains empty list from initialization

                # Common logic for both float and dict formats
                if len(significant_genes) > 0:
                    # Update DE classification if requested and we have a DE column
                    if update_de_classification and inferred_de_column and inferred_de_column in adata.var.columns:
                        old_count = np.sum(adata.var[inferred_de_column])
                        adata.var[inferred_de_column] = significant_mask
                        new_count = np.sum(adata.var[inferred_de_column])
                        if isinstance(significance_threshold, dict):
                            logger.info(f"Updated DE classification: {old_count} → {new_count} significant genes with multiple thresholds")
                        else:
                            logger.info(f"Updated DE classification: {old_count} → {new_count} significant genes at {y_axis_type} {threshold_comparison} {significance_threshold}")

                    # Filter de_data for significant genes
                    sig_de_genes_df = de_data[de_data["gene"].isin(significant_genes)]

                    # Count up and down regulated significant genes
                    up_sig_genes = sig_de_genes_df[sig_de_genes_df["lfc"] > 0]["gene"].tolist()
                    down_sig_genes = sig_de_genes_df[sig_de_genes_df["lfc"] < 0]["gene"].tolist()

                    if up_sig_genes:
                        highlight_groups.append({
                            "genes": up_sig_genes,
                            "name": f"Higher in {condition2} ({len(up_sig_genes)})" if condition2 else f"Up-regulated ({len(up_sig_genes)})"
                        })
                    if down_sig_genes:
                        highlight_groups.append({
                            "genes": down_sig_genes,
                            "name": f"Higher in {condition1} ({len(down_sig_genes)})" if condition1 else f"Down-regulated ({len(down_sig_genes)})"
                        })

                    if isinstance(significance_threshold, dict):
                        logger.info(f"Highlighting {len(significant_genes):,} genes with multiple thresholds ({len(up_sig_genes)} up, {len(down_sig_genes)} down)")
                    else:
                        logger.info(f"Highlighting {len(significant_genes):,} genes at {y_axis_type} {threshold_comparison} {significance_threshold} ({len(up_sig_genes)} up, {len(down_sig_genes)} down)")
                else:
                    # No significant genes found - fallback to top genes
                    if isinstance(significance_threshold, dict):
                        logger.info("No genes found with multiple thresholds - falling back to top genes highlighting")
                    else:
                        logger.info(f"No genes found at {y_axis_type} {threshold_comparison} {significance_threshold} - falling back to top genes highlighting")

                    # Fallback to top genes when no significant genes are found (if n_top_genes specified)
                    fallback_n = n_top_genes or 10  # Use 10 as fallback if n_top_genes is None
                    top_genes = de_data.sort_values("sort_val", ascending=False).head(fallback_n)
                    highlight_groups.append(
                        {"genes": top_genes["gene"].tolist(), "name": f"Top {fallback_n} genes (no genes at threshold)"}
                    )
                    logger.info(f"Highlighting top {fallback_n:,} genes by score as fallback")
            
            # Regular DE column logic (when no other highlighting mechanism is specified or failed)
            # Use DE column for highlighting when no threshold is specified, regardless of background coloring
            elif (
                inferred_de_column
                and inferred_de_column in adata.var.columns
                and highlight_genes is None  # Only use DE column if no specific genes highlighted
            ):
                # Get genes marked as DE by filtering de_data using gene names
                is_de_mask = de_data["gene"].isin(adata.var_names[adata.var[inferred_de_column]])
                de_genes_df = de_data[is_de_mask]
                de_genes = de_genes_df["gene"].tolist()
                
                if de_genes:
                    # Count up and down regulated DE genes
                    up_de_genes = de_genes_df[de_genes_df["lfc"] > 0]["gene"].tolist()
                    down_de_genes = de_genes_df[de_genes_df["lfc"] < 0]["gene"].tolist()

                    # Always add separate highlight groups for up/down DE genes
                    if up_de_genes:
                        highlight_groups.append({
                            "genes": up_de_genes,
                            "name": f"Higher in {condition2} ({len(up_de_genes)})" if condition2 else f"Up-regulated ({len(up_de_genes)})"
                        })
                    if down_de_genes:
                        highlight_groups.append({
                            "genes": down_de_genes,
                            "name": f"Higher in {condition1} ({len(down_de_genes)})" if condition1 else f"Down-regulated ({len(down_de_genes)})"
                        })

                    logger.info(f"Highlighting {len(de_genes):,} genes marked as DE ({len(up_de_genes)} up, {len(down_de_genes)} down)")
                else:
                    logger.info("No genes marked as DE found - falling back to top genes highlighting")
                    # Fallback to top genes when no DE genes are found
                    fallback_n = n_top_genes or 10  # Use 10 as fallback if n_top_genes is None
                    top_genes = de_data.sort_values("sort_val", ascending=False).head(fallback_n)
                    highlight_groups.append(
                        {"genes": top_genes["gene"].tolist(), "name": f"Top {fallback_n} genes (no DE genes found)"}
                    )
                    logger.info(f"Highlighting top {fallback_n:,} genes by {sort_key or score_key} as fallback")
            else:
                # Fallback to top genes approach if DE column not found
                fallback_n = n_top_genes or 10  # Use 10 as fallback if n_top_genes is None
                logger.info(f"DE column '{inferred_de_column}' not found. Falling back to top {fallback_n} genes by score.")
                top_genes = de_data.sort_values("sort_val", ascending=False).head(fallback_n)
                highlight_groups.append(
                    {"genes": top_genes["gene"].tolist(), "name": f"Top {fallback_n} genes"}
                )
                logger.info(f"Highlighting top {fallback_n:,} genes by {sort_key or score_key}")

    # Plot background genes
    if background_color_key is not None:
        if bg_color_is_categorical:
            # Create a scatter plot for each category to add to legend
            for category, color in category_colors.items():
                mask = de_data["bg_color"] == category
                if mask.any():  # Only plot if we have points with this category
                    # Need to use color as a string or RGB tuple, not as 'c' parameter
                    # to avoid the single numeric RGB warning
                    ax.scatter(
                        de_data.loc[mask, "lfc"].values,
                        de_data.loc[mask, "score"].values,
                        alpha=alpha_background,
                        s=point_size,
                        color=color,  # Use 'color' parameter instead of 'c'
                        label=category,
                        **kwargs,
                    )
        else:
            # Continuous coloring
            scatter = ax.scatter(
                de_data["lfc"].values,
                de_data["score"].values,
                alpha=alpha_background,
                s=point_size,
                c=bg_colors,
                cmap=cmap,
                norm=bg_norm,
                **kwargs,
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
                    bbox.x0 + bbox.width + 0.01,  # x position (just to the right of the plot)
                    sidebar_bottom_center
                    - cax_height / 2,  # y position (centered in lower portion)
                    0.02,  # width (thin)
                    cax_height,  # height (20% of plot height)
                ]

                # Create a small vertical colorbar
                cax = fig.add_axes(cax_rect)
                cbar = fig.colorbar(scatter, cax=cax, orientation="vertical")

                # Adjust label and ticks for better visibility
                cbar.set_label(background_color_key, fontsize=10)
                cbar.ax.tick_params(labelsize=8)
    else:
        # Standard background coloring
        ax.scatter(
            de_data["lfc"].values,
            de_data["score"].values,
            alpha=alpha_background,
            s=point_size,
            c=color_background,
            label="All genes" if show_legend else None,
            **kwargs,
        )

    # Process each highlight group
    for group in highlight_groups:
        # Get genes for this group
        genes = group["genes"]
        group_df = de_data[de_data["gene"].isin(genes)].copy()

        # Determine how to color genes in this group
        if "colors" in group:
            # Dictionary of per-gene colors
            for _, gene_row in group_df.iterrows():
                gene_name = gene_row["gene"]
                color = group["colors"].get(gene_name)
                if color is None:
                    # Use default color based on direction
                    color = color_up if gene_row["lfc"] > 0 else color_down

                # Plot this gene
                ax.scatter(gene_row["lfc"], gene_row["score"], alpha=1, s=point_size * 3, c=color)

                # Add label if requested - will be handled by centralized labeling logic below
                pass
        else:
            # Single color for the whole group (or split by direction)
            group_color = group.get("color")

            if group_color is not None:
                # Use single color for the whole group
                ax.scatter(
                    group_df["lfc"].values,
                    group_df["score"].values,
                    alpha=1,
                    s=point_size * 3,
                    c=group_color,
                    label=group["name"],
                )

                # Labels will be handled by centralized labeling logic below
            else:
                # Split by direction (up/down regulated)
                up_genes = group_df[group_df["lfc"] > 0]
                down_genes = group_df[group_df["lfc"] < 0]

                # Plot up-regulated genes
                if len(up_genes) > 0:
                    ax.scatter(
                        up_genes["lfc"].values,
                        up_genes["score"].values,
                        alpha=1,
                        s=point_size * 3,
                        c=color_up,
                        label=(
                            f"Higher in {condition2} ({len(up_genes)})"
                            if condition2
                            else f"Up-regulated ({len(up_genes)})"
                        ),
                    )

                    # Labels will be handled by centralized labeling logic below

                # Plot down-regulated genes
                if len(down_genes) > 0:
                    ax.scatter(
                        down_genes["lfc"].values,
                        down_genes["score"].values,
                        alpha=1,
                        s=point_size * 3,
                        c=color_down,
                        label=(
                            f"Higher in {condition1} ({len(down_genes)})"
                            if condition1
                            else f"Down-regulated ({len(down_genes)})"
                        ),
                    )

                    # Labels will be handled by centralized labeling logic below

    # Centralized gene labeling logic
    genes_to_label = []
    custom_labels = {}
    
    if gene_labels is not False:
        if gene_labels is True:
            # Label all highlighted genes
            for group in highlight_groups:
                genes_to_label.extend(group["genes"])
            logger.info(f"Labeling all {len(genes_to_label)} highlighted genes")
            
        elif isinstance(gene_labels, int):
            # Label top N genes by score
            top_genes_for_labels = de_data.sort_values("sort_val", ascending=False).head(gene_labels)
            genes_to_label = top_genes_for_labels["gene"].tolist()
            logger.info(f"Labeling top {gene_labels} genes by score")
            
        elif isinstance(gene_labels, list):
            # Label specific genes
            genes_to_label = [g for g in gene_labels if g in adata.var_names]
            if len(genes_to_label) < len(gene_labels):
                missing = set(gene_labels) - set(genes_to_label)
                logger.warning(f"Gene labeling: {len(missing)} genes not found: {', '.join(missing)}")
            logger.info(f"Labeling {len(genes_to_label)} specific genes")
            
        elif isinstance(gene_labels, dict):
            # Label genes with custom labels
            genes_to_label = [g for g in gene_labels.keys() if g in adata.var_names]
            custom_labels = {g: gene_labels[g] for g in genes_to_label}
            if len(genes_to_label) < len(gene_labels):
                missing = set(gene_labels.keys()) - set(genes_to_label)
                logger.warning(f"Gene labeling: {len(missing)} genes not found: {', '.join(missing)}")
            logger.info(f"Labeling {len(genes_to_label)} genes with custom labels")

    # Apply the labels
    if genes_to_label:
        genes_df = de_data[de_data["gene"].isin(genes_to_label)]
        for _, gene_row in genes_df.iterrows():
            gene_name = gene_row["gene"]
            # Use custom label if provided, otherwise use gene name
            label_text = custom_labels.get(gene_name, gene_name)
            ax.annotate(
                label_text,
                (gene_row["lfc"], gene_row["score"]),
                fontsize=font_size,
                **text_kwargs,
            )

    # Create dummy entries for the legend if no highlighted genes
    if len(highlight_groups) == 0 and show_legend:
        ax.scatter(
            [],
            [],
            alpha=1,
            s=point_size * 3,
            c=color_up,
            label=f"Higher in {condition2} (0)" if condition2 else "Up-regulated (0)",
        )
        ax.scatter(
            [],
            [],
            alpha=1,
            s=point_size * 3,
            c=color_down,
            label=f"Higher in {condition1} (0)" if condition1 else "Down-regulated (0)",
        )

    # Add formatting
    ax.axvline(x=0, color="black", linestyle="--", alpha=0.3)

    # Add significance threshold line if applicable (only for single float thresholds, not dictionaries)
    if show_thresholds and significance_threshold is not None and not isinstance(significance_threshold, dict):
        if y_axis_type in ["local_fdr", "tail_fdr", "ptp"] and y_transform is not None:
            # Transform the threshold for display
            threshold_y = y_transform(np.array([significance_threshold]))[0]
            ax.axhline(
                y=threshold_y,
                color="red",
                linestyle="--",
                alpha=0.7,
                linewidth=1,
                label=f"{y_axis_type} = {significance_threshold}",
            )
            logger.info(f"Added {y_axis_type} threshold line at y={threshold_y:.2f} ({y_axis_type}={significance_threshold})")
        else:
            # For Mahalanobis distance and custom columns, threshold is used directly
            ax.axhline(
                y=significance_threshold,
                color="red",
                linestyle="--",
                alpha=0.7,
                linewidth=1,
                label=f"{y_axis_type} = {significance_threshold}",
            )
            logger.info(f"Added {y_axis_type} threshold line at y={significance_threshold}")
    elif show_thresholds and isinstance(significance_threshold, dict):
        logger.info("Skipping threshold line drawing for dictionary-format significance_threshold")

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
    has_continuous_colorbar = background_color_key is not None and not bg_color_is_categorical

    # Add legend with appropriate styling
    if show_legend and legend_loc != "none":
        if has_continuous_colorbar:
            # For continuous colorbar case: place legend in top part of right sidebar
            if legend_loc == "best":
                ax.legend(
                    bbox_to_anchor=(1.05, 0.7),  # Position in top part of sidebar
                    loc="upper left",
                    fontsize=legend_fontsize,
                    frameon=False,
                    ncol=legend_ncol or 1,
                )
            else:
                # If user specified a different location, respect it
                ax.legend(
                    loc=legend_loc, fontsize=legend_fontsize, frameon=False, ncol=legend_ncol or 1
                )
        else:
            # Standard legend placement without colorbar competition
            if legend_loc == "best":
                ax.legend(
                    bbox_to_anchor=(1.05, 1),
                    loc="upper left",
                    fontsize=legend_fontsize,
                    frameon=False,
                    ncol=legend_ncol or 1,
                )
            else:
                ax.legend(
                    loc=legend_loc, fontsize=legend_fontsize, frameon=False, ncol=legend_ncol or 1
                )

    if grid:
        ax.grid(**grid_kwargs)

    # Instead of tight_layout, manually adjust the plot's spacing
    # This avoids issues with colorbars and other axes elements
    if has_continuous_colorbar or (show_legend and legend_loc == "best"):
        # Make room for the right sidebar with legend and/or colorbar
        plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)
    else:
        # Standard spacing when no sidebar is needed
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    # Save figure if path provided
    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight")

    # Return figure and axes if requested
    if return_fig:
        return fig, ax
    elif save is None:
        # Only show if not saving and not returning
        # Check if the current backend allows for interactive display
        if plt.get_backend().lower() not in ["agg", "pdf", "svg", "ps", "cairo", "template"]:
            plt.show()
