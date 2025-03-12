"""Utility functions for heatmap plotting."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Optional, Union, List, Tuple, Dict, Any, Sequence, Literal, Callable, Set
from anndata import AnnData
import pandas as pd
import logging

from ...utils import get_run_from_history
from ..volcano import _extract_conditions_from_key

logger = logging.getLogger("kompot")


def _infer_heatmap_keys(
    adata: AnnData,
    run_id: Optional[int] = None,
    lfc_key: Optional[str] = None,
    score_key: Optional[str] = None,
):
    """
    Infer heatmap keys from AnnData object.

    Parameters
    ----------
    adata : AnnData
        AnnData object with differential expression results
    run_id : int, optional
        Run ID to use. If None, uses latest run (-1).
    lfc_key : str, optional
        Log fold change key. If provided, will be returned as is.
    score_key : str, optional
        Score key. If None, will be inferred from run information.

    Returns
    -------
    tuple
        (lfc_key, score_key) with the inferred keys
    """
    # If both keys already provided, return them
    if lfc_key is not None and score_key is not None:
        return lfc_key, score_key

    # Get run info from kompot_de for the specified run_id
    effective_run_id = -1 if run_id is None else run_id
    run_info = get_run_from_history(adata, effective_run_id, analysis_type="de")
    
    if run_info is None:
        logger.warning(f"No valid run found with run_id={effective_run_id}")
        return lfc_key, score_key
        
    # Extract keys from field_names if present
    inferred_lfc_key = lfc_key
    inferred_score_key = score_key
    
    if "field_names" in run_info:
        field_names = run_info["field_names"]

        # Get lfc_key from field_names if not provided
        if inferred_lfc_key is None and "mean_lfc_key" in field_names:
            inferred_lfc_key = field_names["mean_lfc_key"]

        # Get score_key from field_names if not provided
        if inferred_score_key is None and "mahalanobis_key" in field_names:
            inferred_score_key = field_names["mahalanobis_key"]

    # If lfc_key not found, raise error
    if inferred_lfc_key is None:
        raise ValueError(
            "Could not infer lfc_key from the specified run. Please specify manually."
        )

    return inferred_lfc_key, inferred_score_key


def _prepare_gene_list(
    adata: AnnData,
    var_names: Optional[Union[List[str], Sequence[str]]] = None,
    gene_list: Optional[Union[List[str], Sequence[str]]] = None,
    n_top_genes: int = 20,
    lfc_key: Optional[str] = None,
    score_key: Optional[str] = None,
    sort_genes: bool = True,
    run_id: Optional[int] = None,
) -> Tuple[List[str], Optional[str], Optional[str], Dict]:
    """
    Prepare the list of genes to be included in the heatmap.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object containing expression data
    var_names : list, optional
        List of gene names (legacy parameter)
    gene_list : list, optional
        Explicit list of genes to include
    n_top_genes : int
        Number of top genes to include
    lfc_key : str, optional
        Key for log fold change values. If None, inferred from run_id.
    score_key : str, optional
        Key for significance scores. If None, inferred from run_id.
    sort_genes : bool
        Whether to sort genes by score
    run_id : int, optional
        Run ID to use. If None, uses latest run (-1).
        
    Returns
    -------
    Tuple containing:
    - List of gene names
    - LFC key used
    - Score key used 
    - Run info dictionary
    """
    # Normalize run_id to use -1 (latest run) if None
    effective_run_id = -1 if run_id is None else run_id
    
    # Get run info from history once
    run_info = get_run_from_history(adata, effective_run_id, analysis_type="de")
    
    # If gene_list is provided, use it directly
    if gene_list is not None:
        var_names = gene_list
        logger.info(f"Using provided gene_list with {len(gene_list)} genes/features")
        return var_names, lfc_key, score_key, run_info
        
    # If var_names is provided, use it directly
    if var_names is not None:
        return var_names, lfc_key, score_key, run_info
        
    # If var_names not provided and no gene_list, get top genes based on DE results
    # Infer keys using the helper function
    lfc_key, score_key = _infer_heatmap_keys(adata, effective_run_id, lfc_key, score_key)

    # Extract condition information for logging
    condition1 = condition2 = None
    if run_info is not None and "params" in run_info:
        params = run_info["params"]
        if "conditions" in params and len(params["conditions"]) == 2:
            condition1 = params["conditions"][0]
            condition2 = params["conditions"][1]

    # If conditions not found in run_info, try to extract from lfc_key
    if (condition1 is None or condition2 is None) and lfc_key is not None:
        conditions = _extract_conditions_from_key(lfc_key)
        if conditions:
            condition1, condition2 = conditions

    # Convert negative run_id to positive value for logging
    if effective_run_id < 0 and "kompot_de" in adata.uns and "run_history" in adata.uns["kompot_de"]:
        actual_run_id = len(adata.uns["kompot_de"]["run_history"]) + effective_run_id
    else:
        actual_run_id = effective_run_id

    # Log which run is being used
    conditions_str = f": comparing {condition1} vs {condition2}" if condition1 and condition2 else ""
    log_message = f"Using DE run {actual_run_id} for heatmap{conditions_str}"
    logger.info(log_message)

    # Log the fields being used
    logger.info(f"Using fields for heatmap - lfc_key: '{lfc_key}', score_key: '{score_key}'")

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
    return var_names, lfc_key, score_key, run_info


def _get_expression_matrix(
    adata: AnnData,
    var_names: List[str],
    layer: Optional[str] = None
) -> np.ndarray:
    """
    Extract expression matrix for the specified genes.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object
    var_names : list
        List of gene names to extract
    layer : str, optional
        Layer to use for expression values
        
    Returns
    -------
    numpy.ndarray
        Expression matrix with genes as columns
    """
    # Get expression data from the specified layer or X
    if layer is not None and layer in adata.layers:
        logger.info(f"Using expression data from layer: '{layer}'")
        expr_matrix = (
            adata[:, var_names].layers[layer].toarray()
            if hasattr(adata.layers[layer], "toarray")
            else adata[:, var_names].layers[layer]
        )
    else:
        if layer is not None:
            logger.warning(f"Requested layer '{layer}' not found, falling back to adata.X")
        logger.info(f"Using expression data from adata.X")
        expr_matrix = (
            adata[:, var_names].X.toarray()
            if hasattr(adata.X, "toarray")
            else adata[:, var_names].X
        )
    
    return expr_matrix


def _filter_excluded_groups(
    expr_df: pd.DataFrame, 
    groupby: str, 
    exclude_groups: Optional[Union[str, List[str]]],
    available_groups: List
) -> pd.DataFrame:
    """
    Filter expression dataframe to exclude specified groups.
    
    Parameters
    ----------
    expr_df : pandas.DataFrame
        Expression dataframe
    groupby : str
        Column name for grouping
    exclude_groups : str or list
        Groups to exclude
    available_groups : list
        List of available groups
        
    Returns
    -------
    pandas.DataFrame
        Filtered expression dataframe
    """
    if exclude_groups is None:
        return expr_df
        
    # Convert single group to list for consistent handling
    if isinstance(exclude_groups, str):
        exclude_groups = [exclude_groups]

    # Check for non-existent groups
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
            
    return expr_df


def _apply_scaling(
    data: Union[pd.DataFrame, np.ndarray],
    standard_scale: Optional[Union[str, int]],
    is_split: bool = False,
    has_hierarchical_index: bool = False,
    log_message: bool = True
) -> Union[pd.DataFrame, np.ndarray]:
    """
    Apply scaling to the data (z-scoring).
    
    Parameters
    ----------
    data : DataFrame or ndarray
        Data to scale
    standard_scale : str or int or None
        Type of scaling: 'var'|0 for gene-wise, 'group'|1 for group-wise, None for no scaling
    is_split : bool
        Whether the data has conditions split
    has_hierarchical_index : bool
        Whether the data has a hierarchical index
        
    Returns
    -------
    DataFrame or ndarray
        Scaled data in the same format as input
    """
    if standard_scale is None:
        return data

    # For pandas DataFrame with hierarchical structure (split by condition)
    if isinstance(data, pd.DataFrame):
        if standard_scale == "group" or standard_scale == 1:
            # Scale by group (cols)
            if log_message:
                logger.info("Applying group-wise z-scoring (standard_scale='group')")
            if is_split and has_hierarchical_index:
                # Handle hierarchical columns by scaling each group separately
                for group_idx in data.index:
                    group_data = data.loc[group_idx]
                    # Handle NaN values and zeros
                    group_mean = group_data.mean(skipna=True)
                    group_std = group_data.std(skipna=True)
                    if group_std > 0:
                        data.loc[group_idx] = (group_data - group_mean) / group_std
                    # NaN values remain NaN
            else:
                # Properly handle NaN values
                means = data.mean(axis=0, skipna=True)
                stds = data.std(axis=0, skipna=True)
                # Avoid division by zero
                stds = stds.replace(0, 1)
                # Apply z-scoring
                data = (data - means) / stds

        elif standard_scale == "var" or standard_scale == 0:
            # Scale by gene (rows)
            if log_message:
                logger.info("Applying gene-wise z-scoring (standard_scale='var')")
            if is_split and has_hierarchical_index:
                # Handle hierarchical columns by scaling each gene separately across groups/conditions
                # Get all genes
                all_genes = data.columns
                
                # Process each gene separately
                for gene in all_genes:
                    # Extract this gene's values across all groups and conditions
                    gene_data = data[gene]
                    # Calculate gene-wise mean and std across all values
                    gene_mean = gene_data.mean(skipna=True)
                    gene_std = gene_data.std(skipna=True)
                    # Avoid division by zero
                    if gene_std > 0:
                        # Apply z-scoring to this gene across all groups/conditions
                        data[gene] = (gene_data - gene_mean) / gene_std
                    # NaN values remain NaN
            else:
                # Z-score each gene (column) across all samples
                # Properly handle NaN values
                means = data.mean(axis=0, skipna=True)
                stds = data.std(axis=0, skipna=True)
                # Avoid division by zero
                stds = stds.replace(0, 1)
                # Apply z-scoring directly without transposition
                data = (data - means) / stds

    # For numpy arrays (diagonal split data)
    elif isinstance(data, np.ndarray):
        if standard_scale == "var" or standard_scale == 0:
            # Perform gene-wise z-scoring
            if log_message:
                logger.info("Applying gene-wise z-scoring (standard_scale='var')")
            # For gene-wise z-scoring, compute stats along axis=0 (across columns/samples)
            means = np.nanmean(data, axis=0, keepdims=True)
            stds = np.nanstd(data, axis=0, keepdims=True)
            # Avoid division by zero
            stds[stds == 0] = 1.0
            stds[np.isnan(stds)] = 1.0
            # Z-score the array
            data = (data - means) / stds
            
        elif standard_scale == "group" or standard_scale == 1:
            # Perform group-wise z-scoring
            if log_message:
                logger.info("Applying group-wise z-scoring (standard_scale='group')")
            # For group-wise z-scoring, compute stats along axis=1 (across genes)
            means = np.nanmean(data, axis=1, keepdims=True)
            stds = np.nanstd(data, axis=1, keepdims=True)
            # Avoid division by zero
            stds[stds == 0] = 1.0
            stds[np.isnan(stds)] = 1.0
            # Z-score
            data = (data - means) / stds
    
    return data


def _calculate_figsize(
    n_rows: int, 
    n_cols: int, 
    dendrogram: bool = False,
    cluster_rows: bool = False,
    cluster_cols: bool = False
) -> Tuple[float, float]:
    """
    Calculate appropriate figure size based on data dimensions.
    
    Parameters
    ----------
    n_rows : int
        Number of rows
    n_cols : int
        Number of columns
    dendrogram : bool
        Whether dendrograms are shown
    cluster_rows : bool
        Whether rows are clustered
    cluster_cols : bool
        Whether columns are clustered
        
    Returns
    -------
    tuple
        Figure size as (width, height)
    """
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

    return (width_inches, height_inches)


def _setup_colormap_normalization(data, center, vmin, vmax, cmap):
    """
    Set up colormap normalization for heatmap.
    
    Parameters
    ----------
    data : array-like
        Data to visualize
    center : float or None
        Center value for diverging colormaps
    vmin : float or None
        Minimum value for colormap
    vmax : float or None
        Maximum value for colormap
    cmap : str or matplotlib.colors.Colormap
        Colormap to use
        
    Returns
    -------
    tuple
        (norm, cmap_obj, vmin, vmax) with the normalization settings
    """
    # Determine data range if not explicitly provided
    if vmin is None:
        vmin = np.nanmin(data)
    if vmax is None:
        vmax = np.nanmax(data)
        
    # Ensure vmin is not equal to vmax to avoid normalization issues
    if vmin == vmax:
        vmin -= 0.1
        vmax += 0.1
        
    # Set up normalization
    if center is not None:
        # Use diverging normalization
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
    else:
        # Use standard normalization
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    
    # Get colormap object
    if isinstance(cmap, str):
        try:
            # Use the newer API if available
            cmap_obj = plt.colormaps[cmap]
        except (AttributeError, KeyError):
            # Fall back to older API for compatibility
            cmap_obj = plt.cm.get_cmap(cmap)
    else:
        cmap_obj = cmap
        
    return norm, cmap_obj, vmin, vmax