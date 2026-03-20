"""Functions for visualizing expression patterns for individual genes."""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Union, List, Tuple, Dict, Any, Literal
from anndata import AnnData
import warnings
import logging

from ..utils import KOMPOT_COLORS
from ..anndata.utils import get_run_from_history
from .volcano import _extract_conditions_from_key

# Get the pre-configured logger
logger = logging.getLogger("kompot")


def _infer_expression_keys(adata: AnnData, run_id: int = -1, lfc_key: Optional[str] = None,
                           score_key: Optional[str] = None, strict: bool = False):
    """
    Infer expression-related keys from AnnData object for gene expression visualization.

    This function uses robust field inference that relies on run info and provides
    proper warnings for overwrites and fallbacks.

    Parameters
    ----------
    adata : AnnData
        AnnData object with differential expression results
    run_id : int, optional
        Run ID to use. Default is -1 (the latest run).
    lfc_key : str, optional
        Log fold change key. If provided, will be returned as is.
    score_key : str, optional
        Score key. If provided, will be returned as is unless the default
        value needs to be replaced with a run-specific key.
    strict : bool, optional
        If True, raises errors when keys can't be inferred. If False, returns
        None for missing keys. Default is False.

    Returns
    -------
    tuple
        (lfc_key, score_key) with the inferred keys
    """
    from .field_inference import infer_fields_from_run_info

    # If both keys already provided, just return them
    if lfc_key is not None and score_key is not None:
        return lfc_key, score_key

    # Use robust field inference
    required_fields = []
    if lfc_key is None:
        required_fields.append("mean_lfc_key")
    if score_key is None:
        required_fields.append("mahalanobis_key")

    if required_fields:
        inferred_fields = infer_fields_from_run_info(
            adata=adata,
            analysis_type="de",
            run_id=run_id,
            required_fields=required_fields,
            strict=strict
        )

        # Use inferred fields if not provided by user
        if lfc_key is None:
            lfc_key = inferred_fields.get("mean_lfc_key")
        if score_key is None:
            score_key = inferred_fields.get("mahalanobis_key")

    return lfc_key, score_key

try:
    import scanpy as sc
    _has_scanpy = True
except ImportError:
    _has_scanpy = False


def plot_gene_expression(
    adata: AnnData,
    gene: str,
    lfc_key: Optional[str] = None,
    score_key: Optional[str] = None,
    condition1: Optional[str] = None,
    condition2: Optional[str] = None,
    basis: Optional[str] = "X_umap", 
    figsize: Tuple[float, float] = (12, 12),
    cmap_expression: str = "Spectral_r",
    cmap_fold_change: str = "RdBu_r",
    title: Optional[str] = None,
    run_id: int = -1,
    layer: Optional[str] = None,
    save: Optional[str] = None,
    return_fig: bool = False,
    **kwargs
) -> Optional[plt.Figure]:
    """
    Visualize expression patterns for a specific gene across conditions.
    
    Creates a figure with multiple panels showing original expression, imputed expression
    for each condition, and fold change.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object containing differential expression results
    gene : str
        Name of the gene to visualize
    lfc_key : str, optional
        Key in adata.var for log fold change values.
        If None, will try to infer from ``kompot_de_`` keys.
    score_key : str, optional
        Key in adata.var for significance scores.
        If None, will try to infer from ``kompot_de_`` keys.
    condition1 : str, optional
        Name of condition 1 (denominator in fold change)
    condition2 : str, optional
        Name of condition 2 (numerator in fold change)
    basis : str or None, optional
        Key in adata.obsm for the embedding coordinates (default: "X_umap").
        If None, will use cell index for x-axis instead of embeddings.
    figsize : tuple, optional
        Figure size as (width, height) in inches
    cmap_expression : str, optional
        Colormap for expression plots
    cmap_fold_change : str, optional
        Colormap for fold change plot
    title : str, optional
        Overall figure title. If None, uses gene name.
    run_id : int, optional
        Run ID to use. Default is -1 (latest run).
    layer : str, optional
        Layer in AnnData to use for expression values. If None, uses adata.X or infers from run information.
    save : str, optional
        Path to save figure. If None, figure is not saved
    return_fig : bool, optional
        If True, returns the figure and axes
    **kwargs : 
        Additional parameters passed to scatter plot functions
        
    Returns
    -------
    If return_fig is True, returns (fig, axes)
    """
    # Check if scanpy is available for the scatter plots
    if not _has_scanpy:
        warnings.warn(
            "Scanpy is required for plotting. Install scanpy to use this function."
        )
        return None
        
    # Check if gene is in the dataset
    if gene not in adata.var_names:
        raise ValueError(f"Gene '{gene}' not found in adata.var_names.")
        
    # Get gene index
    gene_idx = adata.var_names.get_loc(gene)
    
    # Infer keys using our specialized helper function
    # Pass strict=False to avoid raising errors when keys can't be inferred
    lfc_key, score_key = _infer_expression_keys(adata, run_id, lfc_key, score_key, strict=False)
    
    # If keys are still None, provide a helpful warning
    if lfc_key is None:
        logger.warning("Could not infer lfc_key. Some functionality may be limited.")
        
    if score_key is None:
        logger.warning("Could not infer score_key. Some functionality may be limited.")
    
    # Calculate the actual (positive) run ID for logging - same as volcano_da
    if run_id < 0:
        # Use get_run_history to get the deserialized run history
        from ..anndata.utils import get_run_history
        run_history = get_run_history(adata, "de")
        if run_history is not None:
            actual_run_id = len(run_history) + run_id
        else:
            actual_run_id = run_id
    else:
        actual_run_id = run_id
    
    # Get run info for both conditions and layer
    run_info = get_run_from_history(adata, run_id, analysis_type="de")
    
    # Extract conditions - prioritize run_info params (stores original names reliably)
    if condition1 is None or condition2 is None:
        # First try run_info params - this is the authoritative source
        if run_info is not None and 'params' in run_info:
            params = run_info['params']
            if condition1 is None and 'condition1' in params:
                condition1 = params['condition1']
            if condition2 is None and 'condition2' in params:
                condition2 = params['condition2']

        # Only if run_info unavailable, fall back to key name extraction
        if condition1 is None or condition2 is None:
            conditions = _extract_conditions_from_key(lfc_key) if lfc_key is not None else None
            if conditions:
                if condition1 is None:
                    condition1 = conditions[0]
                if condition2 is None:
                    condition2 = conditions[1]

        # If still none, provide defaults
        if condition1 is None:
            condition1 = "Condition 1"
            logger.info(f"No condition1 found, using default: '{condition1}'")
        if condition2 is None:
            condition2 = "Condition 2"
            logger.info(f"No condition2 found, using default: '{condition2}'")
    
    # Log which run and fields are being used - same pattern as volcano_da
    conditions_str = f": comparing {condition1} to {condition2}" if condition1 and condition2 else ""
    logger.info(f"Using DE run {actual_run_id}{conditions_str}")
    
    lfc_str = f"'{lfc_key}'" if lfc_key is not None else "None"
    score_str = f"'{score_key}'" if score_key is not None else "None"
    logger.info(f"Using fields for gene expression plot - lfc_key: {lfc_str}, score_key: {score_str}")
    
    # Try to infer layer from run_info if not explicitly provided
    if layer is None and run_info is not None and 'params' in run_info:
        params = run_info['params']
        if 'layer' in params:
            inferred_layer = params['layer']
            # Don't use fold_change layers for expression visualization
            if inferred_layer is not None and "fold_change" not in inferred_layer:
                layer = inferred_layer
                logger.info(f"Using layer '{layer}' inferred from run information")
            elif inferred_layer is not None:
                logger.info(f"Ignoring fold_change layer '{inferred_layer}' inferred from run, using adata.X instead")
            
    # Extract fold change and score for the gene
    gene_lfc = adata.var.loc[gene, lfc_key] if lfc_key is not None and lfc_key in adata.var else "Unknown"
    gene_score = adata.var.loc[gene, score_key] if score_key is not None and score_key in adata.var else "Unknown"
    
    # Check if embedding basis exists
    if basis not in adata.obsm:
        logger.warning(f"Basis '{basis}' not found in adata.obsm. Available bases: {list(adata.obsm.keys())}")
        logger.warning("Falling back to standard coordinates.")
        basis = None
        
    # Determine condition-specific imputed expression layer names from run_info.
    # We never silently fall back to pattern-matching layer names, as that could
    # pick layers from a different run with a different result_key prefix.
    condition1_layer = None
    condition2_layer = None
    fold_change_layer = None

    if run_info is not None:
        # Get the condition names from run_info params for mapping
        param_condition1 = run_info.get('params', {}).get('condition1')
        param_condition2 = run_info.get('params', {}).get('condition2')

        # Determine if the user swapped conditions relative to the run
        swapped = (param_condition1 == condition2 and param_condition2 == condition1)

        # Primary source: imputed_layer_keys (stores the full layer names directly)
        if 'imputed_layer_keys' in run_info:
            imputed_keys = run_info['imputed_layer_keys']

            if swapped:
                condition1_layer = imputed_keys.get('condition2')
                condition2_layer = imputed_keys.get('condition1')
            else:
                condition1_layer = imputed_keys.get('condition1')
                condition2_layer = imputed_keys.get('condition2')
            fold_change_layer = imputed_keys.get('fold_change')

        # Fallback source: field_names dict
        elif 'field_names' in run_info:
            fn = run_info['field_names']
            if 'imputed_key_1' in fn and 'imputed_key_2' in fn:
                if swapped:
                    condition1_layer = fn['imputed_key_2']
                    condition2_layer = fn['imputed_key_1']
                else:
                    condition1_layer = fn['imputed_key_1']
                    condition2_layer = fn['imputed_key_2']
            fold_change_layer = fn.get('fold_change_key')

        # Log what we resolved
        if condition1_layer:
            logger.info(f"Using imputed layer '{condition1_layer}' for '{condition1}'")
        if condition2_layer:
            logger.info(f"Using imputed layer '{condition2_layer}' for '{condition2}'")
        if fold_change_layer:
            logger.info(f"Using fold_change layer '{fold_change_layer}' from run_info")

    # Warn explicitly about missing layers instead of silently guessing
    if condition1_layer is None or condition2_layer is None or fold_change_layer is None:
        missing = []
        if condition1_layer is None:
            missing.append(f"condition1 ('{condition1}') imputed layer")
        if condition2_layer is None:
            missing.append(f"condition2 ('{condition2}') imputed layer")
        if fold_change_layer is None:
            missing.append("fold_change layer")
        logger.warning(
            f"Could not determine layer names from run_info for: {', '.join(missing)}. "
            f"The corresponding panels will be empty. "
            f"Re-run compute_differential_expression or specify layers manually."
        )
            
    # Create figure and axes
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    
    # Set the figure title
    if title is None:
        title = f"Expression Patterns for {gene}"
    fig.suptitle(title, fontsize=16, y=1)
    
    # Add fold change and score information as a subtitle
    if isinstance(gene_lfc, (int, float)) and isinstance(gene_score, (int, float)):
        plt.figtext(
            0.5, 0.965, 
            f"Mean log fold change: {gene_lfc:.2f} | Mahalanobis distance: {gene_score:.2f}",
            ha='center', fontsize=12
        )
    
    # Panel 1: Original expression
    if basis and _has_scanpy:
        # Use scanpy to plot original expression on embedding
        sc.pl.embedding(
            adata, 
            basis=basis.replace("X_", ""),
            color=gene,
            title=f"Original Expression{' ('+layer+')' if layer else ''}",
            color_map=cmap_expression,
            layer=layer,  # Use the inferred or specified layer
            show=False,
            ax=axs[0, 0],
            **kwargs
        )
    else:
        # Fallback to simple scatter plot
        if layer is not None and layer in adata.layers:
            # Use specified/inferred layer
            logger.info(f"Using expression data from layer: '{layer}'")
            orig_values = adata[:, gene].layers[layer]
        else:
            # Use default X matrix
            if layer is not None:
                logger.warning(f"Requested layer '{layer}' not found, falling back to adata.X")
            orig_values = adata[:, gene].X
            
        if hasattr(orig_values, 'toarray'):
            orig_values = orig_values.toarray().flatten()
        else:
            orig_values = orig_values.flatten()
            
        # Sort cells by expression value
        sort_idx = np.argsort(orig_values)
        sorted_values = orig_values[sort_idx]
            
        # Simple scatter without grid - use a small default point size
        scatter_kwargs = {'s': 3}
        # Override with user-provided kwargs
        scatter_kwargs.update(kwargs)
            
        axs[0, 0].scatter(
            np.arange(len(sorted_values)),
            sorted_values,
            c=sorted_values,
            cmap=cmap_expression,
            **scatter_kwargs
        )
        axs[0, 0].set_title(f"Original Expression{' ('+layer+')' if layer else ''}")
        axs[0, 0].set_xlabel("Cell index")
        axs[0, 0].set_ylabel("Expression")
        
        # Clean up plot appearance
        axs[0, 0].grid(False)
        axs[0, 0].spines['top'].set_visible(False)
        axs[0, 0].spines['right'].set_visible(False)
        
    # Panel 2: Condition 1 imputed expression
    if condition1_layer and condition1_layer in adata.layers:
        if basis and _has_scanpy:
            sc.pl.embedding(
                adata,
                basis=basis.replace("X_", ""),
                color=gene,
                title=f"Imputed Expression ({condition1})",
                color_map=cmap_expression,
                layer=condition1_layer,
                show=False,
                ax=axs[0, 1],
                **kwargs
            )
        elif basis:
            # Handle both sparse and dense layers
            values = adata.layers[condition1_layer][:, gene_idx]
            if hasattr(values, 'toarray'):
                values = values.toarray().flatten()
            else:
                values = values.flatten()
                
            # Create scatter kwargs - use a small default point size
            scatter_kwargs = {'s': 3}
            # Override with user-provided kwargs
            scatter_kwargs.update(kwargs)
                
            scatter1 = axs[0, 1].scatter(
                adata.obsm[basis][:, 0],
                adata.obsm[basis][:, 1],
                c=values,
                cmap=cmap_expression,
                **scatter_kwargs
            )
            plt.colorbar(scatter1, ax=axs[0, 1])
            axs[0, 1].set_title(f"Imputed Expression ({condition1})")
            axs[0, 1].set_xlabel("UMAP 1")
            axs[0, 1].set_ylabel("UMAP 2")
            axs[0, 1].grid(False)
            axs[0, 1].spines['top'].set_visible(False)
            axs[0, 1].spines['right'].set_visible(False)
        else:
            # Handle both sparse and dense layers
            values = adata.layers[condition1_layer][:, gene_idx]
            if hasattr(values, 'toarray'):
                values = values.toarray().flatten()
            else:
                values = values.flatten()
                
            # Sort cells by expression value
            sort_idx = np.argsort(values)
            sorted_values = values[sort_idx]
                
            # Create scatter kwargs - use a small default point size
            scatter_kwargs = {'s': 3}
            # Override with user-provided kwargs
            scatter_kwargs.update(kwargs)
                
            scatter1 = axs[0, 1].scatter(
                np.arange(len(sorted_values)),
                sorted_values,
                c=sorted_values,
                cmap=cmap_expression,
                **scatter_kwargs
            )
            plt.colorbar(scatter1, ax=axs[0, 1])
            axs[0, 1].set_title(f"Imputed Expression ({condition1})")
            axs[0, 1].set_xlabel("Cell index")
            axs[0, 1].set_ylabel("Expression")
            axs[0, 1].grid(False)
            axs[0, 1].spines['top'].set_visible(False)
            axs[0, 1].spines['right'].set_visible(False)
    else:
        axs[0, 1].set_title(f"Imputed Expression ({condition1 or 'Condition 1'}): Not available")
        axs[0, 1].set_axis_off()
        
    # Panel 3: Condition 2 imputed expression
    if condition2_layer and condition2_layer in adata.layers:
        if basis and _has_scanpy:
            
            sc.pl.embedding(
                adata,
                basis=basis.replace("X_", ""),
                color=gene,
                title=f"Imputed Expression ({condition2})",
                color_map=cmap_expression,
                layer=condition2_layer,
                show=False,
                ax=axs[1, 0],
                **kwargs
            )
        elif basis:
            # Handle both sparse and dense layers
            values = adata.layers[condition2_layer][:, gene_idx]
            if hasattr(values, 'toarray'):
                values = values.toarray().flatten()
            else:
                values = values.flatten()
                
            # Create scatter kwargs - use a small default point size
            scatter_kwargs = {'s': 3}
            # Override with user-provided kwargs
            scatter_kwargs.update(kwargs)
                
            scatter2 = axs[1, 0].scatter(
                adata.obsm[basis][:, 0],
                adata.obsm[basis][:, 1],
                c=values,
                cmap=cmap_expression,
                **scatter_kwargs
            )
            plt.colorbar(scatter2, ax=axs[1, 0])
            axs[1, 0].set_title(f"Imputed Expression ({condition2})")
            axs[1, 0].set_xlabel("UMAP 1")
            axs[1, 0].set_ylabel("UMAP 2")
            axs[1, 0].grid(False)
            axs[1, 0].spines['top'].set_visible(False)
            axs[1, 0].spines['right'].set_visible(False)
        else:
            # Handle both sparse and dense layers
            values = adata.layers[condition2_layer][:, gene_idx]
            if hasattr(values, 'toarray'):
                values = values.toarray().flatten()
            else:
                values = values.flatten()
                
            # Sort cells by expression value
            sort_idx = np.argsort(values)
            sorted_values = values[sort_idx]
                
            # Create scatter kwargs - use a small default point size
            scatter_kwargs = {'s': 3}
            # Override with user-provided kwargs
            scatter_kwargs.update(kwargs)
                
            scatter2 = axs[1, 0].scatter(
                np.arange(len(sorted_values)),
                sorted_values,
                c=sorted_values,
                cmap=cmap_expression,
                **scatter_kwargs
            )
            plt.colorbar(scatter2, ax=axs[1, 0])
            axs[1, 0].set_title(f"Imputed Expression ({condition2})")
            axs[1, 0].set_xlabel("Cell index")
            axs[1, 0].set_ylabel("Expression")
            axs[1, 0].grid(False)
            axs[1, 0].spines['top'].set_visible(False)
            axs[1, 0].spines['right'].set_visible(False)
    else:
        axs[1, 0].set_title(f"Imputed Expression ({condition2 or 'Condition 2'}): Not available")
        axs[1, 0].set_axis_off()
        
    # Panel 4: Fold change
    if fold_change_layer and fold_change_layer in adata.layers:
        if basis and _has_scanpy:
            scatter_kwargs = {'vcenter': 0}
            scatter_kwargs.update(kwargs)
            
            sc.pl.embedding(
                adata,
                basis=basis.replace("X_", ""),
                color=gene,
                title=f"Log Fold Change\n{condition1 or 'Condition 1'} to {condition2 or 'Condition 2'}",
                layer=fold_change_layer,
                color_map=cmap_fold_change,
                show=False,
                ax=axs[1, 1],
                **scatter_kwargs
            )
        elif basis:
            # Handle both sparse and dense layers
            fold_changes = adata.layers[fold_change_layer][:, gene_idx]
            if hasattr(fold_changes, 'toarray'):
                fold_changes = fold_changes.toarray().flatten()
            else:
                fold_changes = fold_changes.flatten()
                
            # Create scatter kwargs - use a small default point size
            scatter_kwargs = {'s': 3, 'vcenter': 0}
            # Override with user-provided kwargs
            scatter_kwargs.update(kwargs)
                
            scatter3 = axs[1, 1].scatter(
                adata.obsm[basis][:, 0],
                adata.obsm[basis][:, 1],
                c=fold_changes,
                cmap=cmap_fold_change,
                **scatter_kwargs
            )
            plt.colorbar(scatter3, ax=axs[1, 1])
            axs[1, 1].set_title(f"Log Fold Change\n{condition1 or 'Condition 1'} to {condition2 or 'Condition 2'}")
            axs[1, 1].set_xlabel("UMAP 1")
            axs[1, 1].set_ylabel("UMAP 2")
            axs[1, 1].grid(False)
            axs[1, 1].spines['top'].set_visible(False)
            axs[1, 1].spines['right'].set_visible(False)
        else:
            # Handle both sparse and dense layers
            fold_changes = adata.layers[fold_change_layer][:, gene_idx]
            if hasattr(fold_changes, 'toarray'):
                fold_changes = fold_changes.toarray().flatten()
            else:
                fold_changes = fold_changes.flatten()
                
            # Sort cells by fold change value (absolute magnitude)
            sort_idx = np.argsort(np.abs(fold_changes))
            sorted_fc = fold_changes[sort_idx]
                
            # Create scatter kwargs - use a small default point size
            scatter_kwargs = {'s': 3}
            # Override with user-provided kwargs
            scatter_kwargs.update(kwargs)
                
            scatter3 = axs[1, 1].scatter(
                np.arange(len(sorted_fc)),
                sorted_fc,
                c=sorted_fc,
                cmap=cmap_fold_change,
                **scatter_kwargs
            )
            plt.colorbar(scatter3, ax=axs[1, 1])
            
            axs[1, 1].set_title(f"Log Fold Change\n{condition1 or 'Condition 1'} to {condition2 or 'Condition 2'}")
            axs[1, 1].set_xlabel("Cell index")
            axs[1, 1].set_ylabel("Fold Change")
            axs[1, 1].grid(False)
            axs[1, 1].spines['top'].set_visible(False)
            axs[1, 1].spines['right'].set_visible(False)
    else:
        axs[1, 1].set_title(f"Log Fold Change: Not available")
        axs[1, 1].set_axis_off()
    
    # Adjust layout
    plt.tight_layout()
    
    if save is not None:
        fig.savefig(save, bbox_inches="tight", dpi=300)
    if return_fig:
        return fig
    return None