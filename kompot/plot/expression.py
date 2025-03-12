"""Functions for visualizing expression patterns for individual genes."""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Union, List, Tuple, Dict, Any, Literal
from anndata import AnnData
import warnings
import logging

from ..utils import get_run_from_history, KOMPOT_COLORS
from .volcano import _extract_conditions_from_key, _infer_de_keys

# Get the pre-configured logger
logger = logging.getLogger("kompot")

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
) -> Optional[Tuple[plt.Figure, plt.Axes]]:
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
    
    # Infer keys using helper function
    lfc_key, score_key = _infer_de_keys(adata, run_id, lfc_key, score_key)
    
    # Calculate the actual (positive) run ID for logging - same as volcano_da
    if run_id < 0:
        if 'kompot_de' in adata.uns and 'run_history' in adata.uns['kompot_de']:
            actual_run_id = len(adata.uns['kompot_de']['run_history']) + run_id
        else:
            actual_run_id = run_id
    else:
        actual_run_id = run_id
    
    # Get run info for both conditions and layer
    run_info = get_run_from_history(adata, run_id, analysis_type="de")
    
    # Extract conditions from lfc_key if not provided
    if condition1 is None or condition2 is None:
        # Try to extract conditions from the key name
        conditions = _extract_conditions_from_key(lfc_key)
        if conditions:
            condition1, condition2 = conditions
        else:
            # If not in key, try getting from run info
            if run_info is not None and 'params' in run_info:
                params = run_info['params']
                if 'conditions' in params and len(params['conditions']) == 2:
                    condition1 = params['conditions'][0]
                    condition2 = params['conditions'][1]
    
    # Log which run and fields are being used - same pattern as volcano_da
    conditions_str = f": comparing {condition1} vs {condition2}" if condition1 and condition2 else ""
    logger.info(f"Using DE run {actual_run_id}{conditions_str}")
    logger.info(f"Using fields for gene expression plot - lfc_key: '{lfc_key}', score_key: '{score_key}'")
    
    # Try to infer layer from run_info if not explicitly provided
    if layer is None and run_info is not None and 'params' in run_info:
        params = run_info['params']
        if 'layer' in params:
            inferred_layer = params['layer']
            # Don't use fold_change layers for expression visualization
            if "fold_change" not in inferred_layer:
                layer = inferred_layer
                logger.info(f"Using layer '{layer}' inferred from run information")
            else:
                logger.info(f"Ignoring fold_change layer '{inferred_layer}' inferred from run, using adata.X instead")
            
    # Extract fold change and score for the gene
    gene_lfc = adata.var.loc[gene, lfc_key] if lfc_key in adata.var else "Unknown"
    gene_score = adata.var.loc[gene, score_key] if score_key in adata.var else "Unknown"
    
    # Check if embedding basis exists
    if basis not in adata.obsm:
        logger.warning(f"Basis '{basis}' not found in adata.obsm. Available bases: {list(adata.obsm.keys())}")
        logger.warning("Falling back to standard coordinates.")
        basis = None
        
    # Determine condition-specific imputed expression layer names
    condition1_layer = None
    condition2_layer = None
    fold_change_layer = None
    
    # First try to get them from run_info
    if run_info is not None:
        # Check if imputed_layer_keys is directly in run_info
        if 'imputed_layer_keys' in run_info:
            imputed_keys = run_info['imputed_layer_keys']
            # The keys may be named condition1/condition2, but the actual imputed layers
            # correspond to the specific condition names (e.g., Young/Old)
            if 'condition1' in imputed_keys:
                # Map to the correct condition based on run_info
                if run_info.get('params', {}).get('condition1') == condition1:
                    condition1_layer = imputed_keys['condition1']
                    logger.info(f"Using condition1 imputed layer '{condition1_layer}' for '{condition1}'")
                elif run_info.get('params', {}).get('condition1') == condition2:
                    condition2_layer = imputed_keys['condition1']
                    logger.info(f"Using condition1 imputed layer '{condition2_layer}' for '{condition2}'")
                
            if 'condition2' in imputed_keys:
                # Map to the correct condition based on run_info
                if run_info.get('params', {}).get('condition2') == condition2:
                    condition2_layer = imputed_keys['condition2']
                    logger.info(f"Using condition2 imputed layer '{condition2_layer}' for '{condition2}'")
                elif run_info.get('params', {}).get('condition2') == condition1:
                    condition1_layer = imputed_keys['condition2']
                    logger.info(f"Using condition2 imputed layer '{condition1_layer}' for '{condition1}'")
                
            if 'fold_change' in imputed_keys:
                fold_change_layer = imputed_keys['fold_change']
                logger.info(f"Using fold_change layer '{fold_change_layer}' from run_info")
        
        # Otherwise check field_names
        elif 'field_names' in run_info:
            field_names = run_info['field_names']
            params = run_info.get('params', {})
            
            # Map the imputed layers based on the actual condition names in params
            if 'imputed_key_1' in field_names and 'imputed_key_2' in field_names:
                # Get condition names from params
                param_condition1 = params.get('condition1')
                param_condition2 = params.get('condition2')
                
                # Map to the correct layers based on condition names
                if param_condition1 == condition1 and param_condition2 == condition2:
                    # Standard mapping
                    condition1_layer = field_names['imputed_key_1']
                    condition2_layer = field_names['imputed_key_2']
                    logger.info(f"Using imputed layer '{condition1_layer}' for '{condition1}'")
                    logger.info(f"Using imputed layer '{condition2_layer}' for '{condition2}'")
                elif param_condition1 == condition2 and param_condition2 == condition1:
                    # Reversed mapping
                    condition1_layer = field_names['imputed_key_2']
                    condition2_layer = field_names['imputed_key_1']
                    logger.info(f"Using imputed layer '{condition1_layer}' for '{condition1}'")
                    logger.info(f"Using imputed layer '{condition2_layer}' for '{condition2}'")
                else:
                    # Fall back to position-based mapping if condition names don't match
                    condition1_layer = field_names['imputed_key_1']
                    condition2_layer = field_names['imputed_key_2']
                    logger.info(f"Using imputed layer '{condition1_layer}' for condition 1")
                    logger.info(f"Using imputed layer '{condition2_layer}' for condition 2")
                    
            # Get fold change layer
            if 'fold_change_key' in field_names:
                fold_change_layer = field_names['fold_change_key']
                logger.info(f"Using fold_change layer '{fold_change_layer}' from field_names")
    
    # If layers still not found, try to find them by name pattern
    if condition1_layer is None or condition2_layer is None or fold_change_layer is None:
        logger.info("Some layers not found in run_info, searching by name pattern")
        for layer_name in adata.layers.keys():
            if condition1_layer is None and 'imputed' in layer_name and condition1 and condition1.lower() in layer_name.lower():
                condition1_layer = layer_name
                logger.info(f"Found condition1 imputed layer '{condition1_layer}' by name pattern")
            elif condition2_layer is None and 'imputed' in layer_name and condition2 and condition2.lower() in layer_name.lower():
                condition2_layer = layer_name
                logger.info(f"Found condition2 imputed layer '{condition2_layer}' by name pattern")
            elif fold_change_layer is None and 'fold_change' in layer_name:
                fold_change_layer = layer_name
                logger.info(f"Found fold_change layer '{fold_change_layer}' by name pattern")
            
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
                title=f"Log Fold Change\n{condition2 or 'Condition 2'} vs {condition1 or 'Condition 1'}",
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
            axs[1, 1].set_title(f"Log Fold Change\n{condition2 or 'Condition 2'} vs {condition1 or 'Condition 1'}")
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
            
            axs[1, 1].set_title(f"Log Fold Change\n{condition2 or 'Condition 2'} vs {condition1 or 'Condition 1'}")
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
    
    # Save figure if path provided
    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight")
        
    # Return figure and axes if requested
    if return_fig:
        return fig, axs
    elif save is None:
        # Only show if not saving and not returning
        plt.show()