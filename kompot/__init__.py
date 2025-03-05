"""
Kompot: A package for differential abundance and gene expression analysis
using Mahalanobis distance with JAX backend.
"""

import logging.config
import sys
from typing import Optional, Union, Dict, Any, List, Tuple

import numpy as np

from kompot.version import __version__

# Re-export Mellon tools directly
from mellon import DensityEstimator, FunctionEstimator, Predictor

# Export Kompot's additional functionality
from kompot.differential import DifferentialAbundance, DifferentialExpression
from kompot.utils import compute_mahalanobis_distance, find_landmarks
from kompot.reporter import HTMLReporter

# Configure logging
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "[%(asctime)s] [%(levelname)-8s] %(message)s",
        },
    },
    "handlers": {
        "console": {
            "level": "DEBUG",
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "stream": sys.stdout,
        },
    },
    "loggers": {
        "kompot": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
    },
}

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("kompot")

__all__ = [
    "DensityEstimator", "FunctionEstimator", "Predictor", 
    "DifferentialAbundance", "DifferentialExpression",
    "compute_mahalanobis_distance", "find_landmarks",
    "HTMLReporter", "generate_report", "__version__",
    "compute_differential_abundance", "compute_differential_expression",
    "run_differential_analysis"
]

def compute_differential_abundance(
    adata,
    groupby: str,
    condition1: str,
    condition2: str,
    obsm_key: str = "X_pca",
    n_landmarks: Optional[int] = None,
    log_fold_change_threshold: float = 1.7,
    pvalue_threshold: float = 1e-3,
    jit_compile: bool = False,
    random_state: Optional[int] = None,
    copy: bool = False,
    inplace: bool = True,
    result_key: str = "kompot_da",
    **density_kwargs
) -> Union[Dict[str, np.ndarray], "AnnData"]:
    """
    Compute differential abundance between two conditions directly from an AnnData object.
    
    This function is a scverse-compatible wrapper around the DifferentialAbundance class
    that operates directly on AnnData objects.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object containing cells from both conditions.
    groupby : str
        Column in adata.obs containing the condition labels.
    condition1 : str
        Label in the groupby column identifying the first condition.
    condition2 : str
        Label in the groupby column identifying the second condition.
    obsm_key : str, optional
        Key in adata.obsm containing the cell states (e.g., PCA, diffusion maps),
        by default "X_pca".
    n_landmarks : int, optional
        Number of landmarks to use for approximation. If None, use all points,
        by default None.
    log_fold_change_threshold : float, optional
        Threshold for considering a log fold change significant, by default 1.7.
    pvalue_threshold : float, optional
        Threshold for considering a p-value significant, by default 1e-3.
    jit_compile : bool, optional
        Whether to use JAX just-in-time compilation, by default False.
    random_state : int, optional
        Random seed for reproducible landmark selection when n_landmarks is specified.
        Controls the random selection of points when using approximation, by default None.
    copy : bool, optional
        If True, return a copy of the AnnData object with results added,
        by default False.
    inplace : bool, optional
        If True, modify adata in place, by default True.
    result_key : str, optional
        Key in adata.uns where results will be stored, by default "kompot_da".
    **density_kwargs : dict
        Additional arguments to pass to the DensityEstimator.
        
    Returns
    -------
    Union[Dict[str, np.ndarray], AnnData]
        If copy is True, returns a copy of the AnnData object with results added.
        If inplace is True, returns None (adata is modified in place).
        Otherwise, returns a dictionary of results.
    
    Notes
    -----
    Results are stored in various components of the AnnData object:
    
    - adata.obs[f"{result_key}_log_fold_change"]: Log fold change values for each cell
    - adata.obs[f"{result_key}_log_fold_change_zscore"]: Z-scores for each cell
    - adata.obs[f"{result_key}_log_fold_change_pvalue"]: P-values for each cell
    - adata.obs[f"{result_key}_log_fold_change_direction"]: Direction of change
    - adata.uns[result_key]: Dictionary with additional information and parameters
    """
    try:
        import anndata
    except ImportError:
        raise ImportError(
            "Please install anndata: pip install anndata"
        )
    
    # Make a copy if requested
    if copy:
        adata = adata.copy()
    
    # Extract cell states
    if obsm_key not in adata.obsm:
        raise ValueError(f"Key '{obsm_key}' not found in adata.obsm. Available keys: {list(adata.obsm.keys())}")
    
    if groupby not in adata.obs:
        raise ValueError(f"Column '{groupby}' not found in adata.obs. Available columns: {list(adata.obs.columns)}")
    
    # Create masks for each condition
    mask1 = adata.obs[groupby] == condition1
    mask2 = adata.obs[groupby] == condition2
    
    if np.sum(mask1) == 0:
        raise ValueError(f"Condition '{condition1}' not found in '{groupby}'.")
    if np.sum(mask2) == 0:
        raise ValueError(f"Condition '{condition2}' not found in '{groupby}'.")
    
    logger.info(f"Condition 1 ({condition1}): {np.sum(mask1)} cells")
    logger.info(f"Condition 2 ({condition2}): {np.sum(mask2)} cells")
    
    # Extract cell states for each condition
    X_condition1 = adata.obsm[obsm_key][mask1]
    X_condition2 = adata.obsm[obsm_key][mask2]
    
    # Initialize and fit DifferentialAbundance
    diff_abundance = DifferentialAbundance(
        log_fold_change_threshold=log_fold_change_threshold,
        pvalue_threshold=pvalue_threshold,
        n_landmarks=n_landmarks,
        jit_compile=jit_compile,
        random_state=random_state
    )
    
    diff_abundance.fit(X_condition1, X_condition2, **density_kwargs)
    
    # Prepare results to store in AnnData
    all_cells_mask = mask1 | mask2
    
    if inplace:
        # Add results to adata.obs
        adata.obs[f"{result_key}_log_fold_change"] = np.nan
        adata.obs[f"{result_key}_log_fold_change_zscore"] = np.nan
        adata.obs[f"{result_key}_log_fold_change_pvalue"] = np.nan
        adata.obs[f"{result_key}_log_fold_change_direction"] = ""
        
        # Assign values to masked cells
        adata.obs.loc[all_cells_mask, f"{result_key}_log_fold_change"] = diff_abundance.log_fold_change
        adata.obs.loc[all_cells_mask, f"{result_key}_log_fold_change_zscore"] = diff_abundance.log_fold_change_zscore
        adata.obs.loc[all_cells_mask, f"{result_key}_log_fold_change_pvalue"] = diff_abundance.log_fold_change_pvalue
        adata.obs.loc[all_cells_mask, f"{result_key}_log_fold_change_direction"] = diff_abundance.log_fold_change_direction
        
        # Store model and parameters in adata.uns
        adata.uns[result_key] = {
            "params": {
                "groupby": groupby,
                "condition1": condition1,
                "condition2": condition2,
                "obsm_key": obsm_key,
                "log_fold_change_threshold": log_fold_change_threshold,
                "pvalue_threshold": pvalue_threshold,
            },
            "model": diff_abundance,
        }
        
        if copy:
            return adata
        else:
            return None
    else:
        # Return results as a dictionary
        return {
            "log_fold_change": diff_abundance.log_fold_change,
            "log_fold_change_zscore": diff_abundance.log_fold_change_zscore,
            "log_fold_change_pvalue": diff_abundance.log_fold_change_pvalue,
            "log_fold_change_direction": diff_abundance.log_fold_change_direction,
            "model": diff_abundance,
        }


def compute_differential_expression(
    adata,
    groupby: str,
    condition1: str,
    condition2: str,
    obsm_key: str = "X_pca",
    layer: Optional[str] = None,
    genes: Optional[List[str]] = None,
    n_landmarks: Optional[int] = None,
    use_empirical_variance: bool = False,
    compute_weighted_fold_change: bool = True,
    compute_differential_abundance: bool = True,
    differential_abundance_key: Optional[str] = None,
    sigma: float = 1.0,
    ls: Optional[float] = None,
    compute_mahalanobis: bool = True,
    jit_compile: bool = False,
    random_state: Optional[int] = None,
    copy: bool = False,
    inplace: bool = True,
    result_key: str = "kompot_de",
    **function_kwargs
) -> Union[Dict[str, np.ndarray], "AnnData"]:
    """
    Compute differential expression between two conditions directly from an AnnData object.
    
    This function is a scverse-compatible wrapper around the DifferentialExpression class
    that operates directly on AnnData objects.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object containing cells from both conditions.
    groupby : str
        Column in adata.obs containing the condition labels.
    condition1 : str
        Label in the groupby column identifying the first condition.
    condition2 : str
        Label in the groupby column identifying the second condition.
    obsm_key : str, optional
        Key in adata.obsm containing the cell states (e.g., PCA, diffusion maps),
        by default "X_pca".
    layer : str, optional
        Layer in adata.layers containing gene expression data. If None, use adata.X,
        by default None.
    genes : List[str], optional
        List of gene names to include in the analysis. If None, use all genes,
        by default None.
    n_landmarks : int, optional
        Number of landmarks to use for approximation. If None, use all points,
        by default None.
    use_empirical_variance : bool, optional
        Whether to use empirical variance for uncertainty estimation, by default False.
    compute_weighted_fold_change : bool, optional
        Whether to compute weighted mean log fold change, by default True.
    compute_differential_abundance : bool, optional
        Whether to compute differential abundance if not already provided, by default True.
    differential_abundance_key : str, optional
        Key in adata.uns where differential abundance results are stored, by default None.
        If not None, will reuse existing results instead of computing new ones.
    sigma : float, optional
        Noise level for function estimator, by default 1.0.
    ls : float, optional
        Length scale for the GP kernel. If None, it will be estimated, by default None.
    compute_mahalanobis : bool, optional
        Whether to compute Mahalanobis distances for gene ranking, by default True.
    jit_compile : bool, optional
        Whether to use JAX just-in-time compilation, by default False.
    random_state : int, optional
        Random seed for reproducible landmark selection when n_landmarks is specified.
        Controls the random selection of points when using approximation, by default None.
    copy : bool, optional
        If True, return a copy of the AnnData object with results added,
        by default False.
    inplace : bool, optional
        If True, modify adata in place, by default True.
    result_key : str, optional
        Key in adata.uns where results will be stored, by default "kompot_de".
    **function_kwargs : dict
        Additional arguments to pass to the FunctionEstimator.
        
    Returns
    -------
    Union[Dict[str, np.ndarray], AnnData]
        If copy is True, returns a copy of the AnnData object with results added.
        If inplace is True, returns None (adata is modified in place).
        Otherwise, returns a dictionary of results.
    
    Notes
    -----
    Results are stored in various components of the AnnData object:
    
    - adata.var[f"{result_key}_mahalanobis"]: Mahalanobis distance for each gene
    - adata.var[f"{result_key}_weighted_lfc"]: Weighted mean log fold change for each gene
    - adata.var[f"{result_key}_lfc_std"]: Standard deviation of log fold change for each gene
    - adata.var[f"{result_key}_bidirectionality"]: Bidirectionality score for each gene
    - adata.layers[f"{result_key}_condition1_imputed"]: Imputed expression for condition 1
    - adata.layers[f"{result_key}_condition2_imputed"]: Imputed expression for condition 2
    - adata.layers[f"{result_key}_fold_change"]: Log fold change for each cell and gene
    - adata.uns[result_key]: Dictionary with additional information and parameters
    """
    try:
        import anndata
        from scipy import sparse
    except ImportError:
        raise ImportError(
            "Please install anndata and scipy: pip install anndata scipy"
        )
    
    # Make a copy if requested
    if copy:
        adata = adata.copy()
    
    # Extract cell states
    if obsm_key not in adata.obsm:
        raise ValueError(f"Key '{obsm_key}' not found in adata.obsm. Available keys: {list(adata.obsm.keys())}")
    
    if groupby not in adata.obs:
        raise ValueError(f"Column '{groupby}' not found in adata.obs. Available columns: {list(adata.obs.columns)}")
    
    # Create masks for each condition
    mask1 = adata.obs[groupby] == condition1
    mask2 = adata.obs[groupby] == condition2
    
    if np.sum(mask1) == 0:
        raise ValueError(f"Condition '{condition1}' not found in '{groupby}'.")
    if np.sum(mask2) == 0:
        raise ValueError(f"Condition '{condition2}' not found in '{groupby}'.")
    
    logger.info(f"Condition 1 ({condition1}): {np.sum(mask1)} cells")
    logger.info(f"Condition 2 ({condition2}): {np.sum(mask2)} cells")
    
    # Extract cell states for each condition
    X_condition1 = adata.obsm[obsm_key][mask1]
    X_condition2 = adata.obsm[obsm_key][mask2]
    
    # Extract gene expression
    if layer is not None:
        if layer not in adata.layers:
            raise ValueError(f"Layer '{layer}' not found in adata.layers. Available layers: {list(adata.layers.keys())}")
        expr1 = adata.layers[layer][mask1]
        expr2 = adata.layers[layer][mask2]
    else:
        expr1 = adata.X[mask1]
        expr2 = adata.X[mask2]
    
    # Convert to dense if sparse
    if sparse.issparse(expr1):
        expr1 = expr1.toarray()
    if sparse.issparse(expr2):
        expr2 = expr2.toarray()
    
    # Filter genes if requested
    if genes is not None:
        if not all(gene in adata.var_names for gene in genes):
            missing_genes = [gene for gene in genes if gene not in adata.var_names]
            raise ValueError(f"The following genes were not found in adata.var_names: {missing_genes[:10]}" +
                          (f"... and {len(missing_genes) - 10} more" if len(missing_genes) > 10 else ""))
        
        gene_indices = [list(adata.var_names).index(gene) for gene in genes]
        expr1 = expr1[:, gene_indices]
        expr2 = expr2[:, gene_indices]
        selected_genes = genes
    else:
        selected_genes = adata.var_names.tolist()
    
    # Get differential abundance if needed
    differential_abundance = None
    if compute_weighted_fold_change:
        if differential_abundance_key is not None and differential_abundance_key in adata.uns:
            # Reuse existing differential abundance results
            differential_abundance = adata.uns[differential_abundance_key]["model"]
            logger.info(f"Using existing differential abundance results from adata.uns['{differential_abundance_key}']")
        elif compute_differential_abundance:
            # Compute new differential abundance
            logger.info("Computing differential abundance for weighting...")
            differential_abundance = DifferentialAbundance(
                n_landmarks=n_landmarks,
                jit_compile=jit_compile,
                random_state=random_state
            )
            differential_abundance.fit(X_condition1, X_condition2)
    
    # Initialize and fit DifferentialExpression
    diff_expression = DifferentialExpression(
        n_landmarks=n_landmarks,
        use_empirical_variance=use_empirical_variance,
        compute_weighted_fold_change=compute_weighted_fold_change,
        differential_abundance=differential_abundance,
        jit_compile=jit_compile,
        random_state=random_state
    )
    
    diff_expression.fit(
        X_condition1, expr1,
        X_condition2, expr2,
        sigma=sigma,
        ls=ls,
        compute_mahalanobis=compute_mahalanobis,
        compute_differential_abundance=False,  # We've already handled this above
        **function_kwargs
    )
    
    if inplace:
        # Add gene-level metrics to adata.var
        adata.var[f"{result_key}_mahalanobis"] = np.nan
        adata.var.loc[selected_genes, f"{result_key}_mahalanobis"] = diff_expression.mahalanobis_distances
        
        if diff_expression.weighted_mean_log_fold_change is not None:
            adata.var[f"{result_key}_weighted_lfc"] = np.nan
            adata.var.loc[selected_genes, f"{result_key}_weighted_lfc"] = diff_expression.weighted_mean_log_fold_change
        
        adata.var[f"{result_key}_lfc_std"] = np.nan
        adata.var[f"{result_key}_bidirectionality"] = np.nan
        adata.var.loc[selected_genes, f"{result_key}_lfc_std"] = diff_expression.lfc_stds
        adata.var.loc[selected_genes, f"{result_key}_bidirectionality"] = diff_expression.bidirectionality
        
        # Add cell-gene level results
        all_cells_mask = mask1 | mask2
        n_cells = np.sum(all_cells_mask)
        n_selected_genes = len(selected_genes)
        
        # Process the data to match the shape of the full gene set
        if n_selected_genes < len(adata.var_names):
            # We need to expand the imputed data to the full gene set
            if f"{result_key}_condition1_imputed" not in adata.layers:
                adata.layers[f"{result_key}_condition1_imputed"] = np.zeros_like(adata.X)
            if f"{result_key}_condition2_imputed" not in adata.layers:
                adata.layers[f"{result_key}_condition2_imputed"] = np.zeros_like(adata.X)
            if f"{result_key}_fold_change" not in adata.layers:
                adata.layers[f"{result_key}_fold_change"] = np.zeros_like(adata.X)
            
            # Map the imputed values to the correct positions
            for i, gene in enumerate(selected_genes):
                gene_idx = list(adata.var_names).index(gene)
                adata.layers[f"{result_key}_condition1_imputed"][all_cells_mask, gene_idx] = diff_expression.condition1_imputed[:, i]
                adata.layers[f"{result_key}_condition2_imputed"][all_cells_mask, gene_idx] = diff_expression.condition2_imputed[:, i]
                adata.layers[f"{result_key}_fold_change"][all_cells_mask, gene_idx] = diff_expression.fold_change[:, i]
        else:
            # We're using all genes, so we can directly assign the arrays
            adata.layers[f"{result_key}_condition1_imputed"] = np.zeros_like(adata.X)
            adata.layers[f"{result_key}_condition2_imputed"] = np.zeros_like(adata.X)
            adata.layers[f"{result_key}_fold_change"] = np.zeros_like(adata.X)
            
            adata.layers[f"{result_key}_condition1_imputed"][all_cells_mask] = diff_expression.condition1_imputed
            adata.layers[f"{result_key}_condition2_imputed"][all_cells_mask] = diff_expression.condition2_imputed
            adata.layers[f"{result_key}_fold_change"][all_cells_mask] = diff_expression.fold_change
        
        # Store model and parameters in adata.uns
        adata.uns[result_key] = {
            "params": {
                "groupby": groupby,
                "condition1": condition1,
                "condition2": condition2,
                "obsm_key": obsm_key,
                "layer": layer,
                "genes": genes,
                "n_landmarks": n_landmarks,
                "use_empirical_variance": use_empirical_variance,
                "compute_weighted_fold_change": compute_weighted_fold_change,
            },
            "model": diff_expression,
        }
        
        if copy:
            return adata
        else:
            return None
    else:
        # Return results as a dictionary
        return {
            "mahalanobis_distances": diff_expression.mahalanobis_distances,
            "weighted_mean_log_fold_change": diff_expression.weighted_mean_log_fold_change,
            "lfc_stds": diff_expression.lfc_stds,
            "bidirectionality": diff_expression.bidirectionality,
            "condition1_imputed": diff_expression.condition1_imputed,
            "condition2_imputed": diff_expression.condition2_imputed,
            "fold_change": diff_expression.fold_change,
            "model": diff_expression,
        }


def run_differential_analysis(
    adata,
    groupby: str,
    condition1: str,
    condition2: str,
    obsm_key: str = "X_pca",
    layer: Optional[str] = None,
    genes: Optional[List[str]] = None,
    n_landmarks: Optional[int] = None,
    compute_abundance: bool = True,
    compute_expression: bool = True,
    abundance_key: str = "kompot_da",
    expression_key: str = "kompot_de",
    jit_compile: bool = False,
    random_state: Optional[int] = None,
    copy: bool = False,
    generate_html_report: bool = True,
    report_dir: str = "kompot_report",
    open_browser: bool = True,
    **kwargs
) -> "AnnData":
    """
    Run a complete differential analysis workflow on an AnnData object.
    
    This function computes both differential abundance and differential expression
    between two conditions and stores the results in the AnnData object.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object containing cells from both conditions.
    groupby : str
        Column in adata.obs containing the condition labels.
    condition1 : str
        Label in the groupby column identifying the first condition.
    condition2 : str
        Label in the groupby column identifying the second condition.
    obsm_key : str, optional
        Key in adata.obsm containing the cell states (e.g., PCA, diffusion maps),
        by default "X_pca".
    layer : str, optional
        Layer in adata.layers containing gene expression data. If None, use adata.X,
        by default None.
    genes : List[str], optional
        List of gene names to include in the analysis. If None, use all genes,
        by default None.
    n_landmarks : int, optional
        Number of landmarks to use for approximation. If None, use all points,
        by default None.
    compute_abundance : bool, optional
        Whether to compute differential abundance, by default True.
    compute_expression : bool, optional
        Whether to compute differential expression, by default True.
    abundance_key : str, optional
        Key in adata.uns where differential abundance results will be stored,
        by default "kompot_da".
    expression_key : str, optional
        Key in adata.uns where differential expression results will be stored,
        by default "kompot_de".
    jit_compile : bool, optional
        Whether to use JAX just-in-time compilation, by default False.
    random_state : int, optional
        Random seed for reproducible landmark selection when n_landmarks is specified.
        Controls the random selection of points when using approximation, by default None.
    copy : bool, optional
        If True, return a copy of the AnnData object with results added,
        by default False.
    generate_html_report : bool, optional
        Whether to generate an HTML report with the results, by default True.
    report_dir : str, optional
        Directory where the HTML report will be saved, by default "kompot_report".
    open_browser : bool, optional
        Whether to open the HTML report in a browser, by default True.
    **kwargs : dict
        Additional arguments to pass to compute_differential_abundance and 
        compute_differential_expression.
        
    Returns
    -------
    AnnData
        The AnnData object with analysis results added. If copy is True, this is a
        new object; otherwise, it's the input object modified in place.
    
    Notes
    -----
    This function runs the full Kompot differential analysis workflow and provides
    a simplified interface for both differential abundance and expression analysis.
    Results are stored in the AnnData object's obs, var, layers, and uns attributes.
    """
    try:
        import anndata
    except ImportError:
        raise ImportError(
            "Please install anndata: pip install anndata"
        )
    
    # Make a copy if requested
    if copy:
        adata = adata.copy()
    
    # Separate kwargs for each analysis type
    abundance_kwargs = {k: v for k, v in kwargs.items() if k in [
        'log_fold_change_threshold', 'pvalue_threshold'
    ]}
    
    expression_kwargs = {k: v for k, v in kwargs.items() if k in [
        'use_empirical_variance', 'compute_weighted_fold_change', 'sigma', 'ls',
        'compute_mahalanobis'
    ]}
    
    report_kwargs = {k: v for k, v in kwargs.items() if k in [
        'title', 'subtitle', 'template_dir', 'use_cdn', 'top_n', 'groupby', 'embedding_key'
    ]}
    
    # Run differential abundance if requested
    if compute_abundance:
        logger.info("Computing differential abundance...")
        compute_differential_abundance(
            adata,
            groupby=groupby,
            condition1=condition1,
            condition2=condition2,
            obsm_key=obsm_key,
            n_landmarks=n_landmarks,
            jit_compile=jit_compile,
            random_state=random_state,
            inplace=True,
            result_key=abundance_key,
            **abundance_kwargs
        )
    
    # Run differential expression if requested
    if compute_expression:
        logger.info("Computing differential expression...")
        compute_differential_expression(
            adata,
            groupby=groupby,
            condition1=condition1,
            condition2=condition2,
            obsm_key=obsm_key,
            layer=layer,
            genes=genes,
            n_landmarks=n_landmarks,
            jit_compile=jit_compile,
            random_state=random_state,
            differential_abundance_key=abundance_key if compute_abundance else None,
            inplace=True,
            result_key=expression_key,
            **expression_kwargs
        )
    
    # Generate HTML report if requested
    if generate_html_report and compute_expression:
        logger.info("Generating HTML report...")
        diff_expr = adata.uns[expression_key]["model"]
        report_path = generate_report(
            diff_expr,
            output_dir=report_dir,
            adata=adata,
            condition1_name=condition1,
            condition2_name=condition2,
            open_browser=open_browser,
            **report_kwargs
        )
        logger.info(f"HTML report generated at: {report_path}")
    
    return adata


def generate_report(
    diff_expr,
    output_dir="kompot_report",
    adata=None,
    condition1_name="Condition 1",
    condition2_name="Condition 2",
    **kwargs
):
    """Generate an interactive HTML report for differential expression results.
    
    Parameters
    ----------
    diff_expr : DifferentialExpression
        DifferentialExpression object with results
    output_dir : str, optional
        Directory where the report will be saved, by default "kompot_report"
    adata : AnnData, optional
        AnnData object with cell annotations, by default None
    condition1_name : str, optional
        Name of the first condition, by default "Condition 1"
    condition2_name : str, optional
        Name of the second condition, by default "Condition 2"
    **kwargs : dict
        Additional arguments to pass to HTMLReporter or to reporter methods
        
    Returns
    -------
    str
        Path to the generated report
    """
    # Extract parameters for different methods
    reporter_params = {k: v for k, v in kwargs.items() if k in [
        'title', 'subtitle', 'template_dir', 'use_cdn'
    ]}
    
    diff_expr_params = {k: v for k, v in kwargs.items() if k in [
        'gene_names', 'top_n'
    ]}
    
    anndata_params = {k: v for k, v in kwargs.items() if k in [
        'groupby', 'embedding_key', 'cell_annotations'
    ]}
    
    # Whether to open browser at the end
    open_browser = kwargs.get("open_browser", True)
    
    # Create reporter
    reporter = HTMLReporter(output_dir=output_dir, **reporter_params)
    
    # Add differential expression results
    reporter.add_differential_expression(
        diff_expr,
        condition1_name=condition1_name,
        condition2_name=condition2_name,
        **diff_expr_params
    )
    
    if adata is not None:
        # Try to use default parameters, but allow override
        groupby = anndata_params.get("groupby", "leiden" if hasattr(adata.obs, "get") and adata.obs.get("leiden") is not None else "louvain")
        embedding_key = anndata_params.get("embedding_key", "X_umap" if hasattr(adata.obsm, "get") and adata.obsm.get("X_umap") is not None else "X_tsne")
        
        # Add AnnData only if the required columns exist
        if hasattr(adata.obs, "get") and adata.obs.get(groupby) is not None and hasattr(adata.obsm, "get") and adata.obsm.get(embedding_key) is not None:
            cell_annotations = anndata_params.get("cell_annotations", None)
            reporter.add_anndata(
                adata,
                groupby=groupby, 
                embedding_key=embedding_key,
                cell_annotations=cell_annotations
            )
    
    return reporter.generate(open_browser=open_browser)