"""
Integrated workflows for AnnData objects.
"""

import logging
import numpy as np
import pandas as pd
import datetime
from typing import Optional, Union, Dict, Any, List, Tuple

from .differential_abundance import compute_differential_abundance
from .differential_expression import compute_differential_expression
from .core import _sanitize_name
from ..reporter import HTMLReporter
from ..utils import get_environment_info

logger = logging.getLogger("kompot")


def run_differential_analysis(
    adata,
    groupby: str,
    condition1: str,
    condition2: str,
    obsm_key: str = "DM_EigenVectors",
    layer: Optional[str] = None,
    genes: Optional[List[str]] = None,
    n_landmarks: Optional[int] = None,
    landmarks: Optional[np.ndarray] = None,
    compute_abundance: bool = True,
    compute_expression: bool = True,
    abundance_key: str = "kompot_da",
    expression_key: str = "kompot_de",
    share_landmarks: bool = True,
    ls_factor: float = 10.0,
    jit_compile: bool = False,
    random_state: Optional[int] = None,
    copy: bool = False,
    generate_html_report: bool = True,
    report_dir: str = "kompot_report",
    open_browser: bool = True,
    overwrite: Optional[bool] = None,
    store_landmarks: bool = False,
    **kwargs
) -> Dict[str, Any]:
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
        by default "DM_EigenVectors".
    layer : str, optional
        Layer in adata.layers containing gene expression data. If None, use adata.X,
        by default None.
    genes : List[str], optional
        List of gene names to include in the analysis. If None, use all genes,
        by default None.
    n_landmarks : int, optional
        Number of landmarks to use for approximation. If None, use all points,
        by default None. Ignored if landmarks is provided.
    landmarks : np.ndarray, optional
        Pre-computed landmarks to use. If provided, n_landmarks will be ignored.
        Shape (n_landmarks, n_features).
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
    share_landmarks : bool, optional
        Whether to share landmarks between abundance and expression analyses,
        by default True.
    ls_factor : float, optional
        Multiplication factor to apply to length scale when it's automatically inferred,
        by default 10.0. Only used when length scale is not explicitly provided.
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
    overwrite : bool, optional
        Controls behavior when results with the same result_key already exist:
        
        - If None (default): Warn about existing results but proceed with overwriting
        - If True: Silently overwrite existing results
        - If False: Raise an error if results would be overwritten
    store_landmarks : bool, optional
        Whether to store landmarks in adata.uns for future reuse, by default False.
        Setting to True will allow reusing landmarks with future analyses but may 
        significantly increase the AnnData file size.
    **kwargs : dict
        Additional arguments to pass to compute_differential_abundance and 
        compute_differential_expression.
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
            - "adata": The AnnData object with analysis results added (a new object if copy=True)
            - "differential_abundance": The DifferentialAbundance model if compute_abundance=True
            - "differential_expression": The DifferentialExpression model if compute_expression=True
    
    Notes
    -----
    This function runs the full Kompot differential analysis workflow and provides
    a simplified interface for both differential abundance and expression analysis.
    Results are stored in the AnnData object's obs, var, layers, and uns attributes.

    The `log_fold_change_direction` column in adata.obs is assigned categorical values
    ('up', 'down', or 'neutral'), and matching colors are stored in adata.uns with
    the '_colors' postfix for easy visualization in scanpy and other tools:
    
    .. code-block:: python
    
        # Color scheme used
        direction_colors = {"up": "#d73027", "down": "#4575b4", "neutral": "#d3d3d3"}
        
        # This allows direct use with scanpy's plotting functions
        import scanpy as sc
        sc.pl.umap(adata, color=f"{abundance_key}_log_fold_change_direction")
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
        'log_fold_change_threshold', 'pvalue_threshold', 'batch_size', 'sample_col'
    ]}
    
    expression_kwargs = {k: v for k, v in kwargs.items() if k in [
        'sample_col', 'compute_weighted_fold_change', 'sigma', 'ls',
        'compute_mahalanobis'
    ]}
    
    report_kwargs = {k: v for k, v in kwargs.items() if k in [
        'title', 'subtitle', 'template_dir', 'use_cdn', 'top_n', 'groupby', 'embedding_key'
    ]}
    
    # Run differential abundance if requested
    abundance_result = None
    abundance_landmarks = None
    if compute_abundance:
        logger.info("Computing differential abundance...")
        abundance_result = compute_differential_abundance(
            adata,
            groupby=groupby,
            condition1=condition1,
            condition2=condition2,
            obsm_key=obsm_key,
            n_landmarks=n_landmarks,
            landmarks=landmarks,
            ls_factor=ls_factor,
            jit_compile=jit_compile,
            random_state=random_state,
            inplace=True,
            result_key=abundance_key,
            overwrite=overwrite,
            store_landmarks=store_landmarks,
            **abundance_kwargs
        )
        
        # Check if landmarks are stored in abundance_key
        if store_landmarks and abundance_key in adata.uns and 'landmarks' in adata.uns[abundance_key]:
            abundance_landmarks = adata.uns[abundance_key]['landmarks']
            if share_landmarks:
                logger.info(f"Will reuse landmarks from differential abundance analysis for expression analysis")
                # Update abundance_key uns to indicate landmarks were shared
                if 'params' in adata.uns[abundance_key]:
                    adata.uns[abundance_key]['params']['landmarks_shared_with_expression'] = True
        elif share_landmarks and abundance_result is not None and 'model' in abundance_result and hasattr(abundance_result['model'], 'computed_landmarks'):
            # If landmarks weren't stored but are available in the model, use them for sharing
            abundance_landmarks = abundance_result['model'].computed_landmarks
            logger.info(f"Using landmarks from abundance model for expression analysis (not stored in adata)")
    
    # Run differential expression if requested
    expression_result = None
    if compute_expression:
        logger.info("Computing differential expression...")
        # Check if the abundance_key log density fields exist
        diff_abund_key = abundance_key if compute_abundance else None
        
        # Make sure log density columns exist if we're trying to use differential abundance key
        if diff_abund_key is not None:
            # Sanitize condition names for use in column names
            cond1_safe = _sanitize_name(condition1)
            cond2_safe = _sanitize_name(condition2)
            
            # Check for condition-specific column names
            specific_cols = [f"{diff_abund_key}_log_density_{cond1_safe}", f"{diff_abund_key}_log_density_{cond2_safe}"]
            
            if not all(col in adata.obs for col in specific_cols):
                logger.warning(f"Log density columns not found in adata.obs. "
                              f"Expected: {specific_cols}. "
                              f"Will not compute weighted mean fold changes.")
                diff_abund_key = None
        
        # Use abundance landmarks for expression analysis if available and sharing is enabled
        expr_landmarks = landmarks
        if share_landmarks and abundance_landmarks is not None:
            expr_landmarks = abundance_landmarks
        
        expression_result = compute_differential_expression(
            adata,
            groupby=groupby,
            condition1=condition1,
            condition2=condition2,
            obsm_key=obsm_key,
            layer=layer,
            genes=genes,
            n_landmarks=n_landmarks,
            landmarks=expr_landmarks,
            ls_factor=ls_factor,
            jit_compile=jit_compile,
            random_state=random_state,
            differential_abundance_key=diff_abund_key,
            inplace=True,
            result_key=expression_key,
            overwrite=overwrite,
            store_landmarks=store_landmarks,
            **expression_kwargs
        )
    
    # Generate HTML report if requested
    if generate_html_report and compute_expression and expression_result is not None:
        logger.info("Generating HTML report...")
        # Get the model from the expression_result
        diff_expr = expression_result["model"]
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
        
    # Store information about landmark sharing in adata.uns
    if compute_abundance and compute_expression and share_landmarks and abundance_landmarks is not None:
        # Create a shared landmarks entry if needed
        if 'landmarks_info' not in adata.uns:
            adata.uns['landmarks_info'] = {}
        
        # Record that landmarks were shared between abundance and expression
        adata.uns['landmarks_info']['shared_between_analyses'] = True
        adata.uns['landmarks_info']['source'] = abundance_key
        adata.uns['landmarks_info']['targets'] = [expression_key]
        
        # Store timestamp of sharing
        adata.uns['landmarks_info']['timestamp'] = datetime.datetime.now().isoformat()
        
        # Record that landmarks were shared during computation, but don't synchronize storage
        if store_landmarks:
            logger.info(f"Landmarks shared during computation between {abundance_key} and {expression_key}")
    
    # Store combined run information with enhanced environment data
    # Get environment info
    env_info = get_environment_info()
    
    # Create a kompot_run_history entry if it doesn't exist
    if 'kompot_run_history' not in adata.uns:
        adata.uns['kompot_run_history'] = []
    
    # Add current run info to the history with a run_id for reference
    run_id = len(adata.uns['kompot_run_history'])
    
    # Create parameters dictionary
    parameters_dict = {
        "obsm_key": obsm_key,
        "share_landmarks": share_landmarks,
        "ls_factor": ls_factor,
        "generate_html_report": generate_html_report,
        "groupby": groupby,
        "condition1": condition1,
        "condition2": condition2,
        "layer": layer,
        "genes": genes,
        "n_landmarks": n_landmarks,
        "compute_abundance": compute_abundance,
        "compute_expression": compute_expression
    }
    
    run_info = {
        "run_id": run_id,
        "timestamp": env_info["timestamp"],
        "function": "run_differential_analysis",
        "analysis_type": "combined",
        "result_key": abundance_key if compute_abundance else (expression_key if compute_expression else None),
        "abundance_key": abundance_key if compute_abundance else None,  # Keep for reference
        "expression_key": expression_key if compute_expression else None,  # Keep for reference
        "conditions": {
            "groupby": groupby,
            "condition1": condition1,
            "condition2": condition2
        },
        "params": parameters_dict,
        "environment": env_info  # Add all environment info
    }
    
    # Add to global history
    adata.uns['kompot_run_history'].append(run_info)
    
    # Also add to the appropriate specific history locations
    if compute_abundance:
        # Make sure kompot_da exists and has a run_history
        if 'kompot_da' not in adata.uns:
            adata.uns['kompot_da'] = {}
        if 'run_history' not in adata.uns['kompot_da']:
            adata.uns['kompot_da']['run_history'] = []
        
        # Add to DA specific history
        adata.uns['kompot_da']['run_history'].append(run_info)
    
    if compute_expression:
        # Make sure kompot_de exists and has a run_history
        if 'kompot_de' not in adata.uns:
            adata.uns['kompot_de'] = {}
        if 'run_history' not in adata.uns['kompot_de']:
            adata.uns['kompot_de']['run_history'] = []
        
        # Add to DE specific history
        adata.uns['kompot_de']['run_history'].append(run_info)
    
    # Store the latest run as a separate key for easy access
    adata.uns['kompot_latest_run'] = run_info
    
    # Note: The individual analyses (DA and DE) already store their runs 
    # in the fixed storage locations during their execution.
    # The global history is maintained separately for combined runs.
    
    # Return the results along with the AnnData
    result_dict = {
        "adata": adata,
        "differential_abundance": abundance_result["model"] if abundance_result else None,
        "differential_expression": expression_result["model"] if expression_result else None,
    }
    
    # Add landmarks if they were computed and shared
    if abundance_landmarks is not None:
        result_dict["landmarks"] = abundance_landmarks
    elif compute_expression and expression_result and "landmarks" in expression_result:
        result_dict["landmarks"] = expression_result["landmarks"]
    
    return result_dict


def generate_report(
    diff_expr,
    output_dir: str,
    adata,
    condition1_name: str,
    condition2_name: str,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    template_dir: Optional[str] = None,
    use_cdn: bool = True,
    open_browser: bool = True,
    top_n: int = 50,
    embedding_key: Optional[str] = None,
    groupby: Optional[str] = None,
    **kwargs,
) -> str:
    """Generate interactive HTML report for differential expression results.

    Parameters
    ----------
    diff_expr : DifferentialExpression
        The fitted DifferentialExpression model.
    output_dir : str
        Directory to save the HTML report.
    adata : AnnData
        The AnnData object used for analysis.
    condition1_name : str
        Name of condition 1.
    condition2_name : str
        Name of condition 2.
    title : str, optional
        Report title.
    subtitle : str, optional
        Report subtitle.
    template_dir : str, optional
        Directory containing custom templates.
    use_cdn : bool, optional
        Use CDN for JavaScript libraries instead of local files, by default True.
    open_browser : bool, optional
        Whether to open browser after generating the report, by default True.
    top_n : int, optional
        Number of top genes to include in the report, by default 50.
    embedding_key : str, optional
        Key in adata.obsm for embedding coordinates, by default None.
    groupby : str, optional
        Column in adata.obs to use for coloring, by default None.
    **kwargs
        Additional arguments for HTMLReporter.

    Returns
    -------
    str
        Path to the generated HTML report.
    """
    # Set default title if not provided
    if title is None:
        title = f"Differential Expression: {condition1_name} vs {condition2_name}"
    
    # Set default subtitle if not provided
    if subtitle is None:
        subtitle = f"Generated with Kompot on {datetime.datetime.now().strftime('%Y-%m-%d')}"
    
    # Create reporter
    reporter = HTMLReporter(
        diff_expr=diff_expr,
        output_dir=output_dir,
        adata=adata,
        condition1_name=condition1_name,
        condition2_name=condition2_name,
        title=title,
        subtitle=subtitle,
        template_dir=template_dir,
        use_cdn=use_cdn,
        top_n=top_n,
        embedding_key=embedding_key,
        groupby=groupby,
        **kwargs
    )
    
    # Generate report
    return reporter.generate_report(open_browser=open_browser)