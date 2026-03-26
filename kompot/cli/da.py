"""CLI command for differential abundance analysis."""
import argparse
import sys
from pathlib import Path
import logging

import anndata as ad

from ..anndata import da
from ..settings import GPSettings, DAThresholdSettings, StorageSettings, OutputSettings
from .utils import load_config, merge_args_with_config, validate_anndata_path
from .compute_config import configure_compute


logger = logging.getLogger("kompot.cli")


def add_da_parser(subparsers) -> argparse.ArgumentParser:
    """
    Add differential abundance subcommand parser.

    Parameters
    ----------
    subparsers
        Subparsers object from argparse

    Returns
    -------
    argparse.ArgumentParser
        The DA parser
    """
    parser = subparsers.add_parser(
        'da',
        help='Differential abundance analysis',
        description='Compute differential abundance between two conditions'
    )

    # Required arguments
    parser.add_argument(
        'input',
        type=str,
        help='Input AnnData file (.h5ad or .zarr)'
    )

    # Output
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Output AnnData file (.h5ad or .zarr). Required unless --table-output is specified.'
    )

    parser.add_argument(
        '-t', '--table-output',
        type=str,
        help='Output only the DA results as a table (.csv or .tsv). Contains cell-level statistics from adata.obs.'
    )

    # Config file
    parser.add_argument(
        '-c', '--config',
        type=str,
        help='YAML or JSON config file with advanced parameters'
    )

    # Basic required parameters
    parser.add_argument(
        '--groupby',
        type=str,
        help='Column in adata.obs containing condition labels'
    )

    parser.add_argument(
        '--condition1',
        type=str,
        help='Label for first condition (reference)'
    )

    parser.add_argument(
        '--condition2',
        type=str,
        help='Label for second condition (comparison)'
    )

    # Common optional parameters
    parser.add_argument(
        '--obsm-key',
        type=str,
        help='Key in adata.obsm for cell states (default: DM_EigenVectors)'
    )

    parser.add_argument(
        '--result-key',
        type=str,
        help='Key for storing results in adata.uns (default: kompot_da)'
    )

    parser.add_argument(
        '--n-landmarks',
        type=int,
        help='Number of landmarks for approximation (default: None, use all points)'
    )

    parser.add_argument(
        '--sample-col',
        type=str,
        help='Column in adata.obs with sample labels for sample variance estimation'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        help='Batch size for memory-efficient processing (default: 0, no batching)'
    )

    parser.add_argument(
        '--log-fold-change-threshold',
        type=float,
        help='Threshold for log fold change significance (default: 1.0)'
    )

    parser.add_argument(
        '--ptp-threshold',
        type=float,
        help='Posterior tail probability threshold (default: 0.05)'
    )

    parser.add_argument(
        '--ls-factor',
        type=float,
        help='Length scale multiplication factor (default: 10.0)'
    )

    parser.add_argument(
        '--random-state',
        type=int,
        help='Random seed for reproducible landmark selection (default: None)'
    )

    # Boolean flags
    parser.add_argument(
        '--store-landmarks',
        action='store_true',
        help='Store landmarks in AnnData for reuse'
    )

    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing results without warning'
    )

    parser.add_argument(
        '--no-progress',
        action='store_true',
        help='Disable progress bars'
    )

    # Compute configuration
    parser.add_argument(
        '--use-gpu',
        action='store_true',
        help='Use GPU for computation (requires CUDA-enabled JAX)'
    )

    parser.add_argument(
        '--threads',
        type=int,
        help='Number of threads to use for JAX, NumPy, and Dask (default: all available cores)'
    )

    parser.set_defaults(func=run_da)

    return parser


def run_da(args):
    """
    Run differential abundance analysis.

    Parameters
    ----------
    args
        Parsed arguments from argparse
    """
    # Validate output arguments
    if not args.output and not args.table_output:
        logger.error("Either --output or --table-output must be specified")
        sys.exit(1)

    # Validate input file
    input_path = validate_anndata_path(args.input)

    logger.info(f"Loading AnnData from {input_path}")
    adata = ad.read_h5ad(input_path) if str(input_path).endswith('.h5ad') else ad.read_zarr(input_path)
    logger.info(f"Loaded AnnData: {adata.n_obs} cells × {adata.n_vars} genes")

    # Load config if provided
    config = {}
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)

    # Configure compute resources (must be done AFTER mellon import in compute_differential_abundance)
    # Extract compute config before other processing
    use_gpu = getattr(args, 'use_gpu', False)
    n_threads = getattr(args, 'threads', None)

    # Log configuration before compute setup
    if use_gpu:
        logger.info("GPU acceleration: ENABLED")
    else:
        logger.info("GPU acceleration: DISABLED (using CPU)")
    if n_threads:
        logger.info(f"Thread limit: {n_threads}")
    else:
        logger.info("Thread limit: NONE (using all available cores)")

    # Convert args to dict, removing None values and CLI-specific args
    args_dict = {
        k: v for k, v in vars(args).items()
        if v is not None and k not in ['input', 'output', 'table_output', 'config', 'func', 'verbose', 'command', 'use_gpu', 'threads']
    }

    # Handle no_progress -> progress conversion
    if 'no_progress' in args_dict:
        args_dict['progress'] = not args_dict.pop('no_progress')

    # Merge with config (CLI args take precedence)
    params = merge_args_with_config(args_dict, config)

    # Validate required parameters
    required = ['groupby', 'condition1', 'condition2']
    missing = [p for p in required if p not in params]
    if missing:
        logger.error(f"Missing required parameters: {', '.join(missing)}")
        logger.error("Provide them via CLI arguments or config file")
        sys.exit(1)

    logger.info("Starting differential abundance analysis")
    logger.info(f"  Groupby: {params['groupby']}")
    logger.info(f"  Condition 1: {params['condition1']}")
    logger.info(f"  Condition 2: {params['condition2']}")
    logger.info(f"  ObsM key: {params.get('obsm_key', 'X_pca')}")

    # Configure computational backend
    # This must be called AFTER mellon import (which happens in compute_differential_abundance)
    # So we do a "lazy" import here to trigger mellon import, then configure
    logger.info("")
    logger.info("Configuring computational backend...")
    try:
        # Import mellon to trigger its JAX configuration
        import mellon
        # Now configure our settings (will override mellon's CPU-only default if needed)
        configure_compute(use_gpu=use_gpu, n_threads=n_threads)
    except Exception as e:
        logger.warning(f"Could not configure compute backend: {e}")
        logger.warning("Proceeding with default configuration")
    logger.info("")

    # Extract required params
    groupby = params.pop('groupby')
    condition1 = params.pop('condition1')
    condition2 = params.pop('condition2')
    obsm_key = params.pop('obsm_key', 'DM_EigenVectors')
    sample_col = params.pop('sample_col', None)

    # Build Settings from remaining params
    gp_keys = {'n_landmarks', 'landmarks', 'ls_factor', 'batch_size',
               'jit_compile', 'random_state'}
    gp_kwargs = {k: params.pop(k) for k in list(params) if k in gp_keys}
    gp = GPSettings(**gp_kwargs) if gp_kwargs else None

    threshold_kwargs = {}
    if 'log_fold_change_threshold' in params:
        threshold_kwargs['lfc_threshold'] = params.pop('log_fold_change_threshold')
    if 'ptp_threshold' in params:
        threshold_kwargs['ptp_threshold'] = params.pop('ptp_threshold')
    threshold = DAThresholdSettings(**threshold_kwargs) if threshold_kwargs else None

    storage_keys = {'result_key', 'overwrite', 'store_landmarks'}
    storage_kwargs = {k: params.pop(k) for k in list(params) if k in storage_keys}
    storage = StorageSettings(**storage_kwargs) if storage_kwargs else None

    output_keys = {'copy', 'inplace', 'return_full_results', 'allow_single_condition_variance', 'progress'}
    output_kwargs = {k: params.pop(k) for k in list(params) if k in output_keys}

    # Handle return_full_results for table output
    if args.table_output:
        output_kwargs['return_full_results'] = True
    output = OutputSettings(**output_kwargs) if output_kwargs else None

    # Run analysis
    try:
        result_dict = da(
            adata,
            groupby=groupby,
            condition1=condition1,
            condition2=condition2,
            obsm_key=obsm_key,
            sample_col=sample_col,
            gp=gp,
            threshold=threshold,
            storage=storage,
            output=output,
            **params,  # remaining params forwarded as density_kwargs
        )
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise

    # Save AnnData output if specified
    if args.output:
        output_path = Path(args.output)
        logger.info(f"Saving results to {output_path}")

        if str(output_path).endswith('.h5ad'):
            adata.write_h5ad(output_path)
        elif str(output_path).endswith('.zarr'):
            adata.write_zarr(output_path)
        else:
            logger.error(f"Unsupported output format: {output_path.suffix}. Use .h5ad or .zarr")
            sys.exit(1)

    # Save table output if specified
    if args.table_output:
        table_path = Path(args.table_output)
        logger.info(f"Saving DA results table to {table_path}")

        output_df = result_dict["table"]

        # Determine separator based on file extension
        if str(table_path).endswith('.tsv'):
            output_df.to_csv(table_path, sep='\t')
        elif str(table_path).endswith('.csv'):
            output_df.to_csv(table_path)
        else:
            logger.error(f"Unsupported table format: {table_path.suffix}. Use .csv or .tsv")
            sys.exit(1)

        logger.info(f"Saved {len(output_df.columns)} columns for {len(output_df)} cells")

    logger.info("Differential abundance analysis completed successfully")
