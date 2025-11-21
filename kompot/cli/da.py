"""CLI command for differential abundance analysis."""
import argparse
import sys
from pathlib import Path
import logging

import anndata as ad

from ..anndata import compute_differential_abundance
from .utils import load_config, merge_args_with_config, validate_anndata_path


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
        required=True,
        help='Output AnnData file (.h5ad or .zarr)'
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
        default='X_pca',
        help='Key in adata.obsm for cell states (default: X_pca)'
    )

    parser.add_argument(
        '--result-key',
        type=str,
        default='kompot_da',
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
        help='Batch size for memory-efficient processing (default: None)'
    )

    parser.add_argument(
        '--log-fold-change-threshold',
        type=float,
        default=1.0,
        help='Threshold for log fold change significance (default: 1.0)'
    )

    parser.add_argument(
        '--ptp-threshold',
        type=float,
        default=0.05,
        help='Posterior tail probability threshold (default: 0.05)'
    )

    parser.add_argument(
        '--ls-factor',
        type=float,
        default=10.0,
        help='Length scale multiplication factor (default: 10.0)'
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

    # Convert args to dict, removing None values and CLI-specific args
    args_dict = {
        k: v for k, v in vars(args).items()
        if v is not None and k not in ['input', 'output', 'config', 'func', 'verbose']
    }

    # Rename CLI args to match function parameters
    if 'obsm_key' in args_dict:
        args_dict['obsm_key'] = args_dict.pop('obsm_key')
    if 'result_key' in args_dict:
        args_dict['result_key'] = args_dict.pop('result_key')
    if 'n_landmarks' in args_dict:
        args_dict['n_landmarks'] = args_dict.pop('n_landmarks')
    if 'sample_col' in args_dict:
        args_dict['sample_col'] = args_dict.pop('sample_col')
    if 'batch_size' in args_dict:
        args_dict['batch_size'] = args_dict.pop('batch_size')
    if 'log_fold_change_threshold' in args_dict:
        args_dict['log_fold_change_threshold'] = args_dict.pop('log_fold_change_threshold')
    if 'ptp_threshold' in args_dict:
        args_dict['ptp_threshold'] = args_dict.pop('ptp_threshold')
    if 'ls_factor' in args_dict:
        args_dict['ls_factor'] = args_dict.pop('ls_factor')
    if 'store_landmarks' in args_dict:
        args_dict['store_landmarks'] = args_dict.pop('store_landmarks')

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

    # Run analysis
    try:
        compute_differential_abundance(adata, **params)
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise

    # Save output
    output_path = Path(args.output)
    logger.info(f"Saving results to {output_path}")

    if str(output_path).endswith('.h5ad'):
        adata.write_h5ad(output_path)
    elif str(output_path).endswith('.zarr'):
        adata.write_zarr(output_path)
    else:
        logger.error(f"Unsupported output format: {output_path.suffix}. Use .h5ad or .zarr")
        sys.exit(1)

    logger.info("Differential abundance analysis completed successfully")
