"""CLI command for differential expression analysis."""
import argparse
import sys
from pathlib import Path
from typing import Optional
import logging

import anndata as ad

from ..anndata import compute_differential_expression
from .utils import load_config, merge_args_with_config, validate_anndata_path


logger = logging.getLogger("kompot.cli")


def add_de_parser(subparsers) -> argparse.ArgumentParser:
    """
    Add differential expression subcommand parser.

    Parameters
    ----------
    subparsers
        Subparsers object from argparse

    Returns
    -------
    argparse.ArgumentParser
        The DE parser
    """
    parser = subparsers.add_parser(
        'de',
        help='Differential expression analysis',
        description='Compute differential expression between two conditions'
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
        '--layer',
        type=str,
        help='Layer in adata.layers for expression data (default: None, use X)'
    )

    parser.add_argument(
        '--result-key',
        type=str,
        default='kompot_de',
        help='Key for storing results in adata.uns (default: kompot_de)'
    )

    parser.add_argument(
        '--n-landmarks',
        type=int,
        default=5000,
        help='Number of landmarks for approximation (default: 5000)'
    )

    parser.add_argument(
        '--sample-col',
        type=str,
        help='Column in adata.obs with sample labels for sample variance estimation'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Batch size for memory-efficient processing (default: 100)'
    )

    parser.add_argument(
        '--fdr-threshold',
        type=float,
        default=0.05,
        help='FDR threshold for significance (default: 0.05)'
    )

    parser.add_argument(
        '--null-genes',
        type=int,
        default=2000,
        help='Number of null genes for FDR estimation (default: 2000)'
    )

    parser.add_argument(
        '--null-seed',
        type=int,
        default=42,
        help='Random seed for null gene selection and shuffling (default: 42)'
    )

    # Boolean flags
    parser.add_argument(
        '--no-progress',
        action='store_true',
        help='Disable progress bars'
    )

    parser.add_argument(
        '--store-landmarks',
        action='store_true',
        help='Store landmarks in AnnData for reuse'
    )

    parser.add_argument(
        '--store-additional-stats',
        action='store_true',
        help='Store additional statistics'
    )

    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing results without warning'
    )

    parser.set_defaults(func=run_de)

    return parser


def run_de(args):
    """
    Run differential expression analysis.

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
        if v is not None and k not in ['input', 'output', 'config', 'func', 'verbose', 'command']
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
    if 'fdr_threshold' in args_dict:
        args_dict['fdr_threshold'] = args_dict.pop('fdr_threshold')
    if 'null_genes' in args_dict:
        args_dict['null_genes'] = args_dict.pop('null_genes')
    if 'store_landmarks' in args_dict:
        args_dict['store_landmarks'] = args_dict.pop('store_landmarks')
    if 'store_additional_stats' in args_dict:
        args_dict['store_additional_stats'] = args_dict.pop('store_additional_stats')
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

    logger.info("Starting differential expression analysis")
    logger.info(f"  Groupby: {params['groupby']}")
    logger.info(f"  Condition 1: {params['condition1']}")
    logger.info(f"  Condition 2: {params['condition2']}")
    logger.info(f"  ObsM key: {params.get('obsm_key', 'X_pca')}")
    if params.get('layer'):
        logger.info(f"  Layer: {params['layer']}")

    # Run analysis
    try:
        compute_differential_expression(adata, **params)
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

    logger.info("Differential expression analysis completed successfully")
