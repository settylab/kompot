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
        help='Output AnnData file (.h5ad or .zarr). Required unless --table-output is specified.'
    )

    parser.add_argument(
        '-t', '--table-output',
        type=str,
        help='Output only the DE results as a table (.csv or .tsv). Contains gene-level statistics from adata.var.'
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
        '--layer',
        type=str,
        help='Layer in adata.layers for expression data (default: None, use X)'
    )

    parser.add_argument(
        '--result-key',
        type=str,
        help='Key for storing results in adata.uns (default: kompot_de)'
    )

    parser.add_argument(
        '--n-landmarks',
        type=int,
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
        help='Batch size for memory-efficient processing (default: 0, no batching)'
    )

    parser.add_argument(
        '--fdr-threshold',
        type=float,
        help='FDR threshold for significance (default: 0.05)'
    )

    parser.add_argument(
        '--null-genes',
        type=int,
        help='Number of null genes for FDR estimation (default: 2000)'
    )

    parser.add_argument(
        '--null-seed',
        type=int,
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

    # Convert args to dict, removing None values and CLI-specific args
    args_dict = {
        k: v for k, v in vars(args).items()
        if v is not None and k not in ['input', 'output', 'table_output', 'config', 'func', 'verbose', 'command']
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

    # Run analysis - use return_full_results if table output is requested
    try:
        if args.table_output:
            result_dict = compute_differential_expression(adata, return_full_results=True, **params)
        else:
            compute_differential_expression(adata, **params)
            result_dict = None
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
        logger.info(f"Saving DE results table to {table_path}")

        output_df = result_dict["table"]

        # Determine separator based on file extension
        if str(table_path).endswith('.tsv'):
            output_df.to_csv(table_path, sep='\t')
        elif str(table_path).endswith('.csv'):
            output_df.to_csv(table_path)
        else:
            logger.error(f"Unsupported table format: {table_path.suffix}. Use .csv or .tsv")
            sys.exit(1)

        logger.info(f"Saved {len(output_df.columns)} columns for {len(output_df)} genes")

    logger.info("Differential expression analysis completed successfully")
