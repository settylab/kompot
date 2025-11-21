"""CLI command for computing diffusion maps."""
import argparse
import sys
from pathlib import Path
import logging

import anndata as ad

from .utils import load_config, merge_args_with_config, validate_anndata_path


logger = logging.getLogger("kompot.cli")


def add_dm_parser(subparsers) -> argparse.ArgumentParser:
    """
    Add diffusion maps subcommand parser.

    Parameters
    ----------
    subparsers
        Subparsers object from argparse

    Returns
    -------
    argparse.ArgumentParser
        The DM parser
    """
    parser = subparsers.add_parser(
        'dm',
        help='Compute diffusion maps (preprocessing)',
        description='Compute diffusion maps using Palantir for cell state representation'
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

    # Main parameters
    parser.add_argument(
        '--pca-key',
        type=str,
        default='X_pca',
        help='Key in adata.obsm with PCA coordinates (default: X_pca)'
    )

    parser.add_argument(
        '--n-components',
        type=int,
        default=10,
        help='Number of diffusion components to compute (default: 10)'
    )

    parser.add_argument(
        '--knn',
        type=int,
        default=30,
        help='Number of nearest neighbors for graph (default: 30)'
    )

    parser.add_argument(
        '--alpha',
        type=float,
        default=0,
        help='Alpha parameter for diffusion (default: 0, controls decay)'
    )

    parser.add_argument(
        '--use-adjacency-matrix',
        action='store_true',
        help='Use adjacency matrix if available'
    )

    parser.set_defaults(func=run_dm)

    return parser


def run_dm(args):
    """
    Run diffusion maps computation.

    Parameters
    ----------
    args
        Parsed arguments from argparse
    """
    # Check if palantir is available
    try:
        import palantir
    except ImportError:
        logger.error("Palantir is not installed. Install it with: pip install palantir")
        logger.error("Or install kompot with: pip install kompot[recommended]")
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
        if v is not None and k not in ['input', 'output', 'config', 'func', 'verbose']
    }

    # Rename CLI args to match function parameters
    if 'pca_key' in args_dict:
        args_dict['pca_key'] = args_dict.pop('pca_key')
    if 'n_components' in args_dict:
        args_dict['n_components'] = args_dict.pop('n_components')
    if 'use_adjacency_matrix' in args_dict:
        args_dict['use_adjacency_matrix'] = args_dict.pop('use_adjacency_matrix')

    # Merge with config (CLI args take precedence)
    params = merge_args_with_config(args_dict, config)

    # Check if PCA exists
    pca_key = params.get('pca_key', 'X_pca')
    if pca_key not in adata.obsm:
        logger.error(f"PCA coordinates not found at adata.obsm['{pca_key}']")
        logger.error("Run PCA first using scanpy: sc.pp.pca(adata)")
        sys.exit(1)

    logger.info("Computing diffusion maps")
    logger.info(f"  PCA key: {pca_key}")
    logger.info(f"  n_components: {params.get('n_components', 10)}")
    logger.info(f"  knn: {params.get('knn', 30)}")

    # Run diffusion maps
    try:
        palantir.utils.run_diffusion_maps(adata, **params)
    except Exception as e:
        logger.error(f"Diffusion maps computation failed: {str(e)}")
        raise

    # Check results
    if 'DM_EigenVectors' in adata.obsm:
        logger.info(f"Computed {adata.obsm['DM_EigenVectors'].shape[1]} diffusion components")
        logger.info("Results stored in adata.obsm['DM_EigenVectors']")
    else:
        logger.warning("DM_EigenVectors not found in output. Check for errors.")

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

    logger.info("Diffusion maps computation completed successfully")
