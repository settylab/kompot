"""CLI command for expression imputation."""

import argparse
import sys
from pathlib import Path
import logging

import anndata as ad

from ..anndata import impute_expression
from ..settings import GPSettings, StorageSettings, OutputSettings
from .utils import load_config, merge_args_with_config, validate_anndata_path
from .compute_config import configure_compute


logger = logging.getLogger("kompot.cli")


def add_impute_parser(subparsers) -> argparse.ArgumentParser:
    """Add expression imputation subcommand parser.

    Parameters
    ----------
    subparsers
        Subparsers object from argparse

    Returns
    -------
    argparse.ArgumentParser
        The impute parser
    """
    parser = subparsers.add_parser(
        "impute",
        help="Expression imputation via GP smoothing",
        description="Impute gene expression for a single condition using GP smoothing",
    )

    # Required arguments
    parser.add_argument("input", type=str, help="Input AnnData file (.h5ad or .zarr)")

    # Output
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output AnnData file (.h5ad or .zarr). Required unless --table-output is specified.",
    )

    parser.add_argument(
        "-t",
        "--table-output",
        type=str,
        help="Output gene-level summary table (.csv or .tsv) with mean imputed values and std.",
    )

    # Config file
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="YAML or JSON config file with advanced parameters",
    )

    # Condition selection
    parser.add_argument(
        "--groupby",
        type=str,
        help="Column in adata.obs containing condition labels",
    )

    parser.add_argument(
        "--condition",
        type=str,
        help="Which condition to impute (requires --groupby). If omitted, all cells are used.",
    )

    # Common optional parameters
    parser.add_argument(
        "--obsm-key",
        type=str,
        help="Key in adata.obsm for cell states (default: DM_EigenVectors)",
    )

    parser.add_argument(
        "--layer",
        type=str,
        help="Layer in adata.layers for expression data (default: None, use X)",
    )

    parser.add_argument(
        "--result-key",
        type=str,
        help="Key for storing results in adata.uns (default: kompot_impute)",
    )

    parser.add_argument(
        "--n-landmarks",
        type=int,
        help="Number of landmarks for Nystrom approximation (default: 5000)",
    )

    parser.add_argument(
        "--sample-col",
        type=str,
        help="Column in adata.obs with sample labels for sample variance estimation",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        help="Cells per batch during prediction (default: 500)",
    )

    # GP parameters
    parser.add_argument(
        "--sigma",
        type=float,
        help="Noise level for the GP (default: 1.0)",
    )

    parser.add_argument(
        "--ls-factor",
        type=float,
        help="Length scale multiplication factor (default: 10.0)",
    )

    parser.add_argument(
        "--eps",
        type=float,
        help="Numerical stability constant (default: 1e-8)",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        help="Random seed for landmark selection (default: None)",
    )

    # Gene selection
    parser.add_argument(
        "--genes",
        type=str,
        nargs="+",
        help="Subset of genes to impute (default: all)",
    )

    # Boolean flags
    parser.add_argument(
        "--use-empirical-variance",
        action="store_true",
        help="Estimate per-gene heteroscedastic noise from GP residuals",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing results without warning",
    )

    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars",
    )

    # Compute configuration
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU for computation (requires CUDA-enabled JAX)",
    )

    parser.add_argument(
        "--threads",
        type=int,
        help="Number of threads to use for JAX, NumPy, and Dask (default: all available cores)",
    )

    parser.set_defaults(func=run_impute)

    return parser


def run_impute(args):
    """Run expression imputation.

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
    adata = (
        ad.read_h5ad(input_path)
        if str(input_path).endswith(".h5ad")
        else ad.read_zarr(input_path)
    )
    logger.info(f"Loaded AnnData: {adata.n_obs} cells × {adata.n_vars} genes")

    # Load config if provided
    config = {}
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)

    # Configure compute resources
    use_gpu = getattr(args, "use_gpu", False)
    n_threads = getattr(args, "threads", None)

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
        k: v
        for k, v in vars(args).items()
        if v is not None
        and k
        not in [
            "input",
            "output",
            "table_output",
            "config",
            "func",
            "verbose",
            "command",
            "use_gpu",
            "threads",
        ]
    }

    # Handle no_progress -> progress conversion
    if "no_progress" in args_dict:
        args_dict["progress"] = not args_dict.pop("no_progress")

    # Merge with config (CLI args take precedence)
    params = merge_args_with_config(args_dict, config)

    logger.info("Starting expression imputation")
    if params.get("groupby") and params.get("condition"):
        logger.info(f"  Groupby: {params['groupby']}")
        logger.info(f"  Condition: {params['condition']}")
    else:
        logger.info("  Using all cells")
    logger.info(f"  ObsM key: {params.get('obsm_key', 'DM_EigenVectors')}")
    if params.get("layer"):
        logger.info(f"  Layer: {params['layer']}")

    # Configure computational backend
    logger.info("")
    logger.info("Configuring computational backend...")
    try:
        configure_compute(use_gpu=use_gpu, n_threads=n_threads)
    except Exception as e:
        logger.warning(f"Could not configure compute backend: {e}")
        logger.warning("Proceeding with default configuration")
    logger.info("")

    # Extract top-level params
    groupby = params.pop("groupby", None)
    condition = params.pop("condition", None)
    obsm_key = params.pop("obsm_key", "DM_EigenVectors")
    layer = params.pop("layer", None)
    genes = params.pop("genes", None)
    sample_col = params.pop("sample_col", None)

    # Build Settings from remaining params
    gp_keys = {
        "sigma",
        "ls",
        "ls_factor",
        "n_landmarks",
        "use_empirical_variance",
        "batch_size",
        "eps",
        "random_state",
    }
    gp_kwargs = {k: params.pop(k) for k in list(params) if k in gp_keys}
    gp = GPSettings(**gp_kwargs) if gp_kwargs else None

    storage_keys = {"result_key", "overwrite"}
    storage_kwargs = {k: params.pop(k) for k in list(params) if k in storage_keys}
    storage = StorageSettings(**storage_kwargs) if storage_kwargs else None

    output_keys = {"progress", "return_full_results"}
    output_kwargs = {k: params.pop(k) for k in list(params) if k in output_keys}

    # Handle return_full_results for table output
    if args.table_output:
        output_kwargs["return_full_results"] = True
    output = OutputSettings(**output_kwargs) if output_kwargs else None

    # Run analysis
    try:
        result = impute_expression(
            adata,
            groupby=groupby,
            condition=condition,
            obsm_key=obsm_key,
            layer=layer,
            genes=genes,
            sample_col=sample_col,
            gp=gp,
            storage=storage,
            output=output,
            **params,  # remaining params forwarded as function_kwargs
        )
    except Exception as e:
        logger.error(f"Imputation failed: {str(e)}")
        raise

    # Save AnnData output if specified
    if args.output:
        output_path = Path(args.output)
        logger.info(f"Saving results to {output_path}")

        if str(output_path).endswith(".h5ad"):
            adata.write_h5ad(output_path)
        elif str(output_path).endswith(".zarr"):
            adata.write_zarr(output_path)
        else:
            logger.error(
                f"Unsupported output format: {output_path.suffix}. Use .h5ad or .zarr"
            )
            sys.exit(1)

    # Save table output if specified
    if args.table_output:
        table_path = Path(args.table_output)
        logger.info(f"Saving imputation summary to {table_path}")

        output_df = result["table"]

        if str(table_path).endswith(".tsv"):
            output_df.to_csv(table_path, sep="\t")
        elif str(table_path).endswith(".csv"):
            output_df.to_csv(table_path)
        else:
            logger.error(
                f"Unsupported table format: {table_path.suffix}. Use .csv or .tsv"
            )
            sys.exit(1)

        logger.info(
            f"Saved {len(output_df.columns)} columns for {len(output_df)} genes"
        )

    logger.info("Expression imputation completed successfully")
