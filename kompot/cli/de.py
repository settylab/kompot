"""CLI command for differential expression analysis."""

import argparse
import json
import sys
from pathlib import Path
import logging

import anndata as ad

from ..anndata import de
from ..settings import (
    GPSettings,
    FDRSettings,
    FilterSettings,
    StorageSettings,
    OutputSettings,
)
from .utils import load_config, merge_args_with_config, validate_anndata_path
from .compute_config import configure_compute


logger = logging.getLogger("kompot.cli")


def _json_default(obj):
    """JSON serializer for numpy types."""
    import numpy as np

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


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
        "de",
        help="Differential expression analysis",
        description="Compute differential expression between two conditions",
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
        help="Output only the DE results as a table (.csv or .tsv). Contains gene-level statistics from adata.var.",
    )

    # Config file
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="YAML or JSON config file with advanced parameters",
    )

    # Basic required parameters
    parser.add_argument(
        "--groupby", type=str, help="Column in adata.obs containing condition labels"
    )

    parser.add_argument(
        "--condition1", type=str, help="Label for first condition (reference)"
    )

    parser.add_argument(
        "--condition2", type=str, help="Label for second condition (comparison)"
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
        help="Key for storing results in adata.uns (default: kompot_de)",
    )

    parser.add_argument(
        "--n-landmarks",
        type=int,
        help="Number of landmarks for approximation (default: 5000)",
    )

    parser.add_argument(
        "--sample-col",
        type=str,
        help="Column in adata.obs with sample labels for sample variance estimation",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size for memory-efficient processing (default: 0, no batching)",
    )

    parser.add_argument(
        "--fdr-threshold",
        type=float,
        help="FDR threshold for significance (default: 0.05)",
    )

    parser.add_argument(
        "--null-genes",
        type=int,
        help="Number of null genes for FDR estimation (default: 2000)",
    )

    parser.add_argument(
        "--null-seed",
        type=int,
        help="Random seed for null gene selection and shuffling (default: 42)",
    )

    # Boolean flags
    parser.add_argument(
        "--no-progress", action="store_true", help="Disable progress bars"
    )

    parser.add_argument(
        "--store-landmarks",
        action="store_true",
        help="Store landmarks in AnnData for reuse",
    )

    parser.add_argument(
        "--store-additional-stats",
        action="store_true",
        help="Store additional statistics",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing results without warning",
    )

    parser.add_argument(
        "--use-empirical-variance",
        action="store_true",
        help="Estimate per-gene heteroscedastic noise from squared residuals to deflate significance for high-noise genes",
    )

    # Dry run
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Estimate resource requirements instead of running the analysis. "
        "JSON to stdout, human-readable report to stderr. -o/--output is ignored.",
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
    # Validate output arguments (not required for dry-run)
    if not args.dry_run and not args.output and not args.table_output:
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

    # Configure compute resources (must be done AFTER mellon import in _compute_differential_expression)
    # Extract compute config before other processing
    use_gpu = getattr(args, "use_gpu", False)
    n_threads = getattr(args, "threads", None)

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
            "dry_run",
        ]
    }

    # Handle no_progress -> progress conversion
    if "no_progress" in args_dict:
        args_dict["progress"] = not args_dict.pop("no_progress")

    # Merge with config (CLI args take precedence)
    params = merge_args_with_config(args_dict, config)

    # Validate required parameters
    required = ["groupby", "condition1", "condition2"]
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
    if params.get("layer"):
        logger.info(f"  Layer: {params['layer']}")

    # Configure computational backend
    # This must be called AFTER mellon import (which happens in _compute_differential_expression)
    # So we do a "lazy" import here to trigger mellon import, then configure
    logger.info("")
    logger.info("Configuring computational backend...")
    try:
        # Import mellon to trigger its JAX configuration
        # Now configure our settings (will override mellon's CPU-only default if needed)
        configure_compute(use_gpu=use_gpu, n_threads=n_threads)
    except Exception as e:
        logger.warning(f"Could not configure compute backend: {e}")
        logger.warning("Proceeding with default configuration")
    logger.info("")

    # Extract required params
    groupby = params.pop("groupby")
    condition1 = params.pop("condition1")
    condition2 = params.pop("condition2")
    obsm_key = params.pop("obsm_key", "DM_EigenVectors")
    layer = params.pop("layer", None)
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
        "jit_compile",
        "random_state",
    }
    gp_kwargs = {k: params.pop(k) for k in list(params) if k in gp_keys}
    gp = GPSettings(**gp_kwargs) if gp_kwargs else None

    fdr_kwargs = {}
    if "null_genes" in params:
        fdr_kwargs["null_genes"] = params.pop("null_genes")
    if "null_seed" in params:
        fdr_kwargs["null_seed"] = params.pop("null_seed")
    if "fdr_threshold" in params:
        fdr_kwargs["threshold"] = params.pop("fdr_threshold")
    fdr = FDRSettings(**fdr_kwargs) if fdr_kwargs else None

    filter_keys = {
        "cell_filter",
        "groups",
        "min_cells",
        "min_percentage",
        "check_representation",
    }
    filter_kwargs = {k: params.pop(k) for k in list(params) if k in filter_keys}
    filter_settings = FilterSettings(**filter_kwargs) if filter_kwargs else None

    storage_keys = {
        "result_key",
        "overwrite",
        "store_landmarks",
        "store_posterior_covariance",
        "store_additional_stats",
        "store_arrays_on_disk",
        "disk_storage_dir",
        "max_memory_ratio",
    }
    storage_kwargs = {k: params.pop(k) for k in list(params) if k in storage_keys}
    storage = StorageSettings(**storage_kwargs) if storage_kwargs else None

    output_keys = {
        "copy",
        "inplace",
        "return_full_results",
        "compute_mahalanobis",
        "allow_single_condition_variance",
        "progress",
    }
    output_kwargs = {k: params.pop(k) for k in list(params) if k in output_keys}

    # Handle return_full_results for table output
    if args.table_output:
        output_kwargs["return_full_results"] = True
    output = OutputSettings(**output_kwargs) if output_kwargs else None

    # Build shared call kwargs
    call_kwargs = dict(
        groupby=groupby,
        condition1=condition1,
        condition2=condition2,
        obsm_key=obsm_key,
        layer=layer,
        sample_col=sample_col,
        gp=gp,
        fdr=fdr,
        filter=filter_settings,
        storage=storage,
        output=output,
        **params,  # remaining params forwarded as function_kwargs
    )

    # Dry run: estimate resources, output JSON to stdout, report to stderr
    if args.dry_run:
        import io

        try:
            old_stdout = sys.stdout
            sys.stdout = sys.stderr  # capture de()'s print(plan.format_report())
            try:
                plan = de(adata, dry_run=True, **call_kwargs)
            finally:
                sys.stdout = old_stdout
        except Exception as e:
            logger.error(f"Dry run failed: {str(e)}")
            raise

        # Machine-parseable JSON to stdout
        json.dump(plan.to_dict(), sys.stdout, default=_json_default, indent=2)
        print(file=sys.stdout)  # trailing newline
        sys.exit(0 if plan.is_feasible else 1)

    # Run analysis
    try:
        result_dict = de(adata, **call_kwargs)
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
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
        logger.info(f"Saving DE results table to {table_path}")

        output_df = result_dict["table"]

        # Determine separator based on file extension
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

    logger.info("Differential expression analysis completed successfully")
