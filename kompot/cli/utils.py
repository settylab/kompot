"""Utility functions for CLI."""

import yaml
import json
from pathlib import Path
from typing import Dict, Any
import contextlib
import logging
import sys


logger = logging.getLogger("kompot.cli")


def _nullable_strings_allowed():
    """Opt in to writing pandas nullable-string arrays, where anndata gates it.

    anndata 0.11 and 0.12 refuse to write a ``pd.arrays.StringArray`` (which
    pandas 3 produces for ordinary string columns) unless
    ``anndata.settings.allow_write_nullable_strings`` is set, so every CLI
    write failed under their defaults (settylab/kompot#23). The opt-in is
    scoped to the write rather than set at import, and is a no-op on anndata
    versions without the setting.
    """
    import anndata

    settings = getattr(anndata, "settings", None)
    if settings is None or not hasattr(settings, "allow_write_nullable_strings"):
        return contextlib.nullcontext()
    return settings.override(allow_write_nullable_strings=True)


def write_output(adata, output_path) -> None:
    """Write *adata* to an ``.h5ad`` or ``.zarr`` path, exiting on any other suffix."""
    output_path = Path(output_path)
    if str(output_path).endswith(".h5ad"):
        with _nullable_strings_allowed():
            adata.write_h5ad(output_path)
    elif str(output_path).endswith(".zarr"):
        with _nullable_strings_allowed():
            adata.write_zarr(output_path)
    else:
        logger.error(
            f"Unsupported output format: {output_path.suffix}. Use .h5ad or .zarr"
        )
        sys.exit(1)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.

    Parameters
    ----------
    config_path : str
        Path to config file (.yaml, .yml, or .json)

    Returns
    -------
    Dict[str, Any]
        Configuration dictionary
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, "r") as f:
        if config_file.suffix in [".yaml", ".yml"]:
            config = yaml.safe_load(f)
        elif config_file.suffix == ".json":
            config = json.load(f)
        else:
            raise ValueError(
                f"Unsupported config format: {config_file.suffix}. Use .yaml, .yml, or .json"
            )

    return config if config is not None else {}


def merge_args_with_config(
    args_dict: Dict[str, Any], config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Merge command-line arguments with config file, preferring CLI args.

    Parameters
    ----------
    args_dict : Dict[str, Any]
        Dictionary from argparse namespace
    config : Dict[str, Any]
        Configuration from file

    Returns
    -------
    Dict[str, Any]
        Merged configuration with CLI args taking precedence
    """
    # Start with config
    merged = config.copy()

    # Override with CLI args that are not None
    for key, value in args_dict.items():
        if value is not None:
            merged[key] = value

    return merged


def validate_anndata_path(path: str) -> Path:
    """
    Validate that AnnData file exists and has correct extension.

    Parameters
    ----------
    path : str
        Path to AnnData file

    Returns
    -------
    Path
        Validated Path object
    """
    anndata_path = Path(path)

    if not anndata_path.exists():
        raise FileNotFoundError(f"AnnData file not found: {path}")

    valid_extensions = [".h5ad", ".zarr"]
    if not any(str(anndata_path).endswith(ext) for ext in valid_extensions):
        logger.warning(
            f"File {path} does not have a standard AnnData extension "
            f"({', '.join(valid_extensions)}). Attempting to load anyway."
        )

    return anndata_path


def setup_logging(verbose: bool = False):
    """
    Setup logging configuration.

    Parameters
    ----------
    verbose : bool
        If True, set log level to DEBUG, otherwise INFO
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)-8s] %(message)s",
        level=level,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
