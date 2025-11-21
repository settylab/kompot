"""Utility functions for CLI."""
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging


logger = logging.getLogger("kompot.cli")


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

    with open(config_file, 'r') as f:
        if config_file.suffix in ['.yaml', '.yml']:
            config = yaml.safe_load(f)
        elif config_file.suffix == '.json':
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_file.suffix}. Use .yaml, .yml, or .json")

    return config if config is not None else {}


def merge_args_with_config(args_dict: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
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

    valid_extensions = ['.h5ad', '.zarr']
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
        format='[%(asctime)s] [%(levelname)-8s] %(message)s',
        level=level,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
