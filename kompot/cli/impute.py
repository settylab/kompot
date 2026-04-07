"""Deprecated CLI alias: ``kompot impute`` → ``kompot smooth``.

.. deprecated::
    Use ``kompot smooth`` instead. This alias will be removed in a future
    version.
"""

import logging
import warnings

from .smooth import add_smooth_parser, run_smooth  # noqa: F401

logger = logging.getLogger("kompot.cli")


def add_impute_parser(subparsers):
    """Register hidden ``impute`` alias that delegates to ``smooth``."""
    parser = subparsers.add_parser(
        "impute",
        help="(deprecated, use 'smooth') Expression smoothing via GP regression",
    )
    # Copy arguments from the smooth parser by reusing run_smooth
    # but we need to duplicate the arg definitions. Instead, just
    # delegate at runtime.
    parser.set_defaults(func=_run_impute_alias)
    return parser


def _run_impute_alias(args):
    """Emit deprecation warning, then delegate to run_smooth."""
    warnings.warn(
        "'kompot impute' is deprecated and will be removed in a future "
        "version. Use 'kompot smooth' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    logger.warning(
        "The 'impute' command is deprecated. Use 'kompot smooth' instead."
    )
    return run_smooth(args)
