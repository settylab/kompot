"""Deprecated aliases for expression smoothing plots (formerly imputation).

.. deprecated::
    Use :mod:`kompot.plot.smoothing` instead. This module will be removed
    in a future version.
"""

import warnings

from .smoothing import plot_smoothing  # noqa: F401


def plot_imputation(*args, **kwargs):
    """Deprecated. Use :func:`kompot.plot.plot_smoothing` instead."""
    warnings.warn(
        "plot_imputation() is deprecated and will be removed in a future "
        "version. Use kompot.plot.plot_smoothing() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return plot_smoothing(*args, **kwargs)
