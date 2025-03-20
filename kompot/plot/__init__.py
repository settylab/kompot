"""Plotting functions for Kompot results visualization."""

import logging
import sys
logger = logging.getLogger("kompot")

# Import all plotting functions with error handling
__all__ = []

try:
    from .volcano import volcano_de, volcano_da
    # For backward compatibility
    volcano_plot = volcano_de
    __all__.extend(["volcano_de", "volcano_da", "volcano_plot"])
except (ImportError, TypeError) as e:
    # Provide more specific error message for Python 3.12 metaclass issues
    if sys.version_info >= (3, 12) and isinstance(e, TypeError) and "metaclass conflict" in str(e):
        error_msg = (
            "Volcano plot functions unavailable due to metaclass conflict in scanpy with Python 3.12. "
            "You have two options to fix this:\n"
            "1. Update scanpy to the latest version: pip install --upgrade scanpy\n"
            "2. Use Python 3.9-3.11 instead of 3.12"
        )
        logger.warning(f"Python 3.12 compatibility issue: {error_msg}")
    else:
        logger.warning(f"Could not import volcano plotting functions due to: {e}")
    
    # Create stub functions that raise helpful errors
    def volcano_de(*args, **kwargs):
        raise ImportError("Volcano plot functions unavailable due to scanpy compatibility issues. "
                         "Please update scanpy or use Python 3.9-3.11 instead of 3.12.")
    def volcano_da(*args, **kwargs):
        raise ImportError("Volcano plot functions unavailable due to scanpy compatibility issues. "
                         "Please update scanpy or use Python 3.9-3.11 instead of 3.12.")
    volcano_plot = volcano_de

try:
    from .heatmap import heatmap, direction_barplot
    __all__.extend(["heatmap", "direction_barplot"])
except (ImportError, TypeError) as e:
    # Provide more specific error message for Python 3.12 metaclass issues
    if sys.version_info >= (3, 12) and isinstance(e, TypeError) and "metaclass conflict" in str(e):
        error_msg = (
            "Heatmap functions unavailable due to metaclass conflict in scanpy with Python 3.12. "
            "You have two options to fix this:\n"
            "1. Update scanpy to the latest version: pip install --upgrade scanpy\n"
            "2. Use Python 3.9-3.11 instead of 3.12"
        )
        logger.warning(f"Python 3.12 compatibility issue: {error_msg}")
    else:
        logger.warning(f"Could not import heatmap functions due to: {e}")
    
    def heatmap(*args, **kwargs):
        raise ImportError("Heatmap functions unavailable due to scanpy compatibility issues. "
                         "Please update scanpy or use Python 3.9-3.11 instead of 3.12.")
    def direction_barplot(*args, **kwargs):
        raise ImportError("Direction barplot functions unavailable due to scanpy compatibility issues. "
                         "Please update scanpy or use Python 3.9-3.11 instead of 3.12.")

try:
    from .expression import plot_gene_expression
    __all__.append("plot_gene_expression")
except (ImportError, TypeError) as e:
    # Provide more specific error message for Python 3.12 metaclass issues
    if sys.version_info >= (3, 12) and isinstance(e, TypeError) and "metaclass conflict" in str(e):
        error_msg = (
            "Gene expression plotting functions unavailable due to metaclass conflict in scanpy with Python 3.12. "
            "You have two options to fix this:\n"
            "1. Update scanpy to the latest version: pip install --upgrade scanpy\n"
            "2. Use Python 3.9-3.11 instead of 3.12"
        )
        logger.warning(f"Python 3.12 compatibility issue: {error_msg}")
    else:
        logger.warning(f"Could not import gene expression plotting functions due to: {e}")
    
    def plot_gene_expression(*args, **kwargs):
        raise ImportError("Gene expression plotting functions unavailable due to scanpy compatibility issues. "
                         "Please update scanpy or use Python 3.9-3.11 instead of 3.12.")