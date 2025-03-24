"""Volcano plot functions for visualizing differential expression and abundance results.

This module provides functions for creating volcano plots for visualizing
differential expression and differential abundance results.
"""

from .volcano.de import volcano_de
from .volcano.da import volcano_da
from .volcano.multi_da import multi_volcano_da

__all__ = ["volcano_de", "volcano_da", "multi_volcano_da"]