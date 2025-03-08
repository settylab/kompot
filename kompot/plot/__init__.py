"""Plotting functions for Kompot results visualization."""

from .volcano import volcano_de, volcano_da
from .heatmap import heatmap

# For backward compatibility
volcano_plot = volcano_de

__all__ = ["volcano_de", "volcano_da", "volcano_plot", "heatmap"]