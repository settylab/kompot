"""Plotting functions for Kompot results visualization."""

from .volcano import volcano_de, volcano_da
from .heatmap import heatmap, direction_barplot
from .expression import plot_gene_expression

# For backward compatibility
volcano_plot = volcano_de

__all__ = ["volcano_de", "volcano_da", "volcano_plot", "heatmap", "direction_barplot", "plot_gene_expression"]