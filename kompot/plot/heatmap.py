"""Heatmap and related plotting functions.

This module provides functions for creating heatmap visualizations, 
including split-cell heatmaps that show differences between conditions.
"""

from .heatmap.core import heatmap
from .heatmap.direction_plot import direction_barplot

__all__ = ["heatmap", "direction_barplot"]