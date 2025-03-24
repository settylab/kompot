"""
Example demonstrating how to use Kompot's multi_volcano_da plotting functionality with KDE/violin plots.

This example shows how to create multiple volcano plots for differential abundance results
with KDE or violin plot backgrounds to visualize the distribution of points.
"""

import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import kompot as kp
import pandas as pd

# Load example data
adata = sc.datasets.pbmc3k_processed()

# Add simulated differential abundance results to adata.obs
np.random.seed(42)
n_cells = adata.n_obs

# Generate simulated log fold changes - make them related to cell type
cell_types = adata.obs['louvain'].astype('category').cat.codes.values
lfc_base = np.zeros(n_cells)
for i in range(max(cell_types) + 1):
    mask = cell_types == i
    # Each cell type gets a different base LFC
    lfc_base[mask] = (i - max(cell_types)/2) / (max(cell_types)/2) * 2  
    
# Add random noise
lfc = lfc_base + np.random.normal(0, 0.5, n_cells)
adata.obs["kompot_da_lfc"] = lfc

# Generate simulated p-values - make them related to LFC magnitude
pvals = np.exp(-np.abs(lfc) * 2) * 0.8 + 0.001
adata.obs["kompot_da_pval"] = pvals
adata.obs["kompot_da_neg_log10_fold_change_pvalue"] = -np.log10(pvals)

# Add a group column to color by
adata.obs["group"] = adata.obs["louvain"]

# Add categorical direction column
adata.obs["kompot_da_log_fold_change_direction"] = "neutral"
adata.obs.loc[adata.obs["kompot_da_lfc"] > 1.0, "kompot_da_log_fold_change_direction"] = "up"
adata.obs.loc[adata.obs["kompot_da_lfc"] < -1.0, "kompot_da_log_fold_change_direction"] = "down"
adata.obs["kompot_da_log_fold_change_direction"] = adata.obs["kompot_da_log_fold_change_direction"].astype("category")

# Add colors to adata.uns with the standard naming convention
from kompot.utils import KOMPOT_COLORS
direction_colors = KOMPOT_COLORS["direction"]
adata.uns["kompot_da_log_fold_change_direction_colors"] = [
    direction_colors[cat] for cat in adata.obs["kompot_da_log_fold_change_direction"].cat.categories
]

# Create multiple DA volcano plots with KDE background
kp.plot.multi_volcano_da(
    adata,
    groupby="louvain",
    lfc_key="kompot_da_lfc",
    pval_key="kompot_da_neg_log10_fold_change_pvalue",
    pval_threshold=0.05,
    lfc_threshold=1.0,
    color="kompot_da_log_fold_change_direction",
    alpha_background=0.5,
    background_plot="kde",
    background_alpha=0.5,
    background_color="#E6E6E6",  # Light gray
    background_edgecolor="#808080",  # Medium gray outline
    background_height_factor=0.6,  # Default height
    title="Multiple Volcano Plots with KDE Background",
    show=True,
    save="multi_volcano_da_kde.png"
)

# Create multiple DA volcano plots with violin plot background
kp.plot.multi_volcano_da(
    adata,
    groupby="louvain",
    lfc_key="kompot_da_lfc",
    pval_key="kompot_da_neg_log10_fold_change_pvalue",
    pval_threshold=0.05,
    lfc_threshold=1.0,
    color="kompot_da_log_fold_change_direction",
    alpha_background=0.5,
    background_plot="violin",
    background_alpha=0.5,
    background_color="#D6EAF8",  # Light blue
    background_edgecolor="#3498DB",  # Darker blue outline
    background_height_factor=0.8,  # Slightly taller than default
    background_kwargs={"showmeans": False, "showmedians": False, "showextrema": False},
    title="Multiple Volcano Plots with Violin Plot Background",
    show=True,
    save="multi_volcano_da_violin.png"
)

# Create multiple DA volcano plots with KDE background and 2D contour
kp.plot.multi_volcano_da(
    adata,
    groupby="louvain",
    lfc_key="kompot_da_lfc",
    pval_key="kompot_da_neg_log10_fold_change_pvalue",
    pval_threshold=0.05,
    lfc_threshold=1.0,
    color="kompot_da_log_fold_change_direction",
    alpha_background=0.5,
    background_plot="kde",
    background_alpha=0.5,
    background_color="#E8DAEF",  # Light purple
    background_edgecolor="#8E44AD",  # Darker purple outline
    background_height_factor=0.7,  # Slightly larger than default
    background_kwargs={
        "bw_method": 0.2, 
        "show_2d_kde": True, 
        "contour_cmap": "Purples",
        "contour_alpha": 0.2,
        "contour_levels": 8
    },
    title="Multiple Volcano Plots with KDE Background and 2D Contour",
    show=True,
    save="multi_volcano_da_kde_contour.png"
)

print("Multiple volcano plots with density backgrounds created successfully!")