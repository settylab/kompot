"""
Example demonstrating how to use Kompot's volcano_da plotting functionality.

This example shows how to create a volcano plot for differential abundance results,
visualizing cells with log fold change vs significance values and coloring by groups.
"""

import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import kompot as kp
import pandas as pd

# Load example data (for this example, we'll use a built-in dataset)
# In a real scenario, you would load your own data with sc.read() or similar
adata = sc.datasets.pbmc3k_processed()

# Add simulated differential abundance results to adata.obs
# In a real scenario, these would be results from running kompot.compute_differential_abundance
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
# Higher absolute LFC -> lower p-value
pvals = np.exp(-np.abs(lfc) * 2) * 0.8 + 0.001
adata.obs["kompot_da_pval"] = pvals

# Add negative log10-transformed p-values (higher values = more significant)
# The - sign is important here to convert to negative log10
adata.obs["kompot_da_neg_log10_fold_change_pvalue"] = -np.log10(pvals)

# Add a group column to color by (we'll use the louvain clusters)
adata.obs["group"] = adata.obs["louvain"]

# Create a basic differential abundance volcano plot
plt.figure(figsize=(12, 10))
kp.plot.volcano_da(
    adata,
    lfc_key="kompot_da_lfc",
    pval_key="kompot_da_pval",
    lfc_threshold=1.0,
    pval_threshold=0.05,
    title="Differential Abundance Volcano Plot"
)
plt.savefig("basic_da_volcano.png")

# Create a volcano plot with coloring by group
plt.figure(figsize=(12, 10))
kp.plot.volcano_da(
    adata,
    lfc_key="kompot_da_lfc",
    pval_key="kompot_da_pval",
    lfc_threshold=1.0,
    pval_threshold=0.05,
    color="louvain",  # Color by louvain cluster
    point_size=15,
    title="DA Volcano Plot - Colored by Cluster",
    save="colored_da_volcano.png"
)

# Create a volcano plot with custom styling
plt.figure(figsize=(12, 10))
kp.plot.volcano_da(
    adata,
    lfc_key="kompot_da_lfc",
    pval_key="kompot_da_pval",
    lfc_threshold=1.5,  # More stringent LFC threshold
    pval_threshold=0.01,  # More stringent p-value threshold
    alpha_background=0.2,  # More transparent background points
    point_size=20,  # Larger points
    highlight_color="#ff7f00",  # Custom highlight color if not using group coloring
    grid_kwargs={"alpha": 0.2, "linestyle": "--"},  # Custom grid styling
    title="Custom DA Volcano Plot (A to B)",
    save="custom_da_volcano.png"
)

# Create a volcano plot with a highlight subset (e.g., specific cell types)
# Let's highlight cells in clusters 0 and 1
highlight_mask = adata.obs["louvain"].isin(["0", "1"]).values

plt.figure(figsize=(12, 10))
kp.plot.volcano_da(
    adata,
    lfc_key="kompot_da_lfc",
    pval_key="kompot_da_pval",
    highlight_subset=highlight_mask,
    point_size=15,
    title="DA Volcano Plot - Highlighting Specific Clusters",
    save="highlight_da_volcano.png"
)

# Example with multiple groups in one figure
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# First volcano - using regular p-values (auto-transformed)
kp.plot.volcano_da(
    adata,
    lfc_key="kompot_da_lfc",
    pval_key="kompot_da_pval",
    pval_threshold=0.05,
    lfc_threshold=1.0,
    title="Using Regular p-values",
    ax=axes[0],
    return_fig=True
)

# Second volcano - using pre-transformed negative log10 p-values with new naming
kp.plot.volcano_da(
    adata,
    lfc_key="kompot_da_lfc",
    pval_key="kompot_da_neg_log10_fold_change_pvalue",  # Using standardized negative log10 name
    pval_threshold=0.05,  # Will be interpreted correctly based on the key name
    lfc_threshold=1.0,
    log_transform_pval=False,  # Don't transform again since already transformed
    title="Using negative log10 p-values",
    ax=axes[1],
    return_fig=True
)

plt.tight_layout()
plt.savefig("multiple_da_volcanos.png")

# Example showing automatic detection of negative log10 p-values with standardized naming
plt.figure(figsize=(12, 10))
kp.plot.volcano_da(
    adata,
    lfc_key="kompot_da_lfc",
    pval_key="kompot_da_neg_log10_fold_change_pvalue",  # Using standardized negative log10 name
    pval_threshold=0.01,  # Function will handle this correctly
    lfc_threshold=1.2,
    title="Standard negative log10 p-value naming",
    save="neg_log10_pval_volcano.png"
)

# Demonstrate using the colors that would be stored in adata.uns
# Import from the central color definition
from kompot.utils import KOMPOT_COLORS
direction_colors = KOMPOT_COLORS["direction"]

# Add a categorical direction column to demonstrate scanpy-like coloring
# This simulates what compute_differential_abundance would do
adata.obs["kompot_da_log_fold_change_direction"] = "neutral"
adata.obs.loc[adata.obs["kompot_da_lfc"] > 1.0, "kompot_da_log_fold_change_direction"] = "up"
adata.obs.loc[adata.obs["kompot_da_lfc"] < -1.0, "kompot_da_log_fold_change_direction"] = "down"
adata.obs["kompot_da_log_fold_change_direction"] = adata.obs["kompot_da_log_fold_change_direction"].astype("category")

# Add colors to adata.uns with the standard _colors postfix naming convention
adata.uns["kompot_da_log_fold_change_direction_colors"] = [
    direction_colors[cat] for cat in adata.obs["kompot_da_log_fold_change_direction"].cat.categories
]

# Create a volcano plot colored by the direction categories
plt.figure(figsize=(12, 10))
kp.plot.volcano_da(
    adata,
    lfc_key="kompot_da_lfc",
    pval_key="kompot_da_neg_log10_fold_change_pvalue",
    pval_threshold=0.05,
    lfc_threshold=1.0,
    color="kompot_da_log_fold_change_direction",  # Color by direction category
    title="Volcano Plot with Direction Coloring",
    save="direction_colored_volcano.png"
)

print("Differential abundance volcano plot examples created successfully!")