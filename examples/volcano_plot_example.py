"""
Example demonstrating how to use Kompot's volcano plotting functionality for differential expression.

This example shows how to create a volcano plot from differential expression results
stored in an AnnData object.
"""

import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import kompot as kp
from kompot.utils import KOMPOT_COLORS

# Load your data (for this example, we'll create a simulated dataset)
# In a real scenario, you would load your own data with sc.read() or similar
adata = sc.datasets.pbmc3k_processed()

# Add simulated differential expression results to adata.var
# In a real scenario, these would be the results from running kompot.compute_differential_expression
np.random.seed(42)
n_genes = adata.n_vars

# Generate simulated log fold changes
lfc = np.random.normal(0, 1, n_genes)
adata.var["kompot_de_mean_lfc_groupA_to_groupB"] = lfc

# Generate simulated Mahalanobis distances
# Make some genes strongly significant (correlated with fold change)
mahalanobis = np.abs(lfc) * 2 + np.random.normal(0, 1, n_genes)
adata.var["kompot_de_mahalanobis"] = np.abs(mahalanobis)

# Create a basic volcano plot using default parameters
plt.figure(figsize=(12, 8))
kp.plot.volcano_de(
    adata,
    lfc_key="kompot_de_mean_lfc_groupA_to_groupB",
    score_key="kompot_de_mahalanobis"
)
plt.savefig("basic_volcano.png")

# Create a more customized volcano plot
plt.figure(figsize=(12, 8))
kp.plot.volcano_de(
    adata,
    lfc_key="kompot_de_mean_lfc_groupA_to_groupB",  # Explicitly specify the LFC key
    score_key="kompot_de_mahalanobis",              # Explicitly specify the score key
    condition1="groupB",                            # Explicitly specify condition names
    condition2="groupA",
    n_top_genes=15,                                 # Show more top genes
    color_up="#FF5733",                             # Custom colors
    color_down="#3369FF",
    color_background="#CCCCCC",
    title="Custom Volcano Plot: GroupA to GroupB",
    point_size=8,                                   # Larger point size
    font_size=10,                                   # Larger font size
    grid_kwargs={"alpha": 0.2, "linestyle": "--"},  # Custom grid style
    save="custom_volcano.png"                       # Save to file
)

# Example of creating a figure with multiple volcano plots
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# First volcano plot - custom style
kp.plot.volcano_de(
    adata,
    lfc_key="kompot_de_mean_lfc_groupA_to_groupB",
    score_key="kompot_de_mahalanobis",
    ax=axes[0],
    color_up=KOMPOT_COLORS["direction"]["up"],
    color_down=KOMPOT_COLORS["direction"]["down"],
    title="Volcano Plot: Default Style",
    return_fig=True
)

# Second volcano plot - different style
kp.plot.volcano_de(
    adata,
    lfc_key="kompot_de_mean_lfc_groupA_to_groupB",
    score_key="kompot_de_mahalanobis",
    ax=axes[1],
    color_up="#fc8d59",
    color_down="#91bfdb",
    color_background="#ededed",
    n_top_genes=5,
    title="Volcano Plot: Alternative Style",
    return_fig=True
)

plt.tight_layout()
plt.savefig("multiple_volcano_plots.png")
print("Volcano plot examples created successfully!")

# Example using highlight_genes parameter to specify genes to highlight
plt.figure(figsize=(12, 8))
# Get 5 random gene names for the example
import random
random_genes = random.sample(list(adata.var_names), 5)
print(f"Highlighting specific genes: {', '.join(random_genes)}")

kp.plot.volcano_de(
    adata,
    lfc_key="kompot_de_mean_lfc_groupA_to_groupB",
    score_key="kompot_de_mahalanobis",
    highlight_genes=random_genes,
    title="Volcano Plot with Specific Genes Highlighted",
    color_up="#8856a7",
    color_down="#43a2ca",
    save="specific_genes_volcano.png"
)
