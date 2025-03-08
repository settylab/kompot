"""
Example demonstrating how to use Kompot's heatmap plotting functionality.

This example shows how to create heatmaps to visualize gene expression patterns
across different conditions or groups.
"""

import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import kompot as kp

# Load your data (for this example, we'll use a built-in dataset)
# In a real scenario, you would load your own data with sc.read() or similar
adata = sc.datasets.pbmc3k_processed()

# Add simulated differential expression results to adata.var
# In a real scenario, these would be the results from running kompot.compute_differential_expression
np.random.seed(42)
n_genes = adata.n_vars

# Generate simulated log fold changes
lfc = np.random.normal(0, 1, n_genes)
adata.var["kompot_de_mean_lfc_groupA_vs_groupB"] = lfc

# Generate simulated Mahalanobis distances
# Make some genes strongly significant (correlated with fold change)
mahalanobis = np.abs(lfc) * 2 + np.random.normal(0, 1, n_genes)
adata.var["kompot_de_mahalanobis"] = np.abs(mahalanobis)

# For this example, we'll create a random grouping of cells
# In a real scenario, this would be your experimental groups or conditions
n_cells = adata.n_obs
groups = np.random.choice(['groupA', 'groupB', 'groupC'], size=n_cells)
adata.obs['group'] = groups

# Create a basic heatmap using default parameters
plt.figure(figsize=(12, 10))
kp.plot.heatmap(
    adata,
    groupby='group',
    standard_scale='var',  # Scale by gene (row)
    cmap='viridis',
    title='Top Differential Genes by Group'
)
plt.savefig("basic_heatmap.png")

# Create a more customized heatmap
plt.figure(figsize=(12, 10))
kp.plot.heatmap(
    adata,
    n_top_genes=15,                     # Show top 15 genes
    groupby='group',
    lfc_key="kompot_de_mean_lfc_groupA_vs_groupB",  # Explicitly specify the LFC key
    score_key="kompot_de_mahalanobis",              # Explicitly specify the score key
    standard_scale='var',               # Scale by gene (row)
    cmap='RdBu_r',                      # Red-blue colormap
    gene_labels_size=12,                # Larger gene label size
    group_labels_size=14,               # Larger group label size
    colorbar_title="Z-score",           # Custom colorbar title
    title="Gene Expression Heatmap: Top 15 Genes by Mahalanobis Distance",
    center=0,                           # Center colormap at 0
    save="custom_heatmap.png"           # Save to file
)

# Create a heatmap with clustering and dendrograms
kp.plot.heatmap(
    adata,
    n_top_genes=20,
    groupby='group',
    standard_scale='var',
    cmap='YlGnBu',
    dendrogram=True,                    # Show dendrograms
    title="Clustered Gene Expression Heatmap",
    save="clustered_heatmap.png"
)

# Example of creating multiple heatmaps in one figure
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# First heatmap - standard scaling by gene (row)
kp.plot.heatmap(
    adata,
    n_top_genes=15,
    groupby='group',
    standard_scale='var',
    cmap='YlOrRd',
    title="Scale by Gene",
    ax=axes[0],
    return_fig=True
)

# Second heatmap - standard scaling by group (column)
kp.plot.heatmap(
    adata,
    n_top_genes=15,
    groupby='group',
    standard_scale='group',
    cmap='PuBuGn',
    title="Scale by Group",
    ax=axes[1],
    return_fig=True
)

plt.tight_layout()
plt.savefig("multiple_heatmaps.png")
print("Heatmap examples created successfully!")