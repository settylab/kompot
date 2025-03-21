"""
Example demonstrating how to visualize expression patterns for individual genes using Kompot.

This example shows how to create multi-panel expression visualizations for specific genes,
including original expression, imputed expression, and fold changes between conditions.
"""

import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import kompot as kp
import pandas as pd

# Load or create sample data
adata = sc.datasets.pbmc3k_processed()

# Add simulated condition information
np.random.seed(42)
conditions = np.random.choice(['Old', 'Young'], size=adata.n_obs)
adata.obs['condition'] = pd.Categorical(conditions)

# Add simulated differential expression results to adata.var
n_genes = adata.n_vars

# Generate simulated log fold changes
lfc = np.random.normal(0, 1, n_genes)
adata.var["kompot_de_mean_lfc_Young_to_Old"] = lfc

# Generate simulated Mahalanobis distances
# Make some genes strongly significant (correlated with fold change)
mahalanobis = np.abs(lfc) * 2 + np.random.normal(0, 1, n_genes)
adata.var["kompot_de_mahalanobis"] = np.abs(mahalanobis)

# Create simulated imputed expression and fold change layers
# Each cell has its own imputed value for each condition
old_imputed = adata.X.copy() + np.random.normal(0, 0.2, adata.X.shape)
young_imputed = adata.X.copy() + lfc[:, np.newaxis].T + np.random.normal(0, 0.2, adata.X.shape)
fold_change = young_imputed - old_imputed

adata.layers["kompot_de_Old_imputed"] = old_imputed
adata.layers["kompot_de_Young_imputed"] = young_imputed
adata.layers["kompot_de_fold_change"] = fold_change

# Generate UMAP if not already present
if 'X_umap' not in adata.obsm:
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)

# Find a highly differentially expressed gene
top_de_genes = adata.var.sort_values('kompot_de_mahalanobis', ascending=False)
top_gene = top_de_genes.index[0]

print(f"Visualizing expression patterns for top DE gene: {top_gene}")
print(f"Mahalanobis distance: {top_de_genes.iloc[0]['kompot_de_mahalanobis']:.2f}")
print(f"Log fold change: {top_de_genes.iloc[0]['kompot_de_mean_lfc_Young_to_Old']:.2f}")

# Use the new function for gene expression visualization
kp.plot.plot_gene_expression(
    adata,
    gene=top_gene,
    lfc_key="kompot_de_mean_lfc_Young_to_Old",
    score_key="kompot_de_mahalanobis",
    condition1="Old",
    condition2="Young",
    basis="X_umap",
    save="gene_expression_visualization.png"
)

# Create a second example with a different gene and custom colors
second_gene = top_de_genes.index[5]  # Pick a different gene
kp.plot.plot_gene_expression(
    adata,
    gene=second_gene,
    condition1="Old",
    condition2="Young",
    cmap_expression="plasma",
    cmap_fold_change="coolwarm",
    save="gene_expression_custom_colors.png"
)

print("Gene expression visualizations created successfully!")