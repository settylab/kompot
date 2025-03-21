"""
Test script for the improved heatmap plotting with:
1. Square tiles in both heatmap types
2. Colorbar below the legend
3. Gene-wise z-scoring as the default
"""

import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import kompot as kp

# Set matplotlib to non-interactive mode
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
plt.ioff()  # Turn off interactive mode

# Load test data
adata = sc.datasets.pbmc3k_processed()

# Add simulated differential expression results
np.random.seed(42)
n_genes = adata.n_vars

# Generate simulated log fold changes
lfc = np.random.normal(0, 1, n_genes)
adata.var["kompot_de_mean_lfc_groupA_to_groupB"] = lfc

# Generate simulated Mahalanobis distances
mahalanobis = np.abs(lfc) * 2 + np.random.normal(0, 1, n_genes)
adata.var["kompot_de_mahalanobis"] = np.abs(mahalanobis)

# Create random grouping and conditions
n_cells = adata.n_obs
groups = np.random.choice(['cellType1', 'cellType2', 'cellType3'], size=n_cells)
adata.obs['cell_type'] = groups

conditions = np.random.choice(['conditionA', 'conditionB'], size=n_cells)
adata.obs['condition'] = conditions

# Test 1: Regular heatmap with default gene-wise z-scoring (square tiles, colorbar at right)
fig, ax = plt.subplots(figsize=(12, 10))
kp.plot.heatmap(
    adata,
    groupby='cell_type',
    cmap='viridis',
    diagonal_split=False,
    title='Regular Heatmap - Square Tiles, Gene-wise Z-scoring',
    ax=ax
)
plt.savefig("improved_regular_heatmap.png")
plt.close(fig)

# Test 2: Diagonal split heatmap (square tiles, colorbar below legend)
fig, ax = plt.subplots(figsize=(12, 10))
kp.plot.heatmap(
    adata,
    groupby='cell_type',
    condition_column='condition',
    diagonal_split=True,
    cmap='viridis',
    title='Diagonal Split Heatmap - Square Tiles, Gene-wise Z-scoring',
    ax=ax
)
plt.savefig("improved_diagonal_heatmap.png")
plt.close(fig)

# Test 3: Regular heatmap with explicit standard scaling
fig, ax = plt.subplots(figsize=(12, 10))
kp.plot.heatmap(
    adata,
    groupby='cell_type',
    standard_scale='var',  # Explicitly set gene-wise z-scoring
    cmap='RdBu_r',
    diagonal_split=False,
    title='Regular Heatmap with Explicit Gene-wise Z-scoring',
    ax=ax
)
plt.savefig("improved_regular_heatmap_z_score.png")
plt.close(fig)

# Test 4: Try different colormap to highlight changes
fig, ax = plt.subplots(figsize=(12, 10))
kp.plot.heatmap(
    adata,
    groupby='cell_type',
    condition_column='condition',
    diagonal_split=True,
    standard_scale='var',
    cmap='coolwarm',
    title='Diagonal Split Heatmap with Coolwarm Colormap',
    ax=ax
)
plt.savefig("improved_diagonal_coolwarm.png")
plt.close(fig)

print("Heatmap improvement tests completed and saved!")