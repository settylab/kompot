"""
Test script for the updated heatmap plotting functionality with improved legend placement,
proper z-score labeling, fixed gene-wise z-scoring, and no automatic plot display.
"""

import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import kompot as kp
import pandas as pd

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

# Create controlled non-random grouping and conditions for testing z-scoring
n_cells = adata.n_obs
cell_types = ['cellType1', 'cellType2', 'cellType3']
adata.obs['cell_type'] = 'cellType1'  # Default

# Assign specific expression patterns to test z-scoring
genes_to_test = adata.var_names[:5].tolist()
print(f"Testing z-scoring with genes: {genes_to_test}")

# Create a dataset where:
# - cellType1 has high expression (5-10) for all genes
# - cellType2 has medium expression (0-5) for all genes
# - cellType3 has low expression (-5-0) for all genes
# This way we can clearly see if z-scoring is per gene

# Divide cells into three equal groups
cell_indices = np.array_split(range(n_cells), 3)

# Assign cell types
for i, indices in enumerate(cell_indices):
    adata.obs.iloc[indices, adata.obs.columns.get_loc('cell_type')] = cell_types[i]

# Create artificial expression values for the test genes
for i, gene in enumerate(genes_to_test):
    gene_idx = adata.var_names.get_loc(gene)
    
    # Set expression values based on cell type
    for cell_idx in cell_indices[0]:  # cellType1: high expression
        adata.X[cell_idx, gene_idx] = 5 + i
    
    for cell_idx in cell_indices[1]:  # cellType2: medium expression  
        adata.X[cell_idx, gene_idx] = 0 + i
    
    for cell_idx in cell_indices[2]:  # cellType3: low expression
        adata.X[cell_idx, gene_idx] = -5 + i

# Assign conditions (half and half)
conditions = ['conditionA'] * (n_cells // 2) + ['conditionB'] * (n_cells - n_cells // 2)
adata.obs['condition'] = conditions

# Store the run info in adata.uns to simulate a real kompot run
adata.uns['kompot_de'] = {
    'run_history': [{
        'params': {
            'groupby': 'condition',
            'conditions': ['conditionA', 'conditionB'],
            'condition1': 'conditionA',
            'condition2': 'conditionB'
        },
        'field_names': {
            'mean_lfc_key': 'kompot_de_mean_lfc_groupA_to_groupB',
            'mahalanobis_key': 'kompot_de_mahalanobis'
        }
    }]
}

# Test 1: Regular heatmap with proper per-gene z-score
fig, ax = plt.subplots(figsize=(10, 8))
kp.plot.heatmap(
    adata,
    gene_list=genes_to_test,  # Use our test genes with known expression patterns
    groupby='cell_type',
    standard_scale='var',  # Z-scoring per gene
    cmap='viridis',
    diagonal_split=False,
    title='Regular Heatmap with Per-Gene Z-score',
    ax=ax,
    return_fig=True
)
fig.tight_layout()
plt.savefig("regular_heatmap_zscore_per_gene.png")
plt.close(fig)

# Test 2: Diagonal split heatmap with improved legend placement and non-stretched colorbar
fig, ax = plt.subplots(figsize=(10, 8))
kp.plot.heatmap(
    adata,
    gene_list=genes_to_test,  # Use our test genes with known expression patterns
    groupby='cell_type',
    condition_column='condition',
    diagonal_split=True,
    standard_scale='var',  # Z-scoring per gene
    cmap='RdBu_r',
    title='Diagonal Split Heatmap with Improved Legend',
    ax=ax,
    return_fig=True
)
# Don't use tight_layout() for diagonal split, manually adjust spacing
fig.subplots_adjust(right=0.85)  # Make room for colorbar and legend
plt.savefig("diagonal_heatmap_improved_legend.png", bbox_inches='tight')
plt.close(fig)

# Test 3: Create a dataset with missing values for one condition in a group
# to test the black triangle fix
# First, make a copy of the data
adata_missing = adata.copy()

# Create a mask for cellType3-conditionA cells
condA_mask = (adata_missing.obs['cell_type'] == 'cellType3') & (adata_missing.obs['condition'] == 'conditionA')

# Remove these cells to create a missing condition for cellType3
if condA_mask.sum() > 0:
    adata_missing = adata_missing[~condA_mask].copy()
    print(f"Removed {condA_mask.sum()} cells from cellType3-conditionA to create missing condition test")
else:
    print("No cells found for cellType3-conditionA")

# Test the diagonal split heatmap with missing data (should avoid black triangles)
fig, ax = plt.subplots(figsize=(10, 8))
kp.plot.heatmap(
    adata_missing,
    gene_list=genes_to_test,
    groupby='cell_type',
    condition_column='condition',
    diagonal_split=True,
    standard_scale='var',
    cmap='RdBu_r',
    title='Diagonal Split Heatmap with Missing Data (No Black Triangles)',
    ax=ax,
    return_fig=True
)
# Don't use tight_layout() for diagonal split, manually adjust spacing
fig.subplots_adjust(right=0.85)  # Make room for colorbar and legend
plt.savefig("diagonal_heatmap_no_black_triangles.png", bbox_inches='tight')
plt.close(fig)

print("Heatmap tests completed and saved!")