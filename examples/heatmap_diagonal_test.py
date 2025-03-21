"""
Test script for the improved heatmap functionality with diagonal splits.
"""

import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import kompot as kp

# Load your data (for this example, we'll use a built-in dataset)
adata = sc.datasets.pbmc3k_processed()

# Add simulated differential expression results to adata.var
np.random.seed(42)
n_genes = adata.n_vars

# Generate simulated log fold changes
lfc = np.random.normal(0, 1, n_genes)
adata.var["kompot_de_mean_lfc_conditionA_to_conditionB"] = lfc

# Generate simulated Mahalanobis distances
# Make some genes strongly significant (correlated with fold change)
mahalanobis = np.abs(lfc) * 2 + np.random.normal(0, 1, n_genes)
adata.var["kompot_de_mahalanobis"] = np.abs(mahalanobis)

# Create a condition column with exactly two conditions for diagonal split
n_cells = adata.n_obs
conditions = np.random.choice(['conditionA', 'conditionB'], size=n_cells)
adata.obs['condition'] = conditions

# Create a group column for cell types (we'll use louvain clusters as a substitute)
adata.obs['cell_type'] = adata.obs['louvain']

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
            'mean_lfc_key': 'kompot_de_mean_lfc_conditionA_to_conditionB',
            'mahalanobis_key': 'kompot_de_mahalanobis'
        }
    }]
}

# Test 1: Regular heatmap (non-diagonal)
print("Creating regular heatmap...")
kp.plot.heatmap(
    adata,
    n_top_genes=15,
    groupby='cell_type',
    lfc_key="kompot_de_mean_lfc_conditionA_to_conditionB",
    score_key="kompot_de_mahalanobis",
    standard_scale='var',
    cmap='viridis',
    diagonal_split=False,
    title="Regular Heatmap (Non-Diagonal)",
    save="test_regular_heatmap.png"
)

# Test 2: Diagonal split heatmap
print("Creating diagonal split heatmap...")
kp.plot.heatmap(
    adata,
    n_top_genes=15,
    groupby='cell_type',
    condition_column='condition',
    condition1_name='conditionA',
    condition2_name='conditionB',
    lfc_key="kompot_de_mean_lfc_conditionA_to_conditionB",
    score_key="kompot_de_mahalanobis",
    cmap='viridis',
    diagonal_split=True,  # This is now the default, but being explicit
    title="Diagonal Split Heatmap",
    save="test_diagonal_heatmap.png"
)

print("Heatmap tests completed. Check the output images.")