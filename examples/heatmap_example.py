"""
Example demonstrating how to use Kompot's heatmap plotting functionality.

This example shows how to create heatmaps to visualize gene expression patterns
across different conditions or groups.
"""

import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import kompot as kp

# Set matplotlib to non-interactive mode
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
plt.ioff()  # Turn off interactive mode

# Load your data (for this example, we'll use a built-in dataset)
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

# For this example, we'll create a random grouping of cells
# In a real scenario, this would be your experimental groups or conditions
n_cells = adata.n_obs
groups = np.random.choice(['groupA', 'groupB', 'groupC'], size=n_cells)
adata.obs['group'] = groups

# Create conditions for diagonal split examples
conditions = np.random.choice(['conditionA', 'conditionB'], size=n_cells)
adata.obs['condition'] = conditions

# Store run info in adata.uns to simulate real kompot run
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

# Create a basic heatmap using default parameters
fig1, ax1 = plt.subplots(figsize=(12, 10))
kp.plot.heatmap(
    adata,
    groupby='group',
    standard_scale='var',  # Scale by gene (row)
    cmap='viridis',
    diagonal_split=False,  # Explicitly disable diagonal_split as we don't have condition data
    title='Top Differential Genes by Group',
    ax=ax1,
    return_fig=True
)
fig1.tight_layout()
fig1.savefig("basic_heatmap.png")
plt.close(fig1)

# Create a more customized heatmap
fig2, ax2 = plt.subplots(figsize=(12, 10))
kp.plot.heatmap(
    adata,
    n_top_genes=15,                     # Show top 15 genes
    groupby='group',
    lfc_key="kompot_de_mean_lfc_groupA_to_groupB",  # Explicitly specify the LFC key
    score_key="kompot_de_mahalanobis",              # Explicitly specify the score key
    standard_scale='var',               # Scale by gene (row)
    cmap='RdBu_r',                      # Red-blue colormap
    gene_labels_size=12,                # Larger gene label size
    group_labels_size=14,               # Larger group label size
    colorbar_title="Z-score",           # Custom colorbar title
    diagonal_split=False,               # Explicitly disable diagonal_split
    title="Gene Expression Heatmap: Top 15 Genes by Mahalanobis Distance",
    center=0,                           # Center colormap at 0
    ax=ax2,
    return_fig=True
)
fig2.tight_layout()
fig2.savefig("custom_heatmap.png")
plt.close(fig2)

# Create a heatmap with clustering and dendrograms
fig3, ax3 = plt.subplots(figsize=(12, 10))
kp.plot.heatmap(
    adata,
    n_top_genes=20,
    groupby='group',
    standard_scale='var',
    cmap='YlGnBu',
    dendrogram=True,                    # Show dendrograms
    diagonal_split=False,               # Explicitly disable diagonal_split
    title="Clustered Gene Expression Heatmap",
    ax=ax3,
    return_fig=True
)
fig3.tight_layout()
fig3.savefig("clustered_heatmap.png")
plt.close(fig3)

# Example of creating multiple heatmaps in one figure
fig4, axes = plt.subplots(1, 2, figsize=(20, 8))

# First heatmap - standard scaling by gene (row)
kp.plot.heatmap(
    adata,
    n_top_genes=15,
    groupby='group',
    standard_scale='var',
    cmap='YlOrRd',
    diagonal_split=False,               # Explicitly disable diagonal_split
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
    diagonal_split=False,               # Explicitly disable diagonal_split
    title="Scale by Group",
    ax=axes[1],
    return_fig=True
)

fig4.tight_layout()
fig4.savefig("multiple_heatmaps.png")
plt.close(fig4)

# Create a diagonal split heatmap with improved legend and colorbar
fig5, ax5 = plt.subplots(figsize=(12, 10))
kp.plot.heatmap(
    adata,
    n_top_genes=15,
    groupby='group',
    condition_column='condition',
    diagonal_split=True,
    standard_scale='var',
    cmap='RdBu_r',
    center=0,
    title="Diagonal Split Heatmap with Improved Legend",
    ax=ax5,
    return_fig=True
)
# Don't use tight_layout() for diagonal split, manually adjust spacing
fig5.subplots_adjust(right=0.85)  # Make room for colorbar and legend
fig5.savefig("diagonal_split_heatmap.png", bbox_inches='tight')
plt.close(fig5)

print("Heatmap examples created successfully!")