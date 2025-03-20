"""
Example script demonstrating the use of the embedding function.

This script shows how to use the embedding function with group filtering.
"""

import scanpy as sc
import kompot.plot as kp
import numpy as np
import matplotlib.pyplot as plt

# Load a test dataset
adata = sc.datasets.pbmc3k_processed()

# Add some random categorical annotations
np.random.seed(42)
adata.obs['condition'] = np.random.choice(['control', 'treatment'], size=adata.n_obs)
adata.obs['batch'] = np.random.choice(['batch1', 'batch2', 'batch3'], size=adata.n_obs)
adata.obs['status'] = np.random.choice(['positive', 'negative', 'unknown'], size=adata.n_obs,
                                      p=[0.3, 0.3, 0.4])

# Basic embedding plot (similar to standard scanpy)
print("Plotting basic embedding...")
kp.embedding(adata, color='leiden', title='UMAP by cluster')

# Same plot but using the 'umap' basis name without 'X_' prefix (scanpy style)
print("Plotting with 'umap' basis name (without X_ prefix)...")
kp.embedding(adata, basis='umap', color='leiden', title='UMAP by cluster (without X_ prefix)')

# Filtering by a single group
groups = {'condition': ['treatment']}
print("Plotting with single group filter...")
kp.embedding(adata, 
             groups=groups,
             color='leiden', 
             title='Treatment cells only')

# Filtering by multiple conditions
groups = {'condition': ['treatment'], 'batch': ['batch1']}
print("Plotting with multiple group filters...")
kp.embedding(adata, 
             groups=groups,
             color='leiden', 
             title='Treatment cells from batch1')

# Multiple panels (color as list) with group filtering
groups = {'condition': ['treatment']}
print("Plotting multiple panels with group filter...")
kp.embedding(adata, 
             groups=groups,
             color=['leiden', 'n_genes', 'status'], 
             ncols=2,
             wspace=0.4,
             title=['Clusters', 'Number of genes', 'Status'])

# Hiding background cells
groups = {'status': ['positive']}
print("Plotting without background cells...")
kp.embedding(adata, 
             groups=groups,
             color='leiden', 
             show_background=False,
             title='Positive cells only')

# Custom background color
groups = {'status': ['positive']}
print("Plotting with custom background color...")
kp.embedding(adata, 
             groups=groups,
             color='leiden', 
             background_color='lightblue',
             title='Positive cells with light blue background')

# Using the color_map parameter with a diverging colormap
print("Plotting with a diverging colormap...")
kp.embedding(adata, 
             color='n_genes', 
             color_map='RdBu_r',
             vcenter=np.median(adata.obs['n_genes']),
             title='Number of genes with diverging colormap')

print("All plots completed! Close the plots to exit.")