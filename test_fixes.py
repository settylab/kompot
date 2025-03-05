import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import anndata as ad
import kompot

print(f"Kompot version: {kompot.__version__}")
print(f"Scanpy version: {sc.__version__}")
print(f"AnnData version: {ad.__version__}")

# Create a synthetic AnnData object for demonstration
def create_synthetic_adata(n_cells=1000, n_genes=500):
    """Create a synthetic AnnData object for demonstration purposes"""
    # Generate random counts data
    X = np.random.negative_binomial(5, 0.3, size=(n_cells, n_genes))
    
    # Create cell metadata
    obs = pd.DataFrame({
        'Age': np.random.choice(['Young', 'Old'], size=n_cells),
        'highres_celltype': np.random.choice([
            'HSC', 'MPP', 'LMPP', 'CMP', 'GMP', 'MEP'
        ], size=n_cells)
    })
    
    # Create gene metadata
    var = pd.DataFrame(index=[f'gene_{i}' for i in range(n_genes)])
    
    # Create AnnData object
    adata = ad.AnnData(X=X, obs=obs, var=var)
    
    # Log-normalize the data
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.layers['logged_counts'] = adata.X.copy()
    
    # Generate PCA and UMAP
    sc.pp.scale(adata)
    sc.tl.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    
    return adata

# Create a synthetic dataset
print("Creating a synthetic dataset...")
adata = create_synthetic_adata()

# Set analysis parameters
GROUPING_COLUMN = 'Age'
CONDITIONS = ['Young', 'Old']
CELL_TYPE_COLUMN = 'highres_celltype'
DIMENSIONALITY_REDUCTION = 'X_pca'
LAYER_FOR_EXPRESSION = 'logged_counts'
UMAP_BASIS = 'X_umap'

# Get the conditions
condition1 = CONDITIONS[0]  # Reference condition (e.g., 'Young')
condition2 = CONDITIONS[1]  # Comparison condition (e.g., 'Old')

print(f"Running differential abundance analysis: {condition2} vs {condition1}...")

try:
    # Run differential abundance analysis WITHOUT random_state
    da_results = kompot.compute_differential_abundance(
        adata,
        groupby=GROUPING_COLUMN,
        condition1=condition1,
        condition2=condition2,
        obsm_key=DIMENSIONALITY_REDUCTION,
        n_landmarks=200
    )
    print("Differential abundance analysis successful\!")
except Exception as e:
    print(f"Error during differential abundance analysis: {str(e)}")

print(f"Running differential expression analysis: {condition2} vs {condition1}...")

try:
    # Run differential expression analysis WITH jit_compile=False
    de_results = kompot.compute_differential_expression(
        adata,
        groupby=GROUPING_COLUMN,
        condition1=condition1,
        condition2=condition2,
        layer=LAYER_FOR_EXPRESSION,
        n_landmarks=200,
        jit_compile=False
    )
    print("Differential expression analysis successful\!")
except Exception as e:
    print(f"Error during differential expression analysis: {str(e)}")
