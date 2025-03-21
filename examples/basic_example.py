#!/usr/bin/env python

"""
Basic example demonstrating how to use Kompot for differential abundance and expression analysis
with both the original API and the new scverse-compatible AnnData API.
"""

import numpy as np
import matplotlib.pyplot as plt
import umap
import pandas as pd

# Import Kompot
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import kompot

# Set random seed for reproducibility
np.random.seed(42)

# Generate some sample data
def generate_sample_data(n_cells=1000, n_genes=50, n_components=10):
    """Generate sample data for demonstration."""
    # Generate cell states in lower-dimensional space
    X_condition1 = np.random.randn(n_cells, n_components)
    X_condition2 = np.random.randn(n_cells, n_components) + 0.5  # Shift the second condition
    
    # Generate gene expression values
    def generate_expression(X):
        expression = np.zeros((X.shape[0], n_genes))
        
        # Basic expression patterns
        for gene in range(n_genes):
            # Each gene responds to a random subset of components
            weights = np.random.randn(n_components) * (np.random.rand(n_components) > 0.7)
            expression[:, gene] = np.dot(X, weights) + np.random.randn(X.shape[0]) * 0.2
            
        return expression
    
    y_condition1 = generate_expression(X_condition1)
    y_condition2 = generate_expression(X_condition2)
    
    # Make some genes differentially expressed
    diff_genes = np.random.choice(n_genes, size=int(n_genes * 0.2), replace=False)
    for gene in diff_genes:
        # Add an offset to make it differentially expressed
        y_condition2[:, gene] += np.random.randn() * 2
    
    return X_condition1, y_condition1, X_condition2, y_condition2, diff_genes

# Generate data
print("Generating sample data...")
X_condition1, y_condition1, X_condition2, y_condition2, diff_genes = generate_sample_data()

# Combine data for visualization
X_combined = np.vstack([X_condition1, X_condition2])
condition_labels = np.array(['Condition1'] * len(X_condition1) + ['Condition2'] * len(X_condition2))

# Create a UMAP embedding for visualization
print("Creating UMAP embedding for visualization...")
umap_reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = umap_reducer.fit_transform(X_combined)

print("\n==========================================")
print("PART 1: Using the original Kompot API")
print("==========================================")

# 1. Differential Abundance Analysis
print("\n1. Running Differential Abundance Analysis...")
# Create estimators and fit them
diff_abundance = kompot.DifferentialAbundance(n_landmarks=200)
diff_abundance.fit(X_condition1, X_condition2)

# Run prediction on the combined data to compute fold changes
X_combined = np.vstack([X_condition1, X_condition2])
abundance_results = diff_abundance.predict(X_combined)

# Visualize log fold changes
plt.figure(figsize=(10, 8))
plt.scatter(
    X_umap[:, 0], 
    X_umap[:, 1], 
    c=abundance_results['log_fold_change'],
    cmap='RdBu_r',
    alpha=0.7,
    s=5
)
plt.colorbar(label='Log Fold Change (Density)')
plt.title('Differential Abundance: Log Fold Change')
plt.tight_layout()
plt.savefig('diff_abundance_log_fold_change.png')

# 2. Differential Expression Analysis
print("\n2. Running Differential Expression Analysis...")

# Using the density estimator from the previous step (most efficient)
diff_expression = kompot.DifferentialExpression(
    n_landmarks=200,
    # Note: compute_weighted_fold_change has been removed in the latest version
    # Weighted fold changes can be computed with the standalone compute_weighted_mean_fold_change function
)
diff_expression.fit(
    X_condition1, y_condition1, 
    X_condition2, y_condition2
)

# Run prediction to compute fold changes and other metrics
# Compute Mahalanobis distances as well
expression_results = diff_expression.predict(X_combined, compute_mahalanobis=True)

# Visualize fold changes for a specific gene
gene_idx = diff_genes[0]  # Pick the first differentially expressed gene
plt.figure(figsize=(10, 8))
plt.scatter(
    X_umap[:, 0], 
    X_umap[:, 1], 
    c=expression_results['fold_change'][:, gene_idx],
    cmap='RdBu_r',
    alpha=0.7,
    s=5
)
plt.colorbar(label='Log Fold Change (Gene Expression)')
plt.title(f'Differential Expression: Log Fold Change for Gene {gene_idx}')
plt.tight_layout()
plt.savefig(f'diff_expression_gene_{gene_idx}_fold_change.png')

# Demonstrate cell condition labeling in prediction
print("\nDemonstrating cell condition labeling for prediction...")
# Create a smaller set of cells for prediction
X_new = np.vstack([X_condition1[:50], X_condition2[:50]])
# Create condition labels (0 for condition1, 1 for condition2)
cell_labels = np.array([0] * 50 + [1] * 50)

# Make prediction with cell condition labels
prediction = diff_expression.predict(X_new, cell_condition_labels=cell_labels)

# Visualize condition-specific mean fold changes
# First check if condition-specific fold changes are available in the prediction
has_condition_specific = (
    'condition1_cells_mean_fold_change' in prediction and 
    'condition2_cells_mean_fold_change' in prediction
)

plt.figure(figsize=(10, 6))
width = 0.35
x = np.arange(len(diff_genes[:5]))  # Just show the first 5 differential genes for clarity

if has_condition_specific:
    # Regular mean fold change
    plt.bar(
        x - width,
        prediction['mean_log_fold_change'][diff_genes[:5]],
        width=width,
        label='All Cells Mean Fold Change'
    )
    
    # Condition 1 cells mean fold change
    plt.bar(
        x,
        prediction['condition1_cells_mean_fold_change'][diff_genes[:5]],
        width=width,
        label='Condition 1 Cells Mean Fold Change'
    )
    
    # Condition 2 cells mean fold change
    plt.bar(
        x + width,
        prediction['condition2_cells_mean_fold_change'][diff_genes[:5]],
        width=width,
        label='Condition 2 Cells Mean Fold Change'
    )
else:
    # If condition-specific metrics aren't available, just show the overall mean
    print("Condition-specific fold changes not available in the prediction results.")
    print("This could be because cell_condition_labels weren't provided or weren't properly set.")
    
    # Regular mean fold change
    plt.bar(
        x,
        prediction['mean_log_fold_change'][diff_genes[:5]],
        width=width,
        label='Mean Fold Change'
    )

plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Differential Gene Index')
plt.ylabel('Log Fold Change')
plt.title('Mean Fold Change by Cell Condition')
plt.legend()
plt.tight_layout()
plt.savefig('fold_change_by_cell_condition.png')

# Visualize Mahalanobis distances
plt.figure(figsize=(10, 6))
mahalanobis_distances = expression_results['mahalanobis_distances']
top_gene_indices = np.argsort(mahalanobis_distances)[-20:]  # Top 20 genes
# Use matplotlib instead of seaborn
colors = plt.cm.viridis(np.linspace(0, 1, len(top_gene_indices)))
plt.bar(
    np.arange(len(top_gene_indices)),
    mahalanobis_distances[top_gene_indices],
    color=colors
)
plt.xticks(np.arange(len(top_gene_indices)), top_gene_indices)
plt.axhline(y=np.median(mahalanobis_distances), color='r', linestyle='--', label='Median')
plt.xticks(rotation=45)
plt.xlabel('Gene Index')
plt.ylabel('Mahalanobis Distance')
plt.title('Top 20 Genes by Mahalanobis Distance')
plt.legend()
plt.tight_layout()
plt.savefig('top_genes_mahalanobis.png')

# Visualize mean and weighted mean log fold changes
from kompot.differential import compute_weighted_mean_fold_change

plt.figure(figsize=(10, 6))
width = 0.35
x = np.arange(len(diff_genes))  # Just show the differential genes

# Use the standalone function to compute weighted mean log fold change
weighted_mean_lfc = compute_weighted_mean_fold_change(
    expression_results['fold_change'],
    log_density_condition1=abundance_results['log_density_condition1'],
    log_density_condition2=abundance_results['log_density_condition2']
)

plt.bar(
    x - width/2,
    expression_results['mean_log_fold_change'][diff_genes],
    width=width,
    label='Mean Log Fold Change'
)
plt.bar(
    x + width/2,
    weighted_mean_lfc[diff_genes],
    width=width,
    label='Weighted Mean Log Fold Change'
)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Differential Gene Index')
plt.ylabel('Log Fold Change')
plt.title('Mean vs Weighted Mean Log Fold Change')
plt.legend()
plt.tight_layout()
plt.savefig('mean_to_weighted_mean_log_fold_change.png')

print("\n==========================================")
print("PART 2: Using the scverse-compatible AnnData API")
print("==========================================")

# Convert the data to AnnData format
try:
    import anndata
    
    # Create an AnnData object with all data
    print("\nCreating AnnData object...")
    # Create a combined expression matrix
    y_combined = np.vstack([y_condition1, y_condition2])
    
    # Create gene names
    gene_names = [f"gene_{i}" for i in range(y_combined.shape[1])]
    
    # Create metadata
    obs = pd.DataFrame({
        'condition': condition_labels,
        'is_condition1': np.array([True] * len(X_condition1) + [False] * len(X_condition2)),
        'is_condition2': np.array([False] * len(X_condition1) + [True] * len(X_condition2)),
    })
    
    var = pd.DataFrame(index=gene_names)
    var['is_diff_gene'] = [i in diff_genes for i in range(len(gene_names))]
    
    # Create the AnnData object
    adata = anndata.AnnData(
        X=y_combined,
        obs=obs,
        var=var
    )
    
    # Add cell states to obsm
    adata.obsm['DM_EigenVectors'] = X_combined
    adata.obsm['X_umap'] = X_umap
    
    # Run the complete analysis workflow
    print("\n3. Running complete differential analysis with AnnData API...")
    results = kompot.run_differential_analysis(
        adata, 
        groupby='condition', 
        condition1='Condition1', 
        condition2='Condition2',
        obsm_key='DM_EigenVectors',
        n_landmarks=200,
        generate_html_report=True,
        report_dir='kompot_report',
        open_browser=False  # Set to False for headless environments
    )
    
    # Compare the results between the original API and the AnnData API
    print("\n4. Comparing results between original API and AnnData API...")
    
    # Extract the AnnData object from the results
    adata = results["adata"]
    
    # Get top genes from both approaches - use consistent ordering for both
    # Original API - sort in descending order to match AnnData API
    top_genes_original = np.argsort(-diff_expression.mahalanobis_distances)[:10]
    # AnnData API - already sorts in descending order
    top_genes_anndata = adata.var.sort_values('kompot_de_mahalanobis', ascending=False).index[:10]
    
    print(f"Top 10 genes by original API: {top_genes_original}")
    print(f"Top 10 genes by AnnData API: {[int(g.split('_')[1]) for g in top_genes_anndata]}")
    
    # Plot gene expression for a top differential gene
    top_gene = top_genes_anndata[0]
    top_gene_idx = int(top_gene.split('_')[1])
    
    # Compare the original and anndata results for this gene
    plt.figure(figsize=(12, 6))
    
    # Original API result
    plt.subplot(1, 2, 1)
    plt.scatter(
        X_umap[:, 0], 
        X_umap[:, 1], 
        c=expression_results['fold_change'][:, top_gene_idx],
        cmap='RdBu_r',
        alpha=0.7,
        s=5
    )
    plt.colorbar(label='Log Fold Change')
    plt.title(f'Original API: Log Fold Change for {top_gene}')
    
    # AnnData API result - find the fold change layer (with the new more descriptive names)
    plt.subplot(1, 2, 2)
    # Find layers that contain 'fold_change'
    fold_change_layers = [layer for layer in adata.layers.keys() if 'fold_change' in layer]
    if fold_change_layers:
        fold_change_layer = fold_change_layers[0]  # Use the first matching layer
        plt.scatter(
            X_umap[:, 0], 
            X_umap[:, 1], 
            c=adata.layers[fold_change_layer][:, top_gene_idx],
            cmap='RdBu_r',
            alpha=0.7,
            s=5
        )
    else:
        plt.text(0.5, 0.5, "Fold change layer not found", ha='center', va='center')
    plt.colorbar(label='Log Fold Change')
    plt.title(f'AnnData API: Log Fold Change for {top_gene}')
    
    plt.tight_layout()
    plt.savefig('comparison_anndata_api.png')
    
    print("\nAnnData API workflow complete. Results are stored in the AnnData object.")
    print("Key results locations:")
    
    # Find and print the actual column/layer names that exist in the data
    mahalanobis_col = next((col for col in adata.var.columns if 'mahalanobis' in col), None)
    fold_change_obs = next((col for col in adata.obs.columns if 'log_fold_change' in col), None)
    imputed_layer = next((layer for layer in adata.layers.keys() if 'imputed' in layer), None)
    fold_change_layer = next((layer for layer in adata.layers.keys() if 'fold_change' in layer), None)
    
    print(f"  - Gene scores: adata.var['{mahalanobis_col}']" if mahalanobis_col else "  - Gene scores: Not found")
    print(f"  - Cell scores: adata.obs['{fold_change_obs}']" if fold_change_obs else "  - Cell scores: Not found")
    print(f"  - Imputed expression: adata.layers['{imputed_layer}']" if imputed_layer else "  - Imputed expression: Not found")
    print(f"  - Log fold changes: adata.layers['{fold_change_layer}']" if fold_change_layer else "  - Log fold changes: Not found")
    print("  - Full models: Available in the returned results dictionary, not stored in AnnData")
    
    # Print the log density columns
    print("\nLog density columns in AnnData:")
    for col in adata.obs.columns:
        if 'log_density' in col:
            print(f"  - {col}")
    
    # Print the weighted mean fold change - look for the column with the condition names
    weighted_lfc_cols = [col for col in adata.var.columns if 'weighted_lfc' in col]
    if weighted_lfc_cols:
        weighted_col = weighted_lfc_cols[0]  # Use the first weighted LFC column found
        print(f"\nWeighted mean fold change computed successfully! (column: {weighted_col})")
        print("First few genes with weighted mean LFC values:")
        for gene, value in adata.var[weighted_col].iloc[:5].items():
            print(f"  - {gene}: {value:.4f}")
    
    print("\nHTML report generated at: kompot_report/index.html")
    
except ImportError:
    print("\nAnnData is not installed. Skipping the AnnData API example.")
    print("To run the AnnData example, install anndata: pip install anndata")

print("\nAnalysis complete. Visualizations saved to current directory.")