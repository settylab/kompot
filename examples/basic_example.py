#!/usr/bin/env python

"""
Basic example demonstrating how to use Kompot for differential abundance and expression analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap

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
condition_labels = np.array(['Condition 1'] * len(X_condition1) + ['Condition 2'] * len(X_condition2))

# Create a UMAP embedding for visualization
print("Creating UMAP embedding for visualization...")
umap_reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = umap_reducer.fit_transform(X_combined)

# 1. Differential Abundance Analysis
print("\n1. Running Differential Abundance Analysis...")
diff_abundance = kompot.DifferentialAbundance(n_landmarks=200)
diff_abundance.fit(X_condition1, X_condition2)

# Visualize log fold changes
plt.figure(figsize=(10, 8))
plt.scatter(
    X_umap[:, 0], 
    X_umap[:, 1], 
    c=diff_abundance.log_fold_change,
    cmap='RdBu_r',
    alpha=0.7,
    s=5
)
plt.colorbar(label='Log Fold Change (Density)')
plt.title('Differential Abundance: Log Fold Change')
plt.tight_layout()
plt.savefig('diff_abundance_log_fold_change.png')

# Visualize z-scores
plt.figure(figsize=(10, 8))
plt.scatter(
    X_umap[:, 0], 
    X_umap[:, 1], 
    c=diff_abundance.log_fold_change_zscore,
    cmap='RdBu_r',
    alpha=0.7,
    s=5
)
plt.colorbar(label='Z-score')
plt.title('Differential Abundance: Z-scores')
plt.tight_layout()
plt.savefig('diff_abundance_zscores.png')

# 2. Differential Expression Analysis
print("\n2. Running Differential Expression Analysis...")

# Method 1: Using the density estimator from the previous step (most efficient)
print("Method 1: Using pre-computed differential abundance...")
diff_expression = kompot.DifferentialExpression(
    n_landmarks=200,
    differential_abundance=diff_abundance  # Re-use the density estimator we already computed
)
diff_expression.fit(
    X_condition1, y_condition1, 
    X_condition2, y_condition2
)

# Method 2: Computing everything from scratch (simpler but less efficient)
print("Method 2: Computing everything from scratch...")
diff_expression_scratch = kompot.DifferentialExpression(n_landmarks=200)
diff_expression_scratch.fit(
    X_condition1, y_condition1, 
    X_condition2, y_condition2
)

# Method 3: Disabling weighted fold change computation
print("Method 3: Without weighted fold change computation...")
diff_expression_no_weight = kompot.DifferentialExpression(
    n_landmarks=200,
    compute_weighted_fold_change=False
)
diff_expression_no_weight.fit(
    X_condition1, y_condition1, 
    X_condition2, y_condition2
)

# Visualize fold changes for a specific gene
gene_idx = diff_genes[0]  # Pick the first differentially expressed gene
plt.figure(figsize=(10, 8))
plt.scatter(
    X_umap[:, 0], 
    X_umap[:, 1], 
    c=diff_expression.fold_change[:, gene_idx],
    cmap='RdBu_r',
    alpha=0.7,
    s=5
)
plt.colorbar(label='Log Fold Change (Gene Expression)')
plt.title(f'Differential Expression: Log Fold Change for Gene {gene_idx}')
plt.tight_layout()
plt.savefig(f'diff_expression_gene_{gene_idx}_fold_change.png')

# Visualize Mahalanobis distances
plt.figure(figsize=(10, 6))
top_gene_indices = np.argsort(diff_expression.mahalanobis_distances)[-20:]  # Top 20 genes
sns.barplot(
    x=top_gene_indices,
    y=diff_expression.mahalanobis_distances[top_gene_indices],
    palette='viridis'
)
plt.axhline(y=np.median(diff_expression.mahalanobis_distances), color='r', linestyle='--', label='Median')
plt.xticks(rotation=45)
plt.xlabel('Gene Index')
plt.ylabel('Mahalanobis Distance')
plt.title('Top 20 Genes by Mahalanobis Distance')
plt.legend()
plt.tight_layout()
plt.savefig('top_genes_mahalanobis.png')

# Check if our known differential genes are detected
known_diff_mahalanobis = diff_expression.mahalanobis_distances[diff_genes]
other_genes_mahalanobis = np.delete(diff_expression.mahalanobis_distances, diff_genes)

plt.figure(figsize=(8, 6))
plt.hist(other_genes_mahalanobis, bins=20, alpha=0.5, label='Non-Differential Genes')
plt.hist(known_diff_mahalanobis, bins=20, alpha=0.5, label='Known Differential Genes')
plt.xlabel('Mahalanobis Distance')
plt.ylabel('Frequency')
plt.title('Distribution of Mahalanobis Distances')
plt.legend()
plt.tight_layout()
plt.savefig('mahalanobis_distribution.png')

print("\nAnalysis complete. Visualizations saved to current directory.")
print(f"Known differential genes: {diff_genes}")
print(f"Top genes by Mahalanobis distance: {top_gene_indices}")

# Compute precision/recall statistics for the different methods
top_n = 10
methods = {
    "With pre-computed differential abundance": diff_expression,
    "Computed from scratch": diff_expression_scratch,
    "Without weighted fold change": diff_expression_no_weight
}

print(f"\nEvaluation metrics (top {top_n} genes):")
for name, model in methods.items():
    top_genes = np.argsort(model.mahalanobis_distances)[-top_n:]
    true_positives = len(set(top_genes) & set(diff_genes))
    precision = true_positives / top_n
    recall = true_positives / len(diff_genes)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n{name}:")
    print(f"  Precision: {precision:.2f}")
    print(f"  Recall: {recall:.2f}")
    print(f"  F1 Score: {f1_score:.2f}")
    
    if name == "Without weighted fold change":
        print("  Weighted mean log fold change: Not computed")
    else:
        # Check if any of the actual differential genes are in top weighted fold changes
        gene_ranks_by_weighted_fold_change = np.argsort(np.abs(model.weighted_mean_log_fold_change))[::-1]
        top_by_wfc = gene_ranks_by_weighted_fold_change[:top_n]
        wfc_true_positives = len(set(top_by_wfc) & set(diff_genes))
        wfc_precision = wfc_true_positives / top_n
        print(f"  Top genes by weighted fold change - Precision: {wfc_precision:.2f}")