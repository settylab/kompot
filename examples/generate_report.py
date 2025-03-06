#!/usr/bin/env python

"""
Example demonstrating how to use Kompot's interactive HTML report generation.
"""

import numpy as np
import pandas as pd
import kompot
from kompot.reporter import HTMLReporter

# In a real-world scenario, you would load a real dataset
# and run Kompot analysis on it. Here we'll simulate some data.

# Create synthetic data
np.random.seed(42)
n_cells = 1000
n_genes = 500
n_landmarks = 50

# Simulate two conditions with differences in gene expression
X_condition1 = np.random.normal(0, 1, (n_cells, 2))  # 2D embedding
X_condition2 = np.random.normal(0.5, 1, (n_cells, 2))  # Shifted embedding

# Simulate gene expression
y_condition1 = np.random.normal(0, 1, (n_cells, n_genes))
y_condition2 = np.random.normal(0, 1, (n_cells, n_genes))

# Add some differential expression for a subset of genes
diff_genes = np.random.choice(n_genes, 100, replace=False)
y_condition2[:, diff_genes] += np.random.normal(2, 0.5, (n_cells, len(diff_genes)))

# Generate gene names
gene_names = [f"gene_{i}" for i in range(n_genes)]

# Run Kompot analysis
print("Running differential abundance analysis...")
diff_abundance = kompot.DifferentialAbundance(n_landmarks=n_landmarks)
diff_abundance.fit(X_condition1, X_condition2)

# Combined data for prediction
X_combined = np.vstack([X_condition1, X_condition2])

print("Running differential expression analysis...")
diff_expression = kompot.DifferentialExpression(
    n_landmarks=n_landmarks,
    compute_weighted_fold_change=True,  # Explicitly enable weighted fold change
    differential_abundance=diff_abundance  # Re-use the density estimator we already computed
)
diff_expression.fit(
    X_condition1, y_condition1,
    X_condition2, y_condition2
)

# Compute fold changes and metrics for abundance and expression
abundance_results = diff_abundance.predict(X_combined)
expression_results = diff_expression.predict(X_combined, compute_mahalanobis=True)

# Store class attributes for backward compatibility (will be used by reporter)
diff_abundance.log_fold_change = abundance_results['log_fold_change']
diff_abundance.log_fold_change_direction = abundance_results['log_fold_change_direction']
diff_expression.fold_change = expression_results['fold_change']
diff_expression.mahalanobis_distances = expression_results['mahalanobis_distances']
diff_expression.weighted_mean_log_fold_change = expression_results['weighted_mean_log_fold_change']

# Option 1: Using the convenience function
print("Generating report using convenience function...")
report_path = kompot.generate_report(
    diff_expression,
    output_dir="kompot_report_basic",
    condition1_name="Control",
    condition2_name="Treatment",
    gene_names=gene_names,
    title="Kompot Analysis: Control vs Treatment",
    subtitle="Example report with synthetic data"
)
print(f"Basic report generated at: {report_path}")

# Option 2: Using the HTMLReporter class directly for more control
print("Creating report with more options...")
reporter = HTMLReporter(
    output_dir="kompot_report_advanced",
    title="Advanced Kompot Analysis Report",
    subtitle="With method comparison"
)

# Add Kompot results
reporter.add_differential_expression(
    diff_expression,
    condition1_name="Control",
    condition2_name="Treatment",
    gene_names=gene_names,
    top_n=50  # Show top 50 genes
)

# Add comparison with a simulated other method
# In a real scenario, this would be DESeq2, edgeR, etc.
print("Adding method comparison...")
other_method_results = pd.DataFrame({
    "gene": gene_names,
    "log2FoldChange": np.random.normal(0, 1, n_genes),  # Simulated fold changes
    "pvalue": np.random.uniform(0, 1, n_genes),  # Simulated p-values
})

# Make it somewhat correlated with Kompot results
for i in diff_genes:
    other_method_results.loc[i, "log2FoldChange"] = diff_expression.fold_change[i] * 0.8 + np.random.normal(0, 0.5)
    other_method_results.loc[i, "pvalue"] = np.random.uniform(0, 0.01)

reporter.add_comparison(
    diff_expression,
    {"OtherMethod": other_method_results},
    gene_names=gene_names,
    comparison_name="Kompot vs Other Method"
)

# Generate the report
print("Generating advanced report...")
report_path = reporter.generate(open_browser=True)
print(f"Advanced report generated at: {report_path}")

print("Done! Check your browser for the interactive reports.")