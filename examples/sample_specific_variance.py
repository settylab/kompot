"""
Example script demonstrating the sample-specific variance functionality in Kompot.

This example shows how to use sample indices or sample_col to improve variance estimation 
in differential expression and differential abundance analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from kompot.differential import DifferentialExpression, DifferentialAbundance

def gene_expression_example():
    """Example using sample-specific variance with DifferentialExpression."""
    print("Running differential expression with sample-specific variance example...\n")
    
    # Create sample indices for 3 samples per condition
    n_cells_per_sample = 50
    n_samples = 3
    n_genes = 10
    n_features = 5

    # Create cell data
    X_condition1_samples = []
    y_condition1_samples = []
    X_condition2_samples = []
    y_condition2_samples = []

    # Generate data for each sample with varying characteristics
    for i in range(n_samples):
        # Each sample has slightly different characteristics
        X_sample1 = np.random.randn(n_cells_per_sample, n_features) + i * 0.2
        y_sample1 = np.random.randn(n_cells_per_sample, n_genes) + i * 0.1
        
        X_sample2 = np.random.randn(n_cells_per_sample, n_features) + 0.5 + i * 0.2
        y_sample2 = np.random.randn(n_cells_per_sample, n_genes) + 1.0 + i * 0.1
        
        X_condition1_samples.append(X_sample1)
        y_condition1_samples.append(y_sample1)
        X_condition2_samples.append(X_sample2)
        y_condition2_samples.append(y_sample2)

    # Combine samples
    X_condition1 = np.vstack(X_condition1_samples)
    y_condition1 = np.vstack(y_condition1_samples)
    X_condition2 = np.vstack(X_condition2_samples)
    y_condition2 = np.vstack(y_condition2_samples)

    # Create sample indices
    condition1_sample_indices = np.array([0] * n_cells_per_sample + [1] * n_cells_per_sample + [2] * n_cells_per_sample)
    condition2_sample_indices = np.array([0] * n_cells_per_sample + [1] * n_cells_per_sample + [2] * n_cells_per_sample)

    print(f"Condition 1 data shape: {X_condition1.shape}, {y_condition1.shape}")
    print(f"Condition 2 data shape: {X_condition2.shape}, {y_condition2.shape}")
    print(f"Sample indices shape: {condition1_sample_indices.shape}, {condition2_sample_indices.shape}")

    # Create test data points
    X_test = np.random.randn(100, n_features) * 0.8

    # Train two different models:
    # Model without sample variance
    print("\nTraining differential expression model WITHOUT sample variance...")
    diff_expr_no_variance = DifferentialExpression(
        use_sample_variance=False
    )
    diff_expr_no_variance.fit(X_condition1, y_condition1, X_condition2, y_condition2)

    # Model with sample variance (using sample indices)
    print("\nTraining differential expression model WITH sample variance...")
    diff_expr_with_variance = DifferentialExpression(
        use_sample_variance=True
    )
    diff_expr_with_variance.fit(
        X_condition1, y_condition1, 
        X_condition2, y_condition2,
        condition1_sample_indices=condition1_sample_indices,
        condition2_sample_indices=condition2_sample_indices
    )
    
    # Make predictions with both models
    print("\nMaking predictions with both models...")
    pred_no_variance = diff_expr_no_variance.predict(X_test)
    pred_with_variance = diff_expr_with_variance.predict(X_test)

    # Compare fold changes
    print("\nMean fold change comparison:")
    print(f"Without sample variance: {pred_no_variance['mean_log_fold_change'].mean():.4f}")
    print(f"With sample variance: {pred_with_variance['mean_log_fold_change'].mean():.4f}")

    # Compare z-scores
    print("\nZ-score statistics:")
    print(f"Without sample variance - mean: {pred_no_variance['fold_change_zscores'].mean():.4f}, std: {pred_no_variance['fold_change_zscores'].std():.4f}")
    print(f"With sample variance - mean: {pred_with_variance['fold_change_zscores'].mean():.4f}, std: {pred_with_variance['fold_change_zscores'].std():.4f}")

    # Optional: Create visualizations if matplotlib is available
    try:
        # Calculate z-score means for each gene
        z_no_variance = np.mean(pred_no_variance['fold_change_zscores'], axis=0)
        z_with_variance = np.mean(pred_with_variance['fold_change_zscores'], axis=0)

        # Plot bar chart of z-scores
        plt.figure(figsize=(10, 6))
        x = np.arange(len(z_no_variance))
        width = 0.35

        plt.bar(x - width/2, z_no_variance, width, label='Without Sample Variance')
        plt.bar(x + width/2, z_with_variance, width, label='With Sample Variance')

        plt.xlabel('Gene')
        plt.ylabel('Mean Z-score')
        plt.title('Comparison of Z-scores with and without Sample-Specific Variance')
        plt.xticks(x, [f'Gene {i+1}' for i in range(len(z_no_variance))])
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('expression_variance_comparison.png')
        print("\nVisualization saved as 'expression_variance_comparison.png'")
        
    except ImportError:
        print("\nMatplotlib not available, skipping visualization.")
        
    print("\nIn a real application with AnnData, you would use:")
    print("diff_expr = DifferentialExpression(use_sample_variance=True, sample_col='sample_id')")
    print("# or")
    print("compute_differential_expression(adata, groupby='condition', condition1='control', condition2='treatment', sample_col='sample_id')")

def density_variance_example():
    """Example using sample-specific variance with DifferentialAbundance."""
    print("\n\nRunning differential abundance with sample-specific variance example...\n")
    
    # Create sample indices for 3 samples per condition
    n_cells_per_sample = 50
    n_samples = 3
    n_features = 5

    # Create sample data with distinct sample characteristics
    X_condition1_samples = []
    X_condition2_samples = []
    
    # Simulate different samples with slightly different distributions
    for i in range(n_samples):
        # Each sample has slightly different characteristics
        X_sample1 = np.random.randn(n_cells_per_sample, n_features) + i * 0.2
        X_sample2 = np.random.randn(n_cells_per_sample, n_features) + 0.5 + i * 0.2
        
        X_condition1_samples.append(X_sample1)
        X_condition2_samples.append(X_sample2)
    
    # Combine samples
    X_condition1 = np.vstack(X_condition1_samples)
    X_condition2 = np.vstack(X_condition2_samples)
    
    # Create sample indices
    condition1_sample_indices = np.array([0] * n_cells_per_sample + [1] * n_cells_per_sample + [2] * n_cells_per_sample)
    condition2_sample_indices = np.array([0] * n_cells_per_sample + [1] * n_cells_per_sample + [2] * n_cells_per_sample)
    
    print(f"Condition 1 data shape: {X_condition1.shape}")
    print(f"Condition 2 data shape: {X_condition2.shape}")
    print(f"Sample indices shape: {condition1_sample_indices.shape}, {condition2_sample_indices.shape}")
    
    # Create test data points
    X_test = np.random.randn(100, n_features) * 0.8
    
    # Create and fit model without sample variance
    print("\nTraining differential abundance model WITHOUT sample variance...")
    diff_abundance_no_variance = DifferentialAbundance(
        use_sample_variance=False
    )
    diff_abundance_no_variance.fit(X_condition1, X_condition2)
    
    # Create and fit model with sample variance
    print("\nTraining differential abundance model WITH sample variance...")
    diff_abundance_with_variance = DifferentialAbundance(
        use_sample_variance=True
    )
    diff_abundance_with_variance.fit(
        X_condition1, 
        X_condition2,
        condition1_sample_indices=condition1_sample_indices,
        condition2_sample_indices=condition2_sample_indices
    )
    
    # Make predictions with both models
    print("\nMaking predictions with both models...")
    pred_no_variance = diff_abundance_no_variance.predict(X_test)
    pred_with_variance = diff_abundance_with_variance.predict(X_test)
    
    # Compare log fold changes
    print("\nLog fold change comparison:")
    print(f"Without sample variance - mean: {pred_no_variance['log_fold_change'].mean():.4f}, std: {pred_no_variance['log_fold_change'].std():.4f}")
    print(f"With sample variance - mean: {pred_with_variance['log_fold_change'].mean():.4f}, std: {pred_with_variance['log_fold_change'].std():.4f}")
    
    # Compare z-scores
    print("\nZ-score statistics:")
    print(f"Without sample variance - mean: {pred_no_variance['log_fold_change_zscore'].mean():.4f}, std: {pred_no_variance['log_fold_change_zscore'].std():.4f}")
    print(f"With sample variance - mean: {pred_with_variance['log_fold_change_zscore'].mean():.4f}, std: {pred_with_variance['log_fold_change_zscore'].std():.4f}")
    
    # Compare uncertainty
    print("\nUncertainty comparison:")
    print(f"Without sample variance - mean: {pred_no_variance['log_fold_change_uncertainty'].mean():.4f}")
    print(f"With sample variance - mean: {pred_with_variance['log_fold_change_uncertainty'].mean():.4f}")
    
    # Optional: Create visualizations if matplotlib is available
    try:
        # Plot histogram of log fold change z-scores
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(pred_no_variance['log_fold_change_zscore'], bins=20, alpha=0.7, label='Without Sample Variance')
        plt.hist(pred_with_variance['log_fold_change_zscore'], bins=20, alpha=0.7, label='With Sample Variance')
        plt.xlabel('Log Fold Change Z-score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Z-scores')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.scatter(pred_no_variance['log_fold_change'], -np.log10(np.exp(pred_no_variance['neg_log10_fold_change_pvalue'])), 
                   alpha=0.7, label='Without Sample Variance')
        plt.scatter(pred_with_variance['log_fold_change'], -np.log10(np.exp(pred_with_variance['neg_log10_fold_change_pvalue'])), 
                   alpha=0.7, label='With Sample Variance')
        plt.xlabel('Log Fold Change')
        plt.ylabel('-log10(p-value)')
        plt.title('Volcano Plot Comparison')
        plt.axhline(-np.log10(0.05), color='gray', linestyle='--')
        plt.axvline(-1, color='gray', linestyle='--')
        plt.axvline(1, color='gray', linestyle='--')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('density_variance_comparison.png')
        print("\nVisualization saved as 'density_variance_comparison.png'")
        
    except ImportError:
        print("\nMatplotlib not available, skipping visualization.")
    
    print("\nIn a real application with AnnData, you would use:")
    print("diff_abundance = DifferentialAbundance(use_sample_variance=True, sample_col='sample_id')")
    print("# or")
    print("compute_differential_abundance(adata, groupby='condition', condition1='control', condition2='treatment', sample_col='sample_id')")
    
    # Count the number of significant points in each model
    print("\nSignificant differences:")
    significant_no_variance = np.sum(pred_no_variance['log_fold_change_direction'] != 'neutral')
    significant_with_variance = np.sum(pred_with_variance['log_fold_change_direction'] != 'neutral')
    print(f"Without sample variance: {significant_no_variance} points significant")
    print(f"With sample variance: {significant_with_variance} points significant")

def main():
    """Run both examples."""
    # Example for differential expression
    gene_expression_example()
    
    # Example for differential abundance
    density_variance_example()
    
    print("\nSample-specific variance provides more accurate uncertainty estimation by accounting")
    print("for biological variability between samples, leading to more robust statistical inference.")
    print("This can be enabled for both differential expression and differential abundance analysis.")

if __name__ == "__main__":
    main()