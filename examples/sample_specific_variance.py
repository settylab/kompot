"""
Example script demonstrating the new sample-specific empirical variance functionality in Kompot.

This example shows how to use sample indices to improve variance estimation in differential expression analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from kompot.differential import DifferentialExpression

def main():
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

    # Train three different models:
    # 1. Standard approach
    print("\nTraining standard differential expression model...")
    diff_expr_standard = DifferentialExpression(
        use_empirical_variance=False
    )
    diff_expr_standard.fit(X_condition1, y_condition1, X_condition2, y_condition2)

    # 2. Traditional empirical variance approach
    print("\nTraining traditional empirical variance model...")
    diff_expr_traditional = DifferentialExpression(
        use_empirical_variance=True
    )
    diff_expr_traditional.fit(X_condition1, y_condition1, X_condition2, y_condition2)

    # 3. Sample-specific empirical variance approach
    print("\nTraining sample-specific empirical variance model...")
    diff_expr_sample_specific = DifferentialExpression(
        use_sample_specific_variance=True,  # This also sets use_empirical_variance=True automatically
        condition1_sample_indices=condition1_sample_indices,
        condition2_sample_indices=condition2_sample_indices,
        sample_variance_use_estimators=True  # Use FunctionEstimator for each sample group
    )
    diff_expr_sample_specific.fit(X_condition1, y_condition1, X_condition2, y_condition2)

    # Create test data points
    X_test = np.random.randn(100, n_features) * 0.8

    # Make predictions with each model
    print("\nMaking predictions with all models...")
    pred_standard = diff_expr_standard.predict(X_test)
    pred_traditional = diff_expr_traditional.predict(X_test)
    pred_sample_specific = diff_expr_sample_specific.predict(X_test)

    # Compare fold changes (should be similar for all models)
    print("\nMean fold change comparison:")
    print(f"Standard: {pred_standard['mean_log_fold_change'].mean():.4f}")
    print(f"Traditional: {pred_traditional['mean_log_fold_change'].mean():.4f}")
    print(f"Sample-specific: {pred_sample_specific['mean_log_fold_change'].mean():.4f}")
    
    # NOTE: If you want to compute weighted mean fold change, you would need to:
    # 1. Create a DifferentialAbundance object and fit it
    # 2. Get density predictions 
    # 3. Explicitly provide them to the predict method
    # 4. Or use the standalone compute_weighted_mean_fold_change function
    """
    # Example of computing weighted mean fold change:
    from kompot.differential import DifferentialAbundance, compute_weighted_mean_fold_change
    
    # Fit a density estimator
    diff_abundance = DifferentialAbundance()
    diff_abundance.fit(X_condition1, X_condition2)
    
    # Get density predictions
    density_preds = diff_abundance.predict(X_test)
    
    # Method 1: Pass to predict method
    pred_with_density = diff_expr_standard.predict(
        X_test, 
        density_predictions=density_preds
    )
    # Weighted fold change is in pred_with_density['weighted_mean_log_fold_change']
    
    # Method 2: Use standalone function
    weighted_fc = compute_weighted_mean_fold_change(
        pred_standard['fold_change'],
        log_density_condition1=density_preds['log_density_condition1'],
        log_density_condition2=density_preds['log_density_condition2']
    )
    """

    # Compare z-scores (will differ based on variance calculation)
    print("\nZ-score statistics:")
    print(f"Standard - mean: {pred_standard['fold_change_zscores'].mean():.4f}, std: {pred_standard['fold_change_zscores'].std():.4f}")
    print(f"Traditional - mean: {pred_traditional['fold_change_zscores'].mean():.4f}, std: {pred_traditional['fold_change_zscores'].std():.4f}")
    print(f"Sample-specific - mean: {pred_sample_specific['fold_change_zscores'].mean():.4f}, std: {pred_sample_specific['fold_change_zscores'].std():.4f}")

    # Get variance statistics from the sample-specific model
    if hasattr(diff_expr_sample_specific, 'empiric_variance_estimator') and diff_expr_sample_specific.empiric_variance_estimator is not None:
        variance_summary = diff_expr_sample_specific.empiric_variance_estimator.get_sample_variance_summary()
        
        print("\nSample Variance Statistics:")
        print("Condition 1:")
        for sample_id, stats in variance_summary['condition1'].items():
            print(f"  Sample {sample_id}: {stats['n_cells']} cells, mean error: {stats['mean_error']:.4f}")
        
        print("\nCondition 2:")
        for sample_id, stats in variance_summary['condition2'].items():
            print(f"  Sample {sample_id}: {stats['n_cells']} cells, mean error: {stats['mean_error']:.4f}")

    # Optional: Create visualizations if matplotlib is available
    try:
        # Calculate z-score means for each gene
        z_standard = np.mean(pred_standard['fold_change_zscores'], axis=0)
        z_traditional = np.mean(pred_traditional['fold_change_zscores'], axis=0)
        z_sample = np.mean(pred_sample_specific['fold_change_zscores'], axis=0)

        # Plot bar chart of z-scores
        plt.figure(figsize=(10, 6))
        x = np.arange(len(z_standard))
        width = 0.25

        plt.bar(x - width, z_standard, width, label='Standard')
        plt.bar(x, z_traditional, width, label='Traditional Empirical')
        plt.bar(x + width, z_sample, width, label='Sample-Specific')

        plt.xlabel('Gene')
        plt.ylabel('Mean Z-score')
        plt.title('Comparison of Z-scores across different variance estimation methods')
        plt.xticks(x, [f'Gene {i+1}' for i in range(len(z_standard))])
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('variance_methods_comparison.png')
        print("\nVisualization saved as 'variance_methods_comparison.png'")
        
    except ImportError:
        print("\nMatplotlib not available, skipping visualization.")

    print("\nSample-specific empirical variance provides more accurate variance estimates by accounting for")
    print("biological variability between samples, leading to more robust statistical inference.")

if __name__ == "__main__":
    main()