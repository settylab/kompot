"""
Example script demonstrating the nearest neighbor approach for sample-specific empirical variance in Kompot.

This example shows how to use the nearest neighbor method instead of function estimators
for sample-specific variance estimation in differential expression analysis.
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

    # Train two different sample-specific models:
    
    # 1. Sample-specific with Function Estimators
    print("\nTraining sample-specific model with function estimators...")
    diff_expr_estimator = DifferentialExpression(
        use_sample_specific_variance=True,
        condition1_sample_indices=condition1_sample_indices,
        condition2_sample_indices=condition2_sample_indices,
        sample_variance_use_estimators=True  # Use FunctionEstimator for each sample group
    )
    diff_expr_estimator.fit(X_condition1, y_condition1, X_condition2, y_condition2)

    # 2. Sample-specific with Nearest Neighbors
    print("\nTraining sample-specific model with nearest neighbors...")
    diff_expr_nn = DifferentialExpression(
        use_sample_specific_variance=True,
        condition1_sample_indices=condition1_sample_indices,
        condition2_sample_indices=condition2_sample_indices,
        sample_variance_use_estimators=False,  # Use nearest neighbor approach
        sample_variance_n_neighbors=15  # Number of neighbors to consider
    )
    diff_expr_nn.fit(X_condition1, y_condition1, X_condition2, y_condition2)

    # Create test data points - some from each sample type
    test_points = []
    for i in range(n_samples):
        # Generate points similar to each sample
        test_points.append(np.random.randn(20, n_features) + i * 0.2)
    X_test = np.vstack(test_points)

    # Make predictions with each model
    print("\nMaking predictions with both models...")
    pred_estimator = diff_expr_estimator.predict(X_test)
    pred_nn = diff_expr_nn.predict(X_test)

    # Compare fold changes
    print("\nMean fold change comparison:")
    print(f"Estimator approach: {pred_estimator['mean_log_fold_change'].mean():.4f}")
    print(f"Nearest neighbor approach: {pred_nn['mean_log_fold_change'].mean():.4f}")
    
    # NOTE: For weighted mean fold change computation, see the standalone example in sample_specific_variance.py
    # The DifferentialExpression class no longer auto-computes weighted fold change.
    # You must explicitly provide density predictions to the predict method or use the standalone function.

    # Compare z-scores
    print("\nZ-score statistics:")
    print(f"Estimator - mean: {pred_estimator['fold_change_zscores'].mean():.4f}, std: {pred_estimator['fold_change_zscores'].std():.4f}")
    print(f"Nearest neighbor - mean: {pred_nn['fold_change_zscores'].mean():.4f}, std: {pred_nn['fold_change_zscores'].std():.4f}")

    # Analyze behavior across different neighbor counts
    print("\nAnalyzing effect of different neighbor counts...")
    neighbor_counts = [5, 10, 15, 20, 25]
    mean_zscores = []
    std_zscores = []
    
    for n_neighbors in neighbor_counts:
        print(f"Training model with {n_neighbors} neighbors...")
        diff_expr_temp = DifferentialExpression(
            use_sample_specific_variance=True,
            condition1_sample_indices=condition1_sample_indices,
            condition2_sample_indices=condition2_sample_indices,
            sample_variance_use_estimators=False,
            sample_variance_n_neighbors=n_neighbors
        )
        diff_expr_temp.fit(X_condition1, y_condition1, X_condition2, y_condition2)
        
        pred_temp = diff_expr_temp.predict(X_test)
        mean_zscores.append(pred_temp['fold_change_zscores'].mean())
        std_zscores.append(pred_temp['fold_change_zscores'].std())
    
    # Optional: Create visualizations if matplotlib is available
    try:
        # 1. Compare estimator vs NN approach
        plt.figure(figsize=(10, 6))
        z_estimator = np.mean(pred_estimator['fold_change_zscores'], axis=0)
        z_nn = np.mean(pred_nn['fold_change_zscores'], axis=0)
        
        x = np.arange(len(z_estimator))
        width = 0.35
        
        plt.bar(x - width/2, z_estimator, width, label='Function Estimator')
        plt.bar(x + width/2, z_nn, width, label='Nearest Neighbor')
        
        plt.xlabel('Gene')
        plt.ylabel('Mean Z-score')
        plt.title('Comparison of Estimator vs. Nearest Neighbor Approaches')
        plt.xticks(x, [f'Gene {i+1}' for i in range(len(z_estimator))])
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('nn_to_estimator_comparison.png')
        
        # 2. Effect of neighbor count
        plt.figure(figsize=(10, 6))
        plt.plot(neighbor_counts, mean_zscores, 'o-', label='Mean Z-score')
        plt.plot(neighbor_counts, std_zscores, 's--', label='Std Z-score')
        plt.xlabel('Number of Neighbors')
        plt.ylabel('Z-score Statistics')
        plt.title('Effect of Neighbor Count on Z-score Statistics')
        plt.legend()
        plt.grid(True, alpha=0.7)
        plt.tight_layout()
        plt.savefig('neighbor_count_effect.png')
        
        print("\nVisualizations saved as 'nn_to_estimator_comparison.png' and 'neighbor_count_effect.png'")
    
    except ImportError:
        print("\nMatplotlib not available, skipping visualization.")

    print("\nDifferent approaches to sample-specific variance estimation:")
    print("1. Function Estimator approach: Trains separate function estimators for each sample")
    print("   - Pros: Better for well-separated samples with distinct error patterns")
    print("   - Cons: Requires more training time, less effective with few samples")
    print("\n2. Nearest Neighbor approach: Uses weighted nearest neighbors for variance estimation")
    print("   - Pros: Faster, works with fewer samples, more flexible")
    print("   - Cons: May smooth out important sample-specific variance patterns")
    print("\nChoose the approach that best fits your specific dataset characteristics.")

if __name__ == "__main__":
    main()