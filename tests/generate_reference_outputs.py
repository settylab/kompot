"""Generate reference outputs from current DifferentialExpression code.

Run before refactoring to capture expected outputs:
    uv run --extra dev python tests/generate_reference_outputs.py
"""

import os
import numpy as np
from kompot.differential import DifferentialExpression


def make_synthetic_data(seed=42):
    """Create synthetic data with fixed seed for reproducibility."""
    rng = np.random.RandomState(seed)
    n_cells = 100
    n_genes = 10
    n_features = 5

    X1 = rng.randn(n_cells, n_features)
    X2 = rng.randn(n_cells, n_features) + 0.3

    y1 = rng.randn(n_cells, n_genes) * 0.2
    y2 = rng.randn(n_cells, n_genes) * 0.2
    # Mean shift on first 5 genes
    y2[:, :5] += 1.0
    # High noise on last 3 genes
    y1[:, 7:] += rng.randn(n_cells, 3) * 4.0
    y2[:, 7:] += rng.randn(n_cells, 3) * 4.0

    # Prediction points (use X1 for simplicity)
    X_new = X1

    return X1, y1, X2, y2, X_new


def save_results(results, output_dir, prefix):
    """Save all result arrays as .npy files."""
    keys = [
        "condition1_smoothed",
        "condition2_smoothed",
        "condition1_std",
        "condition2_std",
        "fold_change",
        "fold_change_zscores",
        "mean_log_fold_change",
        "mahalanobis_distances",
    ]
    for key in keys:
        if key in results:
            path = os.path.join(output_dir, f"{prefix}_{key}.npy")
            np.save(path, np.asarray(results[key]))
            print(f"  Saved {path} shape={np.asarray(results[key]).shape}")


def main():
    output_dir = "/tmp/kompot_reference"
    os.makedirs(output_dir, exist_ok=True)

    X1, y1, X2, y2, X_new = make_synthetic_data(seed=42)

    fit_kwargs = dict(sigma=1.0, ls_factor=10.0)
    de_kwargs = dict(n_landmarks=50, batch_size=0, random_state=42)

    # --- Configuration A: Basic (no empirical variance, no sample variance) ---
    print("Fitting basic configuration...")
    de_basic = DifferentialExpression(
        use_empirical_variance=False,
        use_sample_variance=False,
        **de_kwargs,
    )
    de_basic.fit(X1, y1, X2, y2, **fit_kwargs)
    res_basic = de_basic.predict(X_new, compute_mahalanobis=True, progress=False)
    save_results(res_basic, output_dir, "basic")

    # --- Configuration B: With empirical variance ---
    print("Fitting empirical variance configuration...")
    de_empvar = DifferentialExpression(
        use_empirical_variance=True,
        use_sample_variance=False,
        **de_kwargs,
    )
    de_empvar.fit(X1, y1, X2, y2, **fit_kwargs)
    res_empvar = de_empvar.predict(X_new, compute_mahalanobis=True, progress=False)
    save_results(res_empvar, output_dir, "empvar")

    print(f"\nReference outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
