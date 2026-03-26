"""Check refactored DifferentialExpression output against reference outputs.

Run after refactoring to verify consistency:
    uv run --extra dev python tests/check_consistency.py
"""

import os
import sys
import numpy as np
from kompot.differential import DifferentialExpression


def make_synthetic_data(seed=42):
    """Create synthetic data with fixed seed (must match generate_reference_outputs.py)."""
    rng = np.random.RandomState(seed)
    n_cells = 100
    n_genes = 10
    n_features = 5

    X1 = rng.randn(n_cells, n_features)
    X2 = rng.randn(n_cells, n_features) + 0.3

    y1 = rng.randn(n_cells, n_genes) * 0.2
    y2 = rng.randn(n_cells, n_genes) * 0.2
    y2[:, :5] += 1.0
    y1[:, 7:] += rng.randn(n_cells, 3) * 4.0
    y2[:, 7:] += rng.randn(n_cells, 3) * 4.0

    X_new = X1
    return X1, y1, X2, y2, X_new


KEYS = [
    "condition1_imputed",
    "condition2_imputed",
    "condition1_std",
    "condition2_std",
    "fold_change",
    "fold_change_zscores",
    "mean_log_fold_change",
    "mahalanobis_distances",
]


def check_config(name, prefix, results, ref_dir, atol=1e-6):
    """Compare results against reference for one configuration."""
    all_pass = True
    for key in KEYS:
        ref_path = os.path.join(ref_dir, f"{prefix}_{key}.npy")
        if not os.path.exists(ref_path):
            print(f"  [{name}] SKIP {key} — reference file not found")
            continue

        expected = np.load(ref_path)
        actual = np.asarray(results.get(key))
        if actual is None:
            print(f"  [{name}] FAIL {key} — missing from results")
            all_pass = False
            continue

        try:
            np.testing.assert_allclose(actual, expected, atol=atol, rtol=1e-5)
            print(f"  [{name}] PASS {key}")
        except AssertionError:
            max_diff = np.max(np.abs(actual - expected))
            print(f"  [{name}] FAIL {key} — max diff: {max_diff:.2e}")
            all_pass = False

    return all_pass


def main():
    ref_dir = "/tmp/kompot_reference"
    if not os.path.isdir(ref_dir):
        print(f"ERROR: Reference directory {ref_dir} not found. Run generate_reference_outputs.py first.")
        sys.exit(1)

    X1, y1, X2, y2, X_new = make_synthetic_data(seed=42)
    fit_kwargs = dict(sigma=1.0, ls_factor=10.0)
    de_kwargs = dict(n_landmarks=50, batch_size=0, random_state=42)

    all_pass = True

    # Basic
    print("Checking basic configuration...")
    de_basic = DifferentialExpression(
        use_empirical_variance=False,
        use_sample_variance=False,
        **de_kwargs,
    )
    de_basic.fit(X1, y1, X2, y2, **fit_kwargs)
    res_basic = de_basic.predict(X_new, compute_mahalanobis=True, progress=False)
    all_pass &= check_config("basic", "basic", res_basic, ref_dir)

    # Empirical variance
    print("Checking empirical variance configuration...")
    de_empvar = DifferentialExpression(
        use_empirical_variance=True,
        use_sample_variance=False,
        **de_kwargs,
    )
    de_empvar.fit(X1, y1, X2, y2, **fit_kwargs)
    res_empvar = de_empvar.predict(X_new, compute_mahalanobis=True, progress=False)
    all_pass &= check_config("empvar", "empvar", res_empvar, ref_dir)

    if all_pass:
        print("\nAll checks PASSED.")
    else:
        print("\nSome checks FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
