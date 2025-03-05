# Kompot

Kompot is a Python package for differential abundance and gene expression analysis using Mahalanobis distance with JAX backend.

## Overview

Kompot implements methodologies from the Mellon package for computing differential abundance and gene expression, with a focus on using Mahalanobis distance as a measure of differential expression significance. It leverages JAX for efficient computations and provides a scikit-learn like API with `.fit()` and `.predict()` methods.

Key features:

- Computation of differential abundance between conditions
- Gene expression imputation and uncertainty estimation
- Mahalanobis distance calculation for differential expression significance
- Weighted log fold change analysis with density difference weighting
- Support for covariance matrices and optional landmarks
- JAX-accelerated computations
- Empirical variance estimation

## Installation

```bash
pip install kompot
```

## Usage Example

```python
import numpy as np
import kompot

# Example for differential abundance
X_condition1 = np.random.randn(500, 10)  # Cell states for condition 1
X_condition2 = np.random.randn(500, 10)  # Cell states for condition 2

diff_abundance = kompot.DifferentialAbundance()
diff_abundance.fit(X_condition1, X_condition2)

# Access differential abundance results
log_fold_change = diff_abundance.log_fold_change
log_fold_change_zscore = diff_abundance.log_fold_change_zscore
log_fold_change_direction = diff_abundance.log_fold_change_direction

# Example for differential expression
y_condition1 = np.random.randn(500, 50)  # Gene expression for condition 1
y_condition2 = np.random.randn(500, 50)  # Gene expression for condition 2

diff_expression = kompot.DifferentialExpression(n_landmarks=200)
diff_expression.fit(
    X_condition1, y_condition1, 
    X_condition2, y_condition2
)

# Access differential expression results
fold_change = diff_expression.fold_change
fold_change_zscores = diff_expression.fold_change_zscores
mahalanobis_distances = diff_expression.mahalanobis_distances
weighted_mean_log_fold_change = diff_expression.weighted_mean_log_fold_change
```

For a more detailed example, see the [basic example script](examples/basic_example.py) in the examples directory.

## Features

### DifferentialAbundance

Computes differential abundance between two conditions using density estimation:

- **Log density values** for each condition
- **Log fold change** between conditions
- **Z-scores** and **p-values** for each fold change
- **Direction of change** ('up', 'down', or 'neutral')

### DifferentialExpression

Computes differential expression between two conditions:

- **Imputed gene expression** for each condition
- **Fold change** between conditions
- **Fold change z-scores** incorporating uncertainty
- **Mahalanobis distances** for statistical significance
- **Weighted mean log fold change** using density differences
- Support for **empirical variance estimation**

## License

GNU General Public License v3 (GPLv3)