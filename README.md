# Kompot

Kompot is a Python package for differential abundance and gene expression analysis using Gaussian Process models with JAX backend.

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
- **Full scverse compatibility with direct AnnData integration**
- Uses diffusion maps (from Palantir) as the default cell state representation
- **Visualization tools** for differential expression and abundance results

## Installation

```bash
pip install kompot
```

For using the default diffusion map cell state representation:

```bash
pip install palantir
```

For additional plotting functionality with scanpy integration:

```bash
pip install kompot[plot]
```

To install all optional dependencies:

```bash
pip install kompot[all]
```

See [Palantir GitHub](https://github.com/dpeerlab/Palantir) for more installation details.

## Usage Example

### Original NumPy-based API

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

# Generate HTML report
kompot.generate_report(
    diff_expression, 
    output_dir="report", 
    condition1_name="Control", 
    condition2_name="Treatment"
)
```

### scverse-compatible AnnData API

```python
import kompot
import anndata

# Load or create AnnData object
adata = anndata.read_h5ad("my_data.h5ad")

# Generate diffusion maps using Palantir (if not already present)
# This automatically adds DM_EigenVectors to adata.obsm
import palantir
palantir.utils.run_diffusion_maps(adata)

# Run full differential analysis workflow
adata = kompot.run_differential_analysis(
    adata,
    groupby="condition",              # Column in adata.obs with condition labels  
    condition1="control",             # First condition label
    condition2="treatment",           # Second condition label
    obsm_key="DM_EigenVectors",       # Cell states in adata.obsm
    layer="counts",                   # Optional: use specific layer (otherwise uses adata.X)
    n_landmarks=200,                  # Number of landmarks for approximation
    generate_html_report=True         # Generate interactive HTML report
)

# Results are stored in the AnnData object:
# - Gene scores: adata.var['kompot_de_mahalanobis']
# - Cell scores: adata.obs['kompot_da_log_fold_change']
# - Imputed expression: adata.layers['kompot_de_condition1_imputed']
# - Log fold changes: adata.layers['kompot_de_fold_change']
# - Full models: adata.uns['kompot_da']['model'] and adata.uns['kompot_de']['model']

# Alternatively, run specific analyses:
# Just differential abundance
kompot.compute_differential_abundance(
    adata,
    groupby="condition",
    condition1="control",
    condition2="treatment",
    obsm_key="DM_EigenVectors"
)

# Just differential expression
kompot.compute_differential_expression(
    adata,
    groupby="condition", 
    condition1="control",
    condition2="treatment",
    obsm_key="DM_EigenVectors",
    layer="counts"
)
```

For a more detailed example, see the [basic example script](examples/basic_example.py) in the examples directory.

### Visualization Examples

```python
import kompot as kp
import scanpy as sc
import matplotlib.pyplot as plt

# Assuming adata has differential expression results
# Create volcano plot for differential expression
kp.plot.volcano_de(
    adata,
    lfc_key="kompot_de_mean_lfc_condition2_vs_condition1",
    score_key="kompot_de_mahalanobis",
    condition1="condition1",
    condition2="condition2",
    n_top_genes=15
)

# Create volcano plot for differential abundance
kp.plot.volcano_da(
    adata,
    lfc_key="kompot_da_log_fold_change",
    pval_key="kompot_da_pvalue",
    lfc_threshold=1.0,
    pval_threshold=0.05,
    color="louvain"  # Color cells by cluster
)

# Create heatmap of top differentially expressed genes
kp.plot.heatmap(
    adata,
    n_top_genes=20,
    groupby="condition",
    standard_scale="var",
    cmap="viridis"
)
```

For more visualization examples, see the [volcano plot example](examples/volcano_plot_example.py), [volcano DA example](examples/volcano_da_example.py), and [heatmap example](examples/heatmap_example.py) scripts.

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

### AnnData Integration

- Direct support for operating on AnnData objects
- Automatic extraction of cell states and gene expression
- Storage of results in standard AnnData locations
- HTML report generation from AnnData objects
- Full compatibility with the scverse ecosystem

### Visualization Tools

- **Volcano plots** for differential expression (`volcano_de`) with customizable appearance
- **Volcano plots** for differential abundance (`volcano_da`) with optional scanpy-based coloring
- **Heatmaps** for visualizing gene expression patterns across conditions
- All plot functions follow a scanpy-like API with AnnData objects as input

## Documentation

Documentation is available in the `docs/` directory. To build the documentation locally:

```bash
# Install doc dependencies
pip install -e ".[docs]"

# Build docs
cd docs
make html

# View docs
open build/html/index.html  # macOS
# or: xdg-open build/html/index.html  # Linux
# or: start build/html/index.html  # Windows
```

## License

GNU General Public License v3 (GPLv3)