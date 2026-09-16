# Kompot

[![DOI](https://zenodo.org/badge/944121568.svg)](https://zenodo.org/badge/latestdoi/944121568)
[![PyPI](https://img.shields.io/pypi/v/kompot.svg)](https://pypi.org/project/kompot/)
[![Tests](https://github.com/settylab/kompot/actions/workflows/tests.yml/badge.svg)](https://github.com/settylab/kompot/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/settylab/kompot/branch/main/graph/badge.svg)](https://codecov.io/gh/settylab/kompot)
[![Documentation Status](https://readthedocs.org/projects/kompot/badge/?version=latest)](https://kompot.readthedocs.io/en/latest/?badge=latest)

![Kompot Logo](https://github.com/settylab/kompot/blob/main/docs/source/_static/images/kompot_logo.png?raw=true)

Kompot is a Python package for differential abundance and gene expression analysis using Gaussian Process models with JAX backend.

## Overview

Kompot implements methodologies from the Mellon package for computing differential abundance and gene expression, with a focus on using Mahalanobis distance as a measure of differential expression significance. It leverages JAX for efficient computations and provides a scikit-learn like API with `.fit()` and `.predict()` methods.

Key features:

- Computation of differential abundance between conditions
- Gene expression smoothing and uncertainty estimation
- Mahalanobis distance calculation for differential expression significance
- JAX-accelerated computations with optional GPU support
- Disk-backed covariance storage for sample variance estimation
- **Resource estimation and dry run** for planning large analyses
- **Full scverse compatibility** with direct AnnData integration
- **Visualization tools** for volcano plots, heatmaps, and embeddings
- **Command-line interface** for pipeline integration

## Installation

```bash
pip install kompot
```

Or via conda:

```bash
conda install -c bioconda kompot
```

See the [installation guide](https://kompot.readthedocs.io/en/latest/installation.html) for optional dependencies and JAX GPU support.

## Usage

### Python API

```python
import kompot
import anndata as ad

# Load data
adata = ad.read_h5ad("data.h5ad")

# Differential expression
kompot.de(adata, "condition", "control", "treatment")

# Differential abundance
kompot.da(adata, "condition", "control", "treatment")

# With advanced options
from kompot import GPSettings, FDRSettings

kompot.de(
    adata, "condition", "control", "treatment",
    gp=GPSettings(sigma=0.5),
    fdr=FDRSettings(threshold=0.05),
)
```

### Sample variance: run it as a second pass

Passing `sample_col` turns on **sample variance**, which replaces the single
shared posterior covariance with **one `(n_landmarks, n_landmarks)` covariance
matrix per gene, per condition**. Two costs follow, and they respond to
different levers:

- **Memory**, `2 x n_landmarks^2 x n_genes x 8` bytes — about **0.37 GiB per
  gene** at the default 5 000 landmarks. `StorageSettings(store_arrays_on_disk=True)`
  removes this almost entirely, and `n_landmarks` shrinks it quadratically.
- **Compute**, one Cholesky factorisation **per gene** instead of one in total.
  `store_arrays_on_disk` does not help here, but lowering `n_landmarks` does
  (~0.016 s/gene at 500 against ~2.1 s at 5 000, single-threaded), as does
  analysing fewer genes.

So run it in two passes, and price the second one first:

```python
# Pass 1 — all genes, no sample variance
kompot.de(adata, "condition", "Young", "Old")

mahal = "kompot_de_Young_to_Old_mahalanobis"
top_genes = adata.var.sort_values(mahal, ascending=False).head(1000).index

# Pass 2 — sample variance, restricted to the top genes
plan = kompot.de(
    adata, "condition", "Young", "Old",
    sample_col="donor_id",
    genes=top_genes,
    gp=kompot.GPSettings(n_landmarks=2000),   # cost is quadratic in this
    dry_run=True,                             # drop once the plan fits
)
```

`dry_run=True` returns a full resource plan (per-array memory, disk, output
fields, feasibility) without running anything; `kompot de --dry-run` does the
same from the CLI. Details, measured plans, and the remaining levers:
[Planning Memory and Disk](https://kompot.readthedocs.io/en/latest/resource_planning.html).

### Command-Line Interface

```bash
# Differential expression
kompot de input.h5ad -o output.h5ad \
  --groupby condition \
  --condition1 control \
  --condition2 treatment
```

## Documentation

- [Full Documentation](https://kompot.readthedocs.io)
- [Planning Memory and Disk](https://kompot.readthedocs.io/en/latest/resource_planning.html) — what sample variance costs and the two-pass workflow
- [Tutorial Notebooks](https://github.com/settylab/kompot/tree/main/examples)
  - [Getting Started](https://github.com/settylab/kompot/blob/main/examples/01_getting_started.ipynb) — differential expression, end to end
  - [Advanced Differential Expression](https://github.com/settylab/kompot/blob/main/examples/02_differential_expression_detailed.ipynb) — tuning, multiple comparisons, run tracking, resource planning
  - [DE with Sample Variance](https://github.com/settylab/kompot/blob/main/examples/03_sample_variance.ipynb) — replicate-aware significance
  - [Differential Abundance](https://github.com/settylab/kompot/blob/main/examples/04_differential_abundance.ipynb) — cell-state frequency changes, incl. sample variance
  - [Smoothing Expression](https://github.com/settylab/kompot/blob/main/examples/05_smooth_expression.ipynb) — the expression function underneath DE: its two uncertainties, and fit/predict across cells
- [CLI Guide](https://kompot.readthedocs.io/en/latest/cli.html)

## Citation

If you use Kompot in your research, please cite:

```bibtex
@article{Otto2025.06.03.657769,
    author = {Otto, Dominik J. and Arriaga-Gomez, Erica and Thieme, Elana and Yang, Ruijin and Lee, Stanley C. and Setty, Manu},
    title = {Comparing phenotypic manifolds with Kompot: Detecting differential abundance and gene expression at single-cell resolution},
    year = {2025},
    doi = {10.1101/2025.06.03.657769},
    publisher = {Cold Spring Harbor Laboratory},
    journal = {bioRxiv},
    URL = {https://www.biorxiv.org/content/10.1101/2025.06.03.657769}
}
```

## License

GNU General Public License v3 (GPLv3)
