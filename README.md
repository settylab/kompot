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
- Gene expression imputation and uncertainty estimation
- Mahalanobis distance calculation for differential expression significance
- JAX-accelerated computations with optional GPU support
- **Disk-backed storage for large datasets** with dask support
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
kompot.compute_differential_expression(
    adata,
    groupby="condition",
    condition1="control",
    condition2="treatment",
    obsm_key="X_pca"
)
```

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
- [Tutorial Notebooks](https://github.com/settylab/kompot/tree/main/examples)
  - [Getting Started](https://github.com/settylab/kompot/blob/main/examples/01_getting_started.ipynb)
  - [Advanced Differential Expression](https://github.com/settylab/kompot/blob/main/examples/02_differential_expression_detailed.ipynb)
  - [Sample Variance Analysis](https://github.com/settylab/kompot/blob/main/examples/03_sample_variance.ipynb)
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
