# Kompot

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
- Weighted log fold change analysis with density difference weighting
- Support for covariance matrices and optional landmarks
- JAX-accelerated computations
- Empirical variance estimation
- **Disk-backed storage for large datasets** with Dask support
- **Full scverse compatibility with direct AnnData integration**
- **Visualization tools** for differential expression, abundance results, and customizable embedding plots

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

For disk-backed storage with Dask support (recommended for large datasets):

```bash
pip install kompot[dask]
```

To install all optional dependencies:

```bash
pip install kompot[all]
```

### JAX Installation

Kompot depends on JAX for efficient computations. By default, the CPU version of JAX is used, which is recommended for most users as it provides good performance without memory constraints.

See [JAX GitHub](https://github.com/google/jax) for more installation details.

## Usage Example

See the [Tutorial Notebook](https://github.com/settylab/kompot/blob/main/examples/tutorial_notebook.ipynb) and [documentation](https://kompot.readthedocs.io/en/latest/index.html).

## License

GNU General Public License v3 (GPLv3)
