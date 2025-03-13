.. kompot documentation master file

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Modules:

   Differential Analysis <differential>
   AnnData Integration <anndata>
   Plotting <plotting>
   Utilities <utils>

.. toctree::
   :hidden:
   :caption: Tutorials:
   :maxdepth: 2

   Tutorial Notebook <notebooks/tutorial_notebook.ipynb>
   Sample Variance Analysis <notebooks/sample_variance.ipynb>

Kompot
======

Kompot is a Python package for differential abundance and gene expression analysis using Gaussian Process models with JAX backend.

Overview
--------

Kompot implements methodologies from the Mellon package for computing differential abundance and gene expression, with a focus on using Mahalanobis distance as a measure of differential expression significance. It leverages JAX for efficient computations and provides a scikit-learn like API with `.fit()` and `.predict()` methods.

Key features:

- Computation of differential abundance between conditions
- Gene expression imputation and uncertainty estimation
- Mahalanobis distance calculation for differential expression significance
- Weighted log fold change analysis with density difference weighting
- Support for covariance matrices and optional landmarks
- JAX-accelerated computations
- Empirical variance estimation
- Visualization tools (volcano plots, heatmaps, expression plots)
- **Full scverse compatibility with direct AnnData integration**

.. toctree::
   :hidden:
   :caption: Links:

    Setty Lab <http://setty-lab.org>
    Github Repo <https://github.com/settylab/kompot>


Index
=====

* :ref:`genindex`
