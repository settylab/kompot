.. kompot documentation master file

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: User Guide:

   Installation <installation>
   Command-Line Interface <cli>

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: API Reference:

   AnnData Interface <simplified>
   Model Classes <differential>
   Plotting <plotting>
   Utilities <utils>

.. toctree::
   :hidden:
   :caption: Tutorials:
   :maxdepth: 2

   Getting Started <notebooks/01_getting_started.ipynb>
   Advanced Differential Expression <notebooks/02_differential_expression_detailed.ipynb>
   DE with Sample Variance <notebooks/03_sample_variance.ipynb>
   Differential Abundance <notebooks/04_differential_abundance.ipynb>
   Smoothing Expression <notebooks/05_smooth_expression.ipynb>

.. |doi| image:: https://zenodo.org/badge/944121568.svg
   :target: https://zenodo.org/badge/latestdoi/944121568
   :alt: DOI

.. |pypi| image:: https://img.shields.io/pypi/v/kompot.svg
   :target: https://pypi.org/project/kompot/
   :alt: PyPI

.. |tests| image:: https://github.com/settylab/kompot/actions/workflows/tests.yml/badge.svg
   :target: https://github.com/settylab/kompot/actions/workflows/tests.yml
   :alt: Tests

.. |codecov| image:: https://codecov.io/gh/settylab/kompot/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/settylab/kompot
   :alt: Coverage

.. |docs| image:: https://readthedocs.org/projects/kompot/badge/?version=latest
   :target: https://kompot.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

|doi| |pypi| |tests| |codecov| |docs|

.. image:: _static/images/kompot_logo.png
   :alt: Kompot Logo
   :align: center
   :width: 400px


Kompot
======

Kompot is a Python package for differential abundance and gene expression analysis using Gaussian Process models with JAX backend.

Overview
--------

Kompot implements methodologies from the Mellon package for computing differential abundance and gene expression, with a focus on using Mahalanobis distance as a measure of differential expression significance. It leverages JAX for efficient computations and provides a scikit-learn like API with ``.fit()`` and ``.predict()`` methods.

Key features:

- Computation of differential abundance between conditions
- Gene expression smoothing and uncertainty estimation
- Mahalanobis distance calculation for differential expression significance
- Weighted log fold change analysis with density difference weighting
- Support for covariance matrices and optional landmarks
- JAX-accelerated computations
- Empirical variance estimation
- **Resource estimation and dry run** for planning large analyses
- Disk-backed covariance storage for sample variance estimation
- Visualization tools (volcano plots, heatmaps, expression plots)
- **Full scverse compatibility with direct AnnData integration**
- **Command-line interface for pipeline integration** and batch processing

Use Cases
---------

Kompot is particularly useful for:

- Comparing cell type abundances across different samples
- Identifying differentially expressed genes between conditions
- Integrating multi-sample or multi-batch variability
- Handling large covariance tensors during sample variance estimation via disk-backed storage
- Creating visualizations of differential analysis results

Installation
------------

Install from PyPI:

.. code-block:: bash

   pip install kompot

Or via conda/mamba:

.. code-block:: bash

   conda install -c bioconda kompot

For optional dependencies and JAX GPU support, see the :doc:`installation guide <installation>`.

Quick Start
-----------

.. code-block:: python

   import kompot

   # Minimal call — uses sensible defaults
   kompot.de(adata, "condition", "Young", "Old")

   # Differential abundance
   kompot.da(adata, "condition", "Young", "Old")

   # Customize GP and FDR settings
   kompot.de(
       adata, "condition", "Young", "Old",
       sample_col="donor_id",
       gp=kompot.GPSettings(sigma=0.5, use_empirical_variance=True),
       fdr=kompot.FDRSettings(threshold=0.01),
   )

See :doc:`simplified` for the full API and all available settings.

**New to Kompot?** Start with the :doc:`Getting Started <notebooks/01_getting_started>` tutorial: differential expression, end to end, with rich visualizations.

**Ready for more?** Explore advanced topics:

- :doc:`Advanced Differential Expression <notebooks/02_differential_expression_detailed>` - Parameter tuning, multiple comparisons, run tracking, and resource planning
- :doc:`DE with Sample Variance <notebooks/03_sample_variance>` - Replicate-aware significance in multi-sample studies
- :doc:`Differential Abundance <notebooks/04_differential_abundance>` - Cell-state frequency changes, including its sample-variance treatment
- :doc:`Smoothing Expression <notebooks/05_smooth_expression>` - The expression function underneath DE: reading its two uncertainties, and fitting on some cells to predict on others

Command-Line Interface
^^^^^^^^^^^^^^^^^^^^^^^

**For pipeline integration and batch processing**, use the :doc:`Command-Line Interface <cli>`:

.. code-block:: bash

   # Compute diffusion maps (preprocessing)
   kompot dm input.h5ad -o input_dm.h5ad --pca-key X_pca

   # Run differential expression
   kompot de input_dm.h5ad -o results.h5ad \
     --groupby condition --condition1 control --condition2 treatment \
     --obsm-key DM_EigenVectors

See the :doc:`CLI documentation <cli>` for complete usage and examples.

.. toctree::
   :hidden:
   :caption: Links:

    Setty Lab <http://setty-lab.org>
    Github Repo <https://github.com/settylab/kompot>


Citation
========

If you use Kompot in your research, please cite:

   Otto, D. J., Arriaga-Gomez, E., Thieme, E., Yang, R., Lee, S. C., & Setty, M. (2025).
   Comparing phenotypic manifolds with Kompot: Detecting differential abundance and gene expression at single-cell resolution.
   *bioRxiv*. https://doi.org/10.1101/2025.06.03.657769

BibTeX:

.. code-block:: bibtex

   @article{Otto2025.06.03.657769,
       author = {Otto, Dominik J. and Arriaga-Gomez, Erica and Thieme, Elana and Yang, Ruijin and Lee, Stanley C. and Setty, Manu},
       title = {Comparing phenotypic manifolds with Kompot: Detecting differential abundance and gene expression at single-cell resolution},
       year = {2025},
       doi = {10.1101/2025.06.03.657769},
       publisher = {Cold Spring Harbor Laboratory},
       journal = {bioRxiv},
       URL = {https://www.biorxiv.org/content/10.1101/2025.06.03.657769}
   }


Index
=====

* :ref:`genindex`
