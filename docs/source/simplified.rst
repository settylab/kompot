AnnData Interface
=================

The AnnData interface provides high-level functions that work directly with
AnnData objects, handling data flow, parameter management, and result storage
automatically.

See the :doc:`Getting Started <notebooks/01_getting_started>` and
:doc:`Advanced Differential Expression <notebooks/02_differential_expression_detailed>`
tutorials for worked examples.


Differential Expression
-----------------------

.. code-block:: python

   import kompot

   # Minimal call
   kompot.de(adata, "condition", "Young", "Old")

   # With biological replicates
   kompot.de(adata, "condition", "Young", "Old", sample_col="donor_id")

   # Customise GP noise and FDR threshold
   kompot.de(
       adata, "condition", "Young", "Old",
       gp=kompot.GPSettings(sigma=0.5),
       fdr=kompot.FDRSettings(threshold=0.01),
   )

   # Enable empirical variance for heteroscedastic noise
   kompot.de(
       adata, "condition", "Young", "Old",
       gp=kompot.GPSettings(use_empirical_variance=True),
   )

   # Filter to specific cell types
   kompot.de(
       adata, "condition", "Young", "Old",
       filter=kompot.FilterSettings(
           groups="cell_type",
           cell_filter={"cell_type": ["T_cell", "B_cell"]},
       ),
   )

   # Store extra statistics and control output
   kompot.de(
       adata, "condition", "Young", "Old",
       storage=kompot.StorageSettings(store_additional_stats=True),
       output=kompot.OutputSettings(progress=False),
   )

.. autofunction:: kompot.de


Differential Abundance
----------------------

.. autofunction:: kompot.compute_differential_abundance


Expression Imputation
---------------------

.. autofunction:: kompot.impute_expression


Settings
--------

Each settings dataclass groups related parameters.  Any field left at its
default is equivalent to omitting it — you only override what you need.

GPSettings
^^^^^^^^^^

Controls the Gaussian Process model for expression fitting.

.. autoclass:: kompot.GPSettings
   :members:
   :undoc-members:

FDRSettings
^^^^^^^^^^^

Controls false-discovery-rate estimation.

.. autoclass:: kompot.FDRSettings
   :members:
   :undoc-members:

FilterSettings
^^^^^^^^^^^^^^

Controls cell filtering and group subsetting.

.. autoclass:: kompot.FilterSettings
   :members:
   :undoc-members:

StorageSettings
^^^^^^^^^^^^^^^

Controls where and how results are stored in the AnnData object.

.. autoclass:: kompot.StorageSettings
   :members:
   :undoc-members:

OutputSettings
^^^^^^^^^^^^^^

Controls return values and runtime behaviour.

.. autoclass:: kompot.OutputSettings
   :members:
   :undoc-members:


Kernels
-------

Matern52Linear
^^^^^^^^^^^^^^

The default GP kernel for expression modelling.  Combines a Matern-5/2 kernel
(local smoothness) with a Linear kernel (global trends), which improves
extrapolation in sparse regions of the cell-state space.

.. autoclass:: kompot.Matern52Linear
   :show-inheritance:

You can override the kernel via ``cov_func_curry``:

.. code-block:: python

   from mellon.cov import Matern52

   # Use plain Matern-5/2 instead of Matern52Linear
   kompot.de(
       adata, "condition", "Young", "Old",
       cov_func_curry=Matern52,
   )


Resource Estimation
-------------------

Before running resource-intensive differential expression analyses, you can use
the dry run utility to estimate memory and disk requirements.

.. code-block:: python

   plan = kompot.dry_run_differential_expression(
       adata,
       condition1="Young",
       condition2="Old",
       groupby="age",
       use_sample_variance=True,
       sample_column="donor_id",
       verbose=True,
   )
   print(plan.format_report(verbose=True))

.. autofunction:: kompot.dry_run_differential_expression


Run Tracking
------------

.. autoclass:: kompot.anndata.utils.RunInfo
   :members: __init__, get_summary, get_data, compare_with
   :show-inheritance:


Cleanup
-------

.. autofunction:: kompot.cleanup

.. autofunction:: kompot.get_field_status


Representation Analysis
-----------------------

.. autofunction:: kompot.check_underrepresentation
