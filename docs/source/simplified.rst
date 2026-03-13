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

   # Customise GP noise and FDR threshold
   kompot.de(
       adata, "condition", "Young", "Old",
       gp=kompot.GPSettings(sigma=0.5),
       fdr=kompot.FDRSettings(threshold=0.01),
   )

   # Filter to specific cell types
   kompot.de(
       adata, "condition", "Young", "Old",
       filter=kompot.FilterSettings(
           groups="cell_type",
           cell_filter={"cell_type": ["T_cell", "B_cell"]},
       ),
   )

   # Sample variance for biological replicates (limit to top genes)
   kompot.de(
       adata, "condition", "Young", "Old",
       sample_col="donor_id",
       genes=top_genes,  # e.g. top 200 from a previous run
       fdr=kompot.FDRSettings(null_genes=0),
   )

.. autofunction:: kompot.de


Differential Abundance
----------------------

.. code-block:: python

   # Minimal call
   kompot.da(adata, "condition", "Young", "Old")

   # Adjust significance thresholds
   kompot.da(
       adata, "condition", "Young", "Old",
       threshold=kompot.DAThresholdSettings(ptp_threshold=0.01),
   )

.. autofunction:: kompot.da


Expression Imputation
---------------------

.. autofunction:: kompot.impute_expression


Settings
--------

Each settings dataclass groups related parameters.  Any field left at its
default is equivalent to omitting it — you only override what you need.

GPSettings
^^^^^^^^^^

Controls the Gaussian Process model.

.. autoclass:: kompot.GPSettings
   :members:
   :undoc-members:

FDRSettings
^^^^^^^^^^^

Controls false-discovery-rate estimation (DE only).

.. autoclass:: kompot.FDRSettings
   :members:
   :undoc-members:

DAThresholdSettings
^^^^^^^^^^^^^^^^^^^

Significance thresholds for differential abundance.

.. autoclass:: kompot.DAThresholdSettings
   :members:
   :undoc-members:

FilterSettings
^^^^^^^^^^^^^^

Controls cell filtering and group subsetting (DE only).

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

ModelSettings
^^^^^^^^^^^^^

Inject pre-fitted models or predictors to skip internal fitting.

.. autoclass:: kompot.ModelSettings
   :members:
   :undoc-members:


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
