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


Expression Smoothing
--------------------

.. autofunction:: kompot.smooth_expression


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

When using pre-fitted predictors with FDR estimation, null features must be
included in the data *before* fitting the predictors — they need to go
through the same GP smoothing pipeline as the real features.  Pass their
column indices via ``FDRSettings(null_genes=[...])``.  The null features
calibrate the FDR null distribution and are then stripped from all output:
the result table and ``adata`` layers contain only the real genes.

.. code-block:: python

   # Assume predictors were trained on 100 real + 200 null features
   null_indices = list(range(100, 300))

   kompot.de(
       adata, "condition", "WT", "KO",
       model=kompot.ModelSettings(
           function_predictor1=predictor_wt,
           function_predictor2=predictor_ko,
       ),
       fdr=kompot.FDRSettings(null_genes=null_indices),
   )


Resource Estimation
-------------------

Before running resource-intensive differential expression analyses, pass
``dry_run=True`` to estimate memory and disk requirements without running
the actual computation.

.. code-block:: python

   plan = kompot.de(
       adata,
       groupby="age",
       condition1="Young",
       condition2="Old",
       sample_col="donor_id",
       dry_run=True,
   )


Run Tracking
------------

.. autoclass:: kompot.anndata.utils.RunInfo
   :members: __init__, get_summary, get_data, compare_with, to_settings, call_args
   :show-inheritance:

Reproducing and Editing Runs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Every run stores its parameters as nested Settings objects.  Use
:meth:`~kompot.anndata.utils.RunInfo.call_args` to get a kwargs dict
that reproduces the run — then edit it before re-running:

.. code-block:: python

   run = kompot.RunInfo(adata, run_id=0, analysis_type="de")

   # Reproduce exactly
   kwargs = run.call_args()
   kompot.de(adata, **kwargs)

   # Or tweak parameters first
   kwargs = run.call_args()
   kwargs["fdr"].threshold = 0.01          # tighten FDR
   kwargs["condition2"] = "Mid"            # different comparison
   kwargs["gp"].n_landmarks = 3000         # fewer landmarks
   kompot.de(adata, **kwargs)

You can also inspect the Settings objects directly:

.. code-block:: python

   settings = run.to_settings()
   print(settings["gp"])       # GPSettings(sigma=1.0, ls_factor=10.0, ...)
   print(settings["fdr"])      # FDRSettings(null_genes=2000, threshold=0.05, ...)


Cleanup
-------

.. autofunction:: kompot.cleanup

.. autofunction:: kompot.get_field_status


Representation Analysis
-----------------------

.. autofunction:: kompot.check_underrepresentation
