AnnData Integration
===================

The AnnData Integration module provides high-level convenience functions that work directly with AnnData objects. These functions handle the data flow, parameter management, and result storage automatically, making it easy to perform differential analysis with minimal setup.

**When to use AnnData Integration:**
- You want a simple, one-function-call approach to differential analysis
- You're working primarily with AnnData objects in your workflow
- You want automatic result storage and metadata tracking
- You prefer convenience over fine-grained control

**Key advantages:**
- Automatic parameter validation and data preparation
- Built-in result storage with run history tracking
- Seamless integration with plotting functions
- Handles complex data structures (layers, embeddings) automatically

Differential Abundance
----------------------

.. automodule:: kompot.anndata.differential_abundance
   :members:
   :undoc-members:
   :show-inheritance:

Differential Expression
-----------------------

.. automodule:: kompot.anndata.differential_expression
   :members:
   :undoc-members:
   :show-inheritance:

Resource Estimation
-------------------

Before running resource-intensive differential expression analyses, you can use the dry run utility to estimate memory and disk requirements, check for field overwrites, and verify parameters.

**Key features:**

- **Memory and disk estimation**: Calculates expected resource usage for all intermediate arrays and final results
- **Null genes accounting**: Correctly estimates resource inflation from null distribution genes (default 2000 additional genes)
- **Field overwrite detection**: Shows which fields will be overwritten, including their run_id and previous run details
- **Sample variance impact**: Estimates additional memory for sample-specific covariance tensors
- **Disk storage planning**: Estimates disk space needed when using ``store_arrays_on_disk=True``

.. code-block:: python

   import kompot as kp

   # Run a dry run before actual computation
   plan = kp.de(
       adata,
       groupby='age',
       condition1='Young',
       condition2='Old',
       sample_col='donor_id',
       dry_run=True,
   )

The dry run output shows:

- **System Resources**: Available memory and disk space
- **Total Requirements**: Memory and disk needed with percentage of available
- **Memory Allocations**: Detailed breakdown of each array (precision matrices, imputed expression, covariances)
- **Output Fields**: All fields that will be created, with ``[OVERWRITES run_id=X]`` markers for existing fields
- **Warnings**: Field overwrite information showing previous run timestamp, conditions, and parameters
- **Status**: Whether the analysis is feasible given available resources

Utilities
---------

.. autoclass:: kompot.anndata.utils.RunInfo
   :members: __init__, get_summary, get_data, compare_with
   :show-inheritance:

Cleanup Utilities
-----------------

.. autofunction:: kompot.cleanup

.. autofunction:: kompot.get_field_status

Representation Analysis
-----------------------

.. autofunction:: kompot.check_underrepresentation
