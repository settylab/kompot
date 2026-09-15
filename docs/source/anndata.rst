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
   :exclude-members: compute_differential_abundance

Differential Expression
-----------------------

.. automodule:: kompot.anndata.differential_expression
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: compute_differential_expression

Smooth Expression
-----------------

.. automodule:: kompot.anndata.smooth
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: compute_smoothed_expression

Resource Estimation
-------------------

Before running resource-intensive differential expression analyses, use
``dry_run=True`` to estimate memory and disk requirements, check for field
overwrites, and verify parameters.

**Key features:**

- **Memory and disk estimation**: Calculates expected resource usage for all intermediate arrays and final results
- **Null genes accounting**: Estimates resource inflation from null distribution genes when ``null_genes`` is given as an explicit number
- **Field overwrite detection**: Shows which fields will be overwritten, including their run_id and previous run details
- **Sample variance impact**: Estimates the per-gene covariance tensors, which are the dominant allocation whenever ``sample_col`` is set
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
       genes=top_genes,
       dry_run=True,
   )

The dry run output shows:

- **System Resources**: Available memory and disk space
- **Total Requirements**: Memory and disk needed with percentage of available
- **Memory Allocations**: Detailed breakdown of each array (precision matrices, smoothed expression, covariances)
- **Output Fields**: All fields that will be created, with ``[OVERWRITES run_id=X]`` markers for existing fields
- **Warnings**: Field overwrite information showing previous run timestamp, conditions, and parameters
- **Status**: Whether the analysis is feasible given available resources

.. seealso::

   :doc:`Planning Memory and Disk <resource_planning>` explains what drives
   the numbers, why ``sample_col`` must be a second pass over a restricted
   gene list, and where the estimate is known to be optimistic.

Utilities
---------

.. autoclass:: kompot.anndata.utils.RunInfo
   :members: __init__, get_summary, get_data, compare_with, to_settings, call_args
   :show-inheritance:

Cleanup Utilities
-----------------

.. autofunction:: kompot.cleanup

.. autofunction:: kompot.get_field_status

Representation Analysis
-----------------------

.. autofunction:: kompot.check_underrepresentation
