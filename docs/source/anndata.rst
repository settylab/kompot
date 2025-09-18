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

Utilities
---------

.. autoclass:: kompot.anndata.utils.RunInfo
   :members: __init__, get_summary, get_data, compare_with
   :show-inheritance:

Representation Analysis
-----------------------

.. autofunction:: kompot.check_underrepresentation
