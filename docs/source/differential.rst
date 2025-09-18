Differential Analysis
=====================

The Differential Analysis module provides the core analysis classes that perform statistical computations. These classes give you direct control over the analysis process and are designed for users who need flexibility and customization.

**When to use Differential Analysis:**
- You need fine-grained control over the analysis process
- You're working with custom data formats or preprocessing pipelines
- You want to integrate differential analysis into larger computational workflows
- You need access to intermediate results or custom statistical methods

**Key advantages:**
- Direct access to statistical computation engines
- Flexible input/output handling
- Customizable analysis parameters and methods
- Suitable for batch processing and automation

**Difference from AnnData Integration:**
While AnnData Integration provides convenient wrapper functions that handle everything automatically, Differential Analysis classes give you the building blocks to construct your own analysis pipeline. Use AnnData Integration for quick exploratory analysis, and Differential Analysis when you need programmatic control.

DifferentialAbundance
---------------------

.. autoclass:: kompot.DifferentialAbundance
   :members:
   :undoc-members:
   :show-inheritance:

DifferentialExpression
----------------------

.. autoclass:: kompot.DifferentialExpression
   :members:
   :undoc-members:
   :show-inheritance:


SampleVariance
--------------

.. autoclass:: kompot.SampleVarianceEstimator
   :members:
   :undoc-members:
   :show-inheritance: