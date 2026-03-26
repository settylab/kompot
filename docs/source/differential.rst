Model Classes
=============

The core model classes perform statistical computations independently of
AnnData.  They follow a scikit-learn-style API with ``.fit()`` and
``.predict()`` methods, and are useful when you need direct control over the
analysis process or are working with custom data formats.

DifferentialAbundance
---------------------

.. autoclass:: kompot.DifferentialAbundance
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

DifferentialExpression
----------------------

.. autoclass:: kompot.DifferentialExpression
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

ExpressionModel
---------------

.. autoclass:: kompot.ExpressionModel
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:


SampleVariance
--------------

.. autoclass:: kompot.SampleVarianceEstimator
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
