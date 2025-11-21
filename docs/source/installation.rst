Installation
============

Basic Installation
------------------

Install from PyPI:

.. code-block:: bash

   pip install kompot

Or via conda/mamba:

.. code-block:: bash

   conda install -c bioconda kompot

Optional Dependencies
---------------------

Diffusion Maps
^^^^^^^^^^^^^^

For using the default diffusion map cell state representation with Palantir:

.. code-block:: bash

   pip install palantir

Plotting
^^^^^^^^

For additional plotting functionality with scanpy integration:

.. code-block:: bash

   pip install kompot[plot]

All Optional Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^

To install all optional dependencies:

.. code-block:: bash

   pip install kompot[all]

JAX Installation
----------------

Kompot depends on JAX for efficient computations. By default, the CPU version of JAX is installed, which is recommended for most users as it provides good performance without memory constraints.

For GPU Support
^^^^^^^^^^^^^^^

If you want to use GPU acceleration, you'll need to install the appropriate JAX version for your CUDA version. See the `JAX installation guide <https://github.com/google/jax#installation>`_ for detailed instructions.

Example for CUDA 12:

.. code-block:: bash

   pip install --upgrade "jax[cuda12]"

Example for CUDA 11:

.. code-block:: bash

   pip install --upgrade "jax[cuda11]"

.. note::

   GPU acceleration can significantly speed up computations for large datasets, but requires careful memory management. The CPU version is sufficient for most analyses and is recommended for getting started.

Development Installation
------------------------

To install kompot from source for development:

.. code-block:: bash

   git clone https://github.com/settylab/kompot.git
   cd kompot
   pip install -e ".[dev]"

This installs kompot in editable mode with development dependencies including testing and documentation tools.
