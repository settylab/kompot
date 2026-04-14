Command-Line Interface (CLI)
============================

The kompot CLI provides command-line access to differential expression (DE) and differential abundance (DA) analysis for pipeline integration and workflow automation.

Installation
------------

The CLI is installed automatically with kompot:

.. code-block:: bash

   pip install kompot
   # or
   mamba install -c bioconda kompot

Verify installation:

.. code-block:: bash

   kompot --version
   kompot --help

Overview
--------

The CLI provides four main commands:

- ``kompot dm`` - Compute diffusion maps (preprocessing with Palantir)
- ``kompot de`` - Differential expression analysis
- ``kompot da`` - Differential abundance analysis
- ``kompot smooth`` - Smooth gene expression for a single condition

All commands support:

- Direct CLI arguments for common parameters
- YAML/JSON config files for advanced parameters
- Reading/writing ``.h5ad`` and ``.zarr`` AnnData formats

Quick Start
-----------

.. code-block:: bash

   # 1. Compute diffusion maps (preprocessing)
   kompot dm input.h5ad -o input_with_dm.h5ad \
     --pca-key X_pca \
     --n-components 10

   # 2. Run differential expression
   kompot de input_with_dm.h5ad -o de_results.h5ad \
     --groupby condition \
     --condition1 control \
     --condition2 treatment

   # 3. Run differential abundance
   kompot da input_with_dm.h5ad -o da_results.h5ad \
     --groupby condition \
     --condition1 control \
     --condition2 treatment

   # 4. Smooth gene expression for a single condition
   kompot smooth input_with_dm.h5ad -o smoothed.h5ad \
     --groupby condition \
     --condition treatment

Using Config Files
^^^^^^^^^^^^^^^^^^

For complex analyses with many parameters, use config files:

.. code-block:: bash

   # Get template (copy from installed package)
   python -c "from pathlib import Path; import shutil; \
   import kompot; \
   src = Path(kompot.__file__).parent / 'cli' / 'templates' / 'de_config_minimal.yaml'; \
   shutil.copy(src, 'my_de_config.yaml')"

   # Edit config file
   nano my_de_config.yaml

   # Run analysis
   kompot de input.h5ad -o output.h5ad -c my_de_config.yaml

CLI arguments override config file values:

.. code-block:: bash

   kompot de input.h5ad -o output.h5ad \
     -c my_config.yaml \
     --batch-size 50  # Overrides batch_size in config

Diffusion Maps Command
----------------------

The ``dm`` command computes diffusion maps using Palantir, which provides a continuous representation of cell states needed for differential analysis.

Basic Usage
^^^^^^^^^^^

.. code-block:: bash

   kompot dm INPUT -o OUTPUT [OPTIONS]

Prerequisites
^^^^^^^^^^^^^

- Requires Palantir: ``pip install palantir`` or ``pip install kompot[recommended]``
- Input AnnData must contain PCA coordinates in ``adata.obsm``

Common Options
^^^^^^^^^^^^^^

.. code-block:: text

   --pca-key KEY           # PCA coordinates in adata.obsm (default: X_pca)
   --n-components N        # Number of diffusion components (default: 10)
   --knn N                 # Number of nearest neighbors (default: 30)
   --alpha FLOAT           # Diffusion alpha parameter (default: 0)

Output
^^^^^^

Results are stored in:

- ``adata.obsm['DM_EigenVectors']`` - Diffusion map coordinates (n_cells × n_components)
- ``adata.uns['DM_EigenValues']`` - Eigenvalues of diffusion operator

Example: Complete Preprocessing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Starting with raw AnnData (assuming PCA already computed)
   kompot dm bone_marrow.h5ad -o bone_marrow_dm.h5ad \
     --pca-key X_pca \
     --n-components 10 \
     --knn 30

   # Then run differential analysis
   kompot de bone_marrow_dm.h5ad -o results.h5ad \
     --groupby Age \
     --condition1 Young \
     --condition2 Old \
     --obsm-key DM_EigenVectors

Why Diffusion Maps?
^^^^^^^^^^^^^^^^^^^^

Diffusion maps capture continuous cell state transitions better than PCA alone:

- Preserves the geometry of differentiation trajectories
- Reduces noise while maintaining biological structure
- Euclidean distance in this representation better represents biological similarity
- Distance in cell-state representation is used by kompot's covariance kernel

See the `Palantir documentation <https://github.com/dpeerlab/Palantir>`_ for details.

Differential Expression Command
--------------------------------

Basic Usage
^^^^^^^^^^^

.. code-block:: bash

   kompot de INPUT -o OUTPUT [OPTIONS]
   kompot de INPUT -t TABLE_OUTPUT [OPTIONS]

At least one output must be specified: ``-o/--output`` for full AnnData or ``-t/--table-output`` for CSV/TSV table.

Required Parameters
^^^^^^^^^^^^^^^^^^^

Either via CLI or config file:

- ``--groupby COLUMN`` - Column in ``adata.obs`` with condition labels
- ``--condition1 LABEL`` - Reference condition label
- ``--condition2 LABEL`` - Comparison condition label

Output Options
^^^^^^^^^^^^^^

.. code-block:: text

   -o, --output FILE         # Output AnnData file (.h5ad or .zarr)
   -t, --table-output FILE   # Output DE results as table (.csv or .tsv)

The ``--table-output`` option exports only the kompot-produced columns from ``adata.var`` (gene-level statistics like mahalanobis distance, log fold change, FDR, etc.). This is useful for downstream analysis or integration with other tools.

Common Options
^^^^^^^^^^^^^^

.. code-block:: text

   --obsm-key KEY            # Cell state representation (default: DM_EigenVectors)
   --layer LAYER             # Expression data layer (default: None, use X)
   --result-key KEY          # Storage key (default: kompot_de)
   --n-landmarks N           # Number of landmarks (default: 5000)
   --sample-col COLUMN       # Sample ID column for replicates
   --batch-size N            # Cells per batch (default: 100)
   --fdr-threshold FLOAT     # FDR threshold (default: 0.05)
   --null-genes N            # Null genes for FDR (default: 2000)

Boolean Flags
^^^^^^^^^^^^^

.. code-block:: text

   --no-progress             # Disable progress bars
   --store-landmarks           # Store landmarks for reuse
   --store-additional-stats    # Store extra statistics
   --use-empirical-variance   # Estimate per-gene noise from GP residuals
   --overwrite                 # Overwrite without warning

Compute Options
^^^^^^^^^^^^^^^

.. code-block:: text

   --use-gpu                 # Use GPU acceleration (requires CUDA-enabled JAX)
   --threads N               # Number of threads for JAX/NumPy/Dask (default: all cores)

Advanced Options
^^^^^^^^^^^^^^^^

For advanced parameters (gene filtering, cell filtering, GP kernel parameters, memory management, etc.), see the configuration file templates:

- ``kompot/cli/templates/de_config_template.yaml`` - Complete template with all parameters
- ``kompot/cli/templates/de_config_minimal.yaml`` - Minimal template with common parameters

Example: Complete Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   kompot de bone_marrow.h5ad -o results.h5ad \
     --groupby Age \
     --condition1 Young \
     --condition2 Old \
     --obsm-key DM_EigenVectors \
     --layer logged_counts \
     --sample-col Sample \
     --n-landmarks 5000 \
     --batch-size 100 \
     --fdr-threshold 0.05 \
     --null-genes 2000 \
     --store-additional-stats

Differential Abundance Command
-------------------------------

Basic Usage
^^^^^^^^^^^

.. code-block:: bash

   kompot da INPUT -o OUTPUT [OPTIONS]
   kompot da INPUT -t TABLE_OUTPUT [OPTIONS]

At least one output must be specified: ``-o/--output`` for full AnnData or ``-t/--table-output`` for CSV/TSV table.

Required Parameters
^^^^^^^^^^^^^^^^^^^

Either via CLI or config file:

- ``--groupby COLUMN`` - Column in ``adata.obs`` with condition labels
- ``--condition1 LABEL`` - Reference condition label
- ``--condition2 LABEL`` - Comparison condition label

Output Options
^^^^^^^^^^^^^^

.. code-block:: text

   -o, --output FILE         # Output AnnData file (.h5ad or .zarr)
   -t, --table-output FILE   # Output DA results as table (.csv or .tsv)

The ``--table-output`` option exports only the kompot-produced columns from ``adata.obs`` (cell-level statistics like log fold change, z-scores, PTP values, etc.). This is useful for downstream analysis or integration with other tools.

Common Options
^^^^^^^^^^^^^^

.. code-block:: text

   --obsm-key KEY                    # Cell state representation (default: X_pca)
   --result-key KEY                  # Storage key (default: kompot_da)
   --n-landmarks N                   # Number of landmarks (default: None, all points)
   --sample-col COLUMN               # Sample ID column for replicates
   --batch-size N                    # Cells per batch (default: None)
   --log-fold-change-threshold FLOAT # LFC threshold (default: 1.0)
   --ptp-threshold FLOAT             # PTP threshold (default: 0.05)
   --ls-factor FLOAT                 # Length scale factor (default: 10.0)

Boolean Flags
^^^^^^^^^^^^^

.. code-block:: text

   --store-landmarks         # Store landmarks for reuse
   --overwrite               # Overwrite without warning
   --no-progress             # Disable progress bars

Compute Options
^^^^^^^^^^^^^^^

.. code-block:: text

   --use-gpu                 # Use GPU acceleration (requires CUDA-enabled JAX)
   --threads N               # Number of threads for JAX/NumPy/Dask (default: all cores)

Example: Complete Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   kompot da bone_marrow.h5ad -o results.h5ad \
     --groupby Age \
     --condition1 Young \
     --condition2 Old \
     --obsm-key DM_EigenVectors \
     --sample-col Sample \
     --n-landmarks 3000 \
     --log-fold-change-threshold 1.0 \
     --ptp-threshold 0.05

Smooth Command
--------------

The ``smooth`` command smooths gene expression for a single condition using GP regression. This is useful for denoising expression data or preparing smoothed expression for downstream analysis.

Basic Usage
^^^^^^^^^^^

.. code-block:: bash

   kompot smooth INPUT -o OUTPUT [OPTIONS]
   kompot smooth INPUT -t TABLE_OUTPUT [OPTIONS]

At least one output must be specified: ``-o/--output`` for full AnnData or ``-t/--table-output`` for CSV/TSV summary table (mean smoothed values and standard deviations per gene).

Common Options
^^^^^^^^^^^^^^

.. code-block:: text

   --groupby COLUMN          # Column in adata.obs with condition labels
   --condition LABEL         # Which condition to smooth (requires --groupby)
   --obsm-key KEY            # Cell state representation (default: DM_EigenVectors)
   --layer LAYER             # Expression data layer (default: None, use X)
   --result-key KEY          # Storage key (default: kompot_smooth)
   --n-landmarks N           # Number of landmarks (default: 5000)
   --sample-col COLUMN       # Sample ID column for sample variance estimation
   --batch-size N            # Cells per batch (default: 500)
   --genes GENE [GENE ...]   # Subset of genes to smooth (default: all)

GP Options
^^^^^^^^^^

.. code-block:: text

   --sigma FLOAT             # Noise level for the GP (default: 1.0)
   --ls-factor FLOAT         # Length scale factor (default: 10.0)
   --eps FLOAT               # Numerical stability constant (default: 1e-8)
   --random-state INT        # Random seed for landmark selection

Boolean Flags
^^^^^^^^^^^^^

.. code-block:: text

   --use-empirical-variance  # Estimate per-gene heteroscedastic noise from GP residuals
   --overwrite               # Overwrite existing results without warning
   --no-progress             # Disable progress bars
   --use-gpu                 # Use GPU acceleration (requires CUDA-enabled JAX)

Example: Smooth Treatment Condition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   kompot smooth bone_marrow.h5ad -o smoothed.h5ad \
     --groupby Age \
     --condition Old \
     --obsm-key DM_EigenVectors \
     --layer logged_counts \
     --n-landmarks 5000

Example: Gene Summary Table
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   kompot smooth bone_marrow.h5ad \
     -t gene_summary.csv \
     --groupby Age \
     --condition Old \
     --genes CD34 CD38 THY1

Configuration Files
-------------------

Mapping to the Python API
^^^^^^^^^^^^^^^^^^^^^^^^^

The Python API groups parameters into Settings dataclasses
(``GPSettings``, ``FDRSettings``, etc.).  CLI config files use **flat
keys** — the CLI maps them to the correct Settings automatically.
The table below shows how config keys correspond to Python Settings:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Config key
     - Python equivalent
     - Description
   * - ``sigma``, ``ls``, ``ls_factor``, ``n_landmarks``, ``batch_size``, ``eps``, ``jit_compile``, ``random_state``, ``use_empirical_variance``
     - ``gp=GPSettings(...)``
     - GP model parameters
   * - ``null_genes``, ``null_seed``, ``fdr_threshold``
     - ``fdr=FDRSettings(...)``
     - FDR / null distribution (DE)
   * - ``log_fold_change_threshold``, ``ptp_threshold``
     - ``threshold=DAThresholdSettings(...)``
     - Significance thresholds (DA)
   * - ``cell_filter``, ``groups``, ``min_cells``, ``min_percentage``, ``check_representation``
     - ``filter=FilterSettings(...)``
     - Cell / group filtering (DE)
   * - ``result_key``, ``overwrite``, ``store_landmarks``, ``store_arrays_on_disk``, ``disk_storage_dir``, ``max_memory_ratio``, ``store_posterior_covariance``, ``store_additional_stats``
     - ``storage=StorageSettings(...)``
     - Result storage
   * - ``copy``, ``inplace``, ``progress``, ``compute_mahalanobis``, ``allow_single_condition_variance``
     - ``output=OutputSettings(...)``
     - Output control

For example, this config file:

.. code-block:: yaml

   sigma: 0.5
   n_landmarks: 3000
   fdr_threshold: 0.01

is equivalent to:

.. code-block:: python

   kompot.de(adata, ...,
       gp=GPSettings(sigma=0.5, n_landmarks=3000),
       fdr=FDRSettings(threshold=0.01))

YAML Format
^^^^^^^^^^^

Config files use standard YAML syntax with flat keys:

.. code-block:: yaml

   # Required parameters
   groupby: "condition"
   condition1: "control"
   condition2: "treatment"

   # Common parameters
   obsm_key: "X_pca"
   layer: "logged_counts"
   result_key: "kompot_de"

   # Sample variance
   sample_col: "sample_id"

   # GP parameters (→ GPSettings)
   sigma: 1.0
   ls_factor: 10.0
   batch_size: 100
   n_landmarks: 5000

   # FDR parameters (→ FDRSettings)
   fdr_threshold: 0.05
   null_genes: 2000

   # Filtering (→ FilterSettings)
   genes: ["Gene1", "Gene2", "Gene3"]
   cell_filter: {batch: "batch1"}

JSON Format
^^^^^^^^^^^

JSON is also supported:

.. code-block:: json

   {
     "groupby": "condition",
     "condition1": "control",
     "condition2": "treatment",
     "obsm_key": "X_pca",
     "batch_size": 100,
     "fdr_threshold": 0.05
   }

Config Templates
^^^^^^^^^^^^^^^^

Kompot provides ready-to-use templates:

**Minimal templates** (commonly used parameters only):

- ``kompot/cli/templates/dm_config_minimal.yaml``
- ``kompot/cli/templates/de_config_minimal.yaml``
- ``kompot/cli/templates/da_config_minimal.yaml``

**Complete templates** (all available parameters with documentation):

- ``kompot/cli/templates/dm_config_template.yaml``
- ``kompot/cli/templates/de_config_template.yaml``
- ``kompot/cli/templates/da_config_template.yaml``
- ``kompot/cli/templates/smooth_config_template.yaml``

Pipeline Integration
--------------------

Nextflow Example
^^^^^^^^^^^^^^^^

.. code-block:: groovy

   process KOMPOT_DE {
       input:
       path adata
       path config

       output:
       path "results.h5ad"

       script:
       """
       kompot de ${adata} -o results.h5ad -c ${config}
       """
   }

Snakemake Example
^^^^^^^^^^^^^^^^^

.. code-block:: python

   rule kompot_de:
       input:
           adata = "data/{sample}.h5ad",
           config = "configs/de_config.yaml"
       output:
           results = "results/{sample}_de.h5ad"
       shell:
           "kompot de {input.adata} -o {output.results} -c {input.config}"

Shell Script Example
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   #!/bin/bash
   # Process multiple samples with complete workflow

   for sample in sample1 sample2 sample3; do
       echo "Processing ${sample}..."

       # Step 1: Compute diffusion maps
       kompot dm \
           data/${sample}.h5ad \
           -o temp/${sample}_dm.h5ad \
           --pca-key X_pca \
           --n-components 10

       # Step 2: Differential expression
       kompot de \
           temp/${sample}_dm.h5ad \
           -o results/${sample}_de.h5ad \
           --groupby condition \
           --condition1 control \
           --condition2 treatment \
           --obsm-key DM_EigenVectors \
           --batch-size 100

       if [ $? -eq 0 ]; then
           echo "${sample} completed successfully"
           rm temp/${sample}_dm.h5ad  # Cleanup intermediate file
       else
           echo "${sample} failed" >&2
           exit 1
       fi
   done

Output Format
-------------

Differential Expression Output
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Results stored in:

- ``adata.var["kompot_de_{cond1}_to_{cond2}_mahalanobis"]`` - Significance scores
- ``adata.var["kompot_de_{cond1}_to_{cond2}_mean_lfc"]`` - Mean log fold change
- ``adata.var["kompot_de_{cond1}_to_{cond2}_is_de"]`` - Boolean significance flag
- ``adata.var["kompot_de_{cond1}_to_{cond2}_mahalanobis_local_fdr"]`` - Local FDR
- ``adata.uns["kompot_de"]`` - Run metadata and parameters

Differential Abundance Output
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Results stored in:

- ``adata.obs["kompot_da_{cond1}_to_{cond2}_lfc"]`` - Log fold change per cell
- ``adata.obs["kompot_da_{cond1}_to_{cond2}_lfc_zscore"]`` - Z-scores
- ``adata.obs["kompot_da_{cond1}_to_{cond2}_neg_log10_lfc_ptp"]`` - -log10 p-values
- ``adata.obs["kompot_da_{cond1}_to_{cond2}_lfc_direction"]`` - Direction (up/down/neutral)
- ``adata.uns["kompot_da"]`` - Run metadata and parameters

Logging and Verbosity
---------------------

Control logging output:

.. code-block:: bash

   # Standard logging (INFO level)
   kompot de input.h5ad -o output.h5ad --groupby condition ...

   # Verbose logging (DEBUG level)
   kompot -v de input.h5ad -o output.h5ad --groupby condition ...

   # Redirect logs
   kompot de input.h5ad -o output.h5ad ... 2> analysis.log

Error Handling
--------------

The CLI exits with different codes:

- ``0`` - Success
- ``1`` - General error (missing files, invalid parameters, analysis failure)
- ``130`` - Interrupted by user (Ctrl+C)

Check exit codes in scripts:

.. code-block:: bash

   kompot de input.h5ad -o output.h5ad ...
   if [ $? -ne 0 ]; then
       echo "Analysis failed" >&2
       exit 1
   fi

Performance Tips
----------------

Memory Management
^^^^^^^^^^^^^^^^^

For large datasets:

.. code-block:: bash

   # Reduce batch size
   kompot de input.h5ad -o output.h5ad ... --batch-size 50

   # Use fewer landmarks
   kompot de input.h5ad -o output.h5ad ... --n-landmarks 3000

   # Enable disk storage (requires config file)
   # In config.yaml:
   #   store_arrays_on_disk: true
   #   disk_storage_dir: "/tmp/kompot_cache"

Speed Optimization
^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Reduce null genes for faster FDR estimation
   kompot de input.h5ad -o output.h5ad ... --null-genes 1000

   # Use fewer landmarks
   kompot da input.h5ad -o output.h5ad ... --n-landmarks 2000

   # Disable progress bars in scripts
   kompot de input.h5ad -o output.h5ad ... --no-progress

Troubleshooting
---------------

Common Issues
^^^^^^^^^^^^^

**Missing required parameters:**

.. code-block:: text

   Error: Missing required parameters: groupby, condition1, condition2

*Solution:* Provide via CLI args or config file

**File not found:**

.. code-block:: text

   Error: AnnData file not found: input.h5ad

*Solution:* Check file path and ensure it exists

**Invalid condition label:**

.. code-block:: text

   Error: Condition 'X' not found in column 'condition'

*Solution:* Check condition labels in your data

**Memory error:**

.. code-block:: text

   MemoryError or JAX out of memory

*Solution:* Reduce ``--batch-size`` and ``--n-landmarks``

Getting Help
^^^^^^^^^^^^

.. code-block:: bash

   # General help
   kompot --help

   # Command-specific help
   kompot de --help
   kompot da --help
   kompot smooth --help
   kompot dm --help

   # Check version
   kompot --version

Comparison with Python API
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 15

   * - Feature
     - CLI
     - Python API
   * - Basic analysis
     - ✅ Simple
     - ✅ Simple
   * - Advanced parameters
     - ⚠️ Requires config file
     - ✅ Direct access
   * - Pipeline integration
     - ✅ Easy
     - ⚠️ Requires scripting
   * - Interactive exploration
     - ❌ Not suitable
     - ✅ Excellent
   * - Visualization
     - ❌ Requires separate step
     - ✅ Integrated
   * - Debugging
     - ⚠️ Limited
     - ✅ Full access
   * - Documentation
     - ✅ Built-in help
     - ✅ Comprehensive

**Recommendation:**

- Use **CLI** for: automated pipelines, batch processing, workflow integration
- Use **Python API** for: interactive analysis, visualization, parameter exploration, custom workflows

See Also
--------

- :doc:`Python API Documentation <simplified>`
- :doc:`Getting Started Tutorial <notebooks/01_getting_started>`
- :doc:`Sample Variance Guide <notebooks/03_sample_variance>`
- `GitHub Repository <https://github.com/settylab/kompot>`_
