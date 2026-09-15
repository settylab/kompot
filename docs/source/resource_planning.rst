Planning Memory and Disk
========================

Most Kompot runs are unremarkable in their resource use. One option is not:
supplying ``sample_col`` to :func:`kompot.de` turns on **sample variance**,
which replaces a single shared covariance matrix with **one covariance matrix
per gene**. The cost of that step is multiplied by the number of genes
analysed, and at Kompot's default landmark count it reaches hundreds of
gigabytes for a routine gene set.

This page explains where the cost comes from, prescribes the two-pass workflow
that keeps it bounded, and shows how to price a run with ``dry_run=True``
before committing to it.

.. _two-pass-workflow:

The prescribed workflow: two passes
-----------------------------------

**Never run sample variance over the whole transcriptome.** Run it twice
instead:

1. **Pass 1, all genes, no sample variance.** Cheap, and it gives you the
   Mahalanobis ranking.
2. **Pass 2, top genes only, with sample variance.** Restrict to the genes
   that survived pass 1 (on the order of 1 000; see :ref:`how-many-genes`)
   and cut ``n_landmarks`` if the plan is still too large.

.. code-block:: python

   import kompot

   # ---- Pass 1: all genes, no sample variance -------------------------
   kompot.de(adata, "condition", "Young", "Old")

   # ---- Pick the genes worth refining ---------------------------------
   mahal = "kompot_de_Young_to_Old_mahalanobis"
   top_genes = adata.var.sort_values(mahal, ascending=False).head(1000).index

   # ---- Pass 2: sample variance, restricted ---------------------------
   kompot.de(
       adata, "condition", "Young", "Old",
       sample_col="donor_id",
       genes=top_genes,                          # the lever that matters most
       gp=kompot.GPSettings(n_landmarks=2000),   # quadratic; see "Other levers"
       storage=kompot.StorageSettings(
           store_arrays_on_disk=True,            # read "Disk offload" below first
           disk_storage_dir="/scratch/kompot",   # large, fast, node-local
       ),
   )

Price it before you run it. A dry run of exactly that call takes a second and
tells you whether it fits (see :ref:`dry-run`). The two levers in that call
that genuinely shrink the allocation are ``genes`` and ``n_landmarks``;
``store_arrays_on_disk`` is discussed, with measurements, under
:ref:`disk-offload`.

The two passes compose rather than collide, which is what makes the split
practical. Measured on a two-pass run where pass 2 analysed 10 of 60 genes:

* the sample-variance score is written to a **new** column with a
  ``_sample_var`` suffix
  (``kompot_de_Young_to_Old_mahalanobis_sample_var``), populated for the genes
  pass 2 analysed and ``NaN`` elsewhere, while pass 1's
  ``..._mahalanobis`` column stays populated for **all** genes. The two sit
  side by side in ``adata.var`` and can be compared directly;
* ``..._mean_lfc`` is unchanged. Fold changes are unaffected by sample
  variance; only the significance moves;
* the shared layers (``..._smoothed``, ``..._fold_change``) are rewritten for
  the genes pass 2 analysed and left untouched for the rest. Pass 2 refits the
  GP on its gene subset, so those columns change slightly. If you need pass 1's
  layers preserved byte for byte, give pass 2 its own
  ``StorageSettings(result_key=...)``;
* two new layers appear, ``..._<condition>_std``, one per condition.

FDR is disabled automatically on the second pass: with ``sample_col`` set,
``FDRSettings(null_genes="auto")`` resolves to ``0``, because the null
calibration is not yet validated for sample variance. That is also what you
want for cost, since null genes are charged at the same per-gene rate as real
ones (see :ref:`the null-gene note <null-genes-cost>`).


.. _where-the-cost-is:

Where the cost comes from
-------------------------

Without sample variance, the Mahalanobis distance for every gene is taken
against **one** posterior covariance matrix of shape ``(n_landmarks,
n_landmarks)``, factorised once.

With sample variance, Kompot fits a per-sample expression predictor, evaluates
it at the landmarks, and forms the **sample-to-sample covariance between
landmark pairs, separately for every gene**. That array has shape

.. code-block:: text

   (n_landmarks, n_landmarks, n_genes)

and Kompot holds one per condition, plus their sum. So the dominant term is

.. code-block:: text

   bytes  =  k x n_landmarks^2 x n_genes x 8          (float64)

   k = 3   variance1 + variance2 + their sum, all dense in memory
   k = 2   what the planner charges to disk when store_arrays_on_disk=True,
           on the assumption that the sum stays lazy

Read the exponents carefully. The cost is **linear in genes** and **quadratic
in landmarks**. At Kompot's default ``n_landmarks=5000``, a single landmark
covariance matrix is ``5000^2 x 8 B = 190 MiB``, so:

.. admonition:: The number to remember
   :class: warning

   At the default 5 000 landmarks, **every gene added to a sample-variance
   run costs about 0.56 GiB**. One thousand genes is roughly 560 GiB; ten
   thousand is over five terabytes. Nothing else in a Kompot run behaves this
   way, which is why the gene list on pass 2 is the number to decide
   deliberately.

Compute scales with genes as well, and worse than the shared path does. With
one shared covariance, Kompot performs a single Cholesky factorisation and
solves against it for all genes in vectorised batches. With per-gene
covariances it factorises **once per gene**, in a Python loop over genes
(``kompot/utils.py``), and ``GPSettings.batch_size`` does not apply to that
loop.

Two smaller terms also grow with the gene count under sample variance, and
neither is affected by ``store_arrays_on_disk``:

* **Per-sample imputations**, ``2 x n_samples x n_landmarks x n_genes x 8`` B.
  At 5 000 landmarks, 1 000 genes and 6 donors this is 0.45 GiB, negligible
  beside the covariance tensors but not beside everything else.
* **Two extra layers**, ``<result_key>_<condition>_std``, each
  ``n_cells x n_genes x 8`` B, written into ``adata.layers``.


Measured plans
--------------

The figures below are Kompot's own ``dry_run=True`` output for a synthetic
AnnData of 20 000 cells and 20 000 genes, two conditions, 6 donors, at the
default ``n_landmarks=5000``. They are totals for the whole plan, not just the
covariance term. (Kompot's report labels these binary magnitudes ``GB``; they
are the same numbers.)

.. list-table::
   :header-rows: 1
   :widths: 14 20 20 32

   * - genes
     - no sample variance
     - sample variance, RAM
     - sample variance, disk-backed
   * - 200
     - 3.1 GiB
     - 115.0 GiB
     - 3.2 GiB RAM + 74.5 GiB disk
   * - 1 000
     - 6.4 GiB
     - 566.0 GiB
     - 7.2 GiB RAM + 372.5 GiB disk
   * - 2 000
     - 10.6 GiB
     - 1 130 GiB
     - 12.1 GiB RAM + 745.1 GiB disk
   * - 20 000 (all)
     - 85.7 GiB
     - 11 276 GiB
     - 100.6 GiB RAM + 7 451 GiB disk

The last row is the configuration to avoid: sample variance over a full
transcriptome at default settings asks for eleven terabytes. The
no-sample-variance column grows only with ``n_cells x n_genes``, which is why
pass 1 over all 20 000 genes is comfortable at 86 GiB.

The fourth column is what the *planner* charges when
``store_arrays_on_disk=True``. Read it as the scratch footprint the run is
allowed to reach, not as a memory saving you can count on; see the warning
under :ref:`disk-offload`.

Reproduce any row on your own data by substituting your ``adata``:

.. code-block:: python

   for on_disk in (False, True):
       plan = kompot.de(
           adata, "condition", "Young", "Old",
           sample_col="donor_id",
           genes=top_genes,
           storage=kompot.StorageSettings(store_arrays_on_disk=on_disk),
           dry_run=True,
       )
       print(on_disk, plan.total_memory_required, plan.total_disk_required)


.. _dry-run:

The dry run is the planning tool
--------------------------------

``kompot.de(..., dry_run=True)`` estimates the whole plan and returns a
:class:`~kompot.resource_estimation.ResourcePlan` instead of running anything.
It prints a report and exposes the numbers programmatically:

.. code-block:: python

   plan = kompot.de(
       adata, "condition", "Young", "Old",
       sample_col="donor_id",
       genes=top_genes,
       storage=kompot.StorageSettings(store_arrays_on_disk=True),
       dry_run=True,
   )

   plan.total_memory_required   # bytes
   plan.total_disk_required     # bytes
   plan.is_feasible             # False if it cannot fit
   plan.warnings                # overwrite and headroom warnings
   for r in plan.requirements:
       print(r.resource_type, r.shape, r.size_human, r.name)

The report breaks the plan down per array, so the sample-variance term is
visible by name (``Sample covariances (per condition, N samples)``) with its
``(n_landmarks, n_landmarks, n_genes)`` shape. It also lists the AnnData
fields that would be created and flags any that an earlier run already wrote,
with that run's ``run_id``.

The intended use is to price the options against each other before committing
to any of them. Hold the gene set fixed to read what sample variance itself
costs:

.. code-block:: python

   base = dict(groupby="condition", condition1="Young", condition2="Old",
               genes=top_genes, dry_run=True)

   without = kompot.de(adata, **base)
   with_sv = kompot.de(adata, sample_col="donor_id", **base)

   print(with_sv.total_memory_required / without.total_memory_required)

then vary the levers, one at a time, until the plan fits:

.. code-block:: python

   for n_landmarks in (1000, 2000, 5000):
       plan = kompot.de(adata, sample_col="donor_id",
                        gp=kompot.GPSettings(n_landmarks=n_landmarks), **base)
       print(n_landmarks, plan.total_memory_required, plan.is_feasible)

The same estimate is available from the command line as ``kompot de
--dry-run``, which writes JSON to stdout and the human-readable report to
stderr, and exits non-zero when the plan is infeasible. See :doc:`cli`.

.. note::

   ``kompot.resource_estimation.dry_run_differential_expression()`` is
   deprecated. Use ``kompot.de(..., dry_run=True)``.

.. _null-genes-cost:

.. warning::

   The dry run reads ``null_genes`` as you pass it and does not resolve the
   ``"auto"`` default. With ``FDRSettings`` left at its default and no
   ``sample_col``, the real run adds 2 000 null genes but the dry run counts
   none, so the plan is optimistic by that margin. Pass
   ``fdr=kompot.FDRSettings(null_genes=2000)`` to the dry run to see the true
   figure. Runs *with* ``sample_col`` are unaffected, because ``"auto"``
   resolves to ``0`` there and the dry run also counts 0. Tracked in
   `settylab/kompot#25 <https://github.com/settylab/kompot/issues/25>`_.

   Null genes are charged at the same per-gene rate as real ones. If you
   override the default and request FDR alongside sample variance, each null
   gene gets its own landmark covariance matrix too: 1 000 real genes plus
   2 000 null genes is a 3 000-gene tensor.


.. _disk-offload:

Disk offload
------------

``StorageSettings(store_arrays_on_disk=True)`` is intended to change how the
per-gene covariance tensors are represented, so they are consumed one gene at
a time instead of built as three dense in-memory arrays. It is the flag the
dry run's warnings point you at, and the one the estimator prices into its
disk column. **Read the warning at the end of this section before you rely on
it**: measured runs do not show the memory saving.

.. code-block:: python

   kompot.de(
       adata, "condition", "Young", "Old",
       sample_col="donor_id",
       genes=top_genes,
       storage=kompot.StorageSettings(
           store_arrays_on_disk=True,
           disk_storage_dir="/scratch/kompot",   # large, fast, node-local
       ),
   )

What it does mechanically depends on whether ``dask`` is installed:

* **With** ``dask`` (``pip install 'kompot[dask]'``) the tensor is a lazy
  Dask graph, evaluated one gene at a time as the Mahalanobis step reaches it.
  Measured: **no file is written at all**, so the dry run's disk figure is an
  upper bound on scratch rather than a prediction of it.
* **Without** ``dask`` Kompot writes the full tensor for each condition as a
  memory-mapped ``covariance_matrix.npy`` under ``disk_storage_dir``, and the
  dry run's disk figure is then what actually lands there. This path is
  sequential rather than parallel, and the dry run warns that ``dask`` is
  missing.

Either way, **set** ``disk_storage_dir``, and create it before you plan
against it. A real run creates a missing directory, but the **dry run raises**
``FileNotFoundError`` on one, so the estimate fails on exactly the
configuration you were trying to price. Left unset, Kompot writes into the
system temporary directory
(honouring ``TMPDIR``), which on a shared cluster is frequently small,
RAM-backed, or both. Kompot creates a unique per-run subdirectory inside it
and removes that subdirectory when the estimator is collected, so space is
reclaimed at the end of the run rather than during it.

``store_arrays_on_disk`` defaults to ``None``, which means *on if and only if*
``disk_storage_dir`` is set. Nothing turns it on automatically in response to
memory pressure. ``max_memory_ratio`` only sets the threshold at which
warnings escalate; it does not switch storage modes.

.. warning::

   **Measured runs do not show the memory saving this flag promises.**
   ``store_arrays_on_disk=True`` moves the tensor from the memory column of
   the plan to the disk column, but peak memory of the actual ``kompot.de()``
   run is unchanged. Two synthetic configurations, 1 500 cells and 4 donors,
   with peak *anonymous* memory sampled from ``/proc/self/smaps_rollup`` so
   that page cache for a memory-mapped file is excluded:

   .. list-table:: 150 genes, 600 landmarks (tensor 412 MiB per condition)
      :header-rows: 1
      :widths: 34 14 14 16 14 12

      * - run
        - plan RAM
        - plan disk
        - peak anon
        - on disk
        - wall
      * - no sample variance
        - 69 MiB
        - 0
        - 962 MiB
        - 0
        - 49 s
      * - sample variance, in memory
        - 1 313 MiB
        - 0
        - 2 357 MiB
        - 0
        - 87 s
      * - on disk, with ``dask``
        - 77 MiB
        - 824 MiB
        - 2 781 MiB
        - **0**
        - 283 s
      * - on disk, no ``dask``
        - 77 MiB
        - 824 MiB
        - 2 346 MiB
        - 824 MiB
        - 50 s

   .. list-table:: 120 genes, 900 landmarks (tensor 742 MiB per condition)
      :header-rows: 1
      :widths: 34 14 14 16 14 12

      * - run
        - plan RAM
        - plan disk
        - peak anon
        - on disk
        - wall
      * - no sample variance
        - 73 MiB
        - 0
        - 1 052 MiB
        - 0
        - 55 s
      * - sample variance, in memory
        - 2 307 MiB
        - 0
        - 3 505 MiB
        - 0
        - 67 s
      * - on disk, with ``dask``
        - 82 MiB
        - 1 483 MiB
        - 4 347 MiB
        - **0**
        - 259 s
      * - on disk, no ``dask``
        - 82 MiB
        - 1 483 MiB
        - 3 562 MiB
        - 1 483 MiB
        - 113 s

   Two things to read off these. The peak-memory column is flat across the
   three sample-variance rows at both scales, against a plan predicting a 17x
   to 28x drop. And with ``dask`` installed **nothing reaches disk at all**:
   the tensor is a lazy graph rather than a file, so the plan's disk figure
   bounds scratch consumption rather than predicting it, and that path was
   also the slowest of the four. Tracked in `settylab/kompot#26
   <https://github.com/settylab/kompot/issues/26>`_.

   Until that is resolved, treat **restricting genes** and **reducing
   landmarks** as the levers that actually bound memory. They shrink the
   allocation itself, and the dry run prices them correctly.

Other levers
------------

``n_landmarks`` **is the strongest lever**, because the tensor is quadratic in
it. Same 1 000 genes, sample variance on disk, from the dry run:

.. list-table::
   :header-rows: 1
   :widths: 20 20 20

   * - ``n_landmarks``
     - RAM
     - disk
   * - 1 000
     - 6.1 GiB
     - 14.9 GiB
   * - 2 000
     - 6.3 GiB
     - 59.6 GiB
   * - 3 000
     - 6.5 GiB
     - 134.1 GiB
   * - 5 000 (default)
     - 7.2 GiB
     - 372.5 GiB

Halving the landmark count quarters the covariance cost. Landmarks control
the resolution of the cell-state approximation, so this is a genuine accuracy
trade-off rather than a free saving, but 5 000 landmarks is rarely required to
resolve a covariance structure that is being summarised gene by gene.

``batch_size`` (in :class:`~kompot.GPSettings`) bounds the temporary arrays
during prediction and during the *shared*-covariance Mahalanobis computation.
It does **not** bound the per-gene covariance loop, so it is a lever for pass 1
and for peak prediction memory, not for the sample-variance tensor.

``genes`` is the lever that matters for pass 2, and it is the subject of the
next section.

.. _how-many-genes:

How many genes in pass 2?
-------------------------

Cost is linear in the gene count, so this is a budget decision rather than a
statistical one, and Kompot supplies no default: ``genes=None`` means *all of
them*, which is the one setting to avoid here.

**Roughly 1 000 genes is a recommendation chosen from the cost curve, not a
tuned or validated parameter.** It is the point at which the covariance term
is around 560 GiB at default landmarks, or 370 GiB of scratch, while still
covering the interesting tail of a Mahalanobis ranking. Past roughly 2 000
genes the in-memory figure passes a terabyte, and the per-gene Cholesky loop
becomes the dominant runtime term because it is the only stage that grows with
the gene count while the rest of the run is essentially fixed. If you need
more coverage than that, reduce ``n_landmarks`` in the same breath rather than
raising the gene count alone.

Price it rather than guessing. A :ref:`dry run <dry-run>` gives the exact
figure for your dimensions in a second.


See also
--------

* :doc:`DE with Sample Variance <notebooks/03_sample_variance>` for the
  two-pass workflow end to end on real data.
* :doc:`Advanced Differential Expression <notebooks/02_differential_expression_detailed>`
  for resource planning alongside the other tuning knobs.
* :doc:`simplified` for :class:`~kompot.StorageSettings` and
  :class:`~kompot.GPSettings` in full.
* :doc:`cli` for ``kompot de --dry-run``.
