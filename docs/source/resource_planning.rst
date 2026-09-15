Planning Memory and Disk
========================

Most Kompot runs are unremarkable in their resource use. One option is not:
supplying ``sample_col`` to :func:`kompot.de` turns on **sample variance**,
which replaces a single shared covariance matrix with **one covariance matrix
per gene**. Both the memory and the compute of that step are multiplied by the
number of genes analysed.

This page explains where that cost comes from, prescribes the two-pass workflow
that keeps it bounded, and shows how to price a run with ``dry_run=True``
before committing to it.

.. _two-pass-workflow:

The prescribed workflow: two passes
-----------------------------------

**Never run sample variance over the whole transcriptome.** Run it twice
instead:

1. **Pass 1, all genes, no sample variance.** Cheap, and it gives you the
   Mahalanobis ranking.
2. **Pass 2, top genes only, with sample variance.** Restrict to the genes that
   survived pass 1 (on the order of 1 000; see :ref:`how-many-genes`), offload
   the covariance tensors to disk, and cut ``n_landmarks`` if the run is still
   too slow.

.. code-block:: python

   import kompot

   # ---- Pass 1: all genes, no sample variance -------------------------
   kompot.de(adata, "condition", "Young", "Old")

   # ---- Pick the genes worth refining ---------------------------------
   mahal = "kompot_de_Young_to_Old_mahalanobis"
   top_genes = adata.var.sort_values(mahal, ascending=False).head(1000).index

   # ---- Pass 2: sample variance, restricted and offloaded -------------
   kompot.de(
       adata, "condition", "Young", "Old",
       sample_col="donor_id",
       genes=top_genes,                          # linear in both memory and time
       gp=kompot.GPSettings(n_landmarks=2000),   # memory is QUADRATIC in this
       storage=kompot.StorageSettings(
           store_arrays_on_disk=True,            # keeps the tensors out of RAM
           disk_storage_dir="/scratch/kompot",   # large, fast, node-local
       ),
   )

Price it before you run it: a dry run of exactly that call takes a second and
tells you whether it fits (see :ref:`dry-run`).

The reason the split is not optional is that the two costs respond to different
levers. ``store_arrays_on_disk`` removes the memory term almost entirely
(:ref:`disk-offload`), and ``n_landmarks`` shrinks it quadratically — but
neither touches the **per-gene Cholesky factorisation**, which is linear in the
gene count and nothing makes it cheaper except analysing fewer genes. Restrict
the gene list and both costs fall together; offload without restricting and you
have merely moved from running out of memory to running out of time.

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
it at the landmarks, and forms the sample-to-sample covariance between landmark
pairs **separately for every gene**. That array has shape

.. code-block:: text

   (n_landmarks, n_landmarks, n_genes)

and there is one per condition. Their sum is never materialised: the
Mahalanobis step assembles a single ``(n_landmarks, n_landmarks)`` matrix per
gene as it goes. So the dominant term, when the tensors are held in memory, is

.. code-block:: text

   bytes  =  2 x n_landmarks^2 x n_genes x 8          (float64)

Read the exponents carefully. The cost is **linear in genes** and **quadratic
in landmarks**. At Kompot's default ``n_landmarks=5000``, a single landmark
covariance matrix is ``5000^2 x 8 B = 190 MiB``, so:

.. admonition:: The number to remember
   :class: warning

   At the default 5 000 landmarks, **every gene added to an in-memory
   sample-variance run costs about 0.37 GiB**. One thousand genes is roughly
   370 GiB; the whole transcriptome is over seven terabytes. Nothing else in a
   Kompot run behaves this way.

   :ref:`disk-offload` removes that term from memory entirely, which is what
   makes large gene sets possible at all. It does not remove the **compute**
   below, which is why the two-pass workflow remains the prescription either
   way.

.. note::

   Before Kompot 0.9.0 the two tensors were summed into a third dense array
   before use, so the figure above was ``3 x n_landmarks^2 x n_genes x 8``
   bytes, or about 0.56 GiB per gene, and ``store_arrays_on_disk`` did not
   reduce peak memory at all. If you are reading a plan produced by 0.8.0 or
   earlier, those are the numbers it will show
   (`settylab/kompot#26 <https://github.com/settylab/kompot/issues/26>`_).

Compute scales with genes as well, and it scales worse than the shared path
does. With one shared covariance Kompot performs a single Cholesky
factorisation and solves against it for all genes in vectorised batches; the
total is essentially flat in the gene count. With per-gene covariances it
factorises **once per gene**, in a Python loop over genes
(``kompot/utils.py``), and ``GPSettings.batch_size`` does not apply to that
loop. Timed on one machine, 20 genes per measurement:

.. list-table::
   :header-rows: 1
   :widths: 22 26 26 26

   * - ``n_landmarks``
     - shared, 20 genes
     - per-gene, 20 genes
     - per-gene, per gene
   * - 500
     - 0.61 s
     - 10.5 s
     - 0.52 s
   * - 1 000
     - 8.7 s
     - 121.6 s
     - 6.1 s
   * - 1 500
     - 11.8 s
     - 156.7 s
     - 7.8 s
   * - 2 000
     - 10.8 s
     - 170.9 s
     - 8.5 s

Those are wall-clock seconds on one shared machine and will not transfer
exactly. The shared column is not monotone in ``n_landmarks`` because at these
sizes it is dominated by JAX compilation rather than by the factorisation, so
read it as *"a fixed cost, independent of the gene count"* rather than as a
measurement of the solve. The per-gene column is the one to take seriously, and
what it shows is that it is **linear in the gene count** where the shared
column is flat.

Multiply it out, because a per-gene figure is not something anyone can act on.
At 2 000 landmarks and 8.5 s per gene:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - genes in pass 2
     - factorisation time
     - in practice
   * - 1 000
     - ~2.4 hours
     - an overnight job
   * - 2 000
     - ~4.7 hours
     - still an overnight job
   * - 20 000 (all)
     - ~47 hours
     - two days of compute

That, rather than memory, is what bounds a sample-variance run once the
tensors are offloaded, and it is the reason
:ref:`the two-pass workflow <two-pass-workflow>` remains the prescription. A
whole-transcriptome sample-variance run is no longer impossible; it is merely
interminable, which is a worse failure because it looks like progress.

Two smaller terms also grow with the gene count under sample variance, and
neither is affected by ``store_arrays_on_disk``:

* **Per-sample imputations**, ``2 x n_samples x n_landmarks x n_genes x 8`` B.
  At 5 000 landmarks, 1 000 genes and 6 donors this is 0.45 GiB.
* **Two extra layers**, ``<result_key>_<condition>_std``, each
  ``n_cells x n_genes x 8`` B, written into ``adata.layers``.

Measured plans
--------------

Kompot's own ``dry_run=True`` output for a synthetic AnnData of 20 000 cells
and 20 000 genes, two conditions, 6 donors, at the default
``n_landmarks=5000`` and ``null_genes=0``. These are totals for the whole plan,
not just the covariance term. (Kompot's report labels these binary magnitudes
``GB``; they are the same numbers.)

.. list-table::
   :header-rows: 1
   :widths: 12 18 20 22 18

   * - genes
     - no sample variance
     - sample variance, in memory
     - sample variance, ``store_arrays_on_disk``
     - disk used
   * - 200
     - 3.1 GiB
     - 78.3 GiB
     - 3.8 GiB
     - 0
   * - 1 000
     - 6.4 GiB
     - 380.2 GiB
     - 7.7 GiB
     - 0
   * - 2 000
     - 10.6 GiB
     - 757.7 GiB
     - 12.6 GiB
     - 0
   * - 20 000 (all)
     - 85.7 GiB
     - 7 552 GiB
     - 101.2 GiB
     - 0

Three things to read off this.

**The in-memory column is the one that explodes.** 1 000 genes asks for
380 GiB, and the whole transcriptome for seven and a half terabytes, because
that column carries ``2 x n_landmarks^2 x n_genes x 8`` bytes of covariance.

**Disk offload collapses it.** The same 1 000 genes plans at 7.7 GiB, barely
above the 6.4 GiB of a run with no sample variance at all, because the tensors
are never held whole. The disk column reads zero because ``dask`` is installed
in the environment that produced the table, and on that path the tensors are
evaluated lazily rather than written; see :ref:`disk-offload` for the
``dask``-less path, where the same figure is written to
``disk_storage_dir`` instead.

**Memory stops being the binding constraint, and compute takes over.** The
20 000-gene disk-backed plan at 101 GiB would fit on a large node; the 20 000
per-gene Cholesky factorisations behind it would not fit in your week. That is
why :ref:`the two-pass workflow <two-pass-workflow>` is the prescription even
now that the memory problem is solved.

Reproduce any row on your own data by substituting your ``adata``:

.. code-block:: python

   for on_disk in (False, True):
       plan = kompot.de(
           adata, "condition", "Young", "Old",
           sample_col="donor_id",
           genes=top_genes,
           storage=kompot.StorageSettings(store_arrays_on_disk=on_disk),
           fdr=kompot.FDRSettings(null_genes=0),
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

   # null_genes is pinned on BOTH arms. Left at "auto" it resolves to 2 000
   # without sample_col and to 0 with it, so the comparison would silently be
   # between two different gene counts.
   base = dict(groupby="condition", condition1="Young", condition2="Old",
               genes=top_genes, fdr=kompot.FDRSettings(null_genes=0),
               dry_run=True)

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

``StorageSettings(store_arrays_on_disk=True)`` keeps the per-gene covariance
tensors out of memory. It is the single most effective lever once you have
decided which genes to refine, and since Kompot 0.9.0 it is also **cheaper
than the in-memory path rather than dearer**.

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

* **With** ``dask`` (``pip install 'kompot[dask]'``) each condition's tensor is
  a lazy Dask graph, evaluated one gene at a time as the Mahalanobis step
  reaches it. **Nothing is written to disk**, so ``disk_storage_dir`` is used
  only for its free-space check. This is the faster path and the recommended
  one.
* **Without** ``dask`` each condition's tensor is written as a memory-mapped
  ``covariance_matrix.npy`` under ``disk_storage_dir``, so the dry run's disk
  figure is what actually lands there. The per-gene work is then sequential,
  and the dry run warns that ``dask`` is missing.

  **This path is also slower than holding the tensors in memory**, by about
  12% on a measured 900-landmark pair (74 s against 66 s). A gene slice of a
  C-contiguous ``(n_points, n_points, n_genes)`` memory map is **strided** —
  its elements sit ``n_genes * 8`` bytes apart — so reading one gene at a time
  touches the whole file, where holding the tensor reads it once sequentially.
  That is the price of not holding it, you pay it only without ``dask``, and
  installing ``dask`` removes it.

Measured on 1 500 cells, 150 genes, 600 landmarks and 4 donors (412 MiB per
tensor), peak **anonymous** memory sampled from ``/proc/self/smaps_rollup`` so
that page cache for a memory-mapped file cannot flatter the reading:

.. list-table::
   :header-rows: 1
   :widths: 30 16 16 18 20

   * - run
     - peak memory
     - bytes written
     - wall clock
     - vs. no sample variance
   * - no sample variance
     - 972 MiB
     - 0
     - 33 s
     - —
   * - sample variance, in memory
     - 1 973 MiB
     - 0
     - 46 s
     - +1 001 MiB
   * - ``store_arrays_on_disk``, with ``dask``
     - 1 153 MiB
     - **0**
     - 48 s
     - +181 MiB
   * - ``store_arrays_on_disk``, no ``dask``
     - 1 148 MiB
     - 824 MiB
     - 48 s
     - +176 MiB

The offloaded runs cost about a sixth of the extra memory that the in-memory
run does, and the gap widens with the gene count, because the in-memory column
carries a term linear in genes and the offloaded ones do not. Repeated at 900
landmarks and 120 genes (742 MiB per tensor), peak memory was 2 809 MiB
in memory, 1 325 MiB with ``dask`` and 1 341 MiB without, against 1 051 MiB
for a run with no sample variance at all.

The wall-clock column is from a shared machine under load and should be read
as an order of magnitude, not a benchmark. One caveat it does show reliably:
the ``dask``-less path is **modestly slower than it was before 0.9.0** — 74 s
against 66 s on a clean 900-landmark pair — because a gene slice of a
C-contiguous ``(n_points, n_points, n_genes)`` memory map is strided, so
reading one gene at a time touches the whole file where the old code did one
sequential read into RAM. That is the price of not holding the tensor, it is
paid only when ``dask`` is absent, and installing ``dask`` avoids it in both
directions.

Either way, **set** ``disk_storage_dir``, and create it before you plan against
it. A real run creates a missing directory, but the **dry run raises**
``FileNotFoundError`` on one, so the estimate fails on exactly the
configuration you were trying to price. Left unset, Kompot uses the system
temporary directory (honouring ``TMPDIR``), which on a shared cluster is
frequently small, RAM-backed, or both. Kompot creates a unique per-run
subdirectory inside it and removes that subdirectory when the estimator is
collected, so space is reclaimed at the end of the run rather than during it.

``store_arrays_on_disk`` defaults to ``None``, which means *on if and only if*
``disk_storage_dir`` is set. Nothing turns it on automatically in response to
memory pressure. ``max_memory_ratio`` only sets the threshold at which the
plan's warnings escalate; it does not switch storage modes.

.. note::

   Before Kompot 0.9.0 this flag did not reduce peak memory, because the two
   tensors were summed into a dense array before use whatever the storage
   mode. On the same configuration as the table above, 0.8.0 measured
   2 782 MiB with ``dask`` and 2 284 MiB without, against 2 401 MiB for the
   in-memory run — that is, offloading made the run **more** expensive, and
   the ``dask`` path was 4.9x slower as well
   (`settylab/kompot#26 <https://github.com/settylab/kompot/issues/26>`_).

Other levers
------------

``n_landmarks`` **is the strongest memory lever**, because the covariance term
is quadratic in it. Same 1 000 genes, from the dry run:

.. list-table::
   :header-rows: 1
   :widths: 24 26 26

   * - ``n_landmarks``
     - sample variance, in memory
     - with ``store_arrays_on_disk``
   * - 1 000
     - 21.0 GiB
     - 6.1 GiB
   * - 2 000
     - 66.0 GiB
     - 6.4 GiB
   * - 3 000
     - 140.8 GiB
     - 6.7 GiB
   * - 5 000 (default)
     - 380.2 GiB
     - 7.7 GiB

Halving the landmark count quarters the covariance footprint, which is exact
arithmetic rather than a measurement. It also cuts the per-gene factorisation,
though **by how much is not something these four points establish**: the timing
table above rises from 0.52 s to 8.5 s per gene between 500 and 2 000
landmarks, but the per-interval ratios are 11.7x, 1.3x and 1.1x, which is not a
power law. A Cholesky is O(n^3) in flops and the measurement plainly is not
following that at these sizes, so take the table as four measured points in
the range it covers and do not extrapolate a scaling from it. Measure at your
own landmark count if the time matters.

Landmarks control the resolution of the cell-state approximation, so reducing
them is a genuine accuracy trade-off rather than a free saving. But 5 000 is
rarely required to resolve a covariance structure that is being summarised gene
by gene.

``batch_size`` (in :class:`~kompot.GPSettings`) bounds the temporary arrays
during prediction and the gene batches of the *shared*-covariance Mahalanobis
computation. It does **not** bound the per-gene covariance loop, which
processes one gene at a time regardless, so it is a lever for pass 1 and for
peak prediction memory, not for sample variance.

``genes`` is the lever that matters for pass 2, and it is the subject of the
next section.

.. _how-many-genes:

How many genes in pass 2?
-------------------------

Both costs are linear in the gene count, so this is a budget decision rather
than a statistical one, and Kompot supplies no default: ``genes=None`` means
*all of them*, which is the one setting to avoid here.

**Roughly 1 000 genes is a recommendation chosen from the cost curves, not a
tuned or validated parameter.** With ``store_arrays_on_disk=True`` memory is no
longer what decides it — 1 000 genes plans at 7.7 GiB, and even the whole
transcriptome plans at 101 GiB, which would fit on a large node. What decides
it is the per-gene Cholesky: at 2 000 landmarks and roughly 8.5 s per gene,
1 000 genes is about **2.4 hours** and 20 000 genes about **47 hours**. One
thousand genes covers the interesting tail of a Mahalanobis ranking at a cost
you can absorb overnight; the whole transcriptome costs two days for a result
whose last nine-tenths you were never going to read.

If you need more coverage than that, cut ``n_landmarks`` in the same breath.
It reduces the per-gene cost as well as the memory, so it buys back some of the
time the extra genes spend — how much, at your dimensions, is worth a single
timed gene rather than an extrapolation.

Price it rather than guessing. A :ref:`dry run <dry-run>` gives the memory and
disk figures for your dimensions in a second; for the time, multiply your gene
count by a single-gene timing at your landmark count.

See also
--------

* :doc:`DE with Sample Variance <notebooks/03_sample_variance>` for the
  two-pass workflow end to end on real data.
* :doc:`Advanced Differential Expression <notebooks/02_differential_expression_detailed>`
  for resource planning alongside the other tuning knobs.
* :doc:`simplified` for :class:`~kompot.StorageSettings` and
  :class:`~kompot.GPSettings` in full.
* :doc:`cli` for ``kompot de --dry-run``.
