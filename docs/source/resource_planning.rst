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

The two costs respond to different levers, which is why the split is worth
making deliberately rather than by habit. ``store_arrays_on_disk`` removes the
memory term almost entirely (:ref:`disk-offload`) and ``n_landmarks`` shrinks
it quadratically, but neither touches the **per-gene Cholesky factorisation**,
which is linear in the gene count and which only a shorter gene list reduces.
Restricting ``genes`` is the one lever that moves both.

Held in memory, the gene list is a hard constraint: a whole transcriptome at
default landmarks asks for 7 552 GiB. Offloaded, it is a budget rather than a
wall — 101 GiB and about 11.6 hours of single-threaded factorisation. See
:ref:`how-many-genes` for which of those you are in.

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
loop.

Measured inside a real ``kompot.de`` sample-variance run, 20 genes, with
``OMP_NUM_THREADS=1``:

.. list-table::
   :header-rows: 1
   :widths: 24 26 24 26

   * - ``n_landmarks``
     - per gene
     - 1 000 genes
     - 20 000 genes
   * - 500
     - 0.016 s
     - 16 s
     - 5 min
   * - 1 000
     - 0.062 s
     - 1 min
     - 21 min
   * - 2 000
     - 0.246 s
     - 4 min
     - 1.4 h
   * - 5 000 (default)
     - 2.08 s
     - 35 min
     - 11.6 h

.. warning::

   **Pin your BLAS thread count for sample-variance runs on a shared node.**
   The per-gene factorisation is a single multi-threaded LAPACK call on a
   matrix that is small relative to the core count, so on a busy machine it
   spends its time in thread contention rather than in arithmetic. Measured on
   the same host, same commit, same run, ``n_landmarks=2000``:

   .. code-block:: text

      OMP_NUM_THREADS=1   0.246 s/gene   (whole run  52.7 s)
      unrestricted        6.135 s/gene   (whole run 280.8 s)

   A bare ``numpy.linalg.cholesky`` at n=2000 shows the same thing from the
   other side: 0.125 s at 21.3 GFLOPS pinned, against 5.20 s at **0.5 GFLOPS**
   unrestricted. A throughput figure that *collapses* as threads are added is
   contention, not work.

   So export ``OMP_NUM_THREADS=1`` (and ``OPENBLAS_NUM_THREADS`` /
   ``MKL_NUM_THREADS`` to match) before a sample-variance run you are sharing
   a node for. On an idle machine the unrestricted path would win; the point is
   that most people run this on a machine that is not idle.

   Pinning also makes the figure *reproducible*, which is the other reason to
   do it. Two independent measurements of the same configuration at 5 000
   landmarks, single-threaded, taken at 1-minute loads of 45.9 and 62.8 — a
   37% difference — came out at 2.084 and 2.113 s per gene, **1.4% apart**.
   Unpinned, the same step moved by a factor of 25. Pin the threads and the
   measurement stops being a property of the machine's mood.

The table above is therefore a **reproducible floor**, not a forecast: it is
what the step costs when it is not competing for cores. Your own figure depends
on your BLAS build, your core count and your node's load, so time a handful of
genes at your own ``n_landmarks`` before committing to a long run.

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

**Which constraint binds depends entirely on the storage mode, and the gap is
three orders of magnitude.** Held in memory, the whole transcriptome asks for
7 552 GiB and is simply out of reach. Offloaded, it asks for 101 GiB and a
large node has that. So :ref:`the two-pass workflow <two-pass-workflow>` is a
hard requirement in the first case and a matter of value for money in the
second — see :ref:`how-many-genes`, which says which is which rather than
implying a wall that is no longer there.

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

.. warning::

   **The plan is a floor, not a total.** It sums the arrays Kompot itself
   allocates; it does not model the Python interpreter, NumPy/BLAS scratch,
   JAX's device pools or Dask's scheduler. Measured on the 600-landmark /
   150-gene configuration used elsewhere on this page: a run with no sample
   variance plans at 69 MiB and peaks at 958 MiB of anonymous memory, and the
   *difference* sample variance makes plans at 16.7 MiB against a measured
   222 MiB — about 13x optimistic.

   That gap is roughly constant for a given dataset shape rather than growing
   with the covariance term, which is what makes the plan useful anyway: the
   figures it gets *exactly* right are the ones that explode, the
   ``2 x n_landmarks^2 x n_genes x 8`` tensors and the disk footprint. Use it
   to compare configurations and to catch the terabyte-scale plans, and leave
   real headroom above whatever it reports.

.. note::

   ``kompot.resource_estimation.dry_run_differential_expression()`` is
   deprecated. Use ``kompot.de(..., dry_run=True)``.

.. _null-genes-cost:

.. note::

   Null genes are charged at the same per-gene rate as real ones, and the dry
   run accounts for them: it resolves ``null_genes`` exactly as the run would,
   including the ``"auto"`` default, so the plan prices the run it describes.
   If you override the default and request FDR alongside sample variance, each
   null gene gets its own landmark covariance matrix too — 1 000 real genes
   plus 2 000 null genes is a 3 000-gene tensor.

   Before Kompot 0.9.0 the estimate was built before ``"auto"`` was resolved
   and silently counted zero null genes, so a default-settings plan without
   ``sample_col`` under-reported by 2 000 genes
   (`settylab/kompot#25 <https://github.com/settylab/kompot/issues/25>`_).


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
tensor), four BLAS threads. **Two memory instruments, because one is not
enough**: ``Anonymous`` counts private heap pages, ``Rss`` counts those *plus*
resident file-backed pages, and the difference is exactly where a memory map
puts its data.

.. list-table::
   :header-rows: 1
   :widths: 28 13 13 15 12 12

   * - run
     - anon
     - Rss
     - file-backed
     - written
     - wall
   * - no sample variance
     - 958 MiB
     - 1 218 MiB
     - 260 MiB
     - 0
     - 36 s
   * - sample variance, in memory
     - 1 975 MiB
     - 2 238 MiB
     - 263 MiB
     - 0
     - 49 s
   * - ``store_arrays_on_disk``, with ``dask``
     - 1 180 MiB
     - 1 442 MiB
     - 262 MiB
     - **0**
     - 50 s
   * - ``store_arrays_on_disk``, no ``dask``
     - 1 139 MiB
     - **2 224 MiB**
     - **1 085 MiB**
     - 824 MiB
     - 51 s

As a share of the extra memory the in-memory run costs over a run with no
sample variance at all:

.. list-table::
   :header-rows: 1
   :widths: 34 22 22

   * - path
     - on ``Anonymous``
     - on ``Rss``
   * - with ``dask``
     - ~1/5
     - **~1/5**
   * - no ``dask``
     - ~1/5
     - **~97%**

**These are deliberately given to one figure.** The share is a ratio of two
differences — an extra of roughly 200 MiB between peaks of 950 and 1 150 MiB —
so run-to-run noise in either peak swings it hard. Measured across twelve runs
on this host, the ``no sample variance`` baseline spans 938-981 MiB and the
``dask`` peak spans 1 134-1 209 MiB, which brackets that share anywhere from
**14% to 27%** without anything changing but the run. Two independent rigs
landing on 17% and 22% is the same noise, not a methodology difference.

Pinning does not tighten it. Fixing Dask's threaded-scheduler pool at 1, at 4
and leaving it at the default (36) gave 1 173/1 209, 1 152/1 163 and
1 134/1 138 MiB — all inside the spread, and the single-worker runs were if
anything the highest.

What the measurement *does* support is the separation, and it is not close:
the ``dask``-less path costs **about a fifth on Anonymous and essentially all
of it on Rss**, a factor of five apart on the same run. That is the finding;
the second significant figure never was.

The ``dask`` path **wins on every instrument**, and that is the claim to rely
on: about a fifth of the extra memory, nothing written to disk, nothing
file-backed, and 4.8x faster than 0.8.0 at matched thread counts. It is the
recommended path for exactly this reason.

The ``dask``-less path is a different bargain, and **should not be described as
a smaller footprint**. It is 99% of the in-memory extra on ``Rss``: what it
changes is the *kind* of page, converting private anonymous memory into
resident page cache backed by a file on disk. Those pages are reclaimable — the
kernel can evict them under pressure and read them back — so it is a genuine
win for surviving a memory squeeze, and not a reduction in resident footprint.

.. warning::

   **Under a cgroup, page cache is charged to you.** Slurm's ``--mem``, a
   container limit and any other cgroup-based cap count resident file-backed
   pages against the same budget as anonymous ones. So the ``dask``-less path's
   advantage may not exist in exactly the environment most people run this in.
   The figures above were taken on an unconstrained host, so they show what is
   resident, not what an OOM killer would decide; treat the reclaimability
   argument as a mechanism, not as a verdict. If you are under a hard cap,
   install ``dask``.

.. note::

   Earlier revisions of this page reported only ``Anonymous`` and justified it
   as keeping page cache from "flattering the reading". That justification was
   backwards: choosing ``Anonymous`` is precisely what *excludes* the 823 MiB
   the ``dask``-less path moves into file-backed residency, which is why it
   appeared to cost a sixth of the in-memory run rather than nearly all of it.

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
arithmetic rather than a measurement. It cuts the per-gene factorisation
steeply too: pinned to one thread, 0.016 s per gene at 500 landmarks against
2.08 s at 5 000. Those four points are close to the cubic flop count a Cholesky
implies — 500 to 5 000 is 10x the landmarks for 130x the time — but they are
four timings on one machine, not a scaling law, and they move with your BLAS
build and your node's load. Take them as a floor and time your own.

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
tuned or validated parameter — and it is a budget, not a wall.** Be clear about
which constraint you are actually under, because they differ by orders of
magnitude:

* **In memory**, the whole transcriptome is out of reach: 7 552 GiB at default
  landmarks. This is exact arithmetic and it is the hard limit.
* **With** ``store_arrays_on_disk=True``, it is not: 101 GiB, which a large
  node has. The memory wall is the one the offload removes.
* **In time**, the whole transcriptome is a long job rather than an impossible
  one: ~11.6 h at 5 000 landmarks, ~1.4 h at 2 000, single-threaded.

So a whole-transcriptome sample-variance run is *feasible* once the tensors are
offloaded. Restricting to ~1 000 genes is still the right default, for reasons
that are about value rather than capacity:

* pass 1 has to run over all genes anyway, and it is what ranks them;
* 1 000 genes costs about a twentieth of the whole transcriptome on both axes,
  for the genes you were going to read;
* FDR is not calibrated for sample variance (``null_genes`` resolves to ``0``
  when ``sample_col`` is set), so the extra 19 000 genes buy no additional
  testable calls.

If you do want wider coverage, cut ``n_landmarks`` in the same breath: it
reduces the per-gene time and the memory together.

Price it rather than guessing. A :ref:`dry run <dry-run>` gives the memory and
disk figures for your dimensions in a second; for the time, multiply your gene
count by a single-gene timing at your landmark count, pinned.

See also
--------

* :doc:`DE with Sample Variance <notebooks/03_sample_variance>` for the
  two-pass workflow end to end on real data.
* :doc:`Advanced Differential Expression <notebooks/02_differential_expression_detailed>`
  for resource planning alongside the other tuning knobs.
* :doc:`simplified` for :class:`~kompot.StorageSettings` and
  :class:`~kompot.GPSettings` in full.
* :doc:`cli` for ``kompot de --dry-run``.
