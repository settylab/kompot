# Changelog

All notable changes to this project will be documented in this file.

## [0.9.0] - 2026-09-15

### Fixed — sample variance no longer costs more memory than it needs to

Supplying `sample_col` gives every gene its own `(n_landmarks, n_landmarks)`
covariance matrix per condition. Kompot summed those two tensors into a **third**
dense tensor before use, and then added the shared posterior covariance into it
gene by gene with `__setitem__`. Three consequences, all silent:

 - `StorageSettings(store_arrays_on_disk=True)` did not keep the tensors out of
   memory. With `np.memmap` inputs the sum materialised a full
   `(n_landmarks, n_landmarks, n_genes)` array in RAM — precisely the array the
   flag exists to avoid.
 - `compute_mahalanobis_distances` then ran `cov = jnp.array(covariance)` inside
   its gene-specific branch and never read `cov`: a second full-size copy of the
   tensor, for nothing.
 - Under Dask the per-gene `__setitem__` rebuilt the task graph once per gene,
   and the delayed per-gene tasks nested with the lazy slices they materialised,
   so peak memory scaled with concurrency rather than with one gene.

The sum is now assembled one gene at a time by `kompot.utils.LazyGeneCovariance`,
which holds references to the terms and materialises a single
`(n_landmarks, n_landmarks)` matrix when the Mahalanobis step asks for a gene.
Measured on 1 500 cells / 150 genes / 600 landmarks (412 MiB per tensor), peak
**anonymous** memory from `/proc/self/smaps_rollup`, same machine and same
synthetic input on both trees:

| run | 0.8.0 | 0.9.0 | bytes written, 0.8.0 → 0.9.0 | wall, 0.8.0 → 0.9.0 |
|---|---|---|---|---|
| no sample variance | 949 MiB | 972 MiB | 0 → 0 | 33.0 s → 32.6 s |
| sample variance, in memory | 2 401 MiB | 1 973 MiB | 0 → 0 | 45.7 s → 46.0 s |
| `store_arrays_on_disk`, with `dask` | 2 782 MiB | 1 153 MiB | 0 → 0 | 231.8 s → 47.8 s |
| `store_arrays_on_disk`, no `dask` | 2 284 MiB | 1 148 MiB | 824 → 824 MiB | 48.2 s → 47.7 s |

Disk-backed storage is now cheaper than in-memory storage, which it was not
before: 1 153 MiB against 1 973 MiB, where 0.8.0 made it *more* expensive
(2 782 against 2 401). Repeated at 900 landmarks / 120 genes: 2 809 MiB in
memory, 1 325 MiB with Dask and 1 341 MiB without, against 3 502 / 4 341 /
3 079 MiB on 0.8.0.

One cost moved the other way, on the path that is not recommended. Without
`dask` the tensors are memory-mapped, and a gene slice of a C-contiguous
`(n_points, n_points, n_genes)` map is strided, so reading one gene at a time
touches the whole file where the old code did one sequential read into RAM.
Measured on a clean 900-landmark pair, 74 s against 66 s — about 12% slower in
exchange for 57% less memory. Installing `dask` avoids it in both directions:
that path is 4.9x *faster* than 0.8.0 as well as lighter. Wall-clock figures
here come from a shared machine under load and should be read as orders of
magnitude.

**Results are unchanged.** Distances differ only by floating-point summation
order: measured maximum relative difference 1.7e-12 across the in-memory,
Dask and memmap paths, which is the same order as the pre-existing difference
*between* those paths on 0.8.0 alone.

In-memory cost per gene therefore drops from `3 x n_landmarks^2 x 8` to
`2 x n_landmarks^2 x 8` bytes — about 0.37 GiB rather than 0.56 GiB per gene at
the default `n_landmarks=5000`.

Fixes settylab/kompot#26.

### Fixed — the dry run now prices the run it is pricing

`kompot.de(..., dry_run=True)` built its plan *before* `null_genes="auto"` was
resolved, so the estimator received the literal string. That matched neither its
integer nor its list branch and fell through to zero, omitting the 2 000 null
genes the run would add. Measured on a 4 000 x 500 input: the default dry run
reported 0.485 GiB for a run the explicit `null_genes=2000` plan priced at
2.156 GiB — **4.44x optimistic**, and the gap grows with `n_cells`.

Resolution now happens above the dry-run branch, and
`estimate_differential_expression_resources` raises on a `null_genes` it cannot
interpret rather than counting zero.

Fixes settylab/kompot#25.

### Changed — the resource plan describes the code that runs

Following the fix above, `estimate_differential_expression_resources`:

 - no longer charges a third `(n_landmarks, n_landmarks, n_genes)` tensor that
   is never built;
 - charges the covariance tensors to **disk** only on the path that writes
   them. With `dask` installed nothing reaches `disk_storage_dir`, and the plan
   now reports that instead of demanding scratch the run does not use;
 - charges the per-gene working set that replaced the dense sum
   (`3 x n_landmarks^2 x 8` bytes, one gene at a time);
 - reworded the in-memory warning to name the levers that shrink the tensors
   (`genes`, linear; `n_landmarks`, quadratic) rather than only pointing at disk
   storage.

`store_arrays_on_disk` semantics are unchanged; only the accounting moved.

### Documentation

 - **New guide: [Planning Memory and Disk](https://kompot.readthedocs.io/en/latest/resource_planning.html)**,
   the canonical explanation of what `sample_col` costs and how to run it, with
   plans produced by `dry_run=True` at realistic sizes. It prescribes the
   two-pass workflow — a cheap first pass over all genes, then sample variance
   restricted to the top genes — ranks the levers, and shows how to price a run
   before committing to it.
 - The same warning now appears where the decision is made: `kompot.de`'s
   docstring (so `help()` carries it), `GPSettings.n_landmarks`,
   `GPSettings.batch_size`, `StorageSettings`, `SampleVarianceEstimator`,
   `kompot de --sample-col`, the DE config template, the README and the tutorial
   notebooks.
 - The CLI "Complete Analysis" example no longer pairs `--sample-col` with
   `--n-landmarks 5000` over every gene, which was the most expensive
   configuration the package can express; it now shows the restricted second
   pass through a config file.
 - Corrected `StorageSettings.max_memory_ratio`, documented as "Fraction of RAM
   before triggering disk storage". It triggers nothing: `store_arrays_on_disk`
   defaults to `None`, which resolves to `disk_storage_dir is not None`, and the
   ratio only sets the threshold at which the estimator escalates warnings. The
   DE config template's `store_arrays_on_disk: null # (null = auto)` was
   misleading for the same reason.

Fixes settylab/kompot#27.

## [0.8.0] - 2026-07-28

### Changed — statistics now match the manuscript

These numerical changes align Kompot's output with the published method. **Absolute values shift,
so re-tune any hard-coded thresholds carried over from 0.7.0.** The three changes differ sharply in
how far their effect reaches — one is presentational, one moves your call rate, and one moves which
genes are called at all:

 - **Differential-expression Mahalanobis distances are smaller by a factor of √2** (the combined
   posterior covariance is now `Σ_a + Σ_b` rather than `(Σ_a + Σ_b)/2`).

   *Impact: cosmetic in the default configuration.* Every distance is divided by the same √2, and
   the empirical null is built by running shuffled genes through the identical pipeline, so it
   rescales too. Gene rankings, p-values, local FDR and `is_de` calls are therefore **unchanged**;
   only the printed magnitudes move. The one exception is when sample variance is in play
   (`use_sample_variance`, auto-enabled when variance predictors or sample indices are supplied):
   there the denominator is `(Σ_a + Σ_b) + (V_a + V_b)`, and only the posterior term doubled, so
   the two components are reweighted rather than uniformly scaled. Genes dominated by posterior
   uncertainty shrink by the full √2, genes dominated by sample variance shrink less, and rankings
   can shift modestly as a result.

 - **Differential-abundance PTP is now one-sided** — `PTP = Φ(−|z|)` rather than `2Φ(−|z|)` — so
   reported values are halved (`neg_log10_fold_change_ptp` increases by ~0.30 = log₁₀2).

   *Impact: more sensitive at an unchanged threshold.* The transform is monotone, so cell rankings
   are **unchanged**, but every PTP halves and therefore more cells clear any fixed
   `ptp_threshold`. The threshold's meaning has loosened: `1e-3` used to require |z| ≥ 3.29 and now
   requires only |z| ≥ 3.09. **Halve your threshold to preserve the old call rate** — the extra
   calls are real gains in power only if you intended a one-sided test.

 - **`use_empirical_variance` now defaults to `False`** in every entry point
   (`DifferentialExpression`, `ExpressionModel`, `smooth_expression()`, the deprecated `compute_*`
   wrappers, and the CLI config templates), matching `kompot.de()`, which already defaulted to
   `False`. Pass `use_empirical_variance=True` explicitly to keep the old behavior.

   *Impact: this is the one that changes results, not just their scale.* It is not a rescaling.
   With it enabled, the expression GP is fitted with `obs_variance=True`, which smooths
   leverage-corrected squared residuals into a per-gene, input-dependent noise surface and adds it
   to the Mahalanobis denominator; disabling it removes that term. Consistently noisy genes are no
   longer down-weighted for their noise, so they rank higher relative to quiet ones — **gene
   rankings and `is_de` calls genuinely differ**, and results are not comparable to a 0.7.0 run
   that used the old default. It also drops two GP fits, so runs are faster and use less memory.
   If your data have no biological replicates, this was the only per-gene noise model in play;
   consider re-enabling it or supplying sample indices so sample variance takes its place.

### Changed — `volcano_de` no longer highlights a fallback when nothing is significant

 - **When no gene meets the significance criteria, nothing is highlighted.** Previously
   `volcano_de` fell back to highlighting the top 10 genes by score, labelled "Top 10 genes (no
   genes at threshold)". On a volcano the coloured points read as *"these are the hits"* whatever
   the legend says, so a negative result was rendered as a hit list — the failure mode is a reader
   taking ten arbitrary top-scoring genes for significant ones. The same applied when a stored
   `is_de` column marked nothing. Both now highlight nothing and emit a **warning** naming the
   criteria that matched zero genes; every gene still appears in the background colour, so the
   plot reads as the negative result it is.

   The legitimate fallback is unchanged: with **no** significance criterion given, `n_top_genes`
   still highlights the top N by score, because ranking without a threshold claims nothing about
   significance.

   If you relied on the old behaviour to always get some highlighted genes, pass `n_top_genes`
   explicitly instead — that asks for a ranking, which is what the fallback was silently
   substituting.

### Changed — renamed field (action required)

 - The differential-expression posterior tail probability is now stored as **`-log10(PTP)` in the `..._neg_log10_ptp` column** (previously the raw probability in `..._ptp`), so larger means more significant. This preserves ranking resolution that the old linear column lost for most genes. `volcano_de(y_axis_type="ptp")` handles the change for you; **update any code that read the old `_ptp` column.**

### Added

 - **Every run history entry now records which Kompot produced it.** `run_history`
   stored parameters and a timestamp but no version, so a `.h5ad` could not say what
   computed it. Each entry now also carries `kompot_version`, `kompot_git_sha` and
   `kompot_editable`. The sha and the editable flag are the load-bearing part: an
   editable install of `v0.7.0-7-g4432d4f` reports `__version__ == "0.7.0"`, so **a
   released version string alone does not identify the code that ran**, and
   reconstructing it after the fact means crossing run timestamps against release
   dates. The sha is read directly from the `.git`
   directory (no `git` binary required, no new dependency) and resolution never raises:
   fields are set to `null` when they cannot be determined, since a present-but-null
   field is self-describing whereas a missing key is indistinguishable from an older
   store. A wheel install with no work tree correctly reports a `null` sha and
   `kompot_editable: false` rather than borrowing the sha of any surrounding repository.

   **This is forward-only.** It describes runs computed by 0.8.0 and later; it does not
   retroactively describe stores written before it, whose entries simply lack these keys
   and continue to load unchanged. Anything consuming `run_history` should treat the
   three fields as optional.

 - **`kompot de --dry-run`**: estimate memory, disk, and output-field requirements without running the analysis. Prints JSON to stdout and a human-readable report to stderr.
 - **`kompot.plot.dotplot`**: fold-change dotplot (color = mean log-fold-change per group, size = fraction of cells expressing) that embeds into your own Matplotlib axes. Genes are given explicitly or auto-picked as top-N by Mahalanobis.
 - **`kompot.plot.lollipop`**: gene-set-enrichment lollipop plot that embeds into an existing axis. Accepts a `StringDBReport`, its enrichment table, or a generic enrichment table from other tools (gseapy/enrichr, GOATOOLS, clusterProfiler) with case-insensitive column autodetection.
 - **Custom background for `StringDBReport` enrichment**: pass `background=` (e.g. your analyzed `adata.var_names`) so over-representation is tested against the genes you actually measured instead of the whole genome — the correct choice for single-cell and targeted panels. Default behavior is unchanged.
 - **`kompot.configure_logging(stream)`**: redirect the kompot logger to a chosen stream.
 - **`random_state` on `kompot.find_landmarks`**: pass an int to make landmark selection reproducible. The underlying Leiden community detection draws from igraph's global RNG, which was otherwise left unseeded, so repeated calls on identical input could return a different number of landmarks. A supplied seed threads into both the nearest-neighbor construction and the Leiden step, so the same input and seed yield identical landmark indices and coordinates. The default (`random_state=None`) **preserves the historical non-deterministic behavior**, so existing callers are unaffected unless they opt in.

### Fixed

 - **Compatibility with pandas 3 and anndata 0.13.** Grouping by a string column
   (`groups="cell_type"`), passing a `pd.Series` of indices as a group, colouring a
   volcano background by a categorical `var` column, and auto-detecting the condition
   label in `plot_smoothing` all raised on pandas 3 / anndata 0.13. Kompot now detects
   dtypes through `pandas.api.types` rather than `numpy.issubdtype` (which cannot
   interpret pandas extension dtypes such as the new default `StringDtype`), indexes
   Series positionally, and ignores the `None` key that anndata 0.13 uses to expose
   `X` through `adata.layers`. Behaviour on pandas 2 / anndata 0.12 is unchanged, with
   one deliberate exception: a `volcano_de` background column of nullable `boolean`
   dtype is now coloured categorically rather than through a continuous colormap,
   aligning it with the numpy `bool` columns that were already treated as categorical.
 - **`cell_filter=` accepts nullable-integer index Series.** `apply_cell_filter()` probed
   the filter dtype with `numpy.issubdtype`, which cannot interpret the ExtensionArray
   that `Series.values` returns for a pandas extension dtype. Passing a nullable `Int64`
   Series of indices raised `TypeError: Cannot interpret 'Int64Dtype()' as a data type`
   instead of selecting cells, and passing a string Series surfaced the same opaque
   `TypeError` in place of the documented `ValueError`. Both paths are now covered by
   regression tests.

 - **Compatibility with matplotlib ≥ 3.9**: plotting no longer calls the removed `matplotlib.cm.get_cmap` API, so heatmap and volcano plots work on current matplotlib.
 - CLI commands now log to stderr, keeping stdout clean for machine-readable output (dry-run JSON, table output).
 - `kompot smooth` is now documented in the CLI guide; assorted API-documentation fixes.

## [0.7.0] - 2026-04-13

### Breaking changes

 - **Drop Python 3.9 support**: kompot now requires Python ≥ 3.10 (driven by mellon ≥ 1.7.0 dependency).

### New simplified API

 - **`kompot.de()`, `kompot.da()`, and `kompot.smooth_expression()` now use Settings dataclasses** (`GPSettings`, `FDRSettings`, `FilterSettings`, `StorageSettings`, `OutputSettings`) so the common case stays simple while advanced options remain discoverable. The old `compute_differential_*` and `compute_smoothed_expression()` functions still work but emit a deprecation warning.
 - **`dry_run=True`** on `de()` prints a resource plan (memory, disk, field overwrites) without running the analysis. Replaces the standalone `dry_run_differential_expression()`.
 - **`ModelSettings`** lets you inject pre-fitted predictors into `de()`, `da()`, and `smooth_expression()` to skip fitting or reuse models across runs.

### New features

 - **Null distribution inspection**: `return_full_results=True` now includes a `"null"` key in the result dict exposing all null gene data: Mahalanobis distances, smoothed expression, fold changes, z-scores, and standard deviations. A lightweight alternative (`OutputSettings(return_null_data=True)`) returns only the summary table and metadata (gene indices, names, seed, provenance) without the full expression matrices.
 - **External null distributions for FDR**: supply your own null distribution instead of relying on column-shuffled null genes.
   - `FDRSettings(null_mahalanobis=...)`: pre-computed null Mahalanobis distances (e.g., from a control-vs-control run).
   - `FDRSettings(null_expression=(expr1, expr2))`: raw null expression matrices fitted through the same GP model.
   - `FDRSettings(combine_with_internal=True)`: concatenate external and internal null distributions.
 - **`kompot.compute_fdr(real_mahal, null_mahal)`**: standalone FDR computation from Mahalanobis distances (no AnnData needed). Returns a DataFrame with `mahalanobis`, `pvalue`, `local_fdr`, `tail_fdr`, `is_de`.
 - **`kompot.extract_null_distribution(adata)`**: extract Mahalanobis distances from a DE run for reuse as a null distribution elsewhere.
 - **`kompot.recompute_fdr(adata, null_mahalanobis)`**: recompute FDR on existing DE results with a new null distribution, updating `adata.var` in place.
 - **`DifferentialExpression.compute_fdr(null_mahal)`**: sklearn-like method to compute FDR after `predict(compute_mahalanobis=True)`.
 - **Empirical variance** (`GPSettings(use_empirical_variance=True)`): estimates per-gene heteroscedastic noise from GP residuals and adjusts Mahalanobis distances accordingly. Works with or without biological replicates.
 - **`CenteredLinear` kernel** for better extrapolation at cell-state boundaries (opt-in via `cov_func`; default remains Matern52).
 - **More accurate uncertainty**: density estimators now use mellon 1.7.1's default Laplacian optimizer instead of ADVI.

### Run history and reproducibility

 - Run parameters are now stored grouped by Settings dataclass, making them directly reconstructible.
 - **`RunInfo.call_args()`** returns a kwargs dict that reproduces the run — edit it and pass to `de()`/`da()` to re-run with tweaked parameters.
 - **`RunInfo.to_settings()`** returns the Settings objects from a previous run for inspection.

### Improvements

 - **Input validation at construction time**: all Settings dataclasses now validate fields in `__post_init__`. Invalid values like `GPSettings(sigma=-1)` or `FDRSettings(threshold=1.5)` raise immediately with a clear message instead of failing deep inside mellon or JAX. The public API functions (`de()`, `da()`, `smooth_expression()`) also validate AnnData inputs upfront (obsm key shape, condition existence, `condition1 != condition2`, gene names, landmarks dimensions).
 - Plotting functions return `Optional[plt.Figure]` (controlled by `return_fig`) instead of `(fig, ax)` tuples, and no longer call `plt.show()`.
 - Consistent parameter naming across plot functions: `background_color_key` → `color`, `de_column` → `direction_column`, `embedding_key` → `basis`.
 - `RunInfo` HTML display now shows parameters hierarchically by Settings group (`gp.sigma`, `fdr.threshold`, …) instead of a flat list.
 - `RunComparison` shows individual changed fields (e.g. `gp.ls_factor: 10.0 → 5.0`) instead of opaque dict diffs.
 - **`kompot smooth` CLI command** for single-condition GP smoothing from the command line, matching the full Python API (condition selection, gene subsetting, empirical variance, sample variance).
 - `--no-progress` flag added to the DA CLI; progress bars can now be fully suppressed in both DA and DE.
 - DA CLI now exposes `--store-arrays-on-disk`, `--disk-storage-dir`, and `--max-memory-ratio`, matching the DE CLI's StorageSettings coverage.
 - FDR is disabled by default when `sample_col` is provided (not yet calibrated for sample variance). Override with `FDRSettings(null_genes=...)`.
 - Remove `statsmodels` dependency.

### Bug fixes

 - **Restore shared-landmark precomputation in DE** (requires mellon ≥ 1.7.1). Mellon's `compute_landmarks` had a silent string-vs-enum bug where `gp_type="fixed"` did not match `GaussianProcessType.FIXED`, causing the function to return `None` instead of the documented fall-through. Kompot's shared-landmark precomputation in `DifferentialExpression.fit()` and the per-condition fallback in `ExpressionModel.fit()` both routed through this code path, so on every DE call kompot was silently dropping the cross-condition shared landmark grid (each condition ended up with an independent full GP) and ignoring the user-supplied `random_state` for landmark selection (mellon's internal `_compute_landmarks` fell back to the hardcoded `DEFAULT_RANDOM_SEED=42`). Pinning `mellon>=1.7.1` enables the fix transparently — no kompot code changes were required.
 - **Shared landmarks across conditions in DA**. `DifferentialAbundance.fit()` now passes `gp_type="fixed"` to `compute_landmarks` and forwards `gp_type="fixed"` to the per-condition `DensityEstimator`s. Previously, when either condition had fewer cells than `n_landmarks`, mellon's auto-selection fell back to `gp_type=FULL` for that estimator, silently discarding the shared-landmark grid that DA had just computed on the combined data — the two density predictors then used independent full GPs, breaking the symmetry assumption behind the Mahalanobis-style abundance comparison. This brings DA into structural parity with DE.
 - Fix local FDR numerical instability (Grenander estimator replaces statsmodels Poisson GLM).
 - Fix tail FDR: replace Benjamini-Hochberg on empirical p-values (which breaks when `n_null` << `n_genes`) with fdrtool-style survival function ratio `Fdr(d) = S_null(d) / S_mix(d)`.
 - Fix `cell_filter` docs: parameter includes matching cells, not excludes.
 - Fix missing `field_mapping` in DA run history: `append_to_run_history` was called before `field_mapping` was computed, so DA history entries never recorded which fields were written.

## [0.6.3]

 - fix condition extraction across all plotting functions: condition names are now extracted from `run_info` params (authoritative source) instead of fragile `_extract_conditions_from_key()` string-splitting, which was broken for multi-word condition names (e.g. "Pre-treatment", "Wild Type"). Affected functions: `plot_gene_expression`, `volcano_da`, `volcano_de`, `multi_volcano_da`, `direction_barplot`
 - silent fallback to pattern-matched layers/keys from potentially wrong runs has been replaced with explicit warnings in `plot_gene_expression` and `volcano_de` (FDR/PTP key inference)

## [0.6.2]

 - fix differential expression analysis using `groups`
 - increase testing coverage
 - thread and GPU-usage control in CLI
 - fix `volcano_de` plot when the layer is `None`

## [0.6.1]

 - table output for CLI
 - default representation in CLI is diffusion maps
 - replace `results_dict` arrays with table of result
 - set default batch size to 0

## [0.6.0]

 - store kompot and other package versions in run info
 - implement command line tools for pipeline integration
 - comprehensive installation documentation with JAX GPU support
 - Zenodo badge automatically points to latest version

## [0.5.2]

 - CSR→LIL→CSR layer conversion for faster appending of partial differential expression results
 - same argument order in `dry_run_differential_expression` and `compute_differential_expression`
 - bugfix: fdr computation when all p-values are 0
 - increase testing coverage

## [0.5.0]

 - comprehensive FDR implementation for differential expression analysis
 - FDR-based visualization in `volcano_de` plots: support for local/tail FDR y-axes and coloring
 - posterior tail probability for differential expression
 - introduction of "is\_de" boolean column in `adata.var` to indicate differential expression based on significance threshold
 - more flexible `volcano_de` plot with FDR/PTP-based thresholding and y-axis options
 - "signal" and "strength" columns in stringDB gene-set enrichment analysis
 - expand testing
 - rename fields to include comparison, e.g., "A\_to\_B", before statistic name
 - make de significance measures tail fdr, ptp, and zscore optional
 - implement cleanup function
 - bugfix: Prevent silent failure of `compute_differential_abundance` with sample variance
   by making sure enough space is available on disk for covariance tensor.
 - dry run for differential expression
 - split tutorials in 3 parts
 - reduce memory demand when using batching and reflect this in dry run
 - fix disk space checking to respect TMPDIR environment variable consistently
 - include all computed results in full results dictionaries (std, fiel names, etc.)
   

## [0.4.3]

 - make sure all components are packaged

## [0.4.2]

 - avoid naming conflict on import

## [0.4.1]

 - avoid absolute imports

## [0.4.0]

 - StringDBReport class for gene set visualization and reporting
 - make sure da directions categories are always retained and ordered correctly
 - `fold_change_mode` parameter for heatmap to only show fold-change instead of split tiles
 - implement `RunInfo` utility to fetch information about previous runs
 - bugfix passing `ax` to `kompot.plot.embedding`
 - implemented `mgroups` in `kompot.plot.embedding` to plot multiple groupings
 - implement group-wise differential expression through `groups` parameter in `kompot.compute_differential_expression`
 - also return and store uncertainty estimates (stds) in de analysis
 - also return and store z-scores in de analysis
 - implement underrepresentation filtering for de analaysis
 - `plot.embedding` scanpy wrapper can now plot multiple layer
 - make sure modified anndata is writable (use JSON for run info in `.uns`)
 - option to store posterior covariance matrix in differential expression anndata function

## [0.3.3]

 - correct titles in expression plot
 - square patches in heatmap legend

## [0.3.2]

 - Remove default `pval_threshold=0.05` from volcano plots
 - change dfeaults of differential abundance `pval_threshold` to 0.05

## [0.3.1]

### Added
- Multiple volcano plot function `kompot.plot.multi_volcano_da`

### Removed
- Deprecated dependencies and setup.cfg
- Deprecated functions

## [0.3.0] - Previous release
