# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### New features

 - **`--dry-run` flag for `kompot de` CLI**: estimates memory, disk, and output field requirements without running the analysis. Outputs machine-parseable JSON to stdout and a human-readable report to stderr. Exit code reflects feasibility.
 - **`kompot.configure_logging(stream)`**: reconfigure the kompot logger output stream. The CLI now logs to stderr by default, keeping stdout clean for machine-parseable output (dry-run JSON, table output).
 - **`kompot.plot.dotplot`**: ax-embeddable fold-change-per-group dotplot. Color = mean of a per-cell LFC layer within each `groupby` category; size = fraction of cells expressing. Gene selection is either an explicit list or auto-picked top-N by Mahalanobis from run history (with optional `filter_key`, e.g. restricting to `is_de=True`). Pass `axes=(main, cbar, size_legend)` to compose into a larger figure, or leave `axes=None` for a standalone figure. Unlike `scanpy.pl.DotPlot`, this function does not build its own `GridSpec` and does not fight externally-provided axes, which is the whole reason it exists. Shares gene-selection, layer-fetch, and colormap-normalization primitives with `kompot.plot.heatmap` via the existing `heatmap.utils` helpers.

### Improvements

 - **CLI logs to stderr**: all `kompot` CLI commands now write log messages to stderr instead of stdout, so stdout is reserved for data output.
 - **`kompot smooth` documented in CLI guide**: added full command reference, options, and examples to the Sphinx CLI docs.
 - Fix double-backslash rendering in all CLI doc code blocks.
 - Exclude deprecated `compute_differential_*` functions from Sphinx automodule output.
 - Add `smooth_expression()` module to Sphinx API docs.
 - Add `RunInfo.to_settings()` and `call_args()` to documented members.
 - Fix "Gene Expression Imputation" → "Gene Expression Smoothing" in docs toctree.
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
