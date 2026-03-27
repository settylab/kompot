# Changelog

All notable changes to this project will be documented in this file.

## [0.7.0]

### New simplified API

 - **`kompot.de()` and `kompot.da()` replace the old `compute_differential_*` functions.** Related parameters are grouped into Settings dataclasses (`GPSettings`, `FDRSettings`, `FilterSettings`, `StorageSettings`, `OutputSettings`) so the common case stays simple while advanced options remain discoverable. The old functions still work but emit a deprecation warning.
 - **`dry_run=True`** on `de()` prints a resource plan (memory, disk, field overwrites) without running the analysis. Replaces the standalone `dry_run_differential_expression()`.
 - **`ModelSettings`** lets you inject pre-fitted predictors into `de()` / `da()` to skip fitting or reuse models across runs.

### New features

 - **Empirical variance** (`GPSettings(use_empirical_variance=True)`): estimates per-gene heteroscedastic noise from GP residuals and adjusts Mahalanobis distances accordingly. Works with or without biological replicates.
 - **`CenteredLinear` kernel** for better extrapolation at cell-state boundaries (opt-in via `cov_func`; default remains Matern52).
 - **More accurate uncertainty**: density and expression estimators now use mellon 1.7.0's default optimizer instead of ADVI.

### Run history and reproducibility

 - Run parameters are now stored grouped by Settings dataclass, making them directly reconstructible.
 - **`RunInfo.call_args()`** returns a kwargs dict that reproduces the run — edit it and pass to `de()`/`da()` to re-run with tweaked parameters.
 - **`RunInfo.to_settings()`** returns the Settings objects from a previous run for inspection.

### Improvements

 - Plotting functions return `Optional[plt.Figure]` (controlled by `return_fig`) instead of `(fig, ax)` tuples, and no longer call `plt.show()`.
 - Consistent parameter naming across plot functions: `background_color_key` → `color`, `de_column` → `direction_column`, `embedding_key` → `basis`.
 - `--no-progress` flag added to the DA CLI; progress bars can now be fully suppressed in both DA and DE.
 - FDR is disabled by default when `sample_col` is provided (not yet calibrated for sample variance). Override with `FDRSettings(null_genes=...)`.
 - Remove `statsmodels` dependency.

### Bug fixes

 - Fix local FDR numerical instability (Grenander estimator replaces statsmodels Poisson GLM).
 - Fix tail FDR: replace Benjamini-Hochberg on empirical p-values (which breaks when `n_null` << `n_genes`) with fdrtool-style survival function ratio `Fdr(d) = S_null(d) / S_mix(d)`.
 - Fix `cell_filter` docs: parameter includes matching cells, not excludes.

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
