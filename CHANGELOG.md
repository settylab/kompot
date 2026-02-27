# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

 - fix local FDR numerical instability: replace statsmodels Poisson GLM (which caused overflow/divide-by-zero RuntimeWarnings) with Grenander estimator — boundary-corrected KDE + PAVA isotonic regression enforcing monotonically declining densities and local FDR with Mahalanobis distance. PAVA follows fdrtool's numerically stable incremental update. Added ground truth validation tests using exponential and gamma mixture distributions.
 - remove `statsmodels` dependency: Benjamini-Hochberg FDR correction is now implemented directly.
 - fix `cell_filter` documentation: parameter was documented as specifying cells to exclude, but the implementation includes matching cells. Updated docstrings, CLI config template, and docs to correctly describe inclusion semantics.
 - fix missing `field_mapping` in DA run history: `append_to_run_history` was called before `field_mapping` was added to `current_run_info`, so history entries never contained it.

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
