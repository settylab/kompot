# Changelog

All notable changes to this project will be documented in this file.

## [0.5.0]

 - comprehensive FDR implementation for differential expression analysis
 - FDR-based visualization in `volcano_de` plots: support for local/tail FDR y-axes and coloring
 - posterior tail probability for differential expression
 - itroduction of "is\_de" boolean column in `adata.var` to indicate differential expression based on significance threshold
 - more flexible `volcano_de` plot with FDR/PTP-based thresholding and y-axis options
 - "signal" and "strength" columns in stringDB gene-set enrichment analysis
 - expand testing
 - rename fileds to include comparison, e.g., "A\_to\_B", before statistic name
 - make de significnace measures tail fdr, ptp, and zscore optional
 - implement cleanup function
 - bugfix: Prevent silent failure of `compute_differential_abundance` with sample variance
   by making sure enough space is available on disk for covariance tensor.
   

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
