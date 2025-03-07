"""Main class for generating interactive HTML reports from Kompot analyses."""

import os
import json
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Union, Any, Tuple
import webbrowser
import datetime
import uuid
import importlib.resources
import logging

from kompot.differential import DifferentialExpression

logger = logging.getLogger("kompot")


class HTMLReporter:
    """Generate interactive HTML reports from differential expression analysis.
    
    Instead of generating static plots, this class saves the data required for 
    visualization and uses JavaScript to render interactive plots in the browser.
    """
    
    def __init__(
        self,
        output_dir: str = "kompot_report",
        title: str = "Kompot Analysis Report",
        subtitle: str = None,
        template_dir: str = None,
        use_cdn: bool = False,  # Whether to use CDN for JS libraries
    ):
        """Initialize the HTML reporter.
        
        Args:
            output_dir: Directory where the report will be saved
            title: Title of the report
            subtitle: Subtitle of the report
            template_dir: Custom template directory (if None, use default)
            use_cdn: Whether to use CDN for JavaScript libraries instead of local files
        """
        self.output_dir = output_dir
        self.title = title
        self.subtitle = subtitle or ""
        self.template_dir = template_dir or self._get_default_template_dir()
        self.use_cdn = use_cdn
        self.components = []
        self.report_id = str(uuid.uuid4())[:8]  # Unique ID for this report
        
        # Data storage
        self.diff_expr_data = []
        self.anndata_info = None
        self.comparison_data = []
    
    def _get_default_template_dir(self) -> str:
        """Get the path to the default template directory."""
        logger.debug("Getting default template directory")
        try:
            # For Python 3.9+
            with importlib.resources.files("kompot") as pkg_path:
                template_path = str(pkg_path / "reporter/templates")
                logger.debug(f"Using importlib.resources.files, found template path: {template_path}")
                return template_path
        except AttributeError:
            # Fallback for older versions
            import importlib_resources
            logger.debug("Falling back to importlib_resources for older Python versions")
            with importlib_resources.path("kompot", "reporter/templates") as p:
                template_path = str(p)
                logger.debug(f"Using importlib_resources.path, found template path: {template_path}")
                return template_path
    
    def add_differential_expression(
        self, 
        diff_expr: DifferentialExpression,
        condition1_name: str,
        condition2_name: str,
        gene_names: Optional[List[str]] = None,
        top_n: int = 100,
    ):
        """Add differential expression results to the report.
        
        Args:
            diff_expr: DifferentialExpression object with results
            condition1_name: Name of the first condition
            condition2_name: Name of the second condition
            gene_names: List of gene names (optional)
            top_n: Number of top genes to include in the report
        """
        logger.info(f"Adding differential expression results for {condition1_name} vs {condition2_name}")
        
        # Extract the necessary data from the DifferentialExpression object
        # With the new API, we need to ensure predictions have been computed
        logger.debug("Extracting differential expression metrics")
        
        # Check if attributes are populated, if not, run prediction
        if diff_expr.fold_change is None or diff_expr.fold_change_zscores is None:
            logger.info("Model attributes not yet populated. Running prediction to compute values.")
            # Need to get original training data dimensions
            if not hasattr(diff_expr, 'n_condition1') or diff_expr.n_condition1 is None:
                raise ValueError("Model not fitted. Please call fit() and predict() before using with reporter.")
                
            # Create combined data points of the correct size
            n_total = diff_expr.n_condition1 + diff_expr.n_condition2 if diff_expr.n_condition2 is not None else 0
            if n_total == 0:
                raise ValueError("Cannot determine appropriate dimensions. Please call predict() explicitly before using with reporter.")
                
            # Find out if we have landmarks
            has_landmarks = False
            landmarks = None
            if hasattr(diff_expr.function_predictor1, 'landmarks') and diff_expr.function_predictor1.landmarks is not None:
                has_landmarks = True
                landmarks = diff_expr.function_predictor1.landmarks
            
            # Use landmarks if available, otherwise create dummy data of appropriate size
            if has_landmarks:
                logger.info("Using landmarks for predictions")
                X_combined = landmarks
            else:
                logger.warning("No landmarks found. Using random data for predictions. Results may not be accurate.")
                X_combined = np.random.randn(n_total, diff_expr.function_predictor1.dim)
                
            # Run prediction to compute fold changes and metrics
            _ = diff_expr.predict(X_combined, compute_mahalanobis=True)
        
        # Determine number of genes
        if gene_names is not None:
            n_genes = len(gene_names)
        elif hasattr(diff_expr, 'fold_change') and diff_expr.fold_change is not None:
            n_genes = diff_expr.fold_change.shape[1]  # Get number of genes from fold_change
        else:
            # Default to a small number if we can't determine
            n_genes = 50
            logger.warning(f"Could not determine number of genes. Using default: {n_genes}")
            
        # Now extract metrics from the populated model
        # Handle cases where the attributes could be None or not set
        if hasattr(diff_expr, 'fold_change') and diff_expr.fold_change is not None:
            fc = np.mean(diff_expr.fold_change, axis=0)  # Get mean across samples for each gene
        else:
            fc = np.zeros(n_genes)  # Default to zeros if not available
            
        if hasattr(diff_expr, 'fold_change_zscores') and diff_expr.fold_change_zscores is not None:
            fc_zscores = np.mean(diff_expr.fold_change_zscores, axis=0)  # Get mean across samples for each gene
        else:
            fc_zscores = np.zeros(n_genes)  # Default to zeros if not available
        
        # Check if Mahalanobis distances have been computed
        if not hasattr(diff_expr, 'mahalanobis_distances') or diff_expr.mahalanobis_distances is None:
            logger.warning("Mahalanobis distances not computed. Using zeros.")
            m_distances = np.zeros(fc.shape[0])
        else:
            m_distances = diff_expr.mahalanobis_distances  # Already correct shape (n_genes,)
            
        # We no longer use weighted fold change from the DifferentialExpression object
        # Just use regular fold change for this value
        if hasattr(diff_expr, 'fold_change') and diff_expr.fold_change is not None:
            # Use regular mean fold change as weighted fold change is no longer supported in DifferentialExpression
            wfc = np.mean(diff_expr.fold_change, axis=0)
            logger.info("Using regular fold change for report. DifferentialExpression no longer stores weighted_mean_log_fold_change.")
        else:
            logger.info("Fold change not available. Using zeros.")
            wfc = np.zeros(n_genes)  # Default to zeros if not available
            
        # Handle additional metrics
        if hasattr(diff_expr, 'lfc_stds') and diff_expr.lfc_stds is not None:
            lfc_stds = diff_expr.lfc_stds  # Already correct shape (n_genes,)
        else:
            lfc_stds = np.zeros(n_genes)  # Default to zeros if not available
            
        if hasattr(diff_expr, 'bidirectionality') and diff_expr.bidirectionality is not None:
            bidir = diff_expr.bidirectionality  # Already correct shape (n_genes,)
        else:
            bidir = np.zeros(n_genes)  # Default to zeros if not available
        
        # Generate gene names if not provided
        if gene_names is None:
            logger.debug("No gene names provided, generating generic names")
            gene_names = [f"gene_{i}" for i in range(fc.shape[0])]
        else:
            logger.debug(f"Using provided gene names, total: {len(gene_names)}")
        
        # Create a dataframe with all the results
        logger.debug("Creating results DataFrame")
        results_df = pd.DataFrame({
            "gene": gene_names,
            "log2FoldChange": fc,
            "z_score": fc_zscores,
            "mahalanobis_distance": m_distances,
            "weighted_fold_change": wfc,
            "fold_change_std": lfc_stds,
            "bidirectionality": bidir
        })
        
        # Sort by absolute mahalanobis distance
        logger.debug("Sorting results by absolute Mahalanobis distance")
        results_df["abs_mdist"] = np.abs(results_df["mahalanobis_distance"])
        results_df = results_df.sort_values("abs_mdist", ascending=False)
        results_df = results_df.drop(columns=["abs_mdist"])
        
        # Store the top genes data
        logger.debug(f"Storing top {top_n} genes for report")
        self.diff_expr_data.append({
            "condition1": condition1_name,
            "condition2": condition2_name,
            "results_df": results_df,
            "diff_expr": diff_expr,
            "top_n": top_n
        })
        logger.info(f"Successfully added differential expression data for {len(results_df)} genes")
    
    def add_anndata(
        self, 
        adata: "AnnData",
        groupby: str,
        embedding_key: str = "X_umap",
        cell_annotations: List[str] = None,
    ):
        """Add AnnData object with cell annotations for visualizations.
        
        Args:
            adata: AnnData object with cell data
            groupby: Column in adata.obs to group cells by
            embedding_key: Key in adata.obsm for the embedding coordinates
            cell_annotations: List of columns in adata.obs to use for cell annotations
        """
        logger.info(f"Adding AnnData object with grouping by '{groupby}' and embedding '{embedding_key}'")
        
        try:
            import anndata
            logger.debug("Successfully imported anndata package")
        except ImportError:
            logger.error("Failed to import anndata package")
            raise ImportError(
                "The 'anndata' package is required for working with AnnData objects. "
                "Please install it with 'pip install anndata'."
            )
        
        if not isinstance(adata, anndata.AnnData):
            logger.error(f"Expected anndata.AnnData object, but got {type(adata)}")
            raise TypeError("adata must be an AnnData object")
        
        if embedding_key not in adata.obsm:
            logger.error(f"Embedding key '{embedding_key}' not found in adata.obsm. Available keys: {list(adata.obsm.keys())}")
            raise ValueError(f"Embedding key '{embedding_key}' not found in adata.obsm")
        
        if groupby not in adata.obs:
            logger.error(f"Group key '{groupby}' not found in adata.obs. Available columns: {list(adata.obs.columns)}")
            raise ValueError(f"Group key '{groupby}' not found in adata.obs")
        
        # Default to all string columns in adata.obs if cell_annotations is None
        if cell_annotations is None:
            logger.debug("No cell annotations provided, auto-detecting categorical columns")
            cell_annotations = [
                col for col in adata.obs.columns 
                if isinstance(adata.obs[col].dtype, (object, pd.CategoricalDtype))
            ]
            logger.debug(f"Auto-detected {len(cell_annotations)} categorical annotations: {cell_annotations}")
        
        # Validate all cell annotations exist
        for anno in cell_annotations:
            if anno not in adata.obs:
                logger.error(f"Cell annotation '{anno}' not found in adata.obs")
                raise ValueError(f"Cell annotation '{anno}' not found in adata.obs")
        
        # Store the anndata info for later processing
        logger.debug(f"Storing AnnData object with {adata.n_obs} cells and {len(cell_annotations)} annotations")
        self.anndata_info = {
            "adata": adata,
            "groupby": groupby,
            "embedding_key": embedding_key,
            "cell_annotations": cell_annotations
        }
        logger.info(f"Successfully added AnnData object with {adata.n_obs} cells")
    
    def add_comparison(
        self,
        kompot_results: DifferentialExpression,
        other_results: Dict[str, pd.DataFrame],
        gene_names: Optional[List[str]] = None,
        comparison_name: str = "Method Comparison",
    ):
        """Add comparison between Kompot and other methods (e.g., DESeq2, Scanpy).
        
        Args:
            kompot_results: DifferentialExpression object with Kompot results
            other_results: Dict mapping method names to DataFrames with results
            gene_names: List of gene names (optional)
            comparison_name: Name for this comparison
        """
        # Validate the other results
        for method, df in other_results.items():
            required_cols = ["log2FoldChange"]
            optional_cols = ["pvalue", "padj", "z_score", "statistic"]
            
            # Check for at least one of the optional columns
            has_optional = any(col in df.columns for col in optional_cols)
            if not has_optional:
                raise ValueError(
                    f"Results for method '{method}' must have at least one of "
                    f"these columns: {optional_cols}"
                )
            
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(
                        f"Results for method '{method}' must have a '{col}' column"
                    )
        
        # Generate gene names if not provided
        if gene_names is None:
            # Check if fold_change is available
            if hasattr(kompot_results, 'fold_change') and kompot_results.fold_change is not None:
                gene_count = kompot_results.fold_change.shape[1]
            else:
                # Default to a reasonable number if fold_change isn't available
                gene_count = 20
                logger.warning(f"Could not determine gene count from fold_change. Using default: {gene_count}")
            gene_names = [f"gene_{i}" for i in range(gene_count)]
        
        # Extract Kompot data
        df_data = {"gene": gene_names}
        
        # Add fold change if available
        if hasattr(kompot_results, 'fold_change') and kompot_results.fold_change is not None:
            mean_fold_change = np.mean(kompot_results.fold_change, axis=0)
            df_data["log2FoldChange"] = mean_fold_change
            # Use regular fold change for weighted fold change (which is no longer directly available)
            df_data["weighted_fold_change"] = mean_fold_change
        else:
            df_data["log2FoldChange"] = np.zeros(len(gene_names))
            df_data["weighted_fold_change"] = np.zeros(len(gene_names))
            
        # Add z-scores if available
        if hasattr(kompot_results, 'fold_change_zscores') and kompot_results.fold_change_zscores is not None:
            df_data["z_score"] = np.mean(kompot_results.fold_change_zscores, axis=0)
        else:
            df_data["z_score"] = np.zeros(len(gene_names))
            
        # Add Mahalanobis distances if available
        if hasattr(kompot_results, 'mahalanobis_distances') and kompot_results.mahalanobis_distances is not None:
            df_data["mahalanobis_distance"] = kompot_results.mahalanobis_distances
        else:
            df_data["mahalanobis_distance"] = np.zeros(len(gene_names))
            
        kompot_df = pd.DataFrame(df_data)
        
        # Store the comparison data
        self.comparison_data.append({
            "name": comparison_name,
            "kompot": kompot_df,
            "others": other_results
        })
    
    def _prepare_gene_data(self, diff_expr_data):
        """Prepare gene expression data for the report."""
        result = []
        
        for data in diff_expr_data:
            df = data["results_df"].copy()
            top_genes = df.head(data["top_n"])
            
            # Convert to dict format for JSON serialization
            genes_dict = {
                "condition1": data["condition1"],
                "condition2": data["condition2"],
                "genes": top_genes.to_dict(orient="records")
            }
            
            result.append(genes_dict)
        
        return result
    
    def _prepare_umap_data(self):
        """Prepare UMAP data from AnnData object."""
        if self.anndata_info is None:
            return None
        
        adata = self.anndata_info["adata"]
        embedding_key = self.anndata_info["embedding_key"]
        groupby = self.anndata_info["groupby"]
        annotations = self.anndata_info["cell_annotations"]
        
        # Extract UMAP coordinates
        umap_coords = adata.obsm[embedding_key]
        
        # For each annotation, create a category-to-index mapping
        annotation_data = {}
        for anno in annotations:
            if isinstance(adata.obs[anno].dtype, pd.CategoricalDtype):
                categories = adata.obs[anno].cat.categories.tolist()
                values = adata.obs[anno].cat.codes.astype(int).tolist()
            else:
                categories = sorted(adata.obs[anno].unique().tolist())
                cat_map = {cat: i for i, cat in enumerate(categories)}
                values = [cat_map[val] for val in adata.obs[anno]]
            
            annotation_data[anno] = {
                "categories": categories,
                "values": values
            }
        
        # Create a dictionary with all UMAP data
        umap_data = {
            "coordinates": umap_coords.tolist(),
            "annotations": annotation_data,
            "default_annotation": groupby
        }
        
        return umap_data
    
    def _prepare_comparison_data(self):
        """Prepare method comparison data."""
        if not self.comparison_data:
            return None
        
        result = []
        
        for comp in self.comparison_data:
            kompot_df = comp["kompot"]
            kompot_genes = set(kompot_df["gene"])
            
            # Process each comparison method
            method_data = {}
            for method, df in comp["others"].items():
                # Ensure gene column exists
                if "gene" not in df.columns:
                    df = df.copy()
                    df["gene"] = kompot_df["gene"].values
                
                # Convert values to Python types for JSON serialization
                method_data[method] = {}
                for col in df.columns:
                    if col == "gene":
                        continue
                    values = df[col].values
                    if isinstance(values, np.ndarray):
                        values = values.tolist()
                    method_data[method][col] = {
                        "values": values,
                        "genes": df["gene"].tolist()
                    }
            
            # Create comparison dictionary
            kompot_data = {
                "log2FoldChange": kompot_df["log2FoldChange"].tolist(),
                "z_score": kompot_df["z_score"].tolist(),
                "mahalanobis_distance": kompot_df["mahalanobis_distance"].tolist(),
                "genes": kompot_df["gene"].tolist()
            }
            
            # Add weighted_fold_change (using regular fold change)
            kompot_data["weighted_fold_change"] = kompot_df["log2FoldChange"].tolist()
            
            comp_dict = {
                "name": comp["name"],
                "kompot": kompot_data,
                "methods": method_data
            }
            
            result.append(comp_dict)
        
        return result
    
    def _prepare_gene_specific_plots(self):
        """Prepare gene-specific plot data."""
        if not self.diff_expr_data or not hasattr(self.diff_expr_data[0]["diff_expr"], "condition1_imputed"):
            return None
        
        # We'll prepare data for the top N genes across all differential expression analyses
        gene_data = {}
        
        for data in self.diff_expr_data:
            diff_expr = data["diff_expr"]
            df = data["results_df"]
            top_genes = df.head(data["top_n"])["gene"].tolist()
            
            # For each top gene, extract imputed values
            for gene_idx, gene_name in enumerate(top_genes):
                if gene_name not in gene_data:
                    gene_data[gene_name] = {}
                
                # Create a key for this comparison
                comp_key = f"{data['condition1']}_vs_{data['condition2']}"
                
                # Extract the imputed values for this gene
                if (hasattr(diff_expr, "condition1_imputed") and diff_expr.condition1_imputed is not None and 
                    hasattr(diff_expr, "condition2_imputed") and diff_expr.condition2_imputed is not None):
                    
                    # Check dimensions and bounds
                    if (len(diff_expr.condition1_imputed.shape) > 1 and 
                        gene_idx < diff_expr.condition1_imputed.shape[1]):
                        cond1_values = diff_expr.condition1_imputed[:, gene_idx].tolist()
                    else:
                        cond1_values = []
                    
                    if (len(diff_expr.condition2_imputed.shape) > 1 and 
                        gene_idx < diff_expr.condition2_imputed.shape[1]):
                        cond2_values = diff_expr.condition2_imputed[:, gene_idx].tolist()
                    else:
                        cond2_values = []
                else:
                    # If imputed values are not available
                    cond1_values = []
                    cond2_values = []
                
                # Add to the gene data dictionary
                gene_data[gene_name][comp_key] = {
                    "condition1": {
                        "name": data["condition1"],
                        "values": cond1_values
                    },
                    "condition2": {
                        "name": data["condition2"],
                        "values": cond2_values
                    }
                }
        
        return gene_data
    
    def _copy_template_files(self):
        """Copy template files to the output directory."""
        template_dir = Path(self.template_dir)
        output_dir = Path(self.output_dir)
        
        # Create necessary directories
        os.makedirs(output_dir / "js", exist_ok=True)
        os.makedirs(output_dir / "css", exist_ok=True)
        
        # Copy JavaScript files
        js_dir = template_dir / "js"
        for js_file in os.listdir(js_dir):
            shutil.copy(js_dir / js_file, output_dir / "js" / js_file)
        
        # Copy CSS files
        css_dir = template_dir / "css"
        for css_file in os.listdir(css_dir):
            shutil.copy(css_dir / css_file, output_dir / "css" / css_file)
    
    def _generate_html(self):
        """Generate the HTML file for the report."""
        template_path = Path(self.template_dir) / "base.html"
        
        # If the template doesn't exist, create a simple one
        if not template_path.exists():
            html_content = self._generate_default_template()
        else:
            with open(template_path, "r") as f:
                html_content = f.read()
        
        # Replace template variables
        html_content = html_content.replace("{{TITLE}}", self.title)
        html_content = html_content.replace("{{SUBTITLE}}", self.subtitle)
        html_content = html_content.replace("{{REPORT_ID}}", self.report_id)
        html_content = html_content.replace("{{TIMESTAMP}}", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        # Replace CDN links if requested
        if self.use_cdn:
            html_content = html_content.replace(
                '<script src="js/plotly.min.js"></script>',
                '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>'
            )
            html_content = html_content.replace(
                '<script src="js/datatables.min.js"></script>',
                '<script src="https://cdn.datatables.net/1.10.22/js/jquery.dataTables.min.js"></script>'
            )
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Write the HTML file
        output_path = Path(self.output_dir) / "index.html"
        with open(output_path, "w") as f:
            f.write(html_content)
        
        return str(output_path)
    
    def _generate_default_template(self):
        """Generate a default HTML template."""
        return """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{{TITLE}}</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="stylesheet" href="css/styles.css">
    <script src="js/plotly.min.js"></script>
    <script src="js/datatables.min.js"></script>
    <script src="js/kompot_report.js"></script>
</head>
<body>
    <header>
        <h1>{{TITLE}}</h1>
        <p>{{SUBTITLE}}</p>
        <p class="timestamp">Generated: {{TIMESTAMP}}</p>
    </header>
    
    <main>
        <section id="overview">
            <h2>Overview</h2>
            <div id="summary-stats"></div>
        </section>
        
        <section id="gene-table">
            <h2>Top Differentially Expressed Genes</h2>
            <div class="controls">
                <label for="condition-select">Comparison:</label>
                <select id="condition-select"></select>
            </div>
            <div id="gene-table-container"></div>
        </section>
        
        <section id="volcano-plot">
            <h2>Volcano Plot</h2>
            <div class="controls">
                <label for="volcano-x-metric">X-axis:</label>
                <select id="volcano-x-metric">
                    <option value="log2FoldChange">Log2 Fold Change</option>
                    <option value="weighted_fold_change">Weighted Fold Change</option>
                </select>
                <label for="volcano-y-metric">Y-axis:</label>
                <select id="volcano-y-metric">
                    <option value="mahalanobis_distance">Mahalanobis Distance</option>
                    <option value="z_score">Z-score</option>
                </select>
            </div>
            <div id="volcano-plot-container"></div>
        </section>
        
        <section id="umap-plots" style="display: none;">
            <h2>Cell Annotations</h2>
            <div class="controls">
                <label for="umap-annotation">Color by:</label>
                <select id="umap-annotation"></select>
            </div>
            <div id="umap-plot-container"></div>
        </section>
        
        <section id="gene-specific" style="display: none;">
            <h2>Gene-Specific Visualization</h2>
            <div class="controls">
                <label for="gene-search">Search for gene:</label>
                <input type="text" id="gene-search" placeholder="Gene name">
            </div>
            <div id="gene-plots-container"></div>
        </section>
        
        <section id="method-comparison" style="display: none;">
            <h2>Method Comparison</h2>
            <div class="controls">
                <label for="comparison-select">Comparison:</label>
                <select id="comparison-select"></select>
                <label for="method-select">Method:</label>
                <select id="method-select"></select>
            </div>
            <div id="method-comparison-container"></div>
        </section>
    </main>
    
    <footer>
        <p>Generated with Kompot v0.1.0</p>
        <p>Report ID: {{REPORT_ID}}</p>
    </footer>
    
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            initReport('{{REPORT_ID}}');
        });
    </script>
</body>
</html>
"""
    
    def generate(self, open_browser: bool = True):
        """Generate the HTML report.
        
        Args:
            open_browser: Whether to open the browser with the report
            
        Returns:
            The path to the generated report
        """
        logger.info(f"Generating HTML report in directory: {self.output_dir}")
        
        # Create output directory structure
        logger.debug(f"Creating output directory structure: {self.output_dir}")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "data"), exist_ok=True)
        
        # Process all the data and generate JSON files
        data_dir = os.path.join(self.output_dir, "data")
        logger.debug(f"Data directory for JSON files: {data_dir}")
        
        # 1. Gene table data
        logger.info("Preparing gene expression data for the report")
        gene_data = self._prepare_gene_data(self.diff_expr_data)
        gene_data_path = os.path.join(data_dir, "gene_data.json")
        with open(gene_data_path, "w") as f:
            json.dump(gene_data, f)
        logger.debug(f"Saved gene data to {gene_data_path}")
        
        # 2. UMAP data if provided
        if self.anndata_info is not None:
            logger.info("Preparing UMAP data for the report")
            umap_data = self._prepare_umap_data()
            umap_data_path = os.path.join(data_dir, "umap_data.json")
            with open(umap_data_path, "w") as f:
                json.dump(umap_data, f)
            logger.debug(f"Saved UMAP data to {umap_data_path}")
        else:
            logger.debug("No AnnData provided, skipping UMAP data preparation")
        
        # 3. Method comparison data if provided
        if self.comparison_data:
            logger.info("Preparing method comparison data for the report")
            comparison_data = self._prepare_comparison_data()
            comparison_data_path = os.path.join(data_dir, "comparison_data.json")
            with open(comparison_data_path, "w") as f:
                json.dump(comparison_data, f)
            logger.debug(f"Saved comparison data to {comparison_data_path}")
        else:
            logger.debug("No comparison data provided, skipping comparison data preparation")
        
        # 4. Gene-specific plot data
        logger.info("Preparing gene-specific plot data for the report")
        gene_plot_data = self._prepare_gene_specific_plots()
        if gene_plot_data:
            gene_plots_path = os.path.join(data_dir, "gene_plots.json")
            with open(gene_plots_path, "w") as f:
                json.dump(gene_plot_data, f)
            logger.debug(f"Saved gene-specific plot data to {gene_plots_path}")
        else:
            logger.debug("No gene-specific plot data available, skipping")
        
        # Copy template files (JS, CSS)
        logger.info("Copying template files (JS, CSS)")
        self._copy_template_files()
        
        # Generate HTML file
        logger.info("Generating HTML file")
        html_path = self._generate_html()
        
        # Open browser if requested
        if open_browser:
            logger.info(f"Opening browser with the report: {html_path}")
            webbrowser.open(f"file://{os.path.abspath(html_path)}")
        else:
            logger.debug("Skipping browser opening as requested")
        
        logger.info(f"Report generation complete: {html_path}")
        return html_path