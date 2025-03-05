"""
Integration tests for the Kompot HTML reporter module.

These tests ensure all components work together correctly.
"""

import os
import json
import tempfile
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import webbrowser

from kompot.differential import DifferentialExpression, DifferentialAbundance
from kompot.reporter import HTMLReporter
import kompot


# Create a real AnnData for testing
def create_test_anndata(n_cells=100, n_genes=20):
    """Create a real AnnData object for testing."""
    import anndata
    np.random.seed(42)
    
    # Generate random data
    X = np.random.normal(0, 1, (n_cells, n_genes))
    
    # Create UMAP coordinates
    obsm = {
        "X_umap": np.random.normal(0, 1, (n_cells, 2))
    }
    
    # Create observation annotations
    categories = ["TypeA", "TypeB", "TypeC"]
    obs = pd.DataFrame({
        "cell_type": np.random.choice(categories, n_cells),
        "sample": np.random.choice(["sample1", "sample2"], n_cells),
        "condition": np.random.choice(["control", "treatment"], n_cells)
    })
    
    # Make categorical
    obs["cell_type"] = obs["cell_type"].astype("category")
    obs["sample"] = obs["sample"].astype("category")
    obs["condition"] = obs["condition"].astype("category")
    
    # Create AnnData object
    return anndata.AnnData(X=X, obs=obs, obsm=obsm)


def generate_test_data_with_anndata(n_cells=100, n_genes=20, n_landmarks=10):
    """Generate synthetic data for integration testing."""
    np.random.seed(42)
    
    # Create synthetic data
    X_condition1 = np.random.normal(0, 1, (n_cells, 2))
    X_condition2 = np.random.normal(0.5, 1, (n_cells, 2))
    
    # Simulate gene expression
    y_condition1 = np.random.normal(0, 1, (n_cells, n_genes))
    y_condition2 = np.random.normal(0, 1, (n_cells, n_genes))
    
    # Add some differential expression for a subset of genes
    diff_genes = np.random.choice(n_genes, 5, replace=False)
    y_condition2[:, diff_genes] += np.random.normal(2, 0.5, (n_cells, len(diff_genes)))
    
    # Generate gene names
    gene_names = [f"gene_{i}" for i in range(n_genes)]
    
    # Run Kompot analysis
    diff_abundance = DifferentialAbundance(n_landmarks=n_landmarks)
    diff_abundance.fit(X_condition1, X_condition2)
    
    diff_expression = DifferentialExpression(
        n_landmarks=n_landmarks,
        differential_abundance=diff_abundance
    )
    diff_expression.fit(X_condition1, y_condition1, X_condition2, y_condition2)
    
    # Create a real AnnData object
    adata = create_test_anndata(n_cells=n_cells, n_genes=n_genes)
    
    return diff_expression, gene_names, adata


# Mock the webbrowser.open function to avoid opening actual browser during tests
@pytest.fixture
def mock_webbrowser_open(monkeypatch):
    """Mock the webbrowser.open function."""
    def mock_open(*args, **kwargs):
        return True
    
    monkeypatch.setattr(webbrowser, "open", mock_open)


class TestReporterIntegration:
    """Integration tests for HTML reporter."""
    
    def setup_method(self):
        """Set up test data."""
        self.diff_expr, self.gene_names, self.adata = generate_test_data_with_anndata()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name) / "test_report"
    
    def teardown_method(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()
    
    def test_full_report_with_anndata(self, mock_webbrowser_open):
        """Test generating a full report with AnnData."""
        # Skip this test if anndata is not installed
        try:
            import anndata
        except ImportError:
            pytest.skip("anndata not installed, skipping test")
        
        reporter = HTMLReporter(output_dir=str(self.output_dir))
        
        # Add differential expression results
        reporter.add_differential_expression(
            self.diff_expr,
            condition1_name="Control",
            condition2_name="Treatment",
            gene_names=self.gene_names
        )
        
        # Add AnnData for cell visualization
        reporter.add_anndata(
            self.adata,
            groupby="cell_type",
            embedding_key="X_umap",
            cell_annotations=["cell_type", "sample", "condition"]
        )
        
        # Add method comparison
        other_results = pd.DataFrame({
            "gene": self.gene_names,
            "log2FoldChange": np.random.normal(0, 1, len(self.gene_names)),
            "pvalue": np.random.uniform(0, 1, len(self.gene_names))
        })
        
        reporter.add_comparison(
            self.diff_expr,
            {"OtherMethod": other_results},
            gene_names=self.gene_names
        )
        
        # Generate report
        report_path = reporter.generate(open_browser=True)
        
        # Check that report files exist
        assert os.path.exists(report_path)
        assert os.path.exists(os.path.join(self.output_dir, "data", "gene_data.json"))
        assert os.path.exists(os.path.join(self.output_dir, "data", "umap_data.json"))
        assert os.path.exists(os.path.join(self.output_dir, "data", "comparison_data.json"))
        
        # Verify UMAP data contents
        with open(os.path.join(self.output_dir, "data", "umap_data.json"), 'r') as f:
            umap_data = json.load(f)
        
        assert "coordinates" in umap_data
        assert "annotations" in umap_data
        assert "cell_type" in umap_data["annotations"]
        assert "sample" in umap_data["annotations"]
        assert "condition" in umap_data["annotations"]
        assert "default_annotation" in umap_data
        assert umap_data["default_annotation"] == "cell_type"
    
    def test_convenience_function_with_anndata(self, mock_webbrowser_open):
        """Test the generate_report convenience function with AnnData."""
        # Skip this test if anndata is not installed
        try:
            import anndata
        except ImportError:
            pytest.skip("anndata not installed, skipping test")
            
        report_path = kompot.generate_report(
            self.diff_expr,
            output_dir=os.path.join(self.temp_dir.name, "report"),
            adata=self.adata,
            condition1_name="Control",
            condition2_name="Treatment",
            gene_names=self.gene_names,
            groupby="cell_type",
            open_browser=True
        )
        
        assert os.path.exists(report_path)
        assert os.path.exists(os.path.join(self.temp_dir.name, "report", "data", "gene_data.json"))
        assert os.path.exists(os.path.join(self.temp_dir.name, "report", "data", "umap_data.json"))


def test_specific_gene_plots():
    """Test handling of gene-specific plot data."""
    # Create differential expression results with imputed values to test gene plots
    np.random.seed(42)
    n_cells, n_genes = 100, 20
    n_landmarks = 10
    
    # Create synthetic data
    X_condition1 = np.random.normal(0, 1, (n_cells, 2))
    X_condition2 = np.random.normal(0.5, 1, (n_cells, 2))
    
    # Simulate gene expression
    y_condition1 = np.random.normal(0, 1, (n_cells, n_genes))
    y_condition2 = np.random.normal(0, 1, (n_cells, n_genes))
    
    gene_names = [f"gene_{i}" for i in range(n_genes)]
    
    # Create differential expression analysis
    diff_abundance = DifferentialAbundance(n_landmarks=n_landmarks)
    diff_abundance.fit(X_condition1, X_condition2)
    
    diff_expr = DifferentialExpression(
        n_landmarks=n_landmarks,
        differential_abundance=diff_abundance
    )
    diff_expr.fit(X_condition1, y_condition1, X_condition2, y_condition2)
    
    # Manually add condition1_imputed and condition2_imputed attributes
    # for testing gene-specific plots
    diff_expr.condition1_imputed = y_condition1
    diff_expr.condition2_imputed = y_condition2
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create reporter
        reporter = HTMLReporter(output_dir=os.path.join(temp_dir, "report"))
        reporter.add_differential_expression(
            diff_expr,
            condition1_name="Control",
            condition2_name="Treatment",
            gene_names=gene_names
        )
        
        # Generate report
        report_path = reporter.generate(open_browser=False)
        
        # Check for gene plot data
        assert os.path.exists(os.path.join(temp_dir, "report", "data", "gene_plots.json"))
        
        # Verify gene plot data contents
        with open(os.path.join(temp_dir, "report", "data", "gene_plots.json"), 'r') as f:
            gene_plot_data = json.load(f)
        
        # Check the structure of gene plot data
        assert len(gene_plot_data) > 0
        
        # Check a specific gene
        first_gene = gene_names[0]
        assert first_gene in gene_plot_data
        assert "Control_vs_Treatment" in gene_plot_data[first_gene]
        assert "condition1" in gene_plot_data[first_gene]["Control_vs_Treatment"]
        assert "condition2" in gene_plot_data[first_gene]["Control_vs_Treatment"]
        assert "values" in gene_plot_data[first_gene]["Control_vs_Treatment"]["condition1"]
        assert "values" in gene_plot_data[first_gene]["Control_vs_Treatment"]["condition2"]