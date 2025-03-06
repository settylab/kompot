"""
Tests for the Kompot HTML reporter module.
"""

import os
import json
import tempfile
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from kompot.differential import DifferentialExpression, DifferentialAbundance
from kompot.reporter import HTMLReporter
import kompot


def generate_test_data(n_cells=100, n_genes=20, n_landmarks=10):
    """Generate synthetic data for testing."""
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
    
    # Run prediction on abundance to compute fold changes
    X_combined = np.vstack([X_condition1, X_condition2])
    abundance_results = diff_abundance.predict(X_combined)
    
    diff_expression = DifferentialExpression(
        n_landmarks=n_landmarks,
        differential_abundance=diff_abundance
    )
    diff_expression.fit(X_condition1, y_condition1, X_condition2, y_condition2)
    
    # Run prediction to compute fold changes and other metrics
    expression_results = diff_expression.predict(X_combined, compute_mahalanobis=True)
    
    return diff_expression, gene_names


class TestHTMLReporter:
    """Tests for the HTMLReporter class."""
    
    def setup_method(self):
        """Set up test data."""
        self.diff_expr, self.gene_names = generate_test_data()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name) / "test_report"
    
    def teardown_method(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()
    
    def test_init(self):
        """Test initialization of HTMLReporter."""
        reporter = HTMLReporter(
            output_dir=str(self.output_dir),
            title="Test Report",
            subtitle="Test Subtitle"
        )
        
        assert reporter.title == "Test Report"
        assert reporter.subtitle == "Test Subtitle"
        assert reporter.output_dir == str(self.output_dir)
        assert len(reporter.diff_expr_data) == 0
        assert reporter.anndata_info is None
        assert len(reporter.comparison_data) == 0
    
    def test_add_differential_expression(self):
        """Test adding differential expression data."""
        reporter = HTMLReporter(output_dir=str(self.output_dir))
        reporter.add_differential_expression(
            self.diff_expr,
            condition1_name="Control",
            condition2_name="Treatment",
            gene_names=self.gene_names
        )
        
        assert len(reporter.diff_expr_data) == 1
        data = reporter.diff_expr_data[0]
        assert data["condition1"] == "Control"
        assert data["condition2"] == "Treatment"
        assert data["diff_expr"] == self.diff_expr
        assert len(data["results_df"]) == len(self.gene_names)
        assert set(data["results_df"]["gene"].tolist()) == set(self.gene_names)
    
    def test_add_comparison(self):
        """Test adding method comparison data."""
        reporter = HTMLReporter(output_dir=str(self.output_dir))
        
        # Create a mock comparison method
        other_results = pd.DataFrame({
            "gene": self.gene_names,
            "log2FoldChange": np.random.normal(0, 1, len(self.gene_names)),
            "pvalue": np.random.uniform(0, 1, len(self.gene_names))
        })
        
        reporter.add_comparison(
            self.diff_expr,
            {"OtherMethod": other_results},
            gene_names=self.gene_names,
            comparison_name="Test Comparison"
        )
        
        assert len(reporter.comparison_data) == 1
        data = reporter.comparison_data[0]
        assert data["name"] == "Test Comparison"
        assert "OtherMethod" in data["others"]
        assert len(data["kompot"]) == len(self.gene_names)
    
    def test_prepare_gene_data(self):
        """Test preparation of gene data for report."""
        reporter = HTMLReporter(output_dir=str(self.output_dir))
        reporter.add_differential_expression(
            self.diff_expr,
            condition1_name="Control",
            condition2_name="Treatment",
            gene_names=self.gene_names,
            top_n=10
        )
        
        gene_data = reporter._prepare_gene_data(reporter.diff_expr_data)
        
        assert len(gene_data) == 1
        assert gene_data[0]["condition1"] == "Control"
        assert gene_data[0]["condition2"] == "Treatment"
        assert len(gene_data[0]["genes"]) == 10  # Should be top 10
    
    def test_prepare_comparison_data(self):
        """Test preparation of comparison data."""
        reporter = HTMLReporter(output_dir=str(self.output_dir))
        
        # Create a mock comparison method
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
        
        comparison_data = reporter._prepare_comparison_data()
        
        assert len(comparison_data) == 1
        assert "kompot" in comparison_data[0]
        assert "methods" in comparison_data[0]
        assert "OtherMethod" in comparison_data[0]["methods"]
        assert "log2FoldChange" in comparison_data[0]["methods"]["OtherMethod"]
        assert "pvalue" in comparison_data[0]["methods"]["OtherMethod"]
    
    def test_generate_html(self):
        """Test HTML generation."""
        reporter = HTMLReporter(output_dir=str(self.output_dir))
        reporter.add_differential_expression(
            self.diff_expr,
            condition1_name="Control",
            condition2_name="Treatment",
            gene_names=self.gene_names
        )
        
        html_path = reporter._generate_html()
        
        assert os.path.exists(html_path)
        with open(html_path, 'r') as f:
            html_content = f.read()
        
        assert reporter.title in html_content
        assert reporter.report_id in html_content
        assert 'initReport' in html_content
    
    def test_generate_full_report(self):
        """Test generating the full report."""
        reporter = HTMLReporter(output_dir=str(self.output_dir))
        reporter.add_differential_expression(
            self.diff_expr,
            condition1_name="Control",
            condition2_name="Treatment",
            gene_names=self.gene_names
        )
        
        # Create a mock comparison
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
        
        report_path = reporter.generate(open_browser=False)
        
        assert os.path.exists(report_path)
        assert os.path.exists(os.path.join(self.output_dir, "data"))
        assert os.path.exists(os.path.join(self.output_dir, "data", "gene_data.json"))
        assert os.path.exists(os.path.join(self.output_dir, "data", "comparison_data.json"))
        
        # Test gene data JSON
        with open(os.path.join(self.output_dir, "data", "gene_data.json"), 'r') as f:
            gene_data = json.load(f)
        
        assert len(gene_data) == 1
        assert gene_data[0]["condition1"] == "Control"
        assert gene_data[0]["condition2"] == "Treatment"
        assert "genes" in gene_data[0]
        
        # Test comparison data JSON
        with open(os.path.join(self.output_dir, "data", "comparison_data.json"), 'r') as f:
            comparison_data = json.load(f)
        
        assert len(comparison_data) == 1
        assert "kompot" in comparison_data[0]
        assert "methods" in comparison_data[0]
        assert "OtherMethod" in comparison_data[0]["methods"]


def test_generate_report_function():
    """Test the generate_report convenience function."""
    diff_expr, gene_names = generate_test_data()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        report_path = kompot.generate_report(
            diff_expr,
            output_dir=os.path.join(temp_dir, "report"),
            condition1_name="Control",
            condition2_name="Treatment",
            gene_names=gene_names,
            open_browser=False
        )
        
        assert os.path.exists(report_path)
        assert os.path.exists(os.path.join(temp_dir, "report", "data", "gene_data.json"))


# Test with pytest.mark.parametrize to check different configurations
@pytest.mark.parametrize(
    "title,subtitle,use_cdn",
    [
        ("Test Report", None, False),
        ("Report with Subtitle", "This is a subtitle", False),
        ("Report with CDN", None, True),
    ]
)
def test_reporter_configurations(title, subtitle, use_cdn):
    """Test HTMLReporter with different configurations."""
    diff_expr, gene_names = generate_test_data()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        reporter = HTMLReporter(
            output_dir=os.path.join(temp_dir, "report"),
            title=title,
            subtitle=subtitle,
            use_cdn=use_cdn
        )
        
        reporter.add_differential_expression(
            diff_expr,
            condition1_name="Control",
            condition2_name="Treatment",
            gene_names=gene_names
        )
        
        report_path = reporter.generate(open_browser=False)
        
        assert os.path.exists(report_path)
        
        with open(report_path, 'r') as f:
            html_content = f.read()
        
        assert title in html_content
        if subtitle:
            assert subtitle in html_content
        
        if use_cdn:
            assert 'cdn.plot.ly' in html_content or 'cdn.datatables.net' in html_content
        else:
            assert 'js/plotly.min.js' in html_content