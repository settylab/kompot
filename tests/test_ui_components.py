"""
Tests for UI components and JavaScript in the reporter module.

These tests focus on the client-side functionality of the report.
"""

import os
import json
import tempfile
import re
import pytest
from pathlib import Path

from kompot.reporter import HTMLReporter
from tests.test_reporter import generate_test_data


def extract_js_file_path(html_content):
    """Extract the path to the kompot_report.js file from the HTML content."""
    match = re.search(r'<script src="(js/kompot_report\.js)"></script>', html_content)
    if match:
        return match.group(1)
    return None


def test_javascript_file_creation():
    """Test that JavaScript files are created correctly."""
    diff_expr, gene_names = generate_test_data()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        reporter = HTMLReporter(output_dir=os.path.join(temp_dir, "report"))
        reporter.add_differential_expression(
            diff_expr,
            condition1_name="Control",
            condition2_name="Treatment",
            gene_names=gene_names
        )
        
        # Generate report
        report_path = reporter.generate(open_browser=False)
        
        # Check that JS files exist
        js_dir = os.path.join(temp_dir, "report", "js")
        assert os.path.exists(js_dir)
        assert os.path.exists(os.path.join(js_dir, "kompot_report.js"))
        
        # Read the HTML file to find the JS file reference
        with open(report_path, 'r') as f:
            html_content = f.read()
        
        # Extract JS file path from HTML
        js_file_path = extract_js_file_path(html_content)
        assert js_file_path is not None
        
        # Read the kompot_report.js file
        with open(os.path.join(temp_dir, "report", js_file_path), 'r') as f:
            js_content = f.read()
        
        # Check for key JavaScript functions
        assert "function initReport" in js_content
        assert "function loadGeneTable" in js_content
        assert "function initVolcanoPlot" in js_content


def test_javascript_initialization_parameters():
    """Test that JavaScript contains correct initialization parameters."""
    diff_expr, gene_names = generate_test_data()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        reporter = HTMLReporter(
            output_dir=os.path.join(temp_dir, "report"),
            title="Test Title",
            subtitle="Test Subtitle"
        )
        reporter.add_differential_expression(
            diff_expr,
            condition1_name="Control",
            condition2_name="Treatment",
            gene_names=gene_names
        )
        
        # Generate report
        report_path = reporter.generate(open_browser=False)
        
        # Read the HTML file to check initialization
        with open(report_path, 'r') as f:
            html_content = f.read()
        
        # Check title and subtitle
        assert "Test Title" in html_content
        assert "Test Subtitle" in html_content
        
        # Check report ID is passed to JavaScript
        report_id = reporter.report_id
        assert f"initReport('{report_id}')" in html_content


def test_json_data_format():
    """Test format of the JSON data files."""
    diff_expr, gene_names = generate_test_data()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        reporter = HTMLReporter(output_dir=os.path.join(temp_dir, "report"))
        reporter.add_differential_expression(
            diff_expr,
            condition1_name="Control",
            condition2_name="Treatment",
            gene_names=gene_names
        )
        
        # Generate report
        report_path = reporter.generate(open_browser=False)
        
        # Check gene data JSON format
        gene_data_path = os.path.join(temp_dir, "report", "data", "gene_data.json")
        with open(gene_data_path, 'r') as f:
            gene_data = json.load(f)
        
        # Verify it's valid JSON array
        assert isinstance(gene_data, list)
        assert len(gene_data) == 1
        
        # Check first comparison
        comparison = gene_data[0]
        assert "condition1" in comparison
        assert "condition2" in comparison
        assert "genes" in comparison
        assert isinstance(comparison["genes"], list)
        
        # Check gene format
        first_gene = comparison["genes"][0]
        assert "gene" in first_gene
        assert "log2FoldChange" in first_gene
        assert "mahalanobis_distance" in first_gene


def test_html_ui_elements():
    """Test that HTML contains all required UI elements."""
    diff_expr, gene_names = generate_test_data()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        reporter = HTMLReporter(output_dir=os.path.join(temp_dir, "report"))
        reporter.add_differential_expression(
            diff_expr,
            condition1_name="Control",
            condition2_name="Treatment",
            gene_names=gene_names
        )
        
        # Generate report
        report_path = reporter.generate(open_browser=False)
        
        # Read the HTML file
        with open(report_path, 'r') as f:
            html_content = f.read()
        
        # Check for UI sections
        assert '<section id="overview">' in html_content
        assert '<section id="gene-table">' in html_content
        assert '<section id="volcano-plot">' in html_content
        
        # Check controls
        assert '<select id="condition-select">' in html_content
        assert '<select id="volcano-x-metric">' in html_content
        assert '<select id="volcano-y-metric">' in html_content
        
        # Check containers
        assert '<div id="gene-table-container">' in html_content
        assert '<div id="volcano-plot-container">' in html_content


def test_css_file_creation():
    """Test that CSS files are created correctly."""
    diff_expr, gene_names = generate_test_data()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        reporter = HTMLReporter(output_dir=os.path.join(temp_dir, "report"))
        reporter.add_differential_expression(
            diff_expr,
            condition1_name="Control",
            condition2_name="Treatment",
            gene_names=gene_names
        )
        
        # Generate report
        report_path = reporter.generate(open_browser=False)
        
        # Check that CSS files exist
        css_dir = os.path.join(temp_dir, "report", "css")
        assert os.path.exists(css_dir)
        assert os.path.exists(os.path.join(css_dir, "styles.css"))
        
        # Read the HTML file to find the CSS file reference
        with open(report_path, 'r') as f:
            html_content = f.read()
        
        # Check CSS reference
        assert '<link rel="stylesheet" href="css/styles.css">' in html_content
        
        # Read the CSS file
        with open(os.path.join(css_dir, "styles.css"), 'r') as f:
            css_content = f.read()
        
        # Check for CSS styling
        assert "body" in css_content
        assert "header" in css_content
        assert "footer" in css_content