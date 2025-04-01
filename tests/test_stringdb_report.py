"""
Unit tests for the StringDBReport class.
"""

import pytest
import os
import json
import pandas as pd
from unittest.mock import patch, MagicMock
from kompot.plot import StringDBReport

# Define test gene lists
HUMAN_GENES = ["TP53", "BRCA1", "KRAS", "EGFR", "PTEN"]
MOUSE_GENES = ["Trp53", "Brca1", "Kras", "Egfr", "Pten"]


def test_stringdb_report_init():
    """Test initialization of StringDBReport with default parameters."""
    report = StringDBReport(HUMAN_GENES)
    assert report.genes == HUMAN_GENES
    assert report.species_id == 9606
    assert report.include_stringdb is True
    assert report.include_resources is True
    assert "https://string-db.org" in report.string_db_base_url


def test_stringdb_report_custom_init():
    """Test initialization of StringDBReport with custom parameters."""
    report = StringDBReport(
        genes=MOUSE_GENES,
        species_id=10090,
        include_stringdb=False,
        include_resources=False,
        include_enrichment=True,
    )
    assert report.genes == MOUSE_GENES
    assert report.species_id == 10090
    assert report.include_stringdb is False
    assert report.include_resources is False
    assert report.include_enrichment is True
    
    # Check that annotation categories are correctly initialized
    assert 'Process' in report.annotation_categories
    assert 'KEGG' in report.annotation_categories
    assert 'Reactome' in report.annotation_categories


def test_get_species_name():
    """Test getting species name from ID."""
    human_report = StringDBReport(HUMAN_GENES)
    assert human_report.get_species_name() == "Homo sapiens"
    
    mouse_report = StringDBReport(HUMAN_GENES, species_id=10090)
    assert mouse_report.get_species_name() == "Mus musculus"
    
    # Test unknown species ID
    unknown_report = StringDBReport(HUMAN_GENES, species_id=12345)
    assert unknown_report.get_species_name() == "Species ID: 12345"


def test_get_stringdb_url():
    """Test generation of StringDB network URL."""
    report = StringDBReport(HUMAN_GENES)
    url = report.get_stringdb_url()
    
    # Check basic URL structure
    assert "https://string-db.org/cgi/network" in url
    
    # Check that all genes are included
    for gene in HUMAN_GENES:
        assert gene in url
    
    # Check species ID is in the URL
    assert f"species=9606" in url
    
    # Test with additional genes
    additional_genes = ["MDM2", "CDKN1A"]
    url_with_additional = report.get_stringdb_url(additional_genes)
    
    for gene in additional_genes:
        assert gene in url_with_additional


def test_get_stringdb_image_url():
    """Test generation of StringDB image URL."""
    report = StringDBReport(HUMAN_GENES)
    url = report.get_stringdb_image_url()
    
    # Check basic URL structure
    assert "https://string-db.org/api/image/network" in url
    
    # Check that all genes are included
    for gene in HUMAN_GENES:
        assert gene in url
    
    # Check species ID is in the URL
    assert f"species=9606" in url


def test_get_resource_links():
    """Test generation of resource links for a gene."""
    report = StringDBReport(HUMAN_GENES)
    links = report.get_resource_links("TP53")
    
    # Check common resources for all species
    assert "STRING DB" in links
    assert "BioGRID" in links
    assert "Reactome" in links
    assert "GeneCards" in links
    
    # Check human-specific resources
    assert "UniProt" in links
    assert "NCBI Gene" in links
    
    # Test mouse-specific resources
    mouse_report = StringDBReport(MOUSE_GENES, species_id=10090)
    mouse_links = mouse_report.get_resource_links("Trp53")
    
    assert "MGI" in mouse_links


def test_fetch_stringdb_image():
    """Test fetching StringDB network image."""
    report = StringDBReport(HUMAN_GENES)
    
    # Create a mock for the _make_request method
    original_make_request = report._make_request
    
    try:
        # Test successful request
        report._make_request = lambda url, timeout=10: b'fake_image_data'
        image_data = report.fetch_stringdb_image()
        assert image_data == b'fake_image_data'
        
        # Test failed request
        report._make_request = lambda url, timeout=10: None
        image_data = report.fetch_stringdb_image()
        assert image_data is None
    finally:
        # Restore the original method
        report._make_request = original_make_request


def test_to_html():
    """Test HTML generation."""
    report = StringDBReport(HUMAN_GENES)
    html = report.to_html()
    
    # Check basic HTML structure
    assert "<h3>Gene Set Report" in html
    assert f"Species:</strong> {report.get_species_name()}" in html
    
    # Check StringDB section exists
    assert "<h4>StringDB Network</h4>" in html
    assert "View interactive network in StringDB" in html
    
    # Check resource links section exists (now uses collapsible details)
    assert "Resource Links" in html
    assert "<details>" in html
    assert "<table " in html
    
    # Check genes are included in resource links table
    for gene in HUMAN_GENES:
        assert f"<td style=\"text-align:left;\">{gene}</td>" in html


def test_to_dataframe():
    """Test conversion to DataFrame."""
    report = StringDBReport(HUMAN_GENES)
    df = report.to_dataframe()
    
    # Check DataFrame structure
    assert list(df["Gene"]) == HUMAN_GENES
    
    # Check common resource columns
    assert "STRING DB" in df.columns
    assert "BioGRID" in df.columns
    assert "Reactome" in df.columns
    
    # Check values for TP53
    tp53_row = df[df["Gene"] == "TP53"].iloc[0]
    assert "string-db.org" in tp53_row["STRING DB"]
    assert "biogrid.org" in tp53_row["BioGRID"]


def test_get_json():
    """Test JSON representation."""
    report = StringDBReport(HUMAN_GENES)
    data = report.get_json()
    
    # Check JSON structure
    assert data["genes"] == HUMAN_GENES
    assert data["species_id"] == 9606
    assert data["species_name"] == "Homo sapiens"
    
    # Check StringDB section
    assert "stringdb" in data
    assert "url" in data["stringdb"]
    assert "image_url" in data["stringdb"]
    
    # Check resources section
    assert "resources" in data
    assert len(data["resources"]) == len(HUMAN_GENES)
    assert "TP53" in data["resources"]
    assert "STRING DB" in data["resources"]["TP53"]
    
    # Test with enrichment enabled
    enriched_report = StringDBReport(HUMAN_GENES, include_enrichment=True)
    
    # Mock the enrichment method to return sample data
    original_method = enriched_report.get_functional_enrichment
    try:
        # Create a sample enrichment dataframe
        sample_df = pd.DataFrame({
            'term': ['GO:0006281', 'GO:0007049'],
            'description': ['DNA repair', 'Cell cycle regulation'],
            'fdr': [0.001, 0.01]
        })
        
        enriched_report.get_functional_enrichment = lambda **kwargs: sample_df
        
        # Get enriched JSON data
        enriched_data = enriched_report.get_json()
        
        # Check if enrichment data is included
        assert "enrichment" in enriched_data
        assert len(enriched_data["enrichment"]) == 2
        assert enriched_data["enrichment"][0]["description"] == "DNA repair"
    finally:
        # Restore original method
        enriched_report.get_functional_enrichment = original_method
