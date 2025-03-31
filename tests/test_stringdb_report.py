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


def test_get_enrichment_url():
    """Test generation of enrichment URL."""
    report = StringDBReport(HUMAN_GENES)
    
    # Test default category (Process)
    url = report.get_enrichment_url()
    assert "cgi/network.pl" in url
    assert "identifiers=" in url
    assert f"species={report.species_id}" in url
    assert "#enrichment" in url  # Anchor
    
    # Test specific category
    kegg_url = report.get_enrichment_url(category="KEGG")
    assert "cgi/network.pl" in kegg_url
    assert "identifiers=" in kegg_url
    assert f"species={report.species_id}" in kegg_url
    
    # Test invalid category (should default to Process)
    invalid_url = report.get_enrichment_url(category="InvalidCategory")
    assert "cgi/network.pl" in invalid_url
    assert "identifiers=" in invalid_url
    assert f"species={report.species_id}" in invalid_url


@patch('requests.post')
def test_get_functional_enrichment(mock_post):
    """Test functional enrichment API call."""
    # Mock a successful response with sample enrichment data
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    
    # Sample enrichment data for multiple categories
    sample_data = [
        {
            "category": "Process",
            "description": "DNA repair",
            "fdr": 0.001,
            "term": "GO:0006281",
            "number_of_genes": 3,
            "inputGenes": "TP53,BRCA1,PTEN"
        },
        {
            "category": "Process",
            "description": "Cell cycle regulation",
            "fdr": 0.01,
            "term": "GO:0007049",
            "number_of_genes": 2,
            "inputGenes": "TP53,BRCA1"
        },
        {
            "category": "KEGG",
            "description": "p53 signaling pathway",
            "fdr": 0.005,
            "term": "hsa04115",
            "number_of_genes": 3,
            "inputGenes": "TP53,CDKN1A,MDM2"
        }
    ]
    
    mock_response.json.return_value = sample_data
    mock_post.return_value = mock_response
    
    report = StringDBReport(HUMAN_GENES, include_enrichment=True)
    
    # Test Process category
    process_df = report.get_functional_enrichment(category="Process")
    
    # Verify API call
    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert "enrichment" in args[0]
    assert kwargs["data"]["species"] == report.species_id
    assert "identifiers" in kwargs["data"]
    # We no longer send enrichment_category parameter since we filter locally
    assert "enrichment_category" not in kwargs["data"]
    
    # Verify Process result processing
    assert isinstance(process_df, pd.DataFrame)
    assert len(process_df) == 2
    assert "fdr" in process_df.columns
    assert "term" in process_df.columns
    assert "description" in process_df.columns
    
    # Reset mock and test KEGG category 
    mock_post.reset_mock()
    mock_post.return_value = mock_response  # Same mock response with multiple categories
    
    kegg_df = report.get_functional_enrichment(category="KEGG")
    
    # Verify KEGG result processing
    assert isinstance(kegg_df, pd.DataFrame)
    assert len(kegg_df) == 1
    assert kegg_df.iloc[0]["description"] == "p53 signaling pathway"
    assert kegg_df.iloc[0]["category"] == "KEGG"


@patch('requests.post')
def test_get_interaction_partners(mock_post):
    """Test interaction partners API call."""
    # Mock a successful response with sample network data
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    
    # Sample network data
    sample_data = {
        "nodes": [
            {"name": "TP53", "id": "9606.ENSP00000269305"},
            {"name": "BRCA1", "id": "9606.ENSP00000350283"},
            {"name": "MDM2", "id": "9606.ENSP00000258149"},  # Partner
            {"name": "CDKN1A", "id": "9606.ENSP00000244741"}  # Partner
        ],
        "edges": [
            {"from": "TP53", "to": "MDM2", "score": 0.999},
            {"from": "TP53", "to": "CDKN1A", "score": 0.95},
            {"from": "BRCA1", "to": "TP53", "score": 0.85}
        ]
    }
    
    mock_response.json.return_value = sample_data
    mock_post.return_value = mock_response
    
    report = StringDBReport(["TP53", "BRCA1"])
    df = report.get_interaction_partners()
    
    # Verify API call
    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert "network" in args[0]
    assert kwargs["data"]["species"] == report.species_id
    assert "identifiers" in kwargs["data"]
    
    # Verify result processing
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2  # Two partners
    assert "MDM2" in df["name"].values
    assert "CDKN1A" in df["name"].values


@patch('requests.post')
def test_get_tissue_expression(mock_post):
    """Test tissue expression API call."""
    # Mock a successful response with sample tissue data
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    
    # Since we now use TSV format, set the text property instead of json
    tsv_content = "protein\ttissue\tscore\nTP53\tbrain\t0.8\nTP53\tliver\t0.6\nTP53\tkidney\t0.4\nBRCA1\tbrain\t0.2\nBRCA1\tliver\t0.9\nBRCA1\tkidney\t0.5"
    mock_response.text = tsv_content
    mock_post.return_value = mock_response
    
    # Add status code property to avoid 404 condition
    mock_response.status_code = 200
    
    report = StringDBReport(["TP53", "BRCA1"])
    df = report.get_tissue_expression()
    
    # Verify API call
    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert "tissueExpression" in args[0]
    assert kwargs["data"]["species"] == report.species_id
    assert "identifiers" in kwargs["data"]
    
    # Verify result processing
    assert isinstance(df, pd.DataFrame)
    
    # Should be a pivoted dataframe with tissues as index
    assert df.index.name == 'tissue'
    assert "TP53" in df.columns
    assert "BRCA1" in df.columns
    assert "brain" in df.index
    assert "liver" in df.index
    assert "kidney" in df.index
    
    # Check correct values are in pivoted dataframe
    assert df.loc["brain", "TP53"] == 0.8
    assert df.loc["liver", "BRCA1"] == 0.9
    
    # Test 404 response handling
    mock_response.status_code = 404
    mock_post.return_value = mock_response
    
    # Should return None for 404 response
    assert report.get_tissue_expression() is None


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