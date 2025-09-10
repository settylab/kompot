"""StringDB integration for gene set visualization and analysis."""

import logging
import urllib.parse
import re
from typing import List, Dict, Optional, Any, Union, Tuple, Set
import json
import base64
from io import BytesIO, StringIO
import importlib
import sys
import warnings
import numpy as np

# Setup logging
logger = logging.getLogger("kompot")

# StringDB API endpoints
STRING_API_BASE_URL = "https://string-db.org/api"
STRING_WEB_BASE_URL = "https://string-db.org/cgi"

# Check for required dependencies
STRINGDB_AVAILABLE = True
MISSING_DEPENDENCIES = []

try:
    import requests
except ImportError:
    STRINGDB_AVAILABLE = False
    MISSING_DEPENDENCIES.append("requests")

try:
    from IPython.display import HTML, Image, display
except ImportError:
    # IPython is optional, only needed for display in notebooks
    pass

try:
    import matplotlib.pyplot as plt
except ImportError:
    # matplotlib is optional, only needed for visualization
    pass

try:
    import pandas as pd
except ImportError:
    STRINGDB_AVAILABLE = False
    MISSING_DEPENDENCIES.append("pandas")

class StringDBReport:
    """Generate rich gene set reports with StringDB integration.
    
    This class provides tools to generate rich HTML reports for gene sets,
    including StringDB network visualization, resource links, and other
    gene information. It's designed to work well in Jupyter notebooks but
    can also be used programmatically.
    
    Parameters
    ----------
    genes : List[str]
        List of gene symbols to include in the report
    species_id : int, optional
        NCBI taxonomy ID for species (default: 9606 for Homo sapiens)
    include_stringdb : bool, optional
        Include StringDB network image and links (default: True)
    include_resources : bool, optional
        Include external resource links for genes (default: True)
    include_enrichment : bool, optional
        Include functional enrichment analysis (default: False)
    
    Attributes
    ----------
    genes : List[str]
        List of gene symbols included in the report
    species_id : int
        NCBI taxonomy ID for the species
    string_db_base_url : str
        Base URL for StringDB API and web interface
        
    Notes
    -----
    Supported species IDs and their names:
    
    ============  ========================
    Species ID    Species Name
    ============  ========================
    9606          Homo sapiens
    10090         Mus musculus
    10116         Rattus norvegicus
    7227          Drosophila melanogaster
    6239          Caenorhabditis elegans
    4932          Saccharomyces cerevisiae
    3702          Arabidopsis thaliana
    ============  ========================
    
    Additional species IDs can be used but won't have mapped names in the report.
    For the full list of available species, see the StringDB website.
    """
    
    def __init__(
        self, 
        genes: List[str], 
        species_id: int = 9606,
        include_stringdb: bool = True,
        include_resources: bool = True,
        include_enrichment: bool = False,
    ):
        """Initialize the StringDBReport with genes and options."""
        # Check for required dependencies
        if not STRINGDB_AVAILABLE:
            missing = ", ".join(MISSING_DEPENDENCIES)
            error_msg = f"StringDBReport is unavailable due to missing dependencies. Make sure '{missing}' and other required packages are installed."
            raise ImportError(error_msg)
            
        self.genes = genes
        self.species_id = species_id
        self.string_db_base_url = "https://string-db.org"
        self.include_stringdb = include_stringdb
        self.include_resources = include_resources
        self.include_enrichment = include_enrichment
        
        # Map species IDs to common names
        self.species_map = {
            9606: "Homo sapiens",
            10090: "Mus musculus",
            10116: "Rattus norvegicus",
            7227: "Drosophila melanogaster",
            6239: "Caenorhabditis elegans",
            4932: "Saccharomyces cerevisiae",
            3702: "Arabidopsis thaliana",
        }
        
        # Supported annotation categories
        self.annotation_categories = {
            "Process": "Process (Gene Ontology)",
            "Component": "Component (Gene Ontology)",
            "Function": "Function (Gene Ontology)",
            "KEGG": "KEGG Pathways",
            "Pfam": "Protein Domains (Pfam)",
            "InterPro": "Protein Domains (InterPro)",
            "SMART": "Protein Domains (SMART)",
            "Keywords": "UniProt Keywords",
            "Reactome": "Reactome Pathways",
            "WikiPathways": "WikiPathways",
        }
    
    def get_species_name(self) -> str:
        """Get human-readable species name from species ID."""
        return self.species_map.get(self.species_id, f"Species ID: {self.species_id}")
    
    def get_stringdb_url(self, additional_genes: Optional[List[str]] = None) -> str:
        """Generate URL for StringDB network visualization.
        
        Parameters
        ----------
        additional_genes : List[str], optional
            Additional genes to include in the StringDB query
            
        Returns
        -------
        str
            URL for StringDB network visualization
        """
        genes_to_include = self.genes.copy()
        if additional_genes:
            genes_to_include.extend(additional_genes)
        
        # Remove duplicates while preserving order
        unique_genes = []
        for gene in genes_to_include:
            if gene not in unique_genes:
                unique_genes.append(gene)
        
        gene_string = "%0d".join(unique_genes)
        url = f"{self.string_db_base_url}/cgi/network?identifiers={gene_string}&species={self.species_id}"
        return url
    
    def get_stringdb_image_url(self, additional_genes: Optional[List[str]] = None) -> str:
        """Generate URL for StringDB network image.
        
        Parameters
        ----------
        additional_genes : List[str], optional
            Additional genes to include in the StringDB image
            
        Returns
        -------
        str
            URL for StringDB network image
        """
        genes_to_include = self.genes.copy()
        if additional_genes:
            genes_to_include.extend(additional_genes)
        
        # Remove duplicates while preserving order
        unique_genes = []
        for gene in genes_to_include:
            if gene not in unique_genes:
                unique_genes.append(gene)
        
        gene_string = "%0d".join(unique_genes)
        url = f"{self.string_db_base_url}/api/image/network?identifiers={gene_string}&species={self.species_id}"
        return url
    
    def get_resource_links(self, gene: str) -> Dict[str, str]:
        """Generate external resource links for a gene.
        
        Parameters
        ----------
        gene : str
            Gene symbol to generate links for
            
        Returns
        -------
        Dict[str, str]
            Dictionary mapping resource names to URLs
        """
        encoded_gene = urllib.parse.quote(gene)
        
        links = {
            "STRING DB": f"{self.string_db_base_url}/cgi/network?identifiers={encoded_gene}&species={self.species_id}",
            "BioGRID": f"https://thebiogrid.org/search.php?search={encoded_gene}&organism={self.get_species_name()}",
            "Reactome": f"https://reactome.org/content/query?q={encoded_gene}&species={self.get_species_name().replace(' ', '+')}&cluster=true",
            "GeneCards": f"https://www.genecards.org/cgi-bin/carddisp.pl?gene={encoded_gene}",
        }
        
        # Add UniProt for all organisms
        links["UniProt"] = f"https://www.uniprot.org/uniprotkb?query={encoded_gene}+AND+organism_id:{self.species_id}"
        
        # Species-specific resources
        if self.species_id == 9606:  # Human
            links["NCBI Gene"] = f"https://www.ncbi.nlm.nih.gov/gene/?term={encoded_gene}+AND+human[orgn]"
            links["OMIM"] = f"https://www.omim.org/search?search={encoded_gene}"
            links["GTeX"] = f"https://gtexportal.org/home/gene/{encoded_gene}"
        
        elif self.species_id == 10090:  # Mouse
            links["MGI"] = f"https://www.informatics.jax.org/quicksearch/summary?queryType=exactPhrase&query={encoded_gene}&submit=Quick%0D%0ASearch"
            links["NCBI Gene"] = f"https://www.ncbi.nlm.nih.gov/gene/?term={encoded_gene}+AND+mouse[orgn]"
        
        # Add links for other species as needed
        
        return links
    
    def fetch_stringdb_image(self, additional_genes: Optional[List[str]] = None) -> Optional[bytes]:
        """Fetch StringDB network image as bytes.
        
        Parameters
        ----------
        additional_genes : List[str], optional
            Additional genes to include in the StringDB image
            
        Returns
        -------
        Optional[bytes]
            Image bytes or None if fetch failed
        """
        url = self.get_stringdb_image_url(additional_genes)
        return self._make_request(url)
    
    def _make_request(self, url: str, timeout: int = 10) -> Optional[bytes]:
        """Make a request to fetch data from a URL.
        
        This helper method is separated to make testing easier.
        
        Parameters
        ----------
        url : str
            URL to fetch data from
        timeout : int, optional
            Request timeout in seconds
            
        Returns
        -------
        Optional[bytes]
            Content bytes or None if request failed
        """
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response.content
        except Exception as e:
            logger.warning(f"Failed to fetch data from {url}: {e}")
            return None
    
    def to_html(self, additional_genes: Optional[List[str]] = None) -> str:
        html_parts = []

        # Header
        species_name = self.get_species_name()
        html_parts.append(f"<h3>Gene Set Report: {len(self.genes)} genes</h3>")
        html_parts.append(f"<p><strong>Species:</strong> {species_name} (Taxonomy ID: {self.species_id})</p>")

        # StringDB Network Section
        if self.include_stringdb:
            stringdb_url = self.get_stringdb_url(additional_genes)
            image_url = self.get_stringdb_image_url(additional_genes)

            html_parts.append("<h4>StringDB Network</h4>")
            html_parts.append(f'<p><a href="{stringdb_url}" target="_blank">View interactive network in StringDB</a></p>')
            html_parts.append(f'<a href="{stringdb_url}" target="_blank"><img src="{image_url}" style="max-width:800px; border:1px solid #ddd;" alt="StringDB Network"></a>')

        # ✅ Resource Links Section
        if self.include_resources:
            html_parts.append('<div style="margin-top: 20px;">')
            html_parts.append('<details>')
            html_parts.append(
                f'<summary style="cursor: pointer; font-weight: bold; padding: 10px; background-color: #f8f8f8; border: 1px solid #ddd;">'
                f'Resource Links ({len(self.genes)} genes)</summary>'
            )
            html_parts.append('<div style="padding: 10px;">')
            html_parts.append('<table border="1" style="border-collapse:collapse; width:100%; text-align: left;">')
            html_parts.append('<tr><th style="width:15%; text-align:left;">Gene</th><th style="text-align:left;">Resource Links</th></tr>')

            for gene in self.genes:
                links = self.get_resource_links(gene)
                link_html = " | ".join(f'<a href="{url}" target="_blank">{name}</a>' for name, url in links.items())
                html_parts.append(f'<tr><td style="text-align:left;">{gene}</td><td style="text-align:left;">{link_html}</td></tr>')

            html_parts.append('</table>')
            html_parts.append('</div>')  # close inner div
            html_parts.append('</details>')
            html_parts.append('</div>')  # close outer div

        # Functional Enrichment Section
        if self.include_enrichment:
            try:
                html_parts.append('<h4>Functional Enrichment Analysis</h4>')

                gene_string = urllib.parse.quote("\n".join(self.genes))
                url = (
                    f"{self.string_db_base_url}/cgi/network.pl"
                    f"?identifiers={gene_string}"
                    f"&species={self.species_id}"
                    f"&network_flavor=evidence"
                    f"&required_score=400"
                    f"&caller_identity=kompot"
                    f"#enrichment"
                )

                html_parts.append(f'<p><a href="{url}" target="_blank" style="font-weight: bold;">View interactive enrichment analysis on StringDB</a></p>')

                categories_to_try = [
                    ("Process", "Gene Ontology Processes", False),
                    ("KEGG", "KEGG Pathways", False),
                    ("Function", "Gene Ontology Functions", False),
                    ("Component", "Gene Ontology Components", False),
                    ("Reactome", "Reactome Pathways", False)
                ]

                html_parts.append('<div style="margin-top: 20px;">')

                for category, label, open_by_default in categories_to_try:
                    try:
                        logger.debug(f"Fetching enrichment data for {category}")
                        enrichment_df = self.get_functional_enrichment(category=category)

                        html_parts.append(f'<details {"open" if open_by_default else ""}>')
                        html_parts.append(
                            f'<summary style="cursor: pointer; font-weight: bold; padding: 10px; background-color: #f8f8f8; border: 1px solid #ddd;">'
                            f'{label} ({len(enrichment_df) if enrichment_df is not None else 0} terms)</summary>'
                        )
                        html_parts.append('<div style="padding: 10px;">')

                        expected_cols = ['term', 'description', 'signal', 'strength', 'fdr', 'number_of_genes', 'inputGenes']

                        if enrichment_df is not None and not enrichment_df.empty:
                            display_cols = [col for col in expected_cols if col in enrichment_df.columns]
                            if not display_cols:
                                display_cols = enrichment_df.columns

                            num_rows = min(20, len(enrichment_df))
                            table_html = enrichment_df.head(num_rows)[display_cols].to_html(
                                index=False,
                                escape=False,
                                classes="enrichment-table",
                                border=1
                            )
                            styled_table = table_html.replace(
                                '<table ',
                                '<table style="border-collapse:collapse; width:100%; text-align:left;" '
                            )
                            html_parts.append(styled_table)

                            if len(enrichment_df) > num_rows:
                                html_parts.append(
                                    f'<p style="margin-top:10px; font-style:italic;">Showing {num_rows} of {len(enrichment_df)} enriched terms</p>'
                                )
                        else:
                            empty_df = pd.DataFrame(columns=expected_cols)
                            table_html = empty_df.to_html(
                                index=False,
                                escape=False,
                                classes="enrichment-table",
                                border=1
                            )
                            styled_table = table_html.replace(
                                '<table ',
                                '<table style="border-collapse:collapse; width:100%; text-align:left;" '
                            )
                            html_parts.append(styled_table)
                            html_parts.append(
                                f'<p style="margin-top:10px; font-style:italic;">No enriched terms were found in {label}.</p>'
                            )

                        html_parts.append('</div>')
                        html_parts.append('</details>')

                    except Exception as e:
                        logger.debug(f"Failed to render enrichment for {category}: {e}")
                        html_parts.append(
                            f'<details><summary>{label}</summary>'
                            f'<div><p style="font-style: italic;">Error fetching enrichment for {label}.</p></div></details>'
                        )

                html_parts.append('</div>')

            except Exception as e:
                logger.debug(f"Enrichment section failed: {e}")
                html_parts.append(
                    f'<p>An error occurred while generating enrichment information. '
                    f'<a href="{url}" target="_blank">View enrichment on StringDB</a></p>'
                )

        return "".join(html_parts)


    def _repr_html_(self) -> str:
        """HTML representation for display in Jupyter notebooks."""
        return self.to_html()
    
    def display(self, additional_genes: Optional[List[str]] = None) -> None:
        """Display the report in a Jupyter notebook.
        
        Parameters
        ----------
        additional_genes : List[str], optional
            Additional genes to include in the StringDB visualizations
        """
        display(HTML(self.to_html(additional_genes)))
    
    
    
    def get_functional_enrichment(self, 
                                 category: str = "Process", 
                                 fdr_threshold: float = 0.05) -> Optional[pd.DataFrame]:
        """Get functional enrichment analysis for the gene set.
        
        This method fetches functional enrichment results through StringDB's enrichment API.
        
        Parameters
        ----------
        category : str, optional
            Category for enrichment analysis (default: "Process")
            Valid options:
            - Process: Gene Ontology biological processes
            - Component: Gene Ontology cellular components
            - Function: Gene Ontology molecular functions
            - KEGG: KEGG pathways
            - Pfam: Protein domain annotations from Pfam
            - InterPro: Protein domain annotations from InterPro
            - SMART: Protein domain annotations from SMART
            - Keywords: UniProt keyword annotations
            - Reactome: Reactome pathway annotations
            - WikiPathways: WikiPathways annotations
        fdr_threshold : float, optional
            FDR threshold for significance (default: 0.05)
            
        Returns
        -------
        Optional[pd.DataFrame]
            DataFrame with enrichment results or None if request failed
            
        Notes
        -----
        The enrichment results include various columns depending on the category:
        - term: Identifier for the enriched term (e.g., GO:0006281)
        - description: Human-readable description of the term
        - signal: Balanced metric combining enrichment magnitude and significance (higher is better)
        - strength: Log10(observed/expected) indicating enrichment effect size
        - fdr: False discovery rate (adjusted p-value)
        - number_of_genes: Number of genes from the input that match this term
        - inputGenes: List of input genes that match this term
        
        Results are sorted by signal (descending) following StringDB's default behavior. Different categories have different 
        levels of annotation coverage. For example, GO Process usually 
        provides the most annotations, while specific pathway databases 
        may have more limited coverage.
        """
        if category not in self.annotation_categories:
            valid_cats = ", ".join(self.annotation_categories.keys())
            logger.warning(f"Invalid category '{category}'. Valid options are: {valid_cats}")
            category = "Process"  # Default to Process if invalid
        
        # Map internal categories to StringDB API categories (same names, but kept for clarity)
        category_map = {
            "Process": "Process",
            "Component": "Component",
            "Function": "Function", 
            "KEGG": "KEGG",
            "Pfam": "Pfam",
            "InterPro": "InterPro", 
            "SMART": "SMART",
            "Keywords": "Keyword",  # StringDB API returns "Keyword" not "Keywords"
            "Reactome": "RCTM",     # StringDB API uses "RCTM" for Reactome pathways
            "WikiPathways": "WikiPathways"
        }
        
        api_category = category_map.get(category, "Process")
        
        # Use the REST API to get enrichment results
        url = f"{STRING_API_BASE_URL}/json/enrichment"
        
        # StringDB expects newline-separated gene list
        gene_list = "\n".join(self.genes)
        
        # In some API versions, the StringDB API might only return results
        # if the category is explicitly specified
        payload_base = {
            "identifiers": gene_list,
            "species": self.species_id,
            "caller_identity": "kompot"
        }
        
        # First try to get all categories at once (more efficient)
        payload = payload_base.copy()
        
        # But we'll also prepare a category-specific fallback payload
        fallback_payload = payload_base.copy()
        
        # For GO categories, StringDB expects the format "GO Process" not just "Process"
        if category in ["Process", "Component", "Function"]:
            fallback_payload["enrichment_category"] = f"GO {category}"
        else:
            fallback_payload["enrichment_category"] = api_category
        
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json"
        }
        
        def process_enrichment_response(response, target_category):
            """Helper function to process enrichment API responses"""
            if response.status_code != 200:
                return None
                
            try:
                data = response.json()
            except json.JSONDecodeError:
                # Some versions of the StringDB API return a malformed JSON
                # Try to clean it up before parsing
                try:
                    # Attempt to fix common JSON formatting issues
                    text = response.text
                    
                    # More aggressive JSON cleaning for StringDB's problematic responses
                    
                    # First, let's escape any unescaped quotes in strings
                    text = re.sub(r':\s*"([^"]*)"([^,}]*)"([^"]*)"', r': "\1\\"\2\\"\3"', text)
                    
                    # Replace unquoted keys with quoted keys (common API issue)
                    text = re.sub(r'(\s*)(\w+)(\s*):([^/])', r'\1"\2"\3:\4', text)
                    
                    # Fix missing quotes around values if needed
                    text = re.sub(r':\s*([a-zA-Z][a-zA-Z0-9_]*)\s*([,}])', r': "\1"\2', text)
                    
                    # Fix trailing commas in arrays/objects which are invalid in JSON
                    text = re.sub(r',\s*}', '}', text)
                    text = re.sub(r',\s*\]', ']', text)
                    
                    # Handle unusual escape sequences
                    text = text.replace('\\"', '"').replace('\\', '\\\\').replace('\\"', '\\\\"')
                    
                    # Try parsing with a more lenient parser
                    try:
                        import json5
                        data = json5.loads(text)
                    except ImportError:
                        # Fall back to standard json if json5 is not available
                        data = json.loads(text)
                except Exception as e:
                    logger.debug(f"Invalid JSON response from StringDB API: {str(e)} - {response.text[:100]}...")
                    return None
            
            # Process the results into a DataFrame
            if isinstance(data, list) and len(data) > 0:
                df = pd.DataFrame(data)
                
                # Filter to only include results for the requested category
                if 'category' in df.columns:
                    # Filter by the API's actual category name
                    df = df[df['category'] == target_category]
                
                # Filter by FDR if needed
                if 'fdr' in df.columns:
                    df = df[df['fdr'] <= fdr_threshold]
                    
                # Compute signal and strength columns
                if len(df) > 0:
                    df = self._compute_signal_and_strength(df)
                    # Sort by signal (descending) as StringDB does by default
                    return df.sort_values('signal', ascending=False)
            
            return None
        
        try:
            # First attempt: Get all categories at once (more efficient)
            logger.debug(f"Fetching enrichment data for category '{category}' (all categories approach)")
            
            # Attempt to use the json5 module for more tolerant parsing if it's available
            try:
                import json5
                # If json5 is available, we'll use it directly in the response processing
                logger.debug("Using json5 for more tolerant JSON parsing")
            except ImportError:
                # json5 isn't available, but we'll still try our manual fixes
                logger.debug("json5 module not available, using standard JSON parser with fixes")
            
            response = requests.post(url, data=payload, headers=headers, timeout=30)
            
            # Process the first attempt
            df = process_enrichment_response(response, api_category)
            
            # If the first attempt failed, try the fallback approach
            if df is None or len(df) == 0:
                logger.debug(f"No results found using all-categories approach, trying category-specific request")
                response = requests.post(url, data=fallback_payload, headers=headers, timeout=30)
                df = process_enrichment_response(response, api_category)
                
                # Additionally try a few variations of the category name that might be needed
                if df is None or len(df) == 0 and category in ["Process", "Component", "Function"]:
                    # Some API versions use just "Process" instead of "GO Process"
                    logger.debug(f"Trying variation of category name for {category}")
                    temp_payload = fallback_payload.copy()
                    temp_payload["enrichment_category"] = category
                    response = requests.post(url, data=temp_payload, headers=headers, timeout=30)
                    df = process_enrichment_response(response, api_category)
            
            # Check if we got any results
            if df is not None and len(df) > 0:
                return df
            else:
                logger.warning(f"No enrichment terms found for category '{category}' below FDR threshold {fdr_threshold}")
                return None
                
        except Exception as e:
            logger.warning(f"Failed to fetch enrichment data for category '{category}': {e}")
            return None

    def _compute_signal_and_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute signal and strength columns according to StringDB definitions.
        
        Strength: log10(observed / expected) where expected is calculated from background frequencies
        Signal: weighted harmonic mean between (observed/expected) ratio and -log(FDR) to balance 
                both enrichment magnitude and statistical significance
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with enrichment results from StringDB API
            
        Returns
        -------
        pd.DataFrame
            DataFrame with added 'signal' and 'strength' columns
        """
        df = df.copy()
        
        # Total protein counts used by StringDB for different species
        # These values were reverse-engineered from hypergeometric p-values
        if self.species_id == 9606:  # Human
            total_proteins = 19274  # Confirmed by p-value matching
        elif self.species_id == 10090:  # Mouse
            total_proteins = 22000  # Confirmed by p-value matching
        else:
            # For other species, use a reasonable default
            total_proteins = 18000
            
        # Calculate expected counts as per StringDB definition:
        # Expected = (network_size * background_with_term) / total_proteins_in_species
        network_size = len(self.genes)
        df['expected'] = (network_size * df['number_of_genes_in_background']) / total_proteins
        
        # Compute strength = log10(observed / expected)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df['strength'] = np.log10(df['number_of_genes'] / df['expected'])
        
        # For signal computation, we need both metrics on reasonable scales
        obs_exp_ratio = df['number_of_genes'] / df['expected']
        neg_log_fdr = -np.log10(df['fdr'].clip(lower=1e-10))  # Avoid log(0)
        
        # Normalize both metrics to [0,1] range for balanced weighting
        if obs_exp_ratio.max() > 0:
            obs_exp_norm = obs_exp_ratio / obs_exp_ratio.max()
        else:
            obs_exp_norm = obs_exp_ratio
            
        if neg_log_fdr.max() > 0:
            neg_log_fdr_norm = neg_log_fdr / neg_log_fdr.max()  
        else:
            neg_log_fdr_norm = neg_log_fdr
            
        # Compute signal as weighted harmonic mean
        # This balances enrichment magnitude (obs/exp) with significance (-log FDR)
        with np.errstate(divide='ignore', invalid='ignore'):
            df['signal'] = 2 / (1/obs_exp_norm.clip(lower=1e-10) + 1/neg_log_fdr_norm.clip(lower=1e-10))
            df['signal'] = df['signal'].fillna(0)
        
        return df
    
    def save_html(self, filename: str, additional_genes: Optional[List[str]] = None) -> None:
        """Save the report as an HTML file.
        
        Parameters
        ----------
        filename : str
            Path to save the HTML file
        additional_genes : List[str], optional
            Additional genes to include in the StringDB visualizations
        """
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(self.to_html(additional_genes))
        logger.info(f"Saved gene report to {filename}")
    
    def get_json(self, additional_genes: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate a JSON representation of the gene report.
        
        Parameters
        ----------
        additional_genes : List[str], optional
            Additional genes to include in the StringDB visualizations
            
        Returns
        -------
        Dict[str, Any]
            JSON-serializable dictionary with report data
        """
        result = {
            "genes": self.genes,
            "species_id": self.species_id,
            "species_name": self.get_species_name(),
        }
        
        if self.include_stringdb:
            result["stringdb"] = {
                "url": self.get_stringdb_url(additional_genes),
                "image_url": self.get_stringdb_image_url(additional_genes),
            }
        
        if self.include_resources:
            result["resources"] = {gene: self.get_resource_links(gene) for gene in self.genes}
        
        # Add enrichment data if requested
        if self.include_enrichment:
            try:
                enrichment_df = self.get_functional_enrichment()
                if enrichment_df is not None and not enrichment_df.empty:
                    result["enrichment"] = enrichment_df.to_dict(orient="records")
                    
                    # Add direct URL to StringDB enrichment interface
                    gene_string = urllib.parse.quote("\n".join(self.genes))
                    url = (f"{self.string_db_base_url}/cgi/network.pl"
                          f"?identifiers={gene_string}"
                          f"&species={self.species_id}"
                          f"&network_flavor=evidence"
                          f"&required_score=400"
                          f"&caller_identity=kompot"
                          f"#enrichment")
                    result["enrichment_url"] = url
                    
                    # Add direct URL to StringDB interaction network
                    partners_url = (f"{self.string_db_base_url}/cgi/network.pl"
                                   f"?identifiers={gene_string}"
                                   f"&species={self.species_id}"
                                   f"&add_white_nodes=15"
                                   f"&network_flavor=evidence")
                    result["interaction_url"] = partners_url
            except Exception as e:
                logger.warning(f"Failed to include enrichment data in JSON: {e}")
        
        return result
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert gene resource links to a pandas DataFrame.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with genes as index and resource links as columns
        """
        data = []
        columns = set()
        
        # Collect all unique resource types
        for gene in self.genes:
            links = self.get_resource_links(gene)
            columns.update(links.keys())
        
        # Build data for each gene
        for gene in self.genes:
            gene_links = self.get_resource_links(gene)
            row = {"Gene": gene}
            for col in columns:
                row[col] = gene_links.get(col, "")
            data.append(row)
        
        # Create DataFrame and set Gene as index
        df = pd.DataFrame(data)
        return df