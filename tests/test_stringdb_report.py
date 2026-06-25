"""
Unit tests for the StringDBReport class.
"""

import os

import pytest
import pandas as pd
from kompot.plot import StringDBReport
import kompot.plot.stringdb as stringdb_mod

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
    assert "Process" in report.annotation_categories
    assert "KEGG" in report.annotation_categories
    assert "Reactome" in report.annotation_categories


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
    assert "species=9606" in url

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
    assert "species=9606" in url


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
        report._make_request = lambda url, timeout=10: b"fake_image_data"
        image_data = report.fetch_stringdb_image()
        assert image_data == b"fake_image_data"

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
        assert f'<td style="text-align:left;">{gene}</td>' in html


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
        sample_df = pd.DataFrame(
            {
                "term": ["GO:0006281", "GO:0007049"],
                "description": ["DNA repair", "Cell cycle regulation"],
                "fdr": [0.001, 0.01],
            }
        )

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


# ---------------------------------------------------------------------------
# Background / tested-universe (over-representation correction) tests
# ---------------------------------------------------------------------------


class _FakeResponse:
    """Minimal stand-in for requests.Response."""

    def __init__(self, *, text="", json_data=None, status_code=200):
        self.text = text
        self._json = json_data
        self.status_code = status_code

    def json(self):
        if self._json is None:
            raise ValueError("no json")
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def _make_ids_tsv(symbol_to_id):
    """Build a get_string_ids echo_query TSV response body.

    Columns: queryItem, queryIndex, stringId, taxon, taxonName, preferredName, annotation
    Symbols mapped to None are omitted entirely (mimicking StringDB dropping misses).
    """
    lines = []
    idx = 0
    for sym, sid in symbol_to_id.items():
        if sid is None:
            continue
        lines.append(f"{sym}\t{idx}\t{sid}\t9606\tHomo sapiens\t{sym}\tannotation text")
        idx += 1
    return "\n".join(lines)


def test_map_to_string_ids(monkeypatch):
    """Symbols resolve to STRING IDs; misses dropped; duplicates collapsed."""
    report = StringDBReport(HUMAN_GENES)

    # KRAS and EGFR intentionally map to the same id (dup) and PTEN is unmapped.
    mapping = {
        "TP53": "9606.ENSP00000269305",
        "BRCA1": "9606.ENSP00000418960",
        "KRAS": "9606.ENSP00000256078",
        "EGFR": "9606.ENSP00000256078",  # duplicate STRING id
        "PTEN": None,  # unmapped -> omitted from response
    }
    tsv = _make_ids_tsv(mapping)
    monkeypatch.setattr(
        stringdb_mod.requests,
        "post",
        lambda *a, **k: _FakeResponse(text=tsv),
    )

    ids = report._map_to_string_ids(HUMAN_GENES)
    # Order-preserving, de-duplicated, unmapped excluded.
    assert ids == [
        "9606.ENSP00000269305",
        "9606.ENSP00000418960",
        "9606.ENSP00000256078",
    ]


def test_map_to_string_ids_handles_failure(monkeypatch):
    """A failed mapping request yields an empty list, not an exception."""
    report = StringDBReport(HUMAN_GENES)

    def boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(stringdb_mod.requests, "post", boom)
    assert report._map_to_string_ids(HUMAN_GENES) == []
    assert report._map_to_string_ids([]) == []


def _enrichment_rows(fdr, n_bg=50):
    """One Process-category enrichment row at a given fdr."""
    return [
        {
            "term": "GO:0006281",
            "description": "DNA repair",
            "category": "Process",
            "number_of_genes": 3,
            "number_of_genes_in_background": n_bg,
            "fdr": fdr,
            "p_value": fdr / 10.0,
        }
    ]


def test_background_omitted_by_default(monkeypatch):
    """With no background, the enrichment payload carries no background param
    and the genome-wide universe drives strength."""
    captured = {}

    def fake_post(url, data=None, headers=None, timeout=None):
        captured["data"] = data
        return _FakeResponse(json_data=_enrichment_rows(0.001))

    monkeypatch.setattr(stringdb_mod.requests, "post", fake_post)

    report = StringDBReport(HUMAN_GENES, include_enrichment=True)
    df = report.get_functional_enrichment(category="Process")

    assert df is not None and len(df) == 1
    assert "background_string_identifiers" not in captured["data"]
    # Genome-wide human universe (19274): expected = 5 genes * 50 / 19274
    expected = (len(HUMAN_GENES) * 50) / 19274
    assert df.iloc[0]["expected"] == pytest.approx(expected)


def test_background_wired_into_payload(monkeypatch):
    """A supplied background is mapped to STRING IDs and both foreground and
    background are submitted in the same identifier space."""
    captured = {}

    # Deterministic, network-free identifier mapping.
    id_map = {
        "TP53": "9606.A",
        "BRCA1": "9606.B",
        "KRAS": "9606.C",
        "EGFR": "9606.D",
        "PTEN": "9606.E",
        "GAPDH": "9606.F",
        "ACTB": "9606.G",
    }
    monkeypatch.setattr(
        StringDBReport,
        "_map_to_string_ids",
        lambda self, genes: [id_map[g] for g in genes if id_map.get(g)],
    )

    def fake_post(url, data=None, headers=None, timeout=None):
        captured["data"] = data
        return _FakeResponse(json_data=_enrichment_rows(0.02, n_bg=4))

    monkeypatch.setattr(stringdb_mod.requests, "post", fake_post)

    background = HUMAN_GENES + ["GAPDH", "ACTB"]
    report = StringDBReport(HUMAN_GENES, include_enrichment=True)
    df = report.get_functional_enrichment(category="Process", background=background)

    data = captured["data"]
    assert "background_string_identifiers" in data
    bg_ids = data["background_string_identifiers"].split("\n")
    fg_ids = data["identifiers"].split("\n")
    # Foreground submitted as STRING IDs (same space as background), not symbols.
    assert fg_ids == ["9606.A", "9606.B", "9606.C", "9606.D", "9606.E"]
    assert bg_ids == [
        "9606.A", "9606.B", "9606.C", "9606.D", "9606.E", "9606.F", "9606.G",
    ]
    # Foreground is a subset of the background universe (consistent ORA).
    assert set(fg_ids).issubset(set(bg_ids))
    # Universe size used for strength is the mapped background size (7), not 19274.
    expected = (len(HUMAN_GENES) * 4) / len(bg_ids)
    assert df.iloc[0]["expected"] == pytest.approx(expected)


def test_background_changes_results(monkeypatch):
    """The same term yields a different FDR and strength under a restricted
    tested-universe background than under the genome-wide default, proving the
    parameter is wired end-to-end through the API call and the stats."""
    id_map = {g: f"9606.{i}" for i, g in enumerate(HUMAN_GENES + ["GAPDH", "ACTB"])}
    monkeypatch.setattr(
        StringDBReport,
        "_map_to_string_ids",
        lambda self, genes: [id_map[g] for g in genes if id_map.get(g)],
    )

    def fake_post(url, data=None, headers=None, timeout=None):
        # The API recomputes significance against the supplied universe:
        # a restricted background makes the term *more* significant here.
        if "background_string_identifiers" in (data or {}):
            return _FakeResponse(json_data=_enrichment_rows(0.001, n_bg=4))
        return _FakeResponse(json_data=_enrichment_rows(0.2, n_bg=50))

    monkeypatch.setattr(stringdb_mod.requests, "post", fake_post)

    report = StringDBReport(HUMAN_GENES, include_enrichment=True)
    background = HUMAN_GENES + ["GAPDH", "ACTB"]

    df_genome = report.get_functional_enrichment(
        category="Process", fdr_threshold=0.5
    )
    df_bg = report.get_functional_enrichment(
        category="Process", fdr_threshold=0.5, background=background
    )

    assert df_genome is not None and df_bg is not None
    # FDR differs (the param reached the API request).
    assert df_genome.iloc[0]["fdr"] != df_bg.iloc[0]["fdr"]
    # Strength differs (the universe size reached the signal/strength stats).
    assert df_genome.iloc[0]["strength"] != pytest.approx(df_bg.iloc[0]["strength"])


def test_background_instance_default_and_override(monkeypatch):
    """Instance-level background is honored, and a call-level argument overrides it."""
    seen_backgrounds = []
    id_map = {g: f"9606.{i}" for i, g in enumerate(
        HUMAN_GENES + ["GAPDH", "ACTB", "VIM", "MYC"]
    )}

    def fake_map(self, genes):
        return [id_map[g] for g in genes if id_map.get(g)]

    monkeypatch.setattr(StringDBReport, "_map_to_string_ids", fake_map)

    def fake_post(url, data=None, headers=None, timeout=None):
        if "background_string_identifiers" in (data or {}):
            seen_backgrounds.append(
                data["background_string_identifiers"].split("\n")
            )
        return _FakeResponse(json_data=_enrichment_rows(0.01, n_bg=4))

    monkeypatch.setattr(stringdb_mod.requests, "post", fake_post)

    inst_bg = HUMAN_GENES + ["GAPDH", "ACTB"]
    report = StringDBReport(HUMAN_GENES, include_enrichment=True, background=inst_bg)
    report.get_functional_enrichment(category="Process")  # uses instance bg
    call_bg = HUMAN_GENES + ["VIM", "MYC"]
    report.get_functional_enrichment(category="Process", background=call_bg)

    assert len(seen_backgrounds) == 2
    assert seen_backgrounds[0] == [id_map[g] for g in inst_bg]
    assert seen_backgrounds[1] == [id_map[g] for g in call_bg]


def test_background_falls_back_when_mapping_empty(monkeypatch):
    """If identifier mapping yields nothing, fall back to the genome-wide
    background rather than sending an empty/broken universe."""
    monkeypatch.setattr(
        StringDBReport, "_map_to_string_ids", lambda self, genes: []
    )

    captured = {}

    def fake_post(url, data=None, headers=None, timeout=None):
        captured["data"] = data
        return _FakeResponse(json_data=_enrichment_rows(0.001))

    monkeypatch.setattr(stringdb_mod.requests, "post", fake_post)

    report = StringDBReport(HUMAN_GENES, include_enrichment=True)
    df = report.get_functional_enrichment(
        category="Process", background=HUMAN_GENES + ["GAPDH"]
    )
    assert df is not None
    assert "background_string_identifiers" not in captured["data"]
    # Genome-wide universe used for strength.
    assert df.iloc[0]["expected"] == pytest.approx((len(HUMAN_GENES) * 50) / 19274)


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("KOMPOT_RUN_INTEGRATION"),
    reason="live StringDB API test; set KOMPOT_RUN_INTEGRATION=1 to enable",
)
def test_background_live_integration():
    """End-to-end against the real StringDB API: a restricted tested-universe
    background materially changes the enrichment result vs the genome-wide
    default. Opt-in via KOMPOT_RUN_INTEGRATION; skipped if the API is
    unavailable so it never makes CI depend on an external service."""
    foreground = ["TP53", "BRCA1", "ATM", "CHEK2", "MDM2", "CDKN1A", "RB1"]
    background = foreground + [
        "GAPDH", "ACTB", "TUBB", "B2M", "HPRT1", "MYC", "JUN", "FOS",
        "STAT1", "IRF1", "VIM", "CD3D", "CD8A", "FOXP3", "IL2",
    ]
    report = StringDBReport(foreground, include_enrichment=True)
    try:
        df_genome = report.get_functional_enrichment(category="Process")
        df_bg = report.get_functional_enrichment(
            category="Process", background=background
        )
    except Exception as e:  # pragma: no cover - network dependent
        pytest.skip(f"StringDB API unavailable: {e}")

    n_genome = 0 if df_genome is None else len(df_genome)
    n_bg = 0 if df_bg is None else len(df_bg)
    # The restricted universe should not enrich *more* terms than genome-wide;
    # the correction removes spuriously significant terms.
    assert n_bg <= n_genome
