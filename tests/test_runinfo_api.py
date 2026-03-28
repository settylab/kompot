"""Tests for kompot.anndata.utils.runinfo RunInfo and RunComparison classes.

Covers uncovered lines in kompot/anndata/utils/runinfo.py including:
- RunInfo auto-detection, field mapping parsing, overwritten/missing field checks
- RunInfo HTML representation with groups, long lists, dict params, field mapping strings
- RunComparison badges, parameter diffs, shared fields, and edge cases
"""

import json
import pytest
import numpy as np
import pandas as pd
from anndata import AnnData

from kompot.anndata.utils.json_utils import (
    to_json_string,
)


# ---------------------------------------------------------------------------
# Helpers to build mock AnnData with run history
# ---------------------------------------------------------------------------

def _make_run_entry(run_id, params=None, field_mapping=None, timestamp="2025-01-01T00:00:00"):
    """Build a minimal run-history entry."""
    entry = {
        "run_id": run_id,
        "adjusted_run_id": run_id,
        "timestamp": timestamp,
        "params": params or {"condition1": "A", "condition2": "B", "obsm_key": "X_pca", "result_key": "kompot_da"},
        "field_names": {},
        "environment": {},
        "field_mapping": field_mapping or {},
    }
    return entry


def _adata_with_da_history(n_runs=1, field_mapping=None, extra_uns=None):
    """Return a small AnnData with kompot_da run history."""
    adata = AnnData(
        X=np.zeros((5, 3)),
        obs=pd.DataFrame({"group": ["A", "A", "B", "B", "B"]}, index=[f"c{i}" for i in range(5)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(3)]),
    )
    fm = field_mapping or {
        "da_pval": {"location": "obs", "type": "float", "description": "p-value"},
        "da_lfc": {"location": "obs", "type": "float", "description": "log fold change"},
    }
    runs = [_make_run_entry(i, field_mapping=fm) for i in range(n_runs)]
    adata.uns["kompot_da"] = {
        "run_history": to_json_string(runs),
        "anndata_fields": to_json_string({"obs": {"da_pval": n_runs - 1, "da_lfc": n_runs - 1}}),
    }
    # Put the actual columns so fields are "present"
    adata.obs["da_pval"] = 0.05
    adata.obs["da_lfc"] = 1.0
    if extra_uns:
        adata.uns.update(extra_uns)
    return adata


# ===================================================================
# RunInfo tests
# ===================================================================

class TestRunInfo:
    """Cover lines 78, 149-151, 168-169, 180-181, 187, 226-230, 241-244, 248, 253,
    289-291, 303-304, 312, 729, 742, 746, 823-828, 858-862."""

    def test_auto_detect_da(self):
        # line 78: auto-detect 'da' analysis type
        adata = _adata_with_da_history(n_runs=1)
        from kompot.anndata.utils.runinfo import RunInfo
        ri = RunInfo(adata, run_id=0, analysis_type=None)
        assert ri.analysis_type == "da"

    def test_auto_detect_fails(self):
        # lines 82-85: no analysis type detectable
        adata = AnnData(X=np.zeros((2, 2)))
        from kompot.anndata.utils.runinfo import RunInfo
        with pytest.raises(ValueError, match="Could not detect analysis type"):
            RunInfo(adata, run_id=0, analysis_type=None)

    def test_field_mapping_is_string_valid(self):
        # lines 146-148: field_mapping is a JSON string that parses successfully
        from kompot.anndata.utils.runinfo import RunInfo
        fm = {"my_field": {"location": "obs", "type": "float"}}
        fm_str = to_json_string(fm)
        run = _make_run_entry(0, field_mapping=fm_str)
        adata = AnnData(
            X=np.zeros((5, 3)),
            obs=pd.DataFrame({"group": ["A"] * 5, "my_field": [0.1] * 5}, index=[f"c{i}" for i in range(5)]),
            var=pd.DataFrame(index=[f"g{i}" for i in range(3)]),
        )
        adata.uns["kompot_da"] = {
            "run_history": to_json_string([run]),
            "anndata_fields": to_json_string({"obs": {"my_field": 0}}),
        }
        ri = RunInfo(adata, run_id=0, analysis_type="da")
        assert "obs" in ri.adata_fields

    def test_field_mapping_empty_string(self):
        # lines 153-155: field_mapping is empty (falsy) -> returns {}
        from kompot.anndata.utils.runinfo import RunInfo
        run = _make_run_entry(0, field_mapping={})
        adata = AnnData(
            X=np.zeros((5, 3)),
            obs=pd.DataFrame({"group": ["A"] * 5}, index=[f"c{i}" for i in range(5)]),
            var=pd.DataFrame(index=[f"g{i}" for i in range(3)]),
        )
        adata.uns["kompot_da"] = {
            "run_history": to_json_string([run]),
            "anndata_fields": to_json_string({}),
        }
        ri = RunInfo(adata, run_id=0, analysis_type="da")
        assert ri.adata_fields == {}

    def test_field_mapping_value_is_string(self):
        # lines 162-169, 177-181: mapping values are JSON strings
        from kompot.anndata.utils.runinfo import RunInfo
        fm = {
            "my_field": json.dumps({"location": "obs", "type": "float"}),
            "bad_field": "not{json",
        }
        run = _make_run_entry(0, field_mapping=fm)
        adata = AnnData(
            X=np.zeros((5, 3)),
            obs=pd.DataFrame({"group": ["A"] * 5, "my_field": [0.1] * 5}, index=[f"c{i}" for i in range(5)]),
            var=pd.DataFrame(index=[f"g{i}" for i in range(3)]),
        )
        adata.uns["kompot_da"] = {
            "run_history": to_json_string([run]),
            "anndata_fields": to_json_string({"obs": {"my_field": 0}}),
        }
        ri = RunInfo(adata, run_id=0, analysis_type="da")
        assert "obs" in ri.adata_fields
        assert "my_field" in ri.adata_fields["obs"]

    def test_field_mapping_location_not_in_result(self):
        # line 187: location not already in result dict
        from kompot.anndata.utils.runinfo import RunInfo
        fm = {
            "f1": {"location": "obs", "type": "float"},
            "f2": {"location": "var", "type": "float"},
        }
        run = _make_run_entry(0, field_mapping=fm)
        adata = AnnData(
            X=np.zeros((5, 3)),
            obs=pd.DataFrame({"group": ["A"] * 5, "f1": [0.1] * 5}, index=[f"c{i}" for i in range(5)]),
            var=pd.DataFrame({"f2": [0.1] * 3}, index=[f"g{i}" for i in range(3)]),
        )
        adata.uns["kompot_da"] = {
            "run_history": to_json_string([run]),
            "anndata_fields": to_json_string({"obs": {"f1": 0}, "var": {"f2": 0}}),
        }
        ri = RunInfo(adata, run_id=0, analysis_type="da")
        assert "obs" in ri.adata_fields
        assert "var" in ri.adata_fields

    def test_check_overwritten_fields_invalid_string_mapping(self):
        # lines 226-230, 241-244, 248: _check_overwritten_fields with string field_mapping
        from kompot.anndata.utils.runinfo import RunInfo
        # Create two runs, second one overwrites first's fields
        fm = {"da_pval": {"location": "obs", "type": "float"}}
        run0 = _make_run_entry(0, field_mapping=fm)
        run1 = _make_run_entry(1, field_mapping=fm)
        adata = AnnData(
            X=np.zeros((5, 3)),
            obs=pd.DataFrame({"group": ["A"] * 5, "da_pval": [0.1] * 5}, index=[f"c{i}" for i in range(5)]),
            var=pd.DataFrame(index=[f"g{i}" for i in range(3)]),
        )
        adata.uns["kompot_da"] = {
            "run_history": to_json_string([run0, run1]),
            "anndata_fields": to_json_string({"obs": {"da_pval": 1}}),  # owned by run 1
        }
        ri = RunInfo(adata, run_id=0, analysis_type="da")
        # Run 0's field was overwritten by run 1
        assert len(ri.overwritten_fields) > 0

    def test_check_overwritten_fields_string_field_mapping(self):
        # lines 226-230: field_mapping as string in _check_overwritten_fields
        from kompot.anndata.utils.runinfo import RunInfo
        fm_str = to_json_string({"da_pval": {"location": "obs", "type": "float"}})
        run0 = _make_run_entry(0, field_mapping=fm_str)
        adata = AnnData(
            X=np.zeros((5, 3)),
            obs=pd.DataFrame({"group": ["A"] * 5, "da_pval": [0.1] * 5}, index=[f"c{i}" for i in range(5)]),
            var=pd.DataFrame(index=[f"g{i}" for i in range(3)]),
        )
        adata.uns["kompot_da"] = {
            "run_history": to_json_string([run0]),
            "anndata_fields": to_json_string({"obs": {"da_pval": 0}}),
        }
        ri = RunInfo(adata, run_id=0, analysis_type="da")
        # Should work without errors; field is owned by run 0
        assert len(ri.overwritten_fields) == 0

    def test_check_overwritten_fields_empty_field_mapping(self):
        # lines 233-235: empty field_mapping in _check_overwritten_fields
        from kompot.anndata.utils.runinfo import RunInfo
        run0 = _make_run_entry(0, field_mapping={})
        adata = AnnData(
            X=np.zeros((5, 3)),
            obs=pd.DataFrame({"group": ["A"] * 5}, index=[f"c{i}" for i in range(5)]),
            var=pd.DataFrame(index=[f"g{i}" for i in range(3)]),
        )
        adata.uns["kompot_da"] = {
            "run_history": to_json_string([run0]),
            "anndata_fields": to_json_string({}),
        }
        ri = RunInfo(adata, run_id=0, analysis_type="da")
        assert ri.overwritten_fields == []

    def test_check_overwritten_string_mapping_values(self):
        # lines 241-244: mapping values are JSON strings in _check_overwritten_fields
        from kompot.anndata.utils.runinfo import RunInfo
        fm = {
            "da_pval": json.dumps({"location": "obs"}),
            "bad_field": "not{json",
        }
        run0 = _make_run_entry(0, field_mapping=fm)
        run1 = _make_run_entry(1, field_mapping=fm)
        adata = AnnData(
            X=np.zeros((5, 3)),
            obs=pd.DataFrame({"group": ["A"] * 5, "da_pval": [0.1] * 5}, index=[f"c{i}" for i in range(5)]),
            var=pd.DataFrame(index=[f"g{i}" for i in range(3)]),
        )
        adata.uns["kompot_da"] = {
            "run_history": to_json_string([run0, run1]),
            "anndata_fields": to_json_string({"obs": {"da_pval": 1}}),
        }
        ri = RunInfo(adata, run_id=0, analysis_type="da")
        assert len(ri.overwritten_fields) > 0

    def test_check_overwritten_non_dict_mapping(self):
        # line 248: mapping is not a dict after parsing
        from kompot.anndata.utils.runinfo import RunInfo
        fm = {"da_pval": {"location": "obs"}, "bad_field": 42}
        run0 = _make_run_entry(0, field_mapping=fm)
        adata = AnnData(
            X=np.zeros((5, 3)),
            obs=pd.DataFrame({"group": ["A"] * 5, "da_pval": [0.1] * 5}, index=[f"c{i}" for i in range(5)]),
            var=pd.DataFrame(index=[f"g{i}" for i in range(3)]),
        )
        adata.uns["kompot_da"] = {
            "run_history": to_json_string([run0]),
            "anndata_fields": to_json_string({"obs": {"da_pval": 0}}),
        }
        ri = RunInfo(adata, run_id=0, analysis_type="da")
        # Just verify it doesn't crash
        assert isinstance(ri.overwritten_fields, list)

    def test_check_overwritten_field_not_in_tracking(self):
        # line 253: field location or field name not in tracking dict
        from kompot.anndata.utils.runinfo import RunInfo
        fm = {"da_pval": {"location": "obs"}}
        run0 = _make_run_entry(0, field_mapping=fm)
        adata = AnnData(
            X=np.zeros((5, 3)),
            obs=pd.DataFrame({"group": ["A"] * 5, "da_pval": [0.1] * 5}, index=[f"c{i}" for i in range(5)]),
            var=pd.DataFrame(index=[f"g{i}" for i in range(3)]),
        )
        # No anndata_fields tracking -> empty tracking dict
        adata.uns["kompot_da"] = {
            "run_history": to_json_string([run0]),
        }
        ri = RunInfo(adata, run_id=0, analysis_type="da")
        assert ri.overwritten_fields == []

    def test_check_missing_fields_string_mapping(self):
        # lines 289-291, 303-304, 312: _check_missing_fields with various branches
        from kompot.anndata.utils.runinfo import RunInfo
        fm_str = to_json_string({"da_pval": {"location": "obs"}, "missing_col": {"location": "obs"}})
        run0 = _make_run_entry(0, field_mapping=fm_str)
        adata = AnnData(
            X=np.zeros((5, 3)),
            obs=pd.DataFrame({"group": ["A"] * 5, "da_pval": [0.1] * 5}, index=[f"c{i}" for i in range(5)]),
            var=pd.DataFrame(index=[f"g{i}" for i in range(3)]),
        )
        adata.uns["kompot_da"] = {
            "run_history": to_json_string([run0]),
            "anndata_fields": to_json_string({"obs": {"da_pval": 0, "missing_col": 0}}),
        }
        ri = RunInfo(adata, run_id=0, analysis_type="da")
        # "missing_col" not in adata.obs -> should be in missing_fields
        assert any(f["field"] == "missing_col" for f in ri.missing_fields)

    def test_check_missing_fields_empty_mapping(self):
        # lines 294-295: empty field_mapping in _check_missing_fields
        from kompot.anndata.utils.runinfo import RunInfo
        run0 = _make_run_entry(0, field_mapping={})
        adata = AnnData(
            X=np.zeros((5, 3)),
            obs=pd.DataFrame({"group": ["A"] * 5}, index=[f"c{i}" for i in range(5)]),
            var=pd.DataFrame(index=[f"g{i}" for i in range(3)]),
        )
        adata.uns["kompot_da"] = {
            "run_history": to_json_string([run0]),
            "anndata_fields": to_json_string({}),
        }
        ri = RunInfo(adata, run_id=0, analysis_type="da")
        assert ri.missing_fields == []

    def test_check_missing_fields_string_mapping_values(self):
        # lines 303-304: mapping values are JSON strings
        from kompot.anndata.utils.runinfo import RunInfo
        fm = {"da_pval": json.dumps({"location": "obs"}), "bad_field": "not{json"}
        run0 = _make_run_entry(0, field_mapping=fm)
        adata = AnnData(
            X=np.zeros((5, 3)),
            obs=pd.DataFrame({"group": ["A"] * 5, "da_pval": [0.1] * 5}, index=[f"c{i}" for i in range(5)]),
            var=pd.DataFrame(index=[f"g{i}" for i in range(3)]),
        )
        adata.uns["kompot_da"] = {
            "run_history": to_json_string([run0]),
            "anndata_fields": to_json_string({"obs": {"da_pval": 0}}),
        }
        ri = RunInfo(adata, run_id=0, analysis_type="da")
        # Should not crash

    def test_check_missing_fields_no_location(self):
        # line 312: mapping without location key
        from kompot.anndata.utils.runinfo import RunInfo
        fm = {"da_pval": {"type": "float"}}  # no 'location'
        run0 = _make_run_entry(0, field_mapping=fm)
        adata = AnnData(
            X=np.zeros((5, 3)),
            obs=pd.DataFrame({"group": ["A"] * 5}, index=[f"c{i}" for i in range(5)]),
            var=pd.DataFrame(index=[f"g{i}" for i in range(3)]),
        )
        adata.uns["kompot_da"] = {
            "run_history": to_json_string([run0]),
            "anndata_fields": to_json_string({}),
        }
        ri = RunInfo(adata, run_id=0, analysis_type="da")
        assert ri.missing_fields == []


class TestRunInfoHtml:
    """Cover lines 729, 742, 746, 823-828, 858-862 in _repr_html_."""

    def _make_runinfo_with_groups(self):
        from kompot.anndata.utils.runinfo import RunInfo
        fm = {"da_pval": {"location": "obs", "type": "float", "description": "p-value"}}
        run = _make_run_entry(0, field_mapping=fm)
        run["has_groups"] = True
        run["groups_summary"] = {"count": 5, "names": ["G1", "G2", "G3", "G4", "G5"]}
        adata = AnnData(
            X=np.zeros((5, 3)),
            obs=pd.DataFrame({"group": ["A"] * 5, "da_pval": [0.1] * 5}, index=[f"c{i}" for i in range(5)]),
            var=pd.DataFrame(index=[f"g{i}" for i in range(3)]),
        )
        adata.uns["kompot_da"] = {
            "run_history": to_json_string([run]),
            "anndata_fields": to_json_string({"obs": {"da_pval": 0}}),
        }
        return RunInfo(adata, run_id=0, analysis_type="da")

    def test_html_with_groups(self):
        # line 729: has_groups branch in _repr_html_
        ri = self._make_runinfo_with_groups()
        html = ri._repr_html_()
        assert "Groups" in html
        assert "5 total" in html

    def test_html_long_list_value(self):
        # line 746: long list truncation in _fmt_val
        ri = self._make_runinfo_with_groups()
        ri.params["long_list"] = list(range(20))
        html = ri._repr_html_()
        assert "20 items" in html

    def test_html_dict_param_with_subkeys(self):
        # lines 750-761: dict param rendered as settings group
        ri = self._make_runinfo_with_groups()
        ri.params["gp"] = {"sigma": 1.0, "ls": 0.5, "none_val": None}
        html = ri._repr_html_()
        assert "gp.sigma" in html
        assert "gp.ls" in html

    def test_html_field_mapping_string(self):
        # lines 823-828, 858-862: field_mapping string parsing in HTML
        from kompot.anndata.utils.runinfo import RunInfo
        fm_str = to_json_string({
            "da_pval": {"location": "obs", "type": "float", "description": "p-value"},
        })
        run = _make_run_entry(0, field_mapping=fm_str)
        adata = AnnData(
            X=np.zeros((5, 3)),
            obs=pd.DataFrame({"group": ["A"] * 5, "da_pval": [0.1] * 5}, index=[f"c{i}" for i in range(5)]),
            var=pd.DataFrame(index=[f"g{i}" for i in range(3)]),
        )
        adata.uns["kompot_da"] = {
            "run_history": to_json_string([run]),
            "anndata_fields": to_json_string({"obs": {"da_pval": 0}}),
        }
        ri = RunInfo(adata, run_id=0, analysis_type="da")
        html = ri._repr_html_()
        assert "da_pval" in html

    def test_html_field_mapping_string_values(self):
        # lines 858-862: field mapping values as JSON strings in HTML
        from kompot.anndata.utils.runinfo import RunInfo
        fm = {"da_pval": json.dumps({"location": "obs", "type": "float", "description": "p-value"})}
        run = _make_run_entry(0, field_mapping=fm)
        adata = AnnData(
            X=np.zeros((5, 3)),
            obs=pd.DataFrame({"group": ["A"] * 5, "da_pval": [0.1] * 5}, index=[f"c{i}" for i in range(5)]),
            var=pd.DataFrame(index=[f"g{i}" for i in range(3)]),
        )
        adata.uns["kompot_da"] = {
            "run_history": to_json_string([run]),
            "anndata_fields": to_json_string({"obs": {"da_pval": 0}}),
        }
        ri = RunInfo(adata, run_id=0, analysis_type="da")
        html = ri._repr_html_()
        assert "da_pval" in html


# ===================================================================
# RunComparison tests
# ===================================================================

class TestRunComparison:
    """Cover lines 1414, 1416, 1418, 1420, 1491-1497, 1501-1507, 1511-1517,
    1526, 1572, 1592-1626, 1660, 1664-1667."""

    def _make_comparison_adata(self, params1=None, params2=None, fm1=None, fm2=None):
        """Create adata with two runs for comparison testing."""
        default_fm = {"da_pval": {"location": "obs", "type": "float"}}
        fm1 = fm1 or default_fm
        fm2 = fm2 or default_fm

        p1 = params1 or {"condition1": "A", "condition2": "B", "obsm_key": "X_pca", "result_key": "kompot_da"}
        p2 = params2 or {"condition1": "A", "condition2": "B", "obsm_key": "X_pca", "result_key": "kompot_da"}

        run0 = _make_run_entry(0, params=p1, field_mapping=fm1)
        run1 = _make_run_entry(1, params=p2, field_mapping=fm2)

        adata = AnnData(
            X=np.zeros((5, 3)),
            obs=pd.DataFrame({"group": ["A"] * 5, "da_pval": [0.1] * 5}, index=[f"c{i}" for i in range(5)]),
            var=pd.DataFrame(index=[f"g{i}" for i in range(3)]),
        )
        adata.uns["kompot_da"] = {
            "run_history": to_json_string([run0, run1]),
            "anndata_fields": to_json_string({"obs": {"da_pval": 1}}),
        }
        return adata

    def test_comparison_badges_groupby(self):
        # line 1414: 'groupby' in different params
        from kompot.anndata.utils.runinfo import RunComparison
        p1 = {"condition1": "A", "condition2": "B", "obsm_key": "X_pca", "result_key": "k", "groupby": "g1"}
        p2 = {"condition1": "A", "condition2": "B", "obsm_key": "X_pca", "result_key": "k", "groupby": "g2"}
        adata = self._make_comparison_adata(params1=p1, params2=p2)
        rc = RunComparison(adata, 0, 1, "da")
        html = rc._repr_html_()
        assert "Different Group Variable" in html

    def test_comparison_badges_obsm_key(self):
        # line 1416: 'obsm_key' in different params
        from kompot.anndata.utils.runinfo import RunComparison
        p1 = {"condition1": "A", "condition2": "B", "obsm_key": "X_pca", "result_key": "k"}
        p2 = {"condition1": "A", "condition2": "B", "obsm_key": "X_umap", "result_key": "k"}
        adata = self._make_comparison_adata(params1=p1, params2=p2)
        rc = RunComparison(adata, 0, 1, "da")
        html = rc._repr_html_()
        assert "Different Embedding" in html

    def test_comparison_badges_layer(self):
        # line 1418: 'layer' in different params
        from kompot.anndata.utils.runinfo import RunComparison
        p1 = {"condition1": "A", "condition2": "B", "obsm_key": "X_pca", "result_key": "k", "layer": "raw"}
        p2 = {"condition1": "A", "condition2": "B", "obsm_key": "X_pca", "result_key": "k", "layer": "norm"}
        adata = self._make_comparison_adata(params1=p1, params2=p2)
        rc = RunComparison(adata, 0, 1, "da")
        html = rc._repr_html_()
        assert "Different Layer" in html

    def test_comparison_badges_use_sample_variance(self):
        # line 1420: 'use_sample_variance' in different params
        from kompot.anndata.utils.runinfo import RunComparison
        p1 = {"condition1": "A", "condition2": "B", "obsm_key": "X_pca", "result_key": "k", "use_sample_variance": True}
        p2 = {"condition1": "A", "condition2": "B", "obsm_key": "X_pca", "result_key": "k", "use_sample_variance": False}
        adata = self._make_comparison_adata(params1=p1, params2=p2)
        rc = RunComparison(adata, 0, 1, "da")
        html = rc._repr_html_()
        assert "Different Variance Method" in html

    def test_comparison_only_in_run1_params(self):
        # lines 1491-1497: params only in run1
        from kompot.anndata.utils.runinfo import RunComparison
        p1 = {"condition1": "A", "condition2": "B", "obsm_key": "X_pca", "result_key": "k", "extra_param": "val"}
        p2 = {"condition1": "A", "condition2": "B", "obsm_key": "X_pca", "result_key": "k"}
        adata = self._make_comparison_adata(params1=p1, params2=p2)
        rc = RunComparison(adata, 0, 1, "da")
        html = rc._repr_html_()
        assert "Only in Run" in html
        assert "not set" in html

    def test_comparison_only_in_run2_params(self):
        # lines 1501-1507: params only in run2
        from kompot.anndata.utils.runinfo import RunComparison
        p1 = {"condition1": "A", "condition2": "B", "obsm_key": "X_pca", "result_key": "k"}
        p2 = {"condition1": "A", "condition2": "B", "obsm_key": "X_pca", "result_key": "k", "extra_param": "val2"}
        adata = self._make_comparison_adata(params1=p1, params2=p2)
        rc = RunComparison(adata, 0, 1, "da")
        html = rc._repr_html_()
        assert "not set" in html

    def test_comparison_same_params_few(self):
        # lines 1511-1517: <=5 same params shown inline
        from kompot.anndata.utils.runinfo import RunComparison
        p1 = {"condition1": "A", "result_key": "k", "obsm_key": "X_pca"}
        p2 = {"condition1": "A", "result_key": "k", "obsm_key": "X_pca", "extra": "v"}
        adata = self._make_comparison_adata(params1=p1, params2=p2)
        rc = RunComparison(adata, 0, 1, "da")
        html = rc._repr_html_()
        assert "Same Parameters" in html

    def test_comparison_all_params_identical(self):
        # line 1526: all params identical -> "All parameters are identical"
        from kompot.anndata.utils.runinfo import RunComparison
        p = {"condition1": "A", "condition2": "B", "obsm_key": "X_pca", "result_key": "k"}
        adata = self._make_comparison_adata(params1=p, params2=p)
        rc = RunComparison(adata, 0, 1, "da")
        # Force param_comparison to have no diffs / only_in_run for the else branch
        rc.param_comparison = {"same": {}, "different": {}, "only_in_run1": {}, "only_in_run2": {}}
        html = rc._repr_html_()
        assert "All parameters are identical" in html

    def test_comparison_shared_fields_with_ownership(self):
        # lines 1572, 1592-1626: shared fields with ownership info
        from kompot.anndata.utils.runinfo import RunComparison
        fm = {"da_pval": {"location": "obs", "type": "float"}}
        adata = self._make_comparison_adata(fm1=fm, fm2=fm)
        rc = RunComparison(adata, 0, 1, "da")
        html = rc._repr_html_()
        # Should show shared fields section
        assert "Shared Fields" in html or "Defined in both" in html or "da_pval" in html

    def test_comparison_shared_fields_other_owner(self):
        # lines 1616-1618: owner is a third run (not run1 or run2)
        from kompot.anndata.utils.runinfo import RunComparison
        fm = {"da_pval": {"location": "obs", "type": "float"}}
        adata = self._make_comparison_adata(fm1=fm, fm2=fm)
        # Set ownership to a third run (id=99)
        adata.uns["kompot_da"]["anndata_fields"] = to_json_string({"obs": {"da_pval": 99}})
        rc = RunComparison(adata, 0, 1, "da")
        html = rc._repr_html_()
        assert "Other" in html or "Run 99" in html

    def test_comparison_no_fields(self):
        # line 1660: no fields at all -> "No fields found to compare"
        from kompot.anndata.utils.runinfo import RunComparison
        fm1 = {}
        fm2 = {}
        p = {"condition1": "A", "condition2": "B", "obsm_key": "X_pca", "result_key": "k"}
        run0 = _make_run_entry(0, params=p, field_mapping=fm1)
        run1 = _make_run_entry(1, params=p, field_mapping=fm2)
        adata = AnnData(
            X=np.zeros((5, 3)),
            obs=pd.DataFrame({"group": ["A"] * 5}, index=[f"c{i}" for i in range(5)]),
            var=pd.DataFrame(index=[f"g{i}" for i in range(3)]),
        )
        adata.uns["kompot_da"] = {
            "run_history": to_json_string([run0, run1]),
            "anndata_fields": to_json_string({}),
        }
        rc = RunComparison(adata, 0, 1, "da")
        html = rc._repr_html_()
        assert "No fields found to compare" in html

    def test_comparison_overlapping_fields_note(self):
        # lines 1664-1667: note about shared fields
        from kompot.anndata.utils.runinfo import RunComparison
        fm = {"da_pval": {"location": "obs", "type": "float"}}
        adata = self._make_comparison_adata(fm1=fm, fm2=fm)
        rc = RunComparison(adata, 0, 1, "da")
        html = rc._repr_html_()
        assert "shared fields" in html.lower() or "Note on shared fields" in html
