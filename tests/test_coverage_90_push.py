"""Targeted tests to push coverage to 90%.

Covers uncovered lines in:
- kompot/anndata/utils/runinfo.py
- kompot/anndata/_da_helpers.py
- kompot/anndata/utils/json_utils.py
- kompot/anndata/utils/field_tracking.py
"""

import json
import pytest
import numpy as np
import pandas as pd
from anndata import AnnData

from kompot.anndata.utils.json_utils import (
    jsonable_encoder,
    to_json_string,
    from_json_string,
    get_json_metadata,
    set_json_metadata,
)
from kompot.anndata.utils.field_tracking import (
    get_run_history,
    append_to_run_history,
    detect_output_field_overwrite,
    validate_field_run_id,
    get_run_from_history,
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
# json_utils tests
# ===================================================================

class TestJsonableEncoder:
    """Cover lines 35, 39 in json_utils.py."""

    def test_tuple_converted_to_list(self):
        # line 35: tuple branch
        assert jsonable_encoder((1, 2, 3)) == [1, 2, 3]

    def test_zero_dim_ndarray(self):
        # line 39: 0-d array branch
        result = jsonable_encoder(np.array(42))
        assert result == 42
        assert isinstance(result, int)

    def test_non_serializable_object(self):
        # line 53: else branch -> str()
        class Custom:
            def __str__(self):
                return "custom_obj"
        assert jsonable_encoder(Custom()) == "custom_obj"


class TestFromJsonString:
    """Cover line 88 in json_utils.py."""

    def test_non_string_returns_as_is(self):
        # line 88: not isinstance(json_str, str) branch
        assert from_json_string(42) == 42
        assert from_json_string([1, 2]) == [1, 2]
        assert from_json_string(None) is None


class TestGetJsonMetadata:
    """Cover lines 134-136, 146-147, 150-162, 192-193, 206."""

    def test_missing_intermediate_key(self):
        adata = AnnData(X=np.zeros((2, 2)))
        # returns None when intermediate key missing
        assert get_json_metadata(adata, "nonexistent.key") is None

    def test_missing_final_key(self):
        adata = AnnData(X=np.zeros((2, 2)))
        adata.uns["kompot_da"] = {}
        assert get_json_metadata(adata, "kompot_da.missing_key") is None

    def test_string_value_deserialized(self):
        # line 131-133: string value gets deserialized
        adata = AnnData(X=np.zeros((2, 2)))
        adata.uns["kompot_da"] = {"config": json.dumps({"a": 1})}
        result = get_json_metadata(adata, "kompot_da.config")
        assert result == {"a": 1}

    def test_string_value_not_json(self):
        # line 134-136: string that fails to parse
        adata = AnnData(X=np.zeros((2, 2)))
        adata.uns["kompot_da"] = {"config": "not-json{{{"}
        result = get_json_metadata(adata, "kompot_da.config")
        assert result == "not-json{{{"

    def test_list_with_json_strings(self):
        # lines 139-148: list with string elements that are JSON
        adata = AnnData(X=np.zeros((2, 2)))
        adata.uns["kompot_da"] = {"items": [json.dumps({"x": 1}), "plain", json.dumps([1, 2])]}
        result = get_json_metadata(adata, "kompot_da.items")
        assert result == [{"x": 1}, "plain", [1, 2]]

    def test_list_with_non_json_strings(self):
        # line 146-147: string in list that is not valid JSON
        adata = AnnData(X=np.zeros((2, 2)))
        adata.uns["kompot_da"] = {"items": ["not{json", "also{bad"]}
        result = get_json_metadata(adata, "kompot_da.items")
        assert result == ["not{json", "also{bad"]

    def test_dict_with_json_string_values(self):
        # lines 150-160: dict values that are JSON strings
        adata = AnnData(X=np.zeros((2, 2)))
        adata.uns["kompot_da"] = {"meta": {"a": json.dumps({"nested": True}), "b": "plain"}}
        result = get_json_metadata(adata, "kompot_da.meta")
        assert result == {"a": {"nested": True}, "b": "plain"}

    def test_dict_with_non_json_string_values(self):
        # line 157-158: dict value that fails JSON parse
        adata = AnnData(X=np.zeros((2, 2)))
        adata.uns["kompot_da"] = {"meta": {"a": "not{json"}}
        result = get_json_metadata(adata, "kompot_da.meta")
        assert result == {"a": "not{json"}

    def test_plain_value_returned(self):
        # line 162: non-string, non-list, non-dict value
        adata = AnnData(X=np.zeros((2, 2)))
        adata.uns["kompot_da"] = {"count": 42}
        result = get_json_metadata(adata, "kompot_da.count")
        assert result == 42


class TestSetJsonMetadata:
    """Cover lines 192-193, 206."""

    def test_creates_intermediate_dicts(self):
        adata = AnnData(X=np.zeros((2, 2)))
        result = set_json_metadata(adata, "kompot_da.run_history", [{"a": 1}])
        assert result is True
        assert "kompot_da" in adata.uns

    def test_non_dict_intermediate_returns_false(self):
        # line 192-193: intermediate is not a dict
        adata = AnnData(X=np.zeros((2, 2)))
        adata.uns["kompot_da"] = "not_a_dict"
        result = set_json_metadata(adata, "kompot_da.run_history", [])
        assert result is False

    def test_simple_value_stored_directly(self):
        # line 206: simple values stored without serialization
        adata = AnnData(X=np.zeros((2, 2)))
        set_json_metadata(adata, "kompot_da.count", 42)
        assert adata.uns["kompot_da"]["count"] == 42


# ===================================================================
# field_tracking tests
# ===================================================================

class TestGetRunHistory:
    """Cover lines 40, 50-55 in field_tracking.py."""

    def test_returns_empty_when_no_storage_key(self):
        adata = AnnData(X=np.zeros((2, 2)))
        assert get_run_history(adata, "da") == []

    def test_returns_empty_when_none(self):
        # line 40: run_history is None
        adata = AnnData(X=np.zeros((2, 2)))
        adata.uns["kompot_da"] = {"run_history": None}
        assert get_run_history(adata, "da") == []

    def test_non_list_string_parsed(self):
        # lines 44-52: run_history is a string that parses to a list
        adata = AnnData(X=np.zeros((2, 2)))
        adata.uns["kompot_da"] = {"run_history": json.dumps([{"run_id": 0}])}
        result = get_run_history(adata, "da")
        assert len(result) == 1

    def test_non_list_string_not_list_after_parse(self):
        # lines 47-49: parses to non-list
        adata = AnnData(X=np.zeros((2, 2)))
        adata.uns["kompot_da"] = {"run_history": json.dumps({"not": "a list"})}
        assert get_run_history(adata, "da") == []

    def test_non_list_non_string(self):
        # line 54-55: unexpected type
        adata = AnnData(X=np.zeros((2, 2)))
        adata.uns["kompot_da"] = {"run_history": 42}
        assert get_run_history(adata, "da") == []

    def test_items_that_are_json_strings(self):
        # lines 60-66: items in list are JSON strings
        adata = AnnData(X=np.zeros((2, 2)))
        run = {"run_id": 0, "params": {}}
        adata.uns["kompot_da"] = {"run_history": json.dumps([json.dumps(run)])}
        result = get_run_history(adata, "da")
        assert len(result) == 1
        assert result[0]["run_id"] == 0

    def test_items_that_are_invalid_json_strings(self):
        # line 64: invalid JSON string items
        adata = AnnData(X=np.zeros((2, 2)))
        adata.uns["kompot_da"] = {"run_history": json.dumps(["not{valid{json"])}
        result = get_run_history(adata, "da")
        assert len(result) == 0  # skipped


class TestAppendToRunHistory:
    """Cover lines in append_to_run_history."""

    def test_append_creates_storage_key(self):
        adata = AnnData(X=np.zeros((2, 2)))
        result = append_to_run_history(adata, {"run_id": 0}, "da")
        assert result is True
        assert "kompot_da" in adata.uns


class TestDetectOutputFieldOverwrite:
    """Cover lines 386, 423, 427, 429, 431 in field_tracking.py."""

    def test_result_type_abundance(self):
        # line 381: abundance -> kompot_da
        adata = AnnData(X=np.zeros((2, 2)), obs=pd.DataFrame({"da_col": [1, 2]}, index=["c0", "c1"]))
        has, fields, prev = detect_output_field_overwrite(
            adata, result_type="differential abundance",
            output_patterns=["da_col"], overwrite=False
        )
        assert has is True
        assert "obs.da_col" in fields

    def test_result_type_expression(self):
        # line 383-384: expression -> kompot_de
        adata = AnnData(X=np.zeros((2, 2)), obs=pd.DataFrame({"de_col": [1, 2]}, index=["c0", "c1"]))
        has, fields, prev = detect_output_field_overwrite(
            adata, result_type="differential expression",
            output_patterns=["de_col"], overwrite=False
        )
        assert has is True

    def test_unknown_result_type_raises(self):
        # line 386: unknown result_type
        adata = AnnData(X=np.zeros((2, 2)))
        with pytest.raises(ValueError, match="Unknown result_type"):
            detect_output_field_overwrite(adata, result_type="unknown_type", output_patterns=[])

    def test_no_type_raises(self):
        # line 388: neither analysis_type nor result_type
        adata = AnnData(X=np.zeros((2, 2)))
        with pytest.raises(ValueError, match="Either analysis_type or result_type"):
            detect_output_field_overwrite(adata, output_patterns=[])

    def test_overwrite_true_returns_early(self):
        # line 391-392: overwrite=True short-circuits
        adata = AnnData(X=np.zeros((2, 2)), obs=pd.DataFrame({"da_col": [1, 2]}, index=["c0", "c1"]))
        has, fields, prev = detect_output_field_overwrite(
            adata, analysis_type="da", output_patterns=["da_col"], overwrite=True
        )
        assert has is False

    def test_var_field_overwrite(self):
        # line 420: var location
        adata = AnnData(
            X=np.zeros((2, 3)),
            var=pd.DataFrame({"my_var_col": [0.1, 0.2, 0.3]}, index=["g0", "g1", "g2"]),
        )
        has, fields, _ = detect_output_field_overwrite(
            adata, analysis_type="de", output_patterns=["my_var_col"],
            overwrite=False, location="var"
        )
        assert has is True
        assert "var.my_var_col" in fields

    def test_uns_field_overwrite(self):
        # line 422-423: uns location
        adata = AnnData(X=np.zeros((2, 2)))
        adata.uns["my_key"] = "value"
        has, fields, _ = detect_output_field_overwrite(
            adata, analysis_type="da", output_patterns=["my_key"],
            overwrite=False, location="uns"
        )
        assert has is True
        assert "uns.my_key" in fields

    def test_obsm_field_overwrite(self):
        # line 426-427: obsm location
        adata = AnnData(X=np.zeros((2, 2)))
        adata.obsm["X_test"] = np.zeros((2, 3))
        has, fields, _ = detect_output_field_overwrite(
            adata, analysis_type="da", output_patterns=["X_test"],
            overwrite=False, location="obsm"
        )
        assert has is True
        assert "obsm.X_test" in fields

    def test_varm_field_overwrite(self):
        # line 428-429: varm location
        adata = AnnData(X=np.zeros((2, 3)))
        adata.varm["test"] = np.zeros((3, 2))
        has, fields, _ = detect_output_field_overwrite(
            adata, analysis_type="da", output_patterns=["test"],
            overwrite=False, location="varm"
        )
        assert has is True
        assert "varm.test" in fields

    def test_obsp_field_overwrite(self):
        # line 430-431: obsp location
        adata = AnnData(X=np.zeros((2, 2)))
        adata.obsp["test"] = np.zeros((2, 2))
        has, fields, _ = detect_output_field_overwrite(
            adata, analysis_type="da", output_patterns=["test"],
            overwrite=False, location="obsp"
        )
        assert has is True
        assert "obsp.test" in fields

    def test_layers_field_overwrite(self):
        # line 424-425: layers location
        adata = AnnData(X=np.zeros((2, 3)))
        adata.layers["test"] = np.zeros((2, 3))
        has, fields, _ = detect_output_field_overwrite(
            adata, analysis_type="da", output_patterns=["test"],
            overwrite=False, location="layers"
        )
        assert has is True
        assert "layers.test" in fields

    def test_with_tracking_and_result_key(self):
        # line 454: anndata_fields + result_key tracking lookup
        adata = AnnData(X=np.zeros((2, 2)), obs=pd.DataFrame({"col": [1, 2]}, index=["c0", "c1"]))
        run0 = {"run_id": 0, "params": {}, "timestamp": "t0"}
        adata.uns["kompot_da"] = {
            "run_history": to_json_string([run0]),
            "anndata_fields": to_json_string({"uns": {"my_result": 0}, "obs": {"col": 0}}),
        }
        has, fields, prev = detect_output_field_overwrite(
            adata, analysis_type="da", output_patterns=["col"],
            overwrite=False, result_key="my_result"
        )
        assert has is True
        assert prev is not None


class TestValidateFieldRunId:
    """Cover lines 582-583 in field_tracking.py (validate_field_run_id)."""

    def test_field_matches_run_id(self):
        adata = AnnData(X=np.zeros((2, 2)))
        adata.uns["kompot_da"] = {
            "anndata_fields": to_json_string({"obs": {"my_field": 0}})
        }
        valid, actual, msg = validate_field_run_id(adata, "my_field", "obs", 0, "kompot_da")
        assert valid is True
        assert actual == 0
        assert msg is None

    def test_field_mismatches_run_id(self):
        adata = AnnData(X=np.zeros((2, 2)))
        adata.uns["kompot_da"] = {
            "anndata_fields": to_json_string({"obs": {"my_field": 1}})
        }
        valid, actual, msg = validate_field_run_id(adata, "my_field", "obs", 0, "kompot_da")
        assert valid is False
        assert actual == 1
        assert "inconsistent" in msg

    def test_no_tracking_info(self):
        adata = AnnData(X=np.zeros((2, 2)))
        valid, actual, msg = validate_field_run_id(adata, "my_field", "obs", 0, "kompot_da")
        assert valid is True
        assert actual is None


class TestGetRunFromHistory:
    """Cover lines 582-583, 599-609, 613, 620-622, 625-626, 629, 652-655."""

    def test_run_id_none_returns_none(self):
        adata = AnnData(X=np.zeros((2, 2)))
        assert get_run_from_history(adata, run_id=None) is None

    def test_history_key_dotted(self):
        # lines 575-579: dotted history_key
        adata = AnnData(X=np.zeros((2, 2)))
        run0 = {"run_id": 0, "params": {}}
        adata.uns["custom_key"] = {"sub": {"history": to_json_string([run0])}}
        result = get_run_from_history(adata, run_id=0, history_key="custom_key.sub.history")
        assert result is not None

    def test_history_key_no_dot(self):
        # lines 581-583: no dot => storage_key=key, history_path=run_history
        adata = AnnData(X=np.zeros((2, 2)))
        run0 = {"run_id": 0, "params": {}}
        adata.uns["mykey"] = {"run_history": to_json_string([run0])}
        result = get_run_from_history(adata, run_id=0, history_key="mykey")
        assert result is not None

    def test_missing_storage_key(self):
        # line 590-591: storage key not in uns
        adata = AnnData(X=np.zeros((2, 2)))
        assert get_run_from_history(adata, run_id=0, analysis_type="da") is None

    def test_nested_history_path_missing_part(self):
        # line 602-603: nested path where intermediate is missing
        adata = AnnData(X=np.zeros((2, 2)))
        adata.uns["custom_key"] = {"sub": {}}
        result = get_run_from_history(adata, run_id=0, history_key="custom_key.sub.missing.history")
        assert result is None

    def test_nested_history_path_final_missing(self):
        # line 607-608: nested path where final key is missing
        adata = AnnData(X=np.zeros((2, 2)))
        adata.uns["custom_key"] = {"sub": {"other": "stuff"}}
        result = get_run_from_history(adata, run_id=0, history_key="custom_key.sub.history")
        assert result is None

    def test_direct_path_missing(self):
        # line 612-613: direct path not found
        adata = AnnData(X=np.zeros((2, 2)))
        adata.uns["kompot_da"] = {"not_run_history": []}
        assert get_run_from_history(adata, run_id=0, analysis_type="da") is None

    def test_json_string_run_history(self):
        # lines 617-622: run_history is a JSON string
        adata = AnnData(X=np.zeros((2, 2)))
        run0 = {"run_id": 0, "params": {}}
        adata.uns["kompot_da"] = {"run_history": to_json_string([run0])}
        result = get_run_from_history(adata, run_id=0, analysis_type="da")
        assert result is not None

    def test_json_string_invalid(self):
        # line 620-622: invalid JSON string for run_history
        adata = AnnData(X=np.zeros((2, 2)))
        adata.uns["kompot_da"] = {"run_history": "not{valid{json"}
        assert get_run_from_history(adata, run_id=0, analysis_type="da") is None

    def test_non_list_run_history(self):
        # line 624-626: run_history is not a list
        adata = AnnData(X=np.zeros((2, 2)))
        adata.uns["kompot_da"] = {"run_history": {"dict": "not list"}}
        assert get_run_from_history(adata, run_id=0, analysis_type="da") is None

    def test_empty_run_history(self):
        # line 628-629
        adata = AnnData(X=np.zeros((2, 2)))
        adata.uns["kompot_da"] = {"run_history": []}
        assert get_run_from_history(adata, run_id=0, analysis_type="da") is None

    def test_run_info_is_string_parsed(self):
        # lines 644-655: run_info entry is a JSON string
        adata = AnnData(X=np.zeros((2, 2)))
        run0 = {"run_id": 0, "params": {}}
        adata.uns["kompot_da"] = {"run_history": [json.dumps(run0)]}
        result = get_run_from_history(adata, run_id=0, analysis_type="da")
        assert result is not None
        assert result["run_id"] == 0

    def test_run_info_is_string_invalid(self):
        # lines 652-655: invalid JSON string as run entry
        adata = AnnData(X=np.zeros((2, 2)))
        adata.uns["kompot_da"] = {"run_history": ["not{valid"]}
        result = get_run_from_history(adata, run_id=0, analysis_type="da")
        assert result is not None  # returns empty dict
        assert isinstance(result, dict)

    def test_run_info_is_non_dict_non_string(self):
        # line 657-659: entry is not dict or string
        adata = AnnData(X=np.zeros((2, 2)))
        adata.uns["kompot_da"] = {"run_history": [42]}
        result = get_run_from_history(adata, run_id=0, analysis_type="da")
        assert result is not None
        assert isinstance(result, dict)

    def test_negative_index(self):
        # line 632-633: negative run_id
        adata = AnnData(X=np.zeros((2, 2)))
        run0 = {"run_id": 0, "params": {"a": 1}}
        run1 = {"run_id": 1, "params": {"a": 2}}
        adata.uns["kompot_da"] = {"run_history": [run0, run1]}
        result = get_run_from_history(adata, run_id=-1, analysis_type="da")
        assert result["run_id"] == 1

    def test_out_of_range_run_id(self):
        # line 636-637
        adata = AnnData(X=np.zeros((2, 2)))
        adata.uns["kompot_da"] = {"run_history": [{"run_id": 0}]}
        assert get_run_from_history(adata, run_id=99, analysis_type="da") is None


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


# ===================================================================
# _da_helpers tests
# ===================================================================

class TestCheckDaOverwrites:
    """Cover lines 72-82, 100-108, 119, 144-160, 163, 172, 174, 186, 220-221,
    227-235, 255, 276, 283-291, 501, 504."""

    def _make_da_adata_with_results(self, sample_col=None, prev_sample_var=False):
        """Create adata with existing DA results for overwrite testing."""
        n_obs = 10
        obs_data = {
            "group": ["A"] * 5 + ["B"] * 5,
            "da_pval": np.random.rand(n_obs),
            "da_lfc": np.random.rand(n_obs),
        }
        if sample_col:
            obs_data[sample_col] = ["s1", "s2"] * 5
        adata = AnnData(
            X=np.zeros((n_obs, 3)),
            obs=pd.DataFrame(obs_data, index=[f"c{i}" for i in range(n_obs)]),
        )
        adata.obsm["X_pca"] = np.random.rand(n_obs, 5)

        prev_params = {
            "condition1": "A",
            "condition2": "B",
            "groupby": "group",
            "obsm_key": "X_pca",
            "ls_factor": 1.0,
            "use_sample_variance": prev_sample_var,
            "result_key": "kompot_da",
        }
        prev_run = {"run_id": 0, "timestamp": "2025-01-01", "params": prev_params}
        adata.uns["kompot_da"] = {
            "run_history": to_json_string([prev_run]),
            "anndata_fields": to_json_string({
                "obs": {"da_pval": 0, "da_lfc": 0},
                "uns": {"kompot_da": 0},
            }),
        }
        return adata

    def test_overwrite_false_raises(self):
        # lines 91-93: overwrite=False raises ValueError
        from kompot.anndata._da_helpers import _check_da_overwrites
        adata = self._make_da_adata_with_results()
        field_names = {"all_patterns": {"obs": ["da_pval", "da_lfc"]}}
        with pytest.raises(ValueError, match="overwrite=True"):
            _check_da_overwrites(
                adata, result_key="kompot_da", field_names=field_names,
                sample_col=None, overwrite=False,
                groupby="group", condition1="A", condition2="B",
                obsm_key="X_pca", ls_factor=1.0,
            )

    def test_overwrite_none_with_sample_var_change_params_match(self):
        # lines 99-119: overwrite=None, adding sample variance, params match -> info log
        from kompot.anndata._da_helpers import _check_da_overwrites
        adata = self._make_da_adata_with_results(sample_col="sample", prev_sample_var=False)
        field_names = {"all_patterns": {"obs": ["da_pval", "da_lfc"]}}
        # Should not raise, just log info
        _check_da_overwrites(
            adata, result_key="kompot_da", field_names=field_names,
            sample_col="sample", overwrite=None,
            groupby="group", condition1="A", condition2="B",
            obsm_key="X_pca", ls_factor=1.0,
        )

    def test_overwrite_none_with_sample_var_change_params_mismatch(self):
        # lines 100-108: overwrite=None, adding sample_var but different groupby -> warning
        from kompot.anndata._da_helpers import _check_da_overwrites
        adata = self._make_da_adata_with_results(sample_col="sample", prev_sample_var=False)
        field_names = {"all_patterns": {"obs": ["da_pval", "da_lfc"]}}
        # Different groupby => params_match=False
        _check_da_overwrites(
            adata, result_key="kompot_da", field_names=field_names,
            sample_col="sample", overwrite=None,
            groupby="different_col", condition1="A", condition2="B",
            obsm_key="X_pca", ls_factor=1.0,
        )

    def test_overwrite_false_with_sample_var_fields_listed(self):
        # lines 72-82: overwrite=False with prev_sample_var different, fields listed
        from kompot.anndata._da_helpers import _check_da_overwrites
        adata = self._make_da_adata_with_results(sample_col="sample", prev_sample_var=False)
        field_names = {"all_patterns": {"obs": ["da_pval", "da_lfc"]}}
        with pytest.raises(ValueError, match="overwrite=True"):
            _check_da_overwrites(
                adata, result_key="kompot_da", field_names=field_names,
                sample_col="sample", overwrite=False,
                groupby="group", condition1="A", condition2="B",
                obsm_key="X_pca", ls_factor=1.0,
            )

    def test_overwrite_false_prev_sample_var_true_current_false(self):
        # lines 81-87: previous had sample_var=True, current doesn't
        from kompot.anndata._da_helpers import _check_da_overwrites
        adata = self._make_da_adata_with_results(prev_sample_var=True)
        field_names = {"all_patterns": {"obs": ["da_pval", "da_lfc"]}}
        with pytest.raises(ValueError, match="overwrite=True"):
            _check_da_overwrites(
                adata, result_key="kompot_da", field_names=field_names,
                sample_col=None, overwrite=False,
                groupby="group", condition1="A", condition2="B",
                obsm_key="X_pca", ls_factor=1.0,
            )

    def test_overwrite_none_no_sample_var_change(self):
        # line 126-131: overwrite=None, no sample_var change -> warning
        from kompot.anndata._da_helpers import _check_da_overwrites
        adata = self._make_da_adata_with_results(prev_sample_var=False)
        field_names = {"all_patterns": {"obs": ["da_pval", "da_lfc"]}}
        _check_da_overwrites(
            adata, result_key="kompot_da", field_names=field_names,
            sample_col=None, overwrite=None,
            groupby="group", condition1="A", condition2="B",
            obsm_key="X_pca", ls_factor=1.0,
        )


class TestExtractDaData:
    """Cover lines 144-160, 163, 172, 174, 186."""

    def test_missing_obsm_key(self):
        # lines 144-160: obsm_key not found
        from kompot.anndata._da_helpers import _extract_da_data
        adata = AnnData(
            X=np.zeros((5, 3)),
            obs=pd.DataFrame({"group": ["A"] * 3 + ["B"] * 2}, index=[f"c{i}" for i in range(5)]),
        )
        with pytest.raises(ValueError, match="not found in adata.obsm"):
            _extract_da_data(adata, "group", "A", "B", "X_pca", None)

    def test_missing_obsm_DM_EigenVectors(self):
        # lines 148-159: special DM_EigenVectors message
        from kompot.anndata._da_helpers import _extract_da_data
        adata = AnnData(
            X=np.zeros((5, 3)),
            obs=pd.DataFrame({"group": ["A"] * 3 + ["B"] * 2}, index=[f"c{i}" for i in range(5)]),
        )
        with pytest.raises(ValueError, match="Palantir"):
            _extract_da_data(adata, "group", "A", "B", "DM_EigenVectors", None)

    def test_missing_groupby(self):
        # line 163: groupby not in obs
        from kompot.anndata._da_helpers import _extract_da_data
        adata = AnnData(
            X=np.zeros((5, 3)),
            obs=pd.DataFrame({"group": ["A"] * 5}, index=[f"c{i}" for i in range(5)]),
        )
        adata.obsm["X_pca"] = np.zeros((5, 3))
        with pytest.raises(ValueError, match="not found in adata.obs"):
            _extract_da_data(adata, "nonexistent", "A", "B", "X_pca", None)

    def test_condition_not_found(self):
        # lines 172, 174: condition not found
        from kompot.anndata._da_helpers import _extract_da_data
        adata = AnnData(
            X=np.zeros((5, 3)),
            obs=pd.DataFrame({"group": ["A"] * 3 + ["B"] * 2}, index=[f"c{i}" for i in range(5)]),
        )
        adata.obsm["X_pca"] = np.zeros((5, 3))
        with pytest.raises(ValueError, match="not found"):
            _extract_da_data(adata, "group", "C", "B", "X_pca", None)
        with pytest.raises(ValueError, match="not found"):
            _extract_da_data(adata, "group", "A", "C", "X_pca", None)

    def test_missing_sample_col(self):
        # line 186: sample_col not in obs
        from kompot.anndata._da_helpers import _extract_da_data
        adata = AnnData(
            X=np.zeros((5, 3)),
            obs=pd.DataFrame({"group": ["A"] * 3 + ["B"] * 2}, index=[f"c{i}" for i in range(5)]),
        )
        adata.obsm["X_pca"] = np.zeros((5, 3))
        with pytest.raises(ValueError, match="not found in adata.obs"):
            _extract_da_data(adata, "group", "A", "B", "X_pca", "nonexistent_sample")


class TestResolveDaLandmarks:
    """Cover lines 220-221, 227-235, 255, 276."""

    def test_provided_landmarks(self):
        # lines 220-221: landmarks provided directly
        from kompot.anndata._da_helpers import _resolve_da_landmarks
        adata = AnnData(X=np.zeros((5, 3)))
        adata.obsm["X_pca"] = np.zeros((5, 3))
        lm = np.zeros((10, 3))
        result = _resolve_da_landmarks(adata, lm, "X_pca", "kompot_da")
        assert result is lm

    def test_stored_landmarks_matching_dim(self):
        # lines 226-233: stored landmarks with matching dimension
        from kompot.anndata._da_helpers import _resolve_da_landmarks
        adata = AnnData(X=np.zeros((5, 3)))
        adata.obsm["X_pca"] = np.zeros((5, 3))
        adata.uns["kompot_da"] = {"landmarks": np.zeros((10, 3))}
        result = _resolve_da_landmarks(adata, None, "X_pca", "kompot_da")
        assert result is not None
        assert result.shape == (10, 3)

    def test_stored_landmarks_wrong_dim(self):
        # lines 234-238: stored landmarks with wrong dimension
        from kompot.anndata._da_helpers import _resolve_da_landmarks
        adata = AnnData(X=np.zeros((5, 3)))
        adata.obsm["X_pca"] = np.zeros((5, 3))
        adata.uns["kompot_da"] = {"landmarks": np.zeros((10, 5))}  # wrong dim
        result = _resolve_da_landmarks(adata, None, "X_pca", "kompot_da")
        assert result is None

    def test_other_kompot_key_landmarks(self):
        # lines 242-254: kompot_da storage key differs from result_key, check other
        from kompot.anndata._da_helpers import _resolve_da_landmarks
        adata = AnnData(X=np.zeros((5, 3)))
        adata.obsm["X_pca"] = np.zeros((5, 3))
        adata.uns["kompot_da"] = {"landmarks": np.zeros((10, 3))}
        result = _resolve_da_landmarks(adata, None, "X_pca", "other_key")
        assert result is not None

    def test_other_kompot_key_wrong_dim(self):
        # line 255: other kompot key with wrong dimension
        from kompot.anndata._da_helpers import _resolve_da_landmarks
        adata = AnnData(X=np.zeros((5, 3)))
        adata.obsm["X_pca"] = np.zeros((5, 3))
        adata.uns["kompot_da"] = {"landmarks": np.zeros((10, 7))}
        result = _resolve_da_landmarks(adata, None, "X_pca", "other_key")
        assert result is None

    def test_kompot_wildcard_key(self):
        # lines 261-274: check all kompot_* keys
        from kompot.anndata._da_helpers import _resolve_da_landmarks
        adata = AnnData(X=np.zeros((5, 3)))
        adata.obsm["X_pca"] = np.zeros((5, 3))
        adata.uns["kompot_custom"] = {"landmarks": np.zeros((10, 3))}
        result = _resolve_da_landmarks(adata, None, "X_pca", "other_key")
        assert result is not None

    def test_kompot_wildcard_key_wrong_dim(self):
        # line 276: kompot_* with wrong dimension
        from kompot.anndata._da_helpers import _resolve_da_landmarks
        adata = AnnData(X=np.zeros((5, 3)))
        adata.obsm["X_pca"] = np.zeros((5, 3))
        adata.uns["kompot_custom"] = {"landmarks": np.zeros((10, 7))}
        result = _resolve_da_landmarks(adata, None, "X_pca", "other_key")
        assert result is None

    def test_de_landmarks(self):
        # lines 283-289: DE landmarks reuse
        from kompot.anndata._da_helpers import _resolve_da_landmarks
        adata = AnnData(X=np.zeros((5, 3)))
        adata.obsm["X_pca"] = np.zeros((5, 3))
        adata.uns["kompot_de"] = {"landmarks": np.zeros((10, 3))}
        result = _resolve_da_landmarks(adata, None, "X_pca", "other_key")
        assert result is not None

    def test_de_landmarks_wrong_dim(self):
        # lines 291-294: DE landmarks wrong dim
        from kompot.anndata._da_helpers import _resolve_da_landmarks
        adata = AnnData(X=np.zeros((5, 3)))
        adata.obsm["X_pca"] = np.zeros((5, 3))
        adata.uns["kompot_de"] = {"landmarks": np.zeros((10, 7))}
        result = _resolve_da_landmarks(adata, None, "X_pca", "other_key")
        assert result is None


class TestRecordDaRunInfo:
    """Cover lines 501, 504 in _da_helpers.py (_record_da_run_info)."""

    def test_record_merge_existing_anndata_fields(self):
        # lines 498-507: merge into existing anndata_fields
        from kompot.anndata._da_helpers import _record_da_run_info
        adata = AnnData(X=np.zeros((5, 3)))
        # Pre-populate with existing tracking from a previous run
        adata.uns["kompot_da"] = {
            "run_history": to_json_string([]),
            "anndata_fields": to_json_string({"obs": {"old_field": 0}}),
        }
        field_names = {
            "lfc_key": "da_lfc",
            "zscore_key": "da_zscore",
            "ptp_key": "da_ptp",
            "direction_key": "da_dir",
            "density_key_1": "da_d1",
            "density_key_2": "da_d2",
        }
        _record_da_run_info(
            adata, field_names,
            condition1="A", condition2="B",
            sample_col=None, result_key="kompot_da",
            params_dict={"condition1": "A", "condition2": "B"},
        )
        tracking = get_json_metadata(adata, "kompot_da.anndata_fields")
        # old field should still be there
        assert tracking["obs"]["old_field"] == 0
        # new fields should be added
        assert "da_lfc" in tracking["obs"]

    def test_record_existing_none_anndata_fields(self):
        # line 500-501: existing is None -> becomes empty dict
        from kompot.anndata._da_helpers import _record_da_run_info
        adata = AnnData(X=np.zeros((5, 3)))
        # anndata_fields exists but decodes to None (edge case)
        adata.uns["kompot_da"] = {
            "run_history": to_json_string([]),
            "anndata_fields": "null",
        }
        field_names = {
            "lfc_key": "da_lfc",
            "zscore_key": "da_zscore",
            "ptp_key": "da_ptp",
            "direction_key": "da_dir",
            "density_key_1": "da_d1",
            "density_key_2": "da_d2",
        }
        _record_da_run_info(
            adata, field_names,
            condition1="A", condition2="B",
            sample_col=None, result_key="kompot_da",
            params_dict={"condition1": "A", "condition2": "B"},
        )
        tracking = get_json_metadata(adata, "kompot_da.anndata_fields")
        assert "obs" in tracking
        assert "uns" in tracking
