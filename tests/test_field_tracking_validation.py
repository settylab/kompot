"""Tests for kompot.anndata.utils.field_tracking validation and history functions.

Covers uncovered lines in kompot/anndata/utils/field_tracking.py including:
- get_run_history edge cases (None, non-list, JSON string parsing)
- append_to_run_history storage key creation
- detect_output_field_overwrite for various locations and result types
- validate_field_run_id matching and mismatching
- get_run_from_history with various history formats and edge cases
"""

import json
import pytest
import numpy as np
import pandas as pd
from anndata import AnnData

from kompot.anndata.utils.json_utils import (
    to_json_string,
)
from kompot.anndata.utils.field_tracking import (
    get_run_history,
    append_to_run_history,
    detect_output_field_overwrite,
    validate_field_run_id,
    get_run_from_history,
)


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
