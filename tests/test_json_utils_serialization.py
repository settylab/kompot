"""Tests for kompot.anndata.utils.json_utils serialization and metadata functions.

Covers uncovered lines in kompot/anndata/utils/json_utils.py including:
- jsonable_encoder edge cases (tuples, 0-d arrays, non-serializable objects)
- from_json_string non-string passthrough
- get_json_metadata nested key traversal, type handling, and deserialization
- set_json_metadata intermediate dict creation and value storage
"""

import json
import numpy as np
from anndata import AnnData

from kompot.anndata.utils.json_utils import (
    jsonable_encoder,
    from_json_string,
    get_json_metadata,
    set_json_metadata,
)


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
        adata.uns["kompot_da"] = {
            "items": [json.dumps({"x": 1}), "plain", json.dumps([1, 2])]
        }
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
        adata.uns["kompot_da"] = {
            "meta": {"a": json.dumps({"nested": True}), "b": "plain"}
        }
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
