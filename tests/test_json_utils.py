"""Simple tests for JSON utilities to improve coverage."""

import numpy as np
import pytest
import json


class TestJSONUtilsBasic:
    """Basic tests for JSON utilities."""

    def test_jsonable_encoder_basic(self):
        """Test basic jsonable encoder functionality."""
        try:
            from kompot.anndata.utils.json_utils import jsonable_encoder
        except ImportError:
            pytest.skip("Cannot import json utils")

        # Test basic types
        assert jsonable_encoder(42) == 42
        assert jsonable_encoder("string") == "string"
        assert jsonable_encoder([1, 2, 3]) == [1, 2, 3]

    def test_jsonable_encoder_numpy(self):
        """Test jsonable encoder with numpy types."""
        try:
            from kompot.anndata.utils.json_utils import jsonable_encoder
        except ImportError:
            pytest.skip("Cannot import json utils")

        # Test numpy types
        assert jsonable_encoder(np.int32(42)) == 42
        assert jsonable_encoder(np.float64(3.14)) == 3.14

    def test_to_json_string_basic(self):
        """Test basic to_json_string functionality."""
        try:
            from kompot.anndata.utils.json_utils import to_json_string
        except ImportError:
            pytest.skip("Cannot import json utils")

        obj = {"key": "value", "number": 42}
        json_str = to_json_string(obj)
        assert isinstance(json_str, str)

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed == obj

    def test_from_json_string_basic(self):
        """Test basic from_json_string functionality."""
        try:
            from kompot.anndata.utils.json_utils import from_json_string
        except ImportError:
            pytest.skip("Cannot import json utils")

        json_str = '{"key": "value", "number": 42}'
        obj = from_json_string(json_str)
        assert obj == {"key": "value", "number": 42}

    def test_from_json_string_error(self):
        """Test from_json_string with invalid JSON."""
        try:
            from kompot.anndata.utils.json_utils import from_json_string
        except ImportError:
            pytest.skip("Cannot import json utils")

        # from_json_string returns the original string if JSON parsing fails
        result = from_json_string("invalid json")
        assert result == "invalid json"
