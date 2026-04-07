"""Tests for anndata/utils/field_tracking.py to improve coverage."""

import pytest
import numpy as np
import pandas as pd
from anndata import AnnData

from kompot.anndata.utils.field_tracking import (
    get_run_history,
    append_to_run_history,
    get_last_run_info,
    generate_output_field_names,
    get_environment_info,
    detect_output_field_overwrite,
    _sanitize_name,
    validate_field_run_id,
    get_run_from_history,
)
from kompot.anndata.utils.json_utils import to_json_string, set_json_metadata


@pytest.fixture
def sample_adata():
    """Create a simple AnnData object for testing."""
    n_obs = 50
    n_vars = 30

    X = np.random.randn(n_obs, n_vars)
    obs = pd.DataFrame(
        {
            "cell_id": [f"cell_{i}" for i in range(n_obs)],
            "condition": ["A"] * 25 + ["B"] * 25,
        }
    )
    var = pd.DataFrame({"gene_name": [f"Gene_{i}" for i in range(n_vars)]})

    adata = AnnData(X=X, obs=obs, var=var)
    return adata


class TestGetRunHistory:
    """Test get_run_history function."""

    def test_get_run_history_empty(self, sample_adata):
        """Test getting run history when none exists."""
        history = get_run_history(sample_adata, analysis_type="da")
        assert history == []

    def test_get_run_history_missing_key(self, sample_adata):
        """Test getting run history when storage key exists but run_history doesn't."""
        sample_adata.uns["kompot_da"] = {}
        history = get_run_history(sample_adata, analysis_type="da")
        assert history == []

    def test_get_run_history_with_data(self, sample_adata):
        """Test getting run history with valid data."""
        run_info = {"run_id": 0, "condition1": "A", "condition2": "B"}
        append_to_run_history(sample_adata, run_info, analysis_type="da")

        history = get_run_history(sample_adata, analysis_type="da")
        assert len(history) == 1
        assert history[0]["run_id"] == 0

    def test_get_run_history_not_a_list(self, sample_adata):
        """Test handling when run_history is not a list."""
        sample_adata.uns["kompot_da"] = {"run_history": "not_a_list_or_json"}
        history = get_run_history(sample_adata, analysis_type="da")
        assert history == []

    def test_get_run_history_list_with_string_items(self, sample_adata):
        """Test handling list with string items that need JSON parsing."""
        run_info = {"run_id": 0, "condition1": "A"}
        json_str = to_json_string(run_info)

        sample_adata.uns["kompot_da"] = {
            "run_history": [json_str]  # List with JSON string
        }

        history = get_run_history(sample_adata, analysis_type="da")
        assert len(history) == 1
        assert history[0]["run_id"] == 0

    def test_get_run_history_invalid_json_item(self, sample_adata):
        """Test handling list with invalid JSON strings."""
        sample_adata.uns["kompot_da"] = {
            "run_history": ["not_valid_json", '{"run_id": 1}']
        }

        history = get_run_history(sample_adata, analysis_type="da")
        # Should skip invalid item, keep valid one
        assert len(history) == 1
        assert history[0]["run_id"] == 1

    def test_get_run_history_non_dict_item(self, sample_adata):
        """Test handling list with non-dictionary items."""
        sample_adata.uns["kompot_da"] = {
            "run_history": [123, {"run_id": 1}]  # First item is int
        }

        history = get_run_history(sample_adata, analysis_type="da")
        # Should skip non-dict item
        assert len(history) == 1
        assert history[0]["run_id"] == 1

    def test_get_run_history_de_type(self, sample_adata):
        """Test getting run history for DE analysis."""
        run_info = {"run_id": 0, "analysis": "de"}
        append_to_run_history(sample_adata, run_info, analysis_type="de")

        history = get_run_history(sample_adata, analysis_type="de")
        assert len(history) == 1
        assert history[0]["analysis"] == "de"


class TestAppendToRunHistory:
    """Test append_to_run_history function."""

    def test_append_to_run_history_new(self, sample_adata):
        """Test appending to empty history."""
        run_info = {"run_id": 0, "condition1": "A", "condition2": "B"}
        success = append_to_run_history(sample_adata, run_info, analysis_type="da")

        assert success
        assert "kompot_da" in sample_adata.uns
        history = get_run_history(sample_adata, analysis_type="da")
        assert len(history) == 1
        assert history[0]["run_id"] == 0

    def test_append_to_run_history_multiple(self, sample_adata):
        """Test appending multiple items."""
        for i in range(3):
            run_info = {"run_id": i, "iteration": i}
            append_to_run_history(sample_adata, run_info, analysis_type="da")

        history = get_run_history(sample_adata, analysis_type="da")
        assert len(history) == 3
        assert history[2]["run_id"] == 2

    def test_append_to_run_history_de_type(self, sample_adata):
        """Test appending to DE history."""
        run_info = {"run_id": 0, "analysis": "de"}
        append_to_run_history(sample_adata, run_info, analysis_type="de")

        history = get_run_history(sample_adata, analysis_type="de")
        assert len(history) == 1


class TestGetLastRunInfo:
    """Test get_last_run_info function."""

    def test_get_last_run_info_empty(self, sample_adata):
        """Test getting last run info when none exists."""
        last_run = get_last_run_info(sample_adata, analysis_type="da")
        assert last_run is None

    def test_get_last_run_info_with_data(self, sample_adata):
        """Test getting last run info when it exists."""
        run_info = {"run_id": 0, "condition1": "A"}
        set_json_metadata(sample_adata, "kompot_da.last_run_info", run_info)

        last_run = get_last_run_info(sample_adata, analysis_type="da")
        assert last_run is not None
        assert last_run["run_id"] == 0


class TestGenerateOutputFieldNames:
    """Test generate_output_field_names function."""

    def test_generate_field_names_da_basic(self):
        """Test generating DA field names without sample variance."""
        field_names = generate_output_field_names(
            result_key="kompot_da",
            condition1="control",
            condition2="treatment",
            analysis_type="da",
            with_sample_suffix=False,
        )

        assert "lfc_key" in field_names
        assert "zscore_key" in field_names
        assert "ptp_key" in field_names
        assert "direction_key" in field_names
        assert "density_key_1" in field_names
        assert "density_key_2" in field_names
        assert "all_patterns" in field_names
        assert "obs" in field_names["all_patterns"]
        assert len(field_names["all_patterns"]["obs"]) == 6

        # Verify sample variance impacted fields
        assert "sample_variance_impacted_fields" in field_names
        assert "zscore_key" in field_names["sample_variance_impacted_fields"]

    def test_generate_field_names_da_with_sample_suffix(self):
        """Test generating DA field names with sample variance."""
        field_names = generate_output_field_names(
            result_key="kompot_da",
            condition1="control",
            condition2="treatment",
            analysis_type="da",
            with_sample_suffix=True,
            sample_suffix="_sample_var",
        )

        # Check that sample suffix is applied
        assert "_sample_var" in field_names["zscore_key"]
        assert "_sample_var" in field_names["ptp_key"]
        assert "_sample_var" in field_names["direction_key"]

        # LFC and density should NOT have suffix
        assert "_sample_var" not in field_names["lfc_key"]
        assert "_sample_var" not in field_names["density_key_1"]

    def test_generate_field_names_de_basic(self):
        """Test generating DE field names without sample variance."""
        field_names = generate_output_field_names(
            result_key="kompot_de",
            condition1="A",
            condition2="B",
            analysis_type="de",
            with_sample_suffix=False,
        )

        assert "mahalanobis_key" in field_names
        assert "ptp_key" in field_names
        assert "mean_lfc_key" in field_names
        assert "smoothed_key_1" in field_names
        assert "smoothed_key_2" in field_names
        assert "fold_change_key" in field_names
        assert "fold_change_zscores_key" in field_names
        assert "std_key_1" in field_names
        assert "std_key_2" in field_names
        assert "posterior_covariance_key" in field_names

        # FDR fields
        assert "mahalanobis_pvalue_key" in field_names
        assert "mahalanobis_local_fdr_key" in field_names
        assert "is_de_key" in field_names

        # Varm fields
        assert "mean_lfc_varm_key" in field_names
        assert "mahalanobis_varm_key" in field_names

        # Check all_patterns structure
        assert "all_patterns" in field_names
        assert "var" in field_names["all_patterns"]
        assert "layers" in field_names["all_patterns"]
        assert "obsp" in field_names["all_patterns"]

        # Without sample variance, std keys should be in obs
        assert "obs" in field_names["all_patterns"]
        assert field_names["std_key_1"] in field_names["all_patterns"]["obs"]

    def test_generate_field_names_de_with_sample_suffix(self):
        """Test generating DE field names with sample variance."""
        field_names = generate_output_field_names(
            result_key="kompot_de",
            condition1="A",
            condition2="B",
            analysis_type="de",
            with_sample_suffix=True,
            sample_suffix="_sample",
        )

        # Check sample suffix applied to appropriate fields
        assert "_sample" in field_names["mahalanobis_key"]
        assert "_sample" in field_names["ptp_key"]
        assert "_sample" in field_names["fold_change_zscores_key"]
        assert "_sample" in field_names["mahalanobis_varm_key"]

        # Check fields WITHOUT sample suffix
        assert "_sample" not in field_names["mean_lfc_key"]
        assert "_sample" not in field_names["smoothed_key_1"]
        assert "_sample" not in field_names["fold_change_key"]

        # With sample variance, std keys should be in layers
        assert field_names["std_key_1"] in field_names["all_patterns"]["layers"]

    def test_generate_field_names_sanitization(self):
        """Test that condition names are sanitized."""
        field_names = generate_output_field_names(
            result_key="kompot_da",
            condition1="control-group",
            condition2="treatment.group",
            analysis_type="da",
        )

        # Check that special characters are replaced with underscores
        assert "control_group" in field_names["lfc_key"]
        assert "treatment_group" in field_names["lfc_key"]

    def test_generate_field_names_invalid_type(self):
        """Test that invalid analysis type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown analysis_type"):
            generate_output_field_names(
                result_key="kompot_unknown",
                condition1="A",
                condition2="B",
                analysis_type="unknown",
            )


class TestGetEnvironmentInfo:
    """Test get_environment_info function."""

    def test_get_environment_info_structure(self):
        """Test that environment info has expected structure."""
        env_info = get_environment_info()

        assert "timestamp" in env_info
        assert "platform" in env_info
        assert "python_version" in env_info
        assert "hostname" in env_info
        assert "username" in env_info
        assert "pid" in env_info
        assert "package_versions" in env_info

        # Check that key packages are included
        packages = env_info["package_versions"]
        assert "kompot" in packages
        assert "anndata" in packages
        assert "numpy" in packages
        assert "jax" in packages

    def test_get_environment_info_types(self):
        """Test that environment info values have correct types."""
        env_info = get_environment_info()

        assert isinstance(env_info["timestamp"], str)
        assert isinstance(env_info["platform"], str)
        assert isinstance(env_info["python_version"], str)
        assert isinstance(env_info["pid"], int)
        assert isinstance(env_info["package_versions"], dict)


class TestDetectOutputFieldOverwrite:
    """Test detect_output_field_overwrite function."""

    def test_detect_overwrite_empty_adata(self, sample_adata):
        """Test detection with empty AnnData (no existing fields)."""
        field_names = generate_output_field_names(
            result_key="kompot_da", condition1="A", condition2="B", analysis_type="da"
        )

        will_overwrite, overwritten, prev_run = detect_output_field_overwrite(
            sample_adata, analysis_type="da", field_names=field_names, overwrite=False
        )

        assert not will_overwrite
        assert len(overwritten) == 0
        assert prev_run is None

    def test_detect_overwrite_with_existing_fields(self, sample_adata):
        """Test detection with existing fields in obs."""
        # Add some existing fields
        sample_adata.obs["kompot_da_A_to_B_lfc"] = 1.0

        field_names = generate_output_field_names(
            result_key="kompot_da", condition1="A", condition2="B", analysis_type="da"
        )

        will_overwrite, overwritten, prev_run = detect_output_field_overwrite(
            sample_adata, analysis_type="da", field_names=field_names, overwrite=False
        )

        assert will_overwrite
        assert len(overwritten) > 0
        assert "obs.kompot_da_A_to_B_lfc" in overwritten

    def test_detect_overwrite_allowed(self, sample_adata):
        """Test that overwrite=True skips detection."""
        sample_adata.obs["kompot_da_A_to_B_lfc"] = 1.0

        field_names = generate_output_field_names(
            result_key="kompot_da", condition1="A", condition2="B", analysis_type="da"
        )

        will_overwrite, overwritten, prev_run = detect_output_field_overwrite(
            sample_adata, analysis_type="da", field_names=field_names, overwrite=True
        )

        assert not will_overwrite
        assert len(overwritten) == 0

    def test_detect_overwrite_var_location(self, sample_adata):
        """Test detection for var location (DE analysis)."""
        # Add existing field in var
        sample_adata.var["kompot_de_A_to_B_mahalanobis"] = 1.0

        field_names = generate_output_field_names(
            result_key="kompot_de", condition1="A", condition2="B", analysis_type="de"
        )

        will_overwrite, overwritten, prev_run = detect_output_field_overwrite(
            sample_adata,
            analysis_type="de",
            field_names=field_names,
            overwrite=False,
            location="var",
        )

        assert will_overwrite
        assert "var.kompot_de_A_to_B_mahalanobis" in overwritten

    def test_detect_overwrite_with_output_patterns(self, sample_adata):
        """Test detection using output_patterns instead of field_names."""
        sample_adata.obs["field1"] = 1.0

        will_overwrite, overwritten, prev_run = detect_output_field_overwrite(
            sample_adata,
            analysis_type="da",
            output_patterns=["field1", "field2"],
            overwrite=False,
            location="obs",
        )

        assert will_overwrite
        assert "obs.field1" in overwritten
        assert "obs.field2" not in overwritten

    def test_detect_overwrite_result_type_parameter(self, sample_adata):
        """Test using result_type instead of analysis_type."""
        sample_adata.obs["field1"] = 1.0

        will_overwrite, overwritten, prev_run = detect_output_field_overwrite(
            sample_adata,
            result_type="differential_abundance",
            output_patterns=["field1"],
            overwrite=False,
            location="obs",
        )

        assert will_overwrite

    def test_detect_overwrite_layers_location(self, sample_adata):
        """Test detection for layers location."""
        sample_adata.layers["smoothed_A"] = np.random.randn(50, 30)

        will_overwrite, overwritten, prev_run = detect_output_field_overwrite(
            sample_adata,
            analysis_type="de",
            output_patterns=["smoothed_A", "smoothed_B"],
            overwrite=False,
            location="layers",
        )

        assert will_overwrite
        assert "layers.smoothed_A" in overwritten

    def test_detect_overwrite_no_field_names_or_patterns(self, sample_adata):
        """Test that missing both field_names and output_patterns raises error."""
        with pytest.raises(ValueError, match="Either field_names or output_patterns"):
            detect_output_field_overwrite(
                sample_adata, analysis_type="da", overwrite=False
            )

    def test_detect_overwrite_no_analysis_type_or_result_type(self, sample_adata):
        """Test that missing both analysis_type and result_type raises error."""
        with pytest.raises(ValueError, match="Either analysis_type or result_type"):
            detect_output_field_overwrite(
                sample_adata, output_patterns=["field1"], overwrite=False
            )


class TestSanitizeName:
    """Test _sanitize_name function."""

    def test_sanitize_basic_string(self):
        """Test sanitizing basic string."""
        assert _sanitize_name("control") == "control"

    def test_sanitize_with_spaces(self):
        """Test sanitizing string with spaces."""
        assert _sanitize_name("control group") == "control_group"

    def test_sanitize_with_hyphens(self):
        """Test sanitizing string with hyphens."""
        assert _sanitize_name("control-group") == "control_group"

    def test_sanitize_with_dots(self):
        """Test sanitizing string with dots."""
        assert _sanitize_name("control.group") == "control_group"

    def test_sanitize_with_slashes(self):
        """Test sanitizing string with slashes."""
        assert _sanitize_name("control/group") == "control_group"

    def test_sanitize_multiple_special_chars(self):
        """Test sanitizing string with multiple special characters."""
        assert _sanitize_name("control-group.1/test") == "control_group_1_test"

    def test_sanitize_none(self):
        """Test sanitizing None."""
        assert _sanitize_name(None) == "None"

    def test_sanitize_number(self):
        """Test sanitizing number."""
        assert _sanitize_name(123) == "123"


class TestValidateFieldRunId:
    """Test validate_field_run_id function."""

    def test_validate_field_run_id_no_tracking(self, sample_adata):
        """Test validation when no tracking info exists."""
        is_valid, actual_run_id, warning = validate_field_run_id(
            sample_adata,
            field_name="test_field",
            location="obs",
            requested_run_id=0,
            storage_key="kompot_da",
        )

        # Should be valid (True) when no tracking exists
        assert is_valid
        assert actual_run_id is None
        assert warning is None

    def test_validate_field_run_id_matching(self, sample_adata):
        """Test validation with matching run ID."""
        # Set up tracking info
        sample_adata.uns["kompot_da"] = {"anndata_fields": {"obs": {"test_field": 0}}}

        is_valid, actual_run_id, warning = validate_field_run_id(
            sample_adata,
            field_name="test_field",
            location="obs",
            requested_run_id=0,
            storage_key="kompot_da",
        )

        assert is_valid
        assert actual_run_id == 0
        assert warning is None

    def test_validate_field_run_id_mismatch(self, sample_adata):
        """Test validation with mismatching run ID."""
        sample_adata.uns["kompot_da"] = {"anndata_fields": {"obs": {"test_field": 1}}}

        is_valid, actual_run_id, warning = validate_field_run_id(
            sample_adata,
            field_name="test_field",
            location="obs",
            requested_run_id=0,
            storage_key="kompot_da",
        )

        assert not is_valid
        assert actual_run_id == 1
        assert warning is not None
        assert "run_id=1" in warning
        assert "run_id=0" in warning

    def test_validate_field_run_id_field_not_tracked(self, sample_adata):
        """Test validation for field that isn't being tracked."""
        sample_adata.uns["kompot_da"] = {"anndata_fields": {"obs": {"other_field": 0}}}

        is_valid, actual_run_id, warning = validate_field_run_id(
            sample_adata,
            field_name="test_field",
            location="obs",
            requested_run_id=0,
            storage_key="kompot_da",
        )

        # Should be valid when field not tracked
        assert is_valid
        assert actual_run_id is None
        assert warning is None


class TestGetRunFromHistory:
    """Test get_run_from_history function."""

    def test_get_run_from_history_empty(self, sample_adata):
        """Test getting run from empty history."""
        run_info = get_run_from_history(sample_adata, run_id=0, analysis_type="da")
        assert run_info is None

    def test_get_run_from_history_none_run_id(self, sample_adata):
        """Test with None as run_id."""
        run_info = get_run_from_history(sample_adata, run_id=None, analysis_type="da")
        assert run_info is None

    def test_get_run_from_history_positive_index(self, sample_adata):
        """Test getting run with positive index."""
        # Add runs to history
        for i in range(3):
            append_to_run_history(sample_adata, {"run_id": i, "data": f"run_{i}"}, "da")

        run_info = get_run_from_history(sample_adata, run_id=1, analysis_type="da")
        assert run_info is not None
        assert run_info["run_id"] == 1
        assert run_info["adjusted_run_id"] == 1

    def test_get_run_from_history_negative_index(self, sample_adata):
        """Test getting run with negative index (most recent)."""
        for i in range(3):
            append_to_run_history(sample_adata, {"run_id": i, "data": f"run_{i}"}, "da")

        # Get most recent (-1)
        run_info = get_run_from_history(sample_adata, run_id=-1, analysis_type="da")
        assert run_info is not None
        assert run_info["run_id"] == 2
        assert run_info["adjusted_run_id"] == 2

        # Get second most recent (-2)
        run_info = get_run_from_history(sample_adata, run_id=-2, analysis_type="da")
        assert run_info["run_id"] == 1

    def test_get_run_from_history_out_of_range(self, sample_adata):
        """Test getting run with out of range index."""
        append_to_run_history(sample_adata, {"run_id": 0}, "da")

        # Index too high
        run_info = get_run_from_history(sample_adata, run_id=5, analysis_type="da")
        assert run_info is None

        # Negative index too low
        run_info = get_run_from_history(sample_adata, run_id=-10, analysis_type="da")
        assert run_info is None

    def test_get_run_from_history_de_type(self, sample_adata):
        """Test getting DE run from history."""
        append_to_run_history(sample_adata, {"run_id": 0, "analysis": "de"}, "de")

        run_info = get_run_from_history(sample_adata, run_id=0, analysis_type="de")
        assert run_info is not None
        assert run_info["analysis"] == "de"

    def test_get_run_from_history_string_item(self, sample_adata):
        """Test handling when run history item is a JSON string."""
        run_data = {"run_id": 0, "condition1": "A"}
        json_str = to_json_string(run_data)

        # Store as JSON string
        sample_adata.uns["kompot_da"] = {"run_history": [json_str]}

        run_info = get_run_from_history(sample_adata, run_id=0, analysis_type="da")
        assert run_info is not None
        assert run_info["run_id"] == 0

    def test_get_run_from_history_invalid_json(self, sample_adata):
        """Test handling when run history item is invalid JSON."""
        sample_adata.uns["kompot_da"] = {"run_history": ["not_valid_json"]}

        run_info = get_run_from_history(sample_adata, run_id=0, analysis_type="da")
        # Should return empty dict instead of None
        assert run_info is not None
        assert isinstance(run_info, dict)
        assert "adjusted_run_id" in run_info

    def test_get_run_from_history_non_dict_item(self, sample_adata):
        """Test handling when run history item is not a dict."""
        sample_adata.uns["kompot_da"] = {
            "run_history": [123]  # Integer instead of dict
        }

        run_info = get_run_from_history(sample_adata, run_id=0, analysis_type="da")
        # Should return empty dict with adjusted_run_id
        assert run_info is not None
        assert isinstance(run_info, dict)
        assert run_info["adjusted_run_id"] == 0

    def test_get_run_from_history_with_history_key(self, sample_adata):
        """Test using history_key parameter for direct access."""
        # Create custom history location
        sample_adata.uns["custom_storage"] = {
            "run_history": [{"run_id": 0, "custom": True}]
        }

        run_info = get_run_from_history(
            sample_adata, run_id=0, history_key="custom_storage.run_history"
        )

        assert run_info is not None
        assert run_info["custom"] is True

    def test_get_run_from_history_preserves_original(self, sample_adata):
        """Test that getting run info doesn't modify original data."""
        original_run = {"run_id": 0, "condition1": "A"}
        append_to_run_history(sample_adata, original_run, "da")

        run_info = get_run_from_history(sample_adata, run_id=0, analysis_type="da")

        # Modify returned run_info
        run_info["new_field"] = "modified"

        # Get again and verify original is unchanged
        run_info2 = get_run_from_history(sample_adata, run_id=0, analysis_type="da")
        assert "new_field" not in run_info2
