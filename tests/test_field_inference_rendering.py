"""Tests for kompot/plot/field_inference.py targeting uncovered lines.

Covers infer_fields_from_run_info, _check_for_overwrites,
_count_potential_field_writers, _fallback_field_inference,
get_comparison_specific_fields, and overwrite detection paths.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from anndata import AnnData
import json


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


def _make_da_adata(
    n_obs=60,
    n_vars=5,
    n_groups=3,
    with_run_history=True,
    condition1="CondA",
    condition2="CondB",
    result_key="kompot_da",
    extra_obs_cols=None,
):
    """Build a minimal AnnData with DA results and run history."""
    rng = np.random.RandomState(42)
    X = rng.randn(n_obs, n_vars).astype(np.float32)
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_obs)])

    lfc_col = f"{result_key}_{condition1}_to_{condition2}_log_fold_change"
    ptp_col = f"{result_key}_{condition1}_to_{condition2}_ptp"
    direction_col = f"{result_key}_{condition1}_to_{condition2}_direction"

    obs[lfc_col] = rng.randn(n_obs).astype(np.float32)
    obs[ptp_col] = rng.uniform(0.001, 1.0, n_obs).astype(np.float32)
    obs[direction_col] = pd.Categorical(rng.choice(["up", "down", "neutral"], n_obs))

    groups = [f"group_{i}" for i in range(n_groups)]
    obs["cell_type"] = pd.Categorical(rng.choice(groups, n_obs))

    if extra_obs_cols:
        for col_name, col_vals in extra_obs_cols.items():
            obs[col_name] = col_vals

    adata = AnnData(X=X, obs=obs)
    adata.obsm["X_umap"] = rng.randn(n_obs, 2).astype(np.float32)
    adata.var_names = [f"gene_{i}" for i in range(n_vars)]

    if with_run_history:
        adata.uns[result_key] = {
            "run_history": json.dumps(
                [
                    {
                        "timestamp": "2025-01-01T00:00:00",
                        "params": {
                            "condition1": condition1,
                            "condition2": condition2,
                            "log_fold_change_threshold": 0.5,
                            "ptp_threshold": 0.05,
                        },
                        "field_names": {
                            "lfc_key": lfc_col,
                            "ptp_key": ptp_col,
                            "direction_key": direction_col,
                        },
                        "adjusted_run_id": 0,
                    }
                ]
            ),
        }

    return adata


def _make_de_adata(
    n_obs=30,
    n_vars=10,
    condition1="Young",
    condition2="Old",
    result_key="kompot_de",
    with_layers=True,
    with_run_history=True,
):
    """Build a minimal AnnData with DE results and run history."""
    rng = np.random.RandomState(0)
    X = rng.randn(n_obs, n_vars).astype(np.float32)
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)])

    lfc_col = f"{result_key}_{condition1}_to_{condition2}_mean_lfc"
    mahal_col = f"{result_key}_{condition1}_to_{condition2}_mahalanobis"

    var[lfc_col] = rng.randn(n_vars).astype(np.float32)
    var[mahal_col] = rng.uniform(0, 5, n_vars).astype(np.float32)

    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_obs)])
    adata = AnnData(X=X, obs=obs, var=var)
    adata.obsm["X_umap"] = rng.randn(n_obs, 2).astype(np.float32)

    imputed1 = f"{result_key}_{condition1}_imputed"
    imputed2 = f"{result_key}_{condition2}_imputed"
    fc_layer = f"{result_key}_{condition1}_to_{condition2}_fold_change"

    if with_layers:
        adata.layers[imputed1] = rng.randn(n_obs, n_vars).astype(np.float32)
        adata.layers[imputed2] = rng.randn(n_obs, n_vars).astype(np.float32)
        adata.layers[fc_layer] = rng.randn(n_obs, n_vars).astype(np.float32)

    if with_run_history:
        adata.uns[result_key] = {
            "run_history": json.dumps(
                [
                    {
                        "timestamp": "2025-01-01T00:00:00",
                        "params": {
                            "condition1": condition1,
                            "condition2": condition2,
                        },
                        "field_names": {
                            "mean_lfc_key": lfc_col,
                            "mahalanobis_key": mahal_col,
                            "imputed_key_1": imputed1,
                            "imputed_key_2": imputed2,
                            "fold_change_key": fc_layer,
                        },
                        "imputed_layer_keys": {
                            "condition1": imputed1,
                            "condition2": imputed2,
                            "fold_change": fc_layer,
                        },
                        "adjusted_run_id": 0,
                    }
                ]
            ),
        }

    return adata


class TestFieldInference:
    """Tests for field_inference.py targeting uncovered lines."""

    def test_infer_default_da_fields(self):
        """Lines 65, 69: default required_fields for DA."""
        from kompot.plot.field_inference import infer_fields_from_run_info

        adata = _make_da_adata()
        result = infer_fields_from_run_info(
            adata, analysis_type="da", run_id=-1, strict=False
        )
        assert "lfc_key" in result
        assert "direction_key" in result

    def test_infer_unknown_analysis_type(self):
        """Line 69: unknown analysis type -> empty required_fields."""
        from kompot.plot.field_inference import infer_fields_from_run_info

        adata = _make_da_adata()
        result = infer_fields_from_run_info(
            adata, analysis_type="unknown", strict=False
        )
        assert isinstance(result, dict)

    def test_infer_no_run_info(self):
        """Lines 77-79: no run info available."""
        from kompot.plot.field_inference import infer_fields_from_run_info

        adata = _make_de_adata(with_run_history=False)
        result = infer_fields_from_run_info(adata, analysis_type="de", strict=False)
        assert isinstance(result, dict)

    def test_condition_mismatch_warning(self):
        """Lines 257-258, 262-266: user conditions don't match run info."""
        from kompot.plot.field_inference import infer_fields_from_run_info

        adata = _make_da_adata(condition1="CondA", condition2="CondB")
        result = infer_fields_from_run_info(
            adata,
            analysis_type="da",
            condition1="WrongA",
            condition2="WrongB",
            strict=False,
        )
        assert isinstance(result, dict)

    def test_condition_mismatch_strict_raises(self):
        """Lines 275, 283: strict mode raises on condition mismatch."""
        from kompot.plot.field_inference import infer_fields_from_run_info

        adata = _make_da_adata(condition1="CondA", condition2="CondB")
        with pytest.raises(ValueError, match="Condition mismatch"):
            infer_fields_from_run_info(
                adata,
                analysis_type="da",
                condition1="WrongA",
                condition2="WrongB",
                strict=True,
            )

    def test_field_not_in_data(self):
        """Lines 313-315, 321: field in run_info but missing from data."""
        from kompot.plot.field_inference import infer_fields_from_run_info

        adata = _make_da_adata()
        # Rename the actual column so the run_info field name doesn't match
        lfc_col = [c for c in adata.obs.columns if "log_fold_change" in c][0]
        adata.obs.rename(columns={lfc_col: "renamed_lfc"}, inplace=True)
        result = infer_fields_from_run_info(adata, analysis_type="da", strict=False)
        # lfc_key should be None or fallback
        assert isinstance(result, dict)

    def test_strict_missing_required_fields(self):
        """Lines 325-328: strict mode with missing required fields."""
        from kompot.plot.field_inference import infer_fields_from_run_info

        adata = _make_de_adata(with_run_history=False)
        # Remove all recognizable columns
        adata.var = adata.var.drop(columns=adata.var.columns.tolist())
        with pytest.raises(ValueError, match="Could not infer"):
            infer_fields_from_run_info(
                adata,
                analysis_type="de",
                strict=True,
            )

    def test_fallback_multiple_candidates(self):
        """Lines 344-346, 385: multiple candidates in fallback."""
        from kompot.plot.field_inference import infer_fields_from_run_info

        adata = _make_de_adata(with_run_history=False)
        # Add multiple columns that match the pattern
        adata.var["kompot_de_A_to_B_mean_lfc"] = np.random.randn(adata.n_vars)
        adata.var["kompot_de_C_to_D_mean_lfc"] = np.random.randn(adata.n_vars)
        adata.var["kompot_de_A_to_B_mahalanobis"] = np.random.randn(adata.n_vars)
        adata.var["kompot_de_C_to_D_mahalanobis"] = np.random.randn(adata.n_vars)
        result = infer_fields_from_run_info(
            adata,
            analysis_type="de",
            strict=False,
        )
        assert isinstance(result, dict)

    def test_check_for_overwrites(self):
        """Lines 313-315: _check_for_overwrites with tracking."""
        from kompot.plot.field_inference import _check_for_overwrites

        adata = _make_da_adata()
        lfc_col = [c for c in adata.obs.columns if "log_fold_change" in c][0]
        # Set up field tracking
        adata.uns["kompot_da"]["anndata_fields"] = json.dumps({"obs": {lfc_col: 0}})
        warnings_list = []
        fields = {"lfc_key": lfc_col}
        _check_for_overwrites(adata, "da", fields, warnings_list)
        # Should not crash; may or may not add warnings

    def test_count_potential_field_writers(self):
        """Lines 369-394: _count_potential_field_writers."""
        from kompot.plot.field_inference import _count_potential_field_writers

        adata = _make_da_adata()
        lfc_col = [c for c in adata.obs.columns if "log_fold_change" in c][0]
        count = _count_potential_field_writers(adata, "da", "lfc_key", lfc_col)
        assert count >= 1


class TestFieldInferenceOverwrites:
    """More targeted tests for field_inference.py overwrite detection."""

    def test_check_overwrites_with_tracking_data(self):
        """Lines 310-315, 321, 325-328, 344-346, 350-354, 363-366: full overwrite path."""
        from kompot.plot.field_inference import _check_for_overwrites

        adata = _make_da_adata(condition1="CondA", condition2="CondB")
        lfc_col = [c for c in adata.obs.columns if "log_fold_change" in c][0]
        direction_col = [c for c in adata.obs.columns if "direction" in c][0]

        # Set up field tracking with a DIFFERENT run_id than the latest
        adata.uns["kompot_da"]["anndata_fields"] = json.dumps(
            {
                "obs": {lfc_col: 99}  # run_id=99, but latest is 0
            }
        )

        warnings_list = []
        fields = {"lfc_key": lfc_col}
        _check_for_overwrites(adata, "da", fields, warnings_list)
        # Should detect the mismatch
        assert any(
            "overwritten" in w.lower() or "written by" in w.lower()
            for w in warnings_list
        )

    def test_check_overwrites_location_not_in_tracking(self):
        """Line 321: location not in tracking -> early return."""
        from kompot.plot.field_inference import _check_for_overwrites

        adata = _make_da_adata()
        lfc_col = [c for c in adata.obs.columns if "log_fold_change" in c][0]
        # Only has "var" key, but DA needs "obs"
        adata.uns["kompot_da"]["anndata_fields"] = json.dumps({"var": {lfc_col: 0}})
        warnings_list = []
        _check_for_overwrites(adata, "da", {"lfc_key": lfc_col}, warnings_list)
        # Should return early without adding warnings
        assert len(warnings_list) == 0

    def test_check_overwrites_location_tracking_as_string(self):
        """Lines 325-328: location_tracking is a JSON string."""
        from kompot.plot.field_inference import _check_for_overwrites

        adata = _make_da_adata()
        lfc_col = [c for c in adata.obs.columns if "log_fold_change" in c][0]
        adata.uns["kompot_da"]["anndata_fields"] = json.dumps(
            {
                "obs": json.dumps({lfc_col: 0})  # Nested JSON string
            }
        )
        warnings_list = []
        _check_for_overwrites(adata, "da", {"lfc_key": lfc_col}, warnings_list)

    def test_check_overwrites_invalid_tracking(self):
        """Lines 313-315: exception accessing tracking data."""
        from kompot.plot.field_inference import _check_for_overwrites

        adata = _make_da_adata()
        lfc_col = [c for c in adata.obs.columns if "log_fold_change" in c][0]
        # Set a value that will cause from_json_string to raise
        adata.uns["kompot_da"]["anndata_fields"] = "{invalid json"
        warnings_list = []
        _check_for_overwrites(adata, "da", {"lfc_key": lfc_col}, warnings_list)

    def test_check_overwrites_multiple_writers(self):
        """Lines 362-366: multiple potential writers."""
        from kompot.plot.field_inference import _check_for_overwrites

        adata = _make_da_adata()
        lfc_col = [c for c in adata.obs.columns if "log_fold_change" in c][0]
        # Add a second run to history with same field_names
        run_history = json.loads(adata.uns["kompot_da"]["run_history"])
        run_history.append(
            {
                "timestamp": "2025-02-01T00:00:00",
                "params": {"condition1": "CondA", "condition2": "CondB"},
                "field_names": {"lfc_key": lfc_col},
                "adjusted_run_id": 1,
            }
        )
        adata.uns["kompot_da"]["run_history"] = json.dumps(run_history)
        adata.uns["kompot_da"]["anndata_fields"] = json.dumps({"obs": {lfc_col: 1}})
        warnings_list = []
        _check_for_overwrites(adata, "da", {"lfc_key": lfc_col}, warnings_list)
        # Should detect multiple writers
        assert any("written by" in w.lower() for w in warnings_list)

    def test_get_run_from_history_error(self):
        """Lines 77-79: error accessing run history."""
        from kompot.plot.field_inference import infer_fields_from_run_info

        adata = _make_da_adata()
        # Corrupt the run history
        adata.uns["kompot_da"]["run_history"] = "not valid json {{{"
        result = infer_fields_from_run_info(adata, analysis_type="da", strict=False)
        assert isinstance(result, dict)

    def test_de_field_mapping(self):
        """Lines 122-123: DE field mapping path."""
        from kompot.plot.field_inference import infer_fields_from_run_info

        adata = _make_de_adata()
        result = infer_fields_from_run_info(adata, analysis_type="de", strict=False)
        assert "mean_lfc_key" in result
        assert "mahalanobis_key" in result

    def test_fallback_with_condition_filtering(self):
        """Lines 250-258: fallback with condition-based filtering."""
        from kompot.plot.field_inference import _fallback_field_inference

        # Create mock data_section with columns
        data = pd.DataFrame(
            {
                "kompot_da_A_to_B_lfc": [1.0],
                "kompot_da_C_to_D_lfc": [2.0],
            }
        )
        result = _fallback_field_inference(
            data,
            "lfc_key",
            "da",
            condition1="A",
            condition2="B",
            result_key=None,
            strict=False,
        )
        assert result == "kompot_da_A_to_B_lfc"

    def test_fallback_with_result_key_filtering(self):
        """Lines 262-266: fallback with result_key filtering."""
        from kompot.plot.field_inference import _fallback_field_inference

        data = pd.DataFrame(
            {
                "kompot_da_A_to_B_lfc": [1.0],
                "kompot_da_C_to_B_lfc": [2.0],
            }
        )
        result = _fallback_field_inference(
            data,
            "lfc_key",
            "da",
            condition1=None,
            condition2=None,
            result_key="kompot_da_A",
            strict=False,
        )
        assert result is not None

    def test_fallback_unknown_field_type(self):
        """Line 230: unknown field_type returns None."""
        from kompot.plot.field_inference import _fallback_field_inference

        data = pd.DataFrame({"col": [1.0]})
        result = _fallback_field_inference(
            data, "unknown_field", "da", None, None, None, False
        )
        assert result is None

    def test_get_comparison_specific_fields(self):
        """Lines 437-462: get_comparison_specific_fields."""
        from kompot.plot.field_inference import get_comparison_specific_fields

        adata = _make_da_adata(condition1="CondA", condition2="CondB")
        fields = get_comparison_specific_fields(adata, "da", "CondA", "CondB")
        assert isinstance(fields, dict)

    def test_fallback_strict_multiple_candidates(self):
        """Line 280: strict mode returns None for multiple candidates."""
        from kompot.plot.field_inference import _fallback_field_inference

        data = pd.DataFrame(
            {
                "kompot_da_A_to_B_lfc": [1.0],
                "kompot_da_C_to_D_lfc": [2.0],
            }
        )
        result = _fallback_field_inference(
            data,
            "lfc_key",
            "da",
            condition1=None,
            condition2=None,
            result_key=None,
            strict=True,
        )
        # With strict=True and multiple candidates, should return None
        assert result is None
