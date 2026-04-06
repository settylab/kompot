"""Tests for kompot.anndata._da_helpers overwrite checks, data extraction, landmarks, and run info recording.

Covers uncovered lines in kompot/anndata/_da_helpers.py including:
- _check_da_overwrites with various overwrite modes and sample variance changes
- _extract_da_data missing keys and condition validation
- _resolve_da_landmarks stored/provided landmarks with dimension checks
- _record_da_run_info merging into existing anndata_fields
"""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from kompot.anndata.utils.json_utils import (
    get_json_metadata,
    to_json_string,
)


# ---------------------------------------------------------------------------
# Helpers to build mock AnnData with run history
# ---------------------------------------------------------------------------


def _make_run_entry(
    run_id, params=None, field_mapping=None, timestamp="2025-01-01T00:00:00"
):
    """Build a minimal run-history entry."""
    entry = {
        "run_id": run_id,
        "adjusted_run_id": run_id,
        "timestamp": timestamp,
        "params": params
        or {
            "condition1": "A",
            "condition2": "B",
            "obsm_key": "X_pca",
            "result_key": "kompot_da",
        },
        "field_names": {},
        "environment": {},
        "field_mapping": field_mapping or {},
    }
    return entry


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
            "anndata_fields": to_json_string(
                {
                    "obs": {"da_pval": 0, "da_lfc": 0},
                    "uns": {"kompot_da": 0},
                }
            ),
        }
        return adata

    def test_overwrite_false_raises(self):
        # lines 91-93: overwrite=False raises ValueError
        from kompot.anndata._da_helpers import _check_da_overwrites

        adata = self._make_da_adata_with_results()
        field_names = {"all_patterns": {"obs": ["da_pval", "da_lfc"]}}
        with pytest.raises(ValueError, match="overwrite=True"):
            _check_da_overwrites(
                adata,
                result_key="kompot_da",
                field_names=field_names,
                sample_col=None,
                overwrite=False,
                groupby="group",
                condition1="A",
                condition2="B",
                obsm_key="X_pca",
                ls_factor=1.0,
            )

    def test_overwrite_none_with_sample_var_change_params_match(self):
        # lines 99-119: overwrite=None, adding sample variance, params match -> info log
        from kompot.anndata._da_helpers import _check_da_overwrites

        adata = self._make_da_adata_with_results(
            sample_col="sample", prev_sample_var=False
        )
        field_names = {"all_patterns": {"obs": ["da_pval", "da_lfc"]}}
        # Should not raise, just log info
        _check_da_overwrites(
            adata,
            result_key="kompot_da",
            field_names=field_names,
            sample_col="sample",
            overwrite=None,
            groupby="group",
            condition1="A",
            condition2="B",
            obsm_key="X_pca",
            ls_factor=1.0,
        )

    def test_overwrite_none_with_sample_var_change_params_mismatch(self):
        # lines 100-108: overwrite=None, adding sample_var but different groupby -> warning
        from kompot.anndata._da_helpers import _check_da_overwrites

        adata = self._make_da_adata_with_results(
            sample_col="sample", prev_sample_var=False
        )
        field_names = {"all_patterns": {"obs": ["da_pval", "da_lfc"]}}
        # Different groupby => params_match=False
        _check_da_overwrites(
            adata,
            result_key="kompot_da",
            field_names=field_names,
            sample_col="sample",
            overwrite=None,
            groupby="different_col",
            condition1="A",
            condition2="B",
            obsm_key="X_pca",
            ls_factor=1.0,
        )

    def test_overwrite_false_with_sample_var_fields_listed(self):
        # lines 72-82: overwrite=False with prev_sample_var different, fields listed
        from kompot.anndata._da_helpers import _check_da_overwrites

        adata = self._make_da_adata_with_results(
            sample_col="sample", prev_sample_var=False
        )
        field_names = {"all_patterns": {"obs": ["da_pval", "da_lfc"]}}
        with pytest.raises(ValueError, match="overwrite=True"):
            _check_da_overwrites(
                adata,
                result_key="kompot_da",
                field_names=field_names,
                sample_col="sample",
                overwrite=False,
                groupby="group",
                condition1="A",
                condition2="B",
                obsm_key="X_pca",
                ls_factor=1.0,
            )

    def test_overwrite_false_prev_sample_var_true_current_false(self):
        # lines 81-87: previous had sample_var=True, current doesn't
        from kompot.anndata._da_helpers import _check_da_overwrites

        adata = self._make_da_adata_with_results(prev_sample_var=True)
        field_names = {"all_patterns": {"obs": ["da_pval", "da_lfc"]}}
        with pytest.raises(ValueError, match="overwrite=True"):
            _check_da_overwrites(
                adata,
                result_key="kompot_da",
                field_names=field_names,
                sample_col=None,
                overwrite=False,
                groupby="group",
                condition1="A",
                condition2="B",
                obsm_key="X_pca",
                ls_factor=1.0,
            )

    def test_overwrite_none_no_sample_var_change(self):
        # line 126-131: overwrite=None, no sample_var change -> warning
        from kompot.anndata._da_helpers import _check_da_overwrites

        adata = self._make_da_adata_with_results(prev_sample_var=False)
        field_names = {"all_patterns": {"obs": ["da_pval", "da_lfc"]}}
        _check_da_overwrites(
            adata,
            result_key="kompot_da",
            field_names=field_names,
            sample_col=None,
            overwrite=None,
            groupby="group",
            condition1="A",
            condition2="B",
            obsm_key="X_pca",
            ls_factor=1.0,
        )


class TestExtractDaData:
    """Cover lines 144-160, 163, 172, 174, 186."""

    def test_missing_obsm_key(self):
        # lines 144-160: obsm_key not found
        from kompot.anndata._da_helpers import _extract_da_data

        adata = AnnData(
            X=np.zeros((5, 3)),
            obs=pd.DataFrame(
                {"group": ["A"] * 3 + ["B"] * 2}, index=[f"c{i}" for i in range(5)]
            ),
        )
        with pytest.raises(ValueError, match="not found in adata.obsm"):
            _extract_da_data(adata, "group", "A", "B", "X_pca", None)

    def test_missing_obsm_DM_EigenVectors(self):
        # lines 148-159: special DM_EigenVectors message
        from kompot.anndata._da_helpers import _extract_da_data

        adata = AnnData(
            X=np.zeros((5, 3)),
            obs=pd.DataFrame(
                {"group": ["A"] * 3 + ["B"] * 2}, index=[f"c{i}" for i in range(5)]
            ),
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
            obs=pd.DataFrame(
                {"group": ["A"] * 3 + ["B"] * 2}, index=[f"c{i}" for i in range(5)]
            ),
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
            obs=pd.DataFrame(
                {"group": ["A"] * 3 + ["B"] * 2}, index=[f"c{i}" for i in range(5)]
            ),
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
            adata,
            field_names,
            condition1="A",
            condition2="B",
            sample_col=None,
            result_key="kompot_da",
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
            adata,
            field_names,
            condition1="A",
            condition2="B",
            sample_col=None,
            result_key="kompot_da",
            params_dict={"condition1": "A", "condition2": "B"},
        )
        tracking = get_json_metadata(adata, "kompot_da.anndata_fields")
        assert "obs" in tracking
        assert "uns" in tracking
