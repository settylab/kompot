"""Tests for kompot.anndata.utils.params."""

import pytest
import numpy as np

from kompot.settings import (
    GPSettings,
    FDRSettings,
    DAThresholdSettings,
    FilterSettings,
    StorageSettings,
    OutputSettings,
)
from kompot.anndata.utils.params import (
    build_params_dict,
    _settings_to_dict,
    reconstruct_settings,
    params_to_call_args,
    params_get,
)


# ---------------------------------------------------------------------------
# _settings_to_dict
# ---------------------------------------------------------------------------


class TestSettingsToDict:
    def test_basic_dataclass(self):
        fdr = FDRSettings(null_genes=100, null_seed=7, threshold=0.1)
        d = _settings_to_dict(fdr)
        assert d == {
            "null_genes": 100,
            "null_seed": 7,
            "threshold": 0.1,
            "null_mahalanobis": None,
            "null_expression": None,
            "combine_with_internal": False,
            "mode": "raw",
            "null_trend_features": ("log_mean", "log_var"),
            "null_trend_model": "poly3",
        }

    def test_numpy_array_replaced(self):
        landmarks = np.zeros((50, 3))
        gp = GPSettings(landmarks=landmarks)
        d = _settings_to_dict(gp)
        assert d["landmarks"] == "<array shape=(50, 3)>"
        # Other fields should be preserved
        assert d["sigma"] == 1.0

    def test_numpy_1d_array_rejected(self):
        with pytest.raises(ValueError, match="2-D array"):
            GPSettings(landmarks=np.arange(10))

    def test_none_values_preserved(self):
        gp = GPSettings(ls=None, landmarks=None)
        d = _settings_to_dict(gp)
        assert d["ls"] is None
        assert d["landmarks"] is None


# ---------------------------------------------------------------------------
# build_params_dict
# ---------------------------------------------------------------------------


class TestBuildParamsDict:
    def test_top_level_only(self):
        top = {"groupby": "condition", "condition1": "A", "condition2": "B"}
        result = build_params_dict(top)
        assert result == top

    def test_with_settings(self):
        top = {"groupby": "cond"}
        gp = GPSettings(sigma=2.0)
        fdr = FDRSettings(threshold=0.01)
        result = build_params_dict(top, gp=gp, fdr=fdr)
        assert result["groupby"] == "cond"
        assert result["gp"]["sigma"] == 2.0
        assert result["fdr"]["threshold"] == 0.01

    def test_landmarks_provided_flag(self):
        top = {"groupby": "cond"}
        gp = GPSettings()
        result = build_params_dict(top, landmarks_provided=True, gp=gp)
        assert result["gp"]["landmarks"] is True

    def test_landmarks_flag_without_gp_key(self):
        top = {"groupby": "cond"}
        result = build_params_dict(top, landmarks_provided=True)
        # No gp key, so landmarks flag should be ignored silently
        assert "gp" not in result

    def test_extra_kwargs(self):
        top = {"groupby": "cond"}
        extra = {"n_landmarks": 500, "ls": 0.1}
        result = build_params_dict(top, extra_kwargs=extra)
        assert result["extra_kwargs"] == extra

    def test_extra_kwargs_none(self):
        top = {"groupby": "cond"}
        result = build_params_dict(top, extra_kwargs=None)
        assert "extra_kwargs" not in result

    def test_multiple_settings(self):
        top = {"obsm_key": "X_pca"}
        result = build_params_dict(
            top,
            gp=GPSettings(),
            fdr=FDRSettings(),
            threshold=DAThresholdSettings(lfc_threshold=2.0),
            filter=FilterSettings(min_cells=5),
            storage=StorageSettings(result_key="my_key"),
            output=OutputSettings(progress=False),
        )
        assert result["gp"]["sigma"] == 1.0
        assert result["fdr"]["null_genes"] == "auto"
        assert result["threshold"]["lfc_threshold"] == 2.0
        assert result["filter"]["min_cells"] == 5
        assert result["storage"]["result_key"] == "my_key"
        assert result["output"]["progress"] is False

    def test_numpy_array_in_settings(self):
        top = {"groupby": "cond"}
        gp = GPSettings(landmarks=np.ones((10, 2)))
        result = build_params_dict(top, gp=gp)
        assert result["gp"]["landmarks"] == "<array shape=(10, 2)>"


# ---------------------------------------------------------------------------
# reconstruct_settings
# ---------------------------------------------------------------------------


class TestReconstructSettings:
    def test_roundtrip_gp(self):
        original = GPSettings(sigma=0.5, ls_factor=5.0, n_landmarks=1000)
        params = {"gp": _settings_to_dict(original)}
        result = reconstruct_settings(params)
        assert "gp" in result
        reconstructed = result["gp"]
        assert isinstance(reconstructed, GPSettings)
        assert reconstructed.sigma == 0.5
        assert reconstructed.ls_factor == 5.0
        assert reconstructed.n_landmarks == 1000

    def test_roundtrip_fdr(self):
        original = FDRSettings(null_genes=500, threshold=0.01)
        params = {"fdr": _settings_to_dict(original)}
        result = reconstruct_settings(params)
        assert isinstance(result["fdr"], FDRSettings)
        assert result["fdr"].null_genes == 500
        assert result["fdr"].threshold == 0.01

    def test_roundtrip_all_settings(self):
        params = {
            "gp": _settings_to_dict(GPSettings()),
            "fdr": _settings_to_dict(FDRSettings()),
            "threshold": _settings_to_dict(DAThresholdSettings()),
            "filter": _settings_to_dict(FilterSettings()),
            "storage": _settings_to_dict(StorageSettings()),
            "output": _settings_to_dict(OutputSettings()),
        }
        result = reconstruct_settings(params)
        assert len(result) == 6
        assert isinstance(result["gp"], GPSettings)
        assert isinstance(result["fdr"], FDRSettings)
        assert isinstance(result["threshold"], DAThresholdSettings)
        assert isinstance(result["filter"], FilterSettings)
        assert isinstance(result["storage"], StorageSettings)
        assert isinstance(result["output"], OutputSettings)

    def test_missing_keys_ignored(self):
        params = {"fdr": _settings_to_dict(FDRSettings())}
        result = reconstruct_settings(params)
        assert "gp" not in result
        assert "fdr" in result

    def test_extra_keys_in_sub_dict_filtered(self):
        params = {
            "fdr": {
                "null_genes": 100,
                "null_seed": 42,
                "threshold": 0.05,
                "unknown_future_field": "should be dropped",
            }
        }
        result = reconstruct_settings(params)
        fdr = result["fdr"]
        assert isinstance(fdr, FDRSettings)
        assert fdr.null_genes == 100
        assert not hasattr(fdr, "unknown_future_field")

    def test_non_dict_values_skipped(self):
        """Top-level keys that are not dicts (e.g. groupby) should be skipped."""
        params = {"groupby": "condition", "fdr": _settings_to_dict(FDRSettings())}
        result = reconstruct_settings(params)
        assert "groupby" not in result
        assert "fdr" in result

    def test_array_placeholder_replaced_with_none(self):
        gp = GPSettings(landmarks=np.ones((10, 3)))
        params = {"gp": _settings_to_dict(gp)}
        # The serialized dict has a placeholder string
        assert params["gp"]["landmarks"].startswith("<array")
        result = reconstruct_settings(params)
        assert result["gp"].landmarks is None

    def test_empty_params(self):
        result = reconstruct_settings({})
        assert result == {}


# ---------------------------------------------------------------------------
# params_to_call_args
# ---------------------------------------------------------------------------


class TestParamsToCallArgs:
    def _make_nested_params(self, **overrides):
        params = {
            "groupby": "condition",
            "condition1": "Young",
            "condition2": "Old",
            "obsm_key": "X_pca",
            "gp": _settings_to_dict(GPSettings(sigma=0.5)),
            "fdr": _settings_to_dict(FDRSettings(threshold=0.01)),
        }
        params.update(overrides)
        return params

    def test_da_basic(self):
        params = self._make_nested_params(
            threshold=_settings_to_dict(DAThresholdSettings(lfc_threshold=2.0)),
        )
        kwargs = params_to_call_args(params, "da")
        assert kwargs["groupby"] == "condition"
        assert kwargs["condition1"] == "Young"
        assert kwargs["condition2"] == "Old"
        assert kwargs["obsm_key"] == "X_pca"
        assert isinstance(kwargs["gp"], GPSettings)
        assert kwargs["gp"].sigma == 0.5
        assert isinstance(kwargs["threshold"], DAThresholdSettings)
        assert kwargs["threshold"].lfc_threshold == 2.0

    def test_de_basic(self):
        params = self._make_nested_params(layer="raw_counts")
        params["genes"] = ["GeneA", "GeneB"]
        kwargs = params_to_call_args(params, "de")
        assert kwargs["groupby"] == "condition"
        assert kwargs["genes"] == ["GeneA", "GeneB"]
        assert isinstance(kwargs["fdr"], FDRSettings)
        assert kwargs["fdr"].threshold == 0.01

    def test_sample_col_forwarded(self):
        params = self._make_nested_params(sample_col="donor")
        kwargs = params_to_call_args(params, "de")
        assert kwargs["sample_col"] == "donor"

    def test_extra_kwargs_merged(self):
        params = self._make_nested_params(
            extra_kwargs={"density_kwargs": {"n_landmarks": 200}}
        )
        kwargs = params_to_call_args(params, "da")
        assert kwargs["density_kwargs"] == {"n_landmarks": 200}

    def test_missing_optional_keys(self):
        params = {
            "groupby": "cond",
            "condition1": "A",
            "condition2": "B",
            "gp": _settings_to_dict(GPSettings()),
        }
        kwargs = params_to_call_args(params, "de")
        assert "layer" not in kwargs
        assert "genes" not in kwargs
        assert "sample_col" not in kwargs

    def test_layer_forwarded(self):
        params = self._make_nested_params(layer="counts")
        kwargs = params_to_call_args(params, "de")
        assert kwargs["layer"] == "counts"


# ---------------------------------------------------------------------------
# params_get
# ---------------------------------------------------------------------------


class TestParamsGet:
    # -- Nested format --

    def test_nested_top_level(self):
        params = {"groupby": "condition", "gp": {"sigma": 0.5}}
        assert params_get(params, "groupby") == "condition"

    def test_nested_settings_field(self):
        params = {"gp": {"sigma": 0.5, "ls_factor": 5.0}}
        assert params_get(params, "sigma") == 0.5
        assert params_get(params, "ls_factor") == 5.0

    def test_nested_fdr_field(self):
        params = {"fdr": {"threshold": 0.01, "null_genes": 100}}
        assert params_get(params, "fdr_threshold") == 0.01
        assert params_get(params, "null_genes") == 100

    def test_nested_threshold_field(self):
        params = {"threshold": {"lfc_threshold": 2.0, "ptp_threshold": 0.1}}
        assert params_get(params, "log_fold_change_threshold") == 2.0
        assert params_get(params, "ptp_threshold") == 0.1

    def test_nested_filter_field(self):
        params = {"filter": {"min_cells": 10}}
        assert params_get(params, "min_cells") == 10

    def test_nested_storage_field(self):
        params = {"storage": {"result_key": "my_key", "overwrite": True}}
        assert params_get(params, "result_key") == "my_key"
        assert params_get(params, "overwrite") is True

    def test_nested_output_field(self):
        params = {"output": {"progress": False, "copy": True}}
        assert params_get(params, "progress") is False
        assert params_get(params, "copy") is True

    def test_nested_missing_key_returns_default(self):
        params = {"gp": {"sigma": 0.5}}
        assert params_get(params, "nonexistent") is None
        assert params_get(params, "nonexistent", "fallback") == "fallback"

    def test_nested_missing_field_in_group(self):
        params = {"gp": {"sigma": 0.5}}
        # ls is a valid legacy key mapping to gp.ls, but not in the dict
        assert params_get(params, "ls") is None
        assert params_get(params, "ls", 42.0) == 42.0

    # -- Legacy flat format --

    def test_legacy_flat_direct(self):
        params = {"sigma": 0.5, "ls_factor": 10.0, "fdr_threshold": 0.05}
        assert params_get(params, "sigma") == 0.5
        assert params_get(params, "ls_factor") == 10.0
        assert params_get(params, "fdr_threshold") == 0.05

    def test_legacy_flat_top_level(self):
        params = {"groupby": "condition", "sigma": 1.0}
        assert params_get(params, "groupby") == "condition"

    # -- Derived keys --

    def test_use_sample_variance_true(self):
        params = {"sample_col": "donor"}
        assert params_get(params, "use_sample_variance") is True

    def test_use_sample_variance_false_when_none(self):
        params = {"sample_col": None}
        assert params_get(params, "use_sample_variance") is False

    def test_use_sample_variance_false_when_missing(self):
        params = {}
        assert params_get(params, "use_sample_variance") is False

    # -- Edge cases --

    def test_default_parameter(self):
        params = {}
        assert params_get(params, "sigma", 999) == 999

    def test_top_level_takes_precedence_over_legacy(self):
        """When a key is present directly and is not a dict, return it."""
        params = {"sigma": 0.3}
        assert params_get(params, "sigma") == 0.3
