"""Tests for kompot input validation (settings __post_init__ and AnnData checks)."""

import numpy as np
import pandas as pd
import pytest
import anndata

import kompot
from kompot.settings import (
    GPSettings,
    FDRSettings,
    DAThresholdSettings,
    FilterSettings,
    StorageSettings,
    OutputSettings,
    ModelSettings,
)
from kompot.validation import (
    validate_probability,
    validate_non_negative_int,
    validate_non_negative_float,
    validate_obsm_key,
    validate_groupby,
    validate_conditions,
    validate_layer,
    validate_genes,
    validate_sample_col,
    validate_expression_data,
    validate_landmarks_shape,
)


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------


@pytest.fixture
def adata():
    rng = np.random.RandomState(42)
    n_cells, n_features, n_genes = 80, 5, 8
    ad = anndata.AnnData(X=rng.randn(n_cells, n_genes))
    ad.obsm["DM_EigenVectors"] = rng.randn(n_cells, n_features)
    ad.obs["condition"] = ["A"] * 40 + ["B"] * 40
    ad.obs["sample"] = ["s1"] * 20 + ["s2"] * 20 + ["s3"] * 20 + ["s4"] * 20
    ad.var_names = [f"gene_{i}" for i in range(n_genes)]
    ad.layers["log"] = rng.randn(n_cells, n_genes)
    return ad


# -----------------------------------------------------------------------
# GPSettings validation
# -----------------------------------------------------------------------


class TestGPSettingsValidation:
    def test_valid_defaults(self):
        GPSettings()

    def test_valid_custom(self):
        GPSettings(sigma=0.5, ls=2.0, n_landmarks=100, batch_size=50, eps=1e-6)

    def test_negative_sigma(self):
        with pytest.raises(ValueError, match="sigma"):
            GPSettings(sigma=-1.0)

    def test_zero_sigma(self):
        with pytest.raises(ValueError, match="sigma"):
            GPSettings(sigma=0.0)

    def test_negative_ls(self):
        with pytest.raises(ValueError, match="ls"):
            GPSettings(ls=-5.0)

    def test_negative_ls_factor(self):
        with pytest.raises(ValueError, match="ls_factor"):
            GPSettings(ls_factor=-1.0)

    def test_negative_n_landmarks(self):
        with pytest.raises(ValueError, match="n_landmarks"):
            GPSettings(n_landmarks=-100)

    def test_string_n_landmarks(self):
        with pytest.raises((ValueError, TypeError)):
            GPSettings(n_landmarks="5000")

    def test_negative_eps(self):
        with pytest.raises(ValueError, match="eps"):
            GPSettings(eps=-1e-8)

    def test_string_sigma(self):
        with pytest.raises((ValueError, TypeError)):
            GPSettings(sigma="high")

    def test_bool_not_accepted_as_int(self):
        with pytest.raises((ValueError, TypeError)):
            GPSettings(random_state=True)

    def test_landmarks_must_be_2d(self):
        with pytest.raises(ValueError, match="2-D"):
            GPSettings(landmarks=np.arange(10))

    def test_landmarks_2d_accepted(self):
        GPSettings(landmarks=np.zeros((50, 3)))

    def test_use_empirical_variance_must_be_bool(self):
        with pytest.raises(TypeError):
            GPSettings(use_empirical_variance="yes")

    def test_jit_compile_must_be_bool(self):
        with pytest.raises(TypeError):
            GPSettings(jit_compile=1)


# -----------------------------------------------------------------------
# FDRSettings validation
# -----------------------------------------------------------------------


class TestFDRSettingsValidation:
    def test_valid_defaults(self):
        FDRSettings()

    def test_threshold_above_1(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            FDRSettings(threshold=1.5)

    def test_threshold_negative(self):
        with pytest.raises(ValueError, match="threshold"):
            FDRSettings(threshold=-0.1)

    def test_threshold_zero(self):
        with pytest.raises(ValueError, match="threshold"):
            FDRSettings(threshold=0.0)

    def test_null_genes_bad_string(self):
        with pytest.raises(ValueError, match="auto"):
            FDRSettings(null_genes="invalid")

    def test_null_genes_auto_valid(self):
        FDRSettings(null_genes="auto")

    def test_null_genes_none_valid(self):
        FDRSettings(null_genes=None)

    def test_null_genes_zero_valid(self):
        FDRSettings(null_genes=0)

    def test_null_genes_positive_int_valid(self):
        FDRSettings(null_genes=2000)

    def test_null_genes_list_valid(self):
        FDRSettings(null_genes=[0, 1, 2])

    def test_null_genes_negative_int(self):
        with pytest.raises(ValueError, match="null_genes"):
            FDRSettings(null_genes=-5)

    def test_null_genes_list_with_negative(self):
        with pytest.raises(ValueError, match="null_genes"):
            FDRSettings(null_genes=[0, -1, 2])

    def test_null_genes_float(self):
        with pytest.raises(TypeError, match="null_genes"):
            FDRSettings(null_genes=3.5)

    def test_mutual_exclusivity_null_mahalanobis_and_expression(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            FDRSettings(
                null_mahalanobis=np.array([1.0, 2.0]),
                null_expression=(np.array([[1]]), np.array([[2]])),
            )


# -----------------------------------------------------------------------
# DAThresholdSettings validation
# -----------------------------------------------------------------------


class TestDAThresholdSettingsValidation:
    def test_valid_defaults(self):
        DAThresholdSettings()

    def test_negative_ptp_threshold(self):
        with pytest.raises(ValueError, match="ptp_threshold"):
            DAThresholdSettings(ptp_threshold=-0.1)

    def test_ptp_threshold_above_1(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            DAThresholdSettings(ptp_threshold=2.0)

    def test_negative_lfc_threshold(self):
        with pytest.raises(ValueError, match="lfc_threshold"):
            DAThresholdSettings(lfc_threshold=-1.0)

    def test_zero_lfc_valid(self):
        DAThresholdSettings(lfc_threshold=0.0)


# -----------------------------------------------------------------------
# FilterSettings validation
# -----------------------------------------------------------------------


class TestFilterSettingsValidation:
    def test_valid_defaults(self):
        FilterSettings()

    def test_negative_min_cells(self):
        with pytest.raises(ValueError, match="min_cells"):
            FilterSettings(min_cells=-1)

    def test_min_percentage_above_100(self):
        with pytest.raises(ValueError, match="min_percentage"):
            FilterSettings(min_percentage=150.0)

    def test_min_percentage_negative(self):
        with pytest.raises(ValueError, match="min_percentage"):
            FilterSettings(min_percentage=-5.0)

    def test_check_representation_non_bool(self):
        with pytest.raises(TypeError):
            FilterSettings(check_representation="yes")


# -----------------------------------------------------------------------
# StorageSettings validation
# -----------------------------------------------------------------------


class TestStorageSettingsValidation:
    def test_valid_defaults(self):
        StorageSettings()

    def test_max_memory_ratio_above_1(self):
        with pytest.raises(ValueError, match="max_memory_ratio"):
            StorageSettings(max_memory_ratio=2.0)

    def test_max_memory_ratio_negative(self):
        with pytest.raises(ValueError, match="max_memory_ratio"):
            StorageSettings(max_memory_ratio=-0.5)

    def test_result_key_non_string(self):
        with pytest.raises(TypeError, match="result_key"):
            StorageSettings(result_key=123)

    def test_overwrite_non_bool(self):
        with pytest.raises(TypeError):
            StorageSettings(overwrite="yes")

    def test_disk_storage_dir_non_string(self):
        with pytest.raises(TypeError, match="disk_storage_dir"):
            StorageSettings(disk_storage_dir=123)


# -----------------------------------------------------------------------
# OutputSettings validation
# -----------------------------------------------------------------------


class TestOutputSettingsValidation:
    def test_valid_defaults(self):
        OutputSettings()

    def test_copy_non_bool(self):
        with pytest.raises(TypeError):
            OutputSettings(copy="yes")

    def test_inplace_non_bool(self):
        with pytest.raises(TypeError):
            OutputSettings(inplace=1)

    def test_progress_non_bool(self):
        with pytest.raises(TypeError):
            OutputSettings(progress="true")


# -----------------------------------------------------------------------
# ModelSettings validation
# -----------------------------------------------------------------------


class TestModelSettingsValidation:
    def test_valid_defaults(self):
        ModelSettings()

    def test_non_callable_function_predictor(self):
        with pytest.raises(TypeError, match="callable"):
            ModelSettings(function_predictor1="not_a_function")

    def test_non_callable_density_predictor(self):
        with pytest.raises(TypeError, match="callable"):
            ModelSettings(density_predictor1=42)

    def test_callable_accepted(self):
        ModelSettings(function_predictor1=lambda x: x)

    def test_model_objects_not_checked_callable(self):
        # model1/model2 are Any, not required to be callable
        ModelSettings(model1="anything")


# -----------------------------------------------------------------------
# AnnData-level validators
# -----------------------------------------------------------------------


class TestValidateObsmKey:
    def test_missing_key(self, adata):
        with pytest.raises(ValueError, match="not found in adata.obsm"):
            validate_obsm_key(adata, "nonexistent")

    def test_missing_dm_eigenvectors_hint(self, adata):
        del adata.obsm["DM_EigenVectors"]
        with pytest.raises(ValueError, match="Palantir"):
            validate_obsm_key(adata, "DM_EigenVectors")

    def test_valid_key(self, adata):
        validate_obsm_key(adata, "DM_EigenVectors")


class TestValidateGroupby:
    def test_missing_column(self, adata):
        with pytest.raises(ValueError, match="not found in adata.obs"):
            validate_groupby(adata, "nonexistent")

    def test_valid_column(self, adata):
        validate_groupby(adata, "condition")


class TestValidateConditions:
    def test_same_conditions(self, adata):
        with pytest.raises(ValueError, match="must be different"):
            validate_conditions(adata, "condition", "A", "A")

    def test_missing_condition(self, adata):
        with pytest.raises(ValueError, match="not found"):
            validate_conditions(adata, "condition", "A", "C")

    def test_valid_conditions(self, adata):
        validate_conditions(adata, "condition", "A", "B")


class TestValidateLayer:
    def test_missing_layer(self, adata):
        with pytest.raises(ValueError, match="not found in adata.layers"):
            validate_layer(adata, "nonexistent")

    def test_valid_layer(self, adata):
        validate_layer(adata, "log")

    def test_none_is_ok(self, adata):
        validate_layer(adata, None)


class TestValidateGenes:
    def test_missing_genes(self, adata):
        with pytest.raises(ValueError, match="not found"):
            validate_genes(adata, ["gene_0", "not_a_gene"])

    def test_empty_list(self, adata):
        with pytest.raises(ValueError, match="empty"):
            validate_genes(adata, [])

    def test_valid_genes(self, adata):
        validate_genes(adata, ["gene_0", "gene_1"])

    def test_none_is_ok(self, adata):
        validate_genes(adata, None)


class TestValidateSampleCol:
    def test_missing_column(self, adata):
        with pytest.raises(ValueError, match="not found"):
            validate_sample_col(adata, "nonexistent")

    def test_valid_column(self, adata):
        validate_sample_col(adata, "sample")

    def test_none_is_ok(self, adata):
        validate_sample_col(adata, None)


class TestValidateExpressionData:
    def test_nan_raises(self):
        arr = np.ones((10, 5))
        arr[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            validate_expression_data(arr)

    def test_inf_raises(self):
        arr = np.ones((10, 5))
        arr[0, 0] = np.inf
        with pytest.raises(ValueError, match="Inf"):
            validate_expression_data(arr)

    def test_valid_data(self):
        validate_expression_data(np.ones((10, 5)))

    def test_none_is_ok(self):
        validate_expression_data(None)


class TestValidateLandmarksShape:
    def test_feature_mismatch(self):
        with pytest.raises(ValueError, match="features"):
            validate_landmarks_shape(np.zeros((50, 3)), "DM_EigenVectors", 5)

    def test_not_2d(self):
        with pytest.raises(ValueError, match="2-D"):
            validate_landmarks_shape(np.zeros(50), "DM_EigenVectors", 5)

    def test_valid(self):
        validate_landmarks_shape(np.zeros((50, 5)), "DM_EigenVectors", 5)

    def test_none_is_ok(self):
        validate_landmarks_shape(None, "DM_EigenVectors", 5)


# -----------------------------------------------------------------------
# Scalar validators
# -----------------------------------------------------------------------


class TestScalarValidators:
    def test_probability_above_1(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            validate_probability(1.5, "p")

    def test_probability_zero(self):
        with pytest.raises(ValueError, match="positive"):
            validate_probability(0.0, "p")

    def test_probability_valid(self):
        assert validate_probability(0.5, "p") == 0.5

    def test_probability_optional_none(self):
        assert validate_probability(None, "p", optional=True) is None

    def test_non_negative_int_negative(self):
        with pytest.raises(ValueError, match=">= 0"):
            validate_non_negative_int(-1, "n")

    def test_non_negative_int_zero(self):
        assert validate_non_negative_int(0, "n") == 0

    def test_non_negative_int_bool_rejected(self):
        with pytest.raises(TypeError, match="integer"):
            validate_non_negative_int(True, "n")

    def test_non_negative_float_negative(self):
        with pytest.raises(ValueError, match=">= 0"):
            validate_non_negative_float(-0.5, "x")

    def test_non_negative_float_nan(self):
        with pytest.raises(ValueError, match="NaN"):
            validate_non_negative_float(float("nan"), "x")

    def test_non_negative_float_zero(self):
        assert validate_non_negative_float(0.0, "x") == 0.0


# -----------------------------------------------------------------------
# Integration: de()/da() catch condition1 == condition2
# -----------------------------------------------------------------------


class TestAPIValidationIntegration:
    def test_de_same_conditions(self, adata):
        with pytest.raises(ValueError, match="must be different"):
            kompot.de(adata, "condition", "A", "A")

    def test_da_same_conditions(self, adata):
        with pytest.raises(ValueError, match="must be different"):
            kompot.da(adata, "condition", "A", "A")

    def test_de_bad_obsm_key(self, adata):
        with pytest.raises(ValueError, match="not found in adata.obsm"):
            kompot.de(adata, "condition", "A", "B", obsm_key="nope")

    def test_da_bad_obsm_key(self, adata):
        with pytest.raises(ValueError, match="not found in adata.obsm"):
            kompot.da(adata, "condition", "A", "B", obsm_key="nope")

    def test_de_landmarks_shape_mismatch(self, adata):
        with pytest.raises(ValueError, match="features"):
            kompot.de(
                adata,
                "condition",
                "A",
                "B",
                gp=GPSettings(landmarks=np.zeros((50, 99))),
            )

    def test_da_landmarks_shape_mismatch(self, adata):
        with pytest.raises(ValueError, match="features"):
            kompot.da(
                adata,
                "condition",
                "A",
                "B",
                gp=GPSettings(landmarks=np.zeros((50, 99))),
            )
