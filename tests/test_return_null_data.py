"""Tests for return_null_data and null data in return_full_results.

Covers the new ``OutputSettings.return_null_data`` flag and the inclusion
of null gene data in ``result_dict["null"]`` when ``return_full_results``
is True.
"""

import numpy as np
import pandas as pd
import pytest
import anndata as ad

import kompot
from kompot import FDRSettings, OutputSettings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_adata(n_cells=80, n_genes=30, seed=42):
    """Create a simple AnnData with a few differential genes."""
    rng = np.random.RandomState(seed)
    n1 = n_cells // 2
    n2 = n_cells - n1

    X_embed = rng.randn(n_cells, 5)
    expr = rng.negative_binomial(10, 0.3, (n_cells, n_genes)).astype(float)

    # Make first 3 genes differentially expressed
    for g in range(3):
        expr[n1:, g] *= 5.0

    gene_names = [f"Gene_{i:03d}" for i in range(n_genes)]
    adata = ad.AnnData(
        X=expr,
        obs=pd.DataFrame(
            {"condition": ["A"] * n1 + ["B"] * n2},
            index=[f"c{i}" for i in range(n_cells)],
        ),
        var=pd.DataFrame(index=gene_names),
    )
    adata.obsm["DM_EigenVectors"] = X_embed
    return adata


_FAST_GP = dict(n_landmarks=10, batch_size=0)


# =====================================================================
# 1. return_full_results=True includes full null data
# =====================================================================


class TestReturnFullResultsNull:
    """When return_full_results=True and FDR is computed, result_dict
    should contain a 'null' key with both lightweight metadata and full
    expression matrices."""

    def test_null_key_present(self):
        adata = _make_adata()
        result = kompot.de(
            adata, "condition", "A", "B",
            gp=kompot.GPSettings(**_FAST_GP),
            fdr=FDRSettings(null_genes=20, null_seed=42),
            output=OutputSettings(
                return_full_results=True, progress=False,
            ),
        )
        assert "null" in result

    def test_null_table_structure(self):
        adata = _make_adata()
        result = kompot.de(
            adata, "condition", "A", "B",
            gp=kompot.GPSettings(**_FAST_GP),
            fdr=FDRSettings(null_genes=15, null_seed=0),
            output=OutputSettings(
                return_full_results=True, progress=False,
            ),
        )
        null = result["null"]
        assert isinstance(null["table"], pd.DataFrame)
        assert "mahalanobis" in null["table"].columns
        assert "mean_lfc" in null["table"].columns
        assert len(null["table"]) == 15

    def test_null_metadata_fields(self):
        adata = _make_adata()
        result = kompot.de(
            adata, "condition", "A", "B",
            gp=kompot.GPSettings(**_FAST_GP),
            fdr=FDRSettings(null_genes=10, null_seed=99),
            output=OutputSettings(
                return_full_results=True, progress=False,
            ),
        )
        null = result["null"]
        assert null["seed"] == 99
        assert null["n_null"] == 10
        assert null["source"] == "internal"
        assert isinstance(null["gene_indices"], list)
        assert len(null["gene_indices"]) == 10
        assert isinstance(null["gene_names"], list)
        assert len(null["gene_names"]) == 10
        # Null gene names should follow the NULL_i_GeneName pattern
        for name in null["gene_names"]:
            assert name.startswith("NULL_")

    def test_full_expression_matrices_present(self):
        adata = _make_adata()
        n_cells = adata.n_obs
        n_null = 12
        result = kompot.de(
            adata, "condition", "A", "B",
            gp=kompot.GPSettings(**_FAST_GP),
            fdr=FDRSettings(null_genes=n_null, null_seed=42),
            output=OutputSettings(
                return_full_results=True, progress=False,
            ),
        )
        null = result["null"]

        # Core expression matrices should always be present
        for key in (
            "condition1_imputed", "condition2_imputed",
            "fold_change",
        ):
            assert key in null, f"Missing key: {key}"
            assert null[key].shape[1] == n_null, (
                f"{key} has wrong number of null genes: "
                f"{null[key].shape[1]} != {n_null}"
            )
            assert null[key].shape[0] > 0

        # Std matrices are only present when use_empirical_variance=True
        # (otherwise the variance is broadcast as a single column).
        # When present, shape must match.
        for key in ("condition1_std", "condition2_std"):
            if key in null:
                assert null[key].shape[1] == n_null

    def test_std_present_with_empirical_variance(self):
        adata = _make_adata()
        n_null = 10
        result = kompot.de(
            adata, "condition", "A", "B",
            gp=kompot.GPSettings(**_FAST_GP, use_empirical_variance=True),
            fdr=FDRSettings(null_genes=n_null, null_seed=42),
            output=OutputSettings(
                return_full_results=True, progress=False,
            ),
        )
        null = result["null"]
        for key in ("condition1_std", "condition2_std"):
            assert key in null, f"Expected {key} with empirical variance"
            assert null[key].shape[1] == n_null

    def test_fold_change_zscores_present(self):
        adata = _make_adata()
        result = kompot.de(
            adata, "condition", "A", "B",
            gp=kompot.GPSettings(**_FAST_GP),
            fdr=FDRSettings(null_genes=8, null_seed=42),
            output=OutputSettings(
                return_full_results=True, progress=False,
            ),
        )
        null = result["null"]
        assert "fold_change_zscores" in null
        assert null["fold_change_zscores"].shape[1] == 8

    def test_real_table_not_contaminated(self):
        """The main result table should only contain real genes."""
        adata = _make_adata()
        result = kompot.de(
            adata, "condition", "A", "B",
            gp=kompot.GPSettings(**_FAST_GP),
            fdr=FDRSettings(null_genes=20, null_seed=42),
            output=OutputSettings(
                return_full_results=True, progress=False,
            ),
        )
        # Real table should have exactly n_genes rows
        assert len(result["table"]) == adata.n_vars
        # No NULL_ prefixed genes in the real table
        assert not any(
            idx.startswith("NULL_") for idx in result["table"].index
        )

    def test_mahalanobis_values_are_nonnegative(self):
        adata = _make_adata()
        result = kompot.de(
            adata, "condition", "A", "B",
            gp=kompot.GPSettings(**_FAST_GP),
            fdr=FDRSettings(null_genes=10, null_seed=42),
            output=OutputSettings(
                return_full_results=True, progress=False,
            ),
        )
        assert (result["null"]["mahalanobis"] >= 0).all()
        assert (result["null"]["table"]["mahalanobis"] >= 0).all()


# =====================================================================
# 2. return_null_data=True (lightweight, no expression matrices)
# =====================================================================


class TestReturnNullDataLightweight:
    """When return_null_data=True (without return_full_results),
    result_dict should contain null metadata but NOT expression matrices."""

    def test_null_key_present_without_return_full_results(self):
        """return_null_data=True alone should return the result dict."""
        adata = _make_adata()
        result = kompot.de(
            adata, "condition", "A", "B",
            gp=kompot.GPSettings(**_FAST_GP),
            fdr=FDRSettings(null_genes=10, null_seed=42),
            output=OutputSettings(
                return_full_results=False,
                return_null_data=True,
                progress=False,
            ),
        )
        assert isinstance(result, dict)
        assert "null" in result
        assert "table" in result  # result dict always has a table

    def test_lightweight_no_expression_in_null(self):
        """return_null_data=True without return_full_results should return
        the dict but the null entry should NOT have expression matrices."""
        adata = _make_adata()
        result = kompot.de(
            adata, "condition", "A", "B",
            gp=kompot.GPSettings(**_FAST_GP),
            fdr=FDRSettings(null_genes=10, null_seed=42),
            output=OutputSettings(
                return_full_results=False,
                return_null_data=True,
                progress=False,
            ),
        )
        null = result["null"]
        assert isinstance(null["table"], pd.DataFrame)
        assert null["seed"] == 42
        for key in (
            "condition1_imputed", "condition2_imputed",
            "fold_change", "fold_change_zscores",
        ):
            assert key not in null, f"Lightweight mode should not include {key}"

    def test_return_null_data_with_copy(self):
        """return_null_data=True with copy=True returns (dict, adata)."""
        adata = _make_adata()
        result = kompot.de(
            adata, "condition", "A", "B",
            gp=kompot.GPSettings(**_FAST_GP),
            fdr=FDRSettings(null_genes=10, null_seed=42),
            output=OutputSettings(
                return_full_results=False,
                return_null_data=True,
                copy=True,
                progress=False,
            ),
        )
        assert isinstance(result, tuple)
        result_dict, adata_copy = result
        assert "null" in result_dict

    def test_no_expression_matrices_when_lightweight_internal(self):
        """Test _compute_fdr directly: return_null_data without
        return_full_results should not capture expression matrices."""
        from kompot.anndata._de_helpers import _compute_fdr

        n_real = 10
        n_null = 5
        n_cells = 20
        rng = np.random.RandomState(0)

        expression_results = {
            "mean_log_fold_change": rng.randn(n_real + n_null),
            "mahalanobis_distances": np.abs(rng.randn(n_real + n_null)),
            "condition1_imputed": rng.randn(n_cells, n_real + n_null),
            "condition2_imputed": rng.randn(n_cells, n_real + n_null),
            "fold_change": rng.randn(n_cells, n_real + n_null),
            "fold_change_zscores": rng.randn(n_cells, n_real + n_null),
            "condition1_std": rng.randn(n_cells, n_real + n_null),
            "condition2_std": rng.randn(n_cells, n_real + n_null),
            "ptp": rng.rand(n_real + n_null),
        }
        selected = [f"Gene_{i}" for i in range(n_real)]
        expanded = selected + [f"NULL_{i}_X" for i in range(n_null)]
        null_indices = list(range(n_null))

        fdr_results = _compute_fdr(
            expression_results, selected, expanded, null_indices,
            fdr_threshold=0.05,
            return_null_data=True,
            return_full_results=False,
            null_seed=42,
        )

        null = fdr_results["null"]

        # Lightweight metadata should be present
        assert isinstance(null["table"], pd.DataFrame)
        assert null["seed"] == 42
        assert null["n_null"] == n_null
        assert len(null["gene_indices"]) == n_null

        # Expression matrices should NOT be present
        for key in (
            "condition1_imputed", "condition2_imputed",
            "fold_change", "fold_change_zscores",
            "condition1_std", "condition2_std",
        ):
            assert key not in null, f"Lightweight mode should not include {key}"

    def test_full_null_has_expression_matrices(self):
        """return_full_results=True should include expression matrices."""
        from kompot.anndata._de_helpers import _compute_fdr

        n_real = 10
        n_null = 5
        n_cells = 20
        rng = np.random.RandomState(0)

        expression_results = {
            "mean_log_fold_change": rng.randn(n_real + n_null),
            "mahalanobis_distances": np.abs(rng.randn(n_real + n_null)),
            "condition1_imputed": rng.randn(n_cells, n_real + n_null),
            "condition2_imputed": rng.randn(n_cells, n_real + n_null),
            "fold_change": rng.randn(n_cells, n_real + n_null),
            "fold_change_zscores": rng.randn(n_cells, n_real + n_null),
            "condition1_std": rng.randn(n_cells, n_real + n_null),
            "condition2_std": rng.randn(n_cells, n_real + n_null),
            "ptp": rng.rand(n_real + n_null),
        }
        selected = [f"Gene_{i}" for i in range(n_real)]
        expanded = selected + [f"NULL_{i}_X" for i in range(n_null)]
        null_indices = list(range(n_null))

        fdr_results = _compute_fdr(
            expression_results, selected, expanded, null_indices,
            fdr_threshold=0.05,
            return_null_data=True,
            return_full_results=True,
            null_seed=42,
        )

        null = fdr_results["null"]

        # Expression matrices SHOULD be present
        for key in (
            "condition1_imputed", "condition2_imputed",
            "fold_change", "fold_change_zscores",
            "condition1_std", "condition2_std",
        ):
            assert key in null, f"Full mode should include {key}"
            assert null[key].shape == (n_cells, n_null)


# =====================================================================
# 3. No null data when FDR is disabled
# =====================================================================


class TestNoNullWithoutFDR:
    """When FDR is disabled (null_genes=0), no null key should appear."""

    def test_no_null_key_without_fdr(self):
        adata = _make_adata()
        result = kompot.de(
            adata, "condition", "A", "B",
            gp=kompot.GPSettings(**_FAST_GP),
            fdr=FDRSettings(null_genes=0),
            output=OutputSettings(
                return_full_results=True, progress=False,
            ),
        )
        assert "null" not in result

    def test_no_null_key_return_null_data_without_fdr(self):
        """return_null_data=True should be harmless when FDR is off."""
        adata = _make_adata()
        result = kompot.de(
            adata, "condition", "A", "B",
            gp=kompot.GPSettings(**_FAST_GP),
            fdr=FDRSettings(null_genes=0),
            output=OutputSettings(
                return_full_results=True,
                return_null_data=True,
                progress=False,
            ),
        )
        assert "null" not in result


# =====================================================================
# 4. External null distribution
# =====================================================================


class TestNullDataWithExternalNull:
    """Null data should reflect provenance when using external nulls."""

    def test_external_null_mahalanobis_source(self):
        adata = _make_adata()
        ext_null = np.random.RandomState(0).exponential(0.5, 100)
        result = kompot.de(
            adata, "condition", "A", "B",
            gp=kompot.GPSettings(**_FAST_GP),
            fdr=FDRSettings(null_genes=0, null_mahalanobis=ext_null),
            output=OutputSettings(
                return_full_results=True, progress=False,
            ),
        )
        null = result["null"]
        assert null["source"] == "external"
        assert null["n_null"] == 100

    def test_combined_null_source(self):
        adata = _make_adata()
        ext_null = np.random.RandomState(0).exponential(0.5, 50)
        result = kompot.de(
            adata, "condition", "A", "B",
            gp=kompot.GPSettings(**_FAST_GP),
            fdr=FDRSettings(
                null_genes=10, null_seed=42,
                null_mahalanobis=ext_null,
                combine_with_internal=True,
            ),
            output=OutputSettings(
                return_full_results=True, progress=False,
            ),
        )
        null = result["null"]
        assert null["source"] == "combined"
        assert null["n_null"] == 60  # 10 internal + 50 external

    def test_internal_only_source(self):
        adata = _make_adata()
        result = kompot.de(
            adata, "condition", "A", "B",
            gp=kompot.GPSettings(**_FAST_GP),
            fdr=FDRSettings(null_genes=15, null_seed=42),
            output=OutputSettings(
                return_full_results=True, progress=False,
            ),
        )
        assert result["null"]["source"] == "internal"


# =====================================================================
# 5. Reproducibility
# =====================================================================


class TestNullDataReproducibility:
    """Same seed should produce identical null data."""

    def test_same_seed_same_results(self):
        params = dict(
            gp=kompot.GPSettings(**_FAST_GP),
            fdr=FDRSettings(null_genes=10, null_seed=123),
            output=OutputSettings(
                return_full_results=True, progress=False,
            ),
        )
        adata1 = _make_adata()
        adata2 = _make_adata()

        r1 = kompot.de(adata1, "condition", "A", "B", **params)
        r2 = kompot.de(adata2, "condition", "A", "B", **params)

        np.testing.assert_allclose(
            r1["null"]["mahalanobis"], r2["null"]["mahalanobis"],
            rtol=1e-3,
        )
        np.testing.assert_array_equal(
            r1["null"]["gene_indices"], r2["null"]["gene_indices"]
        )
        np.testing.assert_allclose(
            r1["null"]["fold_change"], r2["null"]["fold_change"],
            rtol=1e-3,
        )

    def test_different_seed_different_results(self):
        adata1 = _make_adata()
        adata2 = _make_adata()

        r1 = kompot.de(
            adata1, "condition", "A", "B",
            gp=kompot.GPSettings(**_FAST_GP),
            fdr=FDRSettings(null_genes=10, null_seed=0),
            output=OutputSettings(
                return_full_results=True, progress=False,
            ),
        )
        r2 = kompot.de(
            adata2, "condition", "A", "B",
            gp=kompot.GPSettings(**_FAST_GP),
            fdr=FDRSettings(null_genes=10, null_seed=999),
            output=OutputSettings(
                return_full_results=True, progress=False,
            ),
        )

        # Gene indices should differ with different seeds
        assert not np.array_equal(
            r1["null"]["gene_indices"], r2["null"]["gene_indices"]
        )


# =====================================================================
# 6. Null table can be concatenated with real table
# =====================================================================


class TestNullTableCompatibility:
    """The null table should be concatenatable with the real table for
    easy comparison and plotting."""

    def test_concat_works(self):
        adata = _make_adata()
        result = kompot.de(
            adata, "condition", "A", "B",
            gp=kompot.GPSettings(**_FAST_GP),
            fdr=FDRSettings(null_genes=10, null_seed=42),
            output=OutputSettings(
                return_full_results=True, progress=False,
            ),
        )
        real_table = result["table"]
        null_table = result["null"]["table"]

        # Shared columns should overlap
        shared = set(real_table.columns) & set(null_table.columns)
        assert "mahalanobis" in shared
        assert "mean_lfc" in shared

        # Concat should work
        combined = pd.concat(
            [real_table[list(shared)], null_table[list(shared)]],
        )
        assert len(combined) == len(real_table) + len(null_table)
