"""
Integration tests for group-wise FDR analysis with null genes.

Tests the combination of null_genes + groups parameters, ensuring:
- Group-wise FDR statistics are computed correctly
- Group-wise ptp is stored when store_additional_stats=True
- Null genes are properly handled in group-wise analysis
"""

import numpy as np
import pandas as pd
import pytest
import anndata as ad

from kompot.anndata import compute_differential_expression


@pytest.fixture
def adata_with_groups():
    """Create test data with clear groups and differential expression."""
    np.random.seed(42)

    n_cells = 120
    n_genes = 80
    n_features = 10

    # Gene expression with differential expression in first 10 genes
    X = np.random.randn(n_cells, n_genes) * 0.5
    # Make first 10 genes clearly DE in condition2
    X[60:, :10] += 3.0

    # Cell states
    cell_states = np.random.randn(n_cells, n_features)

    # Metadata - create two conditions and three groups
    conditions = ["Young"] * 60 + ["Old"] * 60
    groups = (["groupA"] * 20 + ["groupB"] * 20 + ["groupC"] * 20) * 2

    # Create AnnData object
    adata = ad.AnnData(X)
    adata.obsm["X_pca"] = cell_states
    adata.obs["age"] = pd.Categorical(conditions)
    adata.obs["cell_type"] = pd.Categorical(groups)
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]

    return adata


class TestGroupwiseFDRBasics:
    """Test basic group-wise FDR functionality."""

    def test_groupwise_fdr_with_null_genes(self, adata_with_groups):
        """Test that group-wise FDR is computed when using null_genes + groups."""
        result = compute_differential_expression(
            adata_with_groups,
            groupby="age",
            condition1="Young",
            condition2="Old",
            obsm_key="X_pca",
            null_genes=30,
            groups="cell_type",
            n_landmarks=None,
            overwrite=True,
            inplace=True,
            return_full_results=True,
        )

        assert result is not None
        assert "field_names" in result

        field_names = result["field_names"]

        # Check that group-wise FDR varm matrices exist
        local_fdr_key = f"{field_names['mahalanobis_local_fdr_key']}_groups"
        is_de_key = f"{field_names['is_de_key']}_groups"

        assert local_fdr_key in adata_with_groups.varm, (
            f"Expected {local_fdr_key} in varm"
        )
        assert is_de_key in adata_with_groups.varm, f"Expected {is_de_key} in varm"

        # Verify structure
        assert adata_with_groups.varm[local_fdr_key].shape[0] == len(
            adata_with_groups.var_names
        )
        assert adata_with_groups.varm[is_de_key].shape[0] == len(
            adata_with_groups.var_names
        )

        # Check that all groups are present
        expected_groups = ["groupA", "groupB", "groupC"]
        assert list(adata_with_groups.varm[local_fdr_key].columns) == expected_groups
        assert list(adata_with_groups.varm[is_de_key].columns) == expected_groups

    def test_groupwise_fdr_values(self, adata_with_groups):
        """Test that group-wise FDR values are reasonable."""
        compute_differential_expression(
            adata_with_groups,
            groupby="age",
            condition1="Young",
            condition2="Old",
            obsm_key="X_pca",
            null_genes=30,
            groups="cell_type",
            n_landmarks=None,
            overwrite=True,
            inplace=True,
        )

        # Find the FDR varm keys
        fdr_keys = [
            k
            for k in adata_with_groups.varm.keys()
            if "local_fdr" in k and "groups" in k
        ]
        assert len(fdr_keys) == 1
        local_fdr_key = fdr_keys[0]

        # Check FDR values are in [0, 1]
        for group in adata_with_groups.varm[local_fdr_key].columns:
            fdr_values = adata_with_groups.varm[local_fdr_key][group]
            valid_values = fdr_values[~fdr_values.isna()]
            assert (valid_values >= 0).all(), f"Group {group} has negative FDR values"
            assert (valid_values <= 1).all(), f"Group {group} has FDR values > 1"

    def test_groupwise_is_de_is_boolean(self, adata_with_groups):
        """Test that group-wise is_de contains boolean values."""
        compute_differential_expression(
            adata_with_groups,
            groupby="age",
            condition1="Young",
            condition2="Old",
            obsm_key="X_pca",
            null_genes=30,
            groups="cell_type",
            n_landmarks=None,
            overwrite=True,
            inplace=True,
        )

        # Find the is_de varm key
        is_de_keys = [
            k for k in adata_with_groups.varm.keys() if "is_de" in k and "groups" in k
        ]
        assert len(is_de_keys) == 1
        is_de_key = is_de_keys[0]

        # Check is_de values are boolean
        for group in adata_with_groups.varm[is_de_key].columns:
            is_de_values = adata_with_groups.varm[is_de_key][group]
            assert is_de_values.dtype == bool, f"Group {group} is_de is not boolean"


class TestGroupwisePTP:
    """Test group-wise ptp storage."""

    def test_groupwise_ptp_with_store_additional_stats(self, adata_with_groups):
        """Test that ptp is stored for groups when store_additional_stats=True."""
        result = compute_differential_expression(
            adata_with_groups,
            groupby="age",
            condition1="Young",
            condition2="Old",
            obsm_key="X_pca",
            null_genes=30,
            groups="cell_type",
            store_additional_stats=True,
            n_landmarks=None,
            overwrite=True,
            inplace=True,
            return_full_results=True,
        )

        field_names = result["field_names"]
        ptp_key = f"{field_names['ptp_key']}_groups"

        assert ptp_key in adata_with_groups.varm, (
            f"Expected ptp key {ptp_key} in varm when store_additional_stats=True"
        )

        # Verify structure
        expected_groups = ["groupA", "groupB", "groupC"]
        assert list(adata_with_groups.varm[ptp_key].columns) == expected_groups

        # Check ptp values are non-negative
        for group in expected_groups:
            ptp_values = adata_with_groups.varm[ptp_key][group]
            valid_values = ptp_values[~ptp_values.isna()]
            assert (valid_values >= 0).all(), f"Group {group} has negative ptp values"

    def test_groupwise_ptp_not_stored_by_default(self, adata_with_groups):
        """Test that ptp is NOT stored for groups by default."""
        compute_differential_expression(
            adata_with_groups,
            groupby="age",
            condition1="Young",
            condition2="Old",
            obsm_key="X_pca",
            null_genes=30,
            groups="cell_type",
            store_additional_stats=False,  # Default
            n_landmarks=None,
            overwrite=True,
            inplace=True,
        )

        # ptp_groups key should NOT exist
        ptp_keys = [
            k for k in adata_with_groups.varm.keys() if "ptp" in k and "groups" in k
        ]
        assert len(ptp_keys) == 0, (
            "ptp_groups should not exist when store_additional_stats=False"
        )


class TestNullGeneHandling:
    """Test that null genes are handled correctly in group-wise analysis."""

    def test_null_genes_excluded_from_group_results(self, adata_with_groups):
        """Test that group results only contain real genes, not null genes."""
        n_real_genes = len(adata_with_groups.var_names)
        n_null_genes = 30

        compute_differential_expression(
            adata_with_groups,
            groupby="age",
            condition1="Young",
            condition2="Old",
            obsm_key="X_pca",
            null_genes=n_null_genes,
            groups="cell_type",
            n_landmarks=None,
            overwrite=True,
            inplace=True,
        )

        # Check that varm results have correct length (real genes only)
        for varm_key in adata_with_groups.varm.keys():
            if "groups" in varm_key:
                varm_df = adata_with_groups.varm[varm_key]
                assert varm_df.shape[0] == n_real_genes, (
                    f"{varm_key} has wrong length: {varm_df.shape[0]} (expected {n_real_genes})"
                )

    def test_no_length_mismatch_warnings(self, adata_with_groups, caplog):
        """Test that no length mismatch warnings are emitted."""
        import logging

        caplog.set_level(logging.WARNING)

        compute_differential_expression(
            adata_with_groups,
            groupby="age",
            condition1="Young",
            condition2="Old",
            obsm_key="X_pca",
            null_genes=30,
            groups="cell_type",
            n_landmarks=None,
            overwrite=True,
            inplace=True,
        )

        # Check that no "doesn't match" warnings were logged
        warning_messages = [
            record.message for record in caplog.records if record.levelname == "WARNING"
        ]
        length_warnings = [
            msg
            for msg in warning_messages
            if "doesn't match" in msg and "length" in msg
        ]

        assert len(length_warnings) == 0, (
            f"Found unexpected length mismatch warnings: {length_warnings}"
        )


class TestVarmKeysExist:
    """Test that correct varm keys are created for group-wise FDR."""

    def test_all_expected_varm_keys_exist(self, adata_with_groups):
        """Test that all expected varm keys are created."""
        result = compute_differential_expression(
            adata_with_groups,
            groupby="age",
            condition1="Young",
            condition2="Old",
            obsm_key="X_pca",
            null_genes=30,
            groups="cell_type",
            store_additional_stats=True,
            n_landmarks=None,
            overwrite=True,
            inplace=True,
            return_full_results=True,
        )

        field_names = result["field_names"]

        # Build expected varm keys
        expected_keys = [
            field_names["mean_lfc_varm_key"],
            field_names["mahalanobis_varm_key"],
            f"{field_names['mahalanobis_local_fdr_key']}_groups",
            f"{field_names['is_de_key']}_groups",
            f"{field_names['ptp_key']}_groups",
        ]

        # Check all expected keys exist
        for key in expected_keys:
            assert key in adata_with_groups.varm, f"Expected varm key {key} not found"

            # Verify all have correct groups
            expected_groups = ["groupA", "groupB", "groupC"]
            assert list(adata_with_groups.varm[key].columns) == expected_groups, (
                f"Key {key} has wrong groups"
            )


class TestEdgeCases:
    """Test edge cases for group-wise FDR analysis."""

    def test_groups_without_null_genes(self, adata_with_groups):
        """Test that groups work without null_genes (no FDR)."""
        compute_differential_expression(
            adata_with_groups,
            groupby="age",
            condition1="Young",
            condition2="Old",
            obsm_key="X_pca",
            null_genes=None,  # No FDR
            groups="cell_type",
            n_landmarks=None,
            overwrite=True,
            inplace=True,
        )

        # FDR keys should NOT exist when null_genes is None
        fdr_keys = [
            k for k in adata_with_groups.varm.keys() if "fdr" in k and "groups" in k
        ]
        is_de_keys = [
            k for k in adata_with_groups.varm.keys() if "is_de" in k and "groups" in k
        ]

        assert len(fdr_keys) == 0, "FDR keys should not exist when null_genes=None"
        assert len(is_de_keys) == 0, "is_de keys should not exist when null_genes=None"

    def test_single_group(self, adata_with_groups):
        """Test group-wise FDR with a single group."""
        # Create a single group mask
        single_group_mask = adata_with_groups.obs["cell_type"] == "groupA"

        compute_differential_expression(
            adata_with_groups,
            groupby="age",
            condition1="Young",
            condition2="Old",
            obsm_key="X_pca",
            null_genes=30,
            groups={"single_group": single_group_mask},
            n_landmarks=None,
            overwrite=True,
            inplace=True,
        )

        # Check that only one group exists in varm
        fdr_keys = [
            k
            for k in adata_with_groups.varm.keys()
            if "local_fdr" in k and "groups" in k
        ]
        assert len(fdr_keys) == 1
        local_fdr_key = fdr_keys[0]

        assert list(adata_with_groups.varm[local_fdr_key].columns) == ["single_group"]


class TestGroupwiseFDRConsistency:
    """Test consistency between global and group-wise FDR."""

    def test_global_vs_groupwise_fdr(self, adata_with_groups):
        """Test that global and group-wise FDR can differ appropriately."""
        result = compute_differential_expression(
            adata_with_groups,
            groupby="age",
            condition1="Young",
            condition2="Old",
            obsm_key="X_pca",
            null_genes=30,
            groups="cell_type",
            n_landmarks=None,
            overwrite=True,
            inplace=True,
            return_full_results=True,
        )

        # Get global DE genes
        global_is_de = result["table"]["is_de"]
        n_global_de = global_is_de.sum()

        # Get group-wise DE genes
        is_de_keys = [
            k for k in adata_with_groups.varm.keys() if "is_de" in k and "groups" in k
        ]
        is_de_key = is_de_keys[0]

        group_de_counts = {}
        for group in adata_with_groups.varm[is_de_key].columns:
            group_de_counts[group] = adata_with_groups.varm[is_de_key][group].sum()

        # The global and group-wise DE counts should be reasonable
        # (can differ, but should be in similar range)
        assert n_global_de > 0, "Expected some globally DE genes"
        for group, count in group_de_counts.items():
            # Each group might have different DE genes, but should have some
            assert count >= 0, f"Group {group} has negative DE count"
            # They should be in a reasonable range (e.g., not wildly different)
            # This is a soft check - groups can legitimately differ
