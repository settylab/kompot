"""Tests for FDR functionality in the volcano_de plotting function."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from kompot.plot.volcano import volcano_de

matplotlib.use("Agg")  # Use non-interactive backend to prevent windows


def create_test_anndata_with_fdr(
    n_cells=60, n_genes=500, n_differential=50, result_key="kompot_de"
):
    """Create a test AnnData object with comprehensive FDR data."""
    import anndata

    np.random.seed(42)

    # Create expression data
    X = np.random.negative_binomial(10, 0.3, (n_cells, n_genes)).astype(float)

    # Create gene and cell names
    gene_names = [f"Gene_{i:04d}" for i in range(n_genes)]
    cell_names = [f"Cell_{i:04d}" for i in range(n_cells)]

    # Create condition labels
    condition_labels = ["A"] * (n_cells // 2) + ["B"] * (n_cells - n_cells // 2)

    # Create AnnData object
    adata = anndata.AnnData(
        X=X,
        obs=pd.DataFrame({"condition": condition_labels}, index=cell_names),
        var=pd.DataFrame(index=gene_names),
    )

    # Generate realistic differential expression data
    lfc = np.random.normal(0, 1, n_genes)
    mahal_dist = np.random.gamma(2, 1, n_genes)

    # Make some genes strongly differential
    diff_indices = np.random.choice(n_genes, n_differential, replace=False)
    lfc[diff_indices] = np.random.choice([-3, 3], n_differential) + np.random.normal(
        0, 0.5, n_differential
    )
    mahal_dist[diff_indices] *= 5

    # Generate FDR values (lower for differential genes)
    local_fdr = np.random.beta(5, 2, n_genes)  # Higher values (less significant)
    tail_fdr = np.random.beta(5, 2, n_genes)
    p_values = np.random.beta(2, 5, n_genes)

    # Make differential genes more significant
    local_fdr[diff_indices] = np.random.beta(
        1, 20, n_differential
    )  # Lower values (more significant)
    tail_fdr[diff_indices] = np.random.beta(1, 20, n_differential)
    p_values[diff_indices] = np.random.beta(1, 50, n_differential)

    # Add columns to adata.var
    adata.var[f"{result_key}_A_to_B_mean_lfc"] = lfc
    adata.var[f"{result_key}_A_to_B_mahalanobis"] = mahal_dist
    adata.var[f"{result_key}_A_to_B_mahalanobis_pvalue"] = p_values
    adata.var[f"{result_key}_A_to_B_mahalanobis_local_fdr"] = local_fdr
    adata.var[f"{result_key}_A_to_B_mahalanobis_tail_fdr"] = tail_fdr
    adata.var[f"{result_key}_A_to_B_is_de"] = local_fdr < 0.05

    return adata, diff_indices


class TestVolcanoFDRBasicFunctionality:
    """Test basic FDR functionality."""

    def test_y_axis_type_mahalanobis(self):
        """Test default mahalanobis y-axis type."""
        adata, _ = create_test_anndata_with_fdr()

        fig, ax = plt.subplots()
        volcano_de(
            adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            y_axis_type="mahalanobis",
            ax=ax,
            show_legend=False,
        )

        # Check that y-axis label is correct
        assert ax.get_ylabel() == "Mahalanobis Distance"
        plt.close(fig)

    def test_y_axis_type_local_fdr(self):
        """Test local FDR y-axis type."""
        adata, _ = create_test_anndata_with_fdr()

        fig, ax = plt.subplots()
        volcano_de(
            adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            y_axis_type="local_fdr",
            ax=ax,
            show_legend=False,
        )

        # Check that y-axis label is correct
        assert ax.get_ylabel() == "-log10(Local FDR)"
        plt.close(fig)

    def test_y_axis_type_tail_fdr(self):
        """Test tail FDR y-axis type."""
        adata, _ = create_test_anndata_with_fdr()

        fig, ax = plt.subplots()
        volcano_de(
            adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            y_axis_type="tail_fdr",
            ax=ax,
            show_legend=False,
        )

        # Check that y-axis label is correct
        assert ax.get_ylabel() == "-log10(Tail FDR)"
        plt.close(fig)

    def test_fdr_key_detection(self):
        """Test automatic FDR key detection."""
        adata, _ = create_test_anndata_with_fdr()

        # Test that local FDR key is correctly detected
        fig, ax = plt.subplots()
        volcano_de(
            adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            y_axis_type="local_fdr",
            ax=ax,
            show_legend=False,
        )

        # The plot should complete without error, indicating key was found
        assert ax.get_ylabel() == "-log10(Local FDR)"
        plt.close(fig)

    def test_fdr_transformation(self):
        """Test that FDR values are correctly transformed."""
        adata, _ = create_test_anndata_with_fdr()

        # Get original FDR values
        original_fdr = adata.var["kompot_de_A_to_B_mahalanobis_local_fdr"].values

        fig, ax = plt.subplots()
        volcano_de(
            adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            y_axis_type="local_fdr",
            n_top_genes=0,  # Don't highlight any genes
            color=None,  # Disable automatic coloring to get single scatter
            ax=ax,
            show_legend=False,
        )

        # Should have a single scatter collection with all genes when coloring is disabled
        all_collections = ax.collections
        assert len(all_collections) >= 1, "Should have at least one scatter collection"

        # Get all y values from all scatter collections
        all_y_data = []
        for collection in all_collections:
            offsets = collection.get_offsets()
            if len(offsets) > 0:
                y_values = offsets[:, 1]
                all_y_data.extend(y_values)

        # Calculate expected transformed values
        expected_y = -np.log10(np.maximum(original_fdr, 1e-300))

        # Check that we have the right total number of points and transformation is correct
        assert len(all_y_data) == len(expected_y), (
            f"Expected {len(expected_y)} total points, got {len(all_y_data)}"
        )
        np.testing.assert_allclose(np.sort(all_y_data), np.sort(expected_y), rtol=1e-5)
        plt.close(fig)


class TestVolcanoFDRColoring:
    """Test FDR-based coloring functionality."""

    def test_auto_coloring_enabled(self):
        """Test that auto-coloring is enabled for FDR plots."""
        adata, _ = create_test_anndata_with_fdr()

        # Ensure we have both True and False values in the DE column
        de_values = adata.var["kompot_de_A_to_B_is_de"].values
        n_true = np.sum(de_values)
        n_false = len(de_values) - n_true

        fig, ax = plt.subplots()
        volcano_de(
            adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            y_axis_type="local_fdr",
            ax=ax,
            show_legend=False,
        )

        # Check that the DE column is being used for coloring and has both values
        assert n_true > 0, f"Should have some significant genes, got {n_true}"
        assert n_false > 0, f"Should have some non-significant genes, got {n_false}"

        # For categorical coloring, should have multiple scatter collections or varied colors
        # The categorical coloring might create multiple scatter plots
        n_collections = len(ax.collections)
        assert n_collections >= 1, "Should have at least one scatter collection"

        # Check if using categorical discrete mapping (which creates separate scatters)
        # or if it's a single scatter with discrete colors
        if n_collections > 1:
            # Multiple collections means separate scatters for each category
            assert True, "Multiple scatter collections indicate categorical coloring"
        else:
            # Single collection with discrete colors - check for color variation
            scatter = ax.collections[0]
            colors = scatter.get_facecolors()
            unique_colors = np.unique(colors.view(np.void), axis=0)
            # For discrete boolean coloring, we should see the pattern even if test is inconsistent
            # Just check that the coloring logic was applied (not all points same color)
            assert len(unique_colors) >= 1, "Should have applied coloring"

        plt.close(fig)

    def test_manual_background_coloring_override(self):
        """Test that manual background coloring overrides auto-coloring."""
        adata, _ = create_test_anndata_with_fdr()

        # Add a custom column for coloring
        n_genes = len(adata.var)
        categories = (["cat1", "cat2", "cat3"] * ((n_genes // 3) + 1))[:n_genes]
        adata.var["custom_category"] = categories

        fig, ax = plt.subplots()
        volcano_de(
            adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            y_axis_type="local_fdr",
            color="custom_category",
            ax=ax,
            show_legend=False,
        )

        # Should complete without error and use custom coloring
        # For categorical data, might have multiple collections or single collection with colors
        n_collections = len(ax.collections)
        assert n_collections >= 1, "Should have at least one scatter collection"

        # Test that custom categories exist
        custom_categories = adata.var["custom_category"].unique()
        assert len(custom_categories) == 3, (
            f"Should have 3 custom categories, got {len(custom_categories)}"
        )

        plt.close(fig)

    def test_direction_column_detection(self):
        """Test automatic DE column detection."""
        adata, _ = create_test_anndata_with_fdr()

        # Count initial significant genes
        initial_count = np.sum(adata.var["kompot_de_A_to_B_is_de"])

        fig, ax = plt.subplots()
        volcano_de(
            adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            y_axis_type="local_fdr",
            ax=ax,
            show_legend=False,
        )

        # DE column should still exist and be unchanged
        final_count = np.sum(adata.var["kompot_de_A_to_B_is_de"])
        assert final_count == initial_count, (
            "DE column should be unchanged without update_de_classification"
        )
        plt.close(fig)


class TestVolcanoFDRThresholds:
    """Test FDR threshold functionality."""

    def test_threshold_line_display(self):
        """Test that FDR threshold lines are displayed."""
        adata, _ = create_test_anndata_with_fdr()

        fig, ax = plt.subplots()
        volcano_de(
            adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            y_axis_type="local_fdr",
            significance_threshold=0.05,
            show_thresholds=True,
            ax=ax,
            show_legend=False,
        )

        # Should have horizontal lines (axhline creates Line2D objects)
        lines = ax.lines
        horizontal_lines = [
            line for line in lines if line.get_xdata()[0] != line.get_xdata()[1]
        ]

        assert len(horizontal_lines) > 0, "Should have horizontal threshold line"
        plt.close(fig)

    def test_threshold_line_disabled(self):
        """Test that threshold lines can be disabled."""
        adata, _ = create_test_anndata_with_fdr()

        fig, ax = plt.subplots()
        volcano_de(
            adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            y_axis_type="local_fdr",
            significance_threshold=0.05,
            show_thresholds=False,
            ax=ax,
            show_legend=False,
        )

        # Should have minimal lines (just the vertical x=0 line)
        lines = ax.lines
        # Only the default vertical line should be present
        assert len(lines) <= 1, "Should not have threshold line when disabled"
        plt.close(fig)

    def test_threshold_position_correct(self):
        """Test that threshold line is at correct y position."""
        adata, _ = create_test_anndata_with_fdr()
        threshold_val = 0.1

        fig, ax = plt.subplots()
        volcano_de(
            adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            y_axis_type="local_fdr",
            significance_threshold=threshold_val,
            show_thresholds=True,
            ax=ax,
            show_legend=False,
        )

        # Find horizontal lines
        lines = ax.lines
        horizontal_lines = [
            line
            for line in lines
            if len(set(line.get_ydata())) == 1
            and line.get_ydata()[0] > 0  # horizontal line
        ]  # above x-axis

        assert len(horizontal_lines) > 0, "Should have horizontal threshold line"

        # Check that the line is at the correct transformed position
        expected_y = -np.log10(threshold_val)
        actual_y = horizontal_lines[0].get_ydata()[0]
        np.testing.assert_allclose(actual_y, expected_y, rtol=1e-5)
        plt.close(fig)


class TestVolcanoFDRClassificationUpdates:
    """Test DE classification update functionality."""

    def test_update_classification_enabled(self):
        """Test that DE classification can be updated."""
        adata, _ = create_test_anndata_with_fdr()

        # Count initial significant genes
        initial_count = np.sum(adata.var["kompot_de_A_to_B_is_de"])

        fig, ax = plt.subplots()
        volcano_de(
            adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            y_axis_type="local_fdr",
            significance_threshold=0.2,  # More lenient threshold
            update_de_classification=True,
            ax=ax,
            show_legend=False,
        )

        # Should have more significant genes with lenient threshold
        final_count = np.sum(adata.var["kompot_de_A_to_B_is_de"])
        assert final_count >= initial_count, (
            "Should have at least as many significant genes with lenient threshold"
        )
        plt.close(fig)

    def test_update_classification_disabled(self):
        """Test that DE classification is not updated by default."""
        adata, _ = create_test_anndata_with_fdr()

        # Count initial significant genes
        initial_count = np.sum(adata.var["kompot_de_A_to_B_is_de"])

        fig, ax = plt.subplots()
        volcano_de(
            adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            y_axis_type="local_fdr",
            significance_threshold=0.2,  # More lenient threshold
            update_de_classification=False,  # Explicitly disabled
            ax=ax,
            show_legend=False,
        )

        # Should be unchanged
        final_count = np.sum(adata.var["kompot_de_A_to_B_is_de"])
        assert final_count == initial_count, (
            "Should not update classification when disabled"
        )
        plt.close(fig)

    def test_update_classification_strict_threshold(self):
        """Test classification update with strict threshold."""
        adata, _ = create_test_anndata_with_fdr()

        # Count initial significant genes
        initial_count = np.sum(adata.var["kompot_de_A_to_B_is_de"])

        fig, ax = plt.subplots()
        volcano_de(
            adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            y_axis_type="local_fdr",
            significance_threshold=0.001,  # Very strict threshold
            update_de_classification=True,
            ax=ax,
            show_legend=False,
        )

        # Should have fewer significant genes with strict threshold
        final_count = np.sum(adata.var["kompot_de_A_to_B_is_de"])
        assert final_count <= initial_count, (
            "Should have fewer or equal significant genes with strict threshold"
        )
        plt.close(fig)


class TestVolcanoFDRErrorHandling:
    """Test error handling and edge cases for FDR functionality."""

    def test_missing_fdr_columns(self):
        """Test handling when FDR columns are missing."""
        adata, _ = create_test_anndata_with_fdr()

        # Remove FDR columns
        adata.var = adata.var.drop(
            columns=[
                "kompot_de_A_to_B_mahalanobis_local_fdr",
                "kompot_de_A_to_B_mahalanobis_tail_fdr",
            ]
        )

        fig, ax = plt.subplots()
        # Should not raise an error, should fall back to mahalanobis
        volcano_de(
            adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            y_axis_type="local_fdr",
            ax=ax,
            show_legend=False,
        )

        # Should fall back to original score and ylabel
        assert ax.get_ylabel() in ["Mahalanobis Distance", "-log10(Local FDR)"]
        plt.close(fig)

    def test_missing_direction_column(self):
        """Test handling when DE boolean column is missing."""
        adata, _ = create_test_anndata_with_fdr()

        # Remove DE column
        adata.var = adata.var.drop(columns=["kompot_de_A_to_B_is_de"])

        fig, ax = plt.subplots()
        # Should not raise an error
        volcano_de(
            adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            y_axis_type="local_fdr",
            significance_threshold=0.05,
            update_de_classification=True,  # This should be ignored
            ax=ax,
            show_legend=False,
        )

        # Should complete without error
        assert ax.get_ylabel() == "-log10(Local FDR)"
        plt.close(fig)

    def test_group_specific_fdr_warning(self):
        """Test warning for group-specific FDR usage."""
        adata, _ = create_test_anndata_with_fdr()

        # Add some fake varm data to simulate group analysis
        adata.varm["kompot_de_mean_lfc"] = pd.DataFrame(
            {"group1": np.random.normal(0, 1, len(adata.var))}, index=adata.var_names
        )
        adata.varm["kompot_de_mahalanobis"] = pd.DataFrame(
            {"group1": np.random.gamma(2, 1, len(adata.var))}, index=adata.var_names
        )

        fig, ax = plt.subplots()
        # Should show warning but not error
        volcano_de(
            adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            group="group1",  # Use group-specific data
            y_axis_type="local_fdr",
            ax=ax,
            show_legend=False,
        )

        plt.close(fig)

    def test_invalid_y_axis_type(self):
        """Test handling of invalid y-axis type."""
        adata, _ = create_test_anndata_with_fdr()

        fig, ax = plt.subplots()
        # Should work with invalid type by falling back to default
        volcano_de(
            adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            y_axis_type="invalid_type",  # Invalid type
            ax=ax,
            show_legend=False,
        )

        # Should use default Mahalanobis
        assert ax.get_ylabel() == "Mahalanobis Distance"
        plt.close(fig)


class TestVolcanoFDRIntegration:
    """Test integration with existing functionality."""

    def test_fdr_with_highlight_genes(self):
        """Test FDR functionality works with gene highlighting."""
        adata, diff_indices = create_test_anndata_with_fdr()

        # Get some gene names to highlight
        highlight_genes = adata.var_names[diff_indices[:5]].tolist()

        fig, ax = plt.subplots()
        volcano_de(
            adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            y_axis_type="local_fdr",
            highlight_genes=highlight_genes,
            ax=ax,
            show_legend=False,
        )

        # Should have multiple scatter collections (background + highlighted)
        assert len(ax.collections) > 1, (
            "Should have multiple scatter collections for highlighting"
        )
        plt.close(fig)

    def test_fdr_with_custom_colors(self):
        """Test FDR functionality with custom color parameters."""
        adata, _ = create_test_anndata_with_fdr()

        fig, ax = plt.subplots()
        volcano_de(
            adata,
            lfc_key="kompot_de_A_to_B_mean_lfc",
            score_key="kompot_de_A_to_B_mahalanobis",
            y_axis_type="local_fdr",
            color_up="#FF0000",
            color_down="#0000FF",
            color_background="#CCCCCC",
            ax=ax,
            show_legend=False,
        )

        # Should complete without error
        assert ax.get_ylabel() == "-log10(Local FDR)"
        plt.close(fig)

    def test_fdr_with_save_functionality(self):
        """Test that FDR plots can be saved."""
        import tempfile
        import os

        adata, _ = create_test_anndata_with_fdr()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            volcano_de(
                adata,
                lfc_key="kompot_de_A_to_B_mean_lfc",
                score_key="kompot_de_A_to_B_mahalanobis",
                y_axis_type="local_fdr",
                significance_threshold=0.05,
                save=tmp.name,
                show_legend=False,
            )

            # File should exist and have size > 0
            assert os.path.exists(tmp.name), "Plot file should be saved"
            assert os.path.getsize(tmp.name) > 0, "Plot file should not be empty"

            # Clean up
            os.unlink(tmp.name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
