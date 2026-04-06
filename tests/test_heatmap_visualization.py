"""Tests for plot/heatmap/visualization.py to improve coverage."""

import pytest
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from kompot.plot.heatmap.visualization import (
    _setup_colormap_normalization,
    _draw_diagonal_split_cell,
    _draw_split_dot_cell,
    _draw_fold_change_cell,
)


class TestSetupColormapNormalization:
    """Test _setup_colormap_normalization function."""

    def test_setup_colormap_with_center(self):
        """Test normalization with centered colormap."""
        data = np.array([-2, -1, 0, 1, 2])
        norm, cmap_obj, vmin, vmax = _setup_colormap_normalization(
            data, center=0, vmin=None, vmax=None, cmap="RdBu_r"
        )

        assert isinstance(norm, mcolors.TwoSlopeNorm)
        assert norm.vcenter == 0
        # vmin and vmax should be equidistant from center
        assert vmin == -2
        assert vmax == 2

    def test_setup_colormap_without_center(self):
        """Test normalization without centering."""
        data = np.array([1, 2, 3, 4, 5])
        norm, cmap_obj, vmin, vmax = _setup_colormap_normalization(
            data, center=None, vmin=None, vmax=None, cmap="viridis"
        )

        assert isinstance(norm, mcolors.Normalize)
        assert vmin == 1
        assert vmax == 5

    def test_setup_colormap_with_explicit_vmin_vmax(self):
        """Test normalization with explicit vmin/vmax."""
        data = np.array([1, 2, 3])
        norm, cmap_obj, vmin, vmax = _setup_colormap_normalization(
            data, center=None, vmin=0, vmax=10, cmap="viridis"
        )

        assert vmin == 0
        assert vmax == 10

    def test_setup_colormap_centered_with_explicit_bounds(self):
        """Test centered normalization with explicit bounds."""
        data = np.array([-1, 0, 1])
        norm, cmap_obj, vmin, vmax = _setup_colormap_normalization(
            data, center=0, vmin=-5, vmax=3, cmap="RdBu_r"
        )

        # Should make bounds equidistant from center
        assert vmin == -5
        assert vmax == 5  # max distance is 5

    def test_setup_colormap_string_cmap(self):
        """Test with string colormap name."""
        data = np.array([1, 2, 3])
        norm, cmap_obj, vmin, vmax = _setup_colormap_normalization(
            data, center=None, vmin=None, vmax=None, cmap="plasma"
        )

        # Should return a colormap object
        assert cmap_obj is not None
        assert hasattr(cmap_obj, "__call__")  # Colormap is callable

    def test_setup_colormap_object_cmap(self):
        """Test with colormap object instead of string."""
        data = np.array([1, 2, 3])
        cmap_input = plt.cm.get_cmap("viridis")
        norm, cmap_obj, vmin, vmax = _setup_colormap_normalization(
            data, center=None, vmin=None, vmax=None, cmap=cmap_input
        )

        assert cmap_obj is cmap_input

    def test_setup_colormap_with_nan_values(self):
        """Test normalization with NaN values in data."""
        data = np.array([1, np.nan, 3, 4, np.nan])
        norm, cmap_obj, vmin, vmax = _setup_colormap_normalization(
            data, center=None, vmin=None, vmax=None, cmap="viridis"
        )

        # Should use nanmin/nanmax
        assert vmin == 1.0
        assert vmax == 4.0


class TestDrawDiagonalSplitCell:
    """Test _draw_diagonal_split_cell function."""

    def setup_method(self):
        """Create a figure and axes for testing."""
        self.fig, self.ax = plt.subplots()

    def teardown_method(self):
        """Close the figure after each test."""
        plt.close(self.fig)

    def test_draw_diagonal_split_cell_basic(self):
        """Test basic diagonal split cell drawing."""
        _draw_diagonal_split_cell(
            self.ax,
            x=0,
            y=0,
            w=1,
            h=1,
            val1=0.5,
            val2=0.8,
            cmap="viridis",
            vmin=0,
            vmax=1,
        )

        # Check that patches were added
        assert len(self.ax.patches) == 2  # Two triangles

    def test_draw_diagonal_split_cell_with_nan(self):
        """Test drawing with NaN values."""
        _draw_diagonal_split_cell(
            self.ax,
            x=0,
            y=0,
            w=1,
            h=1,
            val1=np.nan,
            val2=0.5,
            cmap="viridis",
            vmin=0,
            vmax=1,
        )

        assert len(self.ax.patches) == 2

    def test_draw_diagonal_split_cell_with_string_values(self):
        """Test drawing with string values that can be converted."""
        _draw_diagonal_split_cell(
            self.ax,
            x=0,
            y=0,
            w=1,
            h=1,
            val1="0.5",
            val2="0.8",
            cmap="viridis",
            vmin=0,
            vmax=1,
        )

        assert len(self.ax.patches) == 2

    def test_draw_diagonal_split_cell_with_invalid_string(self):
        """Test drawing with invalid string values."""
        _draw_diagonal_split_cell(
            self.ax,
            x=0,
            y=0,
            w=1,
            h=1,
            val1="not_a_number",
            val2="0.8",
            cmap="viridis",
            vmin=0,
            vmax=1,
        )

        # Should handle invalid strings as NaN
        assert len(self.ax.patches) == 2

    def test_draw_diagonal_split_cell_with_draw_values(self):
        """Test drawing with value annotations."""
        _draw_diagonal_split_cell(
            self.ax,
            x=0,
            y=0,
            w=1,
            h=1,
            val1=0.5,
            val2=0.8,
            cmap="viridis",
            vmin=0,
            vmax=1,
            draw_values=True,
        )

        # Check for text annotations
        assert len(self.ax.texts) == 2  # Two text elements

    def test_draw_diagonal_split_cell_with_custom_norm(self):
        """Test drawing with custom normalization."""
        norm = mcolors.TwoSlopeNorm(vcenter=0.5, vmin=0, vmax=1)
        _draw_diagonal_split_cell(
            self.ax,
            x=0,
            y=0,
            w=1,
            h=1,
            val1=0.3,
            val2=0.7,
            cmap="RdBu_r",
            vmin=0,
            vmax=1,
            norm=norm,
        )

        assert len(self.ax.patches) == 2

    def test_draw_diagonal_split_cell_with_alpha(self):
        """Test drawing with custom alpha value."""
        _draw_diagonal_split_cell(
            self.ax,
            x=0,
            y=0,
            w=1,
            h=1,
            val1=0.5,
            val2=0.8,
            cmap="viridis",
            vmin=0,
            vmax=1,
            alpha=0.5,
        )

        assert len(self.ax.patches) == 2

    def test_draw_diagonal_split_cell_with_edge_params(self):
        """Test drawing with edge color and linewidth."""
        _draw_diagonal_split_cell(
            self.ax,
            x=0,
            y=0,
            w=1,
            h=1,
            val1=0.5,
            val2=0.8,
            cmap="viridis",
            vmin=0,
            vmax=1,
            edgecolor="black",
            linewidth=2,
        )

        assert len(self.ax.patches) == 2

    def test_draw_diagonal_split_cell_invalid_ax(self):
        """Test that invalid axes raises error."""
        with pytest.raises(ValueError, match="valid matplotlib Axes"):
            _draw_diagonal_split_cell(
                None,
                x=0,
                y=0,
                w=1,
                h=1,
                val1=0.5,
                val2=0.8,
                cmap="viridis",
                vmin=0,
                vmax=1,
            )

    def test_draw_diagonal_split_cell_colormap_object(self):
        """Test with colormap object instead of string."""
        cmap_obj = plt.cm.get_cmap("plasma")
        _draw_diagonal_split_cell(
            self.ax,
            x=0,
            y=0,
            w=1,
            h=1,
            val1=0.5,
            val2=0.8,
            cmap=cmap_obj,
            vmin=0,
            vmax=1,
        )

        assert len(self.ax.patches) == 2


class TestDrawSplitDotCell:
    """Test _draw_split_dot_cell function."""

    def setup_method(self):
        """Create a figure and axes for testing."""
        self.fig, self.ax = plt.subplots()

    def teardown_method(self):
        """Close the figure after each test."""
        plt.close(self.fig)

    def test_draw_split_dot_cell_basic(self):
        """Test basic split dot cell drawing."""
        _draw_split_dot_cell(
            self.ax,
            x=0,
            y=0,
            w=1,
            h=1,
            val1=0.5,
            val2=0.8,
            cmap="viridis",
            vmin=0,
            vmax=1,
            cell_count1=10,
            cell_count2=20,
        )

        # Check that wedge patches were added
        assert len(self.ax.patches) == 2  # Two wedges

    def test_draw_split_dot_cell_with_nan(self):
        """Test drawing with NaN values."""
        _draw_split_dot_cell(
            self.ax,
            x=0,
            y=0,
            w=1,
            h=1,
            val1=np.nan,
            val2=0.5,
            cmap="viridis",
            vmin=0,
            vmax=1,
            cell_count1=10,
            cell_count2=20,
        )

        assert len(self.ax.patches) == 2

    def test_draw_split_dot_cell_zero_counts(self):
        """Test drawing with zero cell counts."""
        _draw_split_dot_cell(
            self.ax,
            x=0,
            y=0,
            w=1,
            h=1,
            val1=0.5,
            val2=0.8,
            cmap="viridis",
            vmin=0,
            vmax=1,
            cell_count1=0,
            cell_count2=0,
        )

        # Should use default size
        assert len(self.ax.patches) == 2

    def test_draw_split_dot_cell_none_counts(self):
        """Test drawing with None cell counts."""
        _draw_split_dot_cell(
            self.ax,
            x=0,
            y=0,
            w=1,
            h=1,
            val1=0.5,
            val2=0.8,
            cmap="viridis",
            vmin=0,
            vmax=1,
            cell_count1=None,
            cell_count2=None,
        )

        assert len(self.ax.patches) == 2

    def test_draw_split_dot_cell_with_global_max(self):
        """Test drawing with global max count for scaling."""
        _draw_split_dot_cell(
            self.ax,
            x=0,
            y=0,
            w=1,
            h=1,
            val1=0.5,
            val2=0.8,
            cmap="viridis",
            vmin=0,
            vmax=1,
            cell_count1=10,
            cell_count2=20,
            global_max_count=50,
        )

        assert len(self.ax.patches) == 2

    def test_draw_split_dot_cell_string_counts(self):
        """Test drawing with string cell counts."""
        _draw_split_dot_cell(
            self.ax,
            x=0,
            y=0,
            w=1,
            h=1,
            val1=0.5,
            val2=0.8,
            cmap="viridis",
            vmin=0,
            vmax=1,
            cell_count1="10",
            cell_count2="20",
        )

        assert len(self.ax.patches) == 2

    def test_draw_split_dot_cell_invalid_string_counts(self):
        """Test drawing with invalid string counts."""
        _draw_split_dot_cell(
            self.ax,
            x=0,
            y=0,
            w=1,
            h=1,
            val1=0.5,
            val2=0.8,
            cmap="viridis",
            vmin=0,
            vmax=1,
            cell_count1="invalid",
            cell_count2="20",
        )

        assert len(self.ax.patches) == 2

    def test_draw_split_dot_cell_string_global_max(self):
        """Test with string global_max_count."""
        _draw_split_dot_cell(
            self.ax,
            x=0,
            y=0,
            w=1,
            h=1,
            val1=0.5,
            val2=0.8,
            cmap="viridis",
            vmin=0,
            vmax=1,
            cell_count1=10,
            cell_count2=20,
            global_max_count="50",
        )

        assert len(self.ax.patches) == 2

    def test_draw_split_dot_cell_invalid_global_max(self):
        """Test with invalid global_max_count string."""
        _draw_split_dot_cell(
            self.ax,
            x=0,
            y=0,
            w=1,
            h=1,
            val1=0.5,
            val2=0.8,
            cmap="viridis",
            vmin=0,
            vmax=1,
            cell_count1=10,
            cell_count2=20,
            global_max_count="invalid",
        )

        assert len(self.ax.patches) == 2

    def test_draw_split_dot_cell_with_draw_values(self):
        """Test drawing with value annotations."""
        _draw_split_dot_cell(
            self.ax,
            x=0,
            y=0,
            w=1,
            h=1,
            val1=0.5,
            val2=0.8,
            cmap="viridis",
            vmin=0,
            vmax=1,
            cell_count1=10,
            cell_count2=20,
            draw_values=True,
        )

        assert len(self.ax.texts) == 2  # Two text elements

    def test_draw_split_dot_cell_with_custom_norm(self):
        """Test drawing with custom normalization."""
        norm = mcolors.Normalize(vmin=0, vmax=1)
        _draw_split_dot_cell(
            self.ax,
            x=0,
            y=0,
            w=1,
            h=1,
            val1=0.5,
            val2=0.8,
            cmap="viridis",
            vmin=0,
            vmax=1,
            cell_count1=10,
            cell_count2=20,
            norm=norm,
        )

        assert len(self.ax.patches) == 2

    def test_draw_split_dot_cell_custom_max_size_factor(self):
        """Test with custom max_size_factor."""
        _draw_split_dot_cell(
            self.ax,
            x=0,
            y=0,
            w=1,
            h=1,
            val1=0.5,
            val2=0.8,
            cmap="viridis",
            vmin=0,
            vmax=1,
            cell_count1=10,
            cell_count2=20,
            max_size_factor=0.5,
        )

        assert len(self.ax.patches) == 2

    def test_draw_split_dot_cell_string_values(self):
        """Test with string values."""
        _draw_split_dot_cell(
            self.ax,
            x=0,
            y=0,
            w=1,
            h=1,
            val1="0.5",
            val2="0.8",
            cmap="viridis",
            vmin=0,
            vmax=1,
            cell_count1=10,
            cell_count2=20,
        )

        assert len(self.ax.patches) == 2

    def test_draw_split_dot_cell_invalid_ax(self):
        """Test that invalid axes raises error."""
        with pytest.raises(ValueError, match="valid matplotlib Axes"):
            _draw_split_dot_cell(
                None,
                x=0,
                y=0,
                w=1,
                h=1,
                val1=0.5,
                val2=0.8,
                cmap="viridis",
                vmin=0,
                vmax=1,
            )

    def test_draw_split_dot_cell_colormap_object(self):
        """Test with colormap object instead of string."""
        cmap_obj = plt.cm.get_cmap("plasma")
        _draw_split_dot_cell(
            self.ax,
            x=0,
            y=0,
            w=1,
            h=1,
            val1=0.5,
            val2=0.8,
            cmap=cmap_obj,
            vmin=0,
            vmax=1,
            cell_count1=10,
            cell_count2=20,
        )

        assert len(self.ax.patches) == 2

    def test_draw_split_dot_cell_counts_exceed_global_max(self):
        """Test when counts exceed global_max_count."""
        _draw_split_dot_cell(
            self.ax,
            x=0,
            y=0,
            w=1,
            h=1,
            val1=0.5,
            val2=0.8,
            cmap="viridis",
            vmin=0,
            vmax=1,
            cell_count1=100,
            cell_count2=200,
            global_max_count=50,
        )

        # Should clip counts to global_max
        assert len(self.ax.patches) == 2


class TestDrawFoldChangeCell:
    """Test _draw_fold_change_cell function."""

    def setup_method(self):
        """Create a figure and axes for testing."""
        self.fig, self.ax = plt.subplots()

    def teardown_method(self):
        """Close the figure after each test."""
        plt.close(self.fig)

    def test_draw_fold_change_cell_basic(self):
        """Test basic fold change cell drawing."""
        _draw_fold_change_cell(
            self.ax, x=0, y=0, w=1, h=1, lfc=1.5, cmap="RdBu_r", vmin=-2, vmax=2
        )

        # Check that rectangle was added
        assert len(self.ax.patches) == 1

    def test_draw_fold_change_cell_with_nan(self):
        """Test drawing with NaN value."""
        _draw_fold_change_cell(
            self.ax, x=0, y=0, w=1, h=1, lfc=np.nan, cmap="RdBu_r", vmin=-2, vmax=2
        )

        # Should draw rectangle with gray color
        assert len(self.ax.patches) == 1

    def test_draw_fold_change_cell_string_value(self):
        """Test drawing with string lfc value."""
        _draw_fold_change_cell(
            self.ax, x=0, y=0, w=1, h=1, lfc="1.5", cmap="RdBu_r", vmin=-2, vmax=2
        )

        assert len(self.ax.patches) == 1

    def test_draw_fold_change_cell_invalid_string(self):
        """Test drawing with invalid string lfc."""
        _draw_fold_change_cell(
            self.ax,
            x=0,
            y=0,
            w=1,
            h=1,
            lfc="not_a_number",
            cmap="RdBu_r",
            vmin=-2,
            vmax=2,
        )

        # Should handle as NaN
        assert len(self.ax.patches) == 1

    def test_draw_fold_change_cell_with_draw_values(self):
        """Test drawing with value annotation."""
        _draw_fold_change_cell(
            self.ax,
            x=0,
            y=0,
            w=1,
            h=1,
            lfc=1.5,
            cmap="RdBu_r",
            vmin=-2,
            vmax=2,
            draw_values=True,
        )

        assert len(self.ax.texts) == 1  # One text element

    def test_draw_fold_change_cell_with_custom_norm(self):
        """Test drawing with custom normalization."""
        norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=-2, vmax=2)
        _draw_fold_change_cell(
            self.ax,
            x=0,
            y=0,
            w=1,
            h=1,
            lfc=1.5,
            cmap="RdBu_r",
            vmin=-2,
            vmax=2,
            norm=norm,
        )

        assert len(self.ax.patches) == 1

    def test_draw_fold_change_cell_with_alpha(self):
        """Test drawing with custom alpha value."""
        _draw_fold_change_cell(
            self.ax,
            x=0,
            y=0,
            w=1,
            h=1,
            lfc=1.5,
            cmap="RdBu_r",
            vmin=-2,
            vmax=2,
            alpha=0.5,
        )

        assert len(self.ax.patches) == 1

    def test_draw_fold_change_cell_with_edge_params(self):
        """Test drawing with edge color and linewidth."""
        _draw_fold_change_cell(
            self.ax,
            x=0,
            y=0,
            w=1,
            h=1,
            lfc=1.5,
            cmap="RdBu_r",
            vmin=-2,
            vmax=2,
            edgecolor="black",
            linewidth=2,
        )

        assert len(self.ax.patches) == 1

    def test_draw_fold_change_cell_invalid_ax(self):
        """Test that invalid axes raises error."""
        with pytest.raises(ValueError, match="valid matplotlib Axes"):
            _draw_fold_change_cell(
                None, x=0, y=0, w=1, h=1, lfc=1.5, cmap="RdBu_r", vmin=-2, vmax=2
            )

    def test_draw_fold_change_cell_colormap_object(self):
        """Test with colormap object instead of string."""
        cmap_obj = plt.cm.get_cmap("RdBu_r")
        _draw_fold_change_cell(
            self.ax, x=0, y=0, w=1, h=1, lfc=1.5, cmap=cmap_obj, vmin=-2, vmax=2
        )

        assert len(self.ax.patches) == 1

    def test_draw_fold_change_cell_negative_lfc(self):
        """Test with negative log fold change."""
        _draw_fold_change_cell(
            self.ax, x=0, y=0, w=1, h=1, lfc=-1.5, cmap="RdBu_r", vmin=-2, vmax=2
        )

        assert len(self.ax.patches) == 1

    def test_draw_fold_change_cell_zero_lfc(self):
        """Test with zero log fold change."""
        _draw_fold_change_cell(
            self.ax, x=0, y=0, w=1, h=1, lfc=0.0, cmap="RdBu_r", vmin=-2, vmax=2
        )

        assert len(self.ax.patches) == 1
