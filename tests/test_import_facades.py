"""Test import facades to ensure they're properly exposed and used."""


def test_anndata_utils_facade_used():
    """Test that anndata.utils facade actually imports and can use functions."""
    # Import through facade
    from kompot.anndata.utils import (
        jsonable_encoder,
        _sanitize_name,
        RunInfo,
    )

    # Actually use the functions/classes to ensure they're covered
    result = _sanitize_name("test name")
    assert isinstance(result, str)

    # Ensure jsonable_encoder is callable
    assert callable(jsonable_encoder)

    # Ensure RunInfo class is accessible
    assert RunInfo is not None


def test_plot_heatmap_facade_used():
    """Test that plot.heatmap facade actually imports functions."""
    # Import through facade
    from kompot.plot.heatmap import heatmap, direction_barplot

    # Ensure functions are callable
    assert callable(heatmap)
    assert callable(direction_barplot)


def test_plot_volcano_facade_used():
    """Test that plot.volcano facade actually imports functions."""
    # Import through facade
    from kompot.plot.volcano import volcano_de, volcano_da, multi_volcano_da

    # Ensure functions are callable
    assert callable(volcano_de)
    assert callable(volcano_da)
    assert callable(multi_volcano_da)
