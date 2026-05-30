"""Tests for kompot.plot.lollipop."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Synthetic enrichment frames
# ---------------------------------------------------------------------------
def _stringdb_frame(n: int = 8, seed: int = 0) -> pd.DataFrame:
    """A frame shaped like StringDBReport.get_functional_enrichment output:
    ``term``, ``description``, ``signal``, ``strength``, ``fdr``,
    ``number_of_genes`` — sorted by ``signal`` descending."""
    rng = np.random.default_rng(seed)
    fdr = np.sort(rng.uniform(1e-9, 4e-2, size=n))
    df = pd.DataFrame(
        {
            "term": [f"GO:{i:07d}" for i in range(n)],
            "description": [f"biological process number {i}" for i in range(n)],
            "signal": np.sort(rng.uniform(0.5, 3.5, size=n))[::-1],
            "strength": rng.uniform(0.3, 1.5, size=n),
            "fdr": fdr,
            "number_of_genes": rng.integers(3, 60, size=n),
        }
    )
    return df.sort_values("signal", ascending=False).reset_index(drop=True)


def _generic_frame(n: int = 6, seed: int = 1) -> pd.DataFrame:
    """A gseapy/enrichr-style frame: ``Term``, ``Overlap`` ("k/K" string),
    ``Adjusted P-value``, ``Combined Score`` — different header names so
    autodetection + the ``Overlap`` string parser are exercised."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "Term": [f"pathway {i}" for i in range(n)],
            "Overlap": [f"{k}/300" for k in rng.integers(4, 40, size=n)],
            "Adjusted P-value": rng.uniform(1e-7, 3e-2, size=n),
            "Combined Score": rng.uniform(5, 120, size=n),
        }
    )


# ---------------------------------------------------------------------------
# Import / export
# ---------------------------------------------------------------------------
def test_lollipop_import_and_exported():
    import kompot.plot as kp

    assert hasattr(kp, "lollipop")
    assert "lollipop" in kp.__all__


# ---------------------------------------------------------------------------
# StringDBReport-DataFrame path
# ---------------------------------------------------------------------------
def test_lollipop_stringdb_frame_standalone():
    from kompot.plot import lollipop

    df = _stringdb_frame()
    fig = lollipop(df, n_terms=5, return_fig=True)
    assert isinstance(fig, plt.Figure)
    ax = fig.axes[0]
    # 5 dots in the scatter collection
    assert ax.collections[-1].get_offsets().shape[0] == 5
    # x label is the -log10(FDR) default
    assert "log" in ax.get_xlabel().lower()
    plt.close(fig)


def test_lollipop_default_returns_none():
    from kompot.plot import lollipop

    out = lollipop(_stringdb_frame(), n_terms=4)
    assert out is None
    plt.close("all")


def test_lollipop_top_n_sorted_by_significance():
    from kompot.plot import lollipop

    df = _stringdb_frame(n=10)
    fig = lollipop(df, n_terms=3, return_fig=True)
    ax = fig.axes[0]
    labels = [t.get_text() for t in ax.get_yticklabels()]
    # the three smallest-FDR terms, most significant on top
    expected = list(
        df.sort_values("fdr").head(3)["description"]
    )
    # labels may be soft-wrapped (newline) — compare on the un-wrapped text
    got = [lbl.replace("\n", " ") for lbl in labels]
    assert got == expected
    plt.close(fig)


# ---------------------------------------------------------------------------
# Generic-format path + autodetection + Overlap parsing
# ---------------------------------------------------------------------------
def test_lollipop_generic_frame_autodetect():
    from kompot.plot import lollipop

    df = _generic_frame()
    fig = lollipop(df, n_terms=5, return_fig=True)
    assert isinstance(fig, plt.Figure)
    ax = fig.axes[0]
    assert ax.collections[-1].get_offsets().shape[0] == 5
    plt.close(fig)


def test_lollipop_explicit_column_mapping_and_score_metric():
    from kompot.plot import lollipop

    df = _generic_frame()
    fig = lollipop(
        df,
        x_metric="score",
        term_col="Term",
        score_col="Combined Score",
        count_col="Overlap",
        fdr_col="Adjusted P-value",
        n_terms=4,
        return_fig=True,
    )
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Combined Score"
    # score-metric → sorted by score descending; top dot has the max score
    xs = ax.collections[-1].get_offsets()[:, 0]
    assert xs[0] == pytest.approx(df["Combined Score"].max())
    plt.close(fig)


def test_lollipop_overlap_string_drives_dot_size():
    from kompot.plot import lollipop
    from kompot.plot.lollipop import _coerce_counts

    counts = _coerce_counts(pd.Series(["12/300", "4/300", "40/300"]))
    assert list(counts) == [12.0, 4.0, 40.0]


# ---------------------------------------------------------------------------
# StringDBReport instance path (mocked — no network)
# ---------------------------------------------------------------------------
def test_lollipop_accepts_stringdb_report_instance():
    from kompot.plot import lollipop

    frame = _stringdb_frame()

    class _FakeReport:
        def get_functional_enrichment(self, category="Process", fdr_threshold=0.05):
            assert category == "Process"
            return frame

    fig = lollipop(_FakeReport(), n_terms=4, return_fig=True)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_lollipop_stringdb_report_empty_raises():
    from kompot.plot import lollipop

    class _EmptyReport:
        def get_functional_enrichment(self, category="Process", fdr_threshold=0.05):
            return None

    with pytest.raises(ValueError, match="no terms"):
        lollipop(_EmptyReport())
    plt.close("all")


# ---------------------------------------------------------------------------
# Embedding into a provided ax
# ---------------------------------------------------------------------------
def test_lollipop_embeds_into_provided_ax():
    from kompot.plot import lollipop

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    figs_before = list(plt.get_fignums())
    out = lollipop(_stringdb_frame(), ax=axes[0], n_terms=4)
    assert out is None
    # no new figure created
    assert plt.get_fignums() == figs_before
    assert len(axes[0].collections) >= 1
    plt.close(fig)


def test_lollipop_embedded_return_fig_returns_ax():
    from kompot.plot import lollipop

    fig, ax = plt.subplots()
    out = lollipop(_stringdb_frame(), ax=ax, n_terms=3, return_fig=True)
    assert out is ax
    plt.close(fig)


# ---------------------------------------------------------------------------
# Toggles: FDR guide, legend, annotations
# ---------------------------------------------------------------------------
def test_lollipop_fdr_guide_toggle():
    from kompot.plot import lollipop

    df = _stringdb_frame()
    fig_on = lollipop(df, fdr_line=0.05, return_fig=True)
    n_on = len(fig_on.axes[0].lines)
    plt.close(fig_on)

    fig_off = lollipop(df, fdr_line=None, return_fig=True)
    n_off = len(fig_off.axes[0].lines)
    plt.close(fig_off)

    # one fewer line (the guide) when disabled
    assert n_on == n_off + 1


def test_lollipop_legend_toggle():
    from kompot.plot import lollipop

    fig = lollipop(_stringdb_frame(), legend=False, return_fig=True)
    assert fig.legends == [] and fig.axes[0].get_legend() is None
    plt.close(fig)


def test_lollipop_annotate_toggle():
    from kompot.plot import lollipop

    df = _stringdb_frame(n=4)
    fig_on = lollipop(df, annotate=True, return_fig=True)
    ann_on = [c for c in fig_on.axes[0].texts if "FDR" in c.get_text()]
    assert len(ann_on) == 4
    plt.close(fig_on)

    fig_off = lollipop(df, annotate=False, return_fig=True)
    ann_off = [c for c in fig_off.axes[0].texts if "FDR" in c.get_text()]
    assert ann_off == []
    plt.close(fig_off)


def test_lollipop_custom_annotate_fmt():
    from kompot.plot import lollipop

    fig = lollipop(
        _stringdb_frame(n=3),
        annotate_fmt="{count} genes",
        return_fig=True,
    )
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert any("genes" in t for t in texts)
    plt.close(fig)


# ---------------------------------------------------------------------------
# cmap coloring path
# ---------------------------------------------------------------------------
def test_lollipop_cmap_colors_by_score():
    from kompot.plot import lollipop

    df = _stringdb_frame()
    fig = lollipop(df, cmap="viridis", color_by="signal", return_fig=True)
    # a colorbar axis is added on the standalone figure
    assert len(fig.axes) >= 2
    plt.close(fig)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------
def test_lollipop_missing_term_column_raises():
    from kompot.plot import lollipop

    df = pd.DataFrame({"fdr": [0.01, 0.02], "number_of_genes": [5, 8]})
    with pytest.raises(KeyError, match="term"):
        lollipop(df)
    plt.close("all")


def test_lollipop_neg_log10_without_fdr_raises():
    from kompot.plot import lollipop

    df = pd.DataFrame(
        {"description": ["a", "b"], "signal": [1.0, 2.0]}
    )
    with pytest.raises(KeyError, match="FDR"):
        lollipop(df, x_metric="neg_log10_fdr")
    plt.close("all")


def test_lollipop_score_metric_without_score_raises():
    from kompot.plot import lollipop

    df = pd.DataFrame(
        {"description": ["a", "b"], "fdr": [0.01, 0.02]}
    )
    with pytest.raises(KeyError, match="score"):
        lollipop(df, x_metric="score")
    plt.close("all")


def test_lollipop_empty_frame_raises():
    from kompot.plot import lollipop

    with pytest.raises(ValueError, match="empty"):
        lollipop(pd.DataFrame({"description": [], "fdr": []}))
    plt.close("all")


def test_lollipop_explicit_bad_column_raises():
    from kompot.plot import lollipop

    df = _stringdb_frame()
    with pytest.raises(KeyError, match="not found"):
        lollipop(df, term_col="NoSuchColumn")
    plt.close("all")


def test_lollipop_literal_column_x_metric():
    from kompot.plot import lollipop

    df = _stringdb_frame()
    fig = lollipop(df, x_metric="strength", n_terms=4, return_fig=True)
    assert fig.axes[0].get_xlabel() == "strength"
    plt.close(fig)
