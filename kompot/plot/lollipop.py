"""Gene-set-enrichment lollipop plot.

Ax-embeddable lollipop for functional-enrichment tables: one row per
enriched term, a stem from the axis baseline to a dot whose x-position
encodes significance (``-log10(FDR)`` by default, or any score column)
and whose area encodes the gene count. A dashed ``FDR = 0.05`` guide and
an optional in-axes aesthetic key round out the figure-grade rendering.

The renderer was built for the kompot manuscript (Fig 3 panels G/L) and
is generalized here so it feeds any enrichment result with minimal fuss:

* a :class:`kompot.plot.StringDBReport` instance (its
  :meth:`~kompot.plot.StringDBReport.get_functional_enrichment` is
  called for you), **or**
* the ``signal``-sorted DataFrame that method already returns, **or**
* a generic enrichment table from another tool (gseapy / enrichr,
  GOATOOLS, clusterProfiler exports, …). Column-name mapping params plus
  autodetection bridge the schema differences — see :func:`lollipop`.

Like :func:`kompot.plot.dotplot`, this composes cleanly into an
externally-provided ``ax`` instead of building its own ``GridSpec``, so
it drops into a composite figure without fighting the surrounding layout.
"""

from __future__ import annotations

import logging
from typing import Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger("kompot")

try:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.colors import Normalize, to_rgb
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
except ImportError as e:  # pragma: no cover - exercised via import facade
    raise ImportError(
        "matplotlib is required for plotting: pip install matplotlib"
    ) from e


# ---------------------------------------------------------------------------
# Column autodetection
#
# Ordered candidate lists per logical field. The first column present in
# the frame wins. Names cover StringDB (kompot's StringDBReport), gseapy /
# enrichr, GOATOOLS, and clusterProfiler export conventions. Matching is
# case-insensitive on a normalized (stripped) header.
# ---------------------------------------------------------------------------
_TERM_CANDIDATES = (
    "description",  # StringDB, GOATOOLS go-term name
    "term",  # generic / StringDB term id
    "name",  # gseapy "Name"
    "go_name",
    "pathway",
    "term_name",
    "go_term",
    "id",  # last-ditch identifier
)
_SCORE_CANDIDATES = (
    "signal",  # StringDB balanced metric (its default sort key)
    "combined score",  # enrichr / gseapy
    "nes",  # GSEA normalized enrichment score
    "enrichment_score",
    "score",
    "strength",  # StringDB log10(obs/exp)
    "odds ratio",
)
_COUNT_CANDIDATES = (
    "number_of_genes",  # StringDB
    "count",  # clusterProfiler
    "intersection_size",  # g:Profiler
    "overlap",  # gseapy / enrichr ("k/K" string — numerator is taken)
    "n_genes",
    "gene_count",
    "genes",  # sometimes a list/str of members
    "study_count",  # GOATOOLS
)
_FDR_CANDIDATES = (
    "fdr",  # StringDB
    "adjusted p-value",  # enrichr / gseapy
    "p.adjust",  # clusterProfiler
    "padj",
    "p_fdr_bh",  # GOATOOLS
    "qvalue",
    "q_value",
    "q-value",
    "fdr_q-val",  # gseapy preranked
    "benjamini",
    "p_value",  # fall back to raw p if no adjusted column exists
    "pvalue",
    "p-value",
)


def _find_column(
    df: pd.DataFrame, candidates: Sequence[str], explicit: Optional[str]
) -> Optional[str]:
    """Resolve a logical field to an actual column name.

    ``explicit`` (if given) is honored verbatim and validated. Otherwise
    the first candidate present in ``df`` (case-insensitive) is returned,
    or ``None`` when nothing matches.
    """
    if explicit is not None:
        if explicit not in df.columns:
            raise KeyError(
                f"column '{explicit}' not found in enrichment frame "
                f"(available: {list(df.columns)})"
            )
        return explicit
    lower = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def _coerce_counts(values: pd.Series) -> np.ndarray:
    """Coerce a gene-count column to a float array.

    Handles the three shapes seen in the wild: a plain numeric column
    (StringDB ``number_of_genes``), a ``"k/K"`` overlap string (gseapy /
    enrichr ``Overlap`` — the numerator is the observed count), and a
    delimited gene list (the member count is its length). Unparseable
    entries become ``NaN``.
    """
    if pd.api.types.is_numeric_dtype(values):
        return values.to_numpy(dtype=float)

    out = np.full(len(values), np.nan, dtype=float)
    for i, v in enumerate(values.to_numpy()):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        s = str(v).strip()
        if not s:
            continue
        if "/" in s:  # "12/350" overlap → observed numerator
            head = s.split("/", 1)[0].strip()
            try:
                out[i] = float(head)
                continue
            except ValueError:
                pass
        try:  # plain number stored as text
            out[i] = float(s)
            continue
        except ValueError:
            pass
        # delimited member list → count the members
        for sep in (",", ";", " "):
            if sep in s:
                out[i] = float(len([t for t in s.split(sep) if t.strip()]))
                break
    return out


def _darken(color: str, factor: float = 0.55) -> Tuple[float, float, float]:
    """Return a darker companion of ``color`` for dot outlines."""
    r, g, b = to_rgb(color)
    return (r * factor, g * factor, b * factor)


def _wrap(text: str, width: int) -> str:
    """Soft-wrap a long term description across at most two lines."""
    text = str(text)
    if width <= 0 or len(text) <= width:
        return text
    cut = text.rfind(" ", 0, width + 1)
    if cut <= 0:
        cut = width
    head = text[:cut].rstrip()
    tail = text[cut:].lstrip()
    if len(tail) > width:
        tail = tail[: max(width - 1, 0)].rstrip() + "…"
    return f"{head}\n{tail}"


def _resolve_enrichment(
    data,
    *,
    category: str,
    fdr_threshold: float,
) -> pd.DataFrame:
    """Coerce the ``data`` argument into an enrichment DataFrame.

    Accepts a :class:`~kompot.plot.StringDBReport` (its
    ``get_functional_enrichment`` is called), a DataFrame (used as-is), or
    anything DataFrame-constructible (e.g. a list of record dicts).
    """
    if isinstance(data, pd.DataFrame):
        return data
    # Duck-type the StringDBReport so we don't hard-import it (keeps the
    # optional-dependency story intact) and so any work-alike with the
    # same method works too.
    if hasattr(data, "get_functional_enrichment"):
        enr = data.get_functional_enrichment(
            category=category, fdr_threshold=fdr_threshold
        )
        if enr is None or len(enr) == 0:
            raise ValueError(
                "StringDBReport.get_functional_enrichment returned no terms "
                f"for category '{category}' at FDR ≤ {fdr_threshold}. "
                "The StringDB service may be unavailable, or the gene set "
                "may carry no enrichment; pass a precomputed DataFrame to "
                "plot offline."
            )
        return enr
    try:
        return pd.DataFrame(data)
    except Exception as exc:  # pragma: no cover - defensive
        raise TypeError(
            "`data` must be a StringDBReport, a pandas DataFrame, or a "
            f"DataFrame-constructible object; got {type(data)!r}"
        ) from exc


def lollipop(
    data: Union["pd.DataFrame", object],
    *,
    n_terms: int = 12,
    term_col: Optional[str] = None,
    score_col: Optional[str] = None,
    count_col: Optional[str] = None,
    fdr_col: Optional[str] = None,
    x_metric: str = "neg_log10_fdr",
    sort_by: Optional[str] = "x",
    ascending: Optional[bool] = None,
    category: str = "Process",
    fdr_threshold: float = 0.05,
    color: str = "#d73027",
    edge_color: Optional[str] = None,
    cmap: Optional[str] = None,
    color_by: Optional[str] = None,
    stem_lw: float = 1.8,
    stem_alpha: float = 0.65,
    dot_min: float = 40.0,
    dot_max: float = 320.0,
    dot_scale: float = 22.0,
    dot_const: float = 80.0,
    fdr_line: Optional[float] = 0.05,
    annotate: bool = True,
    annotate_fmt: Optional[str] = None,
    legend: bool = True,
    legend_label: str = "gene set",
    label_width: int = 55,
    label_fontsize: float = 6.5,
    annotate_fontsize: float = 6.0,
    fdr_floor: float = 1e-50,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    title_space: float = 0.18,
    xlabel: Optional[str] = None,
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (7.0, 5.0),
    return_fig: bool = False,
    save: Optional[str] = None,
    **kwargs,
) -> Optional[Union[Figure, Axes]]:
    r"""Gene-set-enrichment lollipop plot.

    Each row is an enriched term. A stem runs from the x-axis baseline to
    a dot whose x-position encodes significance (``x_metric``) and whose
    area encodes the matched-gene count.

    Parameters
    ----------
    data : StringDBReport, DataFrame, or records
        Enrichment source. Three forms are accepted:

        * a :class:`kompot.plot.StringDBReport` instance — its
          :meth:`~kompot.plot.StringDBReport.get_functional_enrichment`
          is called with ``category`` / ``fdr_threshold``;
        * the ``signal``-sorted DataFrame that method returns;
        * any other enrichment-result DataFrame (gseapy / enrichr,
          GOATOOLS, clusterProfiler, …). Use the ``*_col`` params to map
          its columns, or rely on autodetection.

        **Expected schema** (logical field → autodetected column names):

        ===============  ====================================================
        Field            Candidate columns (case-insensitive)
        ===============  ====================================================
        term label       ``description``, ``term``, ``name``, ``pathway``, …
        score            ``signal``, ``Combined Score``, ``NES``, ``score``, …
        gene count       ``number_of_genes``, ``Count``, ``Overlap`` (``k/K``), …
        FDR              ``fdr``, ``Adjusted P-value``, ``p.adjust``, ``padj``, …
        ===============  ====================================================

    n_terms : int, default 12
        Number of top terms to display (after sorting).
    term_col, score_col, count_col, fdr_col : str, optional
        Explicit column names overriding autodetection for the term
        label, the score (used when ``x_metric="score"``), the gene count
        (dot size), and the FDR (x-axis when ``x_metric="neg_log10_fdr"``,
        plus the guide line and annotations).
    x_metric : {"neg_log10_fdr", "score"} or column name, default "neg_log10_fdr"
        What the dot's x-position encodes. ``"neg_log10_fdr"`` plots
        ``-log10(FDR)`` (manuscript default); ``"score"`` plots
        ``score_col`` directly; any other value is treated as a literal
        column name to plot.
    sort_by : str or None, default "x"
        How to order rows before taking the top ``n_terms``. ``"x"`` sorts
        by the plotted value (most significant / highest score on top);
        any column name sorts by that column; ``None`` preserves input
        order (StringDB frames already arrive ``signal``-sorted).
    ascending : bool, optional
        Sort direction override. By default sorting is descending for
        ``"x"`` / score columns and ascending for FDR-like columns.
    category : str, default "Process"
        StringDB enrichment category, used only when ``data`` is a
        StringDBReport. See
        :meth:`~kompot.plot.StringDBReport.get_functional_enrichment`.
    fdr_threshold : float, default 0.05
        FDR cutoff passed through to StringDBReport (StringDBReport path
        only).
    color : str, default ``"#d73027"``
        Lollipop fill color. The default is kompot's "up" direction red
        (:data:`kompot.utils.KOMPOT_COLORS`), matching the manuscript.
    edge_color : str, optional
        Dot outline / stem color. Defaults to a darkened ``color``.
    cmap : str, optional
        If given, dots are colored by ``color_by`` through this colormap
        (a colorbar is added on standalone figures) instead of the solid
        ``color``.
    color_by : str, optional
        Column whose values drive the ``cmap`` coloring. Defaults to the
        resolved score column when ``cmap`` is set.
    stem_lw, stem_alpha : float
        Line width and alpha of the lollipop stems.
    dot_min, dot_max : float, default 40, 320
        Clip bounds (area in pt²) for the gene-count dot sizer
        ``clip(dot_min + dot_scale * sqrt(count), dot_min, dot_max)``.
    dot_scale : float, default 22
        Multiplier in the dot sizer above.
    dot_const : float, default 80
        Constant dot area used when no gene-count column is available.
    fdr_line : float or None, default 0.05
        Draw a dashed vertical guide at this FDR (rendered at
        ``-log10(fdr_line)`` when ``x_metric="neg_log10_fdr"``). ``None``
        disables it. Ignored for non-FDR x metrics.
    annotate : bool, default True
        Annotate each dot with ``n=<count>  FDR=<fdr>`` to its right.
    annotate_fmt : str, optional
        Custom format string for the annotation, receiving ``count`` and
        ``fdr`` as keyword fields, e.g. ``"{count} genes (q={fdr:.1e})"``.
    legend : bool, default True
        Draw the aesthetic key (set swatch, dot-size cue, FDR guide).
    legend_label : str, default ``"gene set"``
        Label for the set swatch in the legend.
    label_width : int, default 55
        Soft-wrap width for term descriptions (two lines max). ``0``
        disables wrapping.
    label_fontsize, annotate_fontsize : float
        Font sizes for the y-axis term labels and the per-dot annotation.
    fdr_floor : float, default 1e-50
        FDRs are clipped to this floor before ``-log10`` to keep the
        x-axis finite.
    title, subtitle : str, optional
        Title (bold) and subtitle. On a standalone figure these sit in a
        reserved band above the axes (so the top row is never covered);
        when embedding into ``ax`` the title becomes the axes title.
    title_space : float, default 0.18
        Fraction of the standalone figure height reserved at the top for
        the title / subtitle / legend band.
    xlabel : str, optional
        X-axis label. Defaults to ``$-\log_{10}(\mathrm{FDR})$`` for the
        FDR metric, otherwise the score/column name.
    ax : matplotlib.axes.Axes, optional
        Embed into this axis. If ``None`` a standalone figure is built.
    figsize : tuple, default ``(7.0, 5.0)``
        Standalone figure size; ignored when ``ax`` is given.
    return_fig : bool, default False
        If ``True``, return the ``Figure`` (standalone) or the ``Axes``
        (embedded) instead of ``None``.
    save : str, optional
        If given, ``fig.savefig(save, bbox_inches="tight")`` is called.
    **kwargs
        Forwarded to the dot :meth:`~matplotlib.axes.Axes.scatter` call.

    Returns
    -------
    matplotlib.figure.Figure or matplotlib.axes.Axes or None
        The figure (standalone) or axis (embedded) when ``return_fig`` is
        ``True``, else ``None``.

    Examples
    --------
    From a StringDBReport (queries StringDB live)::

        import kompot
        report = kompot.plot.StringDBReport(
            ["TP53", "BRCA1", "KRAS", "EGFR", "PTEN"], species_id=9606,
        )
        kompot.plot.lollipop(report, category="Process", n_terms=10,
                             return_fig=True)

    From a precomputed enrichment table (offline, any tool)::

        import pandas as pd
        df = pd.DataFrame({
            "description": ["immune response", "cell cycle", "apoptosis"],
            "fdr": [1e-8, 3e-5, 2e-3],
            "number_of_genes": [42, 18, 9],
            "signal": [3.1, 2.0, 1.2],
        })
        kompot.plot.lollipop(df, n_terms=3, return_fig=True)

    A gseapy/enrichr frame, scored by Combined Score, mapped explicitly::

        kompot.plot.lollipop(
            enrichr_df, x_metric="score",
            term_col="Term", score_col="Combined Score",
            count_col="Overlap", fdr_col="Adjusted P-value",
        )

    Embed into a composite figure::

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        kompot.plot.lollipop(df_a, ax=axes[0], title="Condition A")
        kompot.plot.lollipop(df_b, ax=axes[1], title="Condition B")
    """
    df = _resolve_enrichment(
        data, category=category, fdr_threshold=fdr_threshold
    )
    if not isinstance(df, pd.DataFrame):  # pragma: no cover - defensive
        df = pd.DataFrame(df)
    if len(df) == 0:
        raise ValueError("enrichment frame is empty; nothing to plot")
    df = df.reset_index(drop=True)

    # ---- resolve columns ----------------------------------------
    term_c = _find_column(df, _TERM_CANDIDATES, term_col)
    if term_c is None:
        raise KeyError(
            "could not find a term/description column; pass `term_col=` "
            f"(looked for {list(_TERM_CANDIDATES)}; have {list(df.columns)})"
        )
    fdr_c = _find_column(df, _FDR_CANDIDATES, fdr_col)
    score_c = _find_column(df, _SCORE_CANDIDATES, score_col)
    count_c = _find_column(df, _COUNT_CANDIDATES, count_col)

    # ---- x values -----------------------------------------------
    if x_metric == "neg_log10_fdr":
        if fdr_c is None:
            raise KeyError(
                "x_metric='neg_log10_fdr' needs an FDR column but none was "
                "found; pass `fdr_col=` or choose x_metric='score'."
            )
        fdr_vals = pd.to_numeric(df[fdr_c], errors="coerce")
        x = -np.log10(fdr_vals.clip(lower=fdr_floor).astype(float))
        default_xlabel = r"$-\log_{10}(\mathrm{FDR})$"
    elif x_metric == "score":
        if score_c is None:
            raise KeyError(
                "x_metric='score' needs a score column but none was found; "
                "pass `score_col=` or choose x_metric='neg_log10_fdr'."
            )
        x = pd.to_numeric(df[score_c], errors="coerce")
        default_xlabel = str(score_c)
    else:
        # literal column name
        if x_metric not in df.columns:
            raise KeyError(
                f"x_metric '{x_metric}' is neither a known metric "
                "('neg_log10_fdr', 'score') nor a column in the frame "
                f"({list(df.columns)})"
            )
        x = pd.to_numeric(df[x_metric], errors="coerce")
        default_xlabel = str(x_metric)
    x = np.asarray(x, dtype=float)
    df = df.assign(_x=x)
    df = df[np.isfinite(df["_x"])]
    if len(df) == 0:
        raise ValueError(
            f"no finite x values from x_metric='{x_metric}'; check the "
            "selected column for non-numeric / missing entries"
        )

    # ---- sort + top-N -------------------------------------------
    if sort_by is not None:
        if sort_by == "x":
            sort_key, asc_default = "_x", False
        elif sort_by in df.columns:
            sort_key = sort_by
            # FDR-like → ascending (smaller is better); else descending.
            asc_default = sort_by == fdr_c
        else:
            raise KeyError(
                f"sort_by '{sort_by}' is not 'x' and not a column "
                f"({list(df.columns)})"
            )
        asc = asc_default if ascending is None else bool(ascending)
        df = df.sort_values(sort_key, ascending=asc, kind="stable")
    df = df.head(n_terms).reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("no rows remain after sorting / top-N selection")

    xv = df["_x"].to_numpy(dtype=float)

    # ---- dot sizes ----------------------------------------------
    if count_c is not None:
        counts = _coerce_counts(df[count_c])
        with np.errstate(invalid="ignore"):
            sizes = np.clip(
                dot_min + dot_scale * np.sqrt(np.clip(counts, 0, None)),
                dot_min,
                dot_max,
            )
        sizes = np.where(np.isfinite(sizes), sizes, dot_const)
    else:
        counts = np.full(len(df), np.nan)
        sizes = np.full(len(df), dot_const, dtype=float)

    # ---- fdr (for guide + annotation) ---------------------------
    if fdr_c is not None:
        fdr_series = pd.to_numeric(df[fdr_c], errors="coerce").to_numpy(dtype=float)
    else:
        fdr_series = np.full(len(df), np.nan)

    # ---- colors -------------------------------------------------
    if edge_color is None:
        edge_color = _darken(color)
    use_cmap = cmap is not None
    if use_cmap:
        cb_col = color_by if color_by is not None else score_c
        if cb_col is None or cb_col not in df.columns:
            raise KeyError(
                "cmap was given but no `color_by` column is available; pass "
                "`color_by=` (a numeric column) explicitly."
            )
        cvals = pd.to_numeric(df[cb_col], errors="coerce").to_numpy(dtype=float)
        norm = Normalize(
            vmin=float(np.nanmin(cvals)), vmax=float(np.nanmax(cvals))
        )

    # ---- axes ---------------------------------------------------
    standalone = ax is None
    if standalone:
        fig = plt.figure(figsize=figsize)
        top = max(0.55, 1.0 - float(title_space))
        ax = fig.add_axes([0.48, 0.14, 0.49, top - 0.14])
    else:
        fig = ax.figure

    # ---- stems + dots -------------------------------------------
    y = np.arange(len(df))
    for yi, xi in zip(y, xv):
        ax.plot(
            [0, xi], [yi, yi],
            color=edge_color if not use_cmap else color,
            lw=stem_lw, alpha=stem_alpha, zorder=2,
        )
    scatter_kwargs = dict(
        s=sizes, edgecolors=edge_color, linewidths=0.6, zorder=3,
    )
    scatter_kwargs.update(kwargs)
    if use_cmap:
        sc = ax.scatter(xv, y, c=cvals, cmap=cmap, norm=norm, **scatter_kwargs)
    else:
        sc = ax.scatter(xv, y, color=color, **scatter_kwargs)

    # ---- annotations --------------------------------------------
    if annotate:
        for yi, xi in zip(y, xv):
            cnt = counts[yi]
            fdr_val = fdr_series[yi]
            if annotate_fmt is not None:
                txt = annotate_fmt.format(
                    count=int(cnt) if np.isfinite(cnt) else "?",
                    fdr=fdr_val if np.isfinite(fdr_val) else float("nan"),
                )
            else:
                parts = []
                if np.isfinite(cnt):
                    parts.append(f"n={int(cnt)}")
                if np.isfinite(fdr_val):
                    if fdr_val < 1e-6:
                        parts.append(f"FDR={fdr_val:.1e}")
                    else:
                        parts.append(f"FDR={fdr_val:.2g}")
                txt = "  ".join(parts)
            if txt:
                ax.annotate(
                    txt, xy=(xi, yi), xytext=(8, 0),
                    textcoords="offset points",
                    fontsize=annotate_fontsize, color=edge_color,
                    ha="left", va="center", zorder=4,
                )

    # ---- term labels + axes cosmetics ---------------------------
    labels = [_wrap(t, label_width) for t in df[term_c].tolist()]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=label_fontsize)
    ax.set_xlabel(xlabel if xlabel is not None else default_xlabel)

    draw_guide = (
        fdr_line is not None and x_metric == "neg_log10_fdr" and fdr_line > 0
    )
    if draw_guide:
        ax.axvline(
            -np.log10(fdr_line), color="0.4", ls="--", lw=0.7,
            alpha=0.8, zorder=1,
        )

    # Right-pad so annotations never run off the axes.
    xmax = float(np.nanmax(xv)) if len(xv) else 1.0
    if not np.isfinite(xmax) or xmax <= 0:
        xmax = 1.0
    x_hi = max(xmax * 1.35, xmax + 4)
    x_lo = min(0.0, float(np.nanmin(xv)))
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(len(df) - 0.5, -0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    # ---- colorbar (cmap path, standalone only) ------------------
    if use_cmap and standalone:
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label(str(cb_col), fontsize=annotate_fontsize)
        cb.ax.tick_params(labelsize=annotate_fontsize)

    # ---- legend -------------------------------------------------
    if legend:
        handles = []
        if not use_cmap:
            handles.append(
                Line2D(
                    [0], [0], marker="o", color="w",
                    markerfacecolor=color, markeredgecolor=edge_color,
                    markersize=7, label=legend_label,
                )
            )
        if count_c is not None:
            handles.append(
                Line2D(
                    [0], [0], marker="o", color="w",
                    markerfacecolor=color if not use_cmap else "0.5",
                    markeredgecolor=edge_color, markersize=4,
                    label="dot size ∝ gene count",
                )
            )
        if draw_guide:
            handles.append(
                Line2D(
                    [0], [0], color="0.4", ls="--", lw=0.8,
                    label=f"FDR = {fdr_line:g}",
                )
            )
        if handles:
            if standalone:
                fig.legend(
                    handles=handles, loc="upper right",
                    bbox_to_anchor=(0.97, 0.97), fontsize=5.5,
                    frameon=True, framealpha=0.9, borderpad=0.4,
                    handlelength=1.8,
                )
            else:
                ax.legend(
                    handles=handles, loc="lower right", fontsize=5.5,
                    frameon=True, framealpha=0.9,
                )

    # ---- title / subtitle ---------------------------------------
    if standalone:
        if title:
            fig.text(
                0.02, 0.955, title, fontsize=10, fontweight="bold",
                ha="left", va="top",
            )
        if subtitle:
            fig.text(
                0.02, 1.0 - title_space * 0.5, subtitle, fontsize=6.5,
                color="0.35", ha="left", va="top",
            )
    else:
        if title:
            ax.set_title(title, fontsize=10, fontweight="bold", pad=4)
        if subtitle:
            ax.text(
                0.0, 1.01, subtitle, transform=ax.transAxes, fontsize=6.5,
                color="0.35", ha="left", va="bottom",
            )

    # ---- save / return ------------------------------------------
    if save is not None:
        fig.savefig(save, bbox_inches="tight")

    if return_fig:
        return fig if standalone else ax
    return None
