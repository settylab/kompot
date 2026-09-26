"""The docs build must fail when RST markup survives into the HTML (settylab/kompot#30).

Nested inline markup is invalid reStructuredText; docutils renders the inner
markers as literal text and ``sphinx-build`` exits 0. The check lives in
``docs/source/markup_leak_check.py`` and runs at ``build-finished``.
"""

import sys
from pathlib import Path

import pytest

DOCS_SOURCE = Path(__file__).resolve().parents[1] / "docs" / "source"
sys.path.insert(0, str(DOCS_SOURCE))
from markup_leak_check import find_leaks  # noqa: E402

# The three shapes reported on #30, as docutils actually renders them.
NESTED = [
    "<p><strong>Supplying ``sample_col`` multiplies the cost by the gene count.</strong></p>",
    "<p><strong>The ``dask`` path wins on every instrument</strong>, and that is it.</p>",
    "<p><strong>The ``dask``-less path is a different bargain ...</strong></p>",
]


@pytest.mark.parametrize("html", NESTED, ids=["sample_col", "dask", "dask-less"])
def test_nested_literal_in_bold_is_a_leak(html):
    assert [kind for kind, _ in find_leaks(html)] == ["double-backtick literal"]


def test_role_in_a_code_block_comment_is_a_leak():
    html = '<pre><span class="c1"># see :ref:`dry-run` for the plan</span></pre>'
    assert [kind for kind, _ in find_leaks(html)] == ["role"]


def test_directive_rendered_as_text_is_a_leak():
    html = "<p>.. warning::\nthis never became an admonition</p>"
    assert [kind for kind, _ in find_leaks(html)] == ["directive"]


def test_correctly_rendered_markup_is_clean():
    html = (
        '<p><strong>The</strong> <code class="docutils literal notranslate">'
        '<span class="pre">dask</span></code> <strong>path wins</strong>; see '
        '<a class="reference internal" href="#x"><span class="std std-ref">x</span></a>.</p>'
        "<script>var s = '``not page text``';</script>"
    )
    assert find_leaks(html) == []


def test_one_hit_per_literal_not_per_backtick_pair():
    assert len(find_leaks("<p>a ``x`` b ``y`` c</p>")) == 2


def test_sphinx_build_fails_on_nested_markup(tmp_path):
    """End to end: a real build of the #30 shape must exit non-zero, and a clean one zero."""
    pytest.importorskip("sphinx")
    from sphinx.cmd.build import build_main

    def build(body, name):
        src = tmp_path / name / "src"
        src.mkdir(parents=True)
        (src / "conf.py").write_text(
            "import sys\n"
            f"sys.path.insert(0, {str(DOCS_SOURCE)!r})\n"
            "from markup_leak_check import check_after_build\n"
            "def setup(app):\n"
            "    app.connect('build-finished', check_after_build)\n"
        )
        (src / "index.rst").write_text("Title\n=====\n\n" + body + "\n")
        return build_main(["-q", "-b", "html", str(src), str(tmp_path / name / "out")])

    assert build("**The dask path wins** with ``dask``.", "clean") == 0
    assert build("**The ``dask`` path wins on every instrument**", "nested") != 0
