"""Fail the docs build when reStructuredText markup survives into the HTML.

Nested inline markup (``**the ``dask`` path**``) is invalid RST, and docutils
does not warn about it: it stops treating the inner markers as markup and the
page renders literal backticks, while ``sphinx-build`` exits 0
(settylab/kompot#30). The same silent pass covers a role or directive written
where it is not interpreted, such as inside a code-block comment.

Rather than a rule per construct, this asserts the one property all of them
violate: nothing that is supposed to be markup appears as text in the output.
Run automatically from ``conf.py`` at ``build-finished``; also usable
standalone as ``python markup_leak_check.py <html-build-dir>``.
"""

import html as _html
import re
import sys
from pathlib import Path

# Markup that must never survive rendering, as it would appear in page text.
LEAK_PATTERNS = {
    # One hit per literal: a closed ``pair`` on one line, or a stray opener.
    "double-backtick literal": re.compile(r"``(?:[^`\n]*``)?"),
    "role": re.compile(r":(?:doc|ref|func|class|meth|mod|attr|obj|data|exc|term):`"),
    "directive": re.compile(
        r"^\s*\.\. (?:warning|note|admonition|code-block|list-table|toctree|"
        r"autofunction|autoclass|image|figure|math|seealso|versionadded|"
        r"deprecated)::",
        re.MULTILINE,
    ),
}

# Sphinx's own search index and sources copies are not rendered pages.
_SKIP_DIRS = {"_sources", "_static", "_images", "_modules"}

_TAG = re.compile(r"<[^>]+>")
_SCRIPT_STYLE = re.compile(r"<(script|style)\b.*?</\1>", re.DOTALL | re.IGNORECASE)


def page_text(markup: str) -> str:
    """The visible text of an HTML page: tags removed, entities decoded."""
    markup = _SCRIPT_STYLE.sub("", markup)
    return _html.unescape(_TAG.sub("", markup))


def find_leaks(markup: str):
    """Return ``[(kind, line_text), ...]`` for every leaked construct in *markup*."""
    text = page_text(markup)
    leaks = []
    for kind, pattern in LEAK_PATTERNS.items():
        for match in pattern.finditer(text):
            start = text.rfind("\n", 0, match.start()) + 1
            end = text.find("\n", match.end())
            line = text[start : end if end != -1 else len(text)].strip()
            leaks.append((kind, line[:160]))
    return leaks


def scan_build(outdir):
    """Scan every rendered page under *outdir*; return ``{relpath: leaks}``."""
    outdir = Path(outdir)
    found = {}
    for page in sorted(outdir.rglob("*.html")):
        rel = page.relative_to(outdir)
        if _SKIP_DIRS.intersection(rel.parts):
            continue
        leaks = find_leaks(page.read_text(encoding="utf-8", errors="replace"))
        if leaks:
            found[str(rel)] = leaks
    return found


def format_report(found) -> str:
    lines = [
        f"{sum(len(v) for v in found.values())} unrendered markup construct(s) "
        f"in {len(found)} page(s) -- nested inline markup, or a role/directive "
        "where it is not interpreted (settylab/kompot#30):"
    ]
    for page, leaks in found.items():
        for kind, line in leaks:
            lines.append(f"  {page}: {kind}: {line}")
    return "\n".join(lines)


def check_after_build(app, exception):
    """Sphinx ``build-finished`` hook: fail an HTML build that leaked markup."""
    if exception is not None or app.builder.format != "html":
        return
    found = scan_build(app.outdir)
    if found:
        from sphinx.errors import ExtensionError

        raise ExtensionError(format_report(found))


if __name__ == "__main__":
    result = scan_build(sys.argv[1])
    if result:
        print(format_report(result))
        sys.exit(1)
    print("no unrendered markup found")
