"""Provenance of the running Kompot code, stamped into every run history entry.

A released version string does not identify the code that ran: an editable
install of ``v0.7.0-7-g4432d4f`` still reports ``__version__ == "0.7.0"``. The
git sha and the editable flag are what close that gap, so a stored ``.h5ad``
can say which Kompot actually produced it.

Resolution is stdlib-only and never raises: every field degrades to ``None``
rather than failing a computation that has already succeeded. The git sha is
read straight out of the ``.git`` directory rather than by shelling out to
``git``, so it works with no git binary installed.
"""

import json
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger("kompot")

# Resolved once on first use and reused; provenance cannot change within a
# process, and run stamping must not pay for it per run.
_CACHE: Optional[Dict[str, Any]] = None


def _is_installed_tree(package_dir: str) -> bool:
    """Whether *package_dir* lives inside an installed-packages directory.

    Guards against a false positive that would otherwise be easy to hit: a
    virtualenv created *inside* an unrelated git work tree. Walking up from
    ``site-packages`` would find that repository's ``.git`` and stamp its sha
    onto runs of a perfectly ordinary wheel install.
    """
    parts = os.path.normpath(package_dir).split(os.sep)
    return "site-packages" in parts or "dist-packages" in parts


def _read_text(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read().strip()
    except OSError:
        return None


def _find_git_dir(start: str) -> Optional[str]:
    """Locate the ``.git`` directory governing *start*, or ``None``."""
    current = os.path.abspath(start)
    while True:
        candidate = os.path.join(current, ".git")
        if os.path.isdir(candidate):
            return candidate
        if os.path.isfile(candidate):
            # Worktree or submodule: a `.git` *file* points at the real dir.
            contents = _read_text(candidate) or ""
            if contents.startswith("gitdir:"):
                target = contents[len("gitdir:") :].strip()
                if not os.path.isabs(target):
                    target = os.path.join(current, target)
                return os.path.abspath(target) if os.path.isdir(target) else None
            return None
        parent = os.path.dirname(current)
        if parent == current:
            return None
        current = parent


def _resolve_sha(git_dir: str) -> Optional[str]:
    """Resolve HEAD to a full sha by reading git's on-disk refs."""
    head = _read_text(os.path.join(git_dir, "HEAD"))
    if not head:
        return None

    if not head.startswith("ref:"):
        # Detached HEAD already holds the sha.
        return head if _looks_like_sha(head) else None

    ref = head[len("ref:") :].strip()

    # Loose ref.
    loose = _read_text(os.path.join(git_dir, *ref.split("/")))
    if loose and _looks_like_sha(loose):
        return loose

    # Packed ref.
    packed = _read_text(os.path.join(git_dir, "packed-refs"))
    if packed:
        for line in packed.splitlines():
            if not line or line.startswith(("#", "^")):
                continue
            sha, _, name = line.partition(" ")
            if name.strip() == ref and _looks_like_sha(sha):
                return sha
    return None


def _looks_like_sha(value: str) -> bool:
    return len(value) == 40 and all(c in "0123456789abcdef" for c in value.lower())


def _pep610_editable() -> Optional[bool]:
    """Editable flag from installed metadata (PEP 610), or ``None`` if unknown.

    Only ``.dist-info`` carries ``direct_url.json``; a legacy ``.egg-info`` left
    in a source tree cannot, and ``Distribution.from_name`` will happily return
    that one when the checkout is on ``sys.path``. Reading its silence as "no
    direct URL, therefore an index install" would report an editable install as
    non-editable -- precisely the case this field exists to catch. So egg-info
    metadata is skipped, and absence is only conclusive from a real dist-info.
    """
    try:
        from importlib.metadata import distributions

        fallback: Optional[bool] = None
        for dist in distributions():
            try:
                if (dist.metadata["Name"] or "").lower().replace("_", "-") != "kompot":
                    continue
            except Exception:
                continue
            if str(getattr(dist, "_path", "")).endswith(".egg-info"):
                continue
            raw = dist.read_text("direct_url.json")
            if raw is None:
                # Real dist-info with no direct URL: an index install.
                fallback = False
                continue
            return bool(json.loads(raw).get("dir_info", {}).get("editable", False))
        return fallback
    except Exception:
        return None


def _resolve() -> Dict[str, Any]:
    from . import __version__

    provenance: Dict[str, Any] = {
        "kompot_version": __version__,
        "kompot_git_sha": None,
        "kompot_editable": None,
    }

    package_dir = os.path.dirname(os.path.abspath(__file__))
    editable = _pep610_editable()

    if _is_installed_tree(package_dir):
        # Installed into site-packages: not a source checkout, and any .git
        # found above it belongs to something else.
        provenance["kompot_editable"] = False if editable is None else editable
        return provenance

    git_dir = _find_git_dir(package_dir)
    if git_dir is not None:
        provenance["kompot_git_sha"] = _resolve_sha(git_dir)

    if editable is not None:
        provenance["kompot_editable"] = editable
    else:
        # No usable distribution metadata (e.g. running straight from a
        # checkout): a governing work tree means this is source-checkout code.
        provenance["kompot_editable"] = git_dir is not None

    return provenance


def get_provenance() -> Dict[str, Any]:
    """Return the provenance stamp for the running Kompot code.

    Always returns all three keys. A field is ``None`` when it could not be
    resolved -- a present-but-null field is self-describing, whereas a missing
    key is indistinguishable from a store written before stamping existed.
    """
    global _CACHE
    if _CACHE is None:
        try:
            _CACHE = _resolve()
        except Exception:  # pragma: no cover - defensive
            logger.debug("Could not resolve kompot provenance", exc_info=True)
            _CACHE = {
                "kompot_version": None,
                "kompot_git_sha": None,
                "kompot_editable": None,
            }
    return dict(_CACHE)


def stamp(run_info: Dict[str, Any]) -> Dict[str, Any]:
    """Add provenance fields to *run_info* without overwriting existing ones."""
    for key, value in get_provenance().items():
        run_info.setdefault(key, value)
    return run_info
