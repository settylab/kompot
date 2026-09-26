"""Tests for run-history provenance stamping (kompot._provenance)."""

import json
import os

import anndata
import numpy as np
import pytest

from kompot import _provenance
from kompot._provenance import (
    _find_git_dir,
    _is_installed_tree,
    _looks_like_sha,
    _pep610_editable,
    _resolve_sha,
    get_provenance,
    stamp,
)
from kompot.anndata.utils import append_to_run_history, get_run_history

SHA = "4432d4f39ab6c1d2e3f405162738495a6b7c8d9e"
OTHER_SHA = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture(autouse=True)
def _clear_provenance_cache():
    """Provenance is memoized; tests must not inherit each other's cache."""
    _provenance._CACHE = None
    yield
    _provenance._CACHE = None


def _make_git_dir(root, head="ref: refs/heads/main", loose=SHA, packed=None):
    git_dir = root / ".git"
    (git_dir / "refs" / "heads").mkdir(parents=True)
    (git_dir / "HEAD").write_text(head + "\n")
    if loose is not None:
        (git_dir / "refs" / "heads" / "main").write_text(loose + "\n")
    if packed is not None:
        (git_dir / "packed-refs").write_text(packed)
    return git_dir


# --------------------------------------------------------------------------
# sha resolution
# --------------------------------------------------------------------------


def test_resolve_sha_from_loose_ref(tmp_path):
    git_dir = _make_git_dir(tmp_path)
    assert _resolve_sha(str(git_dir)) == SHA


def test_resolve_sha_from_packed_refs(tmp_path):
    """A freshly cloned repo has its refs packed, with no loose ref file."""
    git_dir = _make_git_dir(
        tmp_path,
        loose=None,
        packed=f"# pack-refs with: peeled fully-peeled sorted\n{SHA} refs/heads/main\n",
    )
    assert _resolve_sha(str(git_dir)) == SHA


def test_resolve_sha_ignores_peeled_tag_lines(tmp_path):
    """`^`-prefixed peel lines in packed-refs must not be mistaken for refs."""
    git_dir = _make_git_dir(
        tmp_path,
        loose=None,
        packed=f"{OTHER_SHA} refs/tags/v0.7.0\n^{SHA}\n{SHA} refs/heads/main\n",
    )
    assert _resolve_sha(str(git_dir)) == SHA


def test_resolve_sha_detached_head(tmp_path):
    git_dir = _make_git_dir(tmp_path, head=SHA, loose=None)
    assert _resolve_sha(str(git_dir)) == SHA


def test_resolve_sha_returns_none_when_unresolvable(tmp_path):
    git_dir = _make_git_dir(tmp_path, loose=None)
    assert _resolve_sha(str(git_dir)) is None


# --------------------------------------------------------------------------
# linked worktrees (settylab/kompot#28, #33)
# --------------------------------------------------------------------------


def _make_linked_worktree_git_dir(root, commondir="../..", loose=SHA, packed=None):
    """Mirror git's layout: the worktree's git dir holds HEAD, the main one holds refs."""
    main = _make_git_dir(root, loose=None, packed=packed)
    if loose is not None:
        (main / "refs" / "heads" / "feature").write_text(loose + "\n")
    wt = main / "worktrees" / "wt"
    wt.mkdir(parents=True)
    (wt / "HEAD").write_text("ref: refs/heads/feature\n")
    (wt / "commondir").write_text((str(main) if commondir is None else commondir) + "\n")
    return wt


def test_resolve_sha_linked_worktree_on_a_branch(tmp_path):
    """Branch refs live in the common dir, not in the per-worktree git dir."""
    wt = _make_linked_worktree_git_dir(tmp_path)
    assert _resolve_sha(str(wt)) == SHA


def test_resolve_sha_linked_worktree_absolute_commondir(tmp_path):
    wt = _make_linked_worktree_git_dir(tmp_path, commondir=None)
    assert _resolve_sha(str(wt)) == SHA


def test_resolve_sha_linked_worktree_packed_refs_in_commondir(tmp_path):
    """After `git gc` the branch ref is packed, and packed-refs is shared too."""
    wt = _make_linked_worktree_git_dir(
        tmp_path, loose=None, packed=f"{SHA} refs/heads/feature\n"
    )
    assert _resolve_sha(str(wt)) == SHA


def _git(cwd, *args):
    import subprocess

    return subprocess.run(
        ["git", "-c", "user.name=kompot-test", "-c", "user.email=test@example.invalid",
         "-c", "commit.gpgsign=false", *args],
        cwd=str(cwd), check=True, capture_output=True, text=True,
    ).stdout.strip()


@pytest.mark.parametrize("pack", [False, True], ids=["loose", "packed"])
@pytest.mark.parametrize("shape", ["clone", "branch-worktree", "detached-worktree"])
def test_resolve_sha_matches_git_rev_parse(tmp_path, shape, pack):
    """End to end against real git: every checkout shape resolves, loose or packed.

    `branch-worktree` is the shape that returned None before #28/#33 were fixed;
    `detached-worktree` passed all along because its HEAD holds the sha.
    """
    import shutil

    if shutil.which("git") is None:
        pytest.skip("git binary not available")
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "f.txt").write_text("x\n")
    _git(repo, "add", "f.txt")
    _git(repo, "commit", "-q", "-m", "init")

    checkout = repo
    if shape == "branch-worktree":
        checkout = tmp_path / "wt"
        _git(repo, "worktree", "add", "-b", "some/branch", str(checkout))
        (checkout / "g.txt").write_text("y\n")
        _git(checkout, "add", "g.txt")
        _git(checkout, "commit", "-q", "-m", "on the branch")
    elif shape == "detached-worktree":
        checkout = tmp_path / "wt"
        _git(repo, "worktree", "add", "--detach", str(checkout))
    if pack:
        _git(repo, "pack-refs", "--all", "--prune")

    pkg = checkout / "kompot"
    pkg.mkdir()
    git_dir = _find_git_dir(str(pkg))
    assert git_dir is not None
    assert _resolve_sha(git_dir) == _git(checkout, "rev-parse", "HEAD")


@pytest.mark.parametrize("shape", ["clone", "branch-worktree"])
def test_vendored_tree_does_not_borrow_an_unrelated_repo_sha(tmp_path, monkeypatch, shape):
    """A kompot tree nested below another repository's root must not stamp its HEAD.

    The enclosing repo resolves perfectly well -- which is the danger: the
    stamp would be a confident, wrong sha. Only a `.git` at the package's
    parent counts. (A copy at another repository's root is not caught; by
    position it looks like Kompot's own checkout.)
    """
    import shutil

    if shutil.which("git") is None:
        pytest.skip("git binary not available")
    repo = tmp_path / "other"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "f.txt").write_text("x\n")
    _git(repo, "add", "f.txt")
    _git(repo, "commit", "-q", "-m", "init")
    checkout = repo
    if shape == "branch-worktree":
        checkout = tmp_path / "wt"
        _git(repo, "worktree", "add", "-b", "some/branch", str(checkout))
    pkg = checkout / "vendor" / "lib" / "kompot"
    pkg.mkdir(parents=True)
    monkeypatch.setattr(_provenance, "__file__", str(pkg / "_provenance.py"))
    monkeypatch.setattr(_provenance, "_pep610_editable", lambda: None)

    # The walk-up alone would find and resolve the unrelated repo...
    assert _resolve_sha(_find_git_dir(str(pkg))) == _git(checkout, "rev-parse", "HEAD")
    # ...and provenance must refuse it.
    assert _provenance._resolve()["kompot_git_sha"] is None


def test_own_checkout_root_is_accepted(tmp_path, monkeypatch):
    _make_git_dir(tmp_path)
    pkg = tmp_path / "kompot"
    pkg.mkdir()
    monkeypatch.setattr(_provenance, "__file__", str(pkg / "_provenance.py"))
    monkeypatch.setattr(_provenance, "_pep610_editable", lambda: True)
    assert _provenance._resolve()["kompot_git_sha"] == SHA


def test_unresolvable_sha_in_a_work_tree_is_logged(tmp_path, monkeypatch, caplog):
    """A None sha inside a checkout is a resolution failure and must be visible."""
    _make_git_dir(tmp_path, loose=None)
    pkg = tmp_path / "kompot"
    pkg.mkdir()
    monkeypatch.setattr(_provenance, "__file__", str(pkg / "_provenance.py"))
    monkeypatch.setattr(_provenance, "_pep610_editable", lambda: True)

    with caplog.at_level("WARNING", logger="kompot"):
        result = _provenance._resolve()
    assert result["kompot_git_sha"] is None
    assert any("could not be resolved" in r.getMessage() for r in caplog.records)


def test_resolved_sha_logs_nothing(tmp_path, monkeypatch, caplog):
    _make_git_dir(tmp_path)
    pkg = tmp_path / "kompot"
    pkg.mkdir()
    monkeypatch.setattr(_provenance, "__file__", str(pkg / "_provenance.py"))
    monkeypatch.setattr(_provenance, "_pep610_editable", lambda: True)

    with caplog.at_level("WARNING", logger="kompot"):
        assert _provenance._resolve()["kompot_git_sha"] == SHA
    assert not caplog.records


def test_looks_like_sha_rejects_abbreviations():
    assert _looks_like_sha(SHA)
    assert not _looks_like_sha(SHA[:7])
    assert not _looks_like_sha("z" * 40)


def test_find_git_dir_walks_up_and_follows_git_file(tmp_path):
    """A worktree/submodule stores a `.git` *file* pointing at the real dir."""
    real = tmp_path / "actual_git"
    real.mkdir()
    pkg = tmp_path / "checkout" / "kompot"
    pkg.mkdir(parents=True)
    (tmp_path / "checkout" / ".git").write_text(f"gitdir: {real}\n")
    assert _find_git_dir(str(pkg)) == str(real)


def test_find_git_dir_returns_none_outside_work_tree(tmp_path):
    pkg = tmp_path / "nowhere" / "kompot"
    pkg.mkdir(parents=True)
    assert _find_git_dir(str(pkg)) is None


# --------------------------------------------------------------------------
# install-shape detection
# --------------------------------------------------------------------------


def test_is_installed_tree_detects_site_packages():
    assert _is_installed_tree(os.path.join("/venv", "lib", "python3.12", "site-packages", "kompot"))
    assert _is_installed_tree(os.path.join("/usr", "lib", "python3", "dist-packages", "kompot"))
    assert not _is_installed_tree(os.path.join("/home", "me", "src", "kompot", "kompot"))


def test_installed_tree_does_not_borrow_a_surrounding_repo_sha(tmp_path, monkeypatch):
    """A venv created *inside* an unrelated git repo must not stamp its sha.

    Walking up from site-packages would otherwise find that repository's `.git`
    and attribute a wheel install to a checkout it has nothing to do with.
    """
    _make_git_dir(tmp_path)
    pkg = tmp_path / "venv" / "lib" / "python3.12" / "site-packages" / "kompot"
    pkg.mkdir(parents=True)

    monkeypatch.setattr(_provenance, "__file__", str(pkg / "_provenance.py"))
    monkeypatch.setattr(_provenance, "_pep610_editable", lambda: False)

    result = _provenance._resolve()
    assert result["kompot_git_sha"] is None
    assert result["kompot_editable"] is False


class _FakeDist:
    def __init__(self, path, name="kompot", direct_url=None):
        self._path = path
        self.metadata = {"Name": name}
        self._direct_url = direct_url

    def read_text(self, filename):
        if filename == "direct_url.json":
            return self._direct_url
        return None


def test_pep610_skips_egg_info_metadata(monkeypatch):
    """Regression: a stale source-tree `.egg-info` must not mask an editable install.

    `.egg-info` predates PEP 610 and can never carry `direct_url.json`, so
    treating its absence as "no direct URL, therefore an index install" reported
    editable installs as non-editable -- the exact case the flag exists for.
    """
    editable_json = json.dumps({"dir_info": {"editable": True}})
    import importlib.metadata

    monkeypatch.setattr(
        importlib.metadata,
        "distributions",
        lambda: [
            _FakeDist("/src/kompot.egg-info"),  # legacy, must be skipped
            _FakeDist("/venv/site-packages/kompot-0.8.0.dist-info", direct_url=editable_json),
        ],
    )
    assert _pep610_editable() is True


def test_pep610_reports_false_for_index_install(monkeypatch):
    import importlib.metadata

    monkeypatch.setattr(
        importlib.metadata,
        "distributions",
        lambda: [_FakeDist("/venv/site-packages/kompot-0.8.0.dist-info")],
    )
    assert _pep610_editable() is False


# --------------------------------------------------------------------------
# contract: always three keys, never raises
# --------------------------------------------------------------------------


def test_provenance_always_has_all_three_keys():
    """A present-but-null field is self-describing; a missing key is not."""
    p = get_provenance()
    assert set(p) == {"kompot_version", "kompot_git_sha", "kompot_editable"}


def test_provenance_is_json_serializable():
    """run_history is persisted as JSON inside .uns."""
    json.dumps(get_provenance())


def test_provenance_never_raises(monkeypatch):
    """A failure to resolve provenance must not fail an analysis that succeeded."""
    monkeypatch.setattr(
        _provenance, "_resolve", lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    _provenance._CACHE = None
    p = get_provenance()
    assert p == {
        "kompot_version": None,
        "kompot_git_sha": None,
        "kompot_editable": None,
    }


def test_stamp_does_not_overwrite_existing_fields():
    run_info = {"kompot_version": "0.6.3", "params": {"a": 1}}
    stamped = stamp(run_info)
    assert stamped["kompot_version"] == "0.6.3"
    assert stamped["params"] == {"a": 1}


# --------------------------------------------------------------------------
# integration with run_history
# --------------------------------------------------------------------------


def _adata():
    return anndata.AnnData(X=np.random.rand(10, 4).astype("float32"))


@pytest.mark.parametrize("analysis_type", ["de", "da", "smooth"])
def test_run_history_entries_are_stamped(analysis_type):
    adata = _adata()
    append_to_run_history(adata, {"params": {"x": 1}}, analysis_type)

    entry = get_run_history(adata, analysis_type)[0]
    assert entry["params"] == {"x": 1}
    for key in ("kompot_version", "kompot_git_sha", "kompot_editable"):
        assert key in entry, f"{key} missing for {analysis_type}"


def test_stamping_is_in_place_so_last_run_info_agrees():
    """Callers persist the same dict as `last_run_info` right after appending.

    Stamping a copy would leave `last_run_info` describing an unversioned run
    while the history entry for that very run claimed a version.
    """
    adata = _adata()
    run_info = {"params": {"x": 1}}
    append_to_run_history(adata, run_info, "de")

    assert run_info["kompot_version"] == get_provenance()["kompot_version"]
    assert "kompot_git_sha" in run_info
    assert "kompot_editable" in run_info


def test_pre_080_stores_without_provenance_still_read():
    """Forward-only: stores written before 0.8.0 lack these keys and must still load."""
    adata = _adata()
    legacy = [{"params": {"x": 1}, "timestamp": "2026-03-24T00:00:00"}]
    from kompot.anndata.utils.json_utils import set_json_metadata

    set_json_metadata(adata, "kompot_de.run_history", legacy)

    history = get_run_history(adata, "de")
    assert len(history) == 1
    assert history[0]["timestamp"] == "2026-03-24T00:00:00"
    assert history[0].get("kompot_version") is None  # absent, not fabricated


def test_appending_to_a_legacy_store_preserves_unstamped_entries():
    """Stamping is forward-only: it must not retroactively describe old runs."""
    adata = _adata()
    from kompot.anndata.utils.json_utils import set_json_metadata

    set_json_metadata(adata, "kompot_de.run_history", [{"params": {"old": True}}])
    append_to_run_history(adata, {"params": {"new": True}}, "de")

    history = get_run_history(adata, "de")
    assert len(history) == 2
    assert "kompot_version" not in history[0]
    assert "kompot_version" in history[1]


def test_source_checkout_resolves_sha_and_editable():
    """The load-bearing case: code running from a git checkout identifies itself.

    CI installs with ``pip install -e .`` inside the repo, so this exercises the
    editable path that a wheel install never reaches. Skipped when the tests are
    run against an installed (non-checkout) kompot, where there is nothing to
    resolve and the wheel-degradation contract applies instead.
    """
    package_dir = os.path.dirname(os.path.abspath(_provenance.__file__))
    # A .git at the package's parent, not merely an enclosing work tree: a tree
    # exported or vendored below another repository's root must skip here.
    if _is_installed_tree(package_dir) or _provenance._find_kompot_git_dir(package_dir) is None:
        pytest.skip("kompot is not running from its own source checkout")

    p = get_provenance()
    assert p["kompot_version"]
    assert p["kompot_git_sha"] is not None, "sha must resolve inside a work tree"
    assert _looks_like_sha(p["kompot_git_sha"])
    assert p["kompot_editable"] is True
