"""Length-scale determination schemes for the two-condition expression fit.

The default (``ls_scheme="condition1"``) estimates the shared length scale from
condition 1's cells only, which makes the contrast depend on which condition is
passed first.  These tests pin the default's behaviour and check that each
alternative does what it says.
"""

import numpy as np
import pytest

from kompot.differential.differential_expression import (
    LS_SCHEMES,
    DifferentialExpression,
    _auto_ls,
    _resolve_ls_scheme,
)
from kompot.settings import GPSettings


def _data(n1=120, n2=40, n_genes=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n1 + n2, 3))
    y = rng.normal(size=(n1 + n2, n_genes))
    return X[:n1], y[:n1], X[n1:], y[n1:]


# --------------------------------------------------------------------------
# scheme resolution
# --------------------------------------------------------------------------


def test_condition1_and_separate_defer_to_the_models():
    X1, _, X2, _ = _data()
    assert _resolve_ls_scheme("condition1", X1, X2, 10.0) == (None, None)
    assert _resolve_ls_scheme("separate", X1, X2, 10.0) == (None, None)


def test_pooled_uses_the_union_and_is_smaller_than_either_condition():
    X1, _, X2, _ = _data()
    p1, p2 = _resolve_ls_scheme("pooled", X1, X2, 10.0)
    assert p1 == p2
    assert p1 == pytest.approx(_auto_ls(np.vstack([X1, X2]), 10.0))
    # The union is denser than either part, so nearest-neighbour distances --
    # and hence the length scale -- shrink.  This is the property that makes
    # "pooled" a different question from "what scale does this condition
    # support", and it is why it is not the default.
    assert p1 < _auto_ls(X1, 10.0)
    assert p1 < _auto_ls(X2, 10.0)


def test_symmetric_is_the_size_weighted_geometric_mean():
    X1, _, X2, _ = _data()
    got, got2 = _resolve_ls_scheme("symmetric", X1, X2, 10.0)
    assert got == got2
    a, b = _auto_ls(X1, 10.0), _auto_ls(X2, 10.0)
    n1, n2 = X1.shape[0], X2.shape[0]
    assert got == pytest.approx(np.exp((n1 * np.log(a) + n2 * np.log(b)) / (n1 + n2)))
    assert min(a, b) <= got <= max(a, b)


def test_symmetric_is_invariant_under_swapping_the_conditions():
    X1, _, X2, _ = _data()
    fwd, _ = _resolve_ls_scheme("symmetric", X1, X2, 10.0)
    rev, _ = _resolve_ls_scheme("symmetric", X2, X1, 10.0)
    assert fwd == pytest.approx(rev, rel=1e-9)


def test_pooled_is_invariant_under_swapping_the_conditions():
    X1, _, X2, _ = _data()
    fwd, _ = _resolve_ls_scheme("pooled", X1, X2, 10.0)
    rev, _ = _resolve_ls_scheme("pooled", X2, X1, 10.0)
    # The union is the same point set either way; the only difference is row
    # order, which the approximate nearest-neighbour index is mildly
    # sensitive to.
    assert fwd == pytest.approx(rev, rel=1e-2)


def test_unknown_scheme_is_rejected():
    X1, _, X2, _ = _data()
    with pytest.raises(ValueError, match="Unknown ls_scheme"):
        _resolve_ls_scheme("nope", X1, X2, 10.0)


def test_gp_settings_rejects_unknown_scheme():
    with pytest.raises(ValueError, match="ls_scheme"):
        GPSettings(ls_scheme="nope")
    for scheme in LS_SCHEMES:
        GPSettings(ls_scheme=scheme)


def test_gp_settings_default_preserves_historical_behaviour():
    assert GPSettings().ls_scheme == "condition1"


# --------------------------------------------------------------------------
# end-to-end through DifferentialExpression.fit
# --------------------------------------------------------------------------


def _fit(scheme, swap=False, **kw):
    X1, y1, X2, y2 = _data()
    if swap:
        X1, y1, X2, y2 = X2, y2, X1, y1
    de = DifferentialExpression(n_landmarks=0)
    de.fit(X1, y1, X2, y2, ls_scheme=scheme, **kw)
    return float(de.model1.ls), float(de.model2.ls)


def test_default_gives_condition2_condition1s_length_scale():
    ls1, ls2 = _fit("condition1")
    assert ls1 == pytest.approx(ls2)
    X1, _, _, _ = _data()
    assert ls1 == pytest.approx(_auto_ls(X1, 10.0), rel=1e-6)


def test_default_is_not_invariant_under_swapping_the_conditions():
    """The defect #310 reports: de(A, B) and de(B, A) smooth differently."""
    fwd, _ = _fit("condition1")
    rev, _ = _fit("condition1", swap=True)
    assert fwd != pytest.approx(rev, rel=1e-3)


def test_separate_gives_each_condition_its_own():
    X1, _, X2, _ = _data()
    ls1, ls2 = _fit("separate")
    assert ls1 == pytest.approx(_auto_ls(X1, 10.0), rel=1e-6)
    assert ls2 == pytest.approx(_auto_ls(X2, 10.0), rel=1e-6)
    assert ls1 != pytest.approx(ls2, rel=1e-3)


@pytest.mark.parametrize("scheme", ["symmetric", "pooled"])
def test_shared_schemes_are_swap_invariant_end_to_end(scheme):
    fwd = _fit(scheme)
    rev = _fit(scheme, swap=True)
    assert fwd[0] == pytest.approx(fwd[1])
    assert rev[0] == pytest.approx(rev[1])
    assert fwd[0] == pytest.approx(rev[0], rel=1e-2)


@pytest.mark.parametrize("scheme", LS_SCHEMES)
def test_explicit_ls_overrides_every_scheme(scheme):
    ls1, ls2 = _fit(scheme, ls=0.5)
    assert ls1 == pytest.approx(0.5)
    assert ls2 == pytest.approx(0.5)
