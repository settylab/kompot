"""``param_scheme``: where the hyperparameters of a two-condition fit come from.

One ``GPSettings`` field drives both entry points.  In differential EXPRESSION
it covers the length scale; in differential ABUNDANCE it covers ``d``, ``mu``
and ``ls`` (the second half of this file).

The expression default (``param_scheme=None``, i.e. ``"condition1"``) estimates
the shared length scale from condition 1's cells only, which makes the contrast
depend on which condition is passed first.  These tests pin the default's
behaviour and check that each alternative does what it says -- including
``"condition2"``, the default's mirror, for which the statement is an exact
equivalence:
``de(X, Y, param_scheme="condition1")`` and ``de(Y, X, param_scheme="condition2")``
are the same computation with the labels exchanged.
"""

import contextlib
import logging

import numpy as np
import pytest

from kompot.differential.differential_expression import (
    DifferentialExpression,
    _auto_ls,
    _resolve_scheme_ls,
)
from kompot.settings import PARAM_SCHEMES, GPSettings


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
    assert _resolve_scheme_ls("condition1", X1, X2, 10.0) == (None, None)
    assert _resolve_scheme_ls("separate", X1, X2, 10.0) == (None, None)


def test_pooled_uses_the_union_and_is_smaller_than_either_condition():
    X1, _, X2, _ = _data()
    p1, p2 = _resolve_scheme_ls("pooled", X1, X2, 10.0)
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
    got, got2 = _resolve_scheme_ls("symmetric", X1, X2, 10.0)
    assert got == got2
    a, b = _auto_ls(X1, 10.0), _auto_ls(X2, 10.0)
    n1, n2 = X1.shape[0], X2.shape[0]
    assert got == pytest.approx(np.exp((n1 * np.log(a) + n2 * np.log(b)) / (n1 + n2)))
    assert min(a, b) <= got <= max(a, b)


def test_symmetric_is_invariant_under_swapping_the_conditions():
    X1, _, X2, _ = _data()
    fwd, _ = _resolve_scheme_ls("symmetric", X1, X2, 10.0)
    rev, _ = _resolve_scheme_ls("symmetric", X2, X1, 10.0)
    assert fwd == pytest.approx(rev, rel=1e-9)


def test_pooled_is_invariant_under_swapping_the_conditions():
    X1, _, X2, _ = _data()
    fwd, _ = _resolve_scheme_ls("pooled", X1, X2, 10.0)
    rev, _ = _resolve_scheme_ls("pooled", X2, X1, 10.0)
    # The union is the same point set either way; the only difference is row
    # order, which the approximate nearest-neighbour index is mildly
    # sensitive to.
    assert fwd == pytest.approx(rev, rel=1e-2)


def test_unknown_scheme_is_rejected():
    X1, _, X2, _ = _data()
    with pytest.raises(ValueError, match="Unknown param_scheme"):
        _resolve_scheme_ls("nope", X1, X2, 10.0)
    with pytest.raises(ValueError, match="param_scheme"):
        DifferentialExpression(n_landmarks=0).fit(*_data(), param_scheme="nope")


def test_gp_settings_rejects_unknown_scheme():
    with pytest.raises(ValueError, match="param_scheme"):
        GPSettings(param_scheme="nope")
    for scheme in PARAM_SCHEMES:
        GPSettings(param_scheme=scheme)


def test_gp_settings_default_is_each_entry_points_own():
    """``None`` defers to the entry point; for expression that is condition1."""
    assert GPSettings().param_scheme is None
    assert _fit(None) == _fit("condition1")


# --------------------------------------------------------------------------
# end-to-end through DifferentialExpression.fit
# --------------------------------------------------------------------------


def _fit(scheme, swap=False, **kw):
    X1, y1, X2, y2 = _data()
    if swap:
        X1, y1, X2, y2 = X2, y2, X1, y1
    de = DifferentialExpression(n_landmarks=0)
    de.fit(X1, y1, X2, y2, param_scheme=scheme, **kw)
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


@pytest.mark.parametrize("scheme", PARAM_SCHEMES)
def test_explicit_ls_overrides_every_scheme(scheme):
    ls1, ls2 = _fit(scheme, ls=0.5)
    assert ls1 == pytest.approx(0.5)
    assert ls2 == pytest.approx(0.5)


# --------------------------------------------------------------------------
# a custom kernel must not silently change the default's sharing behaviour
# --------------------------------------------------------------------------


def _fit_with_curry(scheme):
    from mellon.cov import Matern52

    X1, y1, X2, y2 = _data()
    de = DifferentialExpression(n_landmarks=0)
    de.fit(X1, y1, X2, y2, param_scheme=scheme, cov_func_curry=Matern52)
    return float(de.model1.ls), float(de.model2.ls)


def test_default_still_shares_when_a_cov_func_curry_is_supplied():
    """Regression: `cov_func_curry` must not turn the default into `separate`.

    `function_kwargs` is public and the CLI forwards into it, so a caller who
    never touched `param_scheme` can reach this path.  Before the fix, supplying a
    curry dropped the condition-1 inheritance and let condition 2 estimate its
    own -- i.e. `separate` behaviour, silently, at the default.
    """
    ls1, ls2 = _fit_with_curry("condition1")
    assert ls1 == pytest.approx(ls2), (
        "condition 2 stopped inheriting condition 1's length scale when a "
        "cov_func_curry was supplied"
    )
    # and it is condition 1's own estimate that both are using
    plain1, plain2 = _fit("condition1")
    assert ls1 == pytest.approx(plain1, rel=1e-6)


def test_shared_schemes_still_share_when_a_cov_func_curry_is_supplied():
    for scheme in ("symmetric", "pooled"):
        ls1, ls2 = _fit_with_curry(scheme)
        assert ls1 == pytest.approx(ls2), scheme
        assert ls1 == pytest.approx(_fit(scheme)[0], rel=1e-6), scheme


def test_separate_still_unshares_when_a_cov_func_curry_is_supplied():
    ls1, ls2 = _fit_with_curry("separate")
    assert ls1 != pytest.approx(ls2, rel=1e-3)


# --------------------------------------------------------------------------
# "condition2" -- the mirror of the default
# --------------------------------------------------------------------------


def test_auto_ls_is_bit_identical_to_the_models_internal_estimate():
    """`_auto_ls` must reproduce what `ExpressionModel.fit` derives for itself.

    This is the load-bearing assumption behind the `condition1` / `condition2`
    mirror.  `condition1` resolves lazily -- model1 estimates internally and
    model2 inherits the fitted value -- while `condition2` has to resolve
    *eagerly*, because model1 is fitted first and cannot inherit from a model
    that does not exist yet.  The two schemes are the same computation only for
    as long as the eager and lazy estimates agree exactly.  Measured equal to
    the last bit on numpy 2.4 / jax 0.10 / mellon 1.7.1.
    """
    X1, y1, X2, y2 = _data()
    de = DifferentialExpression(n_landmarks=0)
    de.fit(X1, y1, X2, y2, param_scheme="condition1")
    assert float(de.model1.ls) == _auto_ls(X1, 10.0)


def test_condition2_shares_condition2s_length_scale():
    X1, _, X2, _ = _data()
    a, b = _resolve_scheme_ls("condition2", X1, X2, 10.0)
    assert a == b
    assert a == pytest.approx(_auto_ls(X2, 10.0))


def test_condition2_end_to_end_gives_condition1_condition2s_length_scale():
    X1, _, X2, _ = _data()
    ls1, ls2 = _fit("condition2")
    assert ls1 == pytest.approx(ls2)
    assert ls1 == pytest.approx(_auto_ls(X2, 10.0), rel=1e-6)


def test_condition2_is_not_invariant_under_swapping_the_conditions():
    """Characterisation: `condition2` is as asymmetric as the default, by design.

    It is not an alternative to the default -- it is the default's mirror, and
    its purpose is diagnostic.  If this ever starts passing, the scheme has
    stopped being a mirror and the equivalence below is no longer meaningful.
    """
    fwd, _ = _fit("condition2")
    rev, _ = _fit("condition2", swap=True)
    assert fwd != pytest.approx(rev, rel=1e-3)


def test_condition2_resolves_to_what_condition1_resolves_to_on_the_swapped_pair():
    """The resolver-level statement of the mirror."""
    X1, _, X2, _ = _data()
    c2, _ = _resolve_scheme_ls("condition2", X1, X2, 10.0)
    # `condition1` defers to the model, so its value is `_auto_ls` of whatever
    # is passed first -- which after the swap is X2.
    assert c2 == _auto_ls(X2, 10.0)


def test_condition2_still_shares_when_a_cov_func_curry_is_supplied():
    """Regression for the `487a02e` defect, re-checked at the new value.

    The gate is `ls is None and "ls" not in function_kwargs` -- blind to the
    scheme's value -- so `condition2` inherits the fix rather than needing its
    own.  This test is what makes that "inherits" a measurement.
    """
    ls1, ls2 = _fit_with_curry("condition2")
    assert ls1 == pytest.approx(ls2)
    assert ls1 == pytest.approx(_fit("condition2")[0], rel=1e-6)


# --------------------------------------------------------------------------
# the equivalence: de(X, Y, "condition1") == de(Y, X, "condition2")
# --------------------------------------------------------------------------


def _mirror_pair(n_landmarks=0, landmarks=None, n_genes=6):
    """Fit both orientations and predict both on the same points.

    Returns ``(de_forward, de_mirror, result_forward, result_mirror)``.
    Forward is ``de(X, Y, "condition1")``; mirror is ``de(Y, X, "condition2")``.
    """
    X1, y1, X2, y2 = _data(n_genes=n_genes)
    X_new = np.vstack([X1, X2])

    fwd = DifferentialExpression(n_landmarks=n_landmarks, random_state=0)
    fwd.fit(X1, y1, X2, y2, param_scheme="condition1", landmarks=landmarks)

    mir = DifferentialExpression(n_landmarks=n_landmarks, random_state=0)
    mir.fit(X2, y2, X1, y1, param_scheme="condition2", landmarks=landmarks)

    return (
        fwd,
        mir,
        fwd.predict(X_new, compute_mahalanobis=True, progress=False),
        mir.predict(X_new, compute_mahalanobis=True, progress=False),
    )


# Measured bit-identical on the reference platform (numpy 2.4.4, jax 0.10.0,
# mellon 1.7.1, CPU); the tolerance is here so a different BLAS or an
# accelerator does not turn an equivalence into a flake.
_MIRROR_TOL = dict(rtol=1e-6, atol=1e-8)


def test_the_mirror_is_a_relabelling_of_the_same_fit():
    """`de(X, Y, "condition1")` and `de(Y, X, "condition2")` fit the same GPs.

    Both schemes put a single shared length scale on both models; forward takes
    it from X (condition 1 there) and the mirror takes it from X as well (X is
    condition 2 there).  So the same GP is fitted to X and the same GP to Y,
    with the model1 / model2 *roles* exchanged.
    """
    fwd, mir, rf, rm = _mirror_pair()

    assert float(fwd.model1.ls) == pytest.approx(float(fwd.model2.ls))
    assert float(mir.model1.ls) == pytest.approx(float(mir.model2.ls))
    assert float(fwd.model1.ls) == pytest.approx(float(mir.model1.ls))

    # the per-condition surfaces, with the roles exchanged
    np.testing.assert_allclose(
        rf["condition1_smoothed"], rm["condition2_smoothed"], **_MIRROR_TOL
    )
    np.testing.assert_allclose(
        rf["condition2_smoothed"], rm["condition1_smoothed"], **_MIRROR_TOL
    )
    np.testing.assert_allclose(
        rf["condition1_std"], rm["condition2_std"], **_MIRROR_TOL
    )
    np.testing.assert_allclose(
        rf["condition2_std"], rm["condition1_std"], **_MIRROR_TOL
    )


def test_the_mirror_flips_the_sign_of_every_signed_quantity():
    """Fold change is `condition2 - condition1`, so exchanging the labels negates it.

    This is the half of the equivalence that is *not* an equality, and stating
    it as a sign flip rather than weakening the test to "both ran" is the point.
    """
    _, _, rf, rm = _mirror_pair()

    # positive control: the fold change is not ~0, so the flip below is a real
    # statement and not two zero arrays agreeing.
    assert np.abs(np.asarray(rf["fold_change"])).max() > 1e-2
    assert not np.allclose(rf["fold_change"], rm["fold_change"])

    for key in ("fold_change", "mean_log_fold_change", "fold_change_zscores"):
        np.testing.assert_allclose(
            np.asarray(rf[key]), -np.asarray(rm[key]), **_MIRROR_TOL
        )


def test_the_mirror_leaves_the_unsigned_statistics_unchanged():
    """Mahalanobis distance and its tail probability are blind to the exchange.

    The fold change enters the Mahalanobis form quadratically and the
    denominator is `cov1 + cov2`, which is symmetric in the two models -- so
    the sign flip above cancels and the gene-level statistics are identical.
    This is what licenses reading a published `condition1` swap-invariance
    number as a `condition2` number under label exchange.
    """
    _, _, rf, rm = _mirror_pair()

    assert np.asarray(rf["mahalanobis_distances"]).max() > 0
    np.testing.assert_allclose(
        rf["mahalanobis_distances"], rm["mahalanobis_distances"], **_MIRROR_TOL
    )
    np.testing.assert_allclose(
        rf["neg_log10_ptp"], rm["neg_log10_ptp"], **_MIRROR_TOL
    )


def test_the_mirror_holds_with_landmarks_when_the_landmarks_are_shared():
    """The equivalence survives the Nystrom path -- given the *same* landmarks.

    `fit()` derives automatic landmarks from `np.vstack([X_condition1,
    X_condition2])`, whose ROW ORDER differs between the two orientations, so
    the automatic path evaluates the two runs at different points.  That is a
    pre-existing, scheme-independent asymmetry; no `param_scheme` removes it.  Pass
    one landmark array to both runs and the equivalence is exact again.
    """
    from mellon.parameters import compute_landmarks

    X1, _, X2, _ = _data()
    shared = np.asarray(
        compute_landmarks(
            np.vstack([X1, X2]), gp_type="fixed", n_landmarks=40, random_state=7
        )
    )
    _, _, rf, rm = _mirror_pair(n_landmarks=40, landmarks=shared)

    np.testing.assert_allclose(
        rf["condition1_smoothed"], rm["condition2_smoothed"], **_MIRROR_TOL
    )
    np.testing.assert_allclose(
        np.asarray(rf["fold_change"]), -np.asarray(rm["fold_change"]), **_MIRROR_TOL
    )
    np.testing.assert_allclose(
        rf["mahalanobis_distances"], rm["mahalanobis_distances"], **_MIRROR_TOL
    )


# --------------------------------------------------------------------------
# every reader of `param_scheme`, not just the writer
# --------------------------------------------------------------------------


def test_param_scheme_is_readable_back_out_of_a_stored_params_dict():
    """`param_scheme` is in the run-parameter match list; it must also be READABLE.

    `build_params_dict` stores it nested under `params["gp"]`, and
    `_check_overwrites` compares it through `params_get`.  `params_get`
    resolves a nested key via `_LEGACY_MAP`; a field missing from that map
    reads back as `None`, so every rerun -- including an identical one --
    compared unequal and was reported as a parameter change.
    """
    from kompot.anndata.utils.params import build_params_dict, params_get

    for scheme in PARAM_SCHEMES:
        params = build_params_dict(
            {"groupby": "g", "condition1": "A", "condition2": "B"},
            gp=GPSettings(param_scheme=scheme),
        )
        assert params["gp"]["param_scheme"] == scheme
        assert params_get(params, "param_scheme") == scheme


def test_the_de_cli_routes_param_scheme_into_gp_settings():
    """A flat config key the CLI does not recognise is not rejected.

    It falls through to `**function_kwargs` and is handed to mellon, so an
    omission from the routing set is silent at the config surface.  `landmarks`
    is deliberately absent -- it is an array, not a scalar config value.
    """
    import dataclasses

    from kompot.cli.de import GP_CONFIG_KEYS

    assert "param_scheme" in GP_CONFIG_KEYS
    fields = {f.name for f in dataclasses.fields(GPSettings)}
    assert GP_CONFIG_KEYS <= fields, GP_CONFIG_KEYS - fields
    assert fields - GP_CONFIG_KEYS == {"landmarks"}


class _CaptureWarnings(logging.Handler):
    """Collect WARNING records straight off the `kompot` logger.

    `caplog` cannot see them: `kompot/__init__.py` configures the logger with
    `"propagate": False`, so records never reach the root handler pytest
    installs.  Using `caplog` here does not fail loudly -- it yields an EMPTY
    record list, which makes a "did not warn" assertion pass for a reason that
    has nothing to do with the code under test.  The three negative tests below
    were vacuous until the positive one flushed this out.
    """

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@contextlib.contextmanager
def _captured_kompot_warnings():
    logger = logging.getLogger("kompot")
    handler = _CaptureWarnings()
    previous = logger.level
    logger.addHandler(handler)
    logger.setLevel(min(previous or logging.WARNING, logging.WARNING))
    try:
        yield handler.messages
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)


def test_the_warning_capture_helper_actually_captures():
    """Positive control for the helper itself, so the negatives below mean something."""
    with _captured_kompot_warnings() as messages:
        logging.getLogger("kompot").warning("canary: same landmarks")
    assert any("same landmarks" in m for m in messages)


def test_condition2_warns_when_the_landmarks_are_not_shared():
    """The equivalence's precondition must be audible at runtime, not just in docs.

    `"condition2"` exists only to be compared against `"condition1"` on the
    swapped orientation, and that comparison is exact only when both runs
    evaluate at the same landmarks.  Automatic landmarks are order-dependent, so
    the default configuration (`n_landmarks=5000`, `landmarks=None`) degrades the
    equivalence to approximate with nothing downstream to signal it.  Measured
    against the tolerance asserted above, six of the nine mirrored comparisons
    miss in EVERY draw and the totals ran 6 to 9 of 9 at the
    `random_state=None` default, where the landmark draw is not even
    reproducible between runs.  A boundary that lives only in prose is
    invisible to the people it protects.
    """
    X1, y1, X2, y2 = _data()

    with _captured_kompot_warnings() as messages:
        DifferentialExpression(n_landmarks=20, random_state=0).fit(
            X1, y1, X2, y2, param_scheme="condition2"
        )
    assert any("same landmarks" in m for m in messages), messages


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param(dict(n_landmarks=0), id="no-landmarks"),
        pytest.param(dict(n_landmarks=20, shared=True), id="landmarks-shared"),
    ],
)
def test_condition2_does_not_warn_when_the_equivalence_is_exact(kwargs):
    """The other half: a warning that always fires teaches nothing."""
    from mellon.parameters import compute_landmarks

    X1, y1, X2, y2 = _data()
    kwargs = dict(kwargs)
    landmarks = None
    if kwargs.pop("shared", False):
        landmarks = np.asarray(
            compute_landmarks(
                np.vstack([X1, X2]), gp_type="fixed", n_landmarks=20, random_state=7
            )
        )

    with _captured_kompot_warnings() as messages:
        DifferentialExpression(random_state=0, **kwargs).fit(
            X1, y1, X2, y2, param_scheme="condition2", landmarks=landmarks
        )
    assert not [m for m in messages if "same landmarks" in m]


def test_the_other_schemes_do_not_warn_about_landmarks():
    """Only `condition2` makes the cross-orientation comparison its purpose."""
    X1, y1, X2, y2 = _data()
    for scheme in ("condition1", "symmetric", "pooled", "separate"):
        with _captured_kompot_warnings() as messages:
            DifferentialExpression(n_landmarks=20, random_state=0).fit(
                X1, y1, X2, y2, param_scheme=scheme
            )
        assert not [m for m in messages if "same landmarks" in m], scheme


# ==========================================================================
# differential ABUNDANCE: the same field covers d, mu and ls
# ==========================================================================

import warnings  # noqa: E402

import anndata  # noqa: E402

import kompot  # noqa: E402
from kompot.differential.differential_abundance import (  # noqa: E402
    DifferentialAbundance,
    _condition_density_params,
    _resolve_scheme_density_params,
)
from kompot.settings import OutputSettings  # noqa: E402

# Separated enough that the fold change is far from zero, so every equality
# below compares arrays that COULD differ.
_DA_SEP, _DA_SCALE = 2.5, 0.45


def _da_data(n1=90, n2=60, seed=0):
    rng = np.random.default_rng(seed)
    X1 = rng.normal(loc=[-_DA_SEP, 0.0], scale=_DA_SCALE, size=(n1, 2))
    X2 = rng.normal(loc=[+_DA_SEP, 0.0], scale=_DA_SCALE, size=(n2, 2))
    return X1, X2


def _da_fit(X1, X2, **fit_kwargs):
    """Fit with no landmarks, so the only orientation effect is the scheme's."""
    model = DifferentialAbundance(n_landmarks=None, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        model.fit(X1, X2, **fit_kwargs)
    return model


def _da_predict(model, X_new):
    return model.predict(X_new, progress=False)


def _shared(model):
    p1, p2 = model.density_predictor1, model.density_predictor2
    return (
        (float(p1.d), float(p1.mu), float(p1.cov_func.ls)),
        (float(p2.d), float(p2.mu), float(p2.cov_func.ls)),
    )


# -- defaults and the deprecated alias --------------------------------------


def test_da_default_is_separate():
    X1, X2 = _da_data()
    Xn = np.vstack([X1, X2])
    a = _da_predict(_da_fit(X1, X2), Xn)
    b = _da_predict(_da_fit(X1, X2, param_scheme="separate"), Xn)
    for key in ("log_density_condition1", "log_density_condition2", "log_fold_change"):
        np.testing.assert_array_equal(a[key], b[key])


@pytest.mark.parametrize("sync, scheme", [(True, "pooled"), (False, "separate")])
def test_sync_parameters_is_an_exact_alias(sync, scheme):
    """The shipped spelling and the new one are the same computation, to the bit."""
    X1, X2 = _da_data()
    Xn = np.vstack([X1, X2])
    model = DifferentialAbundance(n_landmarks=None, random_state=0)
    with pytest.warns(DeprecationWarning, match="param_scheme"):
        model.fit(X1, X2, sync_parameters=sync)
    old = _da_predict(model, Xn)
    new = _da_predict(_da_fit(X1, X2, param_scheme=scheme), Xn)
    for key in (
        "log_density_condition1",
        "log_density_condition2",
        "log_fold_change",
        "log_fold_change_uncertainty",
    ):
        np.testing.assert_array_equal(old[key], new[key])


def test_pooled_and_separate_differ():
    """Positive control for the alias test: the two targets are distinguishable."""
    X1, X2 = _da_data()
    Xn = np.vstack([X1, X2])
    a = _da_predict(_da_fit(X1, X2, param_scheme="pooled"), Xn)
    b = _da_predict(_da_fit(X1, X2, param_scheme="separate"), Xn)
    assert np.abs(a["log_fold_change"] - b["log_fold_change"]).max() > 1e-2


def test_sync_parameters_and_param_scheme_together_raise():
    X1, X2 = _da_data()
    with pytest.raises(ValueError, match="not both"):
        DifferentialAbundance().fit(X1, X2, sync_parameters=True, param_scheme="pooled")


def test_da_rejects_unknown_scheme():
    X1, X2 = _da_data()
    with pytest.raises(ValueError, match="param_scheme"):
        DifferentialAbundance().fit(X1, X2, param_scheme="nope")


# -- what each scheme shares ------------------------------------------------


def test_da_separate_does_not_share():
    X1, X2 = _da_data()
    s1, s2 = _shared(_da_fit(X1, X2, param_scheme="separate"))
    assert s1[0] != s2[0] and s1[1] != s2[1] and s1[2] != s2[2]


@pytest.mark.parametrize("scheme", ["pooled", "condition1", "condition2", "symmetric"])
def test_da_sharing_schemes_share_d_mu_and_ls(scheme):
    X1, X2 = _da_data()
    s1, s2 = _shared(_da_fit(X1, X2, param_scheme=scheme))
    assert s1 == s2


@pytest.mark.parametrize("scheme, side", [("condition1", 0), ("condition2", 1)])
def test_da_one_sided_scheme_shares_that_conditions_own_estimate(scheme, side):
    """The shared values are what that condition's estimator reaches on its own.

    Load-bearing for the mirror below: the scheme resolves eagerly, and it is
    the same computation as ``"separate"`` for the condition it names only if
    the eager estimate equals the estimator's internal one to the last bit.
    """
    X1, X2 = _da_data()
    Xn = np.vstack([X1, X2])
    sep = _da_fit(X1, X2, param_scheme="separate")
    one = _da_fit(X1, X2, param_scheme=scheme)
    assert _shared(one)[side] == _shared(sep)[side]
    key = ("log_density_condition1", "log_density_condition2")[side]
    np.testing.assert_array_equal(_da_predict(one, Xn)[key], _da_predict(sep, Xn)[key])


@pytest.mark.parametrize("scheme", PARAM_SCHEMES)
def test_da_explicit_values_pin_every_scheme(scheme):
    X1, X2 = _da_data()
    s1, s2 = _shared(_da_fit(X1, X2, param_scheme=scheme, d=2.0, mu=-5.0, ls=3.0))
    assert s1 == s2 == (2.0, -5.0, 3.0)


_ONE_SIDED = [("condition1", 0), ("condition2", 1)]


@pytest.mark.parametrize("scheme, side", _ONE_SIDED)
def test_da_one_sided_scheme_derives_mu_at_a_pinned_d(scheme, side):
    """Pinning ``d`` ALONE must reach ``mu``, which the estimator derives from ``d``.

    Reference: the unshared fit with the same pin, i.e. what that condition's
    estimator does on its own.  Pinning d, mu and ls together (the test above)
    cannot see this, because the pinned mu hides the derived one.
    """
    X1, X2 = _da_data()
    sep = _shared(_da_fit(X1, X2, param_scheme="separate", d=2.0))[side]
    got = _shared(_da_fit(X1, X2, param_scheme=scheme, d=2.0))
    assert got[0] == got[1]
    assert got[side] == sep
    assert got[side][0] == 2.0
    # control: the pin moves mu, so equality above is not the unpinned value
    assert got[side][1] != _shared(_da_fit(X1, X2, param_scheme=scheme))[side][1]


def test_da_symmetric_derives_mu_at_a_pinned_d():
    from mellon.parameters import compute_mu

    X1, X2 = _da_data()
    s1, s2 = _shared(_da_fit(X1, X2, param_scheme="symmetric", d=2.0))
    assert s1 == s2
    nn1 = _condition_density_params(X1, 10.0, 42)[1]
    nn2 = _condition_density_params(X2, 10.0, 42)[1]
    assert s1[0] == 2.0
    assert s1[1] == float(compute_mu(np.concatenate([nn1, nn2]), 2.0))
    unpinned = _resolve_scheme_density_params("symmetric", X1, X2, 10.0, 42)
    assert s1[2] == float(unpinned["ls"])  # ls does not depend on d
    assert s1[1] != float(unpinned["mu"])  # control: the pin moves mu


@pytest.mark.parametrize("scheme, side", _ONE_SIDED)
def test_da_one_sided_scheme_with_ls_pinned_alone(scheme, side):
    """Control for the d-pin tests: ``ls`` does not enter d or mu."""
    X1, X2 = _da_data()
    sep = _shared(_da_fit(X1, X2, param_scheme="separate", ls=3.0))[side]
    got = _shared(_da_fit(X1, X2, param_scheme=scheme, ls=3.0))
    assert got[0] == got[1]
    assert got[side] == sep
    assert got[side][2] == 3.0


def test_da_symmetric_with_ls_pinned_alone():
    X1, X2 = _da_data()
    s1, s2 = _shared(_da_fit(X1, X2, param_scheme="symmetric", ls=3.0))
    unpinned = _resolve_scheme_density_params("symmetric", X1, X2, 10.0, 42)
    assert s1 == s2 == (float(unpinned["d"]), float(unpinned["mu"]), 3.0)


# -- swap behaviour ---------------------------------------------------------


def test_da_mirror_is_a_relabelling_of_the_same_fit():
    """``da(X, Y, "condition1")`` and ``da(Y, X, "condition2")``, labels exchanged."""
    X1, X2 = _da_data()
    Xn = np.vstack([X1, X2])
    fwd = _da_predict(_da_fit(X1, X2, param_scheme="condition1"), Xn)
    mir = _da_predict(_da_fit(X2, X1, param_scheme="condition2"), Xn)
    assert np.abs(fwd["log_fold_change"]).max() > 1e-2
    np.testing.assert_array_equal(fwd["log_density_condition1"], mir["log_density_condition2"])
    np.testing.assert_array_equal(fwd["log_density_condition2"], mir["log_density_condition1"])
    np.testing.assert_array_equal(fwd["log_fold_change"], -mir["log_fold_change"])
    np.testing.assert_array_equal(
        fwd["log_fold_change_uncertainty"], mir["log_fold_change_uncertainty"]
    )


def test_da_one_sided_schemes_are_not_swap_invariant():
    """Characterisation: ``condition1`` is asymmetric by design."""
    X1, X2 = _da_data()
    fwd = _shared(_da_fit(X1, X2, param_scheme="condition1"))[0]
    rev = _shared(_da_fit(X2, X1, param_scheme="condition1"))[0]
    assert fwd != rev


@pytest.mark.parametrize("n1, n2", [(90, 60), (700, 300)])
def test_da_symmetric_is_bit_identical_under_a_swap(n1, n2):
    """Swap-invariant by construction, including above the 500-cell ``d`` subsample.

    Positive control: ``"pooled"``'s ``d`` on the same data IS order-dependent
    above 500 combined cells, so the size is one where a union-based rule fails.
    """
    X1, X2 = _da_data(n1, n2)
    fwd = _resolve_scheme_density_params("symmetric", X1, X2, 10.0, 42)
    rev = _resolve_scheme_density_params("symmetric", X2, X1, 10.0, 42)
    assert fwd == rev
    # per-condition references straight from the estimator recipe, so this test
    # reads nothing of the one-sided schemes it is not about
    one = _condition_density_params(X1, 10.0, 42)[0]
    two = _condition_density_params(X2, 10.0, 42)[0]
    for name in ("d", "mu", "ls"):
        assert fwd[name] != one[name] and fwd[name] != two[name], name
        assert min(one[name], two[name]) <= fwd[name] <= max(one[name], two[name]), name
    if n1 + n2 > 500:
        from mellon.parameters import compute_d_factal

        assert compute_d_factal(np.vstack([X1, X2])) != compute_d_factal(
            np.vstack([X2, X1])
        )


def test_da_symmetric_swap_is_exact_end_to_end():
    X1, X2 = _da_data()
    Xn = np.vstack([X1, X2])
    fwd = _da_predict(_da_fit(X1, X2, param_scheme="symmetric"), Xn)
    rev = _da_predict(_da_fit(X2, X1, param_scheme="symmetric"), Xn)
    assert np.abs(fwd["log_fold_change"]).max() > 1e-2
    np.testing.assert_array_equal(fwd["log_fold_change"], -rev["log_fold_change"])
    np.testing.assert_array_equal(
        fwd["log_fold_change_uncertainty"], rev["log_fold_change_uncertainty"]
    )


# -- kompot.da(): GPSettings is honoured ------------------------------------


def _adata(n1=90, n2=60):
    X1, X2 = _da_data(n1, n2)
    X = np.vstack([X1, X2])
    ad = anndata.AnnData(X=np.zeros((X.shape[0], 1), dtype=np.float32))
    ad.obs["cond"] = ["A"] * n1 + ["B"] * n2
    ad.obsm["X"] = X
    return ad


def _da_lfc(gp=None, **kwargs):
    result, _ = kompot.da(
        _adata(),
        "cond",
        "A",
        "B",
        obsm_key="X",
        gp=gp,
        output=OutputSettings(copy=True, return_full_results=True, progress=False),
        **kwargs,
    )
    return result["table"]["lfc"].to_numpy()


_NO_LM = dict(n_landmarks=None, random_state=0)


def test_da_honours_gp_ls():
    """Regression: ``da()`` used to read six GPSettings fields and drop ``ls``.

    The control is the same length scale passed as a density keyword, which has
    always reached mellon.  If ``gp.ls`` were still ignored, the first result
    would equal the default run, not the control.
    """
    default = _da_lfc(GPSettings(**_NO_LM))
    control = _da_lfc(GPSettings(**_NO_LM), ls=0.05)
    via_gp = _da_lfc(GPSettings(ls=0.05, **_NO_LM))
    assert np.abs(control - default).max() > 1e-1
    np.testing.assert_array_equal(via_gp, control)


def test_da_honours_gp_param_scheme():
    separate = _da_lfc(GPSettings(**_NO_LM))
    pooled = _da_lfc(GPSettings(param_scheme="pooled", **_NO_LM))
    with pytest.warns(DeprecationWarning, match="param_scheme"):
        legacy = _da_lfc(GPSettings(**_NO_LM), sync_parameters=True)
    assert np.abs(pooled - separate).max() > 1e-2
    np.testing.assert_array_equal(pooled, legacy)


def test_da_rejects_ls_given_twice_and_scheme_given_twice():
    with pytest.raises(ValueError, match="pass it once"):
        _da_lfc(GPSettings(ls=0.05, **_NO_LM), ls=0.05)
    with pytest.raises(ValueError, match="not both"):
        _da_lfc(GPSettings(param_scheme="pooled", **_NO_LM), sync_parameters=True)


def test_param_scheme_as_a_flat_keyword_is_refused_clearly():
    """It is a GPSettings field; as a keyword it used to die in a TypeError."""
    with pytest.raises(ValueError, match=r"gp=GPSettings\(param_scheme"):
        _da_lfc(GPSettings(**_NO_LM), param_scheme="pooled")
    with pytest.raises(ValueError, match=r"gp=GPSettings\(param_scheme"):
        kompot.de(_adata(), "cond", "A", "B", obsm_key="X", param_scheme="pooled")


def test_da_warns_about_expression_only_gp_fields():
    with _captured_kompot_warnings() as messages:
        _da_lfc(GPSettings(sigma=0.5, **_NO_LM))
    assert any("have no effect in da()" in m for m in messages), messages
    with _captured_kompot_warnings() as messages:
        _da_lfc(GPSettings(**_NO_LM))
    assert not [m for m in messages if "have no effect in da()" in m]


def test_the_da_cli_routes_param_scheme_and_ls_into_gp_settings():
    """Every GPSettings field the DA model uses is routed; the rest fall through."""
    import dataclasses

    from kompot.cli.da import GP_CONFIG_KEYS

    fields = {f.name for f in dataclasses.fields(GPSettings)}
    assert {"param_scheme", "ls"} <= GP_CONFIG_KEYS <= fields
    assert fields - GP_CONFIG_KEYS == {"sigma", "eps", "use_empirical_variance"}
