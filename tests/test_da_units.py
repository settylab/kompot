"""DA must not depend on the length unit of the embedding (settylab/dotto-nexus#455).

With ``normalize=True`` each condition's log density is ``-d_i * ln r + ...`` in
the embedding's units. If the two conditions use different ``d`` (the
``"separate"`` scheme, where each condition's fractal ``d`` is estimated on its
own cells), rescaling every coordinate by ``k`` shifts the log-fold change by
``-(d2 - d1) * ln k``. A scheme that shares ``d``, ``mu`` and ``ls`` cancels the
unit exactly. The known-answer case: both conditions drawn from ONE
distribution at different sizes, so the true log-fold change is 0 everywhere.
"""

import logging

import numpy as np
import pytest

from kompot.differential.differential_abundance import (
    UNITS_TERM_WARN_NATS,
    DifferentialAbundance,
)


def _no_da_data(seed=0, n1=600, n2=200, D=5):
    """One 2-D Gaussian, linearly embedded in D dimensions; n1 != n2; no DA."""
    rng = np.random.default_rng(seed)
    A = np.linalg.qr(rng.normal(size=(D, 2)))[0].T
    X = rng.normal(size=(n1 + n2, 2)) @ A
    return X[:n1], X[n1:]


def _fit_lfc(X1, X2, scheme, k=1.0, **kw):
    m = DifferentialAbundance(n_landmarks=None, random_state=0)
    m.fit(k * X1, k * X2, param_scheme=scheme, **kw)
    r = m.predict(k * np.vstack([X1, X2]), progress=False)
    return m, np.asarray(r["log_fold_change"])


@pytest.fixture(scope="module")
def no_da():
    return _no_da_data()


def test_density_params_are_recorded(no_da):
    X1, X2 = no_da
    m, _ = _fit_lfc(X1, X2, "separate")
    p = m.density_params
    assert p["param_scheme"] == "separate"
    assert p["condition1"]["n_cells"] == len(X1)
    assert p["condition2"]["n_cells"] == len(X2)
    for c in ("condition1", "condition2"):
        assert set(p[c]) == {"d", "mu", "ls", "n_cells"}
        assert np.isfinite([p[c]["d"], p[c]["mu"], p[c]["ls"]]).all()
    # the recorded values are the estimators' own
    assert p["condition1"]["d"] == pytest.approx(float(np.mean(m.density_predictor1.d)))
    assert p["condition2"]["mu"] == pytest.approx(float(m.density_predictor2.mu))


def test_separate_lfc_shifts_by_minus_delta_d_ln_k(no_da):
    """The defect, measured: the shift is exactly the units law, nothing else."""
    X1, X2 = no_da
    m, base = _fit_lfc(X1, X2, "separate")
    d1 = m.density_params["condition1"]["d"]
    d2 = m.density_params["condition2"]["d"]
    assert abs(d2 - d1) > 0.01  # else the test cannot see the law
    k = 1000.0
    _, scaled = _fit_lfc(X1, X2, "separate", k=k)
    shift = scaled - base
    predicted = -(d2 - d1) * np.log(k)
    np.testing.assert_allclose(np.median(shift), predicted, rtol=1e-2, atol=1e-3)


def test_symmetric_lfc_is_unit_free(no_da):
    """The correction: sharing d, mu and ls cancels the length unit."""
    X1, X2 = no_da
    _, base = _fit_lfc(X1, X2, "symmetric")
    for k in (0.01, 1000.0):
        _, scaled = _fit_lfc(X1, X2, "symmetric", k=k)
        # measured max 0.0011 over 10 seeds; separate shifts by ~0.6 here
        np.testing.assert_allclose(scaled, base, atol=5e-3)


# Known answer, averaged over seeds: one seed's median lfc scatters by about
# 0.15 (measured over 10 seeds, settylab/dotto-nexus#455), so a single draw
# cannot carry an "offset ~ 0" claim.
_SEEDS = range(5)


def _median_lfcs(n1, n2, **kw):
    out = []
    for seed in _SEEDS:
        X1, X2 = _no_da_data(seed=seed, n1=n1, n2=n2)
        out.append(float(np.median(_fit_lfc(X1, X2, "symmetric", **kw)[1])))
    return np.array(out)


def test_known_answer_no_da_offset_near_zero_at_the_true_dimension():
    """No true DA, 600 vs 200 cells, shared d = the manifold's true dimension (2).

    normalize=True subtracts ln n, which is exact when d equals the exponent d0
    with which nearest-neighbour distances scale with n. Measured mean over
    seeds 0-4: +0.021 (10 seeds: -0.057, sd 0.14).
    """
    pinned = _median_lfcs(600, 200, d=2.0)
    assert abs(pinned.mean()) < 0.15


def test_shared_fractal_d_leaves_the_size_term():
    """Known limitation, pinned so a change to it is seen: the shared FRACTAL d.

    The fractal estimator reads below d0 (here about 1.47 for d0 = 2), so the
    shared-d lfc keeps the constant (d/d0 - 1) ln(n2/n1): about +0.28 at
    600/200 under the plug-in, +0.22 measured, zero at equal sizes. It is
    unit-free and one constant per comparison, unlike the units term of
    "separate". Measured seeds 0-4: +0.271 at 600/200, -0.005 at 400/400.
    """
    fractal = _median_lfcs(600, 200)
    pinned = _median_lfcs(600, 200, d=2.0)
    balanced = _median_lfcs(400, 400)
    assert (fractal > 0.05).all() and fractal.mean() > 0.15
    assert abs(pinned.mean()) < fractal.mean() / 2
    assert abs(balanced.mean()) < 0.1


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@pytest.fixture
def warnings_seen():
    """kompot's logger does not propagate, so pytest's log capture sees nothing."""
    logger = logging.getLogger("kompot")
    handler, previous = _Capture(), logger.level
    logger.addHandler(handler)
    logger.setLevel(min(previous or logging.WARNING, logging.WARNING))
    try:
        yield handler.messages
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)


def test_the_capture_fixture_captures(warnings_seen):
    """Positive control, so the negatives below mean something."""
    logging.getLogger("kompot").warning("canary: units of the embedding")
    assert any("canary" in m for m in warnings_seen)


def test_separate_warns_when_the_units_term_is_large(no_da, warnings_seen):
    X1, X2 = no_da
    m, _ = _fit_lfc(X1, X2, "separate", k=0.01)
    assert abs(m.density_params["units_term"]) > UNITS_TERM_WARN_NATS
    assert any("depends on the units of the embedding" in w for w in warnings_seen)


@pytest.mark.parametrize("scheme", ["symmetric", "pooled", "condition1"])
def test_shared_schemes_do_not_warn(no_da, warnings_seen, scheme):
    X1, X2 = no_da
    m, _ = _fit_lfc(X1, X2, scheme, k=0.01)
    assert m.density_params["units_term"] == 0.0
    assert not [w for w in warnings_seen if "units of the embedding" in w]


def test_pinned_d_under_separate_does_not_warn(no_da, warnings_seen):
    X1, X2 = no_da
    m, _ = _fit_lfc(X1, X2, "separate", k=0.01, d=1.5)
    assert m.density_params["units_term"] == 0.0
    assert not [w for w in warnings_seen if "units of the embedding" in w]
