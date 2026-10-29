"""``density_normalization="common_d"``: correct the DA normalization bias while each
condition keeps its own density parameters (settylab/dotto-nexus#455).

mellon's nearest-neighbour data term is ``ln rho_hat = -d ln r - ln c_d``. Under
``param_scheme="separate"`` the two conditions use different ``d``, so the standard
normalization (``- ln n``) compares them in different units, ``length^-d1`` vs
``length^-d2``: rescaling the embedding by ``k`` shifts the lfc by ``-(d2 - d1) ln k``.
And when ``d`` is below the exponent ``d_n`` with which nearest-neighbour distances
scale with cell count, ``- ln n`` leaves ``(d/d_n - 1) ln(n2/n1)`` at no true change.

``"common_d"`` maps each condition's own fit to one reference dimension ``d*`` before
subtracting ``ln n``. Both conditions then shift by ``-d* ln k`` (unit-free for any
``d*``), and with ``d* = d_n`` (the default) the size term vanishes.

Known-answer data: one 2-D Gaussian (``d0 = 2``) linearly embedded in 5-D; both
conditions drawn from ONE distribution, so the true lfc is 0 everywhere.
"""

import numpy as np
import pytest

from kompot.differential.differential_abundance import (
    DENSITY_NORMALIZATIONS,
    DifferentialAbundance,
    _log_unit_ball_volume,
    nn_scaling_dimension,
    to_reference_dimension,
)

# The known-answer tolerance of the original test (#455); not loosened.
NO_DA_TOL = 0.15
KNOWN_ANSWER_SEEDS = (0, 1, 2, 3, 4)


def _no_da_data(seed=0, n1=600, n2=200, D=5):
    """One 2-D Gaussian, linearly embedded in D dimensions; no DA."""
    rng = np.random.default_rng(seed)
    A = np.linalg.qr(rng.normal(size=(D, 2)))[0].T
    X = rng.normal(size=(n1 + n2, 2)) @ A
    return X[:n1], X[n1:]


def _fit(X1, X2, k=1.0, scheme="separate", norm=None, **kw):
    m = DifferentialAbundance(n_landmarks=None, random_state=0)
    m.fit(k * X1, k * X2, param_scheme=scheme, density_normalization=norm, **kw)
    r = m.predict(k * np.vstack([X1, X2]), progress=False)
    return m, r


def _lfc(*a, **kw):
    return np.asarray(_fit(*a, **kw)[1]["log_fold_change"])


@pytest.fixture(scope="module")
def no_da():
    return _no_da_data()


# ---- the map itself ----------------------------------------------------------


def test_unit_ball_volume_matches_mellon_constant():
    for d in (1.0, 1.47, 2.0, 3.0):
        # mellon.inference._nearest_neighbors: const = d ln(pi)/2 - gammaln(d/2 + 1)
        from scipy.special import gammaln

        assert _log_unit_ball_volume(d) == pytest.approx(
            d * np.log(np.pi) / 2 - gammaln(d / 2 + 1)
        )
    assert _log_unit_ball_volume(2.0) == pytest.approx(np.log(np.pi))


def test_to_reference_dimension_maps_the_data_term_exactly():
    """-d ln r - ln c_d  ->  -d* ln r - ln c_d*, for any r."""
    r = np.array([1e-4, 0.03, 1.0, 7.0])
    for d_own, d_ref in ((1.47, 2.0), (2.0, 1.2), (1.8, 1.8)):
        f_own = -d_own * np.log(r) - _log_unit_ball_volume(d_own)
        f_ref = -d_ref * np.log(r) - _log_unit_ball_volume(d_ref)
        np.testing.assert_allclose(to_reference_dimension(f_own, d_own, d_ref), f_ref, rtol=1e-12)


# ---- the units law -----------------------------------------------------------


def test_standard_separate_lfc_carries_the_units_term(no_da):
    """Potency control: the defect is present under 0.9.0's normalization."""
    X1, X2 = no_da
    m, r = _fit(X1, X2)
    d1, d2 = float(m.density_predictor1.d), float(m.density_predictor2.d)
    assert abs(d2 - d1) > 0.01  # else no test here can see the law
    k = 1000.0
    shift = _lfc(X1, X2, k=k) - np.asarray(r["log_fold_change"])
    np.testing.assert_allclose(np.median(shift), -(d2 - d1) * np.log(k), rtol=1e-2, atol=1e-3)
    assert abs(np.median(shift)) > 0.05


def test_common_d_lfc_is_unit_free_with_per_condition_d(no_da):
    X1, X2 = no_da
    m, r = _fit(X1, X2, norm="common_d")
    info = m.normalization_info
    # each condition kept its own d: the fit is the separate one
    assert abs(info["d_condition2"] - info["d_condition1"]) > 0.01
    base = np.asarray(r["log_fold_change"])
    for k in (0.01, 1000.0):
        mk, rk = _fit(X1, X2, k=k, norm="common_d")
        assert mk.normalization_info["reference_d"] == pytest.approx(info["reference_d"], rel=1e-9)
        np.testing.assert_allclose(np.asarray(rk["log_fold_change"]), base, atol=1e-3)


def test_nn_scaling_dimension_is_scale_invariant_and_near_d0():
    X1, X2 = _no_da_data(n1=1500, n2=1500)
    X = np.vstack([X1, X2])
    a = nn_scaling_dimension(X, 500, 2000, seed=0)
    b = nn_scaling_dimension(1000.0 * X, 500, 2000, seed=0)
    assert a["d"] == pytest.approx(b["d"], rel=1e-9)
    # a 2-D Gaussian: E ln r_nn scales as -(1/2) ln n
    assert a["d"] == pytest.approx(2.0, abs=0.3)


# ---- the size term: no-DA known answer at the original tolerance -------------


@pytest.mark.parametrize("n1,n2", [(600, 200), (200, 600), (400, 400)])
def test_common_d_no_da_offset_is_zero(n1, n2):
    """Per-condition d, mu, ls (separate); true lfc 0; |median lfc| < 0.15 (seed mean)."""
    meds = []
    for seed in KNOWN_ANSWER_SEEDS:
        X1, X2 = _no_da_data(seed, n1, n2)
        meds.append(float(np.median(_lfc(X1, X2, norm="common_d"))))
    assert abs(np.mean(meds)) < NO_DA_TOL, meds


def test_a_low_reference_d_reintroduces_the_size_term():
    """Potency control for the known-answer test: d* = 1.5 < d0 = 2 leaves
    (d*/d0 - 1) ln(n2/n1) = +0.27 at 600/200, which the tolerance must catch."""
    meds = []
    for seed in KNOWN_ANSWER_SEEDS:
        X1, X2 = _no_da_data(seed, 600, 200)
        meds.append(float(np.median(_lfc(X1, X2, norm="common_d", reference_d=1.5))))
    assert np.mean(meds) > NO_DA_TOL, meds


# ---- off means off -----------------------------------------------------------


def test_standard_is_the_default_and_matches_normalize_true(no_da):
    X1, X2 = no_da
    m0, r0 = _fit(X1, X2)
    _, r1 = _fit(X1, X2, norm="standard")
    assert m0.density_normalization == "standard" and m0.normalization_info is None
    for key in ("log_fold_change", "log_fold_change_zscore", "log_fold_change_uncertainty"):
        np.testing.assert_array_equal(np.asarray(r0[key]), np.asarray(r1[key]))
    Xn = np.vstack([X1, X2])
    expect = np.asarray(m0.density_predictor2(Xn, normalize=True)) - np.asarray(
        m0.density_predictor1(Xn, normalize=True)
    )
    np.testing.assert_array_equal(np.asarray(r0["log_fold_change"]), expect)


def test_common_d_at_the_shared_d_reproduces_standard(no_da):
    """With d1 = d2 = d* the map is the identity: common_d == standard."""
    X1, X2 = no_da
    m, r_std = _fit(X1, X2, scheme="symmetric")
    d_sym = float(m.density_predictor1.d)
    _, r_cd = _fit(X1, X2, scheme="symmetric", norm="common_d", reference_d=d_sym)
    for key in ("log_fold_change", "log_fold_change_uncertainty"):
        np.testing.assert_allclose(np.asarray(r_cd[key]), np.asarray(r_std[key]), rtol=1e-10, atol=1e-10)


def test_common_d_scales_the_variance(no_da):
    X1, X2 = no_da
    m_std, r_std = _fit(X1, X2)
    m, r = _fit(X1, X2, norm="common_d", reference_d=2.0)
    s1 = 2.0 / m.normalization_info["d_condition1"]
    s2 = 2.0 / m.normalization_info["d_condition2"]
    Xn = np.vstack([X1, X2])
    v1 = np.asarray(m_std.density_predictor1.uncertainty(Xn))
    v2 = np.asarray(m_std.density_predictor2.uncertainty(Xn))
    np.testing.assert_allclose(
        np.asarray(r["log_fold_change_uncertainty"]), s1**2 * v1 + s2**2 * v2, rtol=1e-6
    )


# ---- validation and recording ------------------------------------------------


def test_invalid_options_raise(no_da):
    X1, X2 = no_da
    assert DENSITY_NORMALIZATIONS == ("standard", "common_d")
    with pytest.raises(ValueError, match="density_normalization"):
        DifferentialAbundance(n_landmarks=None).fit(X1, X2, density_normalization="bogus")
    with pytest.raises(ValueError, match="reference_d is only used"):
        DifferentialAbundance(n_landmarks=None).fit(X1, X2, reference_d=2.0)
    with pytest.raises(ValueError, match="positive number"):
        DifferentialAbundance(n_landmarks=None).fit(
            X1, X2, density_normalization="common_d", reference_d=-1.0
        )


def test_normalization_info_records_the_measurement(no_da):
    X1, X2 = no_da
    m, _ = _fit(X1, X2, norm="common_d")
    info = m.normalization_info
    assert info["method"] == "common_d"
    assert info["reference_d_source"] == "nn_scaling"
    assert (info["n_condition1"], info["n_condition2"]) == (600, 200)
    assert info["d_condition1"] == pytest.approx(float(m.density_predictor1.d))
    ns = info["nn_scaling"]
    assert (ns["n_small"], ns["n_large"]) == (200, 600)
    assert info["reference_d"] == pytest.approx(
        np.log(600 / 200) / (ns["mean_log_nn_small"] - ns["mean_log_nn_large"])
    )


def test_balanced_sizes_spread_the_scaling_sizes():
    X1, X2 = _no_da_data(n1=400, n2=400)
    m, _ = _fit(X1, X2, norm="common_d")
    ns = m.normalization_info["nn_scaling"]
    assert (ns["n_small"], ns["n_large"]) == (200, 400)


def test_da_wrapper_records_common_d_only_when_on():
    import anndata as ad
    import pandas as pd

    import kompot

    X1, X2 = _no_da_data(n1=300, n2=150)
    X = np.vstack([X1, X2])
    obs = pd.DataFrame(
        {"cond": pd.Categorical(["A"] * len(X1) + ["B"] * len(X2))},
        index=[f"c{i}" for i in range(len(X))],
    )

    def run(**kw):
        a = ad.AnnData(X=np.zeros((len(X), 1), np.float32), obs=obs.copy())
        a.obsm["emb"] = X
        res = kompot.da(
            a, "cond", "A", "B", obsm_key="emb",
            output=kompot.OutputSettings(return_full_results=True, progress=False),
            **kw,
        )
        return a, res

    a_std, res_std = run()
    a_cd, res_cd = run(density_normalization="common_d")
    assert res_cd["model"].normalization_info["method"] == "common_d"
    assert res_std["model"].normalization_info is None
    s_std = str(a_std.uns["kompot_da"])
    s_cd = str(a_cd.uns["kompot_da"])
    assert "density_normalization" not in s_std
    assert "common_d" in s_cd and "reference_d" in s_cd
