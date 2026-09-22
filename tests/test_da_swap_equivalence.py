"""Swap equivalence for differential ABUNDANCE.

`ls_scheme` and `tests/test_ls_scheme.py` are about differential EXPRESSION, where
the shared length scale was estimated from condition 1's cells alone and
`"condition2"` exists to expose that. Abundance is a separate code path and
inherits none of it, so the property is established here rather than assumed.

It holds for a structural reason worth stating: `DifferentialAbundance.fit` builds
BOTH density estimators from the same ``estimator_defaults`` dict --

    density_estimator_condition1 = mellon.DensityEstimator(**estimator_defaults)
    density_estimator_condition2 = mellon.DensityEstimator(**estimator_defaults)

-- with no ``ls_for_model2 = self.model1.ls`` analogue anywhere. There is no
condition-1 inheritance to be asymmetric about, so DA never had the defect
`ls_scheme` was added for, and no DA counterpart of `"condition2"` is warranted.

Two configurations DO break the exactness, and both are covered below: automatic
landmarks, and ``sync_parameters=True``. Both trace to the same ROOT as the DE
case -- ``np.vstack([X_condition1, X_condition2])`` is row-order dependent -- but
they act through DIFFERENT proximate mechanisms, and **a remedy applies at the
proximate level, not the root**: ``compute_landmarks`` (k-means init) for the
first, ``compute_nn_distances`` (which sets ``mu`` and ``ls``) for the second.
So sharing a landmark array fixes the first and cannot reach the second --
landmarks are not an input to ``compute_nn_distances``, and sharing them leaves
the ``sync_parameters`` half at 4.784e-03 against a 4.720e-03 no-landmark
reference.

The remedy for that half is to pass ``mu`` and ``ls`` explicitly -- **and ``d``
as well once the combined data exceeds 500 cells**, because
``compute_d_factal`` subsamples 500 INDICES above that (mellon
``parameters.py``) and so becomes row-order dependent itself. Measured:
``mu``+``ls`` alone is exact at 200+200 and NOT exact at 400+400 (6.085e-04,
with ``d`` reading 1.4972 vs 1.4983); all three explicit is exact at 400+400.
Passing all three is always correct.

Note the trap in that boundary. An earlier version of this file said ``d`` is
identical across orientations. That was measured at 160 combined cells and is
TRUE THERE -- and false above 500, which is most real datasets. A correct local
measurement generalised past the condition that made it true.
"""

import numpy as np
import pytest

from kompot.differential.differential_abundance import DifferentialAbundance

# Well separated so that |log_fold_change| clears the package's OWN default
# `log_fold_change_threshold = 1.0`. No threshold is tuned anywhere in this file:
# an earlier version of these tests used weaker separation, every cell came back
# 'neutral', and the direction check below passed while being vacuous.
_SEPARATION = 2.5
_SCALE = 0.45


def _data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X1 = rng.normal(loc=[-_SEPARATION, 0.0], scale=_SCALE, size=(n, 2))
    X2 = rng.normal(loc=[+_SEPARATION, 0.0], scale=_SCALE, size=(n, 2))
    return X1, X2


def _swap_pair(n_landmarks=None, sync_parameters=False, landmarks=None):
    """Fit both orientations and predict both on the same points."""
    X1, X2 = _data()
    X_new = np.vstack([X1, X2])

    fwd = DifferentialAbundance(n_landmarks=n_landmarks, random_state=0)
    fwd.fit(X1, X2, sync_parameters=sync_parameters, landmarks=landmarks)

    rev = DifferentialAbundance(n_landmarks=n_landmarks, random_state=0)
    rev.fit(X2, X1, sync_parameters=sync_parameters, landmarks=landmarks)

    return (
        fwd.predict(X_new, progress=False),
        rev.predict(X_new, progress=False),
    )


# Measured bit-identical on the reference platform at the DA defaults; the
# tolerance is here so a different BLAS or an accelerator does not turn an
# equivalence into a flake.
_TOL = dict(rtol=1e-6, atol=1e-8)


@pytest.fixture(scope="module")
def default_pair():
    return _swap_pair()


def test_the_swap_is_a_relabelling_of_the_same_densities(default_pair):
    """`da(X, Y)` and `da(Y, X)` fit the same density to each condition.

    Both estimators are constructed from one shared ``estimator_defaults``, and at
    the defaults nothing in that dict depends on the ORDER the conditions were
    passed in -- so each condition's density follows the condition, not the slot.
    """
    fwd, rev = default_pair
    np.testing.assert_allclose(
        fwd["log_density_condition1"], rev["log_density_condition2"], **_TOL
    )
    np.testing.assert_allclose(
        fwd["log_density_condition2"], rev["log_density_condition1"], **_TOL
    )


def test_the_swap_flips_the_sign_of_every_signed_quantity(default_pair):
    """`log_fold_change` is `condition2 - condition1`, so exchanging labels negates it."""
    fwd, rev = default_pair

    # positive control: the fold changes are not ~0, so the flip is a real claim
    assert np.abs(np.asarray(fwd["log_fold_change"])).max() > 1.0
    assert not np.allclose(fwd["log_fold_change"], rev["log_fold_change"])

    for key in ("log_fold_change", "log_fold_change_zscore"):
        np.testing.assert_allclose(
            np.asarray(fwd[key]), -np.asarray(rev[key]), **_TOL
        )


def test_the_swap_leaves_the_unsigned_statistics_unchanged(default_pair):
    """Uncertainty is `u1 + u2` and the PTP is `min(logcdf(z), logcdf(-z))`.

    Both are symmetric in the two conditions -- the first because addition is, the
    second because it is an even function of the z-score -- so the sign flip above
    cancels and these come back identical.
    """
    fwd, rev = default_pair
    np.testing.assert_allclose(
        fwd["log_fold_change_uncertainty"], rev["log_fold_change_uncertainty"], **_TOL
    )
    np.testing.assert_allclose(
        fwd["neg_log10_fold_change_ptp"], rev["neg_log10_fold_change_ptp"], **_TOL
    )


def test_the_swap_exchanges_the_direction_labels(default_pair):
    """`log_fold_change_direction` is categorical: 'up' and 'down' must EXCHANGE.

    This is the one abundance output with no differential-expression counterpart,
    so porting the DE test shape does not cover it.  The control is load-bearing
    rather than decorative: with a weaker density difference every cell comes back
    'neutral', and two all-'neutral' arrays agree perfectly *whether or not* the
    labels are swapped -- so the swap assertion passes while establishing nothing.
    Both halves are asserted here, and the unswapped comparison must DISAGREE.
    """
    fwd, rev = default_pair
    forward = np.asarray(fwd["log_fold_change_direction"])
    reverse = np.asarray(rev["log_fold_change_direction"])

    # control 1: both labels actually occur, so the swap has something to exchange
    assert (forward == "up").sum() > 0
    assert (forward == "down").sum() > 0

    swap = {"up": "down", "down": "up", "neutral": "neutral"}
    expected = np.array([swap[d] for d in reverse], dtype=object)
    assert np.array_equal(forward, expected)

    # control 2: the swap is not a no-op -- without it the two must disagree
    assert not np.array_equal(forward, reverse)


def test_the_significant_set_is_the_same_cells_either_way(default_pair):
    """Significance is `|log_fold_change| > threshold` and a PTP, both sign-blind.

    So the swap may relabel a cell's direction but must never change WHICH cells
    are called.
    """
    fwd, rev = default_pair
    fwd_sig = np.asarray(fwd["log_fold_change_direction"]) != "neutral"
    rev_sig = np.asarray(rev["log_fold_change_direction"]) != "neutral"
    assert fwd_sig.sum() > 0
    assert np.array_equal(fwd_sig, rev_sig)


# --------------------------------------------------------------------------
# where the exactness STOPS -- reported, not hidden
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, why",
    [
        pytest.param(
            dict(n_landmarks=40),
            "automatic landmarks",
            id="automatic-landmarks",
        ),
        pytest.param(
            dict(sync_parameters=True),
            "d / mu / ls synchronised from the stacked union",
            id="sync-parameters",
        ),
    ],
)
def test_the_swap_is_only_approximate_when_the_union_order_matters(kwargs, why):
    """Characterisation: two configurations break the exactness, for one reason.

    ``np.vstack([X_condition1, X_condition2])`` is row-order dependent. It feeds
    ``compute_landmarks`` always, and under ``sync_parameters=True`` it also feeds
    ``compute_d_factal`` / ``compute_mu`` / ``compute_ls``. Either way the two
    orientations are then computed from differently-ordered unions and agreement
    degrades -- silently, which is why it is pinned here.

    This is NOT asserting the gap is small. It asserts the direction of the
    finding: exact at the defaults, inexact here.
    """
    fwd, rev = _swap_pair(**kwargs)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(
            np.asarray(fwd["log_fold_change"]),
            -np.asarray(rev["log_fold_change"]),
            **_TOL,
        )

    # and the defaults, by contrast, ARE exact -- the comparison that gives the
    # statement above its meaning
    d_fwd, d_rev = _swap_pair()
    np.testing.assert_allclose(
        np.asarray(d_fwd["log_fold_change"]), -np.asarray(d_rev["log_fold_change"]), **_TOL
    )


def test_shared_landmarks_restore_the_exact_swap():
    """Share the landmarks, and the automatic-landmark half is exact again.

    Scope: this covers the LANDMARK half only -- ``sync_parameters`` is left at
    its default ``False`` here. The same remedy does not reach the
    ``sync_parameters`` half, and no test asserts that it does.
    """
    from mellon.parameters import compute_landmarks

    X1, X2 = _data()
    shared = np.asarray(
        compute_landmarks(
            np.vstack([X1, X2]), gp_type="fixed", n_landmarks=40, random_state=7
        )
    )
    fwd, rev = _swap_pair(n_landmarks=40, landmarks=shared)
    np.testing.assert_allclose(
        np.asarray(fwd["log_fold_change"]), -np.asarray(rev["log_fold_change"]), **_TOL
    )
    np.testing.assert_allclose(
        fwd["neg_log10_fold_change_ptp"], rev["neg_log10_fold_change_ptp"], **_TOL
    )
