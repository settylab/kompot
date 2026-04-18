"""Variance-stratified residualization of Mahalanobis distances.

The raw per-gene Mahalanobis statistic ``D^2`` scales with per-gene
expression variance when ``use_empirical_variance=False`` (kompot's
default).  Because the gene-shuffled null inherits each gene's ``(mean,
variance)``, the null distribution of ``log(1 + D^2)`` is itself a
smooth monotone function of those two summaries.  Fitting that surface
on the null draws and subtracting it from the observed ``log(1 + D^2)``
gives a condition-specific test statistic whose null is centred at
zero with known scale — exactly the quantity kompot's 1-D local FDR
machinery wants.

See ``docs/variance_stratified_fdr.rst`` for the full derivation and
the Tal1 chimera validation.

The module is pure numpy / scipy and does not touch AnnData or JAX —
it can be called post-hoc on any ``(real_mahalanobis, null_mahalanobis,
null_gene_indices, per_gene_features)`` tuple.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger("kompot")


# ---------------------------------------------------------------------------
# Per-gene (log_mean, log_var) features
# ---------------------------------------------------------------------------


def compute_gene_features(
    expr_list: Sequence[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute ``(log_mean, log_var)`` for each gene on the stacked expression.

    Parameters
    ----------
    expr_list : sequence of 2-D arrays
        Each entry has shape ``(n_cells, n_genes)`` with the same
        ``n_genes``.  Expression matrices for the two (or more)
        conditions are stacked along the cell axis before computing the
        per-gene mean and variance.

    Returns
    -------
    log_mean, log_var : ndarray of shape ``(n_genes,)``
        ``log1p`` of per-gene mean and per-gene variance computed over
        all stacked cells.  Zero-variance genes get ``log_var = 0``.
    """
    arrays = [np.asarray(a, dtype=float) for a in expr_list]
    if not arrays:
        raise ValueError("expr_list must contain at least one expression matrix.")
    n_genes = arrays[0].shape[1]
    for a in arrays:
        if a.ndim != 2 or a.shape[1] != n_genes:
            raise ValueError(
                "All expression matrices must be 2-D with matching gene count."
            )
    stacked = np.vstack(arrays)

    mean = stacked.mean(axis=0)
    var = stacked.var(axis=0)

    # log1p handles zeros gracefully and compresses the dynamic range.
    return np.log1p(mean), np.log1p(np.maximum(var, 0.0))


# ---------------------------------------------------------------------------
# Null-trend fitting
# ---------------------------------------------------------------------------


def _design_poly3(m: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Degree-3 tensor design matrix with one cross term.

    Columns: ``[1, m, m^2, m^3, v, v^2, m*v]``.  Seven parameters, no
    regularisation — adequate for ``n_null >> 7`` which is always the
    case for kompot's default ``null_genes=2000``.
    """
    m = np.asarray(m, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    return np.column_stack([
        np.ones_like(m),
        m,
        m * m,
        m * m * m,
        v,
        v * v,
        m * v,
    ])


def _design_mean_only(m: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Mean-only design: ``[1, m, m^2, m^3]`` — ignores variance feature."""
    m = np.asarray(m, dtype=float).ravel()
    return np.column_stack([np.ones_like(m), m, m * m, m * m * m])


_DESIGN_BUILDERS = {
    "poly3": _design_poly3,
    "poly3_mean_only": _design_mean_only,
}


@dataclass
class NullTrend:
    """Fitted null-trend surface ``phi_hat(m, v)``.

    Attributes
    ----------
    coef : ndarray
        Regression coefficients solved via ordinary least squares on
        ``log1p(null_mahalanobis)``.
    sigma : float
        Homoscedastic residual standard deviation of the fit, used to
        standardise ``R_g`` to ``Z_g``.
    model : str
        Name of the design family.
    features : tuple of str
        Feature names in the order they appear in the design matrix
        inputs (``('log_mean', 'log_var')`` or ``('log_mean',)``).
    n_null : int
        Number of null draws used to fit the surface.
    fit_r2 : float
        Coefficient of determination on the fit.  Reported for
        diagnostics; a value below ~0.3 usually means the residual
        correction will do little.
    """

    coef: np.ndarray
    sigma: float
    model: str
    features: Tuple[str, ...]
    n_null: int
    fit_r2: float

    def predict(self, log_mean: np.ndarray, log_var: np.ndarray) -> np.ndarray:
        """Evaluate ``phi_hat`` at ``(m, v)``."""
        X = _DESIGN_BUILDERS[self.model](log_mean, log_var)
        return X @ self.coef


def fit_null_trend(
    null_log_mahalanobis: np.ndarray,
    null_log_mean: np.ndarray,
    null_log_var: np.ndarray,
    model: str = "poly3",
) -> NullTrend:
    """Fit the null-trend surface ``phi_hat(m, v)`` on permutation draws.

    The target is ``log1p(null_mahalanobis)`` — taken by the caller so
    this function can be used with any log1p-transformed statistic.

    Parameters
    ----------
    null_log_mahalanobis : ndarray of shape ``(K,)``
        ``log1p`` of the null Mahalanobis values (one entry per null
        draw).
    null_log_mean, null_log_var : ndarray of shape ``(K,)``
        ``log1p(mean)`` and ``log1p(var)`` of the gene backing each
        null draw.  For kompot's gene-shuffled null, every draw inherits
        the features of its source gene; duplicate gene indices are
        fine.
    model : str
        Design family.  ``'poly3'`` (default) fits a 7-term tensor
        polynomial in ``(m, v)``.  ``'poly3_mean_only'`` drops the
        variance feature (useful as a diagnostic — we expect a large
        drop in ``fit_r2``).

    Returns
    -------
    NullTrend
    """
    y = np.asarray(null_log_mahalanobis, dtype=float).ravel()
    m = np.asarray(null_log_mean, dtype=float).ravel()
    v = np.asarray(null_log_var, dtype=float).ravel()

    if y.shape[0] != m.shape[0] or y.shape[0] != v.shape[0]:
        raise ValueError(
            "null_log_mahalanobis, null_log_mean and null_log_var must have "
            f"matching length (got {y.shape[0]}, {m.shape[0]}, {v.shape[0]})."
        )
    if model not in _DESIGN_BUILDERS:
        raise ValueError(
            f"Unknown null_trend_model '{model}'. "
            f"Supported: {sorted(_DESIGN_BUILDERS)}."
        )
    if y.shape[0] <= _DESIGN_BUILDERS[model](m[:1], v[:1]).shape[1]:
        raise ValueError(
            f"Need more null draws than design columns "
            f"(got {y.shape[0]} for model '{model}')."
        )

    # Drop non-finite rows — e.g. Mahalanobis = 0 entries from degenerate
    # null draws propagate as log1p(0)=0 which is fine, but NaN cannot be
    # inverted by lstsq.
    finite = np.isfinite(y) & np.isfinite(m) & np.isfinite(v)
    if not finite.all():
        dropped = int((~finite).sum())
        logger.warning(
            f"Dropping {dropped} non-finite null draws before fitting phi_hat."
        )
        y = y[finite]
        m = m[finite]
        v = v[finite]

    X = _DESIGN_BUILDERS[model](m, v)
    coef, _res, _rk, _sv = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    sigma = float(np.sqrt(np.mean(resid * resid)))
    # Protect against the degenerate case where every null draw is identical.
    if sigma <= 0.0:
        logger.warning(
            "Null-trend residual scale is zero — falling back to unit sigma."
        )
        sigma = 1.0
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    fit_r2 = 1.0 - float(np.sum(resid * resid)) / ss_tot if ss_tot > 0 else 0.0

    features = ("log_mean", "log_var") if model == "poly3" else ("log_mean",)

    return NullTrend(
        coef=coef,
        sigma=sigma,
        model=model,
        features=features,
        n_null=int(y.shape[0]),
        fit_r2=fit_r2,
    )


# ---------------------------------------------------------------------------
# Residualisation
# ---------------------------------------------------------------------------


@dataclass
class ResidualisedMahalanobis:
    """Result of residualising a set of Mahalanobis values.

    Attributes
    ----------
    residual : ndarray
        ``log1p(D^2) - phi_hat(m, v)`` per real gene.  Centred at zero
        for a variance-matched null gene.
    z : ndarray
        ``residual / sigma_null``.  Standard-normal-like under the
        null.  Negative values are allowed and mean "quieter than a
        variance-matched null", so the downstream 1-D local FDR uses
        ``max(0, z)`` as its Mahalanobis surrogate.
    log_mean, log_var : ndarray
        Per-gene features.  Kept for diagnostic storage.
    trend : NullTrend
        The fitted surface used to residualise.
    null_residual : ndarray
        ``log1p(null_D^2) - phi_hat(null_m, null_v)`` — feeds the
        1-D local FDR as the null distribution of ``z``.
    """

    residual: np.ndarray
    z: np.ndarray
    log_mean: np.ndarray
    log_var: np.ndarray
    trend: NullTrend
    null_residual: np.ndarray

    @property
    def null_z(self) -> np.ndarray:
        return self.null_residual / self.trend.sigma


def residualize_mahalanobis(
    real_mahalanobis: np.ndarray,
    null_mahalanobis: np.ndarray,
    real_log_mean: np.ndarray,
    real_log_var: np.ndarray,
    null_gene_indices: Union[np.ndarray, Sequence[int]],
    model: str = "poly3",
) -> ResidualisedMahalanobis:
    """Fit the null trend and return residualised statistics.

    ``null_gene_indices[k]`` is the column index (into the real-gene
    feature array) of the gene backing ``null_mahalanobis[k]`` — this is
    how kompot's internal null stores provenance.  The gene-shuffling
    scheme preserves each gene's per-cell values, so the backing gene's
    ``(log_mean, log_var)`` is the correct feature for that null draw.
    """
    real_mahalanobis = np.asarray(real_mahalanobis, dtype=float).ravel()
    null_mahalanobis = np.asarray(null_mahalanobis, dtype=float).ravel()
    real_log_mean = np.asarray(real_log_mean, dtype=float).ravel()
    real_log_var = np.asarray(real_log_var, dtype=float).ravel()
    idx = np.asarray(null_gene_indices, dtype=int).ravel()

    if real_log_mean.shape[0] != real_log_var.shape[0]:
        raise ValueError(
            "real_log_mean and real_log_var must have the same length."
        )
    n_real = real_mahalanobis.shape[0]
    if n_real != real_log_mean.shape[0]:
        raise ValueError(
            f"real_mahalanobis length {n_real} does not match "
            f"per-gene feature length {real_log_mean.shape[0]}."
        )
    if idx.shape[0] != null_mahalanobis.shape[0]:
        raise ValueError(
            f"null_gene_indices length {idx.shape[0]} must match "
            f"null_mahalanobis length {null_mahalanobis.shape[0]}."
        )
    bad = (idx < 0) | (idx >= real_log_mean.shape[0])
    if bad.any():
        raise ValueError(
            f"null_gene_indices contains {int(bad.sum())} out-of-range entries."
        )

    null_m = real_log_mean[idx]
    null_v = real_log_var[idx]
    null_log = np.log1p(null_mahalanobis)
    trend = fit_null_trend(null_log, null_m, null_v, model=model)

    null_phi = trend.predict(null_m, null_v)
    null_residual = null_log - null_phi

    real_phi = trend.predict(real_log_mean, real_log_var)
    residual = np.log1p(real_mahalanobis) - real_phi
    z = residual / trend.sigma

    return ResidualisedMahalanobis(
        residual=residual,
        z=z,
        log_mean=real_log_mean,
        log_var=real_log_var,
        trend=trend,
        null_residual=null_residual,
    )


# ---------------------------------------------------------------------------
# FDR on residuals
# ---------------------------------------------------------------------------


def residual_local_fdr(
    residualised: ResidualisedMahalanobis,
    fdr_threshold: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply kompot's 1-D local FDR to the standardised residuals.

    The residual statistic can be negative (a gene that is *quieter*
    than the variance-matched null).  Local FDR on a strictly
    non-negative quantity is what kompot implements, so we feed
    ``max(0, Z)`` — negative ``Z`` maps to zero Mahalanobis-surrogate
    and therefore local FDR near 1.  This is the intended behaviour:
    below-null genes are not DE.

    Returns
    -------
    pvalues, local_fdr, tail_fdr, is_de
        Same shape as ``residualised.residual``.  ``is_de`` is
        ``local_fdr < fdr_threshold``.
    """
    from .anndata.fdr_utils import compute_fdr_statistics

    real_z = np.clip(residualised.z, 0.0, None)
    null_z = np.clip(residualised.null_z, 0.0, None)

    pvalues, local_fdr, tail_fdr, is_de = compute_fdr_statistics(
        real_mahalanobis=real_z,
        null_mahalanobis=null_z,
        fdr_threshold=fdr_threshold,
    )
    return pvalues, local_fdr, tail_fdr, is_de
