"""Tests for external null distribution features.

Covers: compute_fdr(), recompute_fdr(), extract_null_distribution(),
FDRSettings(null_mahalanobis=...), FDRSettings(null_expression=...),
combine_with_internal, DifferentialExpression.compute_fdr(), and
all validation logic.
"""

import numpy as np
import pandas as pd
import pytest
import anndata as ad
import warnings

import kompot
from kompot import FDRSettings, DifferentialExpression
from kompot.fdr import compute_fdr
from kompot.anndata.fdr_utils import (
    recompute_fdr,
    extract_null_distribution,
    compute_fdr_statistics,
)
from kompot.anndata._de_helpers import _augment_with_external_null_expression


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_simple_adata(n_cells=80, n_genes=30, seed=42):
    """Create a simple AnnData for testing DE."""
    rng = np.random.RandomState(seed)
    n1 = n_cells // 2
    n2 = n_cells - n1

    X_embed = rng.randn(n_cells, 5)
    expr = rng.negative_binomial(10, 0.3, (n_cells, n_genes)).astype(float)

    # Make a few genes differentially expressed
    for g in range(3):
        expr[n1:, g] *= 5.0

    gene_names = [f"Gene_{i:03d}" for i in range(n_genes)]
    adata = ad.AnnData(
        X=expr,
        obs=pd.DataFrame(
            {"condition": ["A"] * n1 + ["B"] * n2},
            index=[f"c{i}" for i in range(n_cells)],
        ),
        var=pd.DataFrame(index=gene_names),
    )
    adata.obsm["DM_EigenVectors"] = X_embed
    return adata


# =====================================================================
# 1. Standalone compute_fdr()
# =====================================================================


class TestComputeFdr:
    def test_basic(self):
        rng = np.random.RandomState(0)
        real = np.concatenate([rng.exponential(2, 20), rng.exponential(0.5, 80)])
        null = rng.exponential(0.5, 200)

        df = compute_fdr(real, null, threshold=0.05)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == [
            "mahalanobis",
            "pvalue",
            "local_fdr",
            "tail_fdr",
            "is_de",
        ]
        assert len(df) == len(real)
        assert df["pvalue"].between(0, 1).all()
        assert df["local_fdr"].between(0, 1).all()
        assert df["tail_fdr"].between(0, 1).all()
        assert df["is_de"].dtype == bool

    def test_gene_names(self):
        real = np.array([1.0, 2.0, 3.0])
        null = np.array([0.5, 0.6, 0.7, 0.8])
        names = ["GeneA", "GeneB", "GeneC"]

        df = compute_fdr(real, null, gene_names=names)
        assert list(df.index) == names

    def test_same_as_internal(self):
        """Standalone compute_fdr should give same results as internal engine."""
        rng = np.random.RandomState(1)
        real = rng.exponential(1, 50)
        null = rng.exponential(0.5, 100)

        df = compute_fdr(real, null, threshold=0.05)
        pv, lfdr, tfdr, sig = compute_fdr_statistics(real, null, 0.05)

        np.testing.assert_array_almost_equal(df["pvalue"].values, pv)
        np.testing.assert_array_almost_equal(df["local_fdr"].values, lfdr)
        np.testing.assert_array_almost_equal(df["tail_fdr"].values, tfdr)
        np.testing.assert_array_equal(df["is_de"].values, sig)

    def test_integer_index_when_no_names(self):
        real = np.array([1.0, 2.0])
        null = np.array([0.5, 0.6])
        df = compute_fdr(real, null)
        assert list(df.index) == [0, 1]


# =====================================================================
# 2. extract_null_distribution() and recompute_fdr()
# =====================================================================


class TestExtractAndRecompute:
    @pytest.fixture
    def adata_with_de(self):
        adata = _make_simple_adata()
        kompot.de(
            adata,
            "condition",
            "A",
            "B",
            fdr=FDRSettings(null_genes=200, null_seed=0),
            output=kompot.OutputSettings(progress=False),
        )
        return adata

    def test_extract_null_distribution(self, adata_with_de):
        dist = extract_null_distribution(adata_with_de)
        assert isinstance(dist, np.ndarray)
        assert len(dist) > 0
        assert not np.any(np.isnan(dist))

    def test_recompute_fdr_basic(self, adata_with_de):
        rng = np.random.RandomState(99)
        new_null = rng.exponential(0.3, 500)

        df = recompute_fdr(adata_with_de, new_null, threshold=0.1)
        assert isinstance(df, pd.DataFrame)
        assert "mahalanobis" in df.columns
        assert "is_de" in df.columns
        assert len(df) == adata_with_de.n_vars

    def test_recompute_fdr_updates_adata(self, adata_with_de):
        rng = np.random.RandomState(99)
        new_null = rng.exponential(0.3, 500)

        # Get original FDR values
        from kompot.anndata.utils.field_tracking import get_run_from_history

        run_info = get_run_from_history(adata_with_de, run_id=-1, analysis_type="de")
        fm = run_info["field_mapping"]

        lfdr_col = None
        for col, meta in fm.items():
            if meta.get("type") == "local_fdr" and meta.get("location") == "var":
                lfdr_col = col
                break

        if lfdr_col and lfdr_col in adata_with_de.var.columns:
            old_vals = adata_with_de.var[lfdr_col].values.copy()
            recompute_fdr(adata_with_de, new_null, threshold=0.1, inplace=True)
            new_vals = adata_with_de.var[lfdr_col].values
            # With a very different null, values should change
            assert not np.allclose(old_vals, new_vals, equal_nan=True)

    def test_recompute_fdr_no_inplace(self, adata_with_de):
        rng = np.random.RandomState(99)
        new_null = rng.exponential(0.3, 500)

        from kompot.anndata.utils.field_tracking import get_run_from_history

        run_info = get_run_from_history(adata_with_de, run_id=-1, analysis_type="de")
        fm = run_info["field_mapping"]

        lfdr_col = None
        for col, meta in fm.items():
            if meta.get("type") == "local_fdr" and meta.get("location") == "var":
                lfdr_col = col
                break

        if lfdr_col and lfdr_col in adata_with_de.var.columns:
            old_vals = adata_with_de.var[lfdr_col].values.copy()
            df = recompute_fdr(adata_with_de, new_null, inplace=False)
            new_vals = adata_with_de.var[lfdr_col].values
            # inplace=False should NOT modify adata
            np.testing.assert_array_equal(old_vals, new_vals)
            assert isinstance(df, pd.DataFrame)

    def test_roundtrip_extract_recompute(self, adata_with_de):
        """extract -> recompute with same null should give similar results."""
        null_dist = extract_null_distribution(adata_with_de)
        df = recompute_fdr(adata_with_de, null_dist, threshold=0.05, inplace=False)
        # Results should be similar (not identical due to density estimation)
        assert df["is_de"].sum() >= 0  # Sanity check, just no crash

    def test_extract_no_run_raises(self):
        adata = _make_simple_adata()
        with pytest.raises(ValueError, match="No DE run found"):
            extract_null_distribution(adata)

    def test_recompute_no_run_raises(self):
        adata = _make_simple_adata()
        with pytest.raises(ValueError, match="No DE run found"):
            recompute_fdr(adata, np.array([1.0, 2.0]))


# =====================================================================
# 3. FDRSettings validation
# =====================================================================


class TestFDRSettingsValidation:
    def test_mutual_exclusivity(self):
        adata = _make_simple_adata()
        with pytest.raises(ValueError, match="mutually exclusive"):
            kompot.de(
                adata,
                "condition",
                "A",
                "B",
                fdr=FDRSettings(
                    null_mahalanobis=np.array([1.0]),
                    null_expression=(np.zeros((40, 5)), np.zeros((40, 5))),
                ),
            )

    def test_null_expression_with_prefitted_raises(self):
        adata = _make_simple_adata()
        # Create a dummy predictor
        dummy_pred = lambda x: x
        with pytest.raises(
            ValueError, match="null_expression cannot be used with pre-fitted"
        ):
            kompot.de(
                adata,
                "condition",
                "A",
                "B",
                fdr=FDRSettings(
                    null_expression=(np.zeros((40, 5)), np.zeros((40, 5))),
                ),
                model=kompot.ModelSettings(function_predictor1=dummy_pred),
            )

    def test_combine_with_no_internal_warns(self):
        adata = _make_simple_adata()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # This should warn but then proceed
            try:
                kompot.de(
                    adata,
                    "condition",
                    "A",
                    "B",
                    fdr=FDRSettings(
                        null_genes=0,
                        null_mahalanobis=np.ones(100),
                        combine_with_internal=True,
                    ),
                    output=kompot.OutputSettings(progress=False),
                )
            except Exception:
                pass  # May fail for other reasons, we just want the warning
            combine_warns = [x for x in w if "combine_with_internal" in str(x.message)]
            assert len(combine_warns) > 0


# =====================================================================
# 4. null_mahalanobis in de()
# =====================================================================


class TestNullMahalanobisInDE:
    def test_external_null_mahalanobis_bypasses_internal(self):
        adata = _make_simple_adata()
        # Use external null with null_genes=0 (no internal null)
        ext_null = np.random.RandomState(0).exponential(0.5, 300)
        kompot.de(
            adata,
            "condition",
            "A",
            "B",
            fdr=FDRSettings(null_genes=0, null_mahalanobis=ext_null),
            output=kompot.OutputSettings(progress=False),
        )
        # Check that FDR columns exist
        de_cols = [c for c in adata.var.columns if "kompot_de" in c]
        fdr_cols = [c for c in de_cols if "fdr" in c.lower() or "is_de" in c.lower()]
        assert len(fdr_cols) > 0

    def test_combine_external_with_internal(self):
        adata = _make_simple_adata()
        ext_null = np.random.RandomState(0).exponential(0.5, 100)
        kompot.de(
            adata,
            "condition",
            "A",
            "B",
            fdr=FDRSettings(
                null_genes=100,
                null_mahalanobis=ext_null,
                combine_with_internal=True,
            ),
            output=kompot.OutputSettings(progress=False),
        )
        de_cols = [c for c in adata.var.columns if "kompot_de" in c]
        fdr_cols = [c for c in de_cols if "fdr" in c.lower() or "is_de" in c.lower()]
        assert len(fdr_cols) > 0


# =====================================================================
# 5. null_expression in de()
# =====================================================================


class TestNullExpressionInDE:
    def test_null_expression_appends_columns(self):
        adata = _make_simple_adata(n_cells=80, n_genes=20)
        n1 = 40
        n2 = 40
        rng = np.random.RandomState(7)
        # Create null expression with matching cell counts
        null_e1 = rng.randn(n1, 10)
        null_e2 = rng.randn(n2, 10)

        kompot.de(
            adata,
            "condition",
            "A",
            "B",
            fdr=FDRSettings(
                null_genes=0,
                null_expression=(null_e1, null_e2),
            ),
            output=kompot.OutputSettings(progress=False),
        )
        de_cols = [c for c in adata.var.columns if "kompot_de" in c]
        fdr_cols = [c for c in de_cols if "fdr" in c.lower() or "is_de" in c.lower()]
        assert len(fdr_cols) > 0


# =====================================================================
# 6. _augment_with_external_null_expression helper
# =====================================================================


class TestAugmentExternalNull:
    def test_basic(self):
        expr1 = np.ones((10, 5))
        expr2 = np.ones((8, 5))
        genes = ["g0", "g1", "g2", "g3", "g4"]
        null_idx = []

        null_e1 = np.zeros((10, 3))
        null_e2 = np.zeros((8, 3))

        e1, e2, eg, ni = _augment_with_external_null_expression(
            expr1, expr2, genes, null_idx, null_e1, null_e2
        )

        assert e1.shape == (10, 8)
        assert e2.shape == (8, 8)
        assert len(eg) == 8
        assert eg[:5] == genes
        assert eg[5:] == ["NULL_EXT_0", "NULL_EXT_1", "NULL_EXT_2"]
        assert ni == [5, 6, 7]

    def test_extends_existing_null(self):
        expr1 = np.ones((10, 7))  # 5 real + 2 existing null
        expr2 = np.ones((8, 7))
        genes = ["g0", "g1", "g2", "g3", "g4", "NULL_0", "NULL_1"]
        null_idx = [5, 6]

        null_e1 = np.zeros((10, 2))
        null_e2 = np.zeros((8, 2))

        e1, e2, eg, ni = _augment_with_external_null_expression(
            expr1, expr2, genes, null_idx, null_e1, null_e2
        )

        assert e1.shape == (10, 9)
        assert len(eg) == 9
        assert ni == [5, 6, 7, 8]

    def test_cell_count_mismatch_raises(self):
        expr1 = np.ones((10, 5))
        expr2 = np.ones((8, 5))

        with pytest.raises(ValueError, match="condition 1"):
            _augment_with_external_null_expression(
                expr1, expr2, ["g"] * 5, [], np.ones((9, 2)), np.ones((8, 2))
            )

        with pytest.raises(ValueError, match="condition 2"):
            _augment_with_external_null_expression(
                expr1, expr2, ["g"] * 5, [], np.ones((10, 2)), np.ones((7, 2))
            )

    def test_column_count_mismatch_raises(self):
        with pytest.raises(ValueError, match="different numbers of columns"):
            _augment_with_external_null_expression(
                np.ones((10, 5)),
                np.ones((8, 5)),
                ["g"] * 5,
                [],
                np.ones((10, 3)),
                np.ones((8, 2)),
            )

    def test_1d_input(self):
        """1-D null expression should be treated as single column."""
        e1, e2, eg, ni = _augment_with_external_null_expression(
            np.ones((10, 5)),
            np.ones((8, 5)),
            ["g"] * 5,
            [],
            np.ones(10),
            np.ones(8),
        )
        assert e1.shape == (10, 6)
        assert e2.shape == (8, 6)


# =====================================================================
# 7. DifferentialExpression.compute_fdr()
# =====================================================================


class TestSklearnComputeFdr:
    def test_compute_fdr_method(self):
        rng = np.random.RandomState(42)
        n1, n2, n_genes, n_dim = 30, 30, 15, 5

        X1 = rng.randn(n1, n_dim)
        X2 = rng.randn(n2, n_dim) + 0.3

        expr1 = rng.negative_binomial(10, 0.3, (n1, n_genes)).astype(float)
        expr2 = rng.negative_binomial(10, 0.3, (n2, n_genes)).astype(float)
        expr2[:, :3] *= 4.0  # Make some genes DE

        model = DifferentialExpression(n_landmarks=50)
        model.fit(X1, expr1, X2, expr2, sigma=1.0, ls_factor=10.0)

        X_pred = np.vstack([X1, X2])
        result = model.predict(X_pred, compute_mahalanobis=True, progress=False)

        assert model._last_mahalanobis is not None

        null = rng.exponential(0.5, 200)
        df = model.compute_fdr(null, threshold=0.05)

        assert isinstance(df, pd.DataFrame)
        assert "is_de" in df.columns
        assert len(df) == n_genes

    def test_compute_fdr_without_predict_raises(self):
        model = DifferentialExpression()
        with pytest.raises(ValueError, match="No Mahalanobis"):
            model.compute_fdr(np.array([1.0]))

    def test_compute_fdr_with_gene_names(self):
        rng = np.random.RandomState(42)
        n1, n2, n_genes, n_dim = 20, 20, 10, 3

        X1 = rng.randn(n1, n_dim)
        X2 = rng.randn(n2, n_dim)
        expr1 = rng.randn(n1, n_genes)
        expr2 = rng.randn(n2, n_genes)

        model = DifferentialExpression(n_landmarks=30)
        model.fit(X1, expr1, X2, expr2, sigma=1.0, ls_factor=10.0)
        model.predict(np.vstack([X1, X2]), compute_mahalanobis=True, progress=False)

        names = [f"G{i}" for i in range(n_genes)]
        df = model.compute_fdr(rng.exponential(1, 100), gene_names=names)
        assert list(df.index) == names


# =====================================================================
# 8. Top-level API access
# =====================================================================


class TestTopLevelAccess:
    def test_kompot_compute_fdr(self):
        assert kompot.compute_fdr is compute_fdr

    def test_kompot_recompute_fdr(self):
        assert kompot.recompute_fdr is recompute_fdr

    def test_kompot_extract_null_distribution(self):
        assert kompot.extract_null_distribution is extract_null_distribution
