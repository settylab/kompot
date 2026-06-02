"""Tests for the ptp (posterior tail probability) functionality."""

import numpy as np
import pandas as pd
import pytest
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import jax.scipy.stats as jax_stats
from scipy.stats import chi2 as scipy_chi2

import anndata


def create_test_adata_with_ptp(n_cells=60, n_genes=50):
    """Create test AnnData with realistic ptp values computed from Mahalanobis distances."""
    np.random.seed(42)

    # Create test data
    X = np.random.normal(0, 1, (n_cells, n_genes))
    var_names = [f"gene_{i}" for i in range(n_genes)]

    # Create AnnData
    adata = anndata.AnnData(X, var=pd.DataFrame(index=var_names))
    adata.obs["condition"] = ["A"] * (n_cells // 2) + ["B"] * (n_cells // 2)

    # Create realistic Mahalanobis distances
    mahalanobis_distances = np.abs(np.random.gamma(2, 1, n_genes))  # Positive values

    # Compute the posterior tail probability (PTP) from Mahalanobis distances
    # using the chi2 distribution, stored as -log10(PTP) in log space — the
    # convention kompot now uses (mirrors the DA neg_log10_lfc_ptp field). The
    # log-space form avoids the linear-space saturation to 1.0 that collapses
    # gene-ranking resolution at the head of the distribution.
    degrees_of_freedom = 10  # Typical number of dimensions
    mahalanobis_squared = mahalanobis_distances.astype(np.float64) ** 2
    neg_log10_ptp_values = -scipy_chi2.logsf(
        mahalanobis_squared, df=degrees_of_freedom
    ) / np.log(10)

    # Add differential expression metrics
    adata.var["kompot_de_mahalanobis_A_to_B"] = mahalanobis_distances
    adata.var["kompot_de_neg_log10_ptp_A_to_B"] = neg_log10_ptp_values
    adata.var["kompot_de_mean_lfc_A_to_B"] = np.random.normal(0, 2, n_genes)

    # Add FDR values for comparison
    adata.var["kompot_de_mahalanobis_local_fdr_A_to_B"] = np.random.uniform(
        0, 0.5, n_genes
    )
    adata.var["kompot_de_mahalanobis_tail_fdr_A_to_B"] = np.random.uniform(
        0, 0.5, n_genes
    )
    # Significant at PTP < 0.05, i.e. -log10(PTP) > -log10(0.05)
    adata.var["kompot_de_is_de_A_to_B"] = neg_log10_ptp_values > -np.log10(0.05)

    # Add run history for proper testing
    adata.uns["kompot_de_run_history"] = [
        {
            "params": {"condition1": "A", "condition2": "B"},
            "field_names": {
                "mahalanobis_key": "kompot_de_mahalanobis_A_to_B",
                "ptp_key": "kompot_de_neg_log10_ptp_A_to_B",
                "mean_lfc_key": "kompot_de_mean_lfc_A_to_B",
            },
            "fdr_keys": {
                "local_fdr_key": "kompot_de_mahalanobis_local_fdr_A_to_B",
                "tail_fdr_key": "kompot_de_mahalanobis_tail_fdr_A_to_B",
                "is_de_key": "kompot_de_is_de_A_to_B",
            },
            "ptp_key": "kompot_de_neg_log10_ptp_A_to_B",  # in field_names
        }
    ]

    return adata


class TestPTPFunctionality:
    """Test class for ptp functionality."""

    def test_ptp_computation_basic(self):
        """Test basic ptp computation logic."""
        # Test with known values
        mahalanobis_distances = np.array([1.0, 2.0, 3.0])
        degrees_of_freedom = 5

        # Square distances and compute ptp
        mahalanobis_squared = mahalanobis_distances**2
        ptp = np.array(jax_stats.chi2.sf(mahalanobis_squared, df=degrees_of_freedom))

        # Verify properties
        assert np.all(ptp >= 0), "PTP values should be non-negative"
        assert np.all(ptp <= 1), "PTP values should be <= 1"
        assert np.all(np.diff(ptp) <= 0), (
            "PTP should decrease with increasing Mahalanobis distance"
        )

    def test_ptp_transform_for_volcano(self):
        """Test -log10 transformation of ptp for volcano plots."""
        ptp_values = np.array([0.1, 0.01, 0.001])

        # Apply transformation
        transformed = -np.log10(np.maximum(ptp_values, 1e-300))

        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(transformed, expected, rtol=1e-10)

    def test_volcano_de_with_ptp(self):
        """Test volcano_de plot with ptp y-axis type."""
        from kompot.plot.volcano import volcano_de

        adata = create_test_adata_with_ptp()

        # Test ptp y-axis
        fig = volcano_de(
            adata,
            lfc_key="kompot_de_mean_lfc_A_to_B",
            score_key="kompot_de_mahalanobis_A_to_B",
            y_axis_type="ptp",
            significance_threshold=0.05,
            return_fig=True,
        )

        ax = fig.axes[0]
        assert ax.get_ylabel() == "-log10(Posterior Tail Probability)"
        plt.close(fig)

    def test_volcano_de_ptp_threshold_line(self):
        """Test that significance threshold line is shown for ptp."""
        from kompot.plot.volcano import volcano_de

        adata = create_test_adata_with_ptp()

        fig = volcano_de(
            adata,
            lfc_key="kompot_de_mean_lfc_A_to_B",
            score_key="kompot_de_mahalanobis_A_to_B",
            y_axis_type="ptp",
            significance_threshold=0.01,
            show_thresholds=True,
            return_fig=True,
        )

        # Check that a horizontal line was added
        ax = fig.axes[0]
        hlines = [
            line
            for line in ax.lines
            if hasattr(line, "_y") and len(np.unique(line._y)) == 1
        ]
        assert len(hlines) > 0, "Expected horizontal threshold line to be present"

        plt.close(fig)

    def test_volcano_de_ptp_gene_selection(self):
        """Test gene selection with ptp threshold."""
        from kompot.plot.volcano import volcano_de

        adata = create_test_adata_with_ptp()

        # Set some genes to be clearly significant. Column stores -log10(PTP),
        # so larger = more significant.
        adata.var.loc["gene_0", "kompot_de_neg_log10_ptp_A_to_B"] = -np.log10(
            0.001
        )  # Very significant (PTP=0.001)
        adata.var.loc["gene_1", "kompot_de_neg_log10_ptp_A_to_B"] = -np.log10(
            0.005
        )  # Significant (PTP=0.005)
        adata.var.loc["gene_2", "kompot_de_neg_log10_ptp_A_to_B"] = -np.log10(
            0.1
        )  # Not significant (PTP=0.1)

        fig = volcano_de(
            adata,
            lfc_key="kompot_de_mean_lfc_A_to_B",
            score_key="kompot_de_mahalanobis_A_to_B",
            y_axis_type="ptp",
            significance_threshold=0.01,
            return_fig=True,
        )

        # Should highlight genes with PTP < 0.01, i.e. -log10(PTP) > 2
        plt.close(fig)

    def test_ptp_column_inference(self):
        """Test that ptp column is correctly inferred from mahalanobis key."""
        from kompot.plot.volcano import volcano_de

        adata = create_test_adata_with_ptp()

        # Remove run history to test fallback inference
        del adata.uns["kompot_de_run_history"]

        fig = volcano_de(
            adata,
            lfc_key="kompot_de_mean_lfc_A_to_B",
            score_key="kompot_de_mahalanobis_A_to_B",  # Should infer ptp from this
            y_axis_type="ptp",
            return_fig=True,
        )

        ax = fig.axes[0]
        assert ax.get_ylabel() == "-log10(Posterior Tail Probability)"
        plt.close(fig)

    def test_ptp_vs_fdr_consistency(self):
        """Test that ptp and FDR can be used interchangeably in volcano plots."""
        from kompot.plot.volcano import volcano_de

        adata = create_test_adata_with_ptp()

        # Test both ptp and local_fdr
        fig1 = volcano_de(
            adata,
            lfc_key="kompot_de_mean_lfc_A_to_B",
            score_key="kompot_de_mahalanobis_A_to_B",
            y_axis_type="ptp",
            significance_threshold=0.05,
            return_fig=True,
        )

        fig2 = volcano_de(
            adata,
            lfc_key="kompot_de_mean_lfc_A_to_B",
            score_key="kompot_de_mahalanobis_A_to_B",
            y_axis_type="local_fdr",
            significance_threshold=0.05,
            return_fig=True,
        )

        # Both should work without errors
        ax1 = fig1.axes[0]
        ax2 = fig2.axes[0]
        assert ax1.get_ylabel() == "-log10(Posterior Tail Probability)"
        assert ax2.get_ylabel() == "-log10(Local FDR)"

        plt.close(fig1)
        plt.close(fig2)

    def test_custom_column_name(self):
        """Test using custom column name for y-axis."""
        from kompot.plot.volcano import volcano_de

        adata = create_test_adata_with_ptp()

        # Test using ptp column directly
        fig = volcano_de(
            adata,
            lfc_key="kompot_de_mean_lfc_A_to_B",
            score_key="kompot_de_mahalanobis_A_to_B",
            y_axis_type="kompot_de_neg_log10_ptp_A_to_B",  # Custom column name
            return_fig=True,
        )

        # Should work without applying transformation (since it's not a known type)
        plt.close(fig)

    def test_ptp_error_handling(self):
        """Test error handling for missing ptp columns."""
        from kompot.plot.volcano import volcano_de

        adata = create_test_adata_with_ptp()

        # Remove ptp column
        del adata.var["kompot_de_neg_log10_ptp_A_to_B"]

        # Should fall back to mahalanobis when ptp not found
        fig = volcano_de(
            adata,
            lfc_key="kompot_de_mean_lfc_A_to_B",
            score_key="kompot_de_mahalanobis_A_to_B",
            y_axis_type="ptp",
            return_fig=True,
        )

        # Should use mahalanobis as fallback
        plt.close(fig)

    def test_significance_threshold_parameter(self):
        """Test the new significance_threshold parameter vs old fdr_threshold."""
        from kompot.plot.volcano import volcano_de

        adata = create_test_adata_with_ptp()

        # Test with significance_threshold
        fig = volcano_de(
            adata,
            lfc_key="kompot_de_mean_lfc_A_to_B",
            score_key="kompot_de_mahalanobis_A_to_B",
            y_axis_type="ptp",
            significance_threshold=0.01,
            return_fig=True,
        )

        plt.close(fig)

        # Test with mahalanobis threshold
        fig = volcano_de(
            adata,
            lfc_key="kompot_de_mean_lfc_A_to_B",
            score_key="kompot_de_mahalanobis_A_to_B",
            y_axis_type="mahalanobis",
            significance_threshold=2.0,
            return_fig=True,
        )

        plt.close(fig)


def _create_de_data(n_cells=80, n_genes=60, n_dims=10, seed=0):
    """AnnData with a clear DE signal and a moderate embedding dimension so the
    chi-squared df (= embedding dim) is large enough for the linear-space
    saturation to bite."""
    rng = np.random.RandomState(seed)
    n1 = n_cells // 2
    n2 = n_cells - n1
    # Embedding with a real shift between conditions in a few dimensions
    shift = np.zeros(n_dims)
    shift[:4] = 1.2
    X = np.vstack(
        [rng.normal(0, 1, (n1, n_dims)), rng.normal(shift, 1, (n2, n_dims))]
    )
    expr = rng.negative_binomial(10, 0.3, (n_cells, n_genes)).astype(float)
    gene_names = [f"Gene_{i:04d}" for i in range(n_genes)]
    cell_names = [f"Cell_{i:04d}" for i in range(n_cells)]
    adata = anndata.AnnData(
        expr,
        obs=pd.DataFrame(
            {"condition": ["A"] * n1 + ["B"] * n2}, index=cell_names
        ),
        var=pd.DataFrame(index=gene_names),
    )
    adata.obsm["X_pca"] = X
    return adata


class TestNegLog10PTPRegression:
    """Regression guards for the linear-space PTP saturation bug.

    The DE posterior tail probability is a strictly monotone transform of the
    Mahalanobis distance, so it must preserve the gene ranking. Storing it in
    linear space (``chi2.sf``) collapses every gene below the chi-squared mean
    onto values numerically indistinguishable from 1.0, destroying that ranking
    at the head of the distribution. Storing ``-log10(PTP)`` from ``chi2.logsf``
    in float64 keeps every value distinct. These tests would have failed against
    the old linear-space storage.
    """

    def test_linear_space_saturates_log_space_does_not(self):
        """Pure-math guard at a realistic df: linear ``sf`` saturates to 1.0 and
        loses distinct values; ``-log10(PTP)`` from ``logsf`` does not."""
        from scipy.stats import spearmanr

        rng = np.random.RandomState(0)
        df = 40  # realistic embedding dimension
        # Most genes are near-null -> D^2 well below the chi2 mean (= df).
        d2 = np.r_[rng.chisquare(5, 2000), rng.chisquare(60, 100)]

        linear_sf = scipy_chi2.sf(d2, df=df)
        neg_log10 = -scipy_chi2.logsf(d2, df=df) / np.log(10)

        # Linear space: a substantial fraction collapse to EXACTLY 1.0 ...
        assert np.mean(linear_sf == 1.0) > 0.1
        # ... so the distinct-value count is destroyed.
        assert len(np.unique(linear_sf)) < len(d2)

        # Log space: every gene keeps a distinct value.
        assert len(np.unique(neg_log10)) == len(d2)
        # And the ranking is exactly the Mahalanobis ranking.
        assert spearmanr(neg_log10, d2).correlation == pytest.approx(1.0)

    def test_stored_field_preserves_mahalanobis_ranking(self):
        """End-to-end: the stored ``neg_log10_ptp`` field ranks genes identically
        to the Mahalanobis distance, has dynamic range beyond [0, 1] (impossible
        for the old linear ``sf`` field, whose max was <= 1), and shows no mass
        saturation onto a single value."""
        try:
            from kompot.anndata import compute_differential_expression
        except ImportError:
            pytest.skip("anndata not installed")
        from scipy.stats import spearmanr

        adata = _create_de_data()
        compute_differential_expression(
            adata,
            groupby="condition",
            condition1="A",
            condition2="B",
            obsm_key="X_pca",
            result_key="reg",
            null_genes=10,
            null_seed=0,
            store_additional_stats=True,
            progress=False,
            n_landmarks=10,
        )

        mahal = adata.var["reg_A_to_B_mahalanobis"].values
        ptp = adata.var["reg_A_to_B_neg_log10_ptp"].values

        # Strictly monotone transform of the distance -> identical ranking.
        finite = np.isfinite(mahal) & np.isfinite(ptp)
        assert finite.sum() >= 2
        assert spearmanr(ptp[finite], mahal[finite]).correlation == pytest.approx(
            1.0
        )

        # -log10(PTP) is always non-negative.
        assert np.all(ptp[finite] >= 0)

        # Dynamic range the old linear-space field could not represent: at least
        # one gene exceeds 1.0 (i.e. PTP < 0.1). The old field stored sf in
        # [0, 1], so its maximum was structurally <= 1.
        assert np.nanmax(ptp) > 1.0

        # No mass saturation: no single stored value dominates the field.
        _, counts = np.unique(ptp[finite], return_counts=True)
        assert counts.max() / finite.sum() < 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
