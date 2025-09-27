"""Comprehensive tests for the robust field inference system."""

import pytest
import numpy as np
import pandas as pd
import anndata
import logging
from unittest.mock import patch, MagicMock

from kompot.plot.field_inference import (
    infer_fields_from_run_info,
    get_comparison_specific_fields,
    _fallback_field_inference,
    _check_for_overwrites
)


class TestFieldInference:
    """Test field inference from run info."""

    def setup_method(self):
        """Set up test data."""
        # Create basic AnnData object
        n_cells, n_genes = 100, 50
        X = np.random.negative_binomial(10, 0.3, (n_cells, n_genes)).astype(float)

        self.adata = anndata.AnnData(
            X=X,
            obs=pd.DataFrame({
                "condition": ["A"] * 50 + ["B"] * 50
            }, index=[f"Cell_{i}" for i in range(n_cells)]),
            var=pd.DataFrame(index=[f"Gene_{i}" for i in range(n_genes)])
        )

    def test_infer_fields_with_valid_run_info(self):
        """Test field inference with valid run info."""
        # Add run history
        run_history = [
            {
                "params": {"condition1": "A", "condition2": "B"},
                "field_names": {
                    "mean_lfc_key": "kompot_de_A_to_B_mean_lfc",
                    "mahalanobis_key": "kompot_de_A_to_B_mahalanobis",
                    "is_de_key": "kompot_de_A_to_B_is_de"
                }
            }
        ]
        self.adata.uns["kompot_de"] = {"run_history": run_history}

        # Add corresponding columns
        self.adata.var["kompot_de_A_to_B_mean_lfc"] = np.random.normal(0, 1, 50)
        self.adata.var["kompot_de_A_to_B_mahalanobis"] = np.random.gamma(2, 1, 50)
        self.adata.var["kompot_de_A_to_B_is_de"] = np.random.choice([True, False], 50)

        fields = infer_fields_from_run_info(
            adata=self.adata,
            analysis_type="de",
            run_id=-1,
            required_fields=["mean_lfc_key", "mahalanobis_key", "is_de_key"],
            strict=True
        )

        assert fields["mean_lfc_key"] == "kompot_de_A_to_B_mean_lfc"
        assert fields["mahalanobis_key"] == "kompot_de_A_to_B_mahalanobis"
        assert fields["is_de_key"] == "kompot_de_A_to_B_is_de"

    def test_infer_fields_with_missing_run_info(self):
        """Test field inference when run info is missing."""
        # No run history - should fall back to pattern matching
        self.adata.var["kompot_de_A_to_B_mean_lfc"] = np.random.normal(0, 1, 50)
        self.adata.var["kompot_de_A_to_B_mahalanobis"] = np.random.gamma(2, 1, 50)

        with patch('kompot.plot.field_inference.logger') as mock_logger:
            fields = infer_fields_from_run_info(
                adata=self.adata,
                analysis_type="de",
                run_id=-1,
                required_fields=["mean_lfc_key", "mahalanobis_key"],
                strict=False
            )

            # Should warn about missing run info
            mock_logger.warning.assert_any_call("No DE run info found for run_id=-1")
            # Should warn about fallback inference
            mock_logger.warning.assert_any_call("Attempting fallback inference for missing fields: ['mean_lfc_key', 'mahalanobis_key']")

        assert fields["mean_lfc_key"] == "kompot_de_A_to_B_mean_lfc"
        assert fields["mahalanobis_key"] == "kompot_de_A_to_B_mahalanobis"

    def test_infer_fields_with_condition_mismatch(self):
        """Test field inference with condition mismatch."""
        # Add run history with different conditions
        run_history = [
            {
                "params": {"condition1": "A", "condition2": "B"},
                "field_names": {
                    "mean_lfc_key": "kompot_de_A_to_B_mean_lfc"
                }
            }
        ]
        self.adata.uns["kompot_de"] = {"run_history": run_history}
        self.adata.var["kompot_de_A_to_B_mean_lfc"] = np.random.normal(0, 1, 50)

        with patch('kompot.plot.field_inference.logger') as mock_logger:
            fields = infer_fields_from_run_info(
                adata=self.adata,
                analysis_type="de",
                condition1="X",  # Mismatch
                condition2="Y",  # Mismatch
                run_id=-1,
                required_fields=["mean_lfc_key"],
                strict=False
            )

            # Should warn about condition mismatch
            mock_logger.warning.assert_any_call(
                "User-specified conditions (X, Y) don't match run info conditions ('A', 'B')"
            )

        assert fields["mean_lfc_key"] == "kompot_de_A_to_B_mean_lfc"

    def test_infer_fields_strict_mode_with_mismatch(self):
        """Test that strict mode raises error on condition mismatch."""
        run_history = [
            {
                "params": {"condition1": "A", "condition2": "B"},
                "field_names": {
                    "mean_lfc_key": "kompot_de_A_to_B_mean_lfc"
                }
            }
        ]
        self.adata.uns["kompot_de"] = {"run_history": run_history}
        self.adata.var["kompot_de_A_to_B_mean_lfc"] = np.random.normal(0, 1, 50)

        with pytest.raises(ValueError, match="Condition mismatch"):
            infer_fields_from_run_info(
                adata=self.adata,
                analysis_type="de",
                condition1="X",
                condition2="Y",
                run_id=-1,
                required_fields=["mean_lfc_key"],
                strict=True
            )

    def test_infer_fields_with_missing_columns(self):
        """Test field inference when run info specifies missing columns."""
        run_history = [
            {
                "params": {"condition1": "A", "condition2": "B"},
                "field_names": {
                    "mean_lfc_key": "missing_column"
                }
            }
        ]
        self.adata.uns["kompot_de"] = {"run_history": run_history}

        with patch('kompot.plot.field_inference.logger') as mock_logger:
            fields = infer_fields_from_run_info(
                adata=self.adata,
                analysis_type="de",
                run_id=-1,
                required_fields=["mean_lfc_key"],
                strict=False
            )

            # Should warn about missing column
            mock_logger.warning.assert_any_call(
                "Run info specifies mean_lfc_key='missing_column' but column not found in data"
            )

        assert fields["mean_lfc_key"] is None

    def test_infer_fields_da_analysis(self):
        """Test field inference for DA analysis."""
        run_history = [
            {
                "params": {"condition1": "A", "condition2": "B"},
                "field_names": {
                    "lfc_key": "kompot_da_A_to_B_lfc",
                    "direction_key": "kompot_da_A_to_B_lfc_direction"
                }
            }
        ]
        self.adata.uns["kompot_da"] = {"run_history": run_history}

        # Add corresponding columns to obs for DA
        self.adata.obs["kompot_da_A_to_B_lfc"] = np.random.normal(0, 1, 100)
        self.adata.obs["kompot_da_A_to_B_lfc_direction"] = np.random.choice(["up", "down", "neutral"], 100)

        fields = infer_fields_from_run_info(
            adata=self.adata,
            analysis_type="da",
            run_id=-1,
            required_fields=["lfc_key", "direction_key"],
            strict=True
        )

        assert fields["lfc_key"] == "kompot_da_A_to_B_lfc"
        assert fields["direction_key"] == "kompot_da_A_to_B_lfc_direction"


class TestComparisonSpecificFields:
    """Test comparison-specific field extraction."""

    def setup_method(self):
        """Set up test data."""
        n_cells, n_genes = 100, 50
        X = np.random.negative_binomial(10, 0.3, (n_cells, n_genes)).astype(float)

        self.adata = anndata.AnnData(
            X=X,
            obs=pd.DataFrame({
                "condition": ["A"] * 50 + ["B"] * 50
            }, index=[f"Cell_{i}" for i in range(n_cells)]),
            var=pd.DataFrame(index=[f"Gene_{i}" for i in range(n_genes)])
        )

    def test_get_comparison_specific_fields_valid(self):
        """Test getting fields for a specific valid comparison."""
        run_history = [
            {
                "params": {"condition1": "A", "condition2": "B"},
                "field_names": {
                    "mean_lfc_key": "kompot_de_A_to_B_mean_lfc",
                    "mahalanobis_key": "kompot_de_A_to_B_mahalanobis"
                }
            }
        ]
        self.adata.uns["kompot_de"] = {"run_history": run_history}

        self.adata.var["kompot_de_A_to_B_mean_lfc"] = np.random.normal(0, 1, 50)
        self.adata.var["kompot_de_A_to_B_mahalanobis"] = np.random.gamma(2, 1, 50)

        fields = get_comparison_specific_fields(
            adata=self.adata,
            analysis_type="de",
            condition1="A",
            condition2="B",
            run_id=-1
        )

        assert fields["mean_lfc_key"] == "kompot_de_A_to_B_mean_lfc"
        assert fields["mahalanobis_key"] == "kompot_de_A_to_B_mahalanobis"

    def test_get_comparison_specific_fields_invalid_comparison(self):
        """Test error when field doesn't match expected comparison."""
        # Set up two runs - one with X to Y, one with A to B
        # The latest run (X to Y) has a field name that doesn't match its conditions
        run_history = [
            {
                "params": {"condition1": "A", "condition2": "B"},
                "field_names": {
                    "mean_lfc_key": "kompot_de_A_to_B_mean_lfc",
                    "mahalanobis_key": "kompot_de_A_to_B_mahalanobis"
                }
            },
            {
                "params": {"condition1": "X", "condition2": "Y"},
                "field_names": {
                    "mean_lfc_key": "kompot_de_A_to_B_mean_lfc",  # Field name doesn't match conditions
                    "mahalanobis_key": "kompot_de_A_to_B_mahalanobis"  # Field name doesn't match conditions
                }
            }
        ]
        self.adata.uns["kompot_de"] = {"run_history": run_history}
        self.adata.var["kompot_de_A_to_B_mean_lfc"] = np.random.normal(0, 1, 50)
        self.adata.var["kompot_de_A_to_B_mahalanobis"] = np.random.normal(0, 1, 50)

        with pytest.raises(ValueError, match="does not match expected comparison X → Y"):
            get_comparison_specific_fields(
                adata=self.adata,
                analysis_type="de",
                condition1="X",
                condition2="Y",
                run_id=-1
            )


class TestFallbackInference:
    """Test fallback field inference."""

    def test_fallback_single_candidate(self):
        """Test fallback inference with single candidate."""
        data_section = pd.DataFrame({
            "kompot_de_A_to_B_mean_lfc": [1, 2, 3],
            "other_column": [4, 5, 6]
        })

        result = _fallback_field_inference(
            data_section=data_section,
            field_type="mean_lfc_key",
            analysis_type="de",
            condition1="A",
            condition2="B",
            result_key=None,
            strict=False
        )

        assert result == "kompot_de_A_to_B_mean_lfc"

    def test_fallback_multiple_candidates_with_conditions(self):
        """Test fallback inference with multiple candidates filtered by conditions."""
        data_section = pd.DataFrame({
            "kompot_de_A_to_B_mean_lfc": [1, 2, 3],
            "kompot_de_X_to_Y_mean_lfc": [4, 5, 6],
            "other_column": [7, 8, 9]
        })

        result = _fallback_field_inference(
            data_section=data_section,
            field_type="mean_lfc_key",
            analysis_type="de",
            condition1="A",
            condition2="B",
            result_key=None,
            strict=False
        )

        assert result == "kompot_de_A_to_B_mean_lfc"

    def test_fallback_multiple_candidates_strict_mode(self):
        """Test fallback inference in strict mode with multiple candidates."""
        data_section = pd.DataFrame({
            "kompot_de_A_to_B_mean_lfc": [1, 2, 3],
            "kompot_de_X_to_Y_mean_lfc": [4, 5, 6]
        })

        result = _fallback_field_inference(
            data_section=data_section,
            field_type="mean_lfc_key",
            analysis_type="de",
            condition1=None,
            condition2=None,
            result_key=None,
            strict=True
        )

        # In strict mode with multiple candidates, should return None
        assert result is None

    def test_fallback_no_candidates(self):
        """Test fallback inference with no candidates."""
        data_section = pd.DataFrame({
            "other_column": [1, 2, 3]
        })

        result = _fallback_field_inference(
            data_section=data_section,
            field_type="mean_lfc_key",
            analysis_type="de",
            condition1="A",
            condition2="B",
            result_key=None,
            strict=False
        )

        assert result is None


class TestOverwriteDetection:
    """Test overwrite detection functionality."""

    def test_check_for_overwrites_single_run(self):
        """Test overwrite check with single run (no overwrites)."""
        adata = anndata.AnnData(X=np.random.random((10, 5)))

        # Set up run history and field tracking for single run
        run_history = [
            {
                "adjusted_run_id": 0,
                "field_names": {
                    "mean_lfc_key": "test_field"
                }
            }
        ]

        # Set up field tracking - field owned by run 0
        field_tracking = {
            "var": {
                "test_field": 0
            }
        }

        adata.uns["kompot_de"] = {
            "run_history": run_history,
            "anndata_fields": field_tracking
        }
        adata.var["test_field"] = np.random.normal(0, 1, 5)

        inferred_fields = {"mean_lfc_key": "test_field"}
        warnings_issued = []

        _check_for_overwrites(adata, "de", inferred_fields, warnings_issued)

        # No warnings should be issued for single run with consistent ownership
        assert len(warnings_issued) == 0

    def test_check_for_overwrites_multiple_runs(self):
        """Test overwrite check with multiple runs writing to same field."""
        adata = anndata.AnnData(X=np.random.random((10, 5)))

        # Set up run history with two runs writing to the same field
        run_history = [
            {
                "adjusted_run_id": 0,
                "field_names": {
                    "mean_lfc_key": "test_field"
                }
            },
            {
                "adjusted_run_id": 1,
                "field_names": {
                    "mean_lfc_key": "test_field"  # Same field - indicates overwrite
                }
            }
        ]

        # Set up field tracking - field currently owned by run 0 but latest is run 1
        # This simulates an overwrite scenario where run 1 should own the field
        # but tracking shows run 0 still owns it
        field_tracking = {
            "var": {
                "test_field": 0  # Field owned by run 0, but latest run is 1
            }
        }

        adata.uns["kompot_de"] = {
            "run_history": run_history,
            "anndata_fields": field_tracking
        }
        adata.var["test_field"] = np.random.normal(0, 1, 5)

        inferred_fields = {"mean_lfc_key": "test_field"}
        warnings_issued = []

        with patch('kompot.plot.field_inference.logger') as mock_logger:
            _check_for_overwrites(adata, "de", inferred_fields, warnings_issued)

        # Should warn about overwrites - both ownership mismatch and multiple writers
        assert len(warnings_issued) >= 1
        warning_text = " ".join(warnings_issued)
        assert ("overwritten" in warning_text or "written by" in warning_text)

    def test_consistency_with_runinfo_overwrite_detection(self):
        """Test that field inference overwrite detection is consistent with RunInfo."""
        import anndata as ad
        from kompot.anndata.utils.runinfo import RunInfo

        # Create test data that mimics a real scenario
        adata = ad.AnnData(X=np.random.random((100, 50)))

        # Set up complex run history with field mapping and overwrite scenario
        run_history = [
            {
                "adjusted_run_id": 0,
                "field_names": {"mean_lfc_key": "kompot_de_A_to_B_mean_lfc"},
                "params": {"condition1": "A", "condition2": "B"},
                "field_mapping": {
                    "kompot_de_A_to_B_mean_lfc": {
                        "location": "var",
                        "type": "mean_lfc_key"
                    }
                }
            },
            {
                "adjusted_run_id": 1,
                "field_names": {"mean_lfc_key": "kompot_de_A_to_B_mean_lfc"},  # Same field
                "params": {"condition1": "A", "condition2": "B"},
                "field_mapping": {
                    "kompot_de_A_to_B_mean_lfc": {
                        "location": "var",
                        "type": "mean_lfc_key"
                    }
                }
            }
        ]

        # Set up field tracking showing field was overwritten by run 1
        # RunInfo will detect this as an overwrite when checking run 0
        field_tracking = {
            "var": {
                "kompot_de_A_to_B_mean_lfc": 1  # Field currently owned by run 1
            }
        }

        adata.uns["kompot_de"] = {
            "run_history": run_history,
            "anndata_fields": field_tracking
        }
        adata.var["kompot_de_A_to_B_mean_lfc"] = np.random.normal(0, 1, 50)

        # Test field inference overwrite detection
        inferred_fields = {"mean_lfc_key": "kompot_de_A_to_B_mean_lfc"}
        warnings_issued = []

        _check_for_overwrites(adata, "de", inferred_fields, warnings_issued)

        # Test RunInfo overwrite detection for comparison
        run_info = RunInfo(adata, run_id=0, analysis_type="de")  # Check run 0
        runinfo_overwrites = run_info.overwritten_fields

        # Both should detect overwrite issues
        assert len(warnings_issued) > 0, "Field inference should detect overwrite issues"
        assert len(runinfo_overwrites) > 0, "RunInfo should detect overwrite issues"

        # Verify both are detecting the same field as problematic
        field_inference_mentions_field = any("kompot_de_A_to_B_mean_lfc" in w for w in warnings_issued)
        runinfo_mentions_field = any(
            info["field"] == "kompot_de_A_to_B_mean_lfc"
            for info in runinfo_overwrites
        )

        assert field_inference_mentions_field, "Field inference should mention the problematic field"
        assert runinfo_mentions_field, "RunInfo should mention the problematic field"


class TestPlotFunctionIntegration:
    """Test integration with actual plot functions."""

    def setup_method(self):
        """Set up test data."""
        n_cells, n_genes = 100, 50
        X = np.random.negative_binomial(10, 0.3, (n_cells, n_genes)).astype(float)

        self.adata = anndata.AnnData(
            X=X,
            obs=pd.DataFrame({
                "condition": ["A"] * 50 + ["B"] * 50
            }, index=[f"Cell_{i}" for i in range(n_cells)]),
            var=pd.DataFrame(index=[f"Gene_{i}" for i in range(n_genes)])
        )

    def test_expression_plot_field_inference(self):
        """Test field inference in expression plotting function."""
        from kompot.plot.expression import _infer_expression_keys

        # Add run history and data
        run_history = [
            {
                "params": {"condition1": "A", "condition2": "B"},
                "field_names": {
                    "mean_lfc_key": "kompot_de_A_to_B_mean_lfc",
                    "mahalanobis_key": "kompot_de_A_to_B_mahalanobis"
                }
            }
        ]
        self.adata.uns["kompot_de"] = {"run_history": run_history}

        self.adata.var["kompot_de_A_to_B_mean_lfc"] = np.random.normal(0, 1, 50)
        self.adata.var["kompot_de_A_to_B_mahalanobis"] = np.random.gamma(2, 1, 50)

        lfc_key, score_key = _infer_expression_keys(
            adata=self.adata,
            run_id=-1,
            strict=True
        )

        assert lfc_key == "kompot_de_A_to_B_mean_lfc"
        assert score_key == "kompot_de_A_to_B_mahalanobis"

    def test_direction_plot_field_inference(self):
        """Test field inference in direction plotting function."""
        from kompot.plot.heatmap.direction_plot import _infer_direction_key

        # Add run history and data
        run_history = [
            {
                "params": {"condition1": "A", "condition2": "B"},
                "field_names": {
                    "direction_key": "kompot_da_A_to_B_lfc_direction"
                }
            }
        ]
        self.adata.uns["kompot_da"] = {"run_history": run_history}

        self.adata.obs["kompot_da_A_to_B_lfc_direction"] = np.random.choice(["up", "down", "neutral"], 100)

        direction_col, cond1, cond2 = _infer_direction_key(
            adata=self.adata,
            run_id=-1
        )

        assert direction_col == "kompot_da_A_to_B_lfc_direction"
        assert cond1 == "A"
        assert cond2 == "B"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_adata(self):
        """Test with empty AnnData object."""
        adata = anndata.AnnData(X=np.empty((0, 0)))

        fields = infer_fields_from_run_info(
            adata=adata,
            analysis_type="de",
            run_id=-1,
            required_fields=["mean_lfc_key"],
            strict=False
        )

        assert fields["mean_lfc_key"] is None

    def test_invalid_run_id(self):
        """Test with invalid run ID."""
        adata = anndata.AnnData(X=np.random.random((10, 5)))

        with patch('kompot.plot.field_inference.logger') as mock_logger:
            fields = infer_fields_from_run_info(
                adata=adata,
                analysis_type="de",
                run_id=999,  # Invalid run ID
                required_fields=["mean_lfc_key"],
                strict=False
            )

        assert fields["mean_lfc_key"] is None

    def test_corrupted_run_history(self):
        """Test with corrupted run history."""
        adata = anndata.AnnData(X=np.random.random((10, 5)))
        adata.uns["kompot_de_run_history"] = "invalid_data"  # Corrupted

        with patch('kompot.plot.field_inference.logger') as mock_logger:
            fields = infer_fields_from_run_info(
                adata=adata,
                analysis_type="de",
                run_id=-1,
                required_fields=["mean_lfc_key"],
                strict=False
            )

        # Should handle gracefully and fall back
        assert isinstance(fields, dict)

    def test_mixed_analysis_types(self):
        """Test error handling for unsupported analysis types."""
        adata = anndata.AnnData(X=np.random.random((10, 5)))

        fields = infer_fields_from_run_info(
            adata=adata,
            analysis_type="invalid_type",
            run_id=-1,
            required_fields=["some_field"],
            strict=False
        )

        # Should handle gracefully
        assert fields["some_field"] is None

    def test_incomplete_field_names(self):
        """Test with incomplete field names in run info."""
        adata = anndata.AnnData(X=np.random.random((10, 5)))

        run_history = [
            {
                "params": {"condition1": "A", "condition2": "B"},
                "field_names": {
                    # Missing some expected fields
                    "mean_lfc_key": "test_lfc"
                }
            }
        ]
        adata.uns["kompot_de"] = {"run_history": run_history}
        adata.var = pd.DataFrame({"test_lfc": [1, 2, 3, 4, 5]})

        fields = infer_fields_from_run_info(
            adata=adata,
            analysis_type="de",
            run_id=-1,
            required_fields=["mean_lfc_key", "mahalanobis_key"],  # Request missing field
            strict=False
        )

        assert fields["mean_lfc_key"] == "test_lfc"
        assert fields["mahalanobis_key"] is None  # Missing from run info


# Integration test to ensure nothing breaks
def test_module_imports():
    """Test that all modules import correctly."""
    from kompot.plot.field_inference import (
        infer_fields_from_run_info,
        get_comparison_specific_fields
    )
    from kompot.plot import (
        infer_fields_from_run_info as plot_infer,
        get_comparison_specific_fields as plot_get_specific
    )

    # Should be the same functions
    assert infer_fields_from_run_info is plot_infer
    assert get_comparison_specific_fields is plot_get_specific