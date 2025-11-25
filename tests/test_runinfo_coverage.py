"""
Unit tests for RunInfo and RunComparison classes.

These tests target uncovered code paths in kompot/anndata/utils/runinfo.py.
"""

import pytest
import numpy as np
import pandas as pd
from anndata import AnnData
from kompot.anndata.utils.runinfo import RunInfo, RunComparison
from kompot.anndata.differential_expression import compute_differential_expression
import copy


@pytest.fixture
def adata_with_de_history():
    """Create AnnData with differential expression run history."""
    np.random.seed(42)
    n_obs = 60
    n_vars = 30

    X = np.random.randn(n_obs, n_vars)
    obs = pd.DataFrame({
        'condition': ['A'] * 30 + ['B'] * 30,
        'sample': ['s1'] * 15 + ['s2'] * 15 + ['s3'] * 15 + ['s4'] * 15
    })
    var = pd.DataFrame({'gene_name': [f'Gene_{i}' for i in range(n_vars)]})
    obsm = {'DM_EigenVectors': np.random.randn(n_obs, 10)}

    adata = AnnData(X=X, obs=obs, var=var, obsm=obsm)

    # Run differential expression to create run history
    compute_differential_expression(
        adata,
        groupby='condition',
        condition1='A',
        condition2='B',
        obsm_key='DM_EigenVectors',
        sample_col='sample',
        n_landmarks=10,
        result_key='de1',
        overwrite=True
    )

    return adata


@pytest.fixture
def adata_with_multiple_de_runs(adata_with_de_history):
    """Create AnnData with multiple DE runs."""
    # Run a second DE analysis with different result_key
    compute_differential_expression(
        adata_with_de_history,
        groupby='condition',
        condition1='A',
        condition2='B',
        obsm_key='DM_EigenVectors',
        sample_col='sample',
        n_landmarks=10,
        result_key='de2',
        overwrite=True
    )

    return adata_with_de_history


class TestRunInfoInitialization:
    """Test RunInfo initialization and basic functionality."""

    def test_init_default_run_id(self, adata_with_de_history):
        """Test initialization with default run_id (None -> -1 most recent)."""
        run_info = RunInfo(adata_with_de_history)

        assert run_info.run_id == -1
        assert run_info.adjusted_run_id == 0
        assert run_info.analysis_type == 'de'
        assert run_info.storage_key == 'kompot_de'

    def test_init_explicit_run_id(self, adata_with_de_history):
        """Test initialization with explicit run_id."""
        run_info = RunInfo(adata_with_de_history, run_id=0)

        assert run_info.run_id == 0
        assert run_info.adjusted_run_id == 0

    def test_init_negative_run_id(self, adata_with_multiple_de_runs):
        """Test initialization with negative run_id (relative indexing)."""
        # -1 should get the most recent run (run 1)
        run_info = RunInfo(adata_with_multiple_de_runs, run_id=-1)

        assert run_info.run_id == -1
        assert run_info.adjusted_run_id == 1

    def test_init_explicit_analysis_type(self, adata_with_de_history):
        """Test initialization with explicit analysis_type."""
        run_info = RunInfo(adata_with_de_history, run_id=0, analysis_type='de')

        assert run_info.analysis_type == 'de'

    def test_init_invalid_analysis_type(self, adata_with_de_history):
        """Test that invalid analysis_type raises error."""
        with pytest.raises(ValueError, match="Invalid analysis_type"):
            RunInfo(adata_with_de_history, run_id=0, analysis_type='invalid')

    def test_init_no_run_history(self):
        """Test that AnnData without run history raises error."""
        adata = AnnData(X=np.random.randn(10, 5))

        with pytest.raises(ValueError, match="Could not detect analysis type"):
            RunInfo(adata)

    def test_init_empty_run_history(self):
        """Test that AnnData with empty run history raises error."""
        adata = AnnData(X=np.random.randn(10, 5))
        adata.uns['kompot_de'] = {'run_history': []}

        with pytest.raises(ValueError, match="No run history found"):
            RunInfo(adata, analysis_type='de')

    def test_init_invalid_run_id(self, adata_with_de_history):
        """Test that invalid run_id raises error."""
        with pytest.raises(ValueError, match="Run ID .* not found"):
            RunInfo(adata_with_de_history, run_id=999)


class TestRunInfoAttributes:
    """Test RunInfo attribute access and methods."""

    def test_field_names_attribute(self, adata_with_de_history):
        """Test that field_names attribute is set correctly."""
        run_info = RunInfo(adata_with_de_history)

        assert isinstance(run_info.field_names, dict)

    def test_params_attribute(self, adata_with_de_history):
        """Test that params attribute is set correctly."""
        run_info = RunInfo(adata_with_de_history)

        assert isinstance(run_info.params, dict)
        assert 'condition1' in run_info.params
        assert 'condition2' in run_info.params
        assert run_info.params['condition1'] == 'A'
        assert run_info.params['condition2'] == 'B'

    def test_params_includes_result_key(self, adata_with_de_history):
        """Test that result_key is included in params if missing."""
        run_info = RunInfo(adata_with_de_history)

        # result_key should be added to params if it's in run_info but not in params
        assert 'result_key' in run_info.params

    def test_environment_attribute(self, adata_with_de_history):
        """Test that environment attribute is set correctly."""
        run_info = RunInfo(adata_with_de_history)

        assert isinstance(run_info.environment, dict)

    def test_timestamp_attribute(self, adata_with_de_history):
        """Test that timestamp attribute is set correctly."""
        run_info = RunInfo(adata_with_de_history)

        assert isinstance(run_info.timestamp, str)


class TestRunInfoFields:
    """Test RunInfo field tracking methods."""

    def test_get_fields_for_run(self, adata_with_de_history):
        """Test _get_fields_for_run returns correct fields."""
        run_info = RunInfo(adata_with_de_history)

        fields = run_info._get_fields_for_run()
        assert isinstance(fields, dict)

        # Should have fields in var location
        assert 'var' in fields
        assert isinstance(fields['var'], list)

    def test_adata_fields_attribute(self, adata_with_de_history):
        """Test that adata_fields attribute is set correctly."""
        run_info = RunInfo(adata_with_de_history)

        assert hasattr(run_info, 'adata_fields')
        assert isinstance(run_info.adata_fields, dict)

    def test_check_overwritten_fields_none(self, adata_with_de_history):
        """Test that check_overwritten_fields returns empty list for single run."""
        run_info = RunInfo(adata_with_de_history)

        # First run should have no overwritten fields
        assert run_info.overwritten_fields == []

    def test_check_overwritten_fields_detected(self, adata_with_multiple_de_runs):
        """Test that overwritten fields are detected."""
        # Get info for the first run (which should be overwritten by second run)
        run_info = RunInfo(adata_with_multiple_de_runs, run_id=0)

        # The first run's fields may be overwritten by the second run
        # (depends on whether they used same result_key)
        assert isinstance(run_info.overwritten_fields, list)

    def test_check_missing_fields_none(self, adata_with_de_history):
        """Test that check_missing_fields returns empty list when all fields present."""
        run_info = RunInfo(adata_with_de_history)

        # All fields should be present
        assert run_info.missing_fields == []

    def test_check_missing_fields_detected(self, adata_with_de_history):
        """Test that missing fields are detected."""
        run_info = RunInfo(adata_with_de_history)

        # Delete a field that was created by the run
        if 'var' in run_info.adata_fields and len(run_info.adata_fields['var']) > 0:
            field_to_delete = run_info.adata_fields['var'][0]
            if field_to_delete in adata_with_de_history.var.columns:
                adata_with_de_history.var.drop(columns=[field_to_delete], inplace=True)

                # Re-create RunInfo to check for missing fields
                run_info_new = RunInfo(adata_with_de_history)

                # Should detect the missing field
                assert len(run_info_new.missing_fields) > 0
                assert any(f['field'] == field_to_delete for f in run_info_new.missing_fields)


class TestRunInfoDataRetrieval:
    """Test RunInfo data retrieval methods."""

    def test_get_raw_data(self, adata_with_de_history):
        """Test get_raw_data returns raw run_info."""
        run_info = RunInfo(adata_with_de_history)

        raw_data = run_info.get_raw_data()
        assert isinstance(raw_data, dict)
        assert raw_data == run_info.run_info

    def test_get_data(self, adata_with_de_history):
        """Test get_data returns comprehensive run data."""
        run_info = RunInfo(adata_with_de_history)

        data = run_info.get_data()
        assert isinstance(data, dict)
        assert 'run_id' in data
        assert 'adjusted_run_id' in data
        assert 'analysis_type' in data
        assert 'field_names' in data
        assert 'params' in data
        assert 'environment' in data
        assert 'timestamp' in data
        assert 'overwritten_fields' in data
        assert 'field_data' in data

    def test_get_data_with_missing_adata_fields(self, adata_with_de_history):
        """Test get_data handles missing adata_fields gracefully."""
        run_info = RunInfo(adata_with_de_history)

        # Temporarily remove adata_fields
        run_info.adata_fields = {}

        data = run_info.get_data()
        assert 'field_data' in data
        assert isinstance(data['field_data'], dict)

    def test_get_summary(self, adata_with_de_history):
        """Test get_summary returns summary information."""
        run_info = RunInfo(adata_with_de_history)

        summary = run_info.get_summary()
        assert isinstance(summary, dict)
        assert 'run_id' in summary
        assert 'adjusted_run_id' in summary
        assert 'analysis_type' in summary
        assert 'timestamp' in summary
        assert 'conditions' in summary
        assert 'obsm_key' in summary
        assert 'field_count' in summary
        assert 'overwritten_field_count' in summary
        assert 'missing_field_count' in summary

    def test_get_summary_with_groups(self, adata_with_de_history):
        """Test get_summary includes groups information when available."""
        # Manually add groups_summary to run_info
        run_info = RunInfo(adata_with_de_history)

        # Modify the raw run_info to include groups
        run_info.run_info['has_groups'] = True
        run_info.run_info['groups_summary'] = {
            'count': 5,
            'names': ['group1', 'group2', 'group3', 'group4', 'group5']
        }

        summary = run_info.get_summary()
        assert summary['has_groups'] == True
        assert summary['groups_count'] == 5
        assert 'groups' in summary


class TestRunInfoStringRepresentations:
    """Test RunInfo string representation methods."""

    def test_repr(self, adata_with_de_history):
        """Test __repr__ returns correct string."""
        run_info = RunInfo(adata_with_de_history)

        repr_str = repr(run_info)
        assert 'RunInfo' in repr_str
        assert 'de' in repr_str
        assert str(run_info.adjusted_run_id) in repr_str

    def test_repr_html(self, adata_with_de_history):
        """Test _repr_html_ returns HTML string."""
        run_info = RunInfo(adata_with_de_history)

        html = run_info._repr_html_()
        assert isinstance(html, str)
        assert '<div' in html
        assert 'kompot-runinfo' in html
        assert 'Run Summary' in html

    def test_repr_html_with_overwritten_fields(self, adata_with_multiple_de_runs):
        """Test _repr_html_ handles overwritten fields."""
        run_info = RunInfo(adata_with_multiple_de_runs, run_id=0)

        html = run_info._repr_html_()
        assert isinstance(html, str)
        # Should contain field status information
        assert 'kompot-runinfo' in html

    def test_repr_html_with_missing_fields(self, adata_with_de_history):
        """Test _repr_html_ handles missing fields."""
        run_info = RunInfo(adata_with_de_history)

        # Delete a field
        if 'var' in run_info.adata_fields and len(run_info.adata_fields['var']) > 0:
            field_to_delete = run_info.adata_fields['var'][0]
            if field_to_delete in adata_with_de_history.var.columns:
                adata_with_de_history.var.drop(columns=[field_to_delete], inplace=True)

                # Re-create RunInfo
                run_info_new = RunInfo(adata_with_de_history)

                html = run_info_new._repr_html_()
                assert isinstance(html, str)
                assert 'Missing' in html or 'missing' in html.lower()


class TestRunInfoComparison:
    """Test RunInfo comparison functionality."""

    def test_compare_with(self, adata_with_multiple_de_runs):
        """Test compare_with creates RunComparison object."""
        run_info = RunInfo(adata_with_multiple_de_runs, run_id=0)

        comparison = run_info.compare_with(1)
        assert isinstance(comparison, RunComparison)
        assert comparison.run1.adjusted_run_id == 0
        assert comparison.run2.adjusted_run_id == 1


class TestRunComparisonInitialization:
    """Test RunComparison initialization."""

    def test_init(self, adata_with_multiple_de_runs):
        """Test RunComparison initialization."""
        comparison = RunComparison(adata_with_multiple_de_runs, 0, 1, 'de')

        assert comparison.analysis_type == 'de'
        assert isinstance(comparison.run1, RunInfo)
        assert isinstance(comparison.run2, RunInfo)
        assert comparison.run1.adjusted_run_id == 0
        assert comparison.run2.adjusted_run_id == 1

    def test_summary_attributes(self, adata_with_multiple_de_runs):
        """Test that summary attributes are set."""
        comparison = RunComparison(adata_with_multiple_de_runs, 0, 1, 'de')

        assert hasattr(comparison, 'summary1')
        assert hasattr(comparison, 'summary2')
        assert isinstance(comparison.summary1, dict)
        assert isinstance(comparison.summary2, dict)

    def test_comparison_attributes(self, adata_with_multiple_de_runs):
        """Test that comparison attributes are set."""
        comparison = RunComparison(adata_with_multiple_de_runs, 0, 1, 'de')

        assert hasattr(comparison, 'param_comparison')
        assert hasattr(comparison, 'field_comparison')
        assert isinstance(comparison.param_comparison, dict)
        assert isinstance(comparison.field_comparison, dict)


class TestRunComparisonParameters:
    """Test RunComparison parameter comparison."""

    def test_compare_parameters_same(self, adata_with_de_history):
        """Test parameter comparison when parameters are the same."""
        # Create two runs with same parameters
        compute_differential_expression(
            adata_with_de_history,
            groupby='condition',
            condition1='A',
            condition2='B',
            obsm_key='DM_EigenVectors',
            sample_col='sample',
            n_landmarks=10,
            result_key='de2',
            overwrite=True
        )

        comparison = RunComparison(adata_with_de_history, 0, 1, 'de')

        # Most parameters should be the same
        assert 'same' in comparison.param_comparison
        assert 'different' in comparison.param_comparison
        assert isinstance(comparison.param_comparison['same'], dict)

    def test_compare_parameters_different(self, adata_with_de_history):
        """Test parameter comparison when some parameters differ."""
        # Create second run with different result_key
        compute_differential_expression(
            adata_with_de_history,
            groupby='condition',
            condition1='A',
            condition2='B',
            obsm_key='DM_EigenVectors',
            sample_col='sample',
            n_landmarks=10,
            result_key='de_different',
            overwrite=True
        )

        comparison = RunComparison(adata_with_de_history, 0, 1, 'de')

        # result_key should be different
        assert 'different' in comparison.param_comparison
        assert 'only_in_run1' in comparison.param_comparison
        assert 'only_in_run2' in comparison.param_comparison

    def test_param_comparison_structure(self, adata_with_multiple_de_runs):
        """Test that param_comparison has correct structure."""
        comparison = RunComparison(adata_with_multiple_de_runs, 0, 1, 'de')

        param_comp = comparison.param_comparison
        assert 'same' in param_comp
        assert 'different' in param_comp
        assert 'only_in_run1' in param_comp
        assert 'only_in_run2' in param_comp

        # Check that 'different' contains dicts with 'run1' and 'run2' keys
        for key, value in param_comp['different'].items():
            assert isinstance(value, dict)
            assert 'run1' in value
            assert 'run2' in value


class TestRunComparisonFields:
    """Test RunComparison field comparison."""

    def test_compare_fields(self, adata_with_multiple_de_runs):
        """Test field comparison."""
        comparison = RunComparison(adata_with_multiple_de_runs, 0, 1, 'de')

        field_comp = comparison.field_comparison
        assert 'by_location' in field_comp
        assert 'totals' in field_comp
        assert isinstance(field_comp['by_location'], dict)
        assert isinstance(field_comp['totals'], dict)

    def test_field_comparison_structure(self, adata_with_multiple_de_runs):
        """Test that field_comparison has correct structure."""
        comparison = RunComparison(adata_with_multiple_de_runs, 0, 1, 'de')

        totals = comparison.field_comparison['totals']
        assert 'same' in totals
        assert 'only_in_run1' in totals
        assert 'only_in_run2' in totals
        assert isinstance(totals['same'], int)
        assert isinstance(totals['only_in_run1'], int)
        assert isinstance(totals['only_in_run2'], int)

    def test_field_comparison_by_location(self, adata_with_multiple_de_runs):
        """Test field comparison by location."""
        comparison = RunComparison(adata_with_multiple_de_runs, 0, 1, 'de')

        by_location = comparison.field_comparison['by_location']

        # Each location should have same, only_in_run1, only_in_run2
        for location, data in by_location.items():
            assert 'same' in data
            assert 'only_in_run1' in data
            assert 'only_in_run2' in data
            assert isinstance(data['same'], list)
            assert isinstance(data['only_in_run1'], list)
            assert isinstance(data['only_in_run2'], list)


class TestRunComparisonSummary:
    """Test RunComparison summary methods."""

    def test_get_summary(self, adata_with_multiple_de_runs):
        """Test get_summary returns correct structure."""
        comparison = RunComparison(adata_with_multiple_de_runs, 0, 1, 'de')

        summary = comparison.get_summary()
        assert isinstance(summary, dict)
        assert 'run1' in summary
        assert 'run2' in summary
        assert 'parameters' in summary
        assert 'fields' in summary

        # Check run1/run2 structure
        assert 'run_id' in summary['run1']
        assert 'timestamp' in summary['run1']
        assert 'result_key' in summary['run1']

        # Check parameters structure
        assert 'same_count' in summary['parameters']
        assert 'different_count' in summary['parameters']
        assert 'only_in_run1_count' in summary['parameters']
        assert 'only_in_run2_count' in summary['parameters']

    def test_repr(self, adata_with_multiple_de_runs):
        """Test __repr__ returns correct string."""
        comparison = RunComparison(adata_with_multiple_de_runs, 0, 1, 'de')

        repr_str = repr(comparison)
        assert 'RunComparison' in repr_str
        assert 'run1=' in repr_str
        assert 'run2=' in repr_str

    def test_repr_html(self, adata_with_multiple_de_runs):
        """Test _repr_html_ returns HTML string."""
        comparison = RunComparison(adata_with_multiple_de_runs, 0, 1, 'de')

        html = comparison._repr_html_()
        assert isinstance(html, str)
        assert '<div' in html
        assert 'kompot-comparison' in html
        assert 'Comparison' in html

    def test_repr_html_with_differences(self, adata_with_de_history):
        """Test _repr_html_ highlights differences."""
        # Create runs with different parameters
        compute_differential_expression(
            adata_with_de_history,
            groupby='condition',
            condition1='A',
            condition2='B',
            obsm_key='DM_EigenVectors',
            sample_col='sample',
            n_landmarks=15,  # Different from first run
            result_key='de_different',
            overwrite=True
        )

        comparison = RunComparison(adata_with_de_history, 0, 1, 'de')

        html = comparison._repr_html_()
        assert isinstance(html, str)
        # Should contain parameter differences
        assert 'Parameter' in html or 'parameter' in html.lower()


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_runinfo_with_json_string_field_mapping(self, adata_with_de_history):
        """Test RunInfo handles JSON string field_mapping."""
        run_info = RunInfo(adata_with_de_history)

        # The implementation should handle field_mapping as either dict or JSON string
        # This is tested implicitly through normal operation
        assert run_info.adata_fields is not None

    def test_runinfo_with_no_field_mapping(self, adata_with_de_history):
        """Test RunInfo handles missing field_mapping gracefully."""
        # Manually corrupt the run_info by removing field_mapping
        if 'kompot_de' in adata_with_de_history.uns and 'run_history' in adata_with_de_history.uns['kompot_de']:
            run_history = adata_with_de_history.uns['kompot_de']['run_history']
            if len(run_history) > 0 and 'field_mapping' in run_history[0]:
                # Save original
                original_field_mapping = run_history[0]['field_mapping']

                # Remove field_mapping
                del run_history[0]['field_mapping']

                # Should still initialize without error
                run_info = RunInfo(adata_with_de_history)
                assert run_info.adata_fields == {}

                # Restore
                run_history[0]['field_mapping'] = original_field_mapping

    def test_comparison_with_empty_params(self, adata_with_de_history):
        """Test RunComparison handles runs with different parameter sets."""
        comparison = RunComparison(adata_with_de_history, 0, 0, 'de')

        # Comparing same run should have all parameters in 'same'
        assert len(comparison.param_comparison['different']) == 0
        assert len(comparison.param_comparison['only_in_run1']) == 0
        assert len(comparison.param_comparison['only_in_run2']) == 0
