"""Tests for RunInfo HTML rendering and edge cases."""

import json
import pytest
import numpy as np
import pandas as pd
import anndata as ad


class TestRunInfoHTML:
    """Cover the _repr_html_ branches in runinfo.py."""

    def _make_adata_with_da_run(self, result_key="kompot_da", sample_col=None):
        """Create AnnData with a DA run history entry."""
        np.random.seed(42)
        n = 30
        adata = ad.AnnData(np.random.randn(n, 5))
        adata.obsm["X_pca"] = np.random.randn(n, 3)
        adata.obs["condition"] = ["A"] * 15 + ["B"] * 15
        adata.obs_names = [f"cell_{i}" for i in range(n)]
        adata.var_names = [f"gene_{i}" for i in range(5)]

        # Add DA results
        prefix = f"{result_key}_A_to_B"
        adata.obs[f"{prefix}_lfc"] = np.random.randn(n)
        adata.obs[f"{prefix}_lfc_zscore"] = np.random.randn(n)
        adata.obs[f"{prefix}_neg_log10_lfc_ptp"] = np.abs(np.random.randn(n))
        adata.obs[f"{prefix}_lfc_direction"] = pd.Categorical(
            np.random.choice(["up", "down", "neutral"], n)
        )

        # Build run history
        from kompot.anndata.utils.params import build_params_dict
        from kompot.settings import (
            GPSettings,
            DAThresholdSettings,
            StorageSettings,
            OutputSettings,
        )

        params = build_params_dict(
            top_level=dict(
                groupby="condition",
                condition1="A",
                condition2="B",
                obsm_key="X_pca",
                sample_col=sample_col,
            ),
            gp=GPSettings(n_landmarks=None, batch_size=None),
            threshold=DAThresholdSettings(),
            storage=StorageSettings(result_key=result_key),
            output=OutputSettings(),
        )

        field_mapping = {
            f"{prefix}_lfc": {
                "location": "obs",
                "type": "lfc",
                "description": "Log fold change",
            },
            f"{prefix}_lfc_zscore": {
                "location": "obs",
                "type": "zscore",
                "description": "Z-score",
            },
        }

        run_info = {
            "timestamp": "2026-01-01T00:00:00",
            "function": "da",
            "result_key": result_key,
            "analysis_type": "da",
            "field_names": {
                "lfc_key": f"{prefix}_lfc",
                "zscore_key": f"{prefix}_lfc_zscore",
                "ptp_key": f"{prefix}_neg_log10_lfc_ptp",
                "direction_key": f"{prefix}_lfc_direction",
                "density_key_1": f"{result_key}_A_log_density",
                "density_key_2": f"{result_key}_B_log_density",
            },
            "uses_sample_variance": sample_col is not None,
            "has_groups": False,
            "params": params,
            "environment": {
                "python_version": "3.12",
                "package_versions": {"kompot": "0.7.0"},
            },
            "field_mapping": field_mapping,
        }

        adata.uns[result_key] = {
            "run_history": json.dumps([run_info]),
            "last_run_info": json.dumps(run_info),
            "anndata_fields": json.dumps(
                {
                    "obs": {
                        f"{prefix}_lfc": {"run_id": 0},
                        f"{prefix}_lfc_zscore": {"run_id": 0},
                    },
                }
            ),
        }
        return adata

    def test_repr_html_basic(self):
        adata = self._make_adata_with_da_run()
        from kompot import RunInfo

        info = RunInfo(adata, run_id=0, analysis_type="da")
        html = info._repr_html_()
        assert "condition1" in html or "groupby" in html
        assert "gp." in html  # nested params rendered hierarchically

    def test_repr_html_with_sample_variance(self):
        adata = self._make_adata_with_da_run(sample_col="sample")
        from kompot import RunInfo

        info = RunInfo(adata, run_id=0, analysis_type="da")
        html = info._repr_html_()
        assert "sample_col" in html

    def test_repr_html_environment_section(self):
        adata = self._make_adata_with_da_run()
        from kompot import RunInfo

        info = RunInfo(adata, run_id=0, analysis_type="da")
        html = info._repr_html_()
        assert "Environment" in html
        assert "kompot" in html

    def test_repr_html_fields_section(self):
        adata = self._make_adata_with_da_run()
        from kompot import RunInfo

        info = RunInfo(adata, run_id=0, analysis_type="da")
        html = info._repr_html_()
        assert "Fields" in html

    def test_repr_basic(self):
        adata = self._make_adata_with_da_run()
        from kompot import RunInfo

        info = RunInfo(adata, run_id=0, analysis_type="da")
        r = repr(info)
        assert "RunInfo" in r

    def test_get_data(self):
        adata = self._make_adata_with_da_run()
        from kompot import RunInfo

        info = RunInfo(adata, run_id=0, analysis_type="da")
        data = info.get_data()
        assert "params" in data
        assert "field_data" in data

    def test_get_summary(self):
        adata = self._make_adata_with_da_run()
        from kompot import RunInfo

        info = RunInfo(adata, run_id=0, analysis_type="da")
        summary = info.get_summary()
        assert "conditions" in summary
        assert summary["analysis_type"] == "da"

    def test_to_settings(self):
        adata = self._make_adata_with_da_run()
        from kompot import RunInfo

        info = RunInfo(adata, run_id=0, analysis_type="da")
        settings = info.to_settings()
        assert "gp" in settings
        assert "threshold" in settings

    def test_call_args(self):
        adata = self._make_adata_with_da_run()
        from kompot import RunInfo

        info = RunInfo(adata, run_id=0, analysis_type="da")
        kwargs = info.call_args()
        assert "groupby" in kwargs
        assert "gp" in kwargs
        assert kwargs["groupby"] == "condition"

    def test_compare_with(self):
        """Test RunComparison with two DA runs."""
        adata = self._make_adata_with_da_run(result_key="kompot_da")

        # Add a second run with different params
        from kompot.anndata.utils.params import build_params_dict
        from kompot.settings import (
            GPSettings,
            DAThresholdSettings,
            StorageSettings,
            OutputSettings,
        )

        params2 = build_params_dict(
            top_level=dict(
                groupby="condition",
                condition1="A",
                condition2="B",
                obsm_key="X_pca",
                sample_col=None,
            ),
            gp=GPSettings(n_landmarks=100, batch_size=50),
            threshold=DAThresholdSettings(lfc_threshold=2.0),
            storage=StorageSettings(result_key="kompot_da"),
            output=OutputSettings(),
        )

        run_info2 = {
            "timestamp": "2026-01-02T00:00:00",
            "function": "da",
            "result_key": "kompot_da",
            "analysis_type": "da",
            "field_names": {},
            "uses_sample_variance": False,
            "has_groups": False,
            "params": params2,
            "environment": {"python_version": "3.12"},
            "field_mapping": {},
        }

        history = json.loads(adata.uns["kompot_da"]["run_history"])
        history.append(run_info2)
        adata.uns["kompot_da"]["run_history"] = json.dumps(history)

        from kompot import RunInfo

        info = RunInfo(adata, run_id=0, analysis_type="da")
        comparison = info.compare_with(1)
        html = comparison._repr_html_()
        # Should show dotted diff keys
        assert "gp.n_landmarks" in html or "threshold.lfc_threshold" in html

    def test_compare_repr(self):
        """Test RunComparison __repr__."""
        adata = self._make_adata_with_da_run()

        from kompot.anndata.utils.params import build_params_dict
        from kompot.settings import (
            GPSettings,
            DAThresholdSettings,
            StorageSettings,
            OutputSettings,
        )

        params2 = build_params_dict(
            top_level=dict(
                groupby="condition",
                condition1="A",
                condition2="B",
                obsm_key="X_pca",
                sample_col=None,
            ),
            gp=GPSettings(),
            threshold=DAThresholdSettings(),
            storage=StorageSettings(result_key="kompot_da"),
            output=OutputSettings(),
        )
        run2 = {
            "timestamp": "2026-01-02T00:00:00",
            "function": "da",
            "result_key": "kompot_da",
            "analysis_type": "da",
            "field_names": {},
            "uses_sample_variance": False,
            "has_groups": False,
            "params": params2,
            "environment": {},
            "field_mapping": {},
        }
        history = json.loads(adata.uns["kompot_da"]["run_history"])
        history.append(run2)
        adata.uns["kompot_da"]["run_history"] = json.dumps(history)

        from kompot import RunInfo

        comp = RunInfo(adata, run_id=0, analysis_type="da").compare_with(1)
        r = repr(comp)
        assert "RunComparison" in r


class TestRunInfoEdgeCases:
    """Cover edge cases in RunInfo."""

    def test_no_run_history(self):
        adata = ad.AnnData(np.random.randn(10, 5))
        from kompot import RunInfo

        with pytest.raises(ValueError, match="No run history"):
            RunInfo(adata, analysis_type="da")

    def test_invalid_analysis_type(self):
        adata = ad.AnnData(np.random.randn(10, 5))
        from kompot import RunInfo

        with pytest.raises(ValueError, match="Invalid analysis_type"):
            RunInfo(adata, analysis_type="invalid")

    def test_auto_detect_de(self):
        """Test auto-detection of analysis type."""
        adata = ad.AnnData(np.random.randn(10, 5))
        adata.uns["kompot_de"] = {
            "run_history": json.dumps(
                [
                    {
                        "timestamp": "2026-01-01",
                        "function": "de",
                        "result_key": "kompot_de",
                        "analysis_type": "de",
                        "field_names": {},
                        "params": {
                            "groupby": "c",
                            "condition1": "A",
                            "condition2": "B",
                        },
                        "environment": {},
                        "field_mapping": {},
                        "uses_sample_variance": False,
                        "has_groups": False,
                    }
                ]
            ),
        }
        from kompot import RunInfo

        info = RunInfo(adata)
        assert info.analysis_type == "de"

    def test_negative_run_id(self):
        adata = ad.AnnData(np.random.randn(10, 5))
        adata.uns["kompot_da"] = {
            "run_history": json.dumps(
                [
                    {
                        "timestamp": "2026-01-01",
                        "function": "da",
                        "result_key": "kompot_da",
                        "analysis_type": "da",
                        "field_names": {},
                        "params": {
                            "groupby": "c",
                            "condition1": "A",
                            "condition2": "B",
                        },
                        "environment": {},
                        "field_mapping": {},
                        "uses_sample_variance": False,
                        "has_groups": False,
                    }
                ]
            ),
        }
        from kompot import RunInfo

        info = RunInfo(adata, run_id=-1, analysis_type="da")
        assert info.adjusted_run_id == 0
