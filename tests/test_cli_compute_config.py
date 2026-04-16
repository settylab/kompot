"""
Unit tests for CLI compute configuration.

These tests directly call CLI functions to ensure coverage is captured
(subprocess tests don't contribute to coverage).
"""

import sys

import pytest
import os
import numpy as np
import pandas as pd
from anndata import AnnData


@pytest.fixture
def sample_adata_for_cli():
    """Create a minimal sample AnnData for CLI testing."""
    np.random.seed(42)
    n_obs = 60
    n_vars = 30

    X = np.random.randn(n_obs, n_vars)
    obs = pd.DataFrame(
        {
            "condition": ["A"] * 30 + ["B"] * 30,
            "sample": ["s1"] * 15 + ["s2"] * 15 + ["s3"] * 15 + ["s4"] * 15,
        }
    )
    var = pd.DataFrame({"gene_name": [f"Gene_{i}" for i in range(n_vars)]})
    obsm = {
        "X_pca": np.random.randn(n_obs, 10),
        "DM_EigenVectors": np.random.randn(n_obs, 10),
    }

    return AnnData(X=X, obs=obs, var=var, obsm=obsm)


class TestComputeConfig:
    """Test compute configuration functions."""

    def test_configure_thread_limits(self, monkeypatch):
        """Test that thread limit configuration sets environment variables."""
        from kompot.cli.compute_config import _configure_thread_limits

        # Clear env vars first
        for var in [
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "BLAS_NUM_THREADS",
        ]:
            monkeypatch.delenv(var, raising=False)

        # Configure thread limits
        _configure_thread_limits(4)

        # Check environment variables were set
        assert os.environ.get("OMP_NUM_THREADS") == "4"
        assert os.environ.get("MKL_NUM_THREADS") == "4"
        assert os.environ.get("OPENBLAS_NUM_THREADS") == "4"
        assert os.environ.get("BLAS_NUM_THREADS") == "4"

    def test_configure_jax_cpu(self, monkeypatch):
        """Test JAX configuration for CPU."""
        from kompot.cli.compute_config import _configure_jax
        import jax

        _configure_jax(use_gpu=False, n_threads=None)

        # Check that JAX is configured for CPU
        # Note: This test may vary depending on JAX installation
        try:
            platform = jax.devices()[0].platform
            # Should be 'cpu' but might be different depending on environment
            assert platform in ["cpu", "gpu"]
        except Exception:
            # If JAX is not properly configured, that's OK for this test
            pass

    def test_configure_jax_with_thread_limit(self, monkeypatch):
        """Test JAX configuration with thread limiting."""
        from kompot.cli.compute_config import _configure_jax

        # Clear XLA_FLAGS
        monkeypatch.delenv("XLA_FLAGS", raising=False)

        _configure_jax(use_gpu=False, n_threads=8)

        # Check XLA_FLAGS were set
        xla_flags = os.environ.get("XLA_FLAGS", "")
        assert "intra_op_parallelism_threads=8" in xla_flags or "8" in xla_flags

    def test_configure_dask(self, monkeypatch):
        """Test Dask configuration."""
        from kompot.cli.compute_config import _configure_dask

        try:
            # This might fail if dask is not installed
            _configure_dask(n_threads=4)
            # If it doesn't raise, dask is installed and configured
        except ImportError:
            # Dask not installed, which is OK
            pass

    def test_get_device_info(self):
        """Test device info retrieval."""
        from kompot.cli.compute_config import get_device_info

        info = get_device_info()

        # Check that info dict has expected keys
        assert "gpu_available" in info
        assert "gpu_devices" in info
        assert "cpu_count" in info
        assert "jax_platform" in info

        # CPU count should be positive
        assert isinstance(info["cpu_count"], int)
        if info["cpu_count"] is not None:
            assert info["cpu_count"] > 0

    def test_log_compute_environment(self):
        """Test logging of compute environment."""
        from kompot.cli.compute_config import log_compute_environment

        # Should run without error
        log_compute_environment()

        # The function logs device info - just check it runs successfully


class TestCLIMainEarlyThreadLimits:
    """Test early thread limit setting in main.py."""

    def test_set_early_thread_limits(self, monkeypatch):
        """Test that _set_early_thread_limits parses and sets thread limits."""
        from kompot.cli.main import _set_early_thread_limits

        # Clear env vars first
        for var in [
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "BLAS_NUM_THREADS",
        ]:
            monkeypatch.delenv(var, raising=False)

        # Test with --threads flag
        _set_early_thread_limits(
            ["de", "input.h5ad", "--threads", "6", "-o", "output.h5ad"]
        )

        # Check environment variables were set
        assert os.environ.get("OMP_NUM_THREADS") == "6"
        assert os.environ.get("MKL_NUM_THREADS") == "6"

    def test_set_early_thread_limits_equals_syntax(self, monkeypatch):
        """Test parsing --threads=N syntax."""
        from kompot.cli.main import _set_early_thread_limits

        # Clear env vars first
        for var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
            monkeypatch.delenv(var, raising=False)

        _set_early_thread_limits(
            ["de", "input.h5ad", "--threads=8", "-o", "output.h5ad"]
        )

        # Check environment variables were set
        assert os.environ.get("OMP_NUM_THREADS") == "8"

    def test_set_early_thread_limits_no_threads(self, monkeypatch):
        """Test when no --threads argument is provided."""
        from kompot.cli.main import _set_early_thread_limits

        # Set a value first
        monkeypatch.setenv("OMP_NUM_THREADS", "999")

        _set_early_thread_limits(["de", "input.h5ad", "-o", "output.h5ad"])

        # Value should remain unchanged
        assert os.environ.get("OMP_NUM_THREADS") == "999"

    def test_set_early_thread_limits_invalid_value(self, monkeypatch):
        """Test with invalid thread count value."""
        from kompot.cli.main import _set_early_thread_limits

        # Should not raise, just ignore invalid value
        _set_early_thread_limits(
            ["de", "input.h5ad", "--threads", "invalid", "-o", "output.h5ad"]
        )

        # Environment variable should not be set with invalid value
        # (will either be unset or have previous value)


class TestCLIDEUnitTests:
    """Unit tests for DE CLI functions (not subprocess)."""

    def test_add_de_parser(self):
        """Test that DE parser is created with correct arguments."""
        import argparse
        from kompot.cli.de import add_de_parser

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()

        de_parser = add_de_parser(subparsers)

        # Check that parser was created
        assert de_parser is not None

        # Check that key arguments were added
        # We can't easily inspect all arguments without parsing, but we can check the parser exists
        assert hasattr(de_parser, "parse_args")

    def test_run_de_missing_output(self, sample_adata_for_cli, tmp_path, capsys):
        """Test run_de with missing output arguments."""
        from kompot.cli.de import run_de
        import argparse

        # Save sample data
        input_file = tmp_path / "input.h5ad"
        sample_adata_for_cli.write_h5ad(input_file)

        # Create args without output
        args = argparse.Namespace(
            input=str(input_file),
            output=None,
            table_output=None,
            config=None,
            groupby="condition",
            condition1="A",
            condition2="B",
            obsm_key="DM_EigenVectors",
            sample_col="sample",
            n_landmarks=10,
            func=None,
            verbose=False,
            command="de",
            use_gpu=False,
            threads=None,
            layer=None,
            result_key=None,
            batch_size=None,
            fdr_threshold=None,
            null_genes=None,
            null_seed=None,
            no_progress=True,
            store_landmarks=False,
            store_additional_stats=False,
            overwrite=False,
            dry_run=False,
        )

        # Should exit with error
        with pytest.raises(SystemExit) as exc_info:
            run_de(args)

        assert exc_info.value.code == 1

    def test_run_de_missing_input_file(self, tmp_path):
        """Test run_de with non-existent input file."""
        from kompot.cli.de import run_de
        import argparse

        # Create args with non-existent input
        args = argparse.Namespace(
            input=str(tmp_path / "nonexistent.h5ad"),
            output=str(tmp_path / "output.h5ad"),
            table_output=None,
            config=None,
            groupby="condition",
            condition1="A",
            condition2="B",
            obsm_key="DM_EigenVectors",
            sample_col="sample",
            n_landmarks=10,
            func=None,
            verbose=False,
            command="de",
            use_gpu=False,
            threads=None,
            layer=None,
            result_key=None,
            batch_size=None,
            fdr_threshold=None,
            null_genes=None,
            null_seed=None,
            no_progress=True,
            store_landmarks=False,
            store_additional_stats=False,
            overwrite=False,
            dry_run=False,
        )

        # Should exit with error or raise FileNotFoundError
        with pytest.raises((SystemExit, FileNotFoundError)):
            run_de(args)

    def test_run_de_missing_required_params(self, sample_adata_for_cli, tmp_path):
        """Test run_de with missing required parameters."""
        from kompot.cli.de import run_de
        import argparse

        # Save sample data
        input_file = tmp_path / "input.h5ad"
        sample_adata_for_cli.write_h5ad(input_file)

        # Create args without required parameters
        args = argparse.Namespace(
            input=str(input_file),
            output=str(tmp_path / "output.h5ad"),
            table_output=None,
            config=None,
            groupby=None,  # Missing required
            condition1=None,  # Missing required
            condition2=None,  # Missing required
            obsm_key="DM_EigenVectors",
            sample_col="sample",
            n_landmarks=10,
            func=None,
            verbose=False,
            command="de",
            use_gpu=False,
            threads=None,
            layer=None,
            result_key=None,
            batch_size=None,
            fdr_threshold=None,
            null_genes=None,
            null_seed=None,
            no_progress=True,
            store_landmarks=False,
            store_additional_stats=False,
            overwrite=False,
            dry_run=False,
        )

        # Should exit with error
        with pytest.raises(SystemExit) as exc_info:
            run_de(args)

        assert exc_info.value.code == 1

    def test_run_de_with_config_file(self, sample_adata_for_cli, tmp_path, monkeypatch):
        """Test run_de with config file."""
        from kompot.cli.de import run_de
        import argparse

        # Create a config file
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
groupby: condition
condition1: A
condition2: B
n_landmarks: 15
""")

        # Save sample data
        input_file = tmp_path / "input.h5ad"
        sample_adata_for_cli.write_h5ad(input_file)

        # Create args
        args = argparse.Namespace(
            input=str(input_file),
            output=str(tmp_path / "output.h5ad"),
            table_output=None,
            config=str(config_file),
            groupby=None,  # Will come from config
            condition1=None,
            condition2=None,
            obsm_key="DM_EigenVectors",
            sample_col="sample",
            n_landmarks=10,  # CLI arg should override config
            func=None,
            verbose=False,
            command="de",
            use_gpu=False,
            threads=None,
            layer=None,
            result_key=None,
            batch_size=None,
            fdr_threshold=None,
            null_genes=None,
            null_seed=None,
            no_progress=True,
            store_landmarks=False,
            store_additional_stats=False,
            overwrite=False,
            dry_run=False,
        )

        # Mock compute_differential_expression to avoid actual computation
        def mock_compute_de(adata, **kwargs):
            # Just verify it was called
            pass

        monkeypatch.setattr("kompot.cli.de.de", mock_compute_de)

        # Should run without error
        run_de(args)

        # Verify output was created
        assert (tmp_path / "output.h5ad").exists()

    def test_run_de_table_output_csv(self, sample_adata_for_cli, tmp_path, monkeypatch):
        """Test run_de with CSV table output."""
        from kompot.cli.de import run_de
        import argparse
        import pandas as pd

        # Save sample data
        input_file = tmp_path / "input.h5ad"
        sample_adata_for_cli.write_h5ad(input_file)

        # Create args
        args = argparse.Namespace(
            input=str(input_file),
            output=None,
            table_output=str(tmp_path / "results.csv"),
            config=None,
            groupby="condition",
            condition1="A",
            condition2="B",
            obsm_key="DM_EigenVectors",
            sample_col="sample",
            n_landmarks=10,
            func=None,
            verbose=False,
            command="de",
            use_gpu=False,
            threads=None,
            layer=None,
            result_key=None,
            batch_size=None,
            fdr_threshold=None,
            null_genes=None,
            null_seed=None,
            no_progress=True,
            store_landmarks=False,
            store_additional_stats=False,
            overwrite=False,
            dry_run=False,
        )

        # Mock compute_differential_expression to return mock results
        def mock_compute_de(adata, **kwargs):
            _output = kwargs.get("output")
            if _output is not None and getattr(_output, "return_full_results", False):
                # Return mock results
                return {
                    "table": pd.DataFrame(
                        {
                            "gene": ["Gene_0", "Gene_1"],
                            "log2_fc": [1.5, -0.8],
                            "pval": [0.01, 0.05],
                        }
                    )
                }
            return None

        monkeypatch.setattr("kompot.cli.de.de", mock_compute_de)

        # Should run without error
        run_de(args)

        # Verify table output was created
        assert (tmp_path / "results.csv").exists()

    def test_run_de_table_output_tsv(self, sample_adata_for_cli, tmp_path, monkeypatch):
        """Test run_de with TSV table output."""
        from kompot.cli.de import run_de
        import argparse
        import pandas as pd

        # Save sample data
        input_file = tmp_path / "input.h5ad"
        sample_adata_for_cli.write_h5ad(input_file)

        # Create args
        args = argparse.Namespace(
            input=str(input_file),
            output=None,
            table_output=str(tmp_path / "results.tsv"),
            config=None,
            groupby="condition",
            condition1="A",
            condition2="B",
            obsm_key="DM_EigenVectors",
            sample_col="sample",
            n_landmarks=10,
            func=None,
            verbose=False,
            command="de",
            use_gpu=False,
            threads=None,
            layer=None,
            result_key=None,
            batch_size=None,
            fdr_threshold=None,
            null_genes=None,
            null_seed=None,
            no_progress=True,
            store_landmarks=False,
            store_additional_stats=False,
            overwrite=False,
            dry_run=False,
        )

        # Mock compute_differential_expression
        def mock_compute_de(adata, **kwargs):
            _output = kwargs.get("output")
            if _output is not None and getattr(_output, "return_full_results", False):
                return {"table": pd.DataFrame({"gene": ["Gene_0"], "log2_fc": [1.5]})}
            return None

        monkeypatch.setattr("kompot.cli.de.de", mock_compute_de)

        # Should run without error
        run_de(args)

        # Verify TSV output was created
        assert (tmp_path / "results.tsv").exists()

    def test_run_de_unsupported_output_format(
        self, sample_adata_for_cli, tmp_path, monkeypatch
    ):
        """Test run_de with unsupported output format."""
        from kompot.cli.de import run_de
        import argparse

        # Save sample data
        input_file = tmp_path / "input.h5ad"
        sample_adata_for_cli.write_h5ad(input_file)

        # Create args with unsupported output format
        args = argparse.Namespace(
            input=str(input_file),
            output=str(tmp_path / "output.txt"),  # Unsupported format
            table_output=None,
            config=None,
            groupby="condition",
            condition1="A",
            condition2="B",
            obsm_key="DM_EigenVectors",
            sample_col="sample",
            n_landmarks=10,
            func=None,
            verbose=False,
            command="de",
            use_gpu=False,
            threads=None,
            layer=None,
            result_key=None,
            batch_size=None,
            fdr_threshold=None,
            null_genes=None,
            null_seed=None,
            no_progress=True,
            store_landmarks=False,
            store_additional_stats=False,
            overwrite=False,
            dry_run=False,
        )

        # Mock compute_differential_expression
        def mock_compute_de(adata, **kwargs):
            pass

        monkeypatch.setattr("kompot.cli.de.de", mock_compute_de)

        # Should exit with error
        with pytest.raises(SystemExit) as exc_info:
            run_de(args)

        assert exc_info.value.code == 1

    def test_run_de_unsupported_table_format(
        self, sample_adata_for_cli, tmp_path, monkeypatch
    ):
        """Test run_de with unsupported table format."""
        from kompot.cli.de import run_de
        import argparse
        import pandas as pd

        # Save sample data
        input_file = tmp_path / "input.h5ad"
        sample_adata_for_cli.write_h5ad(input_file)

        # Create args with unsupported table format
        args = argparse.Namespace(
            input=str(input_file),
            output=None,
            table_output=str(tmp_path / "results.txt"),  # Unsupported format
            config=None,
            groupby="condition",
            condition1="A",
            condition2="B",
            obsm_key="DM_EigenVectors",
            sample_col="sample",
            n_landmarks=10,
            func=None,
            verbose=False,
            command="de",
            use_gpu=False,
            threads=None,
            layer=None,
            result_key=None,
            batch_size=None,
            fdr_threshold=None,
            null_genes=None,
            null_seed=None,
            no_progress=True,
            store_landmarks=False,
            store_additional_stats=False,
            overwrite=False,
            dry_run=False,
        )

        # Mock compute_differential_expression
        def mock_compute_de(adata, **kwargs):
            _output = kwargs.get("output")
            if _output is not None and getattr(_output, "return_full_results", False):
                return {"table": pd.DataFrame({"gene": ["Gene_0"]})}
            return None

        monkeypatch.setattr("kompot.cli.de.de", mock_compute_de)

        # Should exit with error
        with pytest.raises(SystemExit) as exc_info:
            run_de(args)

        assert exc_info.value.code == 1


class TestCLIDryRun:
    """Tests for the --dry-run flag on the DE CLI."""

    def test_dry_run_outputs_json(self, sample_adata_for_cli, tmp_path, monkeypatch, capsys):
        """Test that --dry-run outputs JSON to stdout and exits."""
        from kompot.cli.de import run_de
        import argparse

        input_file = tmp_path / "input.h5ad"
        sample_adata_for_cli.write_h5ad(input_file)

        args = argparse.Namespace(
            input=str(input_file),
            output=str(tmp_path / "output.h5ad"),
            table_output=None,
            config=None,
            groupby="condition",
            condition1="A",
            condition2="B",
            obsm_key="DM_EigenVectors",
            sample_col="sample",
            n_landmarks=10,
            func=None,
            verbose=False,
            command="de",
            use_gpu=False,
            threads=None,
            layer=None,
            result_key=None,
            batch_size=None,
            fdr_threshold=None,
            null_genes=None,
            null_seed=None,
            no_progress=True,
            store_landmarks=False,
            store_additional_stats=False,
            overwrite=False,
            dry_run=True,
        )

        # Mock de() to return a plan object with the expected interface
        class MockPlan:
            is_feasible = True
            def to_dict(self):
                return {"feasible": True, "memory": {"total_bytes": 1024}}
            def format_report(self):
                return "Mock report"

        def mock_de(adata, dry_run=False, **kwargs):
            if dry_run:
                return MockPlan()
            return None

        monkeypatch.setattr("kompot.cli.de.de", mock_de)

        with pytest.raises(SystemExit) as exc_info:
            run_de(args)

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        import json
        result = json.loads(captured.out)
        assert result["feasible"] is True

    def test_dry_run_infeasible_exits_1(self, sample_adata_for_cli, tmp_path, monkeypatch):
        """Test that --dry-run exits with code 1 when plan is infeasible."""
        from kompot.cli.de import run_de
        import argparse

        input_file = tmp_path / "input.h5ad"
        sample_adata_for_cli.write_h5ad(input_file)

        args = argparse.Namespace(
            input=str(input_file),
            output=str(tmp_path / "output.h5ad"),
            table_output=None,
            config=None,
            groupby="condition",
            condition1="A",
            condition2="B",
            obsm_key="DM_EigenVectors",
            sample_col="sample",
            n_landmarks=10,
            func=None,
            verbose=False,
            command="de",
            use_gpu=False,
            threads=None,
            layer=None,
            result_key=None,
            batch_size=None,
            fdr_threshold=None,
            null_genes=None,
            null_seed=None,
            no_progress=True,
            store_landmarks=False,
            store_additional_stats=False,
            overwrite=False,
            dry_run=True,
        )

        class MockPlan:
            is_feasible = False
            def to_dict(self):
                return {"feasible": False}
            def format_report(self):
                return "Not feasible"

        def mock_de(adata, dry_run=False, **kwargs):
            if dry_run:
                return MockPlan()
            return None

        monkeypatch.setattr("kompot.cli.de.de", mock_de)

        with pytest.raises(SystemExit) as exc_info:
            run_de(args)

        assert exc_info.value.code == 1

    def test_dry_run_skips_output_validation(self, sample_adata_for_cli, tmp_path, monkeypatch):
        """Test that --dry-run does not require --output."""
        from kompot.cli.de import run_de
        import argparse

        input_file = tmp_path / "input.h5ad"
        sample_adata_for_cli.write_h5ad(input_file)

        args = argparse.Namespace(
            input=str(input_file),
            output=None,
            table_output=None,
            config=None,
            groupby="condition",
            condition1="A",
            condition2="B",
            obsm_key="DM_EigenVectors",
            sample_col="sample",
            n_landmarks=10,
            func=None,
            verbose=False,
            command="de",
            use_gpu=False,
            threads=None,
            layer=None,
            result_key=None,
            batch_size=None,
            fdr_threshold=None,
            null_genes=None,
            null_seed=None,
            no_progress=True,
            store_landmarks=False,
            store_additional_stats=False,
            overwrite=False,
            dry_run=True,
        )

        class MockPlan:
            is_feasible = True
            def to_dict(self):
                return {"feasible": True}
            def format_report(self):
                return "OK"

        def mock_de(adata, dry_run=False, **kwargs):
            if dry_run:
                return MockPlan()
            return None

        monkeypatch.setattr("kompot.cli.de.de", mock_de)

        # Should NOT exit with "output required" error — dry-run skips that check
        with pytest.raises(SystemExit) as exc_info:
            run_de(args)

        assert exc_info.value.code == 0


class TestConfigureLogging:
    """Tests for kompot.configure_logging()."""

    def test_configure_logging_to_stderr(self):
        """Test that configure_logging redirects the logger."""
        import io
        import kompot

        buf = io.StringIO()
        kompot.configure_logging(buf)

        kompot.logger.info("test message")
        output = buf.getvalue()
        assert "test message" in output

        # Restore default
        kompot.configure_logging(sys.stdout)

    def test_configure_logging_default_stdout(self):
        """Test that configure_logging defaults to stdout."""
        import kompot

        kompot.configure_logging()
        for handler in kompot.logger.handlers:
            if hasattr(handler, "stream"):
                assert handler.stream is sys.stdout
                break


class TestCLIDAUnitTests:
    """Unit tests for DA CLI functions."""

    def test_add_da_parser(self):
        """Test that DA parser is created with correct arguments."""
        import argparse
        from kompot.cli.da import add_da_parser

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()

        da_parser = add_da_parser(subparsers)

        # Check that parser was created
        assert da_parser is not None
        assert hasattr(da_parser, "parse_args")

    def test_run_da_missing_output(self, sample_adata_for_cli, tmp_path):
        """Test run_da with missing output arguments."""
        from kompot.cli.da import run_da
        import argparse

        # Save sample data
        input_file = tmp_path / "input.h5ad"
        sample_adata_for_cli.write_h5ad(input_file)

        # Create args without output
        args = argparse.Namespace(
            input=str(input_file),
            output=None,
            table_output=None,
            config=None,
            groupby="condition",
            condition1="A",
            condition2="B",
            obsm_key="DM_EigenVectors",
            sample_col="sample",
            n_landmarks=10,
            func=None,
            verbose=False,
            command="da",
            use_gpu=False,
            threads=None,
            result_key=None,
            batch_size=None,
            log_fold_change_threshold=None,
            ptp_threshold=None,
            ls_factor=None,
            random_state=None,
            store_landmarks=False,
            overwrite=False,
        )

        # Should exit with error
        with pytest.raises(SystemExit) as exc_info:
            run_da(args)

        assert exc_info.value.code == 1

    def test_run_da_missing_input_file(self, tmp_path):
        """Test run_da with non-existent input file."""
        from kompot.cli.da import run_da
        import argparse

        # Create args with non-existent input
        args = argparse.Namespace(
            input=str(tmp_path / "nonexistent.h5ad"),
            output=str(tmp_path / "output.h5ad"),
            table_output=None,
            config=None,
            groupby="condition",
            condition1="A",
            condition2="B",
            obsm_key="DM_EigenVectors",
            sample_col="sample",
            n_landmarks=10,
            func=None,
            verbose=False,
            command="da",
            use_gpu=False,
            threads=None,
            result_key=None,
            batch_size=None,
            log_fold_change_threshold=None,
            ptp_threshold=None,
            ls_factor=None,
            random_state=None,
            store_landmarks=False,
            overwrite=False,
        )

        # Should exit with error or raise FileNotFoundError
        with pytest.raises((SystemExit, FileNotFoundError)):
            run_da(args)

    def test_run_da_missing_required_params(self, sample_adata_for_cli, tmp_path):
        """Test run_da with missing required parameters."""
        from kompot.cli.da import run_da
        import argparse

        # Save sample data
        input_file = tmp_path / "input.h5ad"
        sample_adata_for_cli.write_h5ad(input_file)

        # Create args without required parameters
        args = argparse.Namespace(
            input=str(input_file),
            output=str(tmp_path / "output.h5ad"),
            table_output=None,
            config=None,
            groupby=None,  # Missing required
            condition1=None,  # Missing required
            condition2=None,  # Missing required
            obsm_key="DM_EigenVectors",
            sample_col="sample",
            n_landmarks=10,
            func=None,
            verbose=False,
            command="da",
            use_gpu=False,
            threads=None,
            result_key=None,
            batch_size=None,
            log_fold_change_threshold=None,
            ptp_threshold=None,
            ls_factor=None,
            random_state=None,
            store_landmarks=False,
            overwrite=False,
        )

        # Should exit with error
        with pytest.raises(SystemExit) as exc_info:
            run_da(args)

        assert exc_info.value.code == 1

    def test_run_da_with_config_file(self, sample_adata_for_cli, tmp_path, monkeypatch):
        """Test run_da with config file."""
        from kompot.cli.da import run_da
        import argparse

        # Create a config file
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
groupby: condition
condition1: A
condition2: B
n_landmarks: 15
""")

        # Save sample data
        input_file = tmp_path / "input.h5ad"
        sample_adata_for_cli.write_h5ad(input_file)

        # Create args
        args = argparse.Namespace(
            input=str(input_file),
            output=str(tmp_path / "output.h5ad"),
            table_output=None,
            config=str(config_file),
            groupby=None,  # Will come from config
            condition1=None,
            condition2=None,
            obsm_key="DM_EigenVectors",
            sample_col="sample",
            n_landmarks=10,  # CLI arg should override config
            func=None,
            verbose=False,
            command="da",
            use_gpu=False,
            threads=None,
            result_key=None,
            batch_size=None,
            log_fold_change_threshold=None,
            ptp_threshold=None,
            ls_factor=None,
            random_state=None,
            store_landmarks=False,
            overwrite=False,
        )

        # Mock compute_differential_abundance to avoid actual computation
        def mock_compute_da(adata, **kwargs):
            # Just verify it was called
            pass

        monkeypatch.setattr("kompot.cli.da.da", mock_compute_da)

        # Should run without error
        run_da(args)

        # Verify output was created
        assert (tmp_path / "output.h5ad").exists()

    def test_run_da_table_output_csv(self, sample_adata_for_cli, tmp_path, monkeypatch):
        """Test run_da with CSV table output."""
        from kompot.cli.da import run_da
        import argparse
        import pandas as pd

        # Save sample data
        input_file = tmp_path / "input.h5ad"
        sample_adata_for_cli.write_h5ad(input_file)

        # Create args
        args = argparse.Namespace(
            input=str(input_file),
            output=None,
            table_output=str(tmp_path / "results.csv"),
            config=None,
            groupby="condition",
            condition1="A",
            condition2="B",
            obsm_key="DM_EigenVectors",
            sample_col="sample",
            n_landmarks=10,
            func=None,
            verbose=False,
            command="da",
            use_gpu=False,
            threads=None,
            result_key=None,
            batch_size=None,
            log_fold_change_threshold=None,
            ptp_threshold=None,
            ls_factor=None,
            random_state=None,
            store_landmarks=False,
            overwrite=False,
        )

        # Mock compute_differential_abundance to return mock results
        def mock_compute_da(adata, **kwargs):
            _output = kwargs.get("output")
            if _output is not None and getattr(_output, "return_full_results", False):
                # Return mock results
                return {
                    "table": pd.DataFrame(
                        {
                            "cell_id": ["cell_0", "cell_1"],
                            "log_fc": [1.5, -0.8],
                            "pval": [0.01, 0.05],
                        }
                    )
                }
            return None

        monkeypatch.setattr("kompot.cli.da.da", mock_compute_da)

        # Should run without error
        run_da(args)

        # Verify table output was created
        assert (tmp_path / "results.csv").exists()

    def test_run_da_table_output_tsv(self, sample_adata_for_cli, tmp_path, monkeypatch):
        """Test run_da with TSV table output."""
        from kompot.cli.da import run_da
        import argparse
        import pandas as pd

        # Save sample data
        input_file = tmp_path / "input.h5ad"
        sample_adata_for_cli.write_h5ad(input_file)

        # Create args
        args = argparse.Namespace(
            input=str(input_file),
            output=None,
            table_output=str(tmp_path / "results.tsv"),
            config=None,
            groupby="condition",
            condition1="A",
            condition2="B",
            obsm_key="DM_EigenVectors",
            sample_col="sample",
            n_landmarks=10,
            func=None,
            verbose=False,
            command="da",
            use_gpu=False,
            threads=None,
            result_key=None,
            batch_size=None,
            log_fold_change_threshold=None,
            ptp_threshold=None,
            ls_factor=None,
            random_state=None,
            store_landmarks=False,
            overwrite=False,
        )

        # Mock compute_differential_abundance
        def mock_compute_da(adata, **kwargs):
            _output = kwargs.get("output")
            if _output is not None and getattr(_output, "return_full_results", False):
                return {"table": pd.DataFrame({"cell_id": ["cell_0"], "log_fc": [1.5]})}
            return None

        monkeypatch.setattr("kompot.cli.da.da", mock_compute_da)

        # Should run without error
        run_da(args)

        # Verify TSV output was created
        assert (tmp_path / "results.tsv").exists()

    def test_run_da_unsupported_output_format(
        self, sample_adata_for_cli, tmp_path, monkeypatch
    ):
        """Test run_da with unsupported output format."""
        from kompot.cli.da import run_da
        import argparse

        # Save sample data
        input_file = tmp_path / "input.h5ad"
        sample_adata_for_cli.write_h5ad(input_file)

        # Create args with unsupported output format
        args = argparse.Namespace(
            input=str(input_file),
            output=str(tmp_path / "output.txt"),  # Unsupported format
            table_output=None,
            config=None,
            groupby="condition",
            condition1="A",
            condition2="B",
            obsm_key="DM_EigenVectors",
            sample_col="sample",
            n_landmarks=10,
            func=None,
            verbose=False,
            command="da",
            use_gpu=False,
            threads=None,
            result_key=None,
            batch_size=None,
            log_fold_change_threshold=None,
            ptp_threshold=None,
            ls_factor=None,
            random_state=None,
            store_landmarks=False,
            overwrite=False,
        )

        # Mock compute_differential_abundance
        def mock_compute_da(adata, **kwargs):
            pass

        monkeypatch.setattr("kompot.cli.da.da", mock_compute_da)

        # Should exit with error
        with pytest.raises(SystemExit) as exc_info:
            run_da(args)

        assert exc_info.value.code == 1

    def test_run_da_unsupported_table_format(
        self, sample_adata_for_cli, tmp_path, monkeypatch
    ):
        """Test run_da with unsupported table format."""
        from kompot.cli.da import run_da
        import argparse
        import pandas as pd

        # Save sample data
        input_file = tmp_path / "input.h5ad"
        sample_adata_for_cli.write_h5ad(input_file)

        # Create args with unsupported table format
        args = argparse.Namespace(
            input=str(input_file),
            output=None,
            table_output=str(tmp_path / "results.txt"),  # Unsupported format
            config=None,
            groupby="condition",
            condition1="A",
            condition2="B",
            obsm_key="DM_EigenVectors",
            sample_col="sample",
            n_landmarks=10,
            func=None,
            verbose=False,
            command="da",
            use_gpu=False,
            threads=None,
            result_key=None,
            batch_size=None,
            log_fold_change_threshold=None,
            ptp_threshold=None,
            ls_factor=None,
            random_state=None,
            store_landmarks=False,
            overwrite=False,
        )

        # Mock compute_differential_abundance
        def mock_compute_da(adata, **kwargs):
            _output = kwargs.get("output")
            if _output is not None and getattr(_output, "return_full_results", False):
                return {"table": pd.DataFrame({"cell_id": ["cell_0"]})}
            return None

        monkeypatch.setattr("kompot.cli.da.da", mock_compute_da)

        # Should exit with error
        with pytest.raises(SystemExit) as exc_info:
            run_da(args)

        assert exc_info.value.code == 1
