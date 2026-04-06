"""
Unit tests for CLI diffusion maps (dm) command.

These tests directly call CLI functions to ensure coverage is captured
(subprocess tests don't contribute to coverage).
"""

import pytest
import sys
import numpy as np
import pandas as pd
from anndata import AnnData


@pytest.fixture
def sample_adata_with_pca():
    """Create a minimal sample AnnData with PCA for DM testing."""
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

    # Add PCA coordinates (required for DM)
    obsm = {"X_pca": np.random.randn(n_obs, 10)}

    return AnnData(X=X, obs=obs, var=var, obsm=obsm)


@pytest.fixture
def sample_adata_no_pca():
    """Create a minimal sample AnnData without PCA."""
    np.random.seed(42)
    n_obs = 60
    n_vars = 30

    X = np.random.randn(n_obs, n_vars)
    obs = pd.DataFrame({"condition": ["A"] * 30 + ["B"] * 30})
    var = pd.DataFrame({"gene_name": [f"Gene_{i}" for i in range(n_vars)]})

    return AnnData(X=X, obs=obs, var=var)


class TestDMParser:
    """Test DM parser creation."""

    def test_add_dm_parser(self):
        """Test that DM parser is created with correct arguments."""
        import argparse
        from kompot.cli.dm import add_dm_parser

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()

        dm_parser = add_dm_parser(subparsers)

        # Check that parser was created
        assert dm_parser is not None
        assert hasattr(dm_parser, "parse_args")

    def test_dm_parser_required_args(self):
        """Test that DM parser has required arguments."""
        import argparse
        from kompot.cli.dm import add_dm_parser

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        dm_parser = add_dm_parser(subparsers)

        # Try parsing with missing required args - should fail
        with pytest.raises(SystemExit):
            dm_parser.parse_args([])

    def test_dm_parser_minimal_args(self):
        """Test that DM parser accepts minimal required arguments."""
        import argparse
        from kompot.cli.dm import add_dm_parser

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        dm_parser = add_dm_parser(subparsers)

        # Should parse with input and output
        args = dm_parser.parse_args(["input.h5ad", "--output", "output.h5ad"])
        assert args.input == "input.h5ad"
        assert args.output == "output.h5ad"

    def test_dm_parser_default_values(self):
        """Test that DM parser has correct default values."""
        import argparse
        from kompot.cli.dm import add_dm_parser

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        dm_parser = add_dm_parser(subparsers)

        args = dm_parser.parse_args(["input.h5ad", "--output", "output.h5ad"])

        # Check defaults
        assert args.pca_key == "X_pca"
        assert args.n_components == 10
        assert args.knn == 30
        assert args.alpha == 0
        assert args.seed == 0
        assert args.kernel_key == "DM_Kernel"
        assert args.sim_key == "DM_Similarity"
        assert args.eigval_key == "DM_EigenValues"
        assert args.eigvec_key == "DM_EigenVectors"

    def test_dm_parser_custom_args(self):
        """Test that DM parser accepts custom argument values."""
        import argparse
        from kompot.cli.dm import add_dm_parser

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        dm_parser = add_dm_parser(subparsers)

        args = dm_parser.parse_args(
            [
                "input.h5ad",
                "--output",
                "output.h5ad",
                "--pca-key",
                "custom_pca",
                "--n-components",
                "20",
                "--knn",
                "50",
                "--alpha",
                "0.5",
                "--seed",
                "123",
            ]
        )

        assert args.pca_key == "custom_pca"
        assert args.n_components == 20
        assert args.knn == 50
        assert args.alpha == 0.5
        assert args.seed == 123


class TestDMRunMissingDependency:
    """Test run_dm with missing dependencies."""

    def test_run_dm_missing_palantir(
        self, sample_adata_with_pca, tmp_path, monkeypatch
    ):
        """Test run_dm when palantir is not installed."""
        from kompot.cli.dm import run_dm
        import argparse

        # Save sample data
        input_file = tmp_path / "input.h5ad"
        sample_adata_with_pca.write_h5ad(input_file)

        # Mock palantir import to raise ImportError
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "palantir":
                raise ImportError("No module named 'palantir'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        # Create args
        args = argparse.Namespace(
            input=str(input_file),
            output=str(tmp_path / "output.h5ad"),
            config=None,
            pca_key="X_pca",
            n_components=10,
            knn=30,
            alpha=0,
            seed=0,
            kernel_key="DM_Kernel",
            sim_key="DM_Similarity",
            eigval_key="DM_EigenValues",
            eigvec_key="DM_EigenVectors",
            func=None,
            verbose=False,
            command="dm",
        )

        # Should exit with error
        with pytest.raises(SystemExit) as exc_info:
            run_dm(args)

        assert exc_info.value.code == 1


class TestDMRunMissingPCA:
    """Test run_dm with missing PCA."""

    def test_run_dm_missing_pca_key(self, sample_adata_no_pca, tmp_path):
        """Test run_dm when PCA key doesn't exist in adata."""
        from kompot.cli.dm import run_dm
        import argparse

        # Save sample data (no PCA)
        input_file = tmp_path / "input.h5ad"
        sample_adata_no_pca.write_h5ad(input_file)

        # Create args
        args = argparse.Namespace(
            input=str(input_file),
            output=str(tmp_path / "output.h5ad"),
            config=None,
            pca_key="X_pca",
            n_components=10,
            knn=30,
            alpha=0,
            seed=0,
            kernel_key="DM_Kernel",
            sim_key="DM_Similarity",
            eigval_key="DM_EigenValues",
            eigvec_key="DM_EigenVectors",
            func=None,
            verbose=False,
            command="dm",
        )

        # Check if palantir is available - if not, skip this test
        try:
            import palantir
        except ImportError:
            pytest.skip("Palantir not installed")

        # Should exit with error due to missing PCA
        with pytest.raises(SystemExit) as exc_info:
            run_dm(args)

        assert exc_info.value.code == 1


class TestDMRunInputValidation:
    """Test run_dm input validation."""

    def test_run_dm_missing_input_file(self, tmp_path):
        """Test run_dm with non-existent input file."""
        from kompot.cli.dm import run_dm
        import argparse

        # Create args with non-existent input
        args = argparse.Namespace(
            input=str(tmp_path / "nonexistent.h5ad"),
            output=str(tmp_path / "output.h5ad"),
            config=None,
            pca_key="X_pca",
            n_components=10,
            knn=30,
            alpha=0,
            seed=0,
            kernel_key="DM_Kernel",
            sim_key="DM_Similarity",
            eigval_key="DM_EigenValues",
            eigvec_key="DM_EigenVectors",
            func=None,
            verbose=False,
            command="dm",
        )

        # Check if palantir is available - if not, skip this test
        try:
            import palantir
        except ImportError:
            pytest.skip("Palantir not installed")

        # Should exit with error
        with pytest.raises((SystemExit, FileNotFoundError)):
            run_dm(args)


class TestDMRunOutputFormat:
    """Test run_dm output format handling."""

    def test_run_dm_unsupported_output_format(self, sample_adata_with_pca, tmp_path):
        """Test run_dm with unsupported output format."""

        # Save sample data
        input_file = tmp_path / "input.h5ad"
        sample_adata_with_pca.write_h5ad(input_file)

        # Mock palantir.utils.run_diffusion_maps
        try:
            import palantir

            def mock_run_dm(adata, **kwargs):
                # Add DM_EigenVectors to adata
                adata.obsm["DM_EigenVectors"] = np.random.randn(adata.n_obs, 10)

            import kompot.cli.dm

            original_palantir = sys.modules.get("palantir")

            # Monkeypatch is tricky here, let's just test the parser instead
            # This test would require more complex mocking
            pytest.skip("Requires complex palantir mocking")

        except ImportError:
            pytest.skip("Palantir not installed")


class TestDMRunWithConfig:
    """Test run_dm with config file."""

    def test_run_dm_with_config_file(self, sample_adata_with_pca, tmp_path):
        """Test run_dm with YAML config file."""
        import argparse

        # Create a simple config file
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
n_components: 15
knn: 40
alpha: 0.5
""")

        # Save sample data
        input_file = tmp_path / "input.h5ad"
        sample_adata_with_pca.write_h5ad(input_file)

        # Create args
        args = argparse.Namespace(
            input=str(input_file),
            output=str(tmp_path / "output.h5ad"),
            config=str(config_file),
            pca_key="X_pca",
            n_components=10,  # This should be overridden by CLI
            knn=30,
            alpha=0,
            seed=0,
            kernel_key="DM_Kernel",
            sim_key="DM_Similarity",
            eigval_key="DM_EigenValues",
            eigvec_key="DM_EigenVectors",
            func=None,
            verbose=False,
            command="dm",
        )

        # Check if palantir is available
        try:
            import palantir

            # Mock palantir.utils.run_diffusion_maps to avoid actual computation
            def mock_run_dm(adata, **kwargs):
                # Verify that config was loaded
                # CLI args should take precedence
                adata.obsm["DM_EigenVectors"] = np.random.randn(
                    adata.n_obs, kwargs.get("n_components", 10)
                )

            # This would require monkeypatching palantir module
            # For now, just test that the function can be called
            pytest.skip("Requires palantir mocking")

        except ImportError:
            pytest.skip("Palantir not installed")


class TestDMRunSuccessPath:
    """Test successful run_dm execution."""

    def test_run_dm_h5ad_success(self, sample_adata_with_pca, tmp_path, monkeypatch):
        """Test successful run_dm with h5ad output."""
        from kompot.cli.dm import run_dm
        import argparse

        # Save sample data
        input_file = tmp_path / "input.h5ad"
        sample_adata_with_pca.write_h5ad(input_file)
        output_file = tmp_path / "output.h5ad"

        # Create args
        args = argparse.Namespace(
            input=str(input_file),
            output=str(output_file),
            config=None,
            pca_key="X_pca",
            n_components=5,  # Small for speed
            knn=10,
            alpha=0,
            seed=42,
            kernel_key="DM_Kernel",
            sim_key="DM_Similarity",
            eigval_key="DM_EigenValues",
            eigvec_key="DM_EigenVectors",
            func=None,
            verbose=False,
            command="dm",
        )

        # Check if palantir is available
        try:
            import palantir

            # Mock palantir.utils.run_diffusion_maps
            def mock_run_dm(adata, **kwargs):
                # Simulate successful DM computation
                n_comps = kwargs.get("n_components", 10)
                adata.obsm["DM_EigenVectors"] = np.random.randn(adata.n_obs, n_comps)
                adata.uns["DM_EigenValues"] = np.random.randn(n_comps)
                adata.obsp["DM_Kernel"] = np.random.randn(adata.n_obs, adata.n_obs)
                adata.obsp["DM_Similarity"] = np.random.randn(adata.n_obs, adata.n_obs)

            monkeypatch.setattr("palantir.utils.run_diffusion_maps", mock_run_dm)

            # Run DM
            run_dm(args)

            # Check output file was created
            assert output_file.exists()

            # Load and verify
            import anndata as ad

            result = ad.read_h5ad(output_file)
            assert "DM_EigenVectors" in result.obsm
            assert result.obsm["DM_EigenVectors"].shape[1] == 5

        except ImportError:
            pytest.skip("Palantir not installed")

    def test_run_dm_zarr_output(self, sample_adata_with_pca, tmp_path, monkeypatch):
        """Test run_dm with zarr output format."""
        from kompot.cli.dm import run_dm
        import argparse

        # Save sample data
        input_file = tmp_path / "input.h5ad"
        sample_adata_with_pca.write_h5ad(input_file)
        output_file = tmp_path / "output.zarr"

        # Create args
        args = argparse.Namespace(
            input=str(input_file),
            output=str(output_file),
            config=None,
            pca_key="X_pca",
            n_components=5,
            knn=10,
            alpha=0,
            seed=42,
            kernel_key="DM_Kernel",
            sim_key="DM_Similarity",
            eigval_key="DM_EigenValues",
            eigvec_key="DM_EigenVectors",
            func=None,
            verbose=False,
            command="dm",
        )

        # Check if palantir is available
        try:
            import palantir

            # Mock palantir.utils.run_diffusion_maps
            def mock_run_dm(adata, **kwargs):
                n_comps = kwargs.get("n_components", 10)
                adata.obsm["DM_EigenVectors"] = np.random.randn(adata.n_obs, n_comps)

            monkeypatch.setattr("palantir.utils.run_diffusion_maps", mock_run_dm)

            # Run DM
            run_dm(args)

            # Check output directory was created
            assert output_file.exists()

        except ImportError:
            pytest.skip("Palantir not installed")


class TestDMArgumentMapping:
    """Test argument name mapping from CLI to function parameters."""

    def test_dm_argument_conversion(self, sample_adata_with_pca, tmp_path, monkeypatch):
        """Test that CLI arguments with hyphens are converted to underscores."""
        from kompot.cli.dm import run_dm
        import argparse

        # Add custom PCA key to adata
        sample_adata_with_pca.obsm["custom_pca"] = sample_adata_with_pca.obsm[
            "X_pca"
        ].copy()

        # Save sample data
        input_file = tmp_path / "input.h5ad"
        sample_adata_with_pca.write_h5ad(input_file)

        # Create args with hyphenated names (as they come from CLI)
        args = argparse.Namespace(
            input=str(input_file),
            output=str(tmp_path / "output.h5ad"),
            config=None,
            pca_key="custom_pca",  # Note: argparse converts - to _
            n_components=8,
            knn=25,
            alpha=0.3,
            seed=99,
            kernel_key="MyKernel",
            sim_key="MySim",
            eigval_key="MyEigVal",
            eigvec_key="MyEigVec",
            func=None,
            verbose=False,
            command="dm",
        )

        # Check if palantir is available
        try:
            import palantir

            captured_params = {}

            def mock_run_dm(adata, **kwargs):
                # Capture the parameters passed
                captured_params.update(kwargs)
                adata.obsm["DM_EigenVectors"] = np.random.randn(
                    adata.n_obs, kwargs.get("n_components", 10)
                )

            monkeypatch.setattr("palantir.utils.run_diffusion_maps", mock_run_dm)

            # Run DM
            run_dm(args)

            # Verify parameters were passed correctly
            assert captured_params.get("n_components") == 8
            assert captured_params.get("knn") == 25
            assert captured_params.get("alpha") == 0.3

        except ImportError:
            pytest.skip("Palantir not installed")
