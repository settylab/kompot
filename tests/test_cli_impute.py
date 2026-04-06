"""Tests for the impute CLI command."""

import subprocess
import sys
import tempfile
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from anndata import AnnData


@pytest.fixture
def sample_adata():
    """Create a sample AnnData object for testing."""
    np.random.seed(42)
    n_obs, n_vars = 80, 20
    X = np.random.randn(n_obs, n_vars)
    obs = pd.DataFrame(
        {
            "condition": ["A"] * 40 + ["B"] * 40,
            "sample": ["s1"] * 20 + ["s2"] * 20 + ["s3"] * 20 + ["s4"] * 20,
        }
    )
    var = pd.DataFrame(index=[f"Gene_{i}" for i in range(n_vars)])
    adata = AnnData(X=X, obs=obs, var=var, obsm={"X_pca": np.random.randn(n_obs, 5)})
    return adata


@pytest.fixture
def temp_dir():
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir)


class TestCLIImpute:
    def test_impute_basic(self, sample_adata, temp_dir):
        """Test basic impute with h5ad output."""
        input_file = Path(temp_dir) / "input.h5ad"
        output_file = Path(temp_dir) / "output.h5ad"
        sample_adata.write_h5ad(input_file)

        cmd = [
            sys.executable,
            "-m",
            "kompot.cli",
            "impute",
            str(input_file),
            "-o",
            str(output_file),
            "--obsm-key",
            "X_pca",
            "--n-landmarks",
            "20",
            "--batch-size",
            "40",
            "--no-progress",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert output_file.exists()

        import anndata as ad

        adata_out = ad.read_h5ad(output_file)
        assert "kompot_impute_all_imputed" in adata_out.layers
        assert "kompot_impute_all_std" in adata_out.layers

    def test_impute_with_condition(self, sample_adata, temp_dir):
        """Test impute with condition selection."""
        input_file = Path(temp_dir) / "input.h5ad"
        output_file = Path(temp_dir) / "output.h5ad"
        sample_adata.write_h5ad(input_file)

        cmd = [
            sys.executable,
            "-m",
            "kompot.cli",
            "impute",
            str(input_file),
            "-o",
            str(output_file),
            "--obsm-key",
            "X_pca",
            "--groupby",
            "condition",
            "--condition",
            "A",
            "--n-landmarks",
            "20",
            "--no-progress",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        import anndata as ad

        adata_out = ad.read_h5ad(output_file)
        assert "kompot_impute_A_imputed" in adata_out.layers

    def test_impute_table_output(self, sample_adata, temp_dir):
        """Test impute with table output."""
        input_file = Path(temp_dir) / "input.h5ad"
        table_file = Path(temp_dir) / "results.csv"
        sample_adata.write_h5ad(input_file)

        cmd = [
            sys.executable,
            "-m",
            "kompot.cli",
            "impute",
            str(input_file),
            "-t",
            str(table_file),
            "--obsm-key",
            "X_pca",
            "--n-landmarks",
            "20",
            "--no-progress",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert table_file.exists()

        df = pd.read_csv(table_file, index_col=0)
        assert "mean_imputed" in df.columns
        assert "mean_std" in df.columns
        assert len(df) == sample_adata.n_vars

    def test_impute_with_config(self, sample_adata, temp_dir):
        """Test impute with YAML config file."""
        input_file = Path(temp_dir) / "input.h5ad"
        output_file = Path(temp_dir) / "output.h5ad"
        config_file = Path(temp_dir) / "config.yaml"
        sample_adata.write_h5ad(input_file)

        config = {
            "obsm_key": "X_pca",
            "n_landmarks": 20,
            "sigma": 0.5,
            "progress": False,
        }
        with open(config_file, "w") as f:
            yaml.dump(config, f)

        cmd = [
            sys.executable,
            "-m",
            "kompot.cli",
            "impute",
            str(input_file),
            "-o",
            str(output_file),
            "-c",
            str(config_file),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert output_file.exists()

    def test_impute_no_output_fails(self, sample_adata, temp_dir):
        """Test that missing output args fails."""
        input_file = Path(temp_dir) / "input.h5ad"
        sample_adata.write_h5ad(input_file)

        cmd = [
            sys.executable,
            "-m",
            "kompot.cli",
            "impute",
            str(input_file),
            "--obsm-key",
            "X_pca",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode != 0

    def test_impute_with_empirical_variance(self, sample_adata, temp_dir):
        """Test impute with --use-empirical-variance flag."""
        input_file = Path(temp_dir) / "input.h5ad"
        output_file = Path(temp_dir) / "output.h5ad"
        sample_adata.write_h5ad(input_file)

        cmd = [
            sys.executable,
            "-m",
            "kompot.cli",
            "impute",
            str(input_file),
            "-o",
            str(output_file),
            "--obsm-key",
            "X_pca",
            "--n-landmarks",
            "20",
            "--use-empirical-variance",
            "--no-progress",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        import anndata as ad

        adata_out = ad.read_h5ad(output_file)
        assert "kompot_impute_all_obs_variance" in adata_out.layers

    def test_impute_gene_subset(self, sample_adata, temp_dir):
        """Test impute with gene subsetting."""
        input_file = Path(temp_dir) / "input.h5ad"
        output_file = Path(temp_dir) / "output.h5ad"
        sample_adata.write_h5ad(input_file)

        cmd = [
            sys.executable,
            "-m",
            "kompot.cli",
            "impute",
            str(input_file),
            "-o",
            str(output_file),
            "--obsm-key",
            "X_pca",
            "--n-landmarks",
            "20",
            "--genes",
            "Gene_0",
            "Gene_1",
            "--no-progress",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        import anndata as ad

        adata_out = ad.read_h5ad(output_file)
        imputed = adata_out.layers["kompot_impute_all_imputed"]
        # Gene_0 and Gene_1 should have values, others NaN
        assert not np.all(np.isnan(imputed[:, 0]))
        assert np.all(np.isnan(imputed[:, 2]))
