"""Tests for CLI functionality."""
import pytest
import tempfile
import shutil
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
from anndata import AnnData
import subprocess
import sys


@pytest.fixture
def sample_adata():
    """Create a sample AnnData object for testing."""
    np.random.seed(42)
    n_obs = 100
    n_vars = 50

    # Create count matrix
    X = np.random.negative_binomial(5, 0.3, size=(n_obs, n_vars))

    # Create obs with conditions
    obs = pd.DataFrame({
        'condition': ['A'] * 50 + ['B'] * 50,
        'batch': np.random.choice(['batch1', 'batch2'], n_obs)
    })

    # Create var
    var = pd.DataFrame({
        'gene_name': [f'Gene_{i}' for i in range(n_vars)]
    })

    # Create obsm with embedding
    obsm = {
        'X_pca': np.random.randn(n_obs, 10)
    }

    adata = AnnData(X=X, obs=obs, var=var, obsm=obsm)
    return adata


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir)


class TestCLIUtils:
    """Test CLI utility functions."""

    def test_load_yaml_config(self, temp_dir):
        """Test loading YAML config."""
        from kompot.cli.utils import load_config

        config = {
            'groupby': 'condition',
            'condition1': 'A',
            'condition2': 'B',
            'n_landmarks': 100
        }

        config_file = Path(temp_dir) / 'test_config.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(config, f)

        loaded = load_config(str(config_file))
        assert loaded == config

    def test_load_json_config(self, temp_dir):
        """Test loading JSON config."""
        from kompot.cli.utils import load_config
        import json

        config = {
            'groupby': 'condition',
            'condition1': 'A',
            'condition2': 'B'
        }

        config_file = Path(temp_dir) / 'test_config.json'
        with open(config_file, 'w') as f:
            json.dump(config, f)

        loaded = load_config(str(config_file))
        assert loaded == config

    def test_load_config_file_not_found(self):
        """Test error when config file doesn't exist."""
        from kompot.cli.utils import load_config

        with pytest.raises(FileNotFoundError):
            load_config('nonexistent.yaml')

    def test_merge_args_with_config(self):
        """Test merging CLI args with config."""
        from kompot.cli.utils import merge_args_with_config

        args_dict = {
            'groupby': 'condition',
            'condition1': 'A',
            'batch_size': None
        }

        config = {
            'condition1': 'X',  # Should be overridden by CLI arg
            'condition2': 'B',  # Should be kept from config
            'batch_size': 100   # Should be kept since CLI arg is None
        }

        merged = merge_args_with_config(args_dict, config)

        assert merged['groupby'] == 'condition'
        assert merged['condition1'] == 'A'  # CLI takes precedence
        assert merged['condition2'] == 'B'
        assert merged['batch_size'] == 100  # From config since CLI is None

    def test_validate_anndata_path(self, sample_adata, temp_dir):
        """Test AnnData path validation."""
        from kompot.cli.utils import validate_anndata_path

        # Test with existing file
        adata_file = Path(temp_dir) / 'test.h5ad'
        sample_adata.write_h5ad(adata_file)

        validated = validate_anndata_path(str(adata_file))
        assert validated.exists()

        # Test with non-existent file
        with pytest.raises(FileNotFoundError):
            validate_anndata_path(str(Path(temp_dir) / 'nonexistent.h5ad'))


class TestCLIDifferentialExpression:
    """Test differential expression CLI command."""

    def test_de_basic_cli_args(self, sample_adata, temp_dir):
        """Test DE with basic CLI arguments."""
        input_file = Path(temp_dir) / 'input.h5ad'
        output_file = Path(temp_dir) / 'output.h5ad'
        sample_adata.write_h5ad(input_file)

        # Run CLI
        cmd = [
            sys.executable, '-m', 'kompot.cli',
            'de',
            str(input_file),
            '-o', str(output_file),
            '--groupby', 'condition',
            '--condition1', 'A',
            '--condition2', 'B',
            '--obsm-key', 'X_pca',
            '--n-landmarks', '50',
            '--batch-size', '50',
            '--null-genes', '10'
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Check command succeeded
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert output_file.exists()

        # Load and verify output
        import anndata as ad
        adata_out = ad.read_h5ad(output_file)
        assert 'kompot_de' in adata_out.uns
        assert 'kompot_de_A_to_B_mahalanobis' in adata_out.var.columns

    def test_de_with_config_file(self, sample_adata, temp_dir):
        """Test DE with config file."""
        input_file = Path(temp_dir) / 'input.h5ad'
        output_file = Path(temp_dir) / 'output.h5ad'
        config_file = Path(temp_dir) / 'config.yaml'

        sample_adata.write_h5ad(input_file)

        # Create config
        config = {
            'groupby': 'condition',
            'condition1': 'A',
            'condition2': 'B',
            'obsm_key': 'X_pca',
            'n_landmarks': 50,
            'batch_size': 50,
            'null_genes': 10,
            'progress': False
        }

        with open(config_file, 'w') as f:
            yaml.dump(config, f)

        # Run CLI
        cmd = [
            sys.executable, '-m', 'kompot.cli',
            'de',
            str(input_file),
            '-o', str(output_file),
            '-c', str(config_file)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert output_file.exists()

    def test_de_cli_overrides_config(self, sample_adata, temp_dir):
        """Test that CLI args override config file."""
        input_file = Path(temp_dir) / 'input.h5ad'
        output_file = Path(temp_dir) / 'output.h5ad'
        config_file = Path(temp_dir) / 'config.yaml'

        sample_adata.write_h5ad(input_file)

        # Create config with different condition1
        config = {
            'groupby': 'condition',
            'condition1': 'X',  # Wrong value
            'condition2': 'B',
            'obsm_key': 'X_pca',
            'n_landmarks': 50,
            'null_genes': 10
        }

        with open(config_file, 'w') as f:
            yaml.dump(config, f)

        # Run CLI with correct condition1
        cmd = [
            sys.executable, '-m', 'kompot.cli',
            'de',
            str(input_file),
            '-o', str(output_file),
            '-c', str(config_file),
            '--condition1', 'A',  # Override config
            '--batch-size', '50'
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Should succeed because CLI overrides config
        assert result.returncode == 0, f"CLI failed: {result.stderr}"

    def test_de_missing_required_params(self, sample_adata, temp_dir):
        """Test error when required parameters are missing."""
        input_file = Path(temp_dir) / 'input.h5ad'
        output_file = Path(temp_dir) / 'output.h5ad'
        sample_adata.write_h5ad(input_file)

        # Run CLI without required params
        cmd = [
            sys.executable, '-m', 'kompot.cli',
            'de',
            str(input_file),
            '-o', str(output_file)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Should fail
        assert result.returncode != 0
        assert 'Missing required parameters' in result.stderr or 'Missing required parameters' in result.stdout

    def test_de_table_output_csv(self, sample_adata, temp_dir):
        """Test DE with table output to CSV."""
        input_file = Path(temp_dir) / 'input.h5ad'
        table_file = Path(temp_dir) / 'results.csv'
        sample_adata.write_h5ad(input_file)

        # Run CLI with table output only (no AnnData output)
        cmd = [
            sys.executable, '-m', 'kompot.cli',
            'de',
            str(input_file),
            '-t', str(table_file),
            '--groupby', 'condition',
            '--condition1', 'A',
            '--condition2', 'B',
            '--obsm-key', 'X_pca',
            '--n-landmarks', '50',
            '--null-genes', '10',
            '--batch-size', '50'
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert table_file.exists()

        # Verify table content
        import pandas as pd
        df = pd.read_csv(table_file, index_col=0)
        assert len(df) > 0
        # Check that kompot columns are present
        assert any('kompot' in col or 'lfc' in col.lower() for col in df.columns)

    def test_de_table_output_tsv(self, sample_adata, temp_dir):
        """Test DE with table output to TSV."""
        input_file = Path(temp_dir) / 'input.h5ad'
        table_file = Path(temp_dir) / 'results.tsv'
        sample_adata.write_h5ad(input_file)

        cmd = [
            sys.executable, '-m', 'kompot.cli',
            'de',
            str(input_file),
            '-t', str(table_file),
            '--groupby', 'condition',
            '--condition1', 'A',
            '--condition2', 'B',
            '--obsm-key', 'X_pca',
            '--n-landmarks', '50',
            '--null-genes', '10',
            '--batch-size', '50'
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert table_file.exists()

        # Verify TSV format
        import pandas as pd
        df = pd.read_csv(table_file, index_col=0, sep='\t')
        assert len(df) > 0

    def test_de_both_outputs(self, sample_adata, temp_dir):
        """Test DE with both AnnData and table outputs."""
        input_file = Path(temp_dir) / 'input.h5ad'
        output_file = Path(temp_dir) / 'output.h5ad'
        table_file = Path(temp_dir) / 'results.csv'
        sample_adata.write_h5ad(input_file)

        cmd = [
            sys.executable, '-m', 'kompot.cli',
            'de',
            str(input_file),
            '-o', str(output_file),
            '-t', str(table_file),
            '--groupby', 'condition',
            '--condition1', 'A',
            '--condition2', 'B',
            '--obsm-key', 'X_pca',
            '--n-landmarks', '50',
            '--null-genes', '10',
            '--batch-size', '50'
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert output_file.exists()
        assert table_file.exists()

    def test_de_no_output_error(self, sample_adata, temp_dir):
        """Test error when neither output nor table-output is specified."""
        input_file = Path(temp_dir) / 'input.h5ad'
        sample_adata.write_h5ad(input_file)

        cmd = [
            sys.executable, '-m', 'kompot.cli',
            'de',
            str(input_file),
            '--groupby', 'condition',
            '--condition1', 'A',
            '--condition2', 'B',
            '--obsm-key', 'X_pca'
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        assert result.returncode != 0
        # Error message goes to stdout via logger
        combined_output = (result.stdout + result.stderr).lower()
        assert 'output' in combined_output or 'table-output' in combined_output


class TestCLIDifferentialAbundance:
    """Test differential abundance CLI command."""

    def test_da_basic_cli_args(self, sample_adata, temp_dir):
        """Test DA with basic CLI arguments."""
        input_file = Path(temp_dir) / 'input.h5ad'
        output_file = Path(temp_dir) / 'output.h5ad'
        sample_adata.write_h5ad(input_file)

        # Run CLI
        cmd = [
            sys.executable, '-m', 'kompot.cli',
            'da',
            str(input_file),
            '-o', str(output_file),
            '--groupby', 'condition',
            '--condition1', 'A',
            '--condition2', 'B',
            '--obsm-key', 'X_pca'
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Check command succeeded
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert output_file.exists()

        # Load and verify output
        import anndata as ad
        adata_out = ad.read_h5ad(output_file)
        assert 'kompot_da' in adata_out.uns
        assert 'kompot_da_A_to_B_lfc' in adata_out.obs.columns

    def test_da_with_config_file(self, sample_adata, temp_dir):
        """Test DA with config file."""
        input_file = Path(temp_dir) / 'input.h5ad'
        output_file = Path(temp_dir) / 'output.h5ad'
        config_file = Path(temp_dir) / 'config.yaml'

        sample_adata.write_h5ad(input_file)

        # Create config
        config = {
            'groupby': 'condition',
            'condition1': 'A',
            'condition2': 'B',
            'obsm_key': 'X_pca',
            'n_landmarks': 50
        }

        with open(config_file, 'w') as f:
            yaml.dump(config, f)

        # Run CLI
        cmd = [
            sys.executable, '-m', 'kompot.cli',
            'da',
            str(input_file),
            '-o', str(output_file),
            '-c', str(config_file)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert output_file.exists()

    def test_da_with_thresholds(self, sample_adata, temp_dir):
        """Test DA with custom thresholds."""
        input_file = Path(temp_dir) / 'input.h5ad'
        output_file = Path(temp_dir) / 'output.h5ad'
        sample_adata.write_h5ad(input_file)

        # Run CLI with custom thresholds
        cmd = [
            sys.executable, '-m', 'kompot.cli',
            'da',
            str(input_file),
            '-o', str(output_file),
            '--groupby', 'condition',
            '--condition1', 'A',
            '--condition2', 'B',
            '--obsm-key', 'X_pca',
            '--log-fold-change-threshold', '0.5',
            '--ptp-threshold', '0.01'
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

    def test_da_table_output_csv(self, sample_adata, temp_dir):
        """Test DA with table output to CSV."""
        input_file = Path(temp_dir) / 'input.h5ad'
        table_file = Path(temp_dir) / 'results.csv'
        sample_adata.write_h5ad(input_file)

        # Run CLI with table output only (no AnnData output)
        cmd = [
            sys.executable, '-m', 'kompot.cli',
            'da',
            str(input_file),
            '-t', str(table_file),
            '--groupby', 'condition',
            '--condition1', 'A',
            '--condition2', 'B',
            '--obsm-key', 'X_pca'
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert table_file.exists()

        # Verify table content
        import pandas as pd
        df = pd.read_csv(table_file, index_col=0)
        assert len(df) > 0
        # Check that kompot columns are present
        assert any('kompot' in col or 'lfc' in col.lower() for col in df.columns)

    def test_da_table_output_tsv(self, sample_adata, temp_dir):
        """Test DA with table output to TSV."""
        input_file = Path(temp_dir) / 'input.h5ad'
        table_file = Path(temp_dir) / 'results.tsv'
        sample_adata.write_h5ad(input_file)

        cmd = [
            sys.executable, '-m', 'kompot.cli',
            'da',
            str(input_file),
            '-t', str(table_file),
            '--groupby', 'condition',
            '--condition1', 'A',
            '--condition2', 'B',
            '--obsm-key', 'X_pca'
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert table_file.exists()

        # Verify TSV format
        import pandas as pd
        df = pd.read_csv(table_file, index_col=0, sep='\t')
        assert len(df) > 0

    def test_da_no_output_error(self, sample_adata, temp_dir):
        """Test error when neither output nor table-output is specified."""
        input_file = Path(temp_dir) / 'input.h5ad'
        sample_adata.write_h5ad(input_file)

        cmd = [
            sys.executable, '-m', 'kompot.cli',
            'da',
            str(input_file),
            '--groupby', 'condition',
            '--condition1', 'A',
            '--condition2', 'B',
            '--obsm-key', 'X_pca'
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        assert result.returncode != 0
        # Error message goes to stdout via logger
        combined_output = (result.stdout + result.stderr).lower()
        assert 'output' in combined_output or 'table-output' in combined_output


class TestCLIDiffusionMaps:
    """Test diffusion maps CLI command."""

    def test_dm_basic_cli_args(self, sample_adata, temp_dir):
        """Test DM with basic CLI arguments."""
        pytest.importorskip("palantir")

        # Add PCA to sample data
        import scanpy as sc
        sc.pp.pca(sample_adata, n_comps=20)

        input_file = Path(temp_dir) / 'input.h5ad'
        output_file = Path(temp_dir) / 'output.h5ad'
        sample_adata.write_h5ad(input_file)

        # Run CLI
        cmd = [
            sys.executable, '-m', 'kompot.cli',
            'dm',
            str(input_file),
            '-o', str(output_file),
            '--pca-key', 'X_pca',
            '--n-components', '10',
            '--knn', '30'
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Check command succeeded
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert output_file.exists()

        # Load and verify output
        import anndata as ad
        adata_out = ad.read_h5ad(output_file)
        assert 'DM_EigenVectors' in adata_out.obsm
        assert adata_out.obsm['DM_EigenVectors'].shape[1] == 10

    def test_dm_with_config_file(self, sample_adata, temp_dir):
        """Test DM with config file."""
        pytest.importorskip("palantir")

        # Add PCA to sample data
        import scanpy as sc
        sc.pp.pca(sample_adata, n_comps=20)

        input_file = Path(temp_dir) / 'input.h5ad'
        output_file = Path(temp_dir) / 'output.h5ad'
        config_file = Path(temp_dir) / 'config.yaml'

        sample_adata.write_h5ad(input_file)

        # Create config
        config = {
            'pca_key': 'X_pca',
            'n_components': 10,
            'knn': 30
        }

        with open(config_file, 'w') as f:
            yaml.dump(config, f)

        # Run CLI
        cmd = [
            sys.executable, '-m', 'kompot.cli',
            'dm',
            str(input_file),
            '-o', str(output_file),
            '-c', str(config_file)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert output_file.exists()

    def test_dm_missing_pca(self, sample_adata, temp_dir):
        """Test error when PCA is missing."""
        input_file = Path(temp_dir) / 'input.h5ad'
        output_file = Path(temp_dir) / 'output.h5ad'

        # Remove PCA from sample data
        del sample_adata.obsm['X_pca']
        sample_adata.write_h5ad(input_file)

        # Run CLI without PCA in data
        cmd = [
            sys.executable, '-m', 'kompot.cli',
            'dm',
            str(input_file),
            '-o', str(output_file)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Should fail
        assert result.returncode != 0
        assert 'PCA coordinates not found' in result.stderr or 'PCA coordinates not found' in result.stdout


class TestCLIMainEntry:
    """Test main CLI entry point."""

    def test_help_message(self):
        """Test that help message displays correctly."""
        cmd = [sys.executable, '-m', 'kompot.cli', '--help']
        result = subprocess.run(cmd, capture_output=True, text=True)

        assert result.returncode == 0
        assert 'kompot' in result.stdout.lower()
        assert 'differential' in result.stdout.lower()

    def test_version_message(self):
        """Test version flag."""
        cmd = [sys.executable, '-m', 'kompot.cli', '--version']
        result = subprocess.run(cmd, capture_output=True, text=True)

        assert result.returncode == 0
        assert 'kompot' in result.stdout.lower()

    def test_de_help(self):
        """Test DE command help."""
        cmd = [sys.executable, '-m', 'kompot.cli', 'de', '--help']
        result = subprocess.run(cmd, capture_output=True, text=True)

        assert result.returncode == 0
        assert 'differential expression' in result.stdout.lower()
        assert '--groupby' in result.stdout
        assert '--condition1' in result.stdout

    def test_da_help(self):
        """Test DA command help."""
        cmd = [sys.executable, '-m', 'kompot.cli', 'da', '--help']
        result = subprocess.run(cmd, capture_output=True, text=True)

        assert result.returncode == 0
        assert 'differential abundance' in result.stdout.lower()
        assert '--groupby' in result.stdout
        assert '--condition1' in result.stdout

    def test_dm_help(self):
        """Test DM command help."""
        cmd = [sys.executable, '-m', 'kompot.cli', 'dm', '--help']
        result = subprocess.run(cmd, capture_output=True, text=True)

        assert result.returncode == 0
        assert 'diffusion' in result.stdout.lower()
        assert '--pca-key' in result.stdout
        assert '--n-components' in result.stdout

    def test_no_command_shows_help(self):
        """Test that running without command shows help."""
        cmd = [sys.executable, '-m', 'kompot.cli']
        result = subprocess.run(cmd, capture_output=True, text=True)

        assert result.returncode == 0
        assert 'kompot' in result.stdout.lower() or 'usage' in result.stdout.lower()
