"""Shared pytest fixtures and configuration for kompot tests."""

import os
import pytest
import numpy as np
import pandas as pd
import anndata as ad

# Configure JAX to use CPU only for tests (must be set before JAX import)
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "True"


# ===== Pytest Configuration =====


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests (deselect with '-m \"not integration\"')",
    )
    config.addinivalue_line("markers", "memory: marks tests that measure memory usage")


# ===== Shared Test Data Fixtures =====


@pytest.fixture
def tiny_adata():
    """
    Create a tiny AnnData object for fast unit tests.

    - 20 cells (10 per condition)
    - 5 genes
    - 5 features (for cell states)
    - 2 samples per condition

    Runtime: ~0.01s per test using this fixture
    """
    np.random.seed(42)

    n_cells = 20
    n_genes = 5
    n_features = 5

    # Gene expression
    X = np.random.randn(n_cells, n_genes)

    # Cell states (for obsm)
    cell_states = np.random.randn(n_cells, n_features)

    # Metadata
    conditions = ["A"] * 10 + ["B"] * 10
    samples = ["sample1"] * 5 + ["sample2"] * 5 + ["sample3"] * 5 + ["sample4"] * 5

    adata = ad.AnnData(X)
    adata.obsm["X_pca"] = cell_states
    adata.obsm["DM_EigenVectors"] = cell_states.copy()
    adata.obs["condition"] = pd.Categorical(conditions)
    adata.obs["sample"] = pd.Categorical(samples)
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]

    return adata


@pytest.fixture
def small_adata():
    """
    Create a small AnnData object for standard unit tests.

    - 50 cells (25 per condition)
    - 10 genes
    - 10 features
    - 2 samples per condition

    Runtime: ~0.05s per test using this fixture
    """
    np.random.seed(42)

    n_cells = 50
    n_genes = 10
    n_features = 10

    X = np.random.randn(n_cells, n_genes)
    cell_states = np.random.randn(n_cells, n_features)

    conditions = ["A"] * 25 + ["B"] * 25
    samples = ["sample1"] * 12 + ["sample2"] * 13 + ["sample3"] * 12 + ["sample4"] * 13

    adata = ad.AnnData(X)
    adata.obsm["X_pca"] = cell_states
    adata.obsm["DM_EigenVectors"] = cell_states.copy()
    adata.obs["condition"] = pd.Categorical(conditions)
    adata.obs["sample"] = pd.Categorical(samples)
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]

    return adata


@pytest.fixture
def medium_adata():
    """
    Create a medium AnnData object for integration tests.

    - 100 cells (50 per condition)
    - 20 genes
    - 10 features
    - 2-4 samples per condition

    Runtime: ~0.2-0.5s per test using this fixture
    """
    np.random.seed(42)

    n_cells = 100
    n_genes = 20
    n_features = 10

    X = np.random.randn(n_cells, n_genes)
    cell_states = np.random.randn(n_cells, n_features)

    conditions = ["A"] * 50 + ["B"] * 50
    samples = (
        ["sample1"] * 12
        + ["sample2"] * 13
        + ["sample3"] * 12
        + ["sample4"] * 13
        + ["sample5"] * 12
        + ["sample6"] * 13
        + ["sample7"] * 12
        + ["sample8"] * 13
    )

    adata = ad.AnnData(X)
    adata.obsm["X_pca"] = cell_states
    adata.obsm["DM_EigenVectors"] = cell_states.copy()
    adata.obs["condition"] = pd.Categorical(conditions)
    adata.obs["sample"] = pd.Categorical(samples)
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]

    return adata


@pytest.fixture
def adata_with_batch():
    """
    Create AnnData with batch information for batch effect tests.

    - 60 cells
    - 10 genes
    - 3 batches
    - 2 conditions
    """
    np.random.seed(42)

    n_cells = 60
    n_genes = 10
    n_features = 10

    X = np.random.randn(n_cells, n_genes)
    cell_states = np.random.randn(n_cells, n_features)

    conditions = ["A"] * 30 + ["B"] * 30
    batches = ["batch1"] * 20 + ["batch2"] * 20 + ["batch3"] * 20
    samples = (
        ["sample1"] * 10
        + ["sample2"] * 10
        + ["sample3"] * 10
        + ["sample4"] * 10
        + ["sample5"] * 10
        + ["sample6"] * 10
    )

    adata = ad.AnnData(X)
    adata.obsm["X_pca"] = cell_states
    adata.obsm["DM_EigenVectors"] = cell_states.copy()
    adata.obs["condition"] = pd.Categorical(conditions)
    adata.obs["batch"] = pd.Categorical(batches)
    adata.obs["sample"] = pd.Categorical(samples)
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]

    return adata


# ===== Fast Test Parameters =====


@pytest.fixture
def fast_de_params():
    """
    Parameters for fast differential expression tests.

    These parameters prioritize speed over accuracy:
    - Small n_landmarks
    - No FDR computation
    - No progress bars
    """
    return {
        "n_landmarks": 10,
        "null_genes": 0,  # Disable FDR for speed
        "progress": False,
        "batch_size": 0,  # No batching needed for small datasets
    }


@pytest.fixture
def fast_da_params():
    """
    Parameters for fast differential abundance tests.

    These parameters prioritize speed over accuracy:
    - Small n_landmarks
    - No progress bars
    """
    return {
        "n_landmarks": 10,
        "progress": False,
        "batch_size": 0,
    }


# ===== Integration Test Parameters =====


@pytest.fixture
def integration_de_params():
    """
    Parameters for integration differential expression tests.

    These parameters are more realistic:
    - Moderate n_landmarks
    - Enable FDR with reduced null genes
    """
    return {
        "n_landmarks": 50,
        "null_genes": 500,  # Reduced from 2000 for faster tests
        "progress": False,
        "batch_size": 0,
    }


@pytest.fixture
def integration_da_params():
    """
    Parameters for integration differential abundance tests.
    """
    return {
        "n_landmarks": 50,
        "progress": False,
        "batch_size": 0,
    }


# ===== Temporary Directory Fixtures =====


@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory for test files."""
    return tmp_path
