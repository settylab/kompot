"""Minimal test for differential expression with groups and sample_col."""

import numpy as np
import pandas as pd
import anndata as ad


def test_de_groups_with_sample_col():
    """Test compute_differential_expression with groups parameter and sample_col.

    This test validates that group-specific differential expression analysis
    works correctly with sample variance correction.
    """
    from kompot.anndata import compute_differential_expression

    # Set random seed for reproducibility
    np.random.seed(42)

    # Minimal dataset: 60 cells, 10 genes, 2 cell types, 2 conditions, 2 samples per condition
    n_cells = 60
    n_genes = 10
    n_features = 3

    # Create cell type labels: 30 cells of type A, 30 of type B
    cell_types = np.array(["TypeA"] * 30 + ["TypeB"] * 30)

    # Create condition labels: 15 cells per type per condition
    conditions = np.array(
        ["Cond1"] * 15 + ["Cond2"] * 15 + ["Cond1"] * 15 + ["Cond2"] * 15
    )

    # Create sample labels: 2 samples per condition
    # Cond1 has Sample1A and Sample1B, Cond2 has Sample2A and Sample2B
    samples = []
    for i, cond in enumerate(conditions):
        if cond == "Cond1":
            samples.append("Sample1A" if i % 2 == 0 else "Sample1B")
        else:
            samples.append("Sample2A" if i % 2 == 0 else "Sample2B")
    samples = np.array(samples)

    # Create low-dimensional embeddings
    X_pca = np.random.normal(0, 1, (n_cells, n_features))
    # Add slight separation between conditions
    X_pca[conditions == "Cond2", 0] += 0.5

    # Create expression matrix
    expression = np.random.normal(5, 1, (n_cells, n_genes))
    # Make gene 0 differentially expressed in TypeA
    expression[np.logical_and(cell_types == "TypeA", conditions == "Cond2"), 0] += 2.0
    # Make gene 1 differentially expressed in TypeB
    expression[np.logical_and(cell_types == "TypeB", conditions == "Cond2"), 1] += 2.0

    # Create AnnData object
    adata = ad.AnnData(X=expression)
    adata.obsm["X_pca"] = X_pca
    adata.obs["cell_type"] = pd.Categorical(cell_types)
    adata.obs["condition"] = pd.Categorical(conditions)
    adata.obs["sample"] = pd.Categorical(samples)

    # Create groups as a dictionary with boolean masks
    groups_dict = {"TypeA": cell_types == "TypeA", "TypeB": cell_types == "TypeB"}

    # Run differential expression with groups and sample_col
    compute_differential_expression(
        adata,
        groupby="condition",
        condition1="Cond1",
        condition2="Cond2",
        groups=groups_dict,  # Analyze both cell types
        sample_col="sample",  # Enable sample variance correction
        layer=None,  # Use .X
        obsm_key="X_pca",
        n_landmarks=5,  # Minimal for speed
        copy=False,
    )

    # Verify varm matrix exists for groups
    assert "kompot_de_Cond1_to_Cond2_mean_lfc_groups" in adata.varm.keys()

    # Get the group-specific mean LFC matrix (n_genes x n_groups)
    mean_lfc_groups = adata.varm["kompot_de_Cond1_to_Cond2_mean_lfc_groups"].values

    # Verify it has 2 columns (one per group)
    assert mean_lfc_groups.shape == (10, 2), (
        f"Expected shape (10, 2), got {mean_lfc_groups.shape}"
    )

    # Group order should be TypeA (index 0), TypeB (index 1) based on dictionary order
    # Verify that gene 0 has higher fold change in TypeA (column 0)
    assert mean_lfc_groups[0, 0] > 0.5, f"Gene 0 TypeA LFC = {mean_lfc_groups[0, 0]}"

    # Verify that gene 1 has higher fold change in TypeB (column 1)
    assert mean_lfc_groups[1, 1] > 0.5, f"Gene 1 TypeB LFC = {mean_lfc_groups[1, 1]}"

    print("✓ Test passed: groups with sample_col works correctly")
