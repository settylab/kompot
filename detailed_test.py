import kompot
import numpy as np
import pandas as pd
import anndata as ad

# Create a simple AnnData object for testing
def create_test_adata(n_cells=200, n_genes=30):
    np.random.seed(42)
    X = np.random.negative_binomial(5, 0.3, size=(n_cells, n_genes))
    
    obs = pd.DataFrame({
        'condition': np.array(['A'] * (n_cells//2) + ['B'] * (n_cells//2)),
    })
    
    var = pd.DataFrame(index=[f'gene_{i}' for i in range(n_genes)])
    
    adata = ad.AnnData(X=X, obs=obs, var=var)
    
    # Create a PCA representation
    pca = np.random.normal(size=(n_cells, 5))
    adata.obsm['X_pca'] = pca
    
    return adata

print("Creating test AnnData object...")
adata = create_test_adata()

print("Running differential abundance analysis...")
try:
    da_results = kompot.compute_differential_abundance(
        adata,
        groupby='condition',
        condition1='A',
        condition2='B',
        obsm_key='X_pca',
        n_landmarks=50
    )
    print("Differential abundance analysis successful\!")
except Exception as e:
    print(f"Error during differential abundance analysis: {str(e)}")

print("\nRunning differential expression analysis...")
try:
    # Try to see where the error occurs
    from kompot.differential import DifferentialExpression
    print("Imported DifferentialExpression class")
    
    # Print the parameters
    import inspect
    signature = inspect.signature(kompot.compute_differential_expression)
    print("Parameters for compute_differential_expression:")
    for param_name, param in signature.parameters.items():
        print(f"  - {param_name}: {param.default}")
    
    # Run with explicit jit_compile parameter
    de_results = kompot.compute_differential_expression(
        adata,
        groupby='condition',
        condition1='A',
        condition2='B',
        obsm_key='X_pca',
        n_landmarks=50,
        jit_compile=False
    )
    print("Differential expression analysis successful\!")
except Exception as e:
    print(f"Error during differential expression analysis: {str(e)}")
    import traceback
    traceback.print_exc()
