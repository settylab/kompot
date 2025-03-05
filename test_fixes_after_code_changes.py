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
    # Now with random_state parameter
    da_results = kompot.compute_differential_abundance(
        adata,
        groupby='condition',
        condition1='A',
        condition2='B',
        obsm_key='X_pca',
        n_landmarks=50,
        random_state=42
    )
    print("Differential abundance analysis successful\!")
except Exception as e:
    print(f"Error during differential abundance analysis: {str(e)}")

print("\nRunning differential expression analysis...")
try:
    # Now with jit_compile and random_state
    de_results = kompot.compute_differential_expression(
        adata,
        groupby='condition',
        condition1='A',
        condition2='B',
        obsm_key='X_pca',
        n_landmarks=50,
        jit_compile=False, 
        random_state=42
    )
    print("Differential expression analysis successful\!")
except Exception as e:
    print(f"Error during differential expression analysis: {str(e)}")
    import traceback
    traceback.print_exc()

print("\nRunning run_differential_analysis function...")
try:
    # Test the combined function
    adata_result = kompot.run_differential_analysis(
        adata,
        groupby='condition',
        condition1='A',
        condition2='B',
        obsm_key='X_pca',
        n_landmarks=50,
        jit_compile=False,
        random_state=42,
        copy=True,
        generate_html_report=False
    )
    print("Run differential analysis successful\!")
except Exception as e:
    print(f"Error during run_differential_analysis: {str(e)}")
    import traceback
    traceback.print_exc()
