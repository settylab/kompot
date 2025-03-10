"""
Example script for creating a stacked bar chart showing the direction of change by cell type.

This script demonstrates using the direction_barplot function from Kompot.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path if running as script
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from kompot.plot.heatmap import direction_barplot


if __name__ == "__main__":
    # Example usage (requires scanpy)
    import scanpy as sc
    import anndata as ad
    
    print("Loading example data...")
    adata = sc.datasets.pbmc3k_processed()
    
    # Add dummy differential abundance results
    print("Creating example differential abundance data...")
    np.random.seed(42)
    directions = np.random.choice(['up', 'down', 'neutral'], size=adata.n_obs, p=[0.3, 0.3, 0.4])
    adata.obs['kompot_da_log_fold_change_direction'] = directions
    
    # Add a dummy run in the run history
    adata.uns['kompot_da'] = {
        'run_history': [
            {
                'params': {
                    'conditions': ['Control', 'Treatment']
                },
                'field_names': {
                    'direction_key': 'kompot_da_log_fold_change_direction'
                }
            }
        ]
    }
    
    # Plot using the built-in direction_barplot function
    print("Creating plot with automatic parameter detection...")
    # The function will automatically infer direction_column and conditions from run_id
    fig, ax = direction_barplot(
        adata, 
        category_column='louvain',
        return_fig=True
    )
    
    # Save to file
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'direction_barplot_auto.png')
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    # Create another plot with explicit parameters
    print("Creating plot with explicit parameters...")
    fig, ax = direction_barplot(
        adata, 
        category_column='louvain',
        direction_column='kompot_da_log_fold_change_direction',
        condition1='Control',
        condition2='Treatment',
        title="Custom Title: Cell Type Direction Distribution",
        stacked=True,
        return_fig=True
    )
    
    # Save to file
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'direction_barplot_explicit.png')
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    # Show the plot
    plt.show()