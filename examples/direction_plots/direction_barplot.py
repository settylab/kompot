"""
Example script for creating a stacked bar chart showing the direction of change by cell type
using Kompot's centralized color definitions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path if running as script
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from kompot.utils import KOMPOT_COLORS


def plot_direction_by_cell_type(adata, cell_type_column, direction_column='kompot_da_log_fold_change_direction',
                                condition1='Control', condition2='Treatment', figsize=(12, 6)):
    """Create a stacked bar chart showing direction of change by cell type using Kompot colors.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object with differential abundance results
    cell_type_column : str
        Column in adata.obs containing cell type annotations
    direction_column : str
        Column in adata.obs containing direction information
    condition1 : str
        Name of the first condition for the title
    condition2 : str
        Name of the second condition for the title
    figsize : tuple
        Size of the figure
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    # Create crosstab (percentage of each direction by cell type)
    crosstab = (
        pd.crosstab(
            adata.obs[cell_type_column],
            adata.obs[direction_column],
            normalize="index",
        )
        * 100
    )

    # Get colors from Kompot's color palette
    direction_colors = KOMPOT_COLORS["direction"]
    
    # Order columns consistently
    ordered_columns = []
    if "up" in crosstab.columns:
        ordered_columns.append("up")
    if "down" in crosstab.columns:
        ordered_columns.append("down")
    if "neutral" in crosstab.columns:
        ordered_columns.append("neutral")
        
    # Filter columns to only those that exist in the data
    ordered_columns = [col for col in ordered_columns if col in crosstab.columns]
    crosstab = crosstab[ordered_columns]
    
    # Create color list matching the ordered columns
    colors = [direction_colors[col] for col in ordered_columns]
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    crosstab.plot(
        kind="bar",
        stacked=True,
        color=colors,
        ax=ax
    )
    
    # Add labels and styling
    ax.set_xlabel("Cell Type")
    ax.set_ylabel("Percentage (%)")
    ax.set_title(f"Direction of Change by Cell Type\n{condition2} vs {condition1}")
    plt.xticks(rotation=90)
    ax.legend(title="Direction")
    plt.tight_layout()
    
    return fig


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
    
    # Plot and save
    print("Creating plot...")
    fig = plot_direction_by_cell_type(adata, 'louvain', condition1='Control', condition2='Treatment')
    
    # Save to file
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'direction_barplot.png')
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    # Show the plot
    plt.show()