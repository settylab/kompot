#!/usr/bin/env python3
"""
Example demonstrating memory optimization techniques in Kompot.

This script shows how to use the new garbage collection utilities
to optimize memory usage during intensive computations.
"""

import numpy as np
import logging
import kompot
from kompot import (
    DifferentialExpression, 
    no_gc, 
    explicit_cleanup, 
    memory_efficient_loop,
    tune_gc_thresholds,
    get_memory_stats,
    log_memory_stats
)

# Set up logging to see memory optimization messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demonstrate_gc_optimization():
    """Demonstrate how to use GC optimization for memory-intensive operations."""
    
    # Generate synthetic data
    np.random.seed(42)
    n_cells_1, n_cells_2 = 1000, 1200
    n_features = 50
    n_genes = 100
    
    # Create synthetic cell states and gene expression data
    X_condition1 = np.random.randn(n_cells_1, n_features)
    X_condition2 = np.random.randn(n_cells_2, n_features) + 0.5  # Slight shift
    
    # Add some structure to the data
    y_condition1 = np.random.randn(n_cells_1, n_genes) * 0.5
    y_condition2 = np.random.randn(n_cells_2, n_genes) * 0.5
    
    # Make some genes differentially expressed
    diff_genes = [0, 5, 10, 15, 20]  # Indices of differentially expressed genes
    for gene_idx in diff_genes:
        y_condition2[:, gene_idx] += 1.5  # Upregulate in condition 2
    
    logger.info("=" * 60)
    logger.info("MEMORY OPTIMIZATION DEMONSTRATION")
    logger.info("=" * 60)
    
    # Log initial memory stats
    logger.info("\n1. Initial Memory Statistics:")
    log_memory_stats(level=logging.INFO)
    
    # Example 1: Using no_gc context manager for intensive computation
    logger.info("\n2. Demonstrating no_gc context manager:")
    
    with no_gc(generation=0):
        logger.info("   Inside no_gc context - GC disabled for performance")
        
        # Simulate intensive computation that creates many temporary objects
        temp_arrays = []
        for i in range(50):
            temp_array = np.random.randn(100, 100)
            temp_arrays.append(temp_array @ temp_array.T)  # Matrix multiplication
        
        # Explicit cleanup of temporary arrays
        explicit_cleanup(temp_arrays)
        logger.info("   Performed explicit cleanup of temporary arrays")
    
    logger.info("   Exited no_gc context - GC re-enabled and cleanup performed")
    
    # Example 2: Tuning GC thresholds for better performance
    logger.info("\n3. Demonstrating GC threshold tuning:")
    
    # Store original thresholds
    original_thresholds = tune_gc_thresholds(gen0=3000, gen1=15, gen2=15)
    logger.info(f"   Tuned GC thresholds from {original_thresholds} to (3000, 15, 15)")
    
    # Example 3: Using memory_efficient_loop for large datasets
    logger.info("\n4. Demonstrating memory_efficient_loop:")
    
    large_dataset = [np.random.randn(200, 200) for _ in range(20)]
    
    with memory_efficient_loop(tune_thresholds=True, generation_cleanup=0) as cleanup_fn:
        results = []
        for i, data in enumerate(large_dataset):
            # Simulate processing
            result = np.linalg.eigvals(data)  # Compute eigenvalues
            results.append(result)
            
            # Periodic cleanup every 5 iterations
            if i % 5 == 0:
                cleanup_fn([data, result])
                logger.info(f"   Processed {i+1}/{len(large_dataset)} items with periodic cleanup")
    
    logger.info("   Completed memory_efficient_loop processing")
    
    # Example 4: Memory-optimized differential expression analysis
    logger.info("\n5. Running memory-optimized differential expression:")
    
    # The DifferentialExpression class now automatically uses GC optimizations
    diff_expr = DifferentialExpression(
        n_landmarks=100,  # Use landmarks to reduce memory usage
        batch_size=200,   # Process in smaller batches
        store_arrays_on_disk=False,  # Keep in memory for this demo
        jit_compile=True  # Enable JIT compilation for speed
    )
    
    # Fit the model - this now uses internal GC optimizations
    logger.info("   Fitting differential expression model...")
    diff_expr.fit(
        X_condition1=X_condition1,
        y_condition1=y_condition1,
        X_condition2=X_condition2,
        y_condition2=y_condition2,
        sigma=1.0
    )
    
    # Generate test points
    X_test = np.vstack([X_condition1[:50], X_condition2[:50]])
    
    # Predict with Mahalanobis distances - this uses optimized computation
    logger.info("   Computing predictions with Mahalanobis distances...")
    results = diff_expr.predict(
        X_test,
        compute_mahalanobis=True,
        progress=False  # Disable progress bar for cleaner output
    )
    
    logger.info(f"   Successfully computed results for {len(X_test)} test points")
    logger.info(f"   Found {np.sum(~np.isnan(results['mahalanobis_distances']))} valid Mahalanobis distances")
    
    # Restore original GC thresholds
    from kompot.gc_utils import restore_gc_thresholds
    restore_gc_thresholds(original_thresholds)
    logger.info(f"\n6. Restored GC thresholds to {original_thresholds}")
    
    # Final memory stats
    logger.info("\n7. Final Memory Statistics:")
    log_memory_stats(level=logging.INFO)
    
    logger.info("\n" + "=" * 60)
    logger.info("DEMONSTRATION COMPLETE")
    logger.info("=" * 60)
    
    return results

def demonstrate_memory_analysis():
    """Demonstrate memory requirement analysis features."""
    
    logger.info("\n" + "=" * 60)
    logger.info("MEMORY ANALYSIS DEMONSTRATION")
    logger.info("=" * 60)
    
    from kompot.memory_utils import (
        analyze_memory_requirements,
        analyze_covariance_memory_requirements,
        get_available_memory,
        array_size
    )
    
    # Analyze memory requirements for different array sizes
    logger.info("\n1. Analyzing memory requirements for various array shapes:")
    
    shapes_to_analyze = [
        (1000, 100),      # Medium dataset
        (5000, 200),      # Large dataset  
        (10000, 500),     # Very large dataset
    ]
    
    analysis = analyze_memory_requirements(
        shapes=shapes_to_analyze,
        analysis_name="Dataset Size Analysis",
        log_level="info"
    )
    
    logger.info(f"   Total memory required: {analysis['total_size']}")
    logger.info(f"   Available memory: {analysis['available_memory']}")
    logger.info(f"   Memory ratio: {analysis['memory_ratio']:.2f}")
    logger.info(f"   Status: {analysis['status']}")
    
    # Analyze covariance matrix memory requirements
    logger.info("\n2. Analyzing covariance matrix memory requirements:")
    
    cov_analysis = analyze_covariance_memory_requirements(
        n_points=2000,
        n_genes=150,
        analysis_name="Covariance Matrix Analysis",
        log_level="info"
    )
    
    logger.info(f"   Should use disk storage: {cov_analysis['should_use_disk']}")
    
    # Show individual array size calculations
    logger.info("\n3. Individual array size calculations:")
    
    # Large covariance matrix
    cov_shape = (2000, 2000, 100)
    size_str, size_bytes = array_size(cov_shape)
    logger.info(f"   Covariance tensor {cov_shape}: {size_str}")
    
    # Expression data
    expr_shape = (5000, 200)
    size_str, size_bytes = array_size(expr_shape)
    logger.info(f"   Expression matrix {expr_shape}: {size_str}")
    
    # Available memory
    avail_str, avail_bytes = get_available_memory()
    logger.info(f"   Available system memory: {avail_str}")

if __name__ == "__main__":
    # Run the demonstrations
    try:
        results = demonstrate_gc_optimization()
        demonstrate_memory_analysis()
        
        print("\n✅ All demonstrations completed successfully!")
        print("\nKey takeaways:")
        print("1. Use no_gc() context manager for memory-intensive computations")
        print("2. Apply explicit_cleanup() to large containers when done")
        print("3. Use memory_efficient_loop() for processing large datasets")
        print("4. Tune GC thresholds to reduce automatic collection frequency")
        print("5. The DifferentialExpression class now automatically optimizes memory usage")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        raise