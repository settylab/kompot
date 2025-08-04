"""Test gc_utils module functionality."""

import pytest
import gc
import numpy as np
from kompot.gc_utils import (
    no_gc, explicit_cleanup, memory_efficient_loop,
    tune_gc_thresholds, restore_gc_thresholds,
    get_memory_stats, log_memory_stats, WeakContainer
)


class TestGCUtils:
    """Test garbage collection utilities."""

    def test_no_gc_context_manager(self):
        """Test no_gc context manager."""
        # Get initial state
        initial_enabled = gc.isenabled()
        
        # Test with generation 0 cleanup
        with no_gc(generation=0):
            # GC should be disabled inside context
            assert not gc.isenabled()
        
        # Should be restored after context
        assert gc.isenabled() == initial_enabled
    
    def test_no_gc_without_generation(self):
        """Test no_gc without specific generation."""
        initial_enabled = gc.isenabled()
        
        with no_gc():
            assert not gc.isenabled()
        
        assert gc.isenabled() == initial_enabled
    
    def test_explicit_cleanup(self):
        """Test explicit cleanup function."""
        # Create some containers
        large_list = [np.random.randn(100, 100) for _ in range(5)]
        large_dict = {i: np.random.randn(50, 50) for i in range(3)}
        
        # Test cleanup with list of containers
        explicit_cleanup([large_list, large_dict])
        
        # Containers should be cleared
        assert len(large_list) == 0
        assert len(large_dict) == 0
    
    def test_explicit_cleanup_single_container(self):
        """Test explicit cleanup with single container."""
        large_list = [1, 2, 3, 4, 5]
        explicit_cleanup(large_list)
        assert len(large_list) == 0
    
    def test_tune_gc_thresholds(self):
        """Test GC threshold tuning."""
        # Get original thresholds
        original = gc.get_threshold()
        
        # Tune thresholds
        old_thresholds = tune_gc_thresholds(gen0=2000, gen1=15, gen2=15)
        
        # Should return original thresholds
        assert old_thresholds == original
        
        # Should have new thresholds
        new_thresholds = gc.get_threshold()
        assert new_thresholds == (2000, 15, 15)
        
        # Restore original
        restore_gc_thresholds(original)
        assert gc.get_threshold() == original
    
    def test_memory_efficient_loop(self):
        """Test memory efficient loop context manager."""
        original_thresholds = gc.get_threshold()
        
        with memory_efficient_loop(tune_thresholds=True) as cleanup_fn:
            # Should have tuned thresholds
            current = gc.get_threshold()
            assert current != original_thresholds
            
            # Test cleanup function
            test_list = [1, 2, 3]
            cleanup_fn([test_list])
            assert len(test_list) == 0
        
        # Should restore original thresholds
        assert gc.get_threshold() == original_thresholds
    
    def test_get_memory_stats(self):
        """Test memory statistics function."""
        stats = get_memory_stats()
        
        # Should have expected keys
        required_keys = ['enabled', 'thresholds', 'counts']
        for key in required_keys:
            assert key in stats
        
        # Values should be reasonable
        assert isinstance(stats['enabled'], bool)
        assert isinstance(stats['thresholds'], tuple)
        assert len(stats['thresholds']) == 3
        assert isinstance(stats['counts'], tuple)
        assert len(stats['counts']) == 3
    
    def test_log_memory_stats(self):
        """Test memory statistics logging."""
        import logging
        
        # Test that the function runs without error
        # (The actual logging output is visible in test output)
        try:
            log_memory_stats(level=logging.INFO)
            log_memory_stats(level=logging.DEBUG)
            # If we get here, the function works
            assert True
        except Exception as e:
            pytest.fail(f"log_memory_stats failed: {e}")
    
    def test_weak_container(self):
        """Test WeakContainer functionality."""
        # Create an object
        test_obj = [1, 2, 3, 4, 5]
        
        # Create weak container
        weak_container = WeakContainer(test_obj)
        
        # Should be able to access object
        assert weak_container.get() == test_obj
        assert weak_container.is_alive()
        
        # Delete original reference
        del test_obj
        
        # May or may not be alive depending on GC timing
        # Just test that the methods work
        alive = weak_container.is_alive()
        assert isinstance(alive, bool)
        
        obj = weak_container.get()
        # obj might be None or the original object
    
    def test_weak_container_with_non_weakref_object(self):
        """Test WeakContainer with objects that can't have weak references."""
        # Integers can't have weak references
        weak_container = WeakContainer(42)
        
        # Should still work (falls back to strong reference)
        assert weak_container.get() == 42
        assert weak_container.is_alive()
    
    def test_memory_efficient_loop_without_tuning(self):
        """Test memory efficient loop without threshold tuning."""
        original_thresholds = gc.get_threshold()
        
        with memory_efficient_loop(tune_thresholds=False) as cleanup_fn:
            # Should not have changed thresholds
            assert gc.get_threshold() == original_thresholds
            
            # Cleanup function should still work
            test_dict = {'a': 1, 'b': 2}
            cleanup_fn([test_dict])
            assert len(test_dict) == 0
        
        # Thresholds should be unchanged
        assert gc.get_threshold() == original_thresholds


class TestGCUtilsIntegration:
    """Test integration with numpy arrays and real workloads."""
    
    def test_with_numpy_arrays(self):
        """Test GC utilities with numpy arrays."""
        with no_gc(generation=0):
            # Create large arrays
            arrays = [np.random.randn(100, 100) for _ in range(10)]
            
            # Do some computation
            results = [arr @ arr.T for arr in arrays]
            
            # Cleanup
            explicit_cleanup([arrays, results])
            
            assert len(arrays) == 0
            assert len(results) == 0
    
    def test_realistic_computation_loop(self):
        """Test with a realistic computation loop."""
        data = [np.random.randn(50, 50) for _ in range(20)]
        
        with memory_efficient_loop(generation_cleanup=0) as cleanup_fn:
            results = []
            temp_arrays = []
            
            for i, arr in enumerate(data):
                # Simulate computation
                result = np.linalg.eigvals(arr)
                results.append(result)
                
                # Track temporary arrays
                temp_arrays.append(arr)
                
                # Periodic cleanup
                if i % 5 == 0:
                    cleanup_fn(temp_arrays)
                    temp_arrays.clear()
            
            # Final cleanup
            cleanup_fn(temp_arrays)
        
        # Should have results for all input data
        assert len(results) == len(data)