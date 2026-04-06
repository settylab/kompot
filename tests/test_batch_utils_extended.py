"""Tests for batch_utils.py to improve coverage."""

import numpy as np
from kompot.batch_utils import is_jax_memory_error, merge_batch_results


def test_is_jax_memory_error_positive():
    """Test is_jax_memory_error with actual JAX memory errors."""
    # Test with RESOURCE_EXHAUSTED message
    error1 = Exception("RESOURCE_EXHAUSTED: Out of memory")
    assert is_jax_memory_error(error1) is True

    # Test with resource exhausted (lowercase)
    error2 = Exception("resource exhausted in GPU allocation")
    assert is_jax_memory_error(error2) is True

    # Test with out of memory
    error3 = Exception("XLA: Out of memory while allocating")
    assert is_jax_memory_error(error3) is True

    # Test with generic memory error
    error4 = Exception("Insufficient memory available")
    assert is_jax_memory_error(error4) is True


def test_is_jax_memory_error_negative():
    """Test is_jax_memory_error with non-memory errors."""
    # Test with unrelated error
    error1 = Exception("Invalid argument")
    assert is_jax_memory_error(error1) is False

    # Test with ValueError
    error2 = ValueError("Shape mismatch")
    assert is_jax_memory_error(error2) is False

    # Test with KeyError
    error3 = KeyError("missing_key")
    assert is_jax_memory_error(error3) is False


def test_merge_batch_results_empty():
    """Test merge_batch_results with empty list."""
    result = merge_batch_results([])
    assert result == {}


def test_merge_batch_results_dicts():
    """Test merge_batch_results with dictionaries containing arrays."""
    results = [
        {"a": np.array([1, 2]), "b": np.array([10])},
        {"a": np.array([3, 4]), "b": np.array([20])},
        {"a": np.array([5, 6]), "b": np.array([30])},
    ]

    merged = merge_batch_results(results)

    assert isinstance(merged, dict)
    assert "a" in merged
    assert "b" in merged
    np.testing.assert_array_equal(merged["a"], np.array([1, 2, 3, 4, 5, 6]))
    np.testing.assert_array_equal(merged["b"], np.array([10, 20, 30]))


def test_merge_batch_results_dicts_with_missing_keys():
    """Test merge_batch_results with dicts that have different keys."""
    results = [
        {"a": np.array([1, 2])},
        {"a": np.array([3, 4]), "b": np.array([10])},
        {"a": np.array([5, 6]), "b": np.array([20])},
    ]

    merged = merge_batch_results(results)

    assert "a" in merged
    assert "b" in merged
    np.testing.assert_array_equal(merged["a"], np.array([1, 2, 3, 4, 5, 6]))
    np.testing.assert_array_equal(merged["b"], np.array([10, 20]))


def test_merge_batch_results_arrays():
    """Test merge_batch_results with plain arrays."""
    results = [
        np.array([[1, 2], [3, 4]]),
        np.array([[5, 6], [7, 8]]),
        np.array([[9, 10], [11, 12]]),
    ]

    merged = merge_batch_results(results, concat_axis=0)

    expected = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    np.testing.assert_array_equal(merged, expected)


def test_merge_batch_results_lists():
    """Test merge_batch_results with lists."""
    results = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]

    merged = merge_batch_results(results)

    assert merged == [1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_merge_batch_results_scalar_dict_values():
    """Test merge_batch_results with dictionaries containing scalar values."""
    results = [
        {"count": 5, "sum": 10.5},
        {"count": 3, "sum": 7.2},
        {"count": 8, "sum": 15.3},
    ]

    merged = merge_batch_results(results)

    assert isinstance(merged, dict)
    assert "count" in merged
    assert "sum" in merged
    # Scalars should be collected into lists
    assert merged["count"] == [5, 3, 8]
    assert merged["sum"] == [10.5, 7.2, 15.3]


def test_merge_batch_results_mixed_types():
    """Test merge_batch_results with mixed array and list values in dict."""
    results = [
        {"array": np.array([1, 2]), "list": [10, 20]},
        {"array": np.array([3, 4]), "list": [30, 40]},
    ]

    merged = merge_batch_results(results)

    assert isinstance(merged, dict)
    np.testing.assert_array_equal(merged["array"], np.array([1, 2, 3, 4]))
    assert merged["list"] == [10, 20, 30, 40]


def test_merge_batch_results_concat_axis():
    """Test merge_batch_results with different concatenation axes."""
    results = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]

    # Concatenate along axis 0 (rows)
    merged0 = merge_batch_results(results, concat_axis=0)
    expected0 = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    np.testing.assert_array_equal(merged0, expected0)

    # Concatenate along axis 1 (columns)
    merged1 = merge_batch_results(results, concat_axis=1)
    expected1 = np.array([[1, 2, 5, 6], [3, 4, 7, 8]])
    np.testing.assert_array_equal(merged1, expected1)


def test_merge_batch_results_single_element():
    """Test merge_batch_results with single element list."""
    results = [{"a": np.array([1, 2, 3])}]

    merged = merge_batch_results(results)

    assert isinstance(merged, dict)
    np.testing.assert_array_equal(merged["a"], np.array([1, 2, 3]))


def test_merge_batch_results_nested_dicts():
    """Test merge_batch_results doesn't handle nested dicts (returns as list)."""
    results = [{"nested": {"a": 1}}, {"nested": {"a": 2}}]

    merged = merge_batch_results(results)

    # Nested dicts should be collected as list
    assert "nested" in merged
    assert merged["nested"] == [{"a": 1}, {"a": 2}]


def test_merge_batch_results_2d_concat_axis():
    """Test merge_batch_results with 2D arrays and different axes."""
    results = [np.array([[1, 2, 3]]), np.array([[4, 5, 6]])]

    merged = merge_batch_results(results, concat_axis=0)
    expected = np.array([[1, 2, 3], [4, 5, 6]])
    np.testing.assert_array_equal(merged, expected)
