from kompot.batch_utils import apply_batched
import jax.numpy as jnp
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Create test data
X = np.random.rand(100, 10)

# Define a test function
def func(x):
    print(f'Processing shape {x.shape}')
    return np.sum(x, axis=1)

# Test with batch_size=None
print("Testing with batch_size=None")
result = apply_batched(func, X, batch_size=None)
print(f"Result shape: {result.shape}")

# Test with a small batch size
print("\nTesting with batch_size=20")
result = apply_batched(func, X, batch_size=20)
print(f"Result shape: {result.shape}")

# Test memory error fallback
print("\nTesting memory error fallback")

def memory_intensive_func(x):
    print(f'Processing shape {x.shape}')
    # Simulate memory error on full batch but success on smaller batches
    if len(x) > 50:
        raise RuntimeError("RESOURCE_EXHAUSTED: out of memory")
    return np.sum(x, axis=1)

try:
    result = apply_batched(memory_intensive_func, X, batch_size=None)
    print(f"Result shape: {result.shape}")
except Exception as e:
    print(f"Error: {e}")