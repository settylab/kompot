import kompot
print(f"Kompot version: {kompot.__version__}")

# Check if compute_differential_abundance accepts random_state
import inspect
signature = inspect.signature(kompot.compute_differential_abundance)
print(f"random_state in compute_differential_abundance params: {'random_state' in signature.parameters}")

# Check if jit_compile is defined in the module
is_jit_compile_defined = False
try:
    jit_compile = False
    print(f"jit_compile variable can be defined: {jit_compile}")
    is_jit_compile_defined = True
except NameError:
    print("jit_compile variable raises NameError when used")

print(f"jit_compile is defined in this scope: {is_jit_compile_defined}")
