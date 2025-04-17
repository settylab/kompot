import palantir
import contextlib
import warnings
import sys
import os

def init_worker(X):
    warnings.filterwarnings(
        "ignore",
        #category=UserWarning,
        message="Some of the cells were unreachable.",
    )
    warnings.filterwarnings(
        "ignore",
        #category=RuntimeWarning,
        message="os.fork() was called.",
    )
    global X_ms
    X_ms = X
def compute_pseudotime_for_start(start_cell):
    @contextlib.contextmanager
    def suppress_stdout():
        """Temporarily redirect sys.stdout to dev/null."""
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        try:
            yield
        finally:
            sys.stdout.close()
            sys.stdout = original_stdout
    with suppress_stdout():
        pr_res = palantir.core.run_palantir(X_ms, start_cell, n_jobs=1, use_early_cell_as_start=True)
    return pr_res.pseudotime.values
