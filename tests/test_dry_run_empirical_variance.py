"""Tests validating dry run memory estimates for use_empirical_variance.

Covers:
1. Dry run reports additional memory when use_empirical_variance=True
2. Empirical validation: actual memory usage (subprocess maxrss) vs dry run estimates
"""

import numpy as np
import pandas as pd
import pytest
import subprocess
import sys
import json
import anndata as ad

from kompot.resource_estimation import (
    estimate_differential_expression_resources,
    dry_run_differential_expression,
)


@pytest.fixture
def medium_adata():
    """AnnData with enough cells/genes for meaningful memory measurements."""
    rng = np.random.RandomState(42)
    n_cells = 200
    n_genes = 50
    n_features = 5

    X = rng.randn(n_cells, n_genes).astype(np.float64)
    cell_states = rng.randn(n_cells, n_features).astype(np.float64)

    conditions = ["A"] * 100 + ["B"] * 100
    samples = (
        ["s1"] * 50 + ["s2"] * 50 +
        ["s3"] * 50 + ["s4"] * 50
    )

    adata = ad.AnnData(X)
    adata.obsm["X_pca"] = cell_states
    adata.obs["condition"] = pd.Categorical(conditions)
    adata.obs["sample"] = pd.Categorical(samples)
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    return adata


class TestDryRunEmpiricalVarianceEstimates:
    """Verify dry run reports more memory with use_empirical_variance."""

    def test_total_memory_increases(self, medium_adata):
        """Total estimated memory should be higher with empirical variance."""
        plan_off = estimate_differential_expression_resources(
            medium_adata,
            condition1="A",
            condition2="B",
            groupby="condition",
            use_empirical_variance=False,
            n_landmarks=30,
            null_genes=0,
            batch_size=0,
        )
        plan_on = estimate_differential_expression_resources(
            medium_adata,
            condition1="A",
            condition2="B",
            groupby="condition",
            use_empirical_variance=True,
            n_landmarks=30,
            null_genes=0,
            batch_size=0,
        )

        assert plan_on.total_memory_required > plan_off.total_memory_required, (
            f"Empirical variance should increase memory estimate: "
            f"on={plan_on.total_memory_required}, off={plan_off.total_memory_required}"
        )

    def test_precision_matrix_requirements_present(self, medium_adata):
        """Should have precision matrix requirements for variance GPs."""
        plan = estimate_differential_expression_resources(
            medium_adata,
            condition1="A",
            condition2="B",
            groupby="condition",
            use_empirical_variance=True,
            n_landmarks=30,
            null_genes=0,
            batch_size=0,
        )

        req_names = [r.name for r in plan.requirements]
        assert any("Empirical variance GP precision" in n for n in req_names), (
            f"Missing empirical variance GP precision matrix requirement. "
            f"Got: {req_names}"
        )

    def test_factor_trick_requirement_present(self, medium_adata):
        """Should have factor trick memory requirement."""
        plan = estimate_differential_expression_resources(
            medium_adata,
            condition1="A",
            condition2="B",
            groupby="condition",
            use_empirical_variance=True,
            n_landmarks=30,
            null_genes=0,
            batch_size=0,
        )

        req_names = [r.name for r in plan.requirements]
        assert any("factor trick" in n for n in req_names), (
            f"Missing factor trick memory requirement. Got: {req_names}"
        )

    def test_landmark_variance_requirement_present(self, medium_adata):
        """Should have empirical variance at landmarks requirement."""
        plan = estimate_differential_expression_resources(
            medium_adata,
            condition1="A",
            condition2="B",
            groupby="condition",
            use_empirical_variance=True,
            n_landmarks=30,
            null_genes=0,
            batch_size=0,
        )

        req_names = [r.name for r in plan.requirements]
        assert any("Empirical variance at landmarks" in n for n in req_names), (
            f"Missing empirical variance at landmarks requirement. Got: {req_names}"
        )

    def test_fitting_temp_requirement_present(self, medium_adata):
        """Should have temporary fitting arrays requirement."""
        plan = estimate_differential_expression_resources(
            medium_adata,
            condition1="A",
            condition2="B",
            groupby="condition",
            use_empirical_variance=True,
            n_landmarks=30,
            null_genes=0,
            batch_size=0,
        )

        req_names = [r.name for r in plan.requirements]
        assert any("Temporary arrays during empirical variance fitting" in n for n in req_names), (
            f"Missing fitting temp arrays requirement. Got: {req_names}"
        )

    def test_intermediate_array_count_increases(self, medium_adata):
        """Intermediate array count should be higher with empirical variance."""
        plan_off = estimate_differential_expression_resources(
            medium_adata,
            condition1="A",
            condition2="B",
            groupby="condition",
            use_empirical_variance=False,
            n_landmarks=30,
            null_genes=0,
            batch_size=0,
        )
        plan_on = estimate_differential_expression_resources(
            medium_adata,
            condition1="A",
            condition2="B",
            groupby="condition",
            use_empirical_variance=True,
            n_landmarks=30,
            null_genes=0,
            batch_size=0,
        )

        def get_intermediate_req(plan):
            for r in plan.requirements:
                if "Peak intermediate arrays" in r.name:
                    return r
            return None

        req_off = get_intermediate_req(plan_off)
        req_on = get_intermediate_req(plan_on)
        assert req_off is not None and req_on is not None
        assert req_on.size_bytes > req_off.size_bytes

    def test_no_empirical_variance_requirements_when_off(self, medium_adata):
        """No empirical variance requirements should appear when flag is off."""
        plan = estimate_differential_expression_resources(
            medium_adata,
            condition1="A",
            condition2="B",
            groupby="condition",
            use_empirical_variance=False,
            n_landmarks=30,
            null_genes=0,
            batch_size=0,
        )

        req_names = [r.name for r in plan.requirements]
        for name in req_names:
            assert "mpirical variance" not in name and "factor trick" not in name, (
                f"Unexpected empirical variance requirement when disabled: {name}"
            )

    def test_info_message_about_empirical_variance(self, medium_adata):
        """Should have an info message about empirical variance GPs."""
        plan = estimate_differential_expression_resources(
            medium_adata,
            condition1="A",
            condition2="B",
            groupby="condition",
            use_empirical_variance=True,
            n_landmarks=30,
            null_genes=0,
            batch_size=0,
        )

        info_text = " ".join(plan.info)
        assert "mpirical variance" in info_text, (
            f"Missing info about empirical variance. Info: {plan.info}"
        )

    def test_dry_run_wrapper_works(self, medium_adata):
        """dry_run_differential_expression should work with use_empirical_variance."""
        plan = dry_run_differential_expression(
            medium_adata,
            groupby="condition",
            condition1="A",
            condition2="B",
            verbose=False,
            use_empirical_variance=True,
            n_landmarks=30,
            null_genes=0,
            batch_size=0,
        )

        assert plan.total_memory_required > 0
        assert plan.is_feasible


# ===== Subprocess-based memory measurement =====
# JAX allocates memory outside Python's heap (XLA buffers), so tracemalloc
# misses most of it. We use subprocess maxrss (resource.getrusage) which
# captures the full resident set size including JAX/C allocations.

_MEASURE_SCRIPT = '''
import sys, json, resource, tempfile, os
import numpy as np
import pandas as pd
import anndata as ad

# Suppress JAX/mellon logging
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import logging
logging.disable(logging.WARNING)

use_ev = "--use-empirical-variance" in sys.argv

rng = np.random.RandomState(42)
n_cells, n_genes, n_features = 200, 50, 5
X = rng.randn(n_cells, n_genes).astype(np.float64)
cell_states = rng.randn(n_cells, n_features).astype(np.float64)
conditions = ["A"] * 100 + ["B"] * 100

adata = ad.AnnData(X)
adata.obsm["X_pca"] = cell_states
adata.obs["condition"] = pd.Categorical(conditions)
adata.var_names = [f"gene_{i}" for i in range(n_genes)]
adata.obs_names = [f"cell_{i}" for i in range(n_cells)]

from kompot.anndata import compute_differential_expression
compute_differential_expression(
    adata,
    groupby="condition",
    condition1="A",
    condition2="B",
    obsm_key="X_pca",
    n_landmarks=30,
    null_genes=0,
    batch_size=0,
    progress=False,
    use_empirical_variance=use_ev,
)

maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
# macOS reports bytes, Linux reports KB
if sys.platform == "linux":
    maxrss *= 1024
print(json.dumps({"maxrss": maxrss, "use_empirical_variance": use_ev}))
'''


def _run_measure(use_ev: bool) -> int:
    """Run the measurement script in a subprocess and return maxrss in bytes."""
    args = [sys.executable, "-c", _MEASURE_SCRIPT]
    if use_ev:
        args.append("--use-empirical-variance")
    result = subprocess.run(args, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(
            f"Measurement subprocess failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
    # Parse the last JSON line from stdout
    for line in reversed(result.stdout.strip().split("\n")):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)["maxrss"]
    raise RuntimeError(f"No JSON output found in: {result.stdout}")


class TestEmpiricalMemoryValidation:
    """Validate dry run estimates against actual memory usage via subprocess maxrss.

    Uses subprocess isolation so each run gets a fresh process with clean
    memory state, and resource.getrusage captures the full RSS including
    JAX/XLA allocations.
    """

    @pytest.mark.slow
    @pytest.mark.memory
    def test_ev_increases_actual_memory(self):
        """Actual peak RSS should be higher with empirical variance."""
        rss_off = _run_measure(use_ev=False)
        rss_on = _run_measure(use_ev=True)

        print(f"\n[maxrss] Without EV: {rss_off:,} bytes, With EV: {rss_on:,} bytes, "
              f"Increase: {rss_on - rss_off:,} bytes ({(rss_on - rss_off) / max(rss_off, 1) * 100:.1f}%)")

        assert rss_on > rss_off, (
            f"Peak RSS with EV ({rss_on:,}) should exceed without ({rss_off:,})"
        )

    @pytest.mark.slow
    @pytest.mark.memory
    def test_estimate_increase_matches_actual_direction(self, medium_adata):
        """Both estimated and actual memory should increase with empirical variance."""
        # Dry run estimates
        plan_off = dry_run_differential_expression(
            medium_adata,
            groupby="condition",
            condition1="A",
            condition2="B",
            verbose=False,
            use_empirical_variance=False,
            n_landmarks=30,
            null_genes=0,
            batch_size=0,
        )
        plan_on = dry_run_differential_expression(
            medium_adata,
            groupby="condition",
            condition1="A",
            condition2="B",
            verbose=False,
            use_empirical_variance=True,
            n_landmarks=30,
            null_genes=0,
            batch_size=0,
        )
        est_increase = plan_on.total_memory_required - plan_off.total_memory_required

        # Actual memory via subprocess
        rss_off = _run_measure(use_ev=False)
        rss_on = _run_measure(use_ev=True)
        actual_increase = rss_on - rss_off

        print(f"\n[increase] Estimated: {est_increase:,} bytes, Actual: {actual_increase:,} bytes")

        assert est_increase > 0, "Estimated increase should be positive"
        assert actual_increase > 0, "Actual increase should be positive"
