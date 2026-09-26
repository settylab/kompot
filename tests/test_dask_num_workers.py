"""`SampleVarianceEstimator(dask_num_workers=...)` must not claim a limit it does not apply.

settylab/kompot#31: the argument wrote the Dask config key ``pool.num-workers``,
which no scheduler reads (the threaded scheduler reads ``num_workers``), mutated
Dask's *global* config as a side effect, and logged "Configured Dask to use N
workers" regardless. The covariance tensor it governed is returned lazily and is
materialised one gene at a time, so there is no worker pool for the setting to
bound. It is now deprecated and ignored, and says so.
"""

import logging
import warnings

import numpy as np
import pytest

from kompot.differential.sample_variance_estimator import (
    DASK_AVAILABLE,
    SampleVarianceEstimator,
)


def _data():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 3))
    Y = rng.standard_normal((40, 2))
    groups = np.repeat([0, 1, 2], [14, 13, 13])
    return X, Y, groups


def test_dask_num_workers_warns_that_it_has_no_effect():
    with pytest.warns(FutureWarning, match="no effect"):
        SampleVarianceEstimator(dask_num_workers=2)


def test_default_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        SampleVarianceEstimator()


@pytest.mark.skipif(not DASK_AVAILABLE, reason="dask not installed")
def test_disk_backed_predict_leaves_dask_config_alone_and_claims_nothing(tmp_path, caplog):
    import dask

    X, Y, groups = _data()
    before = dict(dask.config.config)
    with pytest.warns(FutureWarning):
        sve = SampleVarianceEstimator(
            store_arrays_on_disk=True,
            disk_storage_dir=str(tmp_path),
            jit_compile=False,
            dask_num_workers=2,
        )
    sve.fit(X, Y, groups)
    with caplog.at_level(logging.INFO, logger="kompot"):
        cov = sve.predict(X[:8], diag=False)

    assert isinstance(cov, dask.array.Array)
    assert np.isfinite(np.asarray(cov[:, :, 0])).all()
    # No global side effect: neither the key the old code wrote nor any other.
    assert dask.config.get("pool.num-workers", None) is None
    assert dict(dask.config.config) == before
    # And no log line asserting a limit that was never applied.
    assert not any("workers" in r.getMessage() for r in caplog.records)
