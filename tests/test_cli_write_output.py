"""The CLI must be able to write its output under every supported anndata.

settylab/kompot#23: anndata 0.11 and 0.12 refuse to write a pandas
``StringArray`` unless ``anndata.settings.allow_write_nullable_strings`` is
set, and pandas 3 produces one for every ordinary string column, so every
``kompot de/da/smooth/dm`` write failed under those defaults. An explicit
``dtype="string"`` column produces the same array under pandas 2 as well, so the
test does not depend on pandas 3 to build its input. (It has been run on anndata
0.12 only with pandas 3, and on anndata 0.10 with pandas 2, where it skips.)
"""

import anndata
import numpy as np
import pandas as pd
import pytest

from kompot.cli.utils import write_output


def _adata_with_nullable_strings():
    adata = anndata.AnnData(X=np.ones((4, 3), dtype=np.float32))
    adata.obs["label"] = pd.array(["a", "b", None, "d"], dtype="string")
    adata.var["symbol"] = pd.array(["G1", "G2", "G3"], dtype="string")
    return adata


def _requires_the_setting():
    """anndata < 0.11 has no opt-in and cannot write a StringArray at all.

    Under pandas 2 those versions produce object columns, never a StringArray,
    so #23 does not arise there; the opt-in is a no-op on them by design.
    """
    settings = getattr(anndata, "settings", None)
    if settings is None or not hasattr(settings, "allow_write_nullable_strings"):
        pytest.skip("anndata has no allow_write_nullable_strings setting (< 0.11)")


@pytest.mark.parametrize("suffix", [".h5ad", ".zarr"])
def test_write_output_writes_nullable_string_columns(tmp_path, suffix):
    _requires_the_setting()
    if suffix == ".zarr":
        pytest.importorskip("zarr")
    adata = _adata_with_nullable_strings()
    out = tmp_path / f"out{suffix}"

    write_output(adata, out)

    back = anndata.read_h5ad(out) if suffix == ".h5ad" else anndata.read_zarr(out)
    assert list(back.obs["label"].astype(object).where(back.obs["label"].notna(), None)) == [
        "a", "b", None, "d"
    ]
    assert list(back.var["symbol"].astype(str)) == ["G1", "G2", "G3"]


def test_write_output_does_not_leak_the_setting(tmp_path):
    """The opt-in is scoped to the write, not left switched on for the caller."""
    _requires_the_setting()
    settings = anndata.settings
    before = settings.allow_write_nullable_strings
    write_output(_adata_with_nullable_strings(), tmp_path / "out.h5ad")
    assert settings.allow_write_nullable_strings == before


def test_write_output_rejects_unknown_suffix(tmp_path):
    with pytest.raises(SystemExit):
        write_output(_adata_with_nullable_strings(), tmp_path / "out.csv")
