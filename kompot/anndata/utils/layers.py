"""Helpers for enumerating AnnData layers across anndata versions."""

from typing import List


def layer_names(adata) -> List[str]:
    """Return the named layers of ``adata``.

    anndata >= 0.13 stores ``X`` as ``layers[None]`` and yields that ``None``
    key when iterating ``adata.layers``, so plain iteration mixes a non-layer
    sentinel in with the layer names. Filtering it out keeps layer enumeration
    identical on anndata < 0.13, where no such key exists.

    Parameters
    ----------
    adata : AnnData
        AnnData object to enumerate layers of.

    Returns
    -------
    list of str
        Layer names, excluding the ``None`` key that aliases ``X``.
    """
    return [key for key in adata.layers if key is not None]
