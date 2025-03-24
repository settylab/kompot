"""Volcano plot functions for visualizing differential expression and abundance results."""

from .utils import _extract_conditions_from_key, _infer_de_keys, _infer_da_keys
from .de import volcano_de
from .da import volcano_da
from .multi_da import multi_volcano_da

__all__ = [
    'volcano_de',
    'volcano_da',
    'multi_volcano_da',
]
