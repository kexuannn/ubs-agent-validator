"""Analysis utilities used by the Memgraph builder."""

from .side_effects import SideEffect, infer_side_effects

__all__ = [
    "SideEffect",
    "infer_side_effects",
]
