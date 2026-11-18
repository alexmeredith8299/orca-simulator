"""Three-vector helpers that are faster than numpy's general ones on the small arrays the
simulator uses (a single vector, or K rows of three). Definitional only; no model values."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

F64 = NDArray[np.float64]


def cross3(a: F64, b: F64) -> F64:
    """Cross product of 3-vectors over the last axis, with broadcasting. It matches
    `numpy.cross` for these inputs, without its axis handling, which costs about ten times more
    on a single vector."""
    a0, a1, a2 = a[..., 0], a[..., 1], a[..., 2]
    b0, b1, b2 = b[..., 0], b[..., 1], b[..., 2]
    out: F64 = np.stack([a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0], axis=-1)
    return out
