"""Which tick of a sender's data a receiver holds at each tick, for the multi-satellite detectors.

Spec: Scenario (clusters) and Model: crosslink. Inside a cluster, a member sends its state every
control tick whenever the link is up. Earth blocks line of sight between the clusters, so anything
between them goes through a ground station. The spec gives no latency, so the relay here is store
and forward: a cluster uploads during its own ground pass and the other cluster downloads during
its next pass. The passes of the two clusters do not overlap in the baseline scenario, so there is
no live relay.

Every function returns, per tick, the index of the sender tick the receiver holds (-1 before it
holds anything). Values read at that index are the held data and its age follows from the tick
difference.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

F64 = NDArray[np.float64]
IntArr = NDArray[np.int64]
BoolArr = NDArray[np.bool_]


def _last_true(flag: BoolArr) -> IntArr:
    """Index of the latest tick at or before each tick where `flag` holds, -1 if none."""
    ticks = np.arange(flag.size, dtype=np.int64)
    return np.asarray(np.maximum.accumulate(np.where(flag, ticks, -1)), dtype=np.int64)


def hold(available: BoolArr, dt_s: float) -> tuple[IntArr, F64]:
    """Zero-order hold of a signal that arrives on the ticks where `available` is true.

    Returns the held tick per tick and its age in seconds (inf before the first arrival).
    """
    idx = _last_true(np.asarray(available, dtype=bool))
    age = np.where(idx >= 0, (np.arange(idx.size) - idx) * dt_s, np.inf)
    return idx, np.asarray(age, dtype=np.float64)


def relay_source(source_pass: BoolArr, dest_pass: BoolArr) -> IntArr:
    """Source tick held by the destination cluster through a ground station.

    At a destination pass tick p the destination gets what the source had uploaded by p, which is
    the source's latest pass tick at or before p, and holds it until its next pass.
    """
    uploaded = _last_true(np.asarray(source_pass, dtype=bool))
    synced = _last_true(np.asarray(dest_pass, dtype=bool))
    return np.where(synced >= 0, uploaded[np.maximum(synced, 0)], -1).astype(np.int64)
