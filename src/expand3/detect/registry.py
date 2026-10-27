"""All detectors by name."""

from __future__ import annotations

from expand3.detect.base import Detector
from expand3.detect.detectors import DETECTORS
from expand3.detect.learned import LEARNED
from expand3.detect.mcdrop import MCD
from expand3.detect.peer import PEER
from expand3.detect.unsup import UNSUP

REGISTRY: dict[str, type[Detector]] = {**DETECTORS, **LEARNED, **MCD, **PEER, **UNSUP}
FAMILIES = ("physics", "learned", "bayesian", "peer", "unsupervised", "classifier")


CLASSIFIER = "clf"
RAW_CLASSIFIER = "clf_raw"
CLASSIFIERS = (CLASSIFIER, RAW_CLASSIFIER)
ALL_FAULTS = {f"F{i}" for i in range(1, 11)}


def describe(name: str) -> tuple[str, bool, set[str]]:
    """(family, detection only, the faults the detector can address)."""
    if name in CLASSIFIERS:
        return "classifier", False, set(ALL_FAULTS)
    cls = REGISTRY[name]
    return cls.family, cls.detection_only, set(ALL_FAULTS) if cls.detection_only else set(
        cls.fault_ids)
