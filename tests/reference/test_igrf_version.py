"""ppigrf must evaluate IGRF-14 (spec: SAA boundary uses IGRF-14), valid through 2030."""

import datetime as dt
from pathlib import Path

import numpy as np
import ppigrf
import ppigrf.ppigrf as ppigrf_impl

from expand3.params import Params


def test_default_coefficients_are_igrf14(params: Params) -> None:
    assert params.links.saa.field_model == "IGRF-14"
    shc = Path(ppigrf_impl.shc_fn)
    assert shc.name == "IGRF14.shc"
    assert shc.read_text().splitlines()[0].strip() == "# IGRF 14"
    g, _ = ppigrf_impl.read_shc(ppigrf_impl.shc_fn)
    assert g.index.min().year == 1900
    assert g.index.max().year == 2030  # covers the 2027 epoch draw


def test_evaluates_in_2027() -> None:
    be, bn, bu = ppigrf.igrf(0.0, 45.0, 525.0, dt.datetime(2027, 6, 1))
    b = float(np.sqrt(be**2 + bn**2 + bu**2)[0])
    assert 30_000 < b < 60_000  # nT, mid-latitude at LEO altitude
