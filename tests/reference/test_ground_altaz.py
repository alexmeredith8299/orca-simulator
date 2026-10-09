"""M1: station elevations match astropy AltAz, and passes respect the 5 deg mask."""

import warnings
from datetime import timedelta

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import GCRS, AltAz, CartesianRepresentation, EarthLocation
from astropy.time import Time

from expand3.constellation import ConstellationRun
from expand3.params import Params
from expand3.timebase import SeedDraws

pytestmark = pytest.mark.slow


def test_elevation_matches_astropy(
    params: Params, draws: SeedDraws, run72: ConstellationRun
) -> None:
    rng = np.random.default_rng(0)
    for s_idx, gs in enumerate(params.scenario.ground_stations):
        loc = EarthLocation.from_geodetic(gs.lon_deg * u.deg, gs.lat_deg * u.deg, 0 * u.m)
        n_idx, k_idx = np.nonzero(run72.elevation_deg[:, :, s_idx] > -20)
        pick = rng.choice(len(n_idx), size=15, replace=False)
        for n, k in zip(n_idx[pick], k_idx[pick], strict=True):
            when = Time(draws.epoch_utc + timedelta(seconds=float(run72.t_s[n])), scale="utc")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                coord = GCRS(CartesianRepresentation(run72.r_km[n, k] * u.km), obstime=when)
                alt = coord.transform_to(AltAz(obstime=when, location=loc)).alt.deg
            assert run72.elevation_deg[n, k, s_idx] == pytest.approx(alt, abs=0.05)


def test_passes_respect_mask(params: Params, run72: ConstellationRun) -> None:
    mask = params.scenario.elevation_mask_deg
    assert run72.passes, "expected some passes in 72 h"
    n_passes = 0
    for (sat, station), intervals in run72.passes.items():
        k, s = run72.sats.index(sat), run72.station_names.index(station)
        el = run72.elevation_deg[:, k, s]
        for t0, t1 in intervals:
            n_passes += 1
            inside = (run72.t_s >= t0) & (run72.t_s <= t1)
            assert inside.any()
            assert np.all(el[inside] >= mask)
        np.testing.assert_array_equal(run72.visible[:, k, s], el >= mask)
    # Svalbard sees a polar orbit on most revolutions: about 45 revs in 72 h, six vehicles.
    svalbard = sum(len(v) for (_, st), v in run72.passes.items() if st == "Svalbard")
    assert svalbard > 6 * 25
    assert n_passes > svalbard
