"""M1: with J2 and drag off, specific energy and angular momentum are conserved over 72 h."""

import numpy as np
import pytest

from expand3.constellation import ConstellationRun, propagate_constellation
from expand3.params import Params
from expand3.timebase import SeedDraws
from expand3.units import S_PER_HOUR
from expand3.vehicle.orbit import Forces

MU = 398600.4418


@pytest.mark.slow
def test_energy_and_momentum_conserved(params: Params, draws: SeedDraws) -> None:
    run = propagate_constellation(
        params, draws, forces=Forces(j2=False, drag=False), duration_s=72 * S_PER_HOUR
    )
    r, v = run.r_km, run.v_km_s
    energy = 0.5 * np.sum(v**2, axis=-1) - MU / np.linalg.norm(r, axis=-1)
    h = np.cross(r, v)
    rel_e = np.abs(energy / energy[0] - 1)
    rel_h = np.linalg.norm(h - h[0], axis=-1) / np.linalg.norm(h[0], axis=-1)
    assert rel_e.max() < 1e-9
    assert rel_h.max() < 1e-9


@pytest.mark.slow
def test_states_finite_and_shadow_bounded(run72: ConstellationRun) -> None:
    assert np.all(np.isfinite(run72.r_km))
    assert np.all(np.isfinite(run72.v_km_s))
    assert np.all((run72.shadow >= 0) & (run72.shadow <= 1))
    assert run72.r_km.shape == (len(run72.t_s), 6, 3)
