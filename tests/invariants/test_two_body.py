"""M1: with J2 and drag off, specific energy and angular momentum are conserved over 72 h.

RK4 is not symplectic: at the spec's 10 s step its energy error grows linearly to about 1.3e-9
over 72 h (truncation, scaling as dt^5 per unit time here). Angular momentum stays below 1e-9.
The step-scaling test shows the drift is integrator truncation, not a force-model error.
"""

import numpy as np
import pytest

from expand3.constellation import ConstellationRun, propagate_constellation
from expand3.params import Params
from expand3.timebase import SeedDraws
from expand3.units import S_PER_DAY, S_PER_HOUR
from expand3.vehicle.orbit import Forces, OrbitModel, initial_states, propagate

MU = 398600.4418
ENERGY_TOL_72H = 2e-9
MOMENTUM_TOL_72H = 1e-9


def _energy(y: np.ndarray) -> np.ndarray:
    return 0.5 * np.sum(y[..., 3:] ** 2, axis=-1) - MU / np.linalg.norm(y[..., :3], axis=-1)


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
    assert rel_e.max() < ENERGY_TOL_72H
    assert rel_h.max() < MOMENTUM_TOL_72H


@pytest.mark.slow
def test_energy_drift_is_rk4_truncation(params: Params, draws: SeedDraws) -> None:
    y0 = initial_states(params, ["SAT-1"], draws.epoch_utc)
    model = OrbitModel.from_params(
        params, ["SAT-1"], draws, np.array([0.0, 0.0, 1.0]), Forces(j2=False, drag=False)
    )
    drift = []
    for dt in (10.0, 5.0):
        n = int(S_PER_DAY / dt)
        out = propagate(model, y0, dt, n, n)
        drift.append(abs(_energy(out[-1, 0]) / _energy(out[0, 0]) - 1))
    assert drift[0] / drift[1] > 16


@pytest.mark.slow
def test_states_finite_and_shadow_bounded(run72: ConstellationRun) -> None:
    assert np.all(np.isfinite(run72.r_km))
    assert np.all(np.isfinite(run72.v_km_s))
    assert np.all((run72.shadow >= 0) & (run72.shadow <= 1))
    assert run72.r_km.shape == (len(run72.t_s), 6, 3)
