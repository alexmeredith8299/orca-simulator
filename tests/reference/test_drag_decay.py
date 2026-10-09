"""M1: drag decays the semi-major axis at the analytic circular-orbit rate."""

import numpy as np
import pytest

from expand3.constellation import propagate_constellation
from expand3.params import Params, ParamView
from expand3.timebase import SeedDraws
from expand3.units import M_PER_KM, S_PER_DAY
from expand3.vehicle.orbit import Forces, rv_to_elements

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("sat", ["SAT-1", "SAT-5"])
def test_semi_major_axis_decay(params: Params, draws: SeedDraws, sat: str) -> None:
    run = propagate_constellation(
        params, draws, forces=Forces(j2=False, drag=True), duration_s=S_PER_DAY, sats=[sat]
    )
    mu = params.scenario.constants.mu_km3_s2
    el = rv_to_elements(run.r_km[:, 0], run.v_km_s[:, 0], mu)
    rate = np.polyfit(run.t_s, el.a_km, 1)[0]  # km/s

    view = ParamView(params, sat)
    a = params.scenario.orbit.a_km
    h = a - params.scenario.constants.R_E_km
    rho = draws.rho0_kg_m3 * np.exp(-(h - params.scenario.drag.h0_km) / draws.H_km)
    b = view.get("drag.Cd") * view.get("drag.area_m2") / view.get("mass.wet_mass_kg")
    # rho [kg/m^3] -> kg/km^3 (x 1e9); B [m^2/kg] -> km^2/kg (x 1e-6)
    analytic = -np.sqrt(mu * a) * rho * M_PER_KM**3 * b / M_PER_KM**2
    assert rate < 0
    assert rate == pytest.approx(analytic, rel=0.10)
