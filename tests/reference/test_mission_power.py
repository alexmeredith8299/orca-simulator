"""M4: nominal six-vehicle power over the 72 h mission thread."""

import numpy as np
import pytest

from expand3.mission.thread import MissionRun
from expand3.params import Params

pytestmark = pytest.mark.slow


def test_nominal_power(params: Params, mission72: MissionRun) -> None:
    mu = params.scenario.constants.mu_km3_s2
    period = 2 * np.pi * np.sqrt(params.scenario.orbit.a_km**3 / mu)
    n_per_orbit = round(period / params.scenario.control_tick_s)
    print()
    for k, sat in enumerate(mission72.sats):
        h = mission72.eps[sat]
        assert len(h.t_s) == len(mission72.t_s)
        assert not h.uv_hold.any(), f"{sat} tripped the undervoltage reflex nominally"
        assert not h.shed.any()
        assert not h.disc_clamped.any()
        assert np.all((h.soc >= 0) & (h.soc <= 1))
        idle = ~(mission72.gate_burn[:, k] | mission72.gate_take[:, k] | mission72.gate_slew[:, k])
        n_orbits = len(h.t_s) // n_per_orbit
        for o in range(n_orbits):
            sl = slice(o * n_per_orbit, (o + 1) * n_per_orbit)
            if idle[sl].all():
                assert h.p_gen_W[sl].mean() > h.demand_W[sl].mean()
        print(f"{sat}: min SoC {h.soc.min():.3f}, min V {h.v_V.min():.2f} V, "
              f"orbit-avg P_gen {h.p_gen_W.mean():.0f} W, D {h.demand_W.mean():.0f} W, "
              f"unmet {h.unmet_W.max():.1f} W")
        assert h.soc.min() > 0.5
