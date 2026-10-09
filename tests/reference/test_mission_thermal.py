"""M5: thermal over the nominal 72 h mission, settling, limits, heaters, and the F7 shape."""

import numpy as np
import pytest

from expand3.mission.thread import MissionRun
from expand3.params import Params, ParamView
from expand3.vehicle.thermal import Thermal, run_offline, thermal_env

pytestmark = pytest.mark.slow
SETTLE_TOL_K = 0.5


def orbit_ticks(params: Params) -> int:
    mu = params.scenario.constants.mu_km3_s2
    period = 2 * np.pi * np.sqrt(params.scenario.orbit.a_km**3 / mu)
    return round(period / params.scenario.control_tick_s)


def settle_index(T: np.ndarray, n_orbit: int) -> int:
    """First tick after which every node repeats to within SETTLE_TOL_K one orbit later."""
    diff = np.abs(T[n_orbit:] - T[:-n_orbit]).max(axis=1)
    bad = np.flatnonzero(diff > SETTLE_TOL_K)
    return int(bad[-1] + 1) if bad.size else 0


def limits_for(params: Params, sat: str, node: str) -> tuple[float, float]:
    lim = params.bus(sat).thermal.limits_K
    return lim.get(node, lim["others"])


def test_thermal_recorded(mission72: MissionRun) -> None:
    for sat in mission72.sats:
        th = mission72.thermal[sat]
        assert th.T_K.shape == (len(mission72.t_s), len(th.names))
        assert np.all(np.isfinite(th.T_K))


def test_settles_and_stays_within_limits(params: Params, mission72: MissionRun) -> None:
    n_orbit = orbit_ticks(params)
    print()
    for sat in mission72.sats:
        th = mission72.thermal[sat]
        n0 = settle_index(th.T_K, n_orbit)
        assert mission72.t_s[n0] < 48 * 3600, f"{sat} did not settle"
        for j, node in enumerate(th.names):
            lo, hi = limits_for(params, sat, node)
            seg = th.T_K[n0:, j]
            assert seg.min() >= lo, (sat, node, seg.min())
            assert seg.max() <= hi, (sat, node, seg.max())
        eps = mission72.eps[sat]
        parts = [f"{sat}: settled after {mission72.t_s[n0] / 3600:.1f} h"]
        for node in ("battery", "payload", "bus_px", "wing_px"):
            seg = th.node(node)[n0:]
            parts.append(f"{node} {seg.min():.1f}-{seg.max():.1f} K")
        parts.append(f"heater duty batt {eps.battery_heater[n0:].mean():.2f} "
                     f"payload {eps.payload_heater[n0:].mean():.2f}")
        print(", ".join(parts))


def test_coupled_temperatures_reach_eps(params: Params, mission72: MissionRun) -> None:
    sat = "SAT-1"
    th, eps = mission72.thermal[sat], mission72.eps[sat]
    k = mission72.sats.index(sat)
    e = Thermal(ParamView(params, sat))
    assert th.names == e.names
    # string power at a sunlit tick follows the M4 coefficient at the computed wing temperature
    n = int(np.flatnonzero(mission72.shadow[:, k] == 1)[200])
    from expand3.vehicle.attitude import quat_to_dcm
    from expand3.vehicle.eps import Eps

    model = Eps(ParamView(params, sat))
    temps = {name: float(th.T_K[n, j]) for j, name in enumerate(th.names)}
    sun_body = quat_to_dcm(mission72.q[n, k]).T @ mission72.sun_unit[n]
    np.testing.assert_allclose(eps.p_string_W[n],
                               model.string_power_w(1.0, sun_body, temps), rtol=1e-12)


def test_f7_shape(params: Params, mission72: MissionRun) -> None:
    sat = "SAT-2"
    k = mission72.sats.index(sat)
    envs = [thermal_env(mission72.q[n, k], mission72.r_km[n, k], mission72.sun_unit[n],
                        float(mission72.shadow[n, k])) for n in range(len(mission72.t_s))]
    heat = mission72.eps[sat].heat_W
    heats = [{node: float(v[n]) for node, v in heat.items()} for n in range(len(mission72.t_s))]
    dt = params.scenario.control_tick_s
    nominal = run_offline(Thermal(ParamView(params, sat)), envs, heats, dt)
    view = ParamView(params, sat)
    view.overrides["thermal.nodes.bus_px.alpha"] = min(1.0, 3 * 0.08)
    view.overrides["thermal.nodes.bus_px.epsilon"] = min(1.0, 3 * 0.03)
    th = Thermal(view)
    faulted = run_offline(th, envs, heats, dt)
    last = slice(-orbit_ticks(params), None)

    def swing(T: np.ndarray, node: str) -> float:
        x = T[last, th.names.index(node)]
        return float(x.max() - x.min())

    def mean(T: np.ndarray, node: str) -> float:
        return float(T[last, th.names.index(node)].mean())

    d_px = swing(faulted, "bus_px") - swing(nominal, "bus_px")
    d_core = swing(faulted, "bus_core") - swing(nominal, "bus_core")
    print(f"\nF7 swing: bus_px {swing(nominal, 'bus_px'):.2f} -> {swing(faulted, 'bus_px'):.2f} K, "
          f"bus_core {swing(nominal, 'bus_core'):.2f} -> {swing(faulted, 'bus_core'):.2f} K")
    assert d_px > 0
    assert 0 <= d_core < d_px
    dm_px = abs(mean(faulted, "bus_px") - mean(nominal, "bus_px"))
    for far in ("bus_nx",):  # not linked to bus_px
        assert abs(swing(faulted, far) - swing(nominal, far)) < d_px
        assert abs(mean(faulted, far) - mean(nominal, far)) < dm_px
