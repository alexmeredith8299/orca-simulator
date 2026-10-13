"""M5: thermal over the nominal 72 h mission, settling, limits, heaters, and the F7 shape."""

import numpy as np
import pytest

from expand3.mission.thread import MissionRun
from expand3.params import Params, ParamView
from expand3.vehicle.eps import Eps
from expand3.vehicle.thermal import (
    Thermal,
    orbit_heat,
    run_offline,
    settled_initial_temps,
    thermal_env,
)

pytestmark = pytest.mark.slow
SETTLE_TOL_K = 0.1
SPREAD_K = 20.0
QUIET = ("SAT-1", "SAT-5", "SAT-6")


def orbit_ticks(params: Params) -> int:
    mu = params.scenario.constants.mu_km3_s2
    period = 2 * np.pi * np.sqrt(params.scenario.orbit.a_km**3 / mu)
    return round(period / params.scenario.control_tick_s)


def drive(params: Params, run: MissionRun, sat: str, view: ParamView | None = None,
          T0: np.ndarray | None = None) -> tuple[Thermal, np.ndarray]:
    """Re-run one vehicle's network offline with the mission's environment and node heat."""
    k = run.sats.index(sat)
    envs = [thermal_env(run.q[n, k], run.r_km[n, k], run.sun_unit[n], float(run.shadow[n, k]))
            for n in range(len(run.t_s))]
    heat = run.eps[sat].heat_W
    heats = [{node: float(v[n]) for node, v in heat.items()} for n in range(len(run.t_s))]
    th = Thermal(view or ParamView(params, sat))
    return th, run_offline(th, envs, heats, params.scenario.control_tick_s, T0)


def limits_for(params: Params, sat: str, node: str) -> tuple[float, float]:
    lim = params.bus(sat).thermal.limits_K
    return lim.get(node, lim["others"])


def test_thermal_recorded(mission72: MissionRun) -> None:
    for sat in mission72.sats:
        th = mission72.thermal[sat]
        assert th.T_K.shape == (len(mission72.t_s), len(th.names))
        assert np.all(np.isfinite(th.T_K))


REFERENCE_ORBITS = 30  # plain spin-up, ~48 h, much longer than the battery's 5.6 h time constant


def settled_start(params: Params, run: MissionRun, sat: str, orbits: int,
                  shoot: bool = True) -> np.ndarray:
    """The settled initializer as run_mission applies it, with a chosen spin-up length."""
    view = ParamView(params, sat)
    k = run.sats.index(sat)
    n_orbit = orbit_ticks(params)
    envs = [thermal_env(run.q[n, k], run.r_km[n, k], run.sun_unit[n], float(run.shadow[n, k]))
            for n in range(n_orbit)]
    th = Thermal(view)
    heats = orbit_heat(Eps(view), th, envs, params.scenario.control_tick_s)
    return settled_initial_temps(th, envs, heats, params.scenario.control_tick_s, orbits,
                                 shoot=shoot).T0


def test_settled_start(params: Params, mission72: MissionRun) -> None:
    """Under settled start the battery is on the converged trajectory almost from t = 0. The
    converged trajectory is the same network started from an independent REFERENCE_ORBITS plain
    spin-up (no shooting)."""
    assert params.scenario.thermal_init == "settled"
    print()
    for sat in QUIET:
        th = mission72.thermal[sat]
        ib = th.names.index("battery")
        start = settled_start(params, mission72, sat, params.scenario.thermal_init_orbits)
        np.testing.assert_allclose(th.T_K[0], start, rtol=1e-12)
        _, run = drive(params, mission72, sat, T0=start)
        _, ref = drive(params, mission72, sat,
                       T0=settled_start(params, mission72, sat, REFERENCE_ORBITS, shoot=False))
        dev = np.abs(run[:, ib] - ref[:, ib])
        after6 = mission72.t_s >= 6 * 3600
        assert dev[after6].max() < 0.5
        bad = np.flatnonzero(dev > SETTLE_TOL_K)
        t01 = float(mission72.t_s[bad[-1] + 1]) if bad.size else 0.0
        n_nodes = len(th.names)
        _, hot = drive(params, mission72, sat, T0=np.full(n_nodes, 293.15 + SPREAD_K))
        _, cold = drive(params, mission72, sat, T0=np.full(n_nodes, 293.15 - SPREAD_K))
        n6 = int(np.searchsorted(mission72.t_s, 6 * 3600))
        print(f"{sat}: battery vs converged |dT| at 0 h {dev[0]:.3f} K, max after 6 h "
              f"{dev[after6].max():.3f} K, within 0.1 K from {t01 / 3600:.1f} h; old +-20 K "
              f"bracket at 6 h: hot {hot[n6, ib] - run[n6, ib]:+.2f} K, "
              f"cold {cold[n6, ib] - run[n6, ib]:+.2f} K")


def test_stays_within_limits(params: Params, mission72: MissionRun) -> None:
    print()
    for sat in mission72.sats:
        th = mission72.thermal[sat]
        for j, node in enumerate(th.names):
            lo, hi = limits_for(params, sat, node)
            assert th.T_K[:, j].min() >= lo, (sat, node, th.T_K[:, j].min())
            assert th.T_K[:, j].max() <= hi, (sat, node, th.T_K[:, j].max())
        eps = mission72.eps[sat]
        parts = [f"{sat} (full run):"]
        for node in ("battery", "payload", "bus_px", "wing_px"):
            seg = th.node(node)
            parts.append(f"{node} {seg.min():.1f}-{seg.max():.1f} K")
        parts.append(f"heater duty batt {eps.battery_heater.mean():.2f} "
                     f"payload {eps.payload_heater.mean():.2f}")
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
    _, nominal = drive(params, mission72, sat)
    view = ParamView(params, sat)
    view.overrides["thermal.nodes.bus_px.alpha"] = min(1.0, 3 * 0.08)
    view.overrides["thermal.nodes.bus_px.epsilon"] = min(1.0, 3 * 0.03)
    th, faulted = drive(params, mission72, sat, view)
    last = slice(-orbit_ticks(params), None)

    def swing(T: np.ndarray, node: str) -> float:
        x = T[last, th.names.index(node)]
        return float(x.max() - x.min())

    def mean(T: np.ndarray, node: str) -> float:
        return float(T[last, th.names.index(node)].mean())

    # In idle Sun hold bus_px is edge-on to the Sun, so F7 adds only albedo absorption and extra
    # IR loss. bus_px is tied to bus_core at 20 W/K, so the vehicle cools as a whole and bus_px
    # cools most. Orbital swings change by well under 0.1 K. See CLAUDE.md Known gaps.
    drop = {n: mean(nominal, n) - mean(faulted, n) for n in th.names}
    print(f"\nF7 mean drop: bus_px {drop['bus_px']:.2f} K, bus_core {drop['bus_core']:.2f} K, "
          f"bus_nx {drop['bus_nx']:.2f} K; bus_px swing {swing(nominal, 'bus_px'):.2f} -> "
          f"{swing(faulted, 'bus_px'):.2f} K")
    assert drop["bus_px"] > 1.0
    assert all(drop["bus_px"] > d for n, d in drop.items() if n != "bus_px")
    assert drop["bus_nx"] < drop["bus_px"]  # bus_nx is not linked to bus_px
    assert swing(faulted, "bus_px") >= swing(nominal, "bus_px") - 1e-3
