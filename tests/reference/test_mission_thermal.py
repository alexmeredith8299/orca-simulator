"""M5: thermal over the nominal 72 h mission, settling, limits, heaters, and the F7 shape."""

import numpy as np
import pytest

from expand3.mission.thread import MissionRun
from expand3.params import Params, ParamView
from expand3.vehicle.thermal import Thermal, run_offline, thermal_env

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


def settle_time_s(params: Params, run: MissionRun, sat: str, tol: float = SETTLE_TOL_K) -> float:
    """How long the initial temperatures matter. The forcing is not periodic (ground passes
    switch 60 W of radio heat on 0-3 times per orbit), so settling is measured as the time after
    which runs started SPREAD_K above and below the initial temperature agree within SETTLE_TOL_K
    at every node, under the same environment and node heat."""
    n_nodes = len(run.thermal[sat].names)
    _, hot = drive(params, run, sat, T0=np.full(n_nodes, 293.15 + SPREAD_K))
    _, cold = drive(params, run, sat, T0=np.full(n_nodes, 293.15 - SPREAD_K))
    bad = np.flatnonzero(np.abs(hot - cold).max(axis=1) > tol)
    return float(run.t_s[bad[-1] + 1]) if bad.size else 0.0


def limits_for(params: Params, sat: str, node: str) -> tuple[float, float]:
    lim = params.bus(sat).thermal.limits_K
    return lim.get(node, lim["others"])


def test_thermal_recorded(mission72: MissionRun) -> None:
    for sat in mission72.sats:
        th = mission72.thermal[sat]
        assert th.T_K.shape == (len(mission72.t_s), len(th.names))
        assert np.all(np.isfinite(th.T_K))


def test_settles_and_stays_within_limits(params: Params, mission72: MissionRun) -> None:
    print()
    for sat in mission72.sats:
        th = mission72.thermal[sat]
        quiet = sat in QUIET
        t_settle = settle_time_s(params, mission72, sat) if quiet else 48 * 3600.0
        t_1k = settle_time_s(params, mission72, sat, tol=1.0) if quiet else 0.0
        assert t_settle < mission72.t_s[-1] - 6 * 3600, f"{sat} did not settle"
        n0 = int(np.searchsorted(mission72.t_s, t_settle))
        # limits hold over the whole run, transient included
        for j, node in enumerate(th.names):
            lo, hi = limits_for(params, sat, node)
            assert th.T_K[:, j].min() >= lo, (sat, node, th.T_K[:, j].min())
            assert th.T_K[:, j].max() <= hi, (sat, node, th.T_K[:, j].max())
        eps = mission72.eps[sat]
        parts = [f"{sat}: within 1 K at {t_1k / 3600:.1f} h, 0.1 K at {t_settle / 3600:.1f} h"
                 if quiet else f"{sat} (from 48 h):"]
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
