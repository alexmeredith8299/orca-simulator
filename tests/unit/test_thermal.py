"""M5: thermal network. Spec: Model: thermal network; Parameters > Thermal nodes, links, limits."""

import dataclasses

import numpy as np
import pytest
from scipy.optimize import fsolve

from expand3.params import Params, ParamView
from expand3.vehicle.eps import Eps, EpsInputs
from expand3.vehicle.thermal import (
    Network,
    Thermal,
    ThermalEnv,
    step_power_and_thermal,
    thermal_env,
)

SIGMA, S = 5.670374e-8, 1361.0
Z = np.array([0.0, 0, 1])
SUNLIT_ZENITH = ThermalEnv(shadow=1.0, sun_body=Z, nadir_body=-Z, r_km=6903.137,
                           sun_dot_nadir=-1.0)


def tiny(n: int, **kw: object) -> Network:
    base: dict[str, object] = dict(
        names=[f"n{i}" for i in range(n)], C=np.full(n, 1000.0), alpha=np.zeros(n),
        eps=np.zeros(n), area_src=np.zeros(n), area_rad=np.zeros(n), normal=np.zeros((n, 3)),
        view=np.zeros(n), G=np.zeros((n, n)), S=S, a_alb=0.3, q_IR=237.0, R_E_km=6378.137,
        sigma=SIGMA,
    )
    base.update(kw)
    return Network(**base)  # type: ignore[arg-type]


def test_single_node_radiative_equilibrium() -> None:
    net = tiny(1, alpha=np.array([0.9]), eps=np.array([0.8]), area_src=np.array([1.0]),
               area_rad=np.array([1.0]), normal=np.array([Z]))
    T = np.array([250.0])
    for _ in range(2000):
        T = net.step(T, np.zeros(1), SUNLIT_ZENITH, 30.0)
    expected = (0.9 * S * 1.0 / (0.8 * 1.0 * SIGMA)) ** 0.25
    assert T[0] == pytest.approx(expected, abs=0.1)


def test_two_node_steady_state_matches_scipy() -> None:
    G = np.array([[0.0, 2.0], [2.0, 0.0]])
    net = tiny(2, C=np.array([2000.0, 500.0]), alpha=np.array([0.5, 0.0]),
               eps=np.array([0.8, 0.0]), area_src=np.array([1.0, 0.0]),
               area_rad=np.array([1.0, 0.0]), normal=np.array([Z, np.zeros(3)]), G=G)
    Q = np.array([0.0, 20.0])
    T = np.array([280.0, 280.0])
    for _ in range(5000):
        T = net.step(T, Q, SUNLIT_ZENITH, 30.0)
    ref = fsolve(lambda x: net.rate(x, Q, SUNLIT_ZENITH), np.array([300.0, 300.0]), xtol=1e-12)
    np.testing.assert_allclose(T, ref, atol=0.01)


@pytest.mark.parametrize("sat", ["SAT-1", "SAT-5"])
def test_conduction_conserves_energy(params: Params, sat: str) -> None:
    th = Thermal(ParamView(params, sat))
    net = th.network()
    net = dataclasses.replace(net, alpha=np.zeros_like(net.alpha), eps=np.zeros_like(net.eps))
    rng = np.random.default_rng(5)
    T = 280.0 + 20 * rng.random(len(net.names))
    for _ in range(100):
        Q = 30 * rng.random(len(net.names))
        T1 = net.step(T, Q, SUNLIT_ZENITH, 30.0)
        assert np.sum(net.C * (T1 - T)) == pytest.approx(Q.sum() * 30.0, rel=1e-9)
        T = T1


@pytest.mark.parametrize(("sat", "spec_s", "n_nodes"), [("SAT-1", 103, 15), ("SAT-5", 124, 14)])
def test_stability_limit_matches_spec(
    params: Params, sat: str, spec_s: float, n_nodes: int
) -> None:
    th = Thermal(ParamView(params, sat))
    assert len(th.names) == n_nodes
    limit = th.network().stability_limit_s(290.0)
    print(f"\n{sat}: explicit Euler limit 2/lambda_max = {limit:.1f} s (spec ~{spec_s} s)")
    assert limit == pytest.approx(spec_s, rel=0.10)
    assert params.scenario.control_tick_s < limit


def test_topology(params: Params) -> None:
    a = Thermal(ParamView(params, "SAT-1"))
    net = a.network()
    G = net.G
    ix = a.names.index
    np.testing.assert_array_equal(G, G.T)
    assert np.all(np.diag(G) == 0)
    for face in ("bus_px", "bus_nx", "bus_py", "bus_ny", "bus_pz", "bus_nz"):
        assert G[ix("bus_core"), ix(face)] == 20
    for p, q in (("bus_px", "bus_py"), ("bus_py", "bus_nx"), ("bus_nx", "bus_ny"),
                 ("bus_ny", "bus_px")):
        assert G[ix(p), ix(q)] == 6.0
    assert G[ix("bus_px"), ix("bus_nx")] == 0
    assert G[ix("bus_pz"), ix("bus_px")] == 4.5
    assert G[ix("bus_nz"), ix("bus_nx")] == 4.5
    assert G[ix("bus_pz"), ix("bus_nx")] == 0
    for w in ("px", "nx", "py", "ny"):
        assert G[ix(f"wing_{w}"), ix(f"bus_{w}")] == 1.5
    for node, g in (("battery", 0.4), ("obc", 9.0), ("radio_pa", 11.0), ("payload", 7.5)):
        assert G[ix(node), ix("bus_core")] == g
        assert np.count_nonzero(G[ix(node)]) == 1
    assert np.count_nonzero(G[ix("wing_px")]) == 1
    # wings absorb on their sunward side, radiate from both
    assert net.area_src[ix("wing_px")] == 0.40
    assert net.area_rad[ix("wing_px")] == 0.80
    b = Thermal(ParamView(params, "SAT-5"))
    assert "wing_ny" not in b.names
    nb = b.network()
    assert nb.area_src[b.names.index("wing_px")] == 0.60
    assert nb.area_rad[b.names.index("wing_px")] == 1.20
    assert nb.area_rad[b.names.index("bus_nz")] == 0.49


def test_environment_terms(params: Params) -> None:
    th = Thermal(ParamView(params, "SAT-1"))
    net = th.network()
    ix = th.names.index
    nadir = np.array([0.0, 0, -1])
    lit = ThermalEnv(1.0, Z, nadir, 6903.137, -0.5)
    dark = dataclasses.replace(lit, shadow=0.0)
    sol_l, alb_l, ir_l = net.absorbed_w(lit)
    sol_d, alb_d, ir_d = net.absorbed_w(dark)
    assert np.all(sol_d == 0)
    assert np.all(alb_d == 0)
    np.testing.assert_array_equal(ir_l, ir_d)
    assert sol_l[ix("bus_nz")] == 0  # faces away from the Sun
    assert sol_l[ix("bus_pz")] > 0
    assert alb_l[ix("bus_nz")] > 0  # faces the Earth
    assert alb_l[ix("bus_pz")] == 0
    f_alb = 0.30 * S * (6378.137 / 6903.137) ** 2 * 0.5
    assert alb_l[ix("bus_nz")] == pytest.approx(0.20 * 0.42 * f_alb)
    f_ir = 237 * (6378.137 / 6903.137) ** 2
    assert ir_l[ix("bus_nz")] == pytest.approx(0.85 * 0.42 * f_ir * 0.20)
    for node in ("battery", "obc", "radio_pa", "payload", "bus_core"):
        assert sol_l[ix(node)] == alb_l[ix(node)] == ir_l[ix(node)] == 0
    # albedo needs the sunlit Earth below: Sun behind the Earth gives none
    night = dataclasses.replace(lit, sun_dot_nadir=0.5)
    assert np.all(net.absorbed_w(night)[1] == 0)


def test_thermal_env(params: Params) -> None:
    q = np.array([1.0, 0, 0, 0])
    env = thermal_env(q, np.array([7000.0, 0, 0]), np.array([0.0, 1, 0]), 0.7)
    np.testing.assert_allclose(env.nadir_body, [-1, 0, 0])
    np.testing.assert_allclose(env.sun_body, [0, 1, 0])
    assert env.sun_dot_nadir == pytest.approx(0)
    assert env.r_km == pytest.approx(7000)
    assert env.shadow == 0.7


def test_f7_override_reaches_network(params: Params) -> None:
    view = ParamView(params, "SAT-2")
    th = Thermal(view)
    i = th.names.index("bus_px")
    assert th.network().alpha[i] == 0.08
    view.overrides["thermal.nodes.bus_px.alpha"] = 0.24
    view.overrides["thermal.nodes.bus_px.epsilon"] = 0.09
    net = th.network()
    assert net.alpha[i] == 0.24
    assert net.eps[i] == 0.09


def test_cold_case_heater_and_inhibit(params: Params) -> None:
    view = ParamView(params, "SAT-1")
    eps, th = Eps(view), Thermal(view)
    T = np.full(len(th.names), 265.0)
    state = eps.initial_state()
    env = SUNLIT_ZENITH
    _, state, T = step_power_and_thermal(
        eps, th, state, T, EpsInputs(1.0, Z, {}), env, 30.0)
    assert state.battery_heater
    assert state.inhibit
    second, state, T2 = step_power_and_thermal(
        eps, th, state, T, EpsInputs(1.0, Z, {}), env, 30.0)
    assert second.p_acc_W == 0
    assert second.curtailed_W > 0
    assert second.loads_W["battery_heater"] == 15
    assert second.heat_W["battery"] >= 15
    # strings use the computed wing temperatures
    np.testing.assert_allclose(second.p_string_W, eps.string_power_w(1.0, Z, th.temps_map(T)))
    warm = eps.string_power_w(1.0, Z, {w: 298.15 for w in th.names})
    assert np.all(second.p_string_W > warm)  # colder cells are more efficient
    ib = th.names.index("battery")
    assert T2[ib] > T[ib]
