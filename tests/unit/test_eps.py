"""M4: electrical power. Spec: Model: electrical power; Parameters > Electrical power, Loads."""

import numpy as np
import pytest

from expand3.params import Params, ParamView
from expand3.vehicle.eps import Eps, EpsInputs, EpsState, placeholder_temperatures

Z = np.array([0.0, 0.0, 1.0])
DT = 30.0
SURVIVAL = {"obc", "radio_receiver", "battery_heater"}


def eps_for(params: Params, sat: str = "SAT-1") -> tuple[ParamView, Eps]:
    view = ParamView(params, sat)
    return view, Eps(view)


def temps(view: ParamView, **kw: float) -> dict[str, float]:
    t = placeholder_temperatures(view)
    t.update(kw)
    return t


def inputs(view: ParamView, **kw: object) -> EpsInputs:
    base: dict[str, object] = {"shadow": 1.0, "sun_body": Z, "node_temps_K": temps(view)}
    base.update(kw)
    return EpsInputs(**base)  # type: ignore[arg-type]


@pytest.mark.parametrize(("sat", "per_string", "n"), [("SAT-1", 124.80, 4), ("SAT-5", 180.75, 3)])
def test_string_power_full_sun(params: Params, sat: str, per_string: float, n: int) -> None:
    view, eps = eps_for(params, sat)
    t = temps(view, **{w: 298.15 for w in view.params.bus(sat).wings.names})
    p = eps.string_power_w(1.0, Z, t)
    # S eta0 A p eta_ab at T0, normal incidence
    assert len(p) == n
    np.testing.assert_allclose(p, per_string, rtol=1e-3)
    s, eta0 = 1361.0, view.get("eps.eta0")
    exact = s * eta0 * view.get("eps.string_area_m2") * 0.85 * 0.93
    np.testing.assert_allclose(p, exact, rtol=1e-12)


def test_temperature_incidence_and_shadow(params: Params) -> None:
    view, eps = eps_for(params)
    names = view.params.bus("SAT-1").wings.names
    base = eps.string_power_w(1.0, Z, temps(view, **{w: 298.15 for w in names}))
    hot = eps.string_power_w(1.0, Z, temps(view, **{w: 308.15 for w in names}))
    np.testing.assert_allclose(hot / base, (0.29 - 3e-4 * 10) / 0.29, rtol=1e-12)
    tilted = eps.string_power_w(1.0, np.array([np.sin(0.5), 0, np.cos(0.5)]), temps(view))
    np.testing.assert_allclose(tilted, eps.string_power_w(1.0, Z, temps(view)) * np.cos(0.5))
    assert np.all(eps.string_power_w(1.0, -Z, temps(view)) == 0)
    assert np.all(eps.string_power_w(0.0, Z, temps(view)) == 0)
    np.testing.assert_allclose(eps.string_power_w(0.4, Z, temps(view)),
                               0.4 * eps.string_power_w(1.0, Z, temps(view)))
    # one string per wing: a cold wing changes only its own string
    t = temps(view)
    t[names[2]] = 250.0
    p = eps.string_power_w(1.0, Z, t)
    ref = eps.string_power_w(1.0, Z, temps(view))
    assert p[2] > ref[2]
    np.testing.assert_array_equal(np.delete(p, 2), np.delete(ref, 2))


def test_f2_string_open(params: Params) -> None:
    view, eps = eps_for(params, "SAT-2")
    nominal = eps.string_power_w(1.0, Z, temps(view))
    view.overrides["eps.string_health[0]"] = 0.0
    faulted = eps.string_power_w(1.0, Z, temps(view))
    assert faulted[0] == 0
    np.testing.assert_array_equal(faulted[1:], nominal[1:])
    assert nominal.sum() - faulted.sum() == pytest.approx(nominal.sum() / 4)


def test_load_gating(params: Params) -> None:
    view, eps = eps_for(params)
    st = eps.initial_state()
    idle = eps.loads_w(st, inputs(view))
    assert idle == {
        "obc": 12, "radio_receiver": 3, "adcs": 20, "edge_computer": 15, "s_band_tx": 0,
        "x_band_downlink": 0, "crosslink": 6, "imager": 10, "battery_heater": 0,
        "payload_heater": 0,
    }  # fmt: skip
    p = eps.loads_w(st, inputs(view, in_pass=True))
    assert (p["s_band_tx"], p["x_band_downlink"]) == (15, 45)
    tk = eps.loads_w(st, inputs(view, in_pass=True, take=True))
    assert (tk["s_band_tx"], tk["x_band_downlink"], tk["adcs"], tk["imager"]) == (0, 0, 25, 60)
    sl = eps.loads_w(st, inputs(view, wheel_power_W=40.0))
    assert sl["adcs"] == 60
    assert eps.loads_w(st, inputs(view, edge_off=True))["edge_computer"] == 0
    heaters = eps.loads_w(EpsState(st.E_J, battery_heater=True, payload_heater=True), inputs(view))
    assert (heaters["battery_heater"], heaters["payload_heater"]) == (15, 10)
    shed = eps.loads_w(
        EpsState(st.E_J, uv_hold=True, battery_heater=True, payload_heater=True),
        inputs(view, in_pass=True, take=True, wheel_power_W=40.0),
    )
    for name, w in shed.items():
        if name in SURVIVAL:
            assert w > 0
        else:
            assert w == 0, name


def test_derate_and_ocv(params: Params) -> None:
    view, eps = eps_for(params)
    e_cap = view.get("eps.E_cap_J")
    for t_k, f in zip([253.15, 273.15, 298.15, 318.15, 333.15], [0.65, 0.88, 1.0, 0.97, 0.85],
                      strict=True):
        assert eps.usable_energy_j(t_k) == pytest.approx(e_cap * f, rel=1e-12)
    assert eps.usable_energy_j(263.15) == pytest.approx(e_cap * (0.65 + 0.88) / 2)
    assert eps.usable_energy_j(200.0) == pytest.approx(e_cap * 0.65)
    assert eps.ocv_v(0.5) == pytest.approx(30.4)
    assert eps.ocv_v(0.2) == pytest.approx(28.8)
    assert eps.ocv_v(1.2) == pytest.approx(33.6)


def run(eps: Eps, view: ParamView, state: EpsState, n: int, **kw: object) -> list:
    out = []
    for _ in range(n):
        step, state = eps.step(state, inputs(view, **kw), DT)
        out.append(step)
    return out


@pytest.mark.parametrize("sat", ["SAT-1", "SAT-5"])
def test_energy_bookkeeping_and_voltage(params: Params, sat: str) -> None:
    view, eps = eps_for(params, sat)
    e_cap = view.get("eps.E_cap_J")
    steps = []
    state = EpsState(0.3 * e_cap)
    for k in range(600):  # alternate sun and eclipse, 30 min each, with passes
        kw = {"shadow": 1.0 if (k // 60) % 2 == 0 else 0.0, "in_pass": k % 7 == 0}
        step, state = eps.step(state, inputs(view, **kw), DT)
        steps.append(step)
    for s in steps:
        assert s.E_next_J - s.E_J == pytest.approx(
            (0.96 * s.p_acc_W - s.p_sup_W / 0.98) * DT, abs=1e-6)
        assert s.p_gen_W - s.demand_W == pytest.approx(s.p_pack_W + s.curtailed_W - s.unmet_W,
                                                       abs=1e-9)
        assert 0 <= s.soc <= 1
        assert 0 <= s.E_next_J <= e_cap * (1 + 1e-12)
        assert s.p_acc_W <= view.get("eps.P_c_max_W") + 1e-12
        assert s.p_acc_W <= (e_cap - s.E_J) / (0.96 * DT) + 1e-9
        assert s.v_V == pytest.approx(s.ocv_V + s.i_A * view.get("eps.R_pack_ohm"), abs=1e-12)
        if s.p_pack_W > 0:
            assert s.v_V > s.ocv_V
        if s.p_pack_W < 0:
            assert s.v_V < s.ocv_V
    assert any(s.curtailed_W > 0 for s in steps)  # charged to full in the sun
    assert any(s.p_sup_W > 0 for s in steps)


def test_undervoltage_reflex(params: Params) -> None:
    view, eps = eps_for(params)
    e_cap = view.get("eps.E_cap_J")
    state = EpsState(0.03 * e_cap)
    steps = run(eps, view, state, 40, shadow=0.0, in_pass=True)
    first = next(i for i, s in enumerate(steps) if s.v_V <= 26.4)
    assert not steps[first].shed
    assert steps[first + 1].shed  # applied from the next tick
    for s in steps[first + 1:]:
        for name, w in s.loads_W.items():
            if name not in SURVIVAL:
                assert w == 0
    # recover in the sun: shed holds until V >= 28.0 V, then releases on the next tick
    state = EpsState(steps[-1].E_next_J, uv_hold=True)
    rec = run(eps, view, state, 400, shadow=1.0)
    release = next(i for i, s in enumerate(rec) if s.v_V >= 28.0)
    assert all(s.shed for s in rec[: release + 1])
    assert not rec[release + 1].shed
    assert all(s.v_V < 28.0 for s in rec[:release])


def test_cold_charge_inhibit_and_heaters(params: Params) -> None:
    view, eps = eps_for(params)
    e_cap = view.get("eps.E_cap_J")
    st = EpsState(0.5 * e_cap)

    def go(state: EpsState, battery: float, payload: float = 293.15) -> EpsState:
        t = temps(view, battery=battery, payload=payload)
        return eps.step(state, inputs(view, node_temps_K=t), DT)[1]

    assert go(st, 273.15).inhibit
    assert not go(st, 273.16).inhibit
    held = EpsState(st.E_J, inhibit=True)
    assert go(held, 276.15).inhibit
    assert not go(held, 276.16).inhibit
    step, _ = eps.step(held, inputs(view, node_temps_K=temps(view, battery=275.0)), DT)
    assert step.p_acc_W == 0  # inhibited: no charge even in full sun
    assert step.curtailed_W > 0

    assert go(st, 283.15).battery_heater
    assert not go(st, 283.16).battery_heater
    on = EpsState(st.E_J, battery_heater=True, payload_heater=True)
    assert go(on, 288.14).battery_heater
    assert not go(on, 288.15).battery_heater
    assert go(st, 290.0, payload=263.15).payload_heater
    assert not go(st, 290.0, payload=263.16).payload_heater
    assert go(on, 290.0, payload=268.14).payload_heater
    assert not go(on, 290.0, payload=268.15).payload_heater
    # true-temperature cutoff forces both off, even with a cold sensed reading
    cut = eps.step(on, inputs(view, node_temps_K=temps(view, battery=313.16, payload=313.16),
                              sensed_battery_K=280.0), DT)[1]
    assert not cut.battery_heater
    assert not cut.payload_heater


def test_heat_conservation(params: Params) -> None:
    view, eps = eps_for(params, "SAT-5")
    e_cap = view.get("eps.E_cap_J")
    state = EpsState(0.5 * e_cap, battery_heater=True, payload_heater=True)
    for k in range(200):
        kw = {"shadow": 1.0 if k < 100 else 0.0, "in_pass": k % 3 == 0, "take": k == 50,
              "wheel_power_W": 24.0 if k == 49 else 0.0}
        step, state = eps.step(state, inputs(view, **kw), DT)
        loads = view.params.bus("SAT-5").loads
        component = sum(w * loads[n].heat_fraction for n, w in step.loads_W.items())
        total = component + step.pack_loss_W + step.array_conv_loss_W + step.load_conv_loss_W
        assert sum(step.heat_W.values()) == pytest.approx(total, rel=1e-12)
        loss = 0.04 * step.p_acc_W + (1 / 0.98 - 1) * step.p_sup_W
        assert step.pack_loss_W == pytest.approx(loss)
        assert step.load_conv_loss_W == pytest.approx(step.demand_W * (1 - 0.92))
        assert set(step.heat_W) <= set(view.params.bus("SAT-5").thermal.nodes)


def test_f1_resistance_ramp(params: Params) -> None:
    view, eps = eps_for(params, "SAT-2")
    fview, feps = eps_for(params, "SAT-2")
    r0 = view.get("eps.R_pack_ohm")
    e_cap = view.get("eps.E_cap_J")
    s_n = s_f = EpsState(0.6 * e_cap)
    ramp = 6 * 3600.0
    sags = []
    for k in range(int(ramp / DT) + 20):
        frac = min(1.0, k * DT / ramp)
        fview.overrides["eps.R_pack_ohm"] = r0 * (1 + (3 - 1) * frac)
        kw = {"shadow": 0.0 if (k // 60) % 2 else 1.0, "in_pass": True}
        a, s_n = eps.step(s_n, inputs(view, **kw), DT)
        b, s_f = feps.step(s_f, inputs(fview, **kw), DT)
        assert b.E_J == a.E_J  # R sets the voltage only
        assert b.soc == a.soc
        if a.p_pack_W < 0:
            r_now = fview.get("eps.R_pack_ohm")
            # sag is I R exactly; for the same pack power I rises slightly as V falls
            assert (b.ocv_V - b.v_V) / -b.i_A == pytest.approx(r_now, rel=1e-12)
            sags.append(((b.ocv_V - b.v_V) / (a.ocv_V - a.v_V), r_now / r0))
    ratio, r_ratio = np.array(sags).T
    np.testing.assert_allclose(ratio, r_ratio, rtol=0.03)
    assert r_ratio.max() == pytest.approx(3.0, abs=0.01)


def test_f1_reflex_fires_earlier(params: Params) -> None:
    view, eps = eps_for(params, "SAT-2")
    fview, feps = eps_for(params, "SAT-2")
    fview.overrides["eps.R_pack_ohm"] = 3 * view.get("eps.R_pack_ohm")
    e0 = 0.12 * view.get("eps.E_cap_J")
    n = run(eps, view, EpsState(e0), 200, shadow=0.0, in_pass=True)
    f = run(feps, fview, EpsState(e0), 200, shadow=0.0, in_pass=True)
    first_n = next(i for i, s in enumerate(n) if s.v_V <= 26.4)
    first_f = next(i for i, s in enumerate(f) if s.v_V <= 26.4)
    assert first_f < first_n
