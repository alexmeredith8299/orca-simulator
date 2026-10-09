"""M0: params/*.yaml load, Bus B merges onto Bus A, and spot values match the spec tables."""

import math

import numpy as np
import pytest

from expand3.params import Params, ParamView

SPOT_CHECKS = [
    # (vehicle, ParamView path, spec value)
    ("SAT-1", "eps.E_cap_J", 3.6e6),
    ("SAT-1", "eps.R_pack_ohm", 0.08),
    ("SAT-1", "eps.P_c_max_W", 300),
    ("SAT-1", "eps.strings", 4),
    ("SAT-1", "thermal.stability_limit_s", 103),
    ("SAT-5", "eps.E_cap_J", 4.32e6),
    ("SAT-5", "eps.R_pack_ohm", 0.064),
    ("SAT-5", "eps.P_c_max_W", 360),
    ("SAT-5", "eps.strings", 3),
    ("SAT-5", "eps.eta0", 0.28),
    ("SAT-5", "eps.eta_c", 0.96),  # inherited from Bus A
    ("SAT-5", "mass.wet_mass_kg", 180),
    ("SAT-5", "wheels.tau_max_Nm", 0.075),
    ("SAT-5", "wheels.spin_axis_angle_off_z_deg", 54.7),  # inherited
    ("SAT-5", "thruster.F_max_N", 4),  # inherited
    ("SAT-5", "thruster.axis_b[1]", 1),
    ("SAT-5", "thermal.nodes.bus_px.area_m2", 0.70),
    ("SAT-5", "thermal.nodes.bus_nz.area_m2", 0.49),
    ("SAT-5", "thermal.nodes.bus_nz.epsilon", 0.85),  # inherited
    ("SAT-5", "thermal.nodes.bus_py.C_J_K", 3000),
    ("SAT-5", "thermal.nodes.wing_px.area_m2", 1.20),
    ("SAT-5", "thermal.nodes.wing_py.sunward_m2", 0.60),
    ("SAT-5", "thermal.nodes.wing_nx.C_J_K", 2800),
    ("SAT-5", "thermal.nodes.battery.C_J_K", 9500),
    ("SAT-5", "thermal.nodes.bus_core.C_J_K", 72000),
    ("SAT-5", "thermal.nodes.obc.C_J_K", 2000),  # unchanged
    ("SAT-5", "thermal.stability_limit_s", 124),
    ("SAT-2", "cdh_upsets.lambda_in_per_s", 1 / 1800),
    ("SAT-2", "cdh_upsets.lambda_out_per_s", 1 / 86400),
    ("SAT-2", "crosslink.drop_probability", 0),
    ("SAT-2", "crosslink.range_km", 1500),
    ("SAT-2", "imager_dark_current.D0_e_s_px", 10),
    ("SAT-2", "scenario.constants.mu_km3_s2", 398600.4418),
    ("SAT-2", "scenario.constants.J2", 1.08262668e-3),
    ("SAT-2", "scenario.orbit.a_km", 6903.137),
    ("SAT-2", "scenario.control_tick_s", 30),
]


@pytest.mark.parametrize(("sat", "path", "expected"), SPOT_CHECKS)
def test_spot_values(params: Params, sat: str, path: str, expected: float) -> None:
    assert ParamView(params, sat).get(path) == pytest.approx(expected, rel=1e-12, abs=0)


def test_all_numbers_are_numbers(params: Params) -> None:
    # PyYAML's YAML 1.1 rules read 3.6e6 as a string; the loader must not.
    assert isinstance(params.buses["A"].eps.E_cap_J, float)
    assert isinstance(params.buses["B"].eps.E_cap_J, float)


def test_bus_assignment(params: Params) -> None:
    assert params.sat_ids == [f"SAT-{i}" for i in range(1, 7)]
    assert params.bus("SAT-4") is params.buses["A"]
    assert params.bus("SAT-5") is params.buses["B"]
    assert params.bus("SAT-6") is params.buses["B"]


def test_node_counts(params: Params) -> None:
    a, b = params.buses["A"], params.buses["B"]
    assert len(a.thermal.nodes) == 15
    assert len(b.thermal.nodes) == 14
    assert "wing_ny" in a.thermal.nodes
    assert "wing_ny" not in b.thermal.nodes
    assert b.wings.names == ["wing_px", "wing_nx", "wing_py"]
    assert b.wings.hinge == a.wings.hinge  # merged from Bus A


def test_bus_a_untouched_by_merge(params: Params) -> None:
    a = params.buses["A"]
    assert a.thermal.nodes["bus_px"].area_m2 == 0.63
    assert a.thermal.nodes["wing_px"].C_J_K == 2000
    assert a.thermal.nodes["bus_core"].C_J_K == 60000


@pytest.mark.parametrize("bus_id", ["A", "B"])
def test_every_load_and_node(params: Params, bus_id: str) -> None:
    bus = params.buses[bus_id]  # type: ignore[index]
    assert set(bus.loads) == {
        "obc", "radio_receiver", "adcs", "edge_computer", "s_band_tx",
        "x_band_downlink", "crosslink", "imager", "battery_heater", "payload_heater",
    }  # fmt: skip
    for load in bus.loads.values():
        assert load.node in bus.thermal.nodes
    for node in bus.thermal.nodes.values():
        assert node.C_J_K > 0


def test_ocv_at_half_soc(params: Params) -> None:
    for bus in params.buses.values():
        ocv = bus.eps.ocv_curve
        assert np.interp(0.5, ocv.soc, ocv.volts) == pytest.approx(30.4)


def test_faults(params: Params) -> None:
    f = params.faults.faults
    assert set(f) == {f"F{i}" for i in range(1, 11)}
    assert f["F1"].m == 3
    assert f["F1"].profile == "ramp"
    assert f["F1"].t_r_s == 21600
    assert f["F3"].set_to == 0.02
    assert f["F5"].paths == ["thruster.eta_F", "thruster.Isp_s"]
    assert f["F7"].cap == 1.0
    assert f["F8"].set_to == 0.9


def test_scenario(params: Params) -> None:
    s = params.scenario
    assert s.run_length_h.low == 48
    assert s.run_length_h.high == 72
    assert s.drag.rho0_kg_m3.draw == "loguniform"
    assert [g.name for g in s.ground_stations] == ["Svalbard", "Fairbanks", "Wallops"]
    assert s.true_anomaly_at_epoch_deg["SAT-6"] == 184
    assert math.isclose(s.orbit.a_km - s.constants.R_E_km, 525, abs_tol=1e-6)


def test_paramview_overrides(params: Params) -> None:
    v = ParamView(params, "SAT-2")
    assert v.get("eps.R_pack_ohm") == 0.08
    v.overrides["eps.R_pack_ohm"] = 0.24
    assert v.get("eps.R_pack_ohm") == 0.24
    assert ParamView(params, "SAT-1").get("eps.R_pack_ohm") == 0.08
    assert params.buses["A"].eps.R_pack_ohm == 0.08

    v.overrides["wheels.k[1]"] = 0.5
    k = v.array("wheels.k")
    assert k.tolist() == [1, 0.5, 1, 1]
    with pytest.raises(ValueError, match="read-only"):
        k[0] = 2.0


def test_params_are_frozen(params: Params) -> None:
    with pytest.raises(ValueError, match="frozen"):
        params.buses["A"].eps.R_pack_ohm = 1.0  # type: ignore[misc]


@pytest.mark.parametrize("sat", ["SAT-1", "SAT-5"])
def test_fault_target_nominals(params: Params, sat: str) -> None:
    v = ParamView(params, sat)
    n_strings = int(v.get("eps.strings"))
    n_wheels = int(v.get("wheels.count"))
    assert v.array("eps.string_health").tolist() == [1] * n_strings
    assert v.array("wheels.k").tolist() == [1] * n_wheels
    assert v.array("wheels.c").tolist() == [0] * n_wheels
    assert (n_strings, n_wheels) == ((4, 4) if sat == "SAT-1" else (3, 4))
