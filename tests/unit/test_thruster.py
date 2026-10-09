"""M3: thruster model and the F5/F6 shapes. Spec: Model: actuators (thruster)."""

import numpy as np
import pytest

from expand3.params import Params, ParamView
from expand3.vehicle.attitude import quat_from_axis_angle, quat_rotate
from expand3.vehicle.thruster import dry_mass_kg, thrust_step

G0 = 9.80665
A_MAG = 1.0 / (6 * 3600)  # m/s^2: 1 m/s over a 6 h burn
LENGTH_S = 6 * 3600.0
DT = 10.0


def fly(view: ParamView) -> tuple[float, float, float]:
    """A 6 h burn at constant ordered acceleration; returns (dv, propellant used, integral F/m)."""
    q = np.array([1.0, 0, 0, 0])
    b = view.array("thruster.axis_b")
    m = view.get("mass.wet_mass_kg")
    prop = view.get("thruster.propellant_kg")
    dv = used = integral = 0.0
    for _ in range(round(LENGTH_S / DT)):
        s = thrust_step(view, A_MAG * quat_rotate(q, b), m, prop, q, DT)
        integral += s.F_N / m * DT
        dv += s.dv_m_s
        used += s.dm_kg
        m -= s.dm_kg
        prop -= s.dm_kg
    return dv, used, integral


def test_dry_mass(params: Params) -> None:
    assert dry_mass_kg(ParamView(params, "SAT-1")) == pytest.approx(142.0)
    assert dry_mass_kg(ParamView(params, "SAT-5")) == pytest.approx(172.0)


def test_command_is_clipped_and_along_axis(params: Params) -> None:
    for sat in ("SAT-4", "SAT-5"):
        view = ParamView(params, sat)
        q = quat_from_axis_angle(np.array([0.3, -0.4, 0.8]), 1.1)
        b_eci = quat_rotate(q, view.array("thruster.axis_b"))
        s = thrust_step(view, 1.0 * b_eci, 150.0, 8.0, q, DT)
        assert s.F_cmd_N == view.get("thruster.F_max_N")
        assert pytest.approx(view.get("thruster.F_max_N")) == s.F_N
        np.testing.assert_allclose(s.a_thr_km_s2, s.F_N / 150.0 / 1000 * b_eci, rtol=1e-12)
        small = thrust_step(view, 1e-4 * b_eci, 150.0, 8.0, q, DT)
        assert small.F_cmd_N == pytest.approx(150.0 * 1e-4)
        assert small.mdot_kg_s == pytest.approx(small.F_N / (220 * G0))


def test_nominal_dv_and_propellant(params: Params) -> None:
    view = ParamView(params, "SAT-4")
    dv, used, integral = fly(view)
    assert dv == pytest.approx(integral, rel=1e-12)
    assert dv == pytest.approx(1.0, rel=1e-9)
    m0 = view.get("mass.wet_mass_kg")
    analytic = m0 * (1 - np.exp(-A_MAG * LENGTH_S / (220 * G0)))
    assert used == pytest.approx(analytic, rel=1e-6)


def test_f5_shape(params: Params) -> None:
    dv0, used0, _ = fly(ParamView(params, "SAT-4"))
    view = ParamView(params, "SAT-4")
    view.overrides["thruster.eta_F"] = 0.7
    view.overrides["thruster.Isp_s"] = 0.7 * 220
    dv, used, integral = fly(view)
    assert dv == pytest.approx(0.7 * dv0, rel=1e-9)
    assert dv == pytest.approx(integral, rel=1e-12)
    assert used == pytest.approx(used0, rel=1e-6)  # mass flow unchanged


def test_f6_shape(params: Params) -> None:
    dv0, used0, _ = fly(ParamView(params, "SAT-4"))
    view = ParamView(params, "SAT-4")
    view.overrides["thruster.eta_F"] = 0.7
    dv, used, _ = fly(view)
    assert dv == pytest.approx(0.7 * dv0, rel=1e-9)
    assert used == pytest.approx(0.7 * used0, rel=1e-3)  # mass flow falls to 0.7x


def test_empty_tank_stops_thrust(params: Params) -> None:
    view = ParamView(params, "SAT-2")
    q = np.array([1.0, 0, 0, 0])
    a = 0.01 * view.array("thruster.axis_b")
    partial = thrust_step(view, a, 150.0, 1e-4, q, DT)
    assert partial.dm_kg == pytest.approx(1e-4)
    assert partial.F_N < 0.01 * 150.0
    empty = thrust_step(view, a, 150.0, 0.0, q, DT)
    assert empty.F_N == 0
    assert empty.dv_m_s == 0
    np.testing.assert_array_equal(empty.a_thr_km_s2, 0)
