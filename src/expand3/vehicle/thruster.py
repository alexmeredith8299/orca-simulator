"""Monopropellant thruster: commanded and delivered thrust, acceleration, tank drain (M3).

Spec: Model: actuators (F5, F6), thruster. One body-fixed thruster with axis b:

    F_cmd = min(m |a_cmd|, F_max),  F = eta_F F_cmd,  a_thr = (F / m) R(q) b,
    m_dot = -F / (I_sp g_0)

eta_F and I_sp are read through ParamView on every call, so F5 (eta_F and I_sp x0.7, mass flow
unchanged) and F6 (eta_F x0.7, mass flow x0.7) apply from their onset. The acceleration is held
over the integration step (zero-order hold) with the mass at the step start. When the tank runs
dry within a step, thrust is cut so that exactly the remaining propellant is used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from expand3.params import ParamView
from expand3.units import M_PER_KM
from expand3.vehicle.attitude import quat_rotate

F64 = NDArray[np.float64]


@dataclass(frozen=True)
class ThrustStep:
    """One integration step of thrust: forces [N], acceleration [km/s^2], flow [kg/s], and the
    delta-v [m/s] and propellant [kg] delivered over the step."""

    F_cmd_N: float
    F_N: float
    a_thr_km_s2: F64
    mdot_kg_s: float
    dv_m_s: float
    dm_kg: float


def dry_mass_kg(view: ParamView) -> float:
    """Wet mass minus the loaded propellant. Spec: Parameters > Vehicle and actuators."""
    return view.get("mass.wet_mass_kg") - view.get("thruster.propellant_kg")


def thrust_step(
    view: ParamView, a_cmd_m_s2: F64, mass_kg: float, propellant_kg: float, q: F64, dt: float
) -> ThrustStep:
    """Spec: Model: actuators, thruster. a_cmd_m_s2 is the ordered acceleration (ECI, m/s^2)."""
    f_cmd = min(mass_kg * float(np.linalg.norm(a_cmd_m_s2)), view.get("thruster.F_max_N"))
    if propellant_kg <= 0:
        return ThrustStep(f_cmd, 0.0, np.zeros(3), 0.0, 0.0, 0.0)
    f = view.get("thruster.eta_F") * f_cmd
    exhaust = view.get("thruster.Isp_s") * view.get("scenario.constants.g0_m_s2")
    dm = f / exhaust * dt
    if dm > propellant_kg:  # tank runs dry inside this step
        f *= propellant_kg / dm
        dm = propellant_kg
    b_eci = quat_rotate(q, view.array("thruster.axis_b"))
    return ThrustStep(
        F_cmd_N=f_cmd,
        F_N=f,
        a_thr_km_s2=f / mass_kg / M_PER_KM * b_eci,
        mdot_kg_s=f / exhaust,
        dv_m_s=f / mass_kg * dt,
        dm_kg=dm,
    )
