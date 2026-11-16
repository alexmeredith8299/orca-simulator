"""Onboard navigation filter: an extended Kalman filter on the GNSS fixes (extended faults).

Not in the spec; a vehicle model beside the sensors (docs/extended-faults-plan.md, Phase A). The
state is r [km] and v [km/s] in ECI. Between fixes it is propagated with the same force model as
the orbit (two-body, J2, exponential drag; Spec: Model: orbit and attitude) but with the nominal
atmosphere, the middle of the seed's draw, because the vehicle does not know the density. Thrust
is not modelled, so a burn is left to the process noise. Each fix updates it with the receiver's
nominal noise (Spec: Signals, GNSS), not the noise a fault may have made of it, and no fix is
rejected. A fault in the fixes therefore enters the navigation solution, and how fast it does
depends on the filter's gain.

The filter reads the fixes and nothing else; the true state is never an input.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm

from expand3.params import Draw, Params, ParamView
from expand3.timebase import SeedDraws, draw_seed
from expand3.units import M_PER_KM
from expand3.vehicle.orbit import Forces, OrbitModel, rk4_step

F64 = NDArray[np.float64]

_AXES = 3
_DIM = 2 * _AXES


@dataclass(frozen=True)
class NavSolution:
    """The filter's estimate at each fix time."""

    t_s: F64
    r_km: F64
    v_km_s: F64


def _nominal(draw: Draw) -> float:
    """The middle of a draw: geometric for a log-uniform one."""
    if draw.low is None or draw.high is None:
        raise ValueError("a nominal value needs a draw with low and high")
    return float(np.sqrt(draw.low * draw.high) if draw.draw == "loguniform"
                 else 0.5 * (draw.low + draw.high))


def nav_model(params: Params, sat: str, pole: F64) -> OrbitModel:
    """The vehicle's force model: J2 and drag, with the middle of the density draw."""
    d = params.scenario.drag
    draws: SeedDraws = replace(
        draw_seed(params, 0), rho0_kg_m3=_nominal(d.rho0_kg_m3), H_km=_nominal(d.H_km))
    return OrbitModel.from_params(params, [sat], draws, pole, Forces())


def _gravity_gradient(model: OrbitModel, r: F64, v: F64) -> F64:
    """d a / d r by central differences of the force model (3x3, 1/s^2). The step is a fraction
    of the position, scaled by its norm."""
    h = float(np.linalg.norm(r)) * 1.0e-4
    eye = np.eye(_AXES) * h
    plus = model.accel(r[None, :] + eye, np.tile(v, (_AXES, 1)))
    minus = model.accel(r[None, :] - eye, np.tile(v, (_AXES, 1)))
    return np.asarray((plus - minus).T / (2.0 * h), dtype=np.float64)


def navigate(view: ParamView, model: OrbitModel, t_s: F64, r_fix_km: F64, v_fix_km_s: F64
             ) -> NavSolution:
    """Run the filter over a vehicle's fixes (rows aligned with `t_s`, ascending)."""
    step_s = view.get("signals.nav.step_s")
    accel_sigma = view.get("signals.nav.accel_sigma_m_s2") / M_PER_KM  # km/s^2
    sig_r = view.get("signals.gnss.pos_sigma_m") / M_PER_KM
    sig_v = view.get("signals.gnss.vel_sigma_m_s") / M_PER_KM
    meas_cov = np.diag([sig_r**2] * _AXES + [sig_v**2] * _AXES)
    eye = np.eye(_DIM)

    n = len(t_s)
    r_out = np.empty((n, _AXES))
    v_out = np.empty((n, _AXES))
    x = np.concatenate([r_fix_km[0], v_fix_km_s[0]])
    cov = meas_cov.copy()
    r_out[0], v_out[0] = x[:_AXES], x[_AXES:]
    for i in range(1, n):
        dt = float(t_s[i] - t_s[i - 1])
        steps = max(round(dt / step_s), 1)
        h = dt / steps
        # transition from the gravity gradient (drag's velocity dependence is far below the noise)
        a = np.zeros((_DIM, _DIM))
        a[:_AXES, _AXES:] = np.eye(_AXES)
        a[_AXES:, :_AXES] = _gravity_gradient(model, x[:_AXES], x[_AXES:])
        phi = expm(a * dt)
        y = x[None, :]
        for _ in range(steps):
            y = rk4_step(model.deriv, y, h)
        x = y[0]
        g = np.concatenate([np.eye(_AXES) * dt**2 / 2.0, np.eye(_AXES) * dt])
        cov = phi @ cov @ phi.T + accel_sigma**2 * (g @ g.T)
        z = np.concatenate([r_fix_km[i], v_fix_km_s[i]])
        gain = cov @ np.linalg.inv(cov + meas_cov)
        x = x + gain @ (z - x)
        ikh = eye - gain
        cov = ikh @ cov @ ikh.T + gain @ meas_cov @ gain.T
        r_out[i], v_out[i] = x[:_AXES], x[_AXES:]
    return NavSolution(np.asarray(t_s, dtype=np.float64), r_out, v_out)
