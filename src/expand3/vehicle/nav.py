"""Onboard navigation filter: an extended Kalman filter on the GNSS fixes (extended faults).

Not in the spec; a vehicle model beside the sensors (docs/extended-faults-plan.md, Phase A). The
state is r [km] and v [km/s] in ECI and a drag scale (the density as a multiple of the nominal
one, 1 at the start). Between fixes it is propagated with the same force model as the orbit
(two-body, J2, exponential drag; Spec: Model: orbit and attitude), but the atmosphere is the
middle of the seed's draw, because the vehicle does not know the density, and the scale learns the
difference. The thrust the vehicle ordered (its own burn command, held between samples) is a known
input; what it delivers beyond that, as under F5 and F6, is left to the process noise. Each fix
updates it with the receiver's nominal noise (Spec: Signals, GNSS), not the noise a fault may
have made of it, and no fix is rejected. A fault in the fixes therefore enters the navigation
solution, and how fast it does depends on the filter's gain.

The filter reads the fixes and nothing else; the true state is never an input.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial

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
_N_STATE = _DIM + 1  # position, velocity and the drag scale


@dataclass(frozen=True)
class NavSolution:
    """The filter's estimate at each fix time."""

    t_s: F64
    r_km: F64
    v_km_s: F64
    drag_scale: F64  # the density as a multiple of the nominal one


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


def navigate(view: ParamView, model: OrbitModel, t_s: F64, r_fix_km: F64, v_fix_km_s: F64,
             t_cmd_s: F64, a_cmd_m_s2: F64) -> NavSolution:
    """Run the filter over a vehicle's fixes (rows aligned with `t_s`, ascending). `a_cmd_m_s2`
    (n, 3) is the thrust the vehicle ordered at each time of `t_cmd_s`, held until the next."""
    step_s = view.get("signals.nav.step_s")
    accel_sigma = view.get("signals.nav.accel_sigma_m_s2") / M_PER_KM  # km/s^2
    scale_sigma = view.get("signals.nav.drag_scale_sigma")
    scale_walk = view.get("signals.nav.drag_scale_walk_per_sqrt_s")
    sig_r = view.get("signals.gnss.pos_sigma_m") / M_PER_KM
    sig_v = view.get("signals.gnss.vel_sigma_m_s") / M_PER_KM
    meas_cov = np.diag([sig_r**2] * _AXES + [sig_v**2] * _AXES)
    h_mat = np.hstack([np.eye(_DIM), np.zeros((_DIM, 1))])
    eye = np.eye(_N_STATE)
    no_drag = replace(model, forces=Forces(j2=True, drag=False))

    def drag_accel(r: F64, v: F64) -> F64:
        """The model's drag at the nominal density (km/s^2); the state scales it."""
        full = model.accel(r[None, :], v[None, :])
        out: F64 = (full - no_drag.accel(r[None, :], v[None, :]))[0]
        return out

    def deriv(y: F64, a_thr: F64) -> F64:
        r, v, scale = y[:_AXES], y[_AXES:_DIM], y[_DIM]
        acc = no_drag.accel(r[None, :], v[None, :])[0] + scale * drag_accel(r, v) + a_thr
        return np.concatenate([v, acc, [0.0]])

    n = len(t_s)
    r_out = np.empty((n, _AXES))
    v_out = np.empty((n, _AXES))
    scale_out = np.ones(n)
    x = np.concatenate([r_fix_km[0], v_fix_km_s[0], [1.0]])
    cov = np.zeros((_N_STATE, _N_STATE))
    cov[:_DIM, :_DIM] = meas_cov
    cov[_DIM, _DIM] = scale_sigma**2
    r_out[0], v_out[0] = x[:_AXES], x[_AXES:_DIM]
    for i in range(1, n):
        dt = float(t_s[i] - t_s[i - 1])
        steps = max(round(dt / step_s), 1)
        h = dt / steps
        # transition: the gravity gradient and the drag acceleration per unit scale
        a = np.zeros((_N_STATE, _N_STATE))
        a[:_AXES, _AXES:_DIM] = np.eye(_AXES)
        a[_AXES:_DIM, :_AXES] = _gravity_gradient(model, x[:_AXES], x[_AXES:_DIM])
        a[_AXES:_DIM, _DIM] = drag_accel(x[:_AXES], x[_AXES:_DIM])
        phi = expm(a * dt)
        for j in range(steps):
            k = max(int(np.searchsorted(t_cmd_s, t_s[i - 1] + j * h, side="right")) - 1, 0)
            a_thr = a_cmd_m_s2[k] / M_PER_KM
            x = rk4_step(partial(deriv, a_thr=a_thr), x, h)
        g = np.concatenate([np.eye(_AXES) * dt**2 / 2.0, np.eye(_AXES) * dt])
        noise = np.zeros((_N_STATE, _N_STATE))
        noise[:_DIM, :_DIM] = accel_sigma**2 * (g @ g.T)
        noise[_DIM, _DIM] = scale_walk**2 * dt
        cov = phi @ cov @ phi.T + noise
        z = np.concatenate([r_fix_km[i], v_fix_km_s[i]])
        gain = cov @ h_mat.T @ np.linalg.inv(h_mat @ cov @ h_mat.T + meas_cov)
        x = x + gain @ (z - h_mat @ x)
        x[_DIM] = max(x[_DIM], 0.0)  # a density cannot be negative
        ikh = eye - gain @ h_mat
        cov = ikh @ cov @ ikh.T + gain @ meas_cov @ gain.T
        r_out[i], v_out[i], scale_out[i] = x[:_AXES], x[_AXES:_DIM], x[_DIM]
    return NavSolution(np.asarray(t_s, dtype=np.float64), r_out, v_out, scale_out)
