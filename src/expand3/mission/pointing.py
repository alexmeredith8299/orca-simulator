"""Pointing law in place of the flight stack: idle Sun hold, burn, imaging take (M2).

Spec: Model: orbit and attitude (pointing law); Simplifications (pointing is ideal and may use the
true state). Each law returns the body-to-inertial quaternion directly. Every law is a TRIAD: a
primary body axis is put exactly on an inertial direction, and the free roll about it is set so a
secondary body axis comes as close as possible to a second direction.

    Idle: array normal on the Sun; roll so body +x is as close as possible to the orbit normal.
    Burn: thrust axis b on the ordered acceleration; roll so the array normal is closest to the
          Sun (closed form: the array normal goes to the Sun direction projected normal to b).
    Take: imager boresight on the target. Roll about the boresight cannot move the array normal
          when the boresight is -z, so roll uses the idle convention (+x toward orbit normal).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from expand3.params import ParamView
from expand3.vehicle.attitude import dcm_to_quat

F64 = NDArray[np.float64]

_AXES = {"x": 0, "y": 1, "z": 2}


def axis_from_label(label: str) -> F64:
    """Body unit vector for a face/axis label such as "+z" or "-x"."""
    sign, name = label.strip()[0], label.strip()[1:]
    if sign not in "+-" or name not in _AXES:
        raise ValueError(f"bad axis label {label!r}")
    v = np.zeros(3)
    v[_AXES[name]] = 1.0 if sign == "+" else -1.0
    return v


def _frame(p: F64, s: F64) -> F64:
    """Orthonormal columns [t1, t2, t1 x t2] with t1 along p and t2 normal to p and s. Written in
    scalars: this runs every tick, and numpy's overhead on three numbers dominates."""
    p0, p1, p2 = float(p[0]), float(p[1]), float(p[2])
    n1 = math.sqrt(p0 * p0 + p1 * p1 + p2 * p2)
    a0, a1, a2 = p0 / n1, p1 / n1, p2 / n1
    c0 = a1 * float(s[2]) - a2 * float(s[1])
    c1 = a2 * float(s[0]) - a0 * float(s[2])
    c2 = a0 * float(s[1]) - a1 * float(s[0])
    n2 = math.sqrt(c0 * c0 + c1 * c1 + c2 * c2)
    if n2 < 1e-12:
        raise ValueError("TRIAD directions are parallel; roll is undefined")
    b0, b1, b2 = c0 / n2, c1 / n2, c2 / n2
    out: F64 = np.array([[a0, b0, a1 * b2 - a2 * b1],
                         [a1, b1, a2 * b0 - a0 * b2],
                         [a2, b2, a0 * b1 - a1 * b0]])
    return out


def triad(b1: F64, i1: F64, b2: F64, i2: F64) -> F64:
    """Quaternion with R b1 = i1 exactly and R b2 as close as possible to i2."""
    return dcm_to_quat(_frame(i1, i2) @ _frame(b1, b2).T)


def _array_normal(view: ParamView) -> F64:
    return axis_from_label(view.params.bus(view.sat_id).wings.normal)


def _boresight(view: ParamView) -> F64:
    return axis_from_label(view.params.bus(view.sat_id).imager.face)


def _roll_reference(view: ParamView) -> F64:
    """Body axis the idle and take laws turn toward the orbit normal: the first body axis normal
    to the array normal (body +x for both buses)."""
    n = _array_normal(view)
    for k in range(3):
        e = np.eye(3)[k]
        if abs(e @ n) < 1e-12:
            out: F64 = e
            return out
    raise AssertionError("unreachable")


def sun_hold(view: ParamView, sun_eci: F64, orbit_normal_eci: F64) -> F64:
    """Idle law: array normal on the Sun, body +x as close as possible to the orbit normal."""
    return triad(_array_normal(view), sun_eci, _roll_reference(view), orbit_normal_eci)


def burn(view: ParamView, a_cmd_eci: F64, sun_eci: F64) -> F64:
    """Burn law: thrust axis b along a_cmd, then roll about it to bring the array normal closest
    to the Sun."""
    b = view.array("thruster.axis_b")
    return triad(b, a_cmd_eci, _array_normal(view), sun_eci)


def take(
    view: ParamView, r_eci: F64, target_ecef: F64, eci_to_ecef: F64, orbit_normal_eci: F64
) -> F64:
    """Take law: imager boresight on the target (ECEF, re-aimed at every evaluation using that
    instant's ECI->ECEF rotation); roll as in sun_hold."""
    los = eci_to_ecef.T @ target_ecef - r_eci
    return triad(_boresight(view), los, _roll_reference(view), orbit_normal_eci)
