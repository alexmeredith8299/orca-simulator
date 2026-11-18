"""Translational dynamics: two-body, J2, exponential drag, thruster; fixed-step RK4 (M1).

Spec: Model: orbit and attitude. State per vehicle is r [km], v [km/s] in ECI (GCRS):

    r_dot = v
    v_dot = -mu r / |r|^3 + a_J2(r) - (1/2) rho(h) (Cd A / m) |v_rel| v_rel + a_thr
    v_rel = v - omega_E x r

J2 and Earth rotation act about Earth's rotation pole (ITRS +z expressed in ECI at the epoch),
not the GCRS z axis; the two differ by about 0.15 deg of precession since J2000.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from numpy.typing import NDArray

from expand3.env.atmosphere import density_kg_m3
from expand3.env.frames import eci_to_ecef, mean_sun_lon_ecef_rad
from expand3.params import Params, ParamView
from expand3.timebase import SeedDraws
from expand3.units import DEG_PER_HOUR_ANGLE, HOURS_PER_DAY, M_PER_KM
from expand3.vec3 import cross3

F64 = NDArray[np.float64]

_KEPLER_ITERATIONS = 12


@dataclass(frozen=True)
class Forces:
    """Switches for the perturbing forces (two-body is always on)."""

    j2: bool = True
    drag: bool = True


@dataclass(frozen=True)
class Elements:
    """Osculating classical elements; angles in rad, measured in an equatorial basis."""

    a_km: F64
    e: F64
    i_rad: F64
    raan_rad: F64
    argp_rad: F64
    nu_rad: F64
    u_rad: F64  # argument of latitude, argp + nu


@dataclass(frozen=True)
class OrbitModel:
    """Force model for K vehicles at once. ballistic_m2_kg is Cd A / m per vehicle."""

    mu_km3_s2: float
    J2: float
    R_E_km: float
    omega_E_rad_s: float
    pole: F64
    ballistic_m2_kg: F64
    rho0_kg_m3: float
    h0_km: float
    H_km: float
    forces: Forces = field(default_factory=Forces)
    cd_area_m2: F64 | None = None

    @classmethod
    def from_params(
        cls, params: Params, sats: list[str], draws: SeedDraws, pole: F64, forces: Forces
    ) -> OrbitModel:
        views = [ParamView(params, s) for s in sats]
        v0 = views[0]
        return cls(
            mu_km3_s2=v0.get("scenario.constants.mu_km3_s2"),
            J2=v0.get("scenario.constants.J2"),
            R_E_km=v0.get("scenario.constants.R_E_km"),
            omega_E_rad_s=v0.get("scenario.constants.omega_E_rad_s"),
            pole=np.asarray(pole, dtype=np.float64),
            ballistic_m2_kg=np.array(
                [
                    v.get("drag.Cd") * v.get("drag.area_m2") / v.get("mass.wet_mass_kg")
                    for v in views
                ]
            ),
            rho0_kg_m3=draws.rho0_kg_m3,
            h0_km=v0.get("scenario.drag.h0_km"),
            H_km=draws.H_km,
            forces=forces,
            cd_area_m2=np.array([v.get("drag.Cd") * v.get("drag.area_m2") for v in views]),
        )

    def accel(
        self, r: F64, v: F64, a_thr: F64 | None = None, mass_kg: F64 | None = None
    ) -> F64:
        """Spec: Model: orbit and attitude, translational dynamics. r, v are (K, 3); a_thr is
        (K, 3) in km/s^2; mass_kg (K,) is the current mass for drag (default: wet mass)."""
        rn = np.sqrt(np.einsum("ki,ki->k", r, r))[:, None]
        acc: F64 = -self.mu_km3_s2 * r / rn**3
        if self.forces.j2:
            # a_J2 = -(3/2) J2 mu R_E^2 / |r|^5 [(1 - 5 z^2/|r|^2) r + 2 z p], z = r . p
            z = (r @ self.pole)[:, None]
            k = -1.5 * self.J2 * self.mu_km3_s2 * self.R_E_km**2 / rn**5
            acc = acc + k * ((1.0 - 5.0 * z**2 / rn**2) * r + 2.0 * z * self.pole)
        if self.forces.drag:
            v_rel = v - self.omega_E_rad_s * cross3(self.pole, r)
            speed = np.sqrt(np.einsum("ki,ki->k", v_rel, v_rel))[:, None]
            rho = density_kg_m3(rn - self.R_E_km, self.rho0_kg_m3, self.h0_km, self.H_km)
            # rho [kg/m^3] * B [m^2/kg] is 1/m; x M_PER_KM gives 1/km, so a is in km/s^2.
            ballistic = (
                self.ballistic_m2_kg
                if mass_kg is None or self.cd_area_m2 is None
                else self.cd_area_m2 / mass_kg
            )
            acc = acc - 0.5 * rho * ballistic[:, None] * M_PER_KM * speed * v_rel
        if a_thr is not None:
            acc = acc + a_thr
        return acc

    def deriv(self, y: F64, a_thr: F64 | None = None, mass_kg: F64 | None = None) -> F64:
        """y is (K, 6) = [r, v]; returns [v, a]."""
        out: F64 = np.concatenate(
            [y[:, 3:], self.accel(y[:, :3], y[:, 3:], a_thr, mass_kg)], axis=1
        )
        return out


def rk4_step(f: Callable[[F64], F64], y: F64, dt: float) -> F64:
    """Classical fixed-step fourth-order Runge-Kutta. Spec: Model: orbit and attitude."""
    k1 = f(y)
    k2 = f(y + 0.5 * dt * k1)
    k3 = f(y + 0.5 * dt * k2)
    k4 = f(y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def propagate(
    model: OrbitModel, y0: F64, step_s: float, n_steps: int, record_every: int
) -> F64:
    """RK4 from y0 (K, 6) for n_steps; returns states (n_steps // record_every + 1, K, 6)
    recorded at step 0 and every record_every steps. Thrust is zero until M3."""
    n_rec = n_steps // record_every + 1
    out = np.empty((n_rec, *y0.shape))
    y = np.array(y0, dtype=np.float64)
    out[0] = y
    f = model.deriv
    for j in range(1, n_rec):
        for _ in range(record_every):
            y = rk4_step(f, y, step_s)
        out[j] = y
    return out


def _rot3(angle: float) -> F64:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot1(angle: float) -> F64:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def elements_to_rv(
    a_km: float, e: float, i_rad: float, raan_rad: float, argp_rad: float, nu_rad: float,
    mu_km3_s2: float, basis: F64 | None = None,
) -> tuple[F64, F64]:
    """Classical elements to ECI r [km], v [km/s]. `basis` (3x3, columns = equatorial x, y, z
    axes in ECI) defines the reference plane; default is the ECI axes."""
    p = a_km * (1.0 - e**2)
    rr = p / (1.0 + e * np.cos(nu_rad))
    r_pf = rr * np.array([np.cos(nu_rad), np.sin(nu_rad), 0.0])
    v_pf = np.sqrt(mu_km3_s2 / p) * np.array([-np.sin(nu_rad), e + np.cos(nu_rad), 0.0])
    q = _rot3(raan_rad) @ _rot1(i_rad) @ _rot3(argp_rad)
    b = np.eye(3) if basis is None else basis
    return b @ q @ r_pf, b @ q @ v_pf


def rv_to_elements(r: F64, v: F64, mu_km3_s2: float, basis: F64 | None = None) -> Elements:
    """Osculating elements of r (..., 3), v (..., 3), angles measured in `basis`."""
    b = np.eye(3) if basis is None else basis
    r = np.asarray(r) @ b
    v = np.asarray(v) @ b
    rn = np.linalg.norm(r, axis=-1)
    h = np.cross(r, v)
    hn = np.linalg.norm(h, axis=-1)
    node = np.stack([-h[..., 1], h[..., 0], np.zeros_like(hn)], axis=-1)
    nn = np.linalg.norm(node, axis=-1)
    n_hat = node / nn[..., None]
    e_vec = np.cross(v, h) / mu_km3_s2 - r / rn[..., None]
    a = 1.0 / (2.0 / rn - np.sum(v * v, axis=-1) / mu_km3_s2)
    m_hat = np.cross(h / hn[..., None], n_hat)  # in-plane, 90 deg ahead of the node
    u = np.arctan2(np.sum(r * m_hat, axis=-1), np.sum(r * n_hat, axis=-1))
    argp = np.arctan2(np.sum(e_vec * m_hat, axis=-1), np.sum(e_vec * n_hat, axis=-1))
    return Elements(
        a_km=a,
        e=np.linalg.norm(e_vec, axis=-1),
        i_rad=np.arccos(np.clip(h[..., 2] / hn, -1.0, 1.0)),
        raan_rad=np.arctan2(h[..., 0], -h[..., 1]),
        argp_rad=argp,
        nu_rad=np.mod(u - argp, 2 * np.pi),
        u_rad=u,
    )


def kepler_propagate(r0: F64, v0: F64, dt_s: F64, mu_km3_s2: float) -> tuple[F64, F64]:
    """Analytic two-body propagation of one elliptic state to times dt_s (M,). Reference only."""
    el = rv_to_elements(r0, v0, mu_km3_s2)
    a, e = float(el.a_km), float(el.e)
    nu0 = float(el.nu_rad)
    e_anom0 = 2.0 * np.arctan2(np.sqrt(1 - e) * np.sin(nu0 / 2), np.sqrt(1 + e) * np.cos(nu0 / 2))
    m = e_anom0 - e * np.sin(e_anom0) + np.sqrt(mu_km3_s2 / a**3) * np.asarray(dt_s)
    ea = np.array(m, dtype=np.float64)
    for _ in range(_KEPLER_ITERATIONS):
        ea = ea - (ea - e * np.sin(ea) - m) / (1.0 - e * np.cos(ea))
    nu = 2.0 * np.arctan2(np.sqrt(1 + e) * np.sin(ea / 2), np.sqrt(1 - e) * np.cos(ea / 2))
    rs, vs = zip(
        *(
            elements_to_rv(
                a, e, float(el.i_rad), float(el.raan_rad), float(el.argp_rad), float(n), mu_km3_s2
            )
            for n in np.atleast_1d(nu)
        ),
        strict=True,
    )
    return np.array(rs), np.array(vs)


def equatorial_basis(pole: F64, x_hint: F64) -> F64:
    """3x3 basis (columns x, y, z in ECI) with z = pole and x along x_hint projected onto the
    equator."""
    z = pole / np.linalg.norm(pole)
    x = x_hint - (x_hint @ z) * z
    x = x / np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def initial_states(params: Params, sats: list[str], epoch_utc: datetime) -> F64:
    """(K, 6) initial ECI states. Spec: Parameters > Scenario and environment (orbit, true anomaly
    at epoch, LTAN 22:30).

    The ascending node is placed at local mean time LTAN: its ECEF longitude is the mean Sun's
    plus (LTAN - 12 h) x 15 deg/h, i.e. RAAN = RA of the mean Sun + (LTAN - 12 h) x 15 deg/h,
    measured in Earth's true equator at the epoch. Inclination is about Earth's pole.
    """
    o = params.scenario.orbit
    mu = params.scenario.constants.mu_km3_s2
    m = eci_to_ecef(epoch_utc, np.zeros(1))[0]
    lon_ms = float(mean_sun_lon_ecef_rad(epoch_utc, np.zeros(1))[0])
    lon_node = lon_ms + np.deg2rad((o.ltan_hours - HOURS_PER_DAY / 2) * DEG_PER_HOUR_ANGLE)
    node_eci = m.T @ np.array([np.cos(lon_node), np.sin(lon_node), 0.0])
    basis = equatorial_basis(m[2], node_eci)
    out = np.empty((len(sats), 6))
    for k, sat in enumerate(sats):
        nu = np.deg2rad(params.scenario.true_anomaly_at_epoch_deg[sat])
        r, v = elements_to_rv(
            o.a_km, o.e, np.deg2rad(o.i_deg), 0.0, np.deg2rad(o.arg_perigee_deg), nu, mu, basis
        )
        out[k] = np.concatenate([r, v])
    return out
