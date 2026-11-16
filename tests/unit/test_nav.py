"""Onboard navigation filter: an EKF on the GNSS fixes with J2 and drag in its dynamics.
Not in the spec (docs/extended-faults-plan.md, Phase A): the filter is a vehicle model, its output
a measured signal beside the GNSS fix. It sees the fixes and nothing else."""

import inspect
from dataclasses import replace

import numpy as np

from expand3.params import Params, ParamView
from expand3.timebase import SeedDraws, draw_seed, rng
from expand3.vehicle.nav import nav_model, navigate
from expand3.vehicle.orbit import Forces, OrbitModel, initial_states, propagate

DT_FIX = 60.0
N_FIX = 720  # 12 h of fixes
POLE = np.array([0.0, 0.0, 1.0])


def _truth(params: Params, draws: SeedDraws, sat: str, n: int) -> tuple[np.ndarray, np.ndarray]:
    """True states every 60 s from the full force model (the truth draws of density)."""
    model = OrbitModel.from_params(params, [sat], draws, POLE, Forces())
    y0 = initial_states(params, [sat], draws.epoch_utc)[:1]
    step = 10.0
    y = propagate(model, y0, step, int(DT_FIX / step) * (n - 1), int(DT_FIX / step))[:, 0]
    return y[:, :3], y[:, 3:]


def _fixes(view: ParamView, r: np.ndarray, v: np.ndarray, seed: int
           ) -> tuple[np.ndarray, np.ndarray]:
    g = rng(seed, "test.nav.fixes")
    sp = view.get("signals.gnss.pos_sigma_m") / 1000.0
    sv = view.get("signals.gnss.vel_sigma_m_s") / 1000.0
    return r + g.normal(0.0, sp, r.shape), v + g.normal(0.0, sv, v.shape)


def _setup(params: Params, sat: str = "SAT-2"
           ) -> tuple[ParamView, np.ndarray, np.ndarray, np.ndarray]:
    view = ParamView(params, sat)
    draws = draw_seed(params, 11)
    r, v = _truth(params, draws, sat, N_FIX)
    t = np.arange(N_FIX) * DT_FIX
    return view, t, r, v


def test_filter_beats_the_raw_fix(params: Params) -> None:
    """The filter averages the fixes against J2 and drag dynamics: its error is well under the
    fix noise (1.5 m a position axis, 5 cm/s a velocity axis)."""
    view, t, r, v = _setup(params)
    zr, zv = _fixes(view, r, v, 1)
    nav = navigate(view, nav_model(params, "SAT-2", POLE), t, zr, zv)
    settled = slice(60, None)
    pos_err = (nav.r_km - r)[settled] * 1000.0
    vel_err = (nav.v_km_s - v)[settled] * 1000.0
    raw_pos = (zr - r)[settled] * 1000.0
    assert np.sqrt((pos_err**2).mean()) < 0.8 * np.sqrt((raw_pos**2).mean())
    assert np.sqrt((vel_err**2).mean()) < 0.05
    assert abs(pos_err.mean()) < 0.2  # unbiased


def test_filter_error_is_consistent_with_its_noise_model(params: Params) -> None:
    """With nothing wrong, the fix minus the nav solution has about the spread the filter
    expects, so the detector's nominal model is a fair one."""
    view, t, r, v = _setup(params)
    zr, zv = _fixes(view, r, v, 2)
    nav = navigate(view, nav_model(params, "SAT-2", POLE), t, zr, zv)
    resid = (zr - nav.r_km)[60:] * 1000.0
    assert 0.5 < resid.std() / 1.5 < 1.1


def test_filter_follows_a_position_offset_in_the_fixes(params: Params) -> None:
    """A filter fed a 10 m offset absorbs it: the nav solution ends up on the offset fixes, not
    the truth, and the transient shows between fix and nav solution."""
    view, t, r, v = _setup(params)
    zr, zv = _fixes(view, r, v, 3)
    arm = 400
    off = zr.copy()
    off[arm:] += 10.0 / 1000.0
    nav = navigate(view, nav_model(params, "SAT-2", POLE), t, off, zv)
    err_after = (nav.r_km - r)[arm + 60:] * 1000.0
    assert np.allclose(err_after.mean(axis=0), 10.0, atol=2.5)
    jump = np.linalg.norm((off - nav.r_km)[arm:arm + 3] * 1000.0, axis=1)
    quiet = np.linalg.norm((off - nav.r_km)[arm - 60:arm] * 1000.0, axis=1)
    assert jump.max() > 3.0 * np.median(quiet)


def test_filter_uses_only_the_fixes_and_the_model(params: Params) -> None:
    """No truth state goes in: the signature takes the fixes and the model, nothing else."""
    sig = inspect.signature(navigate)
    assert list(sig.parameters) == ["view", "model", "t_s", "r_fix_km", "v_fix_km_s"]


def test_filter_models_j2_and_drag_with_the_nominal_atmosphere(params: Params) -> None:
    """The onboard force model has J2 and drag, and the atmosphere is the middle of the draw
    (it does not know the seed's density)."""
    model = nav_model(params, "SAT-2", POLE)
    assert model.forces == Forces(j2=True, drag=True)
    d = params.scenario.drag
    rho, scale = d.rho0_kg_m3, d.H_km
    assert rho.low is not None
    assert rho.high is not None
    assert scale.low is not None
    assert scale.high is not None
    assert np.isclose(model.rho0_kg_m3, np.sqrt(rho.low * rho.high))
    assert np.isclose(model.H_km, 0.5 * (scale.low + scale.high))
    assert replace(model, forces=Forces(j2=False, drag=False)).forces.j2 is False


def test_filter_is_deterministic(params: Params) -> None:
    view, t, r, v = _setup(params)
    zr, zv = _fixes(view, r, v, 4)
    model = nav_model(params, "SAT-2", POLE)
    a = navigate(view, model, t, zr, zv)
    b = navigate(view, model, t, zr, zv)
    np.testing.assert_array_equal(a.r_km, b.r_km)
