"""Onboard navigation filter: an EKF on the GNSS fixes with J2 and drag in its dynamics.
Not in the spec (docs/extended-faults-plan.md, Phase A): the filter is a vehicle model, its output
a measured signal beside the GNSS fix. It sees the fixes and nothing else."""

import inspect
from dataclasses import replace

import numpy as np

from expand3.params import Params, ParamView
from expand3.timebase import SeedDraws, draw_seed, rng
from expand3.vehicle.nav import nav_model, navigate
from expand3.vehicle.orbit import Forces, OrbitModel, initial_states, rk4_step

DT_FIX = 60.0
N_FIX = 720  # 12 h of fixes
POLE = np.array([0.0, 0.0, 1.0])


def _truth(params: Params, draws: SeedDraws, sat: str, n: int, thrust_m_s2: float = 0.0,
           burn_h: tuple[float, float] = (0.0, 0.0), rho_mult: float = 1.0
           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """True states every 60 s from the full force model (the truth draws of density, times
    `rho_mult`) with a prograde burn between the hours in `burn_h`; also the thrust ordered at
    each 30 s tick, [m/s^2]."""
    model = OrbitModel.from_params(params, [sat], draws, POLE, Forces())
    model = replace(model, rho0_kg_m3=model.rho0_kg_m3 * rho_mult)
    y = initial_states(params, [sat], draws.epoch_utc)[:1]
    step = 10.0
    per = int(DT_FIX / step)
    states, cmd = [y[0].copy()], []
    for i in range((n - 1) * per):
        t = i * step
        on = burn_h[0] * 3600.0 <= t < burn_h[1] * 3600.0
        a = thrust_m_s2 * y[:, 3:] / np.linalg.norm(y[:, 3:]) if on else np.zeros((1, 3))
        if i % 3 == 0:
            cmd.append(a[0].copy())
        y = rk4_step(lambda q, a=a: model.deriv(q, a / 1000.0), y, step)
        if (i + 1) % per == 0:
            states.append(y[0].copy())
    out = np.array(states)
    return out[:, :3], out[:, 3:], np.array(cmd)


def _fixes(view: ParamView, r: np.ndarray, v: np.ndarray, seed: int
           ) -> tuple[np.ndarray, np.ndarray]:
    g = rng(seed, "test.nav.fixes")
    sp = view.get("signals.gnss.pos_sigma_m") / 1000.0
    sv = view.get("signals.gnss.vel_sigma_m_s") / 1000.0
    return r + g.normal(0.0, sp, r.shape), v + g.normal(0.0, sv, v.shape)


def _setup(params: Params, sat: str = "SAT-2", **kw: float | tuple[float, float]):  # type: ignore[no-untyped-def]
    view = ParamView(params, sat)
    draws = draw_seed(params, 11)
    r, v, cmd = _truth(params, draws, sat, N_FIX, **kw)  # type: ignore[arg-type]
    t = np.arange(N_FIX) * DT_FIX
    return view, t, r, v, cmd


def _run(params: Params, view: ParamView, t: np.ndarray, zr: np.ndarray, zv: np.ndarray,
         cmd: np.ndarray, use_cmd: bool = True):  # type: ignore[no-untyped-def]
    t_cmd = np.arange(len(cmd)) * 30.0
    return navigate(view, nav_model(params, "SAT-2", POLE), t, zr, zv, t_cmd,
                    cmd if use_cmd else np.zeros_like(cmd))


def test_filter_beats_the_raw_fix(params: Params) -> None:
    """The filter averages the fixes against J2 and drag dynamics: its error is well under the
    fix noise (1.5 m a position axis, 5 cm/s a velocity axis)."""
    view, t, r, v, cmd = _setup(params)
    zr, zv = _fixes(view, r, v, 1)
    nav = _run(params, view, t, zr, zv, cmd)
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
    view, t, r, v, cmd = _setup(params)
    zr, zv = _fixes(view, r, v, 2)
    nav = _run(params, view, t, zr, zv, cmd)
    resid = (zr - nav.r_km)[60:] * 1000.0
    assert 0.5 < resid.std() / 1.5 < 1.1


def test_filter_follows_a_position_offset_in_the_fixes(params: Params) -> None:
    """A filter fed a 10 m offset absorbs it: the nav solution ends up on the offset fixes, not
    the truth, and the transient shows between fix and nav solution."""
    view, t, r, v, cmd = _setup(params)
    zr, zv = _fixes(view, r, v, 3)
    arm = 400
    off = zr.copy()
    off[arm:] += 10.0 / 1000.0
    nav = _run(params, view, t, off, zv, cmd)
    err_after = (nav.r_km - r)[arm + 60:] * 1000.0
    assert np.allclose(err_after.mean(axis=0), 10.0, atol=2.5)
    jump = np.linalg.norm((off - nav.r_km)[arm:arm + 3] * 1000.0, axis=1)
    quiet = np.linalg.norm((off - nav.r_km)[arm - 60:arm] * 1000.0, axis=1)
    assert jump.max() > 3.0 * np.median(quiet)


def test_filter_uses_only_the_fixes_and_the_model(params: Params) -> None:
    """No truth state goes in: the signature takes the fixes, the ordered thrust and the model."""
    sig = inspect.signature(navigate)
    assert list(sig.parameters) == ["view", "model", "t_s", "r_fix_km", "v_fix_km_s", "t_cmd_s",
                                    "a_cmd_m_s2"]


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
    view, t, r, v, cmd = _setup(params)
    zr, zv = _fixes(view, r, v, 4)
    a = _run(params, view, t, zr, zv, cmd)
    b = _run(params, view, t, zr, zv, cmd)
    np.testing.assert_array_equal(a.r_km, b.r_km)


def test_ordered_thrust_keeps_the_solution_on_the_orbit_through_a_burn(params: Params) -> None:
    """A 5e-5 m/s^2 burn (about a station-keeping burn) is a large unmodelled force. Fed the
    ordered acceleration, the filter tracks it; without, its error during the burn is larger."""
    view, t, r, v, cmd = _setup(params, thrust_m_s2=5.0e-5, burn_h=(3.0, 9.0))
    zr, zv = _fixes(view, r, v, 5)
    during = slice(4 * 60, 9 * 60)
    with_cmd = _run(params, view, t, zr, zv, cmd)
    without = _run(params, view, t, zr, zv, cmd, use_cmd=False)
    e_with = np.linalg.norm((with_cmd.r_km - r)[during], axis=1).mean() * 1000.0
    e_without = np.linalg.norm((without.r_km - r)[during], axis=1).mean() * 1000.0
    assert e_with < 0.7 * e_without
    assert e_with < 2.0 * 1.5 * np.sqrt(3)  # near the fix noise


def test_drag_scale_is_learned_from_the_fixes(params: Params) -> None:
    """With the true density far above the filter's, the extra state moves toward it. Drag at
    this altitude is about 1e-10 m/s^2, so 12 h of 1.5 m fixes pin it down only roughly."""
    view, t, r, v, cmd = _setup(params, rho_mult=100.0)
    zr, zv = _fixes(view, r, v, 6)
    nav = _run(params, view, t, zr, zv, cmd)
    true_scale = 100.0 * draw_seed(params, 11).rho0_kg_m3 / nav_model(
        params, "SAT-2", POLE).rho0_kg_m3
    assert nav.drag_scale[-1] > 3.0  # from 1 toward the truth, not past it
    assert nav.drag_scale[-1] < 1.5 * true_scale


def test_drag_scale_stays_near_one_when_the_density_is_the_nominal_one(params: Params) -> None:
    """Whatever the draw, the filter's scale does not run away: the seed's density is within a
    factor of about 3 of the nominal one."""
    view, t, r, v, cmd = _setup(params)
    zr, zv = _fixes(view, r, v, 7)
    nav = _run(params, view, t, zr, zv, cmd)
    assert nav.drag_scale.min() >= 0.0
    assert nav.drag_scale.max() < 20.0
