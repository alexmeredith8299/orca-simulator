"""M3: the mission thread end to end (burns, uplink, takes, slews, gates)."""

import dataclasses
import warnings

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import AltAz, EarthLocation, get_body
from astropy.time import Time

from expand3.env.ground import pass_intervals
from expand3.mission.pointing import sun_hold
from expand3.mission.takes import off_nadir_deg
from expand3.mission.thread import MissionRun, run_mission
from expand3.params import Params, ParamView
from expand3.units import S_PER_HOUR
from expand3.vehicle.attitude import quat_angle_rad, quat_rotate
from expand3.vehicle.orbit import rv_to_elements

pytestmark = pytest.mark.slow
BURNERS = ["SAT-2", "SAT-3", "SAT-4"]
QUIET = ["SAT-1", "SAT-5", "SAT-6"]


def test_performance(mission72: MissionRun) -> None:
    print(f"\nsix-vehicle 72 h mission thread: {mission72.wall_s:.1f} s")
    assert mission72.wall_s < 60


def test_quiet_vehicles(mission72: MissionRun) -> None:
    for sat in QUIET:
        k = mission72.sats.index(sat)
        assert sat not in mission72.burns
        assert sat not in mission72.takes
        assert not mission72.gate_burn[:, k].any()
        assert not mission72.gate_take[:, k].any()
        assert not mission72.gate_slew[:, k].any()
        assert np.all(mission72.dv_m_s[:, k] == 0)


def test_uplink_and_burn(params: Params, mission72: MissionRun) -> None:
    tick = params.scenario.control_tick_s
    for sat in BURNERS:
        k = mission72.sats.index(sat)
        rec = mission72.burns[sat]
        visible = mission72.gate_pass[:, k, :].any(axis=1)
        passes = pass_intervals(mission72.t_s, visible)
        first = next(p for p in passes if p[1] >= rec.command_time_s)
        assert rec.uplink_pass == first
        assert rec.burn_start_s == first[1] + tick
        assert rec.burn_start_s > rec.command_time_s
        burning = mission72.t_s[mission72.gate_burn[:, k]]
        assert burning.min() == rec.burn_start_s
        assert rec.flown_s == pytest.approx(rec.length_s)
        assert rec.dv_m_s == pytest.approx(1.0, rel=1e-9)
        assert mission72.dv_m_s[-1, k] == pytest.approx(rec.dv_m_s)
        used = mission72.propellant_kg[0, k] - mission72.propellant_kg[-1, k]
        assert used == pytest.approx(rec.propellant_used_kg, rel=1e-12)
        view = ParamView(params, sat)
        dry = view.get("mass.wet_mass_kg") - view.get("thruster.propellant_kg")
        np.testing.assert_allclose(mission72.mass_kg[:, k], dry + mission72.propellant_kg[:, k])
        # ordered acceleration is prograde while burning, zero otherwise
        a = mission72.a_cmd_m_s2[:, k]
        on = mission72.gate_burn[:, k]
        assert np.all(a[~on] == 0)
        v_hat = mission72.v_km_s[on, k] / np.linalg.norm(mission72.v_km_s[on, k], axis=1)[:, None]
        np.testing.assert_allclose(a[on] / np.linalg.norm(a[on], axis=1)[:, None], v_hat,
                                   atol=1e-12)
        # during the burn the thrust axis is on the ordered acceleration
        b = view.array("thruster.axis_b")
        for n in np.flatnonzero(on)[::50]:
            axis = quat_rotate(mission72.q[n, k], b)
            ordered = a[n] / np.linalg.norm(a[n])
            assert np.arccos(np.clip(axis @ ordered, -1, 1)) < 1e-6


def test_takes(params: Params, mission72: MissionRun) -> None:
    sl = params.scenario.slew
    for sat in BURNERS:
        k = mission72.sats.index(sat)
        rec = mission72.takes[sat]
        assert rec.t_take_s >= params.scenario.take_window_start_h * S_PER_HOUR
        assert rec.t_end_s == rec.t_take_s + params.scenario.take_duration_s
        assert len(rec.tick_times_s) >= 2
        assert max(rec.off_nadir_deg) <= params.scenario.take_constraints.max_off_nadir_deg
        assert min(rec.sun_elevation_deg) >= 10
        assert max(rec.boresight_error_rad) <= 1e-6
        slew_in, slew_out = mission72.slews[sat][0], mission72.slews[sat][1]
        assert slew_in.converged
        assert slew_out.converged
        assert abs(rec.slew_in_start_s + slew_in.duration_s - rec.t_take_s) <= sl.step_s
        take_ticks = mission72.gate_take[:, k]
        np.testing.assert_allclose(mission72.t_s[take_ticks], rec.tick_times_s)
        # after the slew out the vehicle is back on the idle law
        after = np.flatnonzero(mission72.t_s > rec.slew_out_end_s)[0]
        assert not mission72.gate_slew[after, k]
        if not mission72.gate_burn[after, k]:
            h = np.cross(mission72.r_km[after, k], mission72.v_km_s[after, k])
            h_hat = h / np.linalg.norm(h)
            idle = sun_hold(ParamView(params, sat), mission72.sun_unit[after], h_hat)
            assert quat_angle_rad(mission72.q[after, k], idle) < 1e-9


def test_take_time_brute_force(params: Params, mission72: MissionRun) -> None:
    start = params.scenario.take_window_start_h * S_PER_HOUR
    for sat in BURNERS:
        k = mission72.sats.index(sat)
        rec = mission72.takes[sat]
        idx = np.flatnonzero((mission72.t_s >= start) & (mission72.t_s <= rec.t_take_s))
        r_ecef = np.einsum("nij,nj->ni", mission72.eci_to_ecef[idx], mission72.r_km[idx, k])
        loc = EarthLocation.from_geocentric(*(r_ecef.T * u.km))
        ground = EarthLocation.from_geodetic(loc.lon, loc.lat, 0 * u.m)
        times = Time(mission72.draws.epoch_utc, scale="utc") + mission72.t_s[idx] * u.s
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            frame = AltAz(obstime=times, location=ground)
            alt = get_body("sun", times).transform_to(frame).alt.deg
        assert alt[-1] >= 10 - 0.02
        assert np.all(alt[:-1] < 10 + 0.02)


def test_off_nadir_helper_matches(mission72: MissionRun) -> None:
    rec = mission72.takes["SAT-2"]
    k = mission72.sats.index("SAT-2")
    n = int(np.flatnonzero(mission72.t_s == rec.t_take_s)[0])
    target_eci = mission72.eci_to_ecef[n].T @ rec.target_ecef_km
    assert off_nadir_deg(mission72.r_km[n:n + 1, k], target_eci[None])[0] < 0.3


def test_burn_raises_orbit(params: Params, mission72: MissionRun) -> None:
    no_burn = run_mission(
        params, mission72.draws.seed, duration_s=72 * S_PER_HOUR,
        overrides={"SAT-2": {"scenario.station_keeping_dv_m_s": 0.0}},
    )
    k = mission72.sats.index("SAT-2")
    assert no_burn.burns["SAT-2"].dv_m_s == 0
    mu = params.scenario.constants.mu_km3_s2
    period = 2 * np.pi * np.sqrt(params.scenario.orbit.a_km**3 / mu)
    last = mission72.t_s >= mission72.t_s[-1] - period
    assert mission72.burns["SAT-2"].burn_end_s < mission72.t_s[last][0]
    a_burn = rv_to_elements(mission72.r_km[last, k], mission72.v_km_s[last, k], mu).a_km.mean()
    a_ref = rv_to_elements(no_burn.r_km[last, k], no_burn.v_km_s[last, k], mu).a_km.mean()
    a = params.scenario.orbit.a_km
    v = np.sqrt(mu / a)
    assert a_burn - a_ref == pytest.approx(2 * a * 1.0e-3 / v, rel=0.05)


def test_burn_take_conflict(params: Params, mission72: MissionRun) -> None:
    t_take = mission72.takes["SAT-2"].t_take_s
    cmd = dict(mission72.draws.burn_command_s)
    cmd["SAT-2"] = t_take - 3 * S_PER_HOUR
    draws = dataclasses.replace(mission72.draws, burn_command_s=cmd)
    run = run_mission(params, draws.seed, duration_s=72 * S_PER_HOUR, draws=draws)
    rec = run.burns["SAT-2"]
    assert rec.burn_start_s is not None
    assert rec.burn_start_s < t_take
    assert rec.suspended, "constructed case must overlap the take"
    assert run.takes["SAT-2"].t_take_s == t_take
    assert rec.flown_s == pytest.approx(rec.length_s)
    assert rec.dv_m_s == pytest.approx(1.0, rel=1e-9)
    gap = sum(b - a for a, b in rec.suspended)
    assert rec.burn_end_s == pytest.approx(rec.burn_start_s + rec.length_s + gap)
    k = run.sats.index("SAT-2")
    assert not (run.gate_burn[:, k] & (run.gate_take[:, k] | run.gate_slew[:, k])).any()


def test_determinism(params: Params) -> None:
    a = run_mission(params, 77, duration_s=36 * S_PER_HOUR)
    b = run_mission(params, 77, duration_s=36 * S_PER_HOUR)
    for sat in BURNERS:
        assert dataclasses.asdict(a.burns[sat]) == dataclasses.asdict(b.burns[sat])
        if sat in a.takes:
            assert a.takes[sat].t_take_s == b.takes[sat].t_take_s
    np.testing.assert_array_equal(a.r_km, b.r_km)
    np.testing.assert_array_equal(a.q, b.q)
