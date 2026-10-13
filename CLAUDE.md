# CLAUDE.md

EXPAND.3 Phase I simulator for the Orca SBIR: six SmallSats (SAT-1 to SAT-4 on Bus A, SAT-5 and
SAT-6 on Bus B) with orbit, attitude, power, thermal, sensors, links and a fault injector (F1 to F10).
It produces truth and telemetry only.

- Plan: https://claude.ai/code/artifact/bcb26c94-5084-4482-b003-0b7478912903
- Spec: `docs/spec/` (PDF added by Alex). `params/*.yaml` transcribes its Parameters, Signals and
  Faults tables and cites the section in a header comment.

## Environment

```bash
conda activate orca
pip install -e '.[dev]'
```

Python 3.12. Before every commit, run all three and keep them clean:

```bash
ruff check .
mypy src
pytest
```

## Layout

- `params/`: scenario, bus_a, bus_b (diff onto bus_a), faults, links_saa_imager.
- `src/expand3/params.py`: pydantic models, `load_params(dir)`, `Params.bus(sat_id)` and `ParamView`.
- `src/expand3/timebase.py`: `rng(seed, name)` named substreams, `draw_seed` gives `SeedDraws`,
  and `Clock`.
- `src/expand3/env/`: `frames` (GCRS to ITRS, pole, mean Sun), `sun`, `eclipse`, `atmosphere` and
  `ground`.
- `src/expand3/vehicle/orbit.py`: force model, RK4, element conversions and initial states.
- `src/expand3/vehicle/attitude.py` (quaternions, kinematics, rigid body) and
  `vehicle/wheels.py` (`WheelArray`).
- `src/expand3/mission/pointing.py` (Sun hold, burn and take laws) and `mission/slew.py` (plan
  plus PD slews).
- `src/expand3/vehicle/thruster.py` (`thrust_step`), `mission/burns.py` (burn plan and
  `BurnController`) and `mission/takes.py` (take rule, target, constraints).
- `src/expand3/vehicle/eps.py`: `Eps.step` per control tick (strings, loads, pack, voltage,
  latches, node heat) and the per-tick `EpsHistory`.
- `src/expand3/vehicle/thermal.py`: `Network` (node equation, Euler step, stability limit),
  `Thermal` (topology from params) and `step_power_and_thermal` (the per-tick coupling).
- `src/expand3/mission/thread.py`: `run_mission(params, seed)` runs all six vehicles end to end
  and returns per-tick truth: r, v, q, omega, wheel h, mass, gates, the burn log, the take
  records, `eps[sat]` and `thermal[sat]`.
- `src/expand3/constellation.py`: `propagate_constellation` gives r and v at ticks, plus the Sun,
  the shadow fraction and station visibility.
- `src/expand3/units.py`: definitional unit conversions only.
- Stubs for later milestones: `faults.py`, `sensors.py`, `links.py`, `recorder.py` and
  `run.py`.
- `tests/unit`, `tests/invariants` (conservation and bounds), `tests/reference` (external
  references such as IGRF), `tests/faults` and `tests/regression` (seeded golden runs).

## Conventions

- **Units:** SI internally. The one exception is orbit state, which is in km and km/s as the spec
  states. Field names carry their unit suffix (`_km`, `_J`, `_W_K`, `_per_s`).
- **Quaternions:** scalar first `[w, x, y, z]`, rotating body to inertial.
- **No numeric literals in model code.** Every number comes from `params/` through `ParamView`
  (`view.get("eps.R_pack_ohm")`, `view.array("wheels.k")`). `ParamView.overrides`, keyed by
  the dotted paths in `faults.yaml`, is how faults change parameters, so code that reads
  `Params` directly cannot see faults. Structural constants such as 0, 1 and array indices are
  fine; physical or tuning values are not.
- **Cite the spec.** Each function that implements a spec equation names the spec section in its
  docstring.
- **Randomness** comes only from named `numpy.random.SeedSequence` substreams derived from the run
  seed, for example `noise.gnss.SAT-2` and `draw.fault`. Never use global or unnamed RNGs. Adding a
  stream must not shift the draws of the others.
- **No detection or estimation logic** goes in this repo. It simulates truth plus the sensors and
  telemetry the spec defines.
- **Exit tests come first.** Write each milestone's exit tests before its code.

## Milestones

| M | Scope |
|---|---|
| M0 | Scaffold: params, loader, ParamView, tests, CI |
| M1 | Environment and orbit |
| M2 | Attitude and wheels |
| M3 | Thruster and mission thread |
| M4 | Power |
| M5 | Thermal |
| M6 | Sensors, links, SAA and imager |
| M7 | Fault injector |
| M8 | Batch runs and Parquet output |

## Known gaps

M1 choices where the spec is silent (the default is in force until Alex says otherwise):

- **Constants not in the spec.** `scenario.yaml` constants add omega_E = 7.2921159e-5 rad/s (IERS
  2010) and R_sun = 695700 km (IAU 2015 B3, used by the dual-cone shadow).
- **Frames.** ECI is GCRS and ECEF is ITRS (ERFA `c2t06a`, the same transform as astropy's
  GCRS to ITRS). IERS data is used offline from the bundled tables, never downloaded. Future
  epochs use astropy's predicted UT1 and polar motion, a sub-second UT1 error.
- **J2 and Earth rotation** act about Earth's pole (ITRS +z in ECI at the epoch), held fixed for
  the run. This pole differs from GCRS z by about 0.15 deg of precession.
- **LTAN** is mean local time: RAAN = RA of the mean Sun + (LTAN - 12 h) x 15 deg/h, with the mean
  Sun's Greenwich hour angle (UT1 - 12 h) x 15 deg/h. RAAN and inclination are measured in Earth's
  equator at the epoch.
- **Sun** is the geocentric apparent position (astropy `get_body`). The shadow fraction uses a
  spherical Earth, flat-disk overlap and no atmosphere.
- **Stations** sit on WGS84 at zero height. Elevation is geometric (no refraction) from the
  geodetic horizontal. Altitude for drag is the spherical |r| - R_E, as the spec states.
- **Mass** is the wet mass in M1. Propellant drain arrives with the thruster in M3.
- **Two-body energy tolerance.** RK4 at the spec's 10 s step drifts about 1.3e-9 in energy over
  72 h (truncation, scaling as dt^5). The invariant test uses 2e-9 for energy and 1e-9 for angular
  momentum, and also checks the step scaling.
- **Run length and ticks.** A run length that is not a multiple of the 30 s tick ends at the last
  whole tick.
- **Products at ticks.** States, frames, Sun and shadow are produced at control ticks only, so
  there is no 10 s interpolation yet. Revisit if M2 or M3 need substep environment values.

- PyYAML follows YAML 1.1, which reads `3.6e6` as a string. `params.py` registers a YAML 1.2 float
  resolver, so keep using `load_params`.
- The distribution of the station-keeping burn length is not specified; see `scenario.yaml`.

M2 choices where the spec is silent:

- **Wheel azimuths** are 45, 135, 225 and 315 deg about +z from +x (`wheels.azimuth_deg`). The
  spec gives only the 54.7 deg cant.
- **Wheel momentum update.** The spec's first-order discrete update is used deliberately (Alex's
  decision): h(t + dt) = h(t) - (k u + c h(t)) dt. Under F3 friction alone h_n = h_0 (1 - c dt)^n,
  about 1 % off exp(-c t) after 50 s at the 1 s slew step. Saturation clamps |h| at h_max, and the
  body torque is the matching -A dh/dt, which equals A (k o u + c o h(t)) when unsaturated.
- **Wheel power** is charged per wheel whenever |u_k| > 0, including a saturated wheel that is
  still commanded.
- **Rigid-body integration** uses the momentum form: inertial total momentum
  H = R(q)(I omega + A h) is carried and omega = I^-1 (R^T H - A h). This is equivalent to the
  spec's equation, and torque-free H is conserved to roundoff. The 1 s step is within 1e-5 rad of
  a fine step over 30 s of slewing (fourth order).
- **Gravity gradient** is implemented but off in slews, which the spec allows. It switches on
  when `step` is given a position.
- **Pointing roll conventions.** Idle puts body +x as close as possible to the orbit normal. A
  burn rolls about the thrust axis to bring +z closest to the Sun (closed form via TRIAD). A take
  uses the idle roll, because rolling about the -z boresight cannot change +z.
- **Slew plan.** The plan is a bang-coast-bang eigen-axis rotation. Its acceleration uses
  `slew.torque_fraction` = 0.8 of the unclipped torque the array delivers in the required
  direction I e, rather than a minimum over body axes, so each slew uses its own axis capability.
  The coast rate uses 0.8 of the wheel momentum capacity in that direction. Phases are rounded up
  to whole 1 s steps, so the zero-order-hold feedforward is exact and the slew lands at the
  planned end. That adds at most 3 steps over the continuous bang-coast-bang prediction. The
  10 % duration check is applied to slews predicted at 20 s or more, and every slew must finish
  within the prediction plus 3 steps.
- **PD trim.** w_n = 0.3 rad/s and zeta = 0.9, so K_p = I w_n^2 and K_d = 2 zeta w_n I, with
  gyroscopic feedforward. A slew is done when the error is below 0.1 deg and the rate below
  1e-4 rad/s, and gives up 300 s after the planned end.
- **Slew step** is 1 s (`slew.step_s`), inside the 30 s control tick.

M3 choices where the spec is silent (new values are in `scenario.yaml`, marked "default, spec
silent"):

- **Burn length** is drawn once per seed, U(5, 7) h (`station_keeping_burn_length_draw_h`), and
  rounded to whole 10 s integration steps. All burn timing sits on that step grid.
- **Ordered acceleration** is prograde, re-aimed along v at each 10 s step (zero-order hold).
  Its constant magnitude is sized so the delivered delta-v is `station_keeping_dv_m_s` = 1.0 m/s
  at the nominal (loaded) eta_F. Under F5 or F6 the delivered delta-v falls to 0.7 m/s.
- **Uplink** happens in the first pass (any station) with a visible tick at or after the
  command time, so a command issued during a pass uplinks in that pass. The burn starts at the
  first tick after that pass ends (loss of signal).
- **Mass** is dry mass (wet minus propellant) plus the propellant remaining. Drag and thrust use
  the mass at each step's start, and thrust stops when the tank is empty.
- **Take time** is the first tick at or after 30 h where the Sun is at least 10 deg above the
  horizon at the geodetic sub-satellite point. The target is that WGS84 point, fixed in ECEF, so
  off-nadir is near 0 at take start; it is measured from geocentric nadir. The take lasts
  `take_duration_s` = 60 s, with the take law re-aimed at every tick. At 60 s off-nadir reaches
  about 38 deg, under the 45 deg limit.
- **Slews around a take.** The slew in is a rest-to-rest slew from the attitude of the mode in
  force at its start (idle, or burn if burning) to the take attitude at take start. Its start is
  iterated so its planned duration ends exactly at take start. The slew out goes from the take
  attitude at take end to the idle attitude at that time. Pointing is ideal during the take
  itself.
- **Burn and take conflict.** A burn is suspended over [slew-in start, slew-out end], rounded
  outward to the 10 s grid, and resumes afterwards; its end is extended so the full length is
  flown. The run integrates again when a suspension is needed and replans the take from the new
  trajectory.
- **Between slews.** The body rate is the finite-difference rate of the ideal attitude from tick
  to tick, and wheel momentum is held constant (wheels are modeled only during slews). Switches
  between idle and burn attitude are instantaneous.
- **A zero-magnitude burn** (for example a delta-v override of 0) runs the timeline but orders no
  thrust and keeps the idle attitude.

M4 choices where the spec is silent:

- **Placeholder temperatures** (`scenario.placeholder_node_temperature_K`) are now used only by
  the EPS unit tests. `run_mission` feeds EPS the M5 network's temperatures.
- **Initial state.** SoC is 0.8 (`scenario.eps_initial_soc`) and every latch is off.
- **Strings.** String k sits on wing k (in `wings.names` order) and faces the wing normal (+z).
  Its temperature is that wing node's.
- **Tick stepping.** E advances explicitly over the 30 s tick, and V and I use the SoC at the
  tick start. Tables interpolate linearly and clamp at their ends.
- **Latches.** Undervoltage hold, cold-charge inhibit and heater states are evaluated on a tick's
  values and applied from the next tick. Undervoltage uses the true terminal voltage, and the
  heater thermostats use true node temperatures. The cold-charge inhibit takes a sensed battery
  temperature when given (M6 supplies the thermistor value) and otherwise uses the true one.
- **Gates.** The take load applies to ticks with t_take <= t < t_end. Wheel torquing power is the
  mean over the tick of the M2 slew power (10 or 8 W per torquing wheel). The edge computer's SAA
  gate is off until M6.
- **Unmet demand and curtailment.** Unmet demand (N- minus P_sup) and curtailed array power
  (N+ minus P_acc) are recorded as truth; loads are not browned out except through the
  undervoltage reflex.
- **Voltage discriminant.** A negative OCV^2 + 4 P_pack R is clamped to 0 and flagged per tick
  (`disc_clamped`). It never happens in a nominal run.
- **Converter heat.** The array converter loss uses the accepted array power,
  (P_gen - curtailed)(1/eta_ab - 1), because curtailed power is never converted. The load
  converter loss is D(1 - eta_bl). Both heat bus_nz. Heater heat comes through their load rows.
- **Read-once values.** R (F1) and string health (F2) are read through ParamView every tick. The
  other EPS values are read once per `Eps`, which is fine for faults that set them at construction.

M5 choices where the spec is silent:

- **Topology reading.**
  - bus_core links to each of the six faces at 20 W/K.
  - The side faces form a ring, px-py-nx-ny-px, at 6.0 W/K; opposite faces are not linked.
  - Links named `<a>_to_<b>_W_K` (bus_pz-bus_px, bus_nz-bus_nx) join those two nodes.
  - Each wing links to the face with the same suffix through its hinge.
  - Each internal node links only to bus_core.
  - This reading gives explicit-Euler limits 2/lambda_max of 102.8 s (Bus A) and 124.0 s (Bus B)
    at 290 K, matching the spec's ~103 s and ~124 s.
- **Areas.** A_si is the face area, or a wing's sunward side, for every source; IR uses the same
  A_si and the view factor. A_i is the full area, both sides for a wing.
- **Coupling order.** Each tick, EPS runs with the node temperatures at the tick start, then the
  network takes one Euler step with that tick's EPS node heat.
- **Initial temperature: settled start** (Alex's decision; `scenario.thermal_init: settled`).
  - The first orbit of the run (idle Sun hold; burns and takes come hours later) is the periodic
    environment before epoch.
  - Node heat is EPS with no pass, take, slew or burn gates and heaters held off, flown until the
    pack energy repeats orbit to orbit, so the pack losses are periodic ones, not the initial
    charge.
  - The steady state of the orbit-averaged environment and heat is solved with scipy (nonlinear
    T^4), with a balance residual below 1e-6 W per node.
  - It is then refined to the periodic orbit by Newton shooting on the one-orbit map. Without this
    the battery starts about 2 K off and its 5.6 h time constant (C = 8000 J/K on 0.4 W/K) carries
    the error for many hours, because radiation goes as T^4 and does not average linearly.
  - Finally `thermal_init_orbits` = 3 orbits are flown from it. Because 190 ticks fall 11 s short
    of the orbit period, the replayed orbit has that small phase seam.
  - `thermal_init: fixed` starts every node at `thermal_initial_K` = 293.15 K, kept for tests.
- **Settling under settled start.** The battery is within 0.1 K of an independent 30-orbit plain
  spin-up from t = 0: 0.009 K on Bus A and 0.027 K on Bus B. From the old fixed 293.15 K start,
  runs started 20 K hotter and colder took 44 h (Bus A) and 54 h (Bus B) to agree within 0.1 K,
  because passes make the forcing only approximately periodic.
- **F7 signature.**
  - In idle Sun hold, +z is on the Sun, so bus_px is edge-on and gets no direct Sun. F7
    (alpha, epsilon x3) adds albedo absorption and extra IR loss.
  - bus_px is tied to bus_core at 20 W/K, so the whole vehicle cools about 1.5 K and bus_px cools
    most (1.7 K). Orbital swings change by less than 0.1 K.
  - The spec's "swing grows and the core carries it inward" would need bus_px to see the Sun,
    which it does only in some burn attitudes. The F7 test asserts the measured signature.
- **Heaters.** In a nominal settled-start run the battery stays at 285-294 K and the payload at
  276-291 K, so neither heater ever cycles.

