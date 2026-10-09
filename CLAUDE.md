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
- `src/expand3/constellation.py`: `propagate_constellation` gives r and v at ticks, plus the Sun,
  the shadow fraction and station visibility.
- `src/expand3/units.py`: definitional unit conversions only.
- Stubs for later milestones: `mission/`, `faults.py`, `sensors.py`, `links.py`, `recorder.py` and
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
