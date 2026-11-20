# Plan: extended faults, their simulation and their identification

Source: `Gantt—EXPAND - Fault Signatures & FDI Methods & Metrics.csv` (not in the repo). The CSV is
not the spec. Everything below that the spec does not state is an extension, and each value that
we choose is a default marked "spec silent". Last updated 2026-11-20.

## Status

The CSV lists 46 fault types. Ten are the spec's F1 to F10, so 36 are new.

| State | Faults |
|---|---|
| Simulated, detected and documented (14) | `gnss_jump`, `gnss_noise`, `tx_fade`, `mag_bias`, `mag_noise`, `mag_coupling`, `gyro_bias`, `gyro_scale`, `gyro_noise`, `gyro_stuck` (the posterior does not yet separate `gyro_noise` and `gyro_scale`), `st_dropout`, `st_false_star`, `st_stuck`, `gnss_outage` |
| Not built (22) | See the tables under "Remaining work" |

The documentation of the ten is in `docs/spec/EXPAND3_Extended_Faults_Supplement.md`.

## Decisions taken

- **Keep the existing implementations of `tx_fade` and the three magnetometer faults for now.** They
  differ from the CSV in three ways. We may add the CSV versions later as separate faults.
  - `tx_fade` is a loss of the crosslink transmit power, and it shows only in the signal strength
    that partners read. The CSV describes the transmit power of the vehicle on ground passes,
    measured by its own telemetry (`tx_power_dbm`, with a `ground_pass` flag). A vehicle can detect
    that version alone, so it has a single-satellite detector.
  - `mag_coupling` couples the magnetometer to the pack current, because the simulator has no
    magnetorquers. The CSV couples it to the magnetorquer dipole (`mtq_dipole_am2`).
  - The shapes differ. The CSV gives `mag_noise` as step or intermittent (we have ramp or
    intermittent) and `mag_coupling` as step only (we have step or ramp).
- **Detectors stay as they are.** The method is a dropout MLP with a GLR, in single-satellite and
  multi-satellite forms. The CSV names Boosted Tree and Rules, and we do not follow that.
- **One fault per seed**, as in the spec.

## Constraints on the work

- **Existing seeds stay bit-identical.** The extended draw comes from its own named streams
  (`draw.fault_ext` and others), behind `--fault-set extended`. The golden regression runs and every
  earlier batch must not change.
- **The rules in `CLAUDE.md` apply.** Numbers come from `params/` through `ParamView`, and
  randomness comes from named substreams. The simulator holds no detection logic, and the exit tests
  come before the code. Each docstring cites the CSV row.
- **Vehicles stay isolated.** Every other vehicle matches the run with no fault, except for the
  crosslink messages of the faulted vehicle and the drops on its links.
- **The CSV gives parameter names but no numbers** in its "Vendor expected values" column (for
  example `kt_nm_per_a` and `r_harness_ohm`). We choose the values as defaults, marked "spec silent".

## Step 0: groundwork

1. **Vehicle for each fault.** A fault starts a lead time before the take of its vehicle, and only
   SAT-2, SAT-3 and SAT-4 fly takes. Put every new fault on one of those three. The gyro faults
   are already on SAT-2 and SAT-3. A fault that belongs to a
   physical part, such as the tank and the thruster, fixes its own vehicle (SAT-4 for burns).
2. **New shapes.** The injector has step, ramp, exponential and intermittent. Add **spike**
   (`wheel_cage`) and **recovering** (`latchup`, and `dark_current` as the CSV lists it).
3. **Run time.** The six-vehicle run of 72 h took about 114 s in the last full test run, against a
   limit of 60 s. Profile it before adding signals.
4. **Prior and labels.** The prior becomes 1 over (10 + the number of extended faults). The
   posterior labels must start at the first observable effect, as the F3, F5 and F6 results showed.
5. **Scoring.** Add the metrics from the CSV to `score.py`: range-based F1, point-adjusted F1, NAB
   score, isolation accuracy, classification accuracy with a confusion matrix, unknown rate, false
   alarms per nominal hour (split by eclipse and SAA), missed detection rate and onset error.

## Step 1: finish the gyro faults

Done on 2026-11-19. The detector is `mcdn_gyro` (gyro against the rate from two star-tracker fixes),
with offset, slope and variance GLRs and a repeated-reading count, and a multi-satellite form that
gave no gain. `gyro_bias` and `gyro_stuck` get a posterior. `gyro_noise` and `gyro_scale` do not,
because one score for each fault cannot separate them from a bias or a stuck gyro. Open: decorrelate
the scores or fit the posterior on all four, and raise the size of `gyro_scale` (missed on 38 % of
seeds at 25 %). Both were left as they are for now.

## Step 2: faults with signals that already exist (12)

No new physical model is needed. Each fault is a scheduled change on a parameter.

| Fault | CSV shapes | Injection |
|---|---|---|
| `st_dropout` | intermittent, step | Loss of fix availability |
| `st_false_star` | none given | A fixed attitude offset on a fix that is marked valid |
| `st_stuck` | none given | A frozen fix |
| `gnss_outage` | step, intermittent | Lost fixes |
| `temp_sensor_bias` | step, ramp | Offset on one node |
| `temp_sensor_freeze` | step | Frozen reading on one node |
| `heater_stuck_on`, `heater_stuck_off` | step | Heater latch override |
| `radiator_degradation` | ramp | Emissivity of one radiator node (the node is our choice) |
| `panel_efficiency` | exponential, ramp | String efficiency |
| `capacity_fade` | ramp, step | Pack capacity |
| `cell_short` | step | OCV offset and a capacity step |

Order: the sensor faults, then thermal, then power. The power faults are confusable with
`resistance_growth` (F1), so plan the isolation tests there. The CSV pairs `st_false_star` and
`st_stuck` with `st_dropout`, and the gyro faults with `st_false_star`.

### Step 2 sizes (approved 2026-11-19; every value is a default, spec silent)

| Fault | Vehicle | Size | Shapes |
|---|---|---|---|
| `st_dropout` | SAT-3 | No fix after arm (intermittent: none during "on") | step, intermittent |
| `st_false_star` | SAT-2 | One fixed 0.5 deg rotation about an axis drawn once per seed, fix still valid | step |
| `st_stuck` | SAT-4 | The fix freezes at its last value, still marked valid | step |
| `gnss_outage` | SAT-4 | No fixes after arm; the nav filter coasts | step, intermittent |
| `temp_sensor_bias` | SAT-2 | +5 K on the battery thermistor (positive on purpose: a negative bias would cross the 276 K charge inhibit and change the physics) | step, ramp |
| `temp_sensor_freeze` | SAT-3 | The battery thermistor holds its last reading | step |
| `heater_stuck_on` | SAT-3 | Battery heater forced on (15 W) until the 313 K true-temperature cutoff | step |
| `heater_stuck_off` | SAT-2 | Battery heater never turns on. It shows only in cold cases, because a nominal battery never reaches the 283 K on-threshold (accepted) | step |
| `radiator_degradation` | SAT-2 | `bus_nz` white-paint radiator, emissivity 0.85 to 0.60 (x0.7) | ramp |
| `panel_efficiency` | SAT-2 | All four strings x0.85 (-15 % array power) | exponential, ramp |
| `capacity_fade` | SAT-2 | Pack capacity x0.75 (3.6 MJ to 2.7 MJ). Capacity is read once per `Eps` today, so it needs a per-tick read | ramp, step |
| `cell_short` | SAT-2 | Pack open-circuit voltage -3.75 V (one of eight series cells) and capacity x0.75. May reach the 26.4 V undervoltage shed at low SoC | step |

The sensor group (the first four) is built, with single, multi-satellite and store-and-forward
detectors (`mcdn_stavail`, `mcdn_gnssavail`, `mcdn_statt`). On seeds 100:199, `st_dropout`, `st_stuck` and
`gnss_outage` are found on 100 of 100 seeds in every form. `st_false_star` is found on 59 of 100 by one
vehicle and on 100 of 100 with peers. With peers the first alarm on `st_stuck` does not name `st_stuck` on
64 of 100 seeds. The posterior (mix seeds 1000:1222, prior 1/29) identifies `gnss_outage` (10 of 10),
`st_dropout` (10 of 10, 9 before the take) and `st_stuck` (5 of 5) in every form, and `st_false_star` on
5 of 5 with peers (4 before the take) and 0 of 5 with one vehicle. It also separates `st_stuck` from
`st_false_star`, which the thresholds do not. The sensor group is documented in the
supplement. The seeded extended draw now covers more faults, so a
`--fault-set extended` seed without a forced fault can draw a different fault than before. Forced
batches are not affected.

## Step 3: faults that need new signals (13)

| Order | Faults | CSV shapes | New signals |
|---|---|---|---|
| 1 | `tank_leak`, `pressure_sensor_bias` | step, ramp | Tank pressure and temperature from a blowdown model. This also separates F5 from F6. |
| 2 | `wheel_drag` | exponential, ramp, step | Wheel speed, current and temperature from a motor model |
| 2 | `wheel_cage` | intermittent, spike | Same |
| 2 | `wheel_seize` | step | Same, with the speed dropping to 0 |
| 3 | `clock_drift`, `clock_jump` | step, ramp / step | Clock of the computer against GNSS time |
| 4 | `sw_reset`, `cdh_hang` | step, intermittent / step | Uptime of the computer |
| 5 | `feeder_switch_open` | step, intermittent | Voltage and current for each load, bus voltage |
| 5 | `feeder_sensor_bias`, `feeder_sensor_noise` | step, ramp / step | Same |
| 5 | `latchup` | recovering, step | A current limit, a trip and the unserved power |

The tank comes first, because the CSV confirms that `isp_loss` shows a normal pressure drop and
`authority` a low one. The feeder group is the largest change. `wheel_drag` and `wheel_friction` are
confusable (constant against speed-proportional extra current), and so are `sw_reset` and
`upset_rate`, and `clock_drift` and `clock_jump`.

## Step 4: magnetorquers (1 fault, and a rebuild of two others)

`mtq_stuck` needs magnetorquers and a desaturation law. That changes the wheel momentum and
therefore every wheel baseline, so it stays off by default and out of the golden runs. The CSV also
wants `mtq_dipole_am2` and `mtq_coil_a` as telemetry for `mag_coupling`, `mag_bias` and `mtq_stuck`.
When the magnetorquers exist, add the CSV version of `mag_coupling`. Do this last, and only if wanted.

## For every fault

1. Write the exit tests first. Seeds with no fault stay bit-identical, and the other vehicles are
   unchanged.
2. Add the parameters to `params/` as defaults marked "spec silent", and add the injection point.
3. Run a forced batch of 200 seeds for each shape.
4. Build the detector from `mcdnn.py`. Slow drifts use exogenous inputs only, because history lets
   the network extrapolate the drift. Add a multi-satellite form only where vehicles share a
   reference. The likeliest cases are `clock_drift` against peers and anything with a common
   environment. Per-vehicle mechanical faults (wheels, burns) probably gain nothing, as F3 to F6
   showed.
5. Score it. Check the false alarms and the first-alarm wrong-id rate.
6. Fit the posterior.
7. Add it to the supplement and the artifact.

Two checks belong early for every fault. First, look for a nuisance signal that looks like the
fault. The magnetic field disturbance caused hundreds of false alarms. Second, check the confusion
groups, because the first alarm often names the wrong fault. These groups are known:

- `heater_stuck_off`, `temp_sensor_bias` and `insulation_degradation` (F7)
- `capacity_fade`, `resistance_growth` (F1) and `cell_short`
- `wheel_drag` and `wheel_friction` (F3)
- `gyro_bias` and `st_false_star`
- `sw_reset` and `upset_rate` (F9)
- `clock_drift` and `clock_jump`
- `mag_bias` and `mag_coupling`

## Cost

One batch of 200 seeds took about 23 min on 24 CPUs. With two shapes each, the 26 faults make about
50 batches, or roughly 20 h of simulation, plus the detection runs. Most of it belongs on the server.
Ask before starting any long run.

## Decisions

1. Begin with the gyro faults, then do steps 2-4
2. Ask me for any numbers you're not sure of
3. The magnetorquer work is wanted, but we'll need to implement a desaturation law and rerun the wheel baselines afterwards
4. The simulation is getting really slow. We should focus on making it faster and reusing no-fault runs across different fault experiments when possible
