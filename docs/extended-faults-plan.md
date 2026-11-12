# Plan: 36 additional faults, their simulation and their identification

Source: `Gantt—EXPAND - Fault Signatures & FDI Methods & Metrics.csv` (downloaded; not in the
repo). It is not the spec. Everything below that the spec does not state is an extension, with
spec-silent defaults marked as in the earlier milestones. Nothing here is built yet.

## What the CSV adds

It lists 46 fault types across ADCS, navigation, propulsion, power, thermal and C&DH, comms and
payload. Ten are the spec's F1 to F10:

| Spec | CSV name |
|---|---|
| F1 | `resistance_growth` |
| F2 | `string_open` |
| F3 | `wheel_friction` |
| F4 | `wheel_kt` |
| F5 | `isp_loss` |
| F6 | `authority` |
| F7 | `insulation_degradation` |
| F8 | `link_dropout` |
| F9 | `upset_rate` |
| F10 | `dark_current` |

The other **36** are new. The CSV also uses shapes the injector lacks (exponential,
intermittent, spike, recovering; it has step and ramp), and a metric list: MTTD, MTTI, detection,
persist, identification and confirm latency, range-based F1, point-adjusted F1, NAB score,
isolation accuracy, classification accuracy, unknown rate, false alarm rate per nominal hour
(split by eclipse and SAA), missed detection rate and onset error.

It also resolves the F5/F6 problem found in the detectors. `isp_loss` shows a normal tank
pressure drop and `authority` a low one, so a tank pressure signal would separate them. The
simulator only has the tank gauge (0.10 kg noise per sample against about 0.04 kg drained in a
burn), which cannot.

## Constraints on the work

- **Existing seeds stay bit-identical.** The extended draw comes from its own named stream
  (for example `draw.fault_ext`) behind an option such as `--fault-set extended`. The golden
  regression runs and every earlier batch must not change.
- **Rules from `CLAUDE.md` apply.** Numbers come from `params/` through `ParamView`, randomness
  from named substreams, no detection logic in the simulator, exit tests before code, and a spec
  citation (here, a CSV row) in each docstring.
- **Isolation of vehicles** holds as for F1 to F10: other vehicles are identical to the
  no-fault run except for crosslink messages from the faulted vehicle and drops on its links.

## Phase A: existing signals, new injection points (18 faults)

| Group | Faults | Signals already in the batch | Injection |
|---|---|---|---|
| IMU | `gyro_bias`, `gyro_scale`, `gyro_noise`, `gyro_stuck` | `imu_omega_*`, `st_q_*`, `st_available` | Scheduled bias, scale, sigma and freeze on the IMU sensor model. |
| Star tracker | `st_dropout`, `st_false_star`, `st_stuck` | `st_available`, `st_q_*` | Availability loss, a fixed attitude offset, a frozen fix. |
| GNSS | `gnss_outage`, `gnss_jump`, `gnss_noise` | `gnss_valid`, `gnss_r_*`, `gnss_v_*` | Lost fixes, a position offset, larger noise. |
| Thermistors | `temp_sensor_bias`, `temp_sensor_freeze` | `T_*_K_meas` | Offset or frozen reading on one node. |
| Thermal | `heater_stuck_on`, `heater_stuck_off`, `radiator_degradation` | heater commanded and delivered, `T_*` | Heater latch override; a radiator node's emissivity (which node is a spec-silent choice). |
| Power | `panel_efficiency`, `capacity_fade`, `cell_short` | `string_p_*`, `pack_v`, `pack_i`, `soc_est` | String efficiency (exponential); pack capacity (ramp or step); OCV offset and capacity step. |

Work: extend `faults.yaml` and the injector's onset profiles (exponential, intermittent as
on/off episodes from a named stream, spike, recovering), then scheduled parameters in
`sensors.py`, `vehicle/eps.py` and `vehicle/thermal.py`. No new physical models.

## Phase B: new signals from small models (14 faults)

| Group | Faults | New signal or model needed |
|---|---|---|
| Tank | `tank_leak`, `pressure_sensor_bias` | Tank pressure and temperature from a blowdown (PVT) model with the bookkept propellant. This also makes F5 and F6 separable. |
| Thruster | (F5, F6 gain `cat_bed_temp_k`, valve state) | Catalyst-bed temperature, valve state. |
| Wheels | `wheel_drag`, `wheel_cage`, `wheel_seize` | Wheel speed, current and temperature from a motor model (Kt, idle current, viscous and Coulomb friction); a seized wheel stalls. |
| Feeders and PDU | `feeder_switch_open`, `feeder_sensor_bias`, `feeder_sensor_noise`, `latchup` | Per-load feeder voltage and current, bus voltage, a current limit and trip. |
| C&DH | `sw_reset`, `cdh_hang` | OBC uptime. |
| Clock | `clock_drift`, `clock_jump` | OBC time against the GNSS receiver time. |
| Radio | `tx_fade` | Transmit power on passes. |

Work: new sensor blocks in `sensors.py` and models in the relevant `vehicle/` modules, each
with params, a spec-silent default and an exit test that the no-fault output is unchanged
(except for the added columns).

## Phase C: a new subsystem (4 faults)

`mag_bias`, `mag_noise`, `mag_coupling`, `mtq_stuck` need a body-frame magnetometer (the IGRF
model gives the field magnitude `field_nT` today, not the vector), magnetorquers and a
desaturation law. That changes wheel momentum and therefore every wheel-related baseline, so it
must be off by default and kept out of the golden runs. Do this last, and only if wanted.

## Identification

- **Residual methods** come from the CSV's own "Residual Method" column. Most map to what exists:
  Gaussian GLR for offsets, slope GLR for scale or growth, Poisson and binomial GLR for counts.
- **New tests needed:** a variance GLR (`gyro_noise`, `mag_noise`, `feeder_sensor_noise`,
  `gnss_noise`), a frozen-signal test (`gyro_stuck`, `st_stuck`, `temp_sensor_freeze`), a CUSUM
  for `latchup`, and uptime-jump tests for `sw_reset` and `cdh_hang`. Intermittent faults
  (`wheel_cage`, `gnss_outage`, `st_dropout`) need range-based scoring.
- **MC dropout:** reuse the dropout MLP plus GLR from `mcdnn.py`. Slow drifts take exogenous-only
  inputs, since self-history lets the model extrapolate drift (found for F1 and F7).
- **Multi-satellite:** likely to help only where there is a shared reference. The candidates are
  `gnss_jump` and `gnss_noise` against crosslink-measured partner states, `clock_drift` against
  peers, and `tx_fade` against peers' links. Per-vehicle mechanical faults (wheels, burns)
  probably will not, as F3 to F6 showed.
- **Confusable groups to plan for:**
  - `heater_stuck_off`, `temp_sensor_bias` and `insulation_degradation`.
  - `capacity_fade`, `resistance_growth` and `cell_short`.
  - `wheel_drag` and `wheel_friction` (constant against speed-proportional extra current).
  - `gyro_bias` and `st_false_star`.
  - `sw_reset` and `upset_rate`.
  - `clock_drift` and `clock_jump`.
  - `mag_bias` and `mag_coupling`.
- **Scoring and posterior:** add the CSV's metrics to `score.py` (range-based F1, PA-F1, NAB,
  false alarms per hour split by eclipse and SAA, onset error, unknown rate, and a confusion
  matrix over all classes). The posterior's prior changes from one in ten to one in the new
  class count, and its channels need per-fault labels that start at the first observable effect,
  not at arm (the F3, F5 and F6 lesson).

## Order of work

1. Phase A, in this order: scheduled-parameter plumbing and the new profiles; the sensor faults
   (IMU, star tracker, GNSS, thermistors); then thermal and power.
2. Detectors for Phase A, then a batch with the extended fault set and scoring.
3. Phase B, tank first, since it unlocks F5 against F6. Then wheels, feeders, clock, C&DH, radio.
4. Phase C if wanted.
5. Re-run the posterior and the multi-satellite comparison on the extended set.

## Decisions needed

1. **Authority.** Are the new faults extensions with spec-silent defaults? The CSV's "vendor
   expected values" column lists ratings (wheel max torque and Kt, tank blowdown curve, feeder
   resistances, clock ppm and so on) that the repo does not have.
2. **Vehicle and lead.** Which vehicle does each new fault hit, and does it use the same
   fault-lead draw as F1 to F10?
3. **One fault per seed, or several?** The spec has one per seed.
4. **Phase C.** Is the magnetometer and magnetorquer subsystem wanted, given the risk to wheel
   dynamics?
