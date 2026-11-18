# EXPAND.3 Phase I Extended Faults

Supplement to the Phase I Simulation and Fault Specification · 2026-11-18

## Overview

The spec defines ten faults, F1 to F10. This supplement adds six more. To make the new faults
possible, it also adds three things to the simulator:

1. An onboard navigation filter. Each vehicle uses it to estimate its own position and velocity
   from its GNSS fixes.
2. A signal-strength reading. Each message that one vehicle receives from another carries the
   power at which it arrived.
3. A magnetometer. It measures a magnetic field that is noisier than the model that the vehicle
   carries.

The six new faults are `gnss_jump`, `gnss_noise`, `tx_fade`, `mag_bias`, `mag_noise` and
`mag_coupling`.

Someone who has the spec and this document should be able to rebuild all of it. The spec does not
mention any of these additions. Every value here is a default, and the spec is silent about it. A
value stays in force until the owner of the spec changes it. The section "Choices where the spec is
silent" gives a reason for each choice that needs one.

## Scope

Build:

- A table of extended faults, and four shapes for the start of a fault (step, ramp, exponential,
  intermittent)
- An onboard navigation filter that runs on the GNSS fixes of each vehicle
- Two GNSS faults: `gnss_jump` and `gnss_noise`
- A received-power reading for each crosslink message that is delivered
- One crosslink fault: `tx_fade`
- A true magnetic field, a magnetometer, and a reference field that each vehicle holds on board
- Three magnetometer faults: `mag_bias`, `mag_noise` and `mag_coupling`

Do not build detection logic. The navigation filter and the onboard field reference model what a
vehicle can compute from its own sensors. They use sensor outputs and the commands of the vehicle.
They never use the true state.

This supplement also relies on the base simulator for everything that the spec leaves open: the
30 s control tick, the 10 s integration step, the star tracker, and the other choices that are
recorded as M1 to M8 in the `CLAUDE.md` file of the repository.

## Conventions

Units are SI, except that orbit state is in km and km/s, as in the spec. Each field name ends with
its unit. ECI is GCRS and ECEF is ITRS. The pole of the Earth is held fixed during a run, and one
rotation for each tick takes a vector from ECI to ECEF. Quaternions have the scalar first and
rotate from the body frame to the inertial frame. The inverse rotation takes a vector from the
inertial frame to the body frame.

Each sensor takes a sample on every control tick where the time is a multiple of its rate.

Random numbers come from named streams that are derived from the seed of the run. A stream depends
only on the seed and on its name. The first 128 bits of the SHA-256 hash of the name give the
`SeedSequence` spawn key. Adding a new stream therefore never changes another stream. Every stream
draws a number for every sample, whether or not the sample is used. These streams are new:

| Stream | Used for |
|---|---|
| `draw.fault_ext` | choosing which fault a seeded extended draw picks |
| `draw.fault_shape` | choosing the start shape of that fault |
| `draw.episodes` | the start and end times of intermittent episodes |
| `draw.field` | the magnetic disturbance that all vehicles share, drawn once for each seed |
| `noise.mag.<sat>` | the noise of the magnetometer |
| `noise.rssi.<sender>.<receiver>` | the error of the signal-strength reading, for each ordered pair |

The navigation filter uses no random numbers. Model code takes every number from `ParamView`. A
fault is a function of time that is attached to one parameter path on the `ParamView` of the faulted
vehicle. Each model looks up the value for the current time.

## Model: navigation filter

Each vehicle runs an extended Kalman filter (EKF) on its own GNSS fixes. The filter produces one
navigation solution for each fix, once every 60 s. It has three inputs:

- the fixes, which give position and velocity;
- the acceleration that the vehicle ordered for its own burn (the constant prograde acceleration
  from the spec, which is zero outside a burn and is held constant over the control tick);
- the parameters.

The filter never uses the true state. It also does not know the thrust that was actually delivered,
which F5 and F6 change.

State: x = [r, v, k]. The position r [km] and the velocity v [km/s] are in ECI. The drag scale k is
the density of the atmosphere as a multiple of the nominal density, and it starts at 1.

Dynamics. The force model is the one in the spec: a point mass, J2 and exponential drag. For the
atmosphere, the filter uses the middle of each range that the seeds draw from. It does not use the
draw of the seed. So rho_0 is the geometric mean of its log-uniform range, and H is the midpoint of
its uniform range. A vehicle cannot know the draw, and k takes up the difference.

    v_dot = a_2body(r) + a_J2(r) + k a_drag,nom(r, v) + a_cmd / 1000

Between two fixes, RK4 integrates the state with steps of at most `step_s`. The drag acceleration
is computed once at the start of the interval and is held constant over it.

Covariance. Let dt be the time between two fixes.

    Phi = expm(A dt),   A = [ 0  I  0 ; G  0  a_drag,nom ; 0  0  0 ]
    cov <- Phi cov Phi^T + Q
    Q (r, v block) = sigma_a^2 g g^T,   g = [ (dt^2 / 2) I ; dt I ]
    Q (k) = walk^2 dt

G is the derivative da/dr of two-body gravity plus J2. It is computed by central differences, with a
step of 1e-4 |r| on each axis. Phi is recomputed every `linearize_every_fixes` fixes, and also
whenever dt changes.

Update. The measurement is z = [r_fix, v_fix], and H = [I6 0]. The measurement covariance is
R = diag(sigma_r^2 (three times), sigma_v^2 (three times)). It comes from the nominal noise of the
receiver in the Signals table, and not from the larger noise that a fault may cause.

    K = cov H^T (H cov H^T + R)^-1
    x <- x + K (z - H x),  k <- max(k, 0)
    cov <- (I - K H) cov (I - K H)^T + K R K^T

Initial state: the first fix, with k = 1 and cov = blockdiag(R, drag_scale_sigma^2). The filter does
not reject any fix. A fault in the receiver therefore enters the solution, and the gain of the
filter sets how fast it enters.

Signals added. `nav_r` [km] and `nav_v` [km/s], one for each fix. The truth of each is the true
state at the time of the fix.

A healthy vehicle behaves as follows after the first hour:

| Quantity | Raw fix | Navigation solution |
|---|---|---|
| Position error, rms on each axis | 1.5 m | about 0.6 m |
| Velocity error, rms on each axis | 0.05 m/s | about 0.001 m/s |

The difference between a fix and the solution has a standard deviation of about 1.3 m on each axis.
It is white noise, and it is uncorrelated between vehicles (the correlation is below 0.03). The
solution follows an offset in the fixes with a delay. After a step, the solution takes a few fixes
to catch up. After a ramp, a steady mean difference of 0.1 to 0.65 m remains on each axis, and the
spread grows from 1.3 m to about 2 m.

## Model: GNSS (gnss_jump, gnss_noise)

This is the GNSS signal from the spec, with two terms that a fault can change:

    r_fix = r_true + noise_r x noise_mult + offset_m / 1000        (km, on each axis)
    v_fix = v_true + noise_v x noise_mult

The noise is drawn first and then scaled, so a fault does not change the stream. The offset `offset_m`
is added equally to every ECI axis. Nominally, `offset_m` is 0 and `noise_mult` is 1.

## Model: crosslink signal strength (tx_fade)

Each crosslink message that is delivered carries one reading of the power at which it arrived at the
receiver, in dBm. The antennas are omnidirectional. A vehicle can sense what it receives, but not
what it transmits. A fault in the transmit power of one vehicle therefore shows only in the readings
of its partners.

    lambda   = c / (frequency_GHz x 1e9)                        [m]
    FSPL     = 20 log10(4 pi range / lambda)                    [dB], with range in m
    P_rx     = tx_power + tx_gain + rx_gain - FSPL - tx_loss_dB(t)
    rssi     = P_rx + e,   e ~ N(0, rssi_sigma_dB)

The range is the true distance at the tick. The value `tx_loss_dB` belongs to the sender and is
nominally 0. A message that is not delivered has no reading (NaN). At 1000 km the formula gives
-116.3 dBm. Over a run, the readings range from about -115 to -62 dBm, with a median near -104 dBm.
The state messages, their receive noise and F8 are as in the spec, and they do not depend on the
reading.

## Model: magnetic field and magnetometer

True field. The true field is B_true = B_IGRF + dB, evaluated at the true position in ECEF.

B_IGRF is the IGRF-14 vector at the epoch of the run. The function `ppigrf.igrf_gc(r, colatitude,
longitude, date)` returns (B_r, B_theta, B_phi). Here theta is the colatitude, so B_theta points
south and B_phi points east.

    up = r / |r|
    south = (cos(colat) cos(lon), cos(colat) sin(lon), -sin(colat))
    east  = (-sin(lon), cos(lon), 0)
    B_IGRF = B_r up + B_theta south + B_phi east

Its magnitude is the same field magnitude that the SAA code already uses (the two agree to a
relative 1e-9).

The disturbance dB is the same for every vehicle at the same place and time. It is a sum of K
cosine waves in ECEF, and it is drawn once for each seed from `draw.field`:

    dB(x, t) = sum_k a_k cos(kvec_k . x - w_k t + phi_k)

The stream gives the following for each wave k, in this order:

| Quantity | Draw |
|---|---|
| Direction d_k | a unit vector, taken from a standard normal 3-vector |
| Wavelength | log-uniform from `wavelength_min_km` to `wavelength_max_km`; then kvec_k = 2 pi / wavelength x d_k |
| Period | log-uniform from `period_min_h` to `period_max_h`; then w_k = 2 pi / period |
| Phase phi_k | uniform on [0, 2 pi) |
| Polarisation p_k | a unit vector perpendicular to d_k (the normalised cross product of d_k with a standard normal vector) |
| Direction of travel | w_k gets a random sign |

The amplitude vector is a_k = rms_nT sqrt(2 / K) p_k. With this scaling, the rms of |dB| equals
`rms_nT`.

Magnetometer. It measures three components along the body axes, once every `rate_s`:

    B_meas = B_true,body + noise x noise_mult + bias_nT + coupling_nT_per_A x I_pack
    noise ~ N(0, sigma_nT) on each axis,   stream noise.mag.<sat>

B_true,body is B_true rotated first to ECI and then to the body frame, with the true attitude.
I_pack is the true pack current from the EPS model. The value `bias_nT` is added equally to every
axis. Nominally, bias and coupling are 0 and `noise_mult` is 1. The truth of the signal is
B_true,body without the noise and without the fault terms.

Reference field on board. The vehicle expects a reading that equals the IGRF field alone. It
computes this expectation as follows:

- The position is the navigation solution at the tick, and not the true position.
- The attitude is the quaternion that the star tracker measured. It is NaN on ticks without a fix.
- The body-frame model is B_IGRF(r_nav), rotated from ECEF to ECI and then to the body frame with
  quat_to_dcm(q_meas)^T.
- The magnitude model is |B_IGRF(r_nav)|. It needs only the position, so it always exists.

Signals added: `mag` (3 axes), `mag_model` (3 axes, NaN without a tracker fix) and `mag_model_abs`.
The truth of `mag_model` and `mag_model_abs` is the IGRF field at the true position and attitude.

For a healthy vehicle, the residual in magnitude, |B_meas| - |B_model|, has a standard deviation of
about 177 nT. This is a vector disturbance of 300 nT rms, seen along one direction. The change in
this residual over 60 s has a standard deviation of about 98 nT. That change is made of two parts:
the disturbance itself changes by 88 nT, and the sensor noise, differenced over the same interval,
changes by 42 nT. The star tracker gives the attitude about half of the time in Sun hold.

## Signals

The Signals table in the spec gains these rows:

| Signal | Rate | Noise | Notes |
|---|---|---|---|
| Navigation position and velocity | 60 s | none (output of the filter) | one for each GNSS fix |
| Magnetometer, 3 axes | 60 s | 30 nT on each axis | body frame |
| Onboard field model, 3 axes and magnitude | 60 s | none | NaN without a tracker fix |
| Received power of a crosslink message | one for each delivered message | 0.5 dB | in dBm |

## Faults

These rows are added to the Faults table:

| Fault | Vehicle | Parameter that changes | Nominal | Injected | Profile |
|---|---|---|---|---|---|
| gnss_jump | SAT-2 | `sensor_faults.gnss.offset_m` | 0 m | 10 m on every axis | step or ramp, t_r = 6 h |
| gnss_noise | SAT-3 | `sensor_faults.gnss.noise_mult` | 1 | x5 | ramp (t_r = 6 h) or intermittent |
| tx_fade | SAT-2 | `sensor_faults.radio.tx_loss_dB` | 0 dB | 6 dB | step or ramp, t_r = 6 h |
| mag_bias | SAT-3 | `sensor_faults.mag.bias_nT` | 0 nT | 300 nT on every axis | step or ramp, t_r = 6 h |
| mag_noise | SAT-4 | `sensor_faults.mag.noise_mult` | 1 | x5 | ramp (t_r = 6 h) or intermittent |
| mag_coupling | SAT-2 | `sensor_faults.mag.coupling_nT_per_A` | 0 nT/A | 100 nT/A on every axis | step or ramp, t_r = 6 h |

In the Profile column, the first shape of each row is the default. Intermittent episodes have a mean
"on" time of 1800 s and a mean "off" time of 3600 s.

A fault changes the parameter as in the spec. Here s(t) is the start shape of the fault.

    theta(t) = theta_0 (1 + (m - 1) s(t))          (multiplier m)
    theta(t) = theta_0 + (set_to - theta_0) s(t)    (nominal value zero: set the value directly)

The start shapes are defined for times t at or after the start time t_arm. Before t_arm, s = 0.

| Shape | s(t) |
|---|---|
| step | 1 |
| ramp | min(1, (t - t_arm) / t_r) |
| exponential | 1 - exp(-(t - t_arm) / t_r) (none of these faults uses it) |
| intermittent | 1 inside an "on" episode, and 0 between episodes |

Intermittent episodes begin in the "on" state at t_arm. The length of each "on" or "off" period is
exponential with the mean given above. The lengths are drawn from `draw.episodes`, and they continue
until a boundary falls beyond `episode_horizon_s` (3.6e5 s) after t_arm.

Start time and draw. The start time t_0 is the fault lead before the take of the faulted vehicle, as
in the spec. A seeded extended draw is uniform over F1 to F10 and the extended table. The order is
F1 to F10, and then the extended ids in alphabetical order. The draw uses `draw.fault_ext`. The shape
is uniform over the shapes of that fault, and it uses `draw.fault_shape`. The lead uses
`draw.fault_lead` from the spec, so each seed has one lead. There are ten extended faults, so a
seeded draw has a prior of 1/20 for each fault. A fault can also be forced, with a forced lead and a
forced shape. A forced shape must be one of the shapes of that fault.

## Choices where the spec is silent

Faults and vehicles. Every vehicle, size, ramp time and episode length in the Faults table is a
default. Each fault is on a vehicle that has a take (SAT-2, SAT-3 or SAT-4). The spec requires
this for any fault, because the start of a fault depends on the take of its vehicle. SAT-2 has three
of the six faults. They never occur together, because a seed draws one fault.

Sizes. A GNSS jump of 10 m on every axis is a vector of 17.3 m, which is 11 times the noise. It is
large enough to count as a fault, and small enough that the navigation solution of the vehicle can
take it in. With noise multiplied by 5, the noise is 7.5 m and 25 cm/s. The magnetometer bias of
300 nT is 10 times the noise of the sensor. A coupling of 100 nT/A gives 1000 nT at a pack current of
10 A. A transmit loss of 6 dB cuts the power to one quarter.

Navigation filter. An EKF is enough, because the dynamics are smooth and close to linear over 60 s.
The nominal atmosphere with a density scale is the simplest way to let the filter follow a density
error that it cannot know. The ordered thrust is a known input, because the vehicle knows what it
ordered. Without it, a burn would be a large acceleration that the filter would treat as process
noise. The process noise sets the gain. Because no fix is rejected, the gain also sets how fast a
faulted fix enters the solution. Three changes made the filter faster: holding the drag
acceleration constant over a fix interval, using a 30 s step, and refreshing Phi only every 5 fixes.
Together they cut the cost from about 5.6 s to 0.8 s for each vehicle and day, with no visible loss
of accuracy.

Link budget. A 5 W transmitter and 3 dBi antennas at 2.2 GHz are plausible values for S-band. A
reading error of 0.5 dB is also plausible. The antennas are omnidirectional, so there is no pointing
term. The reading uses the true range, and not the range that the vehicles estimate.

Magnetic field. The true field has a disturbance because the real field differs from IGRF by much
more than the noise of a sensor. All vehicles at the same place and time have the same disturbance,
so cluster mates see nearly the same one. A model that is IGRF alone cannot tell the disturbance
from a magnetometer bias. The size is 300 nT rms, which is about 1 % of the field. The 24 waves,
the wavelengths of 2000 to 8000 km and the periods of 1 to 6 h give a field that is nearly the same
across a cluster, and that decorrelates over half an orbit. The field is not smooth along the track
at the rate of the magnetometer, which is 60 s. A vehicle moves about 450 km between two samples, so
the disturbance changes by about 88 nT (rms) from one sample to the next. The sensor noise,
differenced over the same interval, changes by 42 nT. A single vehicle therefore sees the
disturbance as noise as well as bias. The reference on board uses the navigation position and the
tracker attitude, so that it cannot use the truth.

Coupling. The coupling term uses the true pack current, and not the measured one. It therefore does
not include the bias and the noise of the current sensor.

## Parameters

`params/links_saa_imager.yaml`:

| Parameter | Value |
|---|---|
| `link_budget.tx_power_dBm` | 37 |
| `link_budget.tx_gain_dBi`, `rx_gain_dBi` | 3, 3 |
| `link_budget.frequency_GHz` | 2.2 |
| `link_budget.rssi_sigma_dB` | 0.5 |
| `field_disturbance.rms_nT` | 300 |
| `field_disturbance.modes` | 24 |
| `field_disturbance.wavelength_min_km`, `wavelength_max_km` | 2000, 8000 |
| `field_disturbance.period_min_h`, `period_max_h` | 1, 6 |
| `signals.magnetometer.rate_s`, `sigma_nT` | 60, 30 |
| `signals.nav.rate_s` | 60 |
| `signals.nav.accel_sigma_m_s2` | 1.0e-5 |
| `signals.nav.step_s` | 30 |
| `signals.nav.linearize_every_fixes` | 5 |
| `signals.nav.drag_scale_sigma` | 1.0 |
| `signals.nav.drag_scale_walk_per_sqrt_s` | 1.0e-3 |
| `sensor_faults.gnss` | offset_m 0, noise_mult 1 |
| `sensor_faults.mag` | bias_nT 0, noise_mult 1, coupling_nT_per_A 0 |
| `sensor_faults.radio` | tx_loss_dB 0 |

`params/faults_extended.yaml` has the six rows of the Faults table and `episode_horizon_s` of 3.6e5.
Each row gives its vehicle, its parameter path, and either `m` or `set_to`. It also gives its
default `profile`, its `shapes`, and `t_r_s` (21600). The two noise faults also give `t_on_s` (1800)
and `t_off_s` (3600). The `profile` of a row is the first of its `shapes`. A row has `t_r_s` when at
least one of its shapes is a ramp or an exponential. It has `t_on_s` and `t_off_s` when at least one
of its shapes is intermittent.

## Outputs

`measurements.parquet` gains columns named `<name>_meas` and `<name>_true`. They are NaN where the
signal is not sampled.

| Columns | Content |
|---|---|
| `nav_r_{x,y,z}_km` | navigation position |
| `nav_v_{x,y,z}_km_s` | navigation velocity |
| `mag_{x,y,z}_nT` | magnetometer |
| `mag_model_{x,y,z}_nT` | field model on board, in the body frame |
| `mag_model_abs_nT` | magnitude of the field model on board |

`crosslink.parquet` gains `rssi_dBm`, which is NaN unless the message was delivered. The labels and
the manifests record the shape that was used (`profile`). The manifest also records the fault that
the seed would draw from the extended set.

## Checks

A rebuild should pass these checks:

1. The magnitude of the IGRF vector equals the field magnitude of the SAA code, to a relative
   1e-9.
2. The disturbance has zero mean and an rms of `rms_nT`. It is the same for all vehicles, and it
   changes from seed to seed. Cluster mates differ by much less than vehicles that are half an orbit
   apart. It changes over hours.
3. The noise of the magnetometer is Gaussian with zero mean and a standard deviation of `sigma_nT`.
   The bias is added on every axis. The value `noise_mult` scales only the noise. The coupling adds
   `coupling x I` on every axis.
4. The model on board equals IGRF rotated by the measured attitude. It is NaN without a tracker fix,
   and its magnitude exists at every tick.
5. A reading equals the link budget at the true range, plus the reading error. A fade lowers only the
   outgoing readings of the sender, and it leaves every other crosslink quantity unchanged.
6. The navigation filter is more accurate than the raw fix, and its error agrees with its covariance.
   It follows an offset in the fixes, it does not use the truth, it is deterministic, and it stays on
   the orbit through a burn. Its drag scale moves from 1 toward the true scale.
7. After the fault begins, only the affected signal of the faulted vehicle differs from the run of the
   same seed with no fault. For `tx_fade`, the readings of the messages of that vehicle also differ.
8. The six-vehicle run of 72 h takes less than 60 s on the development machine.

## Known issues

- Three gyro faults are on SAT-1, SAT-5 and SAT-6 (`gyro_bias`, `gyro_scale` and `gyro_noise`). The
  spec says that a fault is on SAT-2, SAT-3 or SAT-4, so these three break the rule. Their start
  time comes from a take that those vehicles do not fly. Any consumer that reads start times from
  the recorded takes therefore cannot use them. `mag_coupling` was first on SAT-1 for the same
  reason, and it was moved to SAT-2 on 2026-11-18. This document does not describe the gyro faults.
- The SAA gate still uses IGRF alone. It does not include the disturbance. F9 and the SAA therefore
  match the spec, and they differ from the field that the magnetometer measures.
- The drag scale learns only part of a density error. In one test it moved from 1 to about 6, while
  the true scale was about 46. A larger `drag_scale_walk_per_sqrt_s` moves it further.
- Two comments in the parameter files are wrong. The link budget gives -116.3 dBm at 1000 km, and
  not about -113 dBm. The comment in `faults_extended.yaml` says a cluster of four vehicles with
  three partners, but each cluster has three vehicles and two partners.
- The most recent full test run took about 114 s for the six-vehicle 72 h run, which is above the
  limit of 60 s in check 8. We have not checked whether other jobs were using the machine at that
  time, and we have not repeated the run on an idle machine.
- A seed carries one fault. Two faults at the same time are not supported.
- Faults of the clock and of the magnetorquers are not built.
