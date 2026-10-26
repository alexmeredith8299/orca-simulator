# Fault detection on the 100-seed batch

Batch: `~/orca-batches/m8_seeds_0_99` (one seeded fault per seed). Detectors: `expand3 detect DIR
--detector all`. Scoring: `expand3 score DIR`. Seeds 0-69 tune, 70-99 test. Full tables are in
`DIR/detect/all/score.md`.

## Method

- **Inputs.** `measurements`, `crosslink`, `wheel_steps` and `takes` only. Nominal values (ratio
  means, drop probability, upset rate, wheel k and c) are fitted from fault-free tune data: other
  vehicles, and the faulted vehicle before arm. The one model read from `params/` is the spec's
  dark-current equation (F10).
- **Thresholds.** One per fault id, the smallest h that gives at most 0.05 alarm episodes per
  fault-free vehicle-day on the tune seeds (951.8 vehicle-days), and never below 8. The floor
  matters where the data are noiseless (drops, upsets, wheel fits), so a tuned threshold would
  otherwise sit at zero.
- **Detectors.**
  - F2: Gaussian GLR (window 10, downward shift) on each string's power over the mean of the
    others, while sunlit (mean of all strings above half its 99th percentile).
  - F8: binomial GLR (window 20 messages) per link, summed over a vehicle's links. Only the
    vehicle with the largest sum can alarm, which keeps the neighbours quiet.
  - F9: Poisson GLR (window 20 ticks) on upsets per tick inside the SAA.
  - F10: Gaussian GLR on log2(D / (D0 2^((Tp - Tref)/dT))) at each take.
  - F3/F4: causal per-wheel least squares of `(h_prev - h)/dt = k u + c h_prev` within a slew,
    from the fourth step on. F4 scores k, F3 scores c, in units of the (floored) fault-free spread.
- **Scoring.** An alarm on the faulted vehicle at or after arm is a detection. Every other alarm
  (other vehicles, and the faulted vehicle before arm) is a false alarm. Delay is from arm and lead
  is the time to the take start of the faulted vehicle.

## Results (target 0.05 false alarms per vehicle-day)

| fault | tune detected | test detected | median delay from arm (tune) | median lead before take (tune) | first-alarm isolation |
|---|---|---|---|---|---|
| F2 | 5/5 | 2/2 | 0 s (max 1050 s) | 9.6e3 s | 7/7 |
| F3 | 7/7 | 2/2 | 6.9e3 s | 27 s | 9/9 |
| F4 | 6/6 | 4/4 | 9.9e3 s | 21 s | 10/10 |
| F8 | 10/10 | 6/6 | 0 s | 7.9e3 s | 16/16 |
| F9 | 7/7 | no F9 seeds | 2.8e3 s (max 8.7e3 s) | 4.6e3 s (min 1.7e3 s) | 7/7 |
| F10 | 5/5 | 6/6 | 7.0e3 s | 0 s | 11/11 |

F1, F5, F6 and F7 have no detector yet and are all missed, as expected. Their only alarms are
the F2 detector's false alarms landing on the faulted vehicle after arm.

False alarms: 0.061 per vehicle-day on tune (58 alarms) and 0.083 on test (34 in 407.9
vehicle-days). Nearly all are F2 (0.049 and 0.071 per vehicle-day), spread over every vehicle and
time; F9 contributes 0.012. F3, F4, F8 and F10 raised none.

Sweeping the target (thresholds re-tuned on the tune seeds, detection rates unchanged at 100 %
for every detector above):

| target | F2 h | F9 h | tune false alarms | test false alarms |
|---|---|---|---|---|
| 0.2 | 9.4 | 8.0 | 201 | 87 |
| 0.05 | 10.7 | 8.0 | 58 | 34 |
| 0.01 | 12.4 | 8.5 | 17 | 6 |
| 0.002 | 13.6 | 10.5 | 1 | 2 |

An F2 fault scores about 4000, so the F2 threshold can be raised far above 13.6 at no cost.

## Caveats

- **F3 and F4 are only visible at the take.** The wheels move only in the slews around the take,
  and arm is 1.6-3.2 h earlier. The alarm comes about 21-27 s before take start, in the slew in,
  and the delay from arm is hours by construction.
- **F10 is only visible at the take.** D is evaluated at take start, so the lead is zero.
- **F8 and F2 alarm at arm.** F8 at p = 0.9 trips on the first dropped messages; F2's string
  reads zero.
- **F9 has no test seeds.** The test split drew none of seeds 70-99, so F9 is judged on the tune
  seeds, where it was also used to tune. Its delay depends on when the vehicle next crosses the SAA.
- **Noiseless truth inputs.** `wheel_steps` is truth, so the wheel fit is exact and its
  threshold sits on the floor. A real detector would see noisy torque and momentum telemetry.
- **Small counts.** Per-fault rates come from 2-16 seeds per split.

## Environment note

The simulator cannot run new seeds in this environment at the moment: astropy refuses IERS
predictions more than 30 days past the bundled table, so `expand3 run` and
`tests/regression/test_batch.py` fail on a clean `main`. The detector tests that run the simulator
set `iers.conf.auto_max_age = None` for their own run. The simulator itself is unchanged.
