# Fault posterior, MC-dropout detectors, and model-based vs data-driven detection

Batch `~/orca-batches/m8_seeds_0_99`; seeds 0-69 tune, 70-99 test. Outputs of the final run are in
`DIR/detect/posterior_all/` (`posterior.md`, `posterior.json`, `posterior.csv`, `alarms.csv`,
`score.md`). Earlier reports: `detection-report.md` (physics) and `detection-datadriven-report.md`.

## Summary

- Two things were added. **MC-dropout detectors** (`mcd_thermal` for F7, `mcd_eps` for F1) and a
  **fault posterior** that turns any detector's score into P(fault k is active on its vehicle | data
  so far), by Bayes' rule with a prior taken from the spec.
- The posterior, thresholded at P >= 0.9 and fed to `expand3 score`, detects every fault except F5
  and F6 (no detector covers SAT-4's burn faults). On the tune seeds it had 0 false alarms in 951.8
  vehicle-days, and on the test seeds 1 in 407.9.
- **Test log loss** (lower is better; the prior alone scores 0.25 for F1, 0.34 for F7 and 0.29 for
  F2):

  | fault | physics | gradient-boosted trees | MC dropout |
  |---|---|---|---|
  | F1 | none exists | 0.0110 | 0.0150 |
  | F2 | 0.0000 (`f2`) | 0.0003 | none |
  | F7 | none exists | 0.2776 | **0.1398** |

  MC dropout is clearly better on F7 and slightly worse on F1. The test split has 1 faulted seed
  for F1 and 2 for F7, so only the F7 gap, which also holds on the tune seeds, is worth leaning on.
- The "physics" detectors in this repo are not first-principles forward-model residuals. There is
  no physics detector for F1 or F7, so on those two faults the comparison is between data-driven
  methods only. See "What the comparison can and cannot say".

## What was built

| piece | file |
|---|---|
| MC-dropout LSTM forecasters `mcd_thermal`, `mcd_eps`; uses CUDA when torch sees it | `src/expand3/detect/mcdrop.py`, extra `detect-dl` |
| Fault posterior: prior, tempered evidence, normalization, per-fault detector selection | `src/expand3/detect/posterior.py` |
| `expand3 posterior BATCH --detector ... [--select] [--nominal PKL]` | `src/expand3/cli.py` |
| Tests (written first): prior, calibration of the fit, normalization, synthetic batches | `tests/detect/test_posterior.py`, `tests/detect/test_mcdrop.py` |

Checks before the first commit: ruff and strict mypy clean, 289 tests passed. The per-candidate
test log loss was added afterwards; the 12 posterior tests pass with it and I did not rerun the
whole suite.

## The shared layer: how a score becomes an alarm or a probability

### Scores are GLRs

Every detector reduces its data to one residual stream, then runs a sliding-window generalized
likelihood ratio (`glr.py`). On the last n samples, maximized over n = 1..W, it tests a known
fault-free parameter against an unknown shifted one:

- Gaussian mean shift: `(sum x - mu0)^2 / (2 sigma^2 n)`.
- Slope (F1): the voltage residual grows in proportion to the current, so the test is
  `(sum x r)^2 / (2 sigma^2 sum x^2)` on residual `r` and measured current `x`.
- Poisson rate (F9) and binomial probability (F8, peer F9).

It is a log likelihood ratio at the best-fit shift and window, so it is 0 when the data favor no
shift. Because it is maximized over the window and the shift, and the residuals are autocorrelated,
it is not a true log likelihood ratio, and `e^S` overstates the evidence.

### Alarms (the older path)

`tune_threshold` bisects a threshold per detector and fault id so the fault-free tune data give at
most 0.05 alarm episodes per vehicle-day, with a floor `h_floor`. An alarm is the first sample of
each episode above it.

### The posterior (new)

For fault k on its vehicle v, at time t:

1. **Prior.** The spec injects one fault per seed, uniform over F1..F10, on the fault's own vehicle,
   armed at (take start - lead) with lead ~ U(5700, 11400) s. So the prior that k is active is
   `pi_k F(t)` with `pi_k = 0.1` and `F` the CDF of the arm time given the vehicle's take start. It is
   exactly 0 before the earliest possible arm, on every other vehicle, and on a vehicle with no
   take (no fault is armed). `F` and `pi` are read from `params/`; the take start comes from the
   mission plan in `takes.parquet`.
2. **Evidence.** `log Lambda_k = a_k S_k - c_k`. The tempering `a_k >= 0` and offset `c_k` are fitted
   by maximum likelihood on the tune seeds: a logistic regression of "fault k is active" on `S_k`
   with the log prior odds as a fixed offset (empirical Bayes). `a=1, c=0` would be the raw GLR.
3. **Normalization.** `P_k = o_k / (1 + sum_j o_j)` over the channels on that vehicle, with
   `o_k = pi_k F e^{log Lambda_k} / (1 - F sum_j pi_j)`. A fault with no detector keeps its prior mass
   on the null, because the data cannot tell it from no fault.
4. **Selection.** With `--select`, a fault covered by several detectors keeps the one with the lowest
   tune log loss. The test seeds play no part in the choice.

The fitted scales show how overconfident the raw GLRs were: `a` is 0.037 for F1 (`lrn_eps_hgb`) and
0.095 for F7 (`mcd_thermal`). The count and rate GLRs are closer to calibrated (F9 1.18, F8 2.18,
the latter slightly underconfident). Where a score separates the classes perfectly (F2, F10) `a` and
`c` are not identified, and the wheel scores are discussed below.

## How each detector works

All detectors read measurement-side files only (`measurements`, `crosslink`, `takes`,
`wheel_steps`); the reader refuses `truth.parquet` and any `_true` column. Nominal values come from
fault-free tune data (unfaulted vehicles plus each faulted vehicle before arm).

### Physics family: `f2 f8 f9 f10 wheels`

These test a known structure of the signal rather than learn one. None simulates the vehicle.

- **`f2`** (F2, solar string open). For each string, the ratio of its power to the mean of the
  other strings, taken only while sunlit (mean power above half its 99th percentile). Strings on one
  array see the same Sun, so the ratio cancels the orbit, eclipse and attitude and sits at a
  fault-free mean. One-sided Gaussian GLR for a drop, window 10; the string is the alarm detail.
- **`f8`** (F8, crosslink margin loss). Binomial GLR, window 20, on each link's delivered flags
  against a pooled fault-free drop rate. A vehicle's score is the sum over its incident links and
  only the highest-scoring vehicle can alarm, which keeps its neighbours quiet.
- **`f9`** (F9, SAA upset rate). Poisson GLR, window 20, on upsets per tick while inside the SAA,
  against the fault-free rate times the tick length.
- **`f10`** (F10, imager dark current). Uses the spec's equation `D = D0 2^((Tp - Tref)/dT)` with
  its parameters read from `params/`. The statistic is a Gaussian GLR on log2(measured / model), one
  sample per take. This is the one detector that is a true model residual.
- **`wheels`** (F3, F4). A causal least-squares fit, per wheel and per slew, of the spec's discrete
  update `(h_prev - h)/dt = k u + c h_prev`. The score is the largest deviation over wheels of the
  fitted `k` (F4) or `c` (F3) from the fault-free fit, in units of the fault-free spread.

### Learned residuals: `lrn_*`

Predict a measured signal from context and run the GLR on the miss. The context (`features.py`, about
29 columns) is built from measured signals only: sunlit state inferred from the string powers, time
since the last eclipse entry and exit, the operation gates (pass, take, slew, burn, SAA, shed),
delivered heater states, and first-order filters of these with time constants up to 10 h. No node
temperature is an input to another node's model, because F7 cools the neighbours and they would
absorb it. Models are per bus, trained on every 4th tick of nominal data (at most 150,000 rows). The
residual spread comes from seeds held out of the fit (fit on even seeds, score odd, and the reverse).

- **Thermal (F7).** One regressor per node temperature from context alone. The score is the worst
  node's Gaussian GLR over 120 samples.
- **EPS (F1).** Pack voltage from measured current, SoC estimate and battery temperature. Slope
  GLR of the residual on the current; a plain mean-shift GLR would cancel across charge and
  discharge. **(F2)** each string's power while sunlit from context, GLR for a drop.
- **Ridge** is closed-form on standardized features (cubic for the voltage model). **HGB** is
  scikit-learn `HistGradientBoostingRegressor` (150 iterations, learning rate 0.1, 31 leaves) on the
  same features. HGB is the gradient-boosted-trees detector in the comparison above.

### Peer differencing: `peer_*`

SAT-2 and SAT-3 minus SAT-1, matched by orbit phase (the lag that best aligns the sunlit patterns
over the first hours). Ticks with a pass, burn, take or slew on either vehicle are dropped. This
cancels whatever the environment does to both vehicles, so no model of it is needed. Thermal
differences are regressed on the difference of the context features; strings use a ratio; voltage
uses a ridge on the current, SoC and battery-temperature differences; SAA upsets use a binomial
test on which vehicle took the upsets in matched exposure.

### MC dropout: `mcd_thermal`, `mcd_eps`

After Sadr, Zhu and Hu (2022). A 2-layer LSTM (hidden 32) with dropout 0.2 inside the layers and
before the output, kept stochastic at test time, trained with Adam (20 epochs, learning rate 2e-3, MSE
on standardized targets).

- **Inputs:** the same exogenous context as the learned residuals (for F1, also the current, SoC
  estimate and battery temperature), at every 10th tick, as a window of 24 steps (2 h). Targets are
  all node temperatures (F7) or pack voltage (F1). One model per bus; nominal data only.
- **Uncertainty:** 20 stochastic passes give a mean and a variance (epistemic). An aleatoric term is
  fitted on tune seeds the network did not train on, as the residual variance the passes miss.
- **Score:** `z = (measured - mean) / sqrt(aleatoric + epistemic)` into the same GLR (Gaussian, window
  12 samples, worst node for F7; slope on current for F1).
- **Why the window holds exogenous inputs only:** the paper forecasts a signal from its own
  history, which adapts to a slow drift, the form of F7 and F1.

Whether the model trained on the GPU is **not verified**: the device is chosen automatically and I
only checked that torch sees the card.

## Results (test seeds 70-99)

Log loss is the per-sample binary cross-entropy of P(fault active), over samples where the prior is
nonzero, against whether the fault was truly active. Candidates are fitted on the tune seeds and
all are scored on the test seeds. Faulted test seeds per fault: F1 1, F2 2, F3 2, F4 4, F7 2, F8 6,
F10 6, **F9 0**.

| fault | detector | family | tune log loss | test log loss | prior only (test) |
|---|---|---|---|---|---|
| F1 | `lrn_eps_hgb` | GB trees | 0.0287 | 0.0110 | 0.2484 |
| F1 | `lrn_eps_ridge` | ridge | 0.0293 | 0.0116 | |
| F1 | `mcd_eps` | MC dropout | 0.0315 | 0.0150 | |
| F1 | `peer_eps` | peer | 0.0408 | 0.0086 | |
| F2 | `f2` | physics | 0.0000 | 0.0000 | 0.2915 |
| F2 | `lrn_eps_hgb` | GB trees | 0.0000 | 0.0003 | |
| F2 | `lrn_eps_ridge`, `peer_eps` | ridge, peer | 0.0000 | 0.0000 | |
| F7 | `lrn_thermal_hgb` | GB trees | 0.3160 | 0.2776 | 0.3371 |
| F7 | `lrn_thermal_ridge` | ridge | 0.2844 | 0.2430 | |
| F7 | `peer_thermal` | peer | 0.2186 | 0.1798 | |
| F7 | `mcd_thermal` | MC dropout | **0.1791** | **0.1398** | |
| F3 | `wheels` | physics | 0.0764 | 0.0556 | 0.2985 |
| F4 | `wheels` | physics | 0.0676 | 0.0944 | 0.5428 |
| F8 | `f8` | physics | 0.0000 | 0.0000 | 0.4879 |
| F9 | `f9` | physics | 0.0054 | 0.0008 | 0.2165 |
| F9 | `peer_saa` | peer | 0.0703 | 0.1989 | |
| F10 | `f10` | physics | 0.0000 | 0.0000 | 0.5004 |

What the numbers say:

- **F7: MC dropout beats gradient-boosted trees** by a factor of two (0.140 vs 0.278), on both
  splits (tune 0.179 vs 0.316). Relative to the prior alone it removes 59 % of the loss; HGB removes
  18 %, ridge 28 % and peer differencing 47 %. This is the only fault where the choice of data-driven
  model matters much, and the ordering is the same on 8 faulted tune seeds and 2 test seeds.
- **F1: the data-driven models are within noise of each other.** HGB 0.0110, ridge 0.0116, MC
  dropout 0.0150, peer 0.0086, all removing 94-97 % of the prior's loss. The test set has one faulted
  seed. HGB wins on tune (0.0287 vs 0.0315), peer wins on test, but peer was the worst on tune
  (0.0408). I would not rank them.
- **F2, F8, F10 are solved** by every candidate that exists. Several reach exactly 0.0000 on both
  splits, which means the score separates fault from no fault perfectly. The choice among tied
  candidates is arbitrary, and calibration there is not tested. F10 also has only 30 test rows.
- **F9:** the test split has no F9 seed, so its loss of 0.0008 only reflects that the detector
  stays quiet; it says nothing about detection. `peer_saa` is worse (0.1989).
- **Wheels (F3, F4) are not what they look like.** `a` came out as 1.0000 for both because the score
  is nearly binary: with a noiseless simulator the fault-free spread falls to its floor (1e-3), and
  the tune-data median score is 0 and the 90th to 100th percentiles are all 20 (F3) or all 500 (F4).
  The loss is flat in `a`, so only `a S - c` at those levels is identified. I refit with Nelder-Mead
  from starting values of 0.1, 1 and 3: F3 gives `a` = 211, 74 and 75, and F4 gives 1.05, 0.08 and
  0.08, with identical loss each time (`c` unchanged). So F3 and F4 behave as calibrated yes/no
  detectors, not graded ones.
- **F5 and F6 have no detector** in the registry (SAT-4 burn faults). Their prior mass stays on the
  null and they are never detected.

## Model-based vs data-driven detection: what the comparison can and cannot say

**What "physics" means here.** Of the five physics detectors, two use an equation from the spec
(`f10` the dark-current model, `wheels` the actuator update), one uses a physical redundancy (`f2`:
strings on one array see the same Sun), and two use the known statistical form of the process
(`f8` Bernoulli drops, `f9` Poisson upsets). None integrates a forward model of the vehicle. There is
no physics detector for F1 or F7, the two faults on slow, environment-driven signals, so the
requested head-to-head (physics vs trees vs MC dropout) is only possible on F2, where physics and
trees tie, and on F1 and F7 the comparison is trees vs MC dropout vs peer. A model-based competitor
there would be an observer built from the spec's thermal network (the topology and parameters are in
`params/`) or the pack's voltage equation. **I did not build one**, so this report cannot say how a
true model-based detector would do on F1 and F7.

**Where each approach gets its strength**, as far as these experiments show:

| | model/structure-based | data-driven (trees, MC dropout) | peer differencing |
|---|---|---|---|
| assumes | the signal's structure is known (an equation, a symmetry, a noise law) | the context features carry everything that drives the signal | a healthy twin sees the same environment |
| learns from | a handful of fault-free scalars (mean, spread, rate) | thousands of nominal rows per bus | the same, for a small correction |
| strong when | the fault has a clean, low-dimensional signature (F2, F8, F9, F10, F3/F4) | the signal depends on history in a way no one wrote down (F7 temperatures) | the environment is common and hard to model |
| weak when | the signal is driven by an environment the equation doesn't contain | the fault looks like a drift the model can absorb, or the nominal data miss a regime | no healthy twin (SAT-4 for F5, F6), or the twin differs |
| tells you which component | yes (string, wheel, link) | worst node only | worst node or string |
| calibration | needs a variance floor on a noiseless simulator, which then dominates (wheels) | residual spread from held-out seeds; MC dropout adds a per-sample spread | spread from the nominal difference |

**What I think explains the F7 result, and what is untested.** F7 cools the whole vehicle by about
1.5 K against orbital swings, and the thermal state has long memory. Trees see the history only
through hand-built exponential filters of the inputs; the LSTM reads a 2 h window of them directly and
also weights each miss by its own uncertainty. These are hypotheses: I ran no ablation (for example
trees on lagged windows, or MC dropout with the variance fixed), so I cannot attribute the gap to
sequence modeling versus uncertainty weighting versus the different sampling (every 10th tick vs
every tick).

**Noise caveat for the whole comparison.** The simulator is noiseless in the channels the physics
detectors use (string ratios, drops, upsets, wheel fits), which is why they reach 0.0000. On flight
data, the same detectors would need real variance floors and would lose some of that margin. The
data-driven detectors were tested where the signal carries real sensor noise (thermistors, voltage,
current), so they were the harder test. Near-zero loss on F2, F8 and F10 is a property of this
simulator, not a ranking of methods.

## Limits

- **Small test set.** F1 has 1 faulted test seed, F2 2, F3 2, F7 2. Rows within a seed are strongly
  correlated, so the row counts (for example 116,420 for F1) overstate the information. Treat
  differences smaller than about 2x as unresolved.
- **Selection used tune data only**, but the tune seeds also trained every detector, so tune loss
  is optimistic. Test loss is the number to cite.
- **The prior uses the planned take time**, which is known in advance but leans on the spec's
  definition of fault onset.
- **Channels are combined as if independent** given the hypothesis; they are not.
- **Fits are per channel** (binary logistic with a fixed offset) and then normalized, not a joint
  multi-class fit.
- **Nothing here covers F5, F6**, and `unsup_*`, `clf` and `clf_raw` are not part of the posterior
  (unsupervised detectors name subsystems, not faults, and the classifiers were not wired in).
- **Posterior alarms are not comparable one-for-one with the older threshold alarms.** The posterior
  also uses the prior to rule out alarms before the earliest arm and on other vehicles, which is
  why its false alarms fall from 94 (tune) to 3 for the same F1/F7 detectors.

## Reproduce

```bash
conda activate orca && pip install -e '.[dev,detect-ml,detect-dl]'
# MC-dropout detectors on a CUDA machine need the matching torch wheel
pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu130
B=~/orca-batches/m8_seeds_0_99
expand3 detect $B --detector mcd_thermal,mcd_eps --workers 8 --out $B/detect/mcd
expand3 posterior $B --detector all --select --workers 8 --out $B/detect/posterior_all
# or skip calibration with a pickle of the nominal values of the detectors:
expand3 posterior $B --detector all --select --nominal NOMINAL.pkl --out $B/detect/posterior_all
expand3 score $B --alarms $B/detect/posterior_all/alarms.csv
```
