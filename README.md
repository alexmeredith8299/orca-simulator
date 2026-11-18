# orca-simulator

Satellite fault simulator and fault detectors for the Orca SBIR project.

`expand3` simulates the EXPAND.3 Phase I scenario: six SmallSats in two clusters on two bus
designs. It models orbit and attitude, power, thermal, sensors, crosslinks, and injectable faults.
Runs are seeded, and each run writes truth and telemetry to Parquet files. The detectors in
`src/expand3/detect/` read those files and never use the simulator.

## What it covers

- **Spec faults.** F1 to F10, from the Phase I spec (`docs/spec/`). The values in `params/*.yaml`
  come from the tables in the spec.
- **Extended faults.** Six more faults that the spec does not define: `gnss_jump`, `gnss_noise`,
  `tx_fade`, `mag_bias`, `mag_noise` and `mag_coupling`. They need three additions to the
  simulator: an onboard navigation filter, a signal-strength reading on each crosslink message,
  and a magnetometer. `docs/spec/EXPAND3_Extended_Faults_Supplement.md` describes all of them.
  Every value in the supplement is a default, because the spec is silent.
- **Detectors.** Threshold detectors, learned detectors, Monte Carlo dropout detectors (single
  satellite and multi-satellite), and a posterior that turns detector scores into a probability for
  each fault.

## Setup

```bash
conda activate orca        # Python 3.12
pip install -e '.[dev]'
```

The learned detectors need an extra package set. Use `'.[dev,detect-ml]'` for the scikit-learn
detectors. The Monte Carlo dropout detectors also need `detect-dl`, which installs PyTorch. On a
machine with a CUDA GPU, install PyTorch from the wheel index that matches the GPU first.

## Run it

```bash
expand3 run --seeds 0:100 --out /path/outside/repo --workers 8      # simulate
expand3 report /path/outside/repo                                   # validation report
expand3 detect /path/outside/repo --detector all --workers 8        # run detectors
expand3 score /path/outside/repo                                    # score the alarms
expand3 posterior /path/outside/repo --detector <names>             # fault probabilities
```

- Use `--fault-set extended` with `run` to include the extended faults.
- Force one fault with `--fault <id>`. For an extended fault, choose its start shape with
  `--profile` (`step`, `ramp`, `exponential` or `intermittent`, whichever the fault allows).
- Use `--duration-h` to shorten a run. This is useful for tests.
- Write batches outside the repository. A batch of 100 seeds is large.
- Tune detectors on seeds 0 to 69. Test them on seeds 70 to 99, or on a fresh batch.
- Do not edit `params/` or `src/` while a batch is running. Each seed reloads the parameters from
  disk.

`scripts/run_multisat_batches.sh` runs the batches for the multi-satellite faults and their
detectors on a server. Run `expand3 <command> --help` for every option.

## Where things are

| Path | Content |
|---|---|
| `params/` | Parameters from the spec tables, one YAML file for each topic |
| `src/expand3/` | The simulator: environment, orbit, attitude, power, thermal, sensors, faults |
| `src/expand3/detect/` | Fault detectors, scoring and the posterior |
| `docs/spec/` | The spec (PDF) and the supplement for the extended faults |
| `docs/detection-*.md` | Reports on detection results |
| `docs/extended-faults-plan.md` | The plan for the extended faults |
| `tests/` | Unit, invariant, reference, fault, regression and detector tests |

`CLAUDE.md` has the conventions, the milestone list and every choice that the spec leaves open.

## Checks

Run these before every commit:

```bash
ruff check .
mypy src
pytest
```

The full `pytest` run is long. Each slow test is marked, and `pytest -m "not slow"` skips them.
