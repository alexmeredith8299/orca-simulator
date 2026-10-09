# orca-simulator

Satellite fault simulator for the Orca SBIR project.

`expand3` simulates the EXPAND.3 Phase I scenario: six SmallSats in two clusters on two bus
designs. It models orbit and attitude, power, thermal, sensors, links, and ten injectable
faults (F1 to F10). Runs are seeded and write truth and telemetry for fault-detection work
that lives outside this repo. Parameters come from the spec tables in `params/*.yaml`.

Status: M0 scaffold (parameter loading and validation). See `CLAUDE.md` for conventions
and the milestone plan.

## Setup

```bash
conda activate orca        # Python 3.12
pip install -e '.[dev]'
```

## Checks

```bash
ruff check .
mypy src
pytest
```
