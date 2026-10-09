from pathlib import Path

import pytest

from expand3.constellation import ConstellationRun, propagate_constellation
from expand3.params import Params, load_params
from expand3.timebase import SeedDraws, draw_seed
from expand3.units import S_PER_HOUR

PARAMS_DIR = Path(__file__).resolve().parents[1] / "params"
TEST_SEED = 20270
LONG_RUN_H = 72.0


@pytest.fixture(scope="session")
def params() -> Params:
    return load_params(PARAMS_DIR)


@pytest.fixture(scope="session")
def draws(params: Params) -> SeedDraws:
    return draw_seed(params, TEST_SEED)


@pytest.fixture(scope="session")
def run72(params: Params, draws: SeedDraws) -> ConstellationRun:
    """All six vehicles, full force model, 72 h (the longest spec run)."""
    return propagate_constellation(params, draws, duration_s=LONG_RUN_H * S_PER_HOUR)
