from pathlib import Path

import pytest

from expand3.params import Params, load_params

PARAMS_DIR = Path(__file__).resolve().parents[1] / "params"


@pytest.fixture(scope="session")
def params() -> Params:
    return load_params(PARAMS_DIR)
