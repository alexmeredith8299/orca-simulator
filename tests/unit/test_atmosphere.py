"""M1: exponential atmosphere rho(h) = rho0 exp(-(h - h0)/H)."""

import numpy as np
import pytest

from expand3.env.atmosphere import density_kg_m3


def test_density() -> None:
    h = np.array([525.0, 585.0, 465.0])
    rho = density_kg_m3(h, rho0_kg_m3=4e-13, h0_km=525.0, H_km=60.0)
    assert rho[0] == pytest.approx(4e-13)
    assert rho[1] == pytest.approx(4e-13 / np.e)
    assert rho[2] == pytest.approx(4e-13 * np.e)
